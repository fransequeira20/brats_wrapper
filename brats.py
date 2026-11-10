"""
######### Python wrapper for the Broadband Radio Astronomy Tools (BRATS) ########

This is updated wrapper based on the Python wrapper for BRATS made by 
Jeremy Harwood.

This early version currently only supports the running of command files in which 
BRATS commands, tasks, and parameters should be placed in the order by which they
are to be processed, with one command per line.

Single core and multiprocessing are both supported. 

A new function was created to allow a command by command process. 

***************************** IMPORTANT PLEASE READ  *****************************

 Note there is no BRATS command and syntax error checking in this early version!
 Only Python errors are handled. Incorrect commands are ignored which may lead to
 unexpected results or failure to complete the run, particularly when setting
 paramaters within a task.
 It is therefore suggested that a manual test run is performed on a sample dataset
 by simply running BRATS, then copying and pasting the contents of the file into
 the terminal.

 *********************************************************************************

@author: Francisco Sequeira Murillo
@email: sequeiramuri@wisc.edu

##################################################################################
"""

import os.path
import subprocess
import signal
import multiprocessing as mp

# Base class for the BRATS wrapper
class Bind:

    def __init__(self, install_dir: str):
        self._install = install_dir

    def execfile(self, file_loc: str) -> dict:
        return _Script._execfile(self, file_loc)

    def multiexec(self, file_list: list, num_proc: int) -> dict:
        return _Multiexec._multiexec(self, file_list, num_proc)
    
    def execBratsWrapper(self, command: str) -> dict:
        return _MultiPixel._execBratsWrapper(self, command)

# Child class: processes command files and executes 
class _Script(Bind):
     
    def __init__(self): 
        super(_Script, self).__init__()

    def _execfile(self, file_loc: str) -> dict:

        try:
            commandsfile = open(file_loc, "r")
        except FileNotFoundError:
            print("\n*** Error: File not found. Ensure the commands file %s exists ***\n" % (file_loc))
            return {file_loc: 404}
        except Exception as e:
            print("\n*** Error: %s ***\n*** Unable to open the commands file %s for reading. Check the file exists and the appropriate permissions are set ***\n" % (str(e), file_loc))
            return {file_loc: 401}

        try:
            __process = subprocess.Popen([self._install], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
            data = commandsfile.read()
            __process.communicate(input=data.strip())
            commandsfile.close()
            return {file_loc: 0}
        except Exception as e:
            print("\n*** Error: %s ***\n*** BRATS processing failed***\n" % (e))
            return {self._install: 404}

# Child class: handles multiprocessing for command files
class _Multiexec(_Script):
    
    def __init__(self): 
        super(_Multiexec, self).__init__()

    def _multiexec(self, file_list: list, num_proc: int) -> dict:

        if os.path.exists(self._install) == False:
            print("\n*** Error: File not found. Ensure the brats installation %s exists ***\n" % (self._install))
            return {"Error": 404}
        try: 
            pool = mp.Pool(num_proc) # Run on 2 cores
            returnvalue = pool.map(self.execfile, file_list)
            pool.close()
            pool.join()
            return returnvalue
        except Exception as e:
            print("\n*** Error: %s ***\n*** BRATS processing failed***\n" % (e))
            return {"Error": str(e)}
        

# MultiPixel class: processes line by line the commands from bratsWrapper
class _MultiPixel(Bind):
     
    def __init__(self): 
        super(_MultiPixel, self).__init__()

    def _execBratsWrapper(self, commands: list) -> dict:
        if not isinstance(commands, list):
            raise ValueError("Expected a list of commands.")
        results = {}
        __process = None
        try:
            __process = subprocess.Popen([self._install], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, start_new_session=True)
            
            data = "\n".join(commands)
            outs, errs = __process.communicate(input=data.strip())
            results["status"] = 0
            results['outs'] = outs
            results['errs'] = errs
        except Exception as e:
            # Catches TimeoutException - kill BRATS so it doesn't keep running
            if __process is not None and __process.poll() is None:
                os.killpg(__process.pid, signal.SIGKILL)
                __process.communicate()
            print("\n*** Error: %s ***\n*** BRATS processing failed ***\n" % (e))
            results[self._install] = 404
        return results
