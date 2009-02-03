/*===-- FunctionProfiling.c - Support library for function profiling ------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source      
|* License. See LICENSE.TXT for details.                                      
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file implements the call back routines for the function profiling
|* instrumentation pass.  This should be used with the
|* -insert-function-profiling LLVM pass.
|*
\*===----------------------------------------------------------------------===*/

#include "Profiling.h"
#include <stdlib.h>

static unsigned *ArrayStart;
static unsigned NumElements;

/* FuncProfAtExitHandler - When the program exits, just write out the profiling
 * data.
 */
static void FuncProfAtExitHandler() {
  /* Just write out the data we collected.
   */
  write_profiling_data(FunctionInfo, ArrayStart, NumElements);
}


/* llvm_start_func_profiling - This is the main entry point of the function
 * profiling library.  It is responsible for setting up the atexit handler.
 */
int llvm_start_func_profiling(int argc, const char **argv,
                              unsigned *arrayStart, unsigned numElements) {
  int Ret = save_arguments(argc, argv);
  ArrayStart = arrayStart;
  NumElements = numElements;
  atexit(FuncProfAtExitHandler);
  return Ret;
}
