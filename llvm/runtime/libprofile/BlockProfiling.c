/*===-- BlockProfiling.c - Support library for block profiling ------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source      
|* License. See LICENSE.TXT for details.                                      
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file implements the call back routines for the block profiling
|* instrumentation pass.  This should be used with the -insert-block-profiling
|* LLVM pass.
|*
\*===----------------------------------------------------------------------===*/

#include "Profiling.h"
#include <stdlib.h>

static unsigned *ArrayStart;
static unsigned NumElements;

/* BlockProfAtExitHandler - When the program exits, just write out the profiling
 * data.
 */
static void BlockProfAtExitHandler() {
  /* Note that if this were doing something more intelligent with the
   * instrumentation, we could do some computation here to expand what we
   * collected into simple block profiles. (Or we could do it in llvm-prof.)
   * Regardless, we directly count each block, so no expansion is necessary.
   */
  write_profiling_data(BlockInfo, ArrayStart, NumElements);
}


/* llvm_start_block_profiling - This is the main entry point of the block
 * profiling library.  It is responsible for setting up the atexit handler.
 */
int llvm_start_block_profiling(int argc, const char **argv,
                               unsigned *arrayStart, unsigned numElements) {
  int Ret = save_arguments(argc, argv);
  ArrayStart = arrayStart;
  NumElements = numElements;
  atexit(BlockProfAtExitHandler);
  return Ret;
}
