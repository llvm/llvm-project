/*===-- OptimalEdgeProfiling.c - Support library for opt. edge profiling --===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source      
|* License. See LICENSE.TXT for details.                                      
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file implements the call back routines for the edge profiling
|* instrumentation pass.  This should be used with the
|* -insert-opt-edge-profiling LLVM pass.
|*
\*===----------------------------------------------------------------------===*/

#include "Profiling.h"
#include <stdlib.h>

static unsigned *ArrayStart;
static unsigned NumElements;

/* OptEdgeProfAtExitHandler - When the program exits, just write out the
 * profiling data.
 */
static void OptEdgeProfAtExitHandler(void) {
  /* Note that, although the array has a counter for each edge, not all
   * counters are updated, the ones that are not used are initialised with -1.
   * When loading this information the counters with value -1 have to be
   * recalculated, it is guaranteed that this is possible.
   */
  write_profiling_data(OptEdgeInfo, ArrayStart, NumElements);
}


/* llvm_start_opt_edge_profiling - This is the main entry point of the edge
 * profiling library.  It is responsible for setting up the atexit handler.
 */
int llvm_start_opt_edge_profiling(int argc, const char **argv,
                                  unsigned *arrayStart, unsigned numElements) {
  int Ret = save_arguments(argc, argv);
  ArrayStart = arrayStart;
  NumElements = numElements;
  atexit(OptEdgeProfAtExitHandler);
  return Ret;
}
