/*===-- BasicBlockTracing.c - Support library for basic block tracing -----===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file implements the call back routines for the basic block tracing
|* instrumentation pass.  This should be used with the -trace-basic-blocks
|* LLVM pass.
|*
\*===----------------------------------------------------------------------===*/

#include "Profiling.h"
#include <stdlib.h>
#include <stdio.h>

static unsigned *ArrayStart, *ArrayEnd, *ArrayCursor;

/* WriteAndFlushBBTraceData - write out the currently accumulated trace data
 * and reset the cursor to point to the beginning of the buffer.
 */
static void WriteAndFlushBBTraceData () {
  write_profiling_data(BBTraceInfo, ArrayStart, (ArrayCursor - ArrayStart));
  ArrayCursor = ArrayStart;
}

/* BBTraceAtExitHandler - When the program exits, just write out any remaining 
 * data and free the trace buffer.
 */
static void BBTraceAtExitHandler() {
  WriteAndFlushBBTraceData ();
  free (ArrayStart);
}

/* llvm_trace_basic_block - called upon hitting a new basic block. */
void llvm_trace_basic_block (unsigned BBNum) {
  *ArrayCursor++ = BBNum;
  if (ArrayCursor == ArrayEnd)
    WriteAndFlushBBTraceData ();
}

/* llvm_start_basic_block_tracing - This is the main entry point of the basic
 * block tracing library.  It is responsible for setting up the atexit
 * handler and allocating the trace buffer.
 */
int llvm_start_basic_block_tracing(int argc, const char **argv,
                              unsigned *arrayStart, unsigned numElements) {
  int Ret;
  const unsigned BufferSize = 128 * 1024;
  unsigned ArraySize;

  Ret = save_arguments(argc, argv);

  /* Allocate a buffer to contain BB tracing data */
  ArraySize = BufferSize / sizeof (unsigned);
  ArrayStart = malloc (ArraySize * sizeof (unsigned));
  ArrayEnd = ArrayStart + ArraySize;
  ArrayCursor = ArrayStart;

  /* Set up the atexit handler. */
  atexit (BBTraceAtExitHandler);

  return Ret;
}
