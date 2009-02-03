/*===-- ProfileInfoTypes.h - Profiling info shared constants ------*- C -*-===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
|*===----------------------------------------------------------------------===*|
|*
|* This file defines constants shared by the various different profiling
|* runtime libraries and the LLVM C++ profile info loader. It must be a
|* C header because, at present, the profiling runtimes are written in C.
|*
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_ANALYSIS_PROFILEINFOTYPES_H
#define LLVM_ANALYSIS_PROFILEINFOTYPES_H

enum ProfilingType {
  ArgumentInfo  = 1,   /* The command line argument block */
  FunctionInfo  = 2,   /* Function profiling information  */
  BlockInfo     = 3,   /* Block profiling information     */
  EdgeInfo      = 4,   /* Edge profiling information      */
  PathInfo      = 5,   /* Path profiling information      */
  BBTraceInfo   = 6    /* Basic block trace information   */
};

#endif /* LLVM_ANALYSIS_PROFILEINFOTYPES_H */
