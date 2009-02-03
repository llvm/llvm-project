/*===-- Profiling.h - Profiling support library support routines --*- C -*-===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source      
|* License. See LICENSE.TXT for details.                                      
|*
|*===----------------------------------------------------------------------===*|
|*
|* This file defines functions shared by the various different profiling
|* implementations.
|*
\*===----------------------------------------------------------------------===*/

#ifndef PROFILING_H
#define PROFILING_H

#include "llvm/Analysis/ProfileInfoTypes.h" /* for enum ProfilingType */

/* save_arguments - Save argc and argv as passed into the program for the file
 * we output.
 */
int save_arguments(int argc, const char **argv);

/* write_profiling_data - Write out a typed packet of profiling data to the
 * current output file.
 */
void write_profiling_data(enum ProfilingType PT, unsigned *Start,
                          unsigned NumElements);

#endif
