/*===-- ProfileInfoTypes.h - Profiling info shared constants --------------===*\
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

/* Included by libprofile. */
#if defined(__cplusplus)
extern "C" {
#endif

/* IDs to distinguish between those path counters stored in hashses vs arrays */
enum ProfilingStorageType {
  ProfilingArray = 1,
  ProfilingHash = 2
};

#include "llvm/Analysis/ProfileDataTypes.h"

/*
 * The header for tables that map path numbers to path counters.
 */
typedef struct {
  unsigned fnNumber; /* function number for these counters */
  unsigned numEntries;   /* number of entries stored */
} PathProfileHeader;

/*
 * Describes an entry in a tagged table for path counters.
 */
typedef struct {
  unsigned pathNumber;
  unsigned pathCounter;
} PathProfileTableEntry;

#if defined(__cplusplus)
}
#endif

#endif /* LLVM_ANALYSIS_PROFILEINFOTYPES_H */
