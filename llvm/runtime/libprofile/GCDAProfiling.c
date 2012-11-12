/*===- GCDAProfiling.c - Support library for GCDA file emission -----------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This file implements the call back routines for the gcov profiling
|* instrumentation pass. Link against this library when running code through
|* the -insert-gcov-profiling LLVM pass.
|*
|* We emit files in a corrupt version of GCOV's "gcda" file format. These files
|* are only close enough that LCOV will happily parse them. Anything that lcov
|* ignores is missing.
|*
|* TODO: gcov is multi-process safe by having each exit open the existing file
|* and append to it. We'd like to achieve that and be thread-safe too.
|*
\*===----------------------------------------------------------------------===*/

#include "llvm/Support/DataTypes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#endif

/* #define DEBUG_GCDAPROFILING */

/*
 * --- GCOV file format I/O primitives ---
 */

static FILE *output_file = NULL;

static void write_int32(uint32_t i) {
  fwrite(&i, 4, 1, output_file);
}

static void write_int64(uint64_t i) {
  uint32_t lo = i >>  0;
  uint32_t hi = i >> 32;
  write_int32(lo);
  write_int32(hi);
}

static uint32_t length_of_string(const char *s) {
  return (strlen(s) / 4) + 1;
}

static void write_string(const char *s) {
  uint32_t len = length_of_string(s);
  write_int32(len);
  fwrite(s, strlen(s), 1, output_file);
  fwrite("\0\0\0\0", 4 - (strlen(s) % 4), 1, output_file);
}

static char *mangle_filename(const char *orig_filename) {
  /* TODO: handle GCOV_PREFIX_STRIP */
  const char *prefix;
  char *filename = 0;

  prefix = getenv("GCOV_PREFIX");

  if (!prefix)
    return strdup(orig_filename);

  filename = malloc(strlen(prefix) + 1 + strlen(orig_filename) + 1);
  strcpy(filename, prefix);
  strcat(filename, "/");
  strcat(filename, orig_filename);
  return filename;
}

static void recursive_mkdir(const char *filename) {
  char *pathname;
  int i, e;

  for (i = 1, e = strlen(filename); i != e; ++i) {
    if (filename[i] != '/') continue;
    pathname = malloc(i + 1);
    strncpy(pathname, filename, i);
    pathname[i] = '\0';
#ifdef _WIN32
    _mkdir(pathname);
#else
    mkdir(pathname, 0750);  /* some of these will fail, ignore it. */
#endif
    free(pathname);
  }
}

/*
 * --- LLVM line counter API ---
 */

/* A file in this case is a translation unit. Each .o file built with line
 * profiling enabled will emit to a different file. Only one file may be
 * started at a time.
 */
void llvm_gcda_start_file(const char *orig_filename) {
  char *filename;
  filename = mangle_filename(orig_filename);
  recursive_mkdir(filename);
  output_file = fopen(filename, "w+b");

  if (!output_file) {
    const char *cptr = strrchr(orig_filename, '/');
    output_file = fopen(cptr ? cptr + 1 : orig_filename, "w+b");

    if (!output_file) {
      fprintf(stderr, "LLVM profiling runtime: cannot open '%s': ",
              cptr ? cptr + 1 : orig_filename);
      perror("");
      free(filename);
      return;
    }
  }

  /* gcda file, version 404*, stamp LLVM. */
#ifdef __APPLE__
  fwrite("adcg*204MVLL", 12, 1, output_file);
#else
  fwrite("adcg*404MVLL", 12, 1, output_file);
#endif

#ifdef DEBUG_GCDAPROFILING
  printf("llvmgcda: [%s]\n", orig_filename);
#endif

  free(filename);
}

/* Given an array of pointers to counters (counters), increment the n-th one,
 * where we're also given a pointer to n (predecessor).
 */
void llvm_gcda_increment_indirect_counter(uint32_t *predecessor,
                                          uint64_t **counters) {
  uint64_t *counter;
  uint32_t pred;

  pred = *predecessor;
  if (pred == 0xffffffff)
    return;
  counter = counters[pred];

  /* Don't crash if the pred# is out of sync. This can happen due to threads,
     or because of a TODO in GCOVProfiling.cpp buildEdgeLookupTable(). */
  if (counter)
    ++*counter;
#ifdef DEBUG_GCDAPROFILING
  else
    printf("llvmgcda: increment_indirect_counter counters=%x, pred=%u\n",
           state_table_row, *predecessor);
#endif
}

void llvm_gcda_emit_function(uint32_t ident, const char *function_name) {
#ifdef DEBUG_GCDAPROFILING
  printf("llvmgcda: function id=%x\n", ident);
#endif
  if (!output_file) return;

  /* function tag */  
  fwrite("\0\0\0\1", 4, 1, output_file);
  write_int32(3 + 1 + length_of_string(function_name));
  write_int32(ident);
  write_int32(0);
  write_int32(0);
  write_string(function_name);
}

void llvm_gcda_emit_arcs(uint32_t num_counters, uint64_t *counters) {
  uint32_t i;

  /* Counter #1 (arcs) tag */
  if (!output_file) return;
  fwrite("\0\0\xa1\1", 4, 1, output_file);
  write_int32(num_counters * 2);
  for (i = 0; i < num_counters; ++i)
    write_int64(counters[i]);

#ifdef DEBUG_GCDAPROFILING
  printf("llvmgcda:   %u arcs\n", num_counters);
  for (i = 0; i < num_counters; ++i)
    printf("llvmgcda:   %llu\n", (unsigned long long)counters[i]);
#endif
}

void llvm_gcda_end_file() {
  /* Write out EOF record. */
  if (!output_file) return;
  fwrite("\0\0\0\0\0\0\0\0", 8, 1, output_file);
  fclose(output_file);
  output_file = NULL;

#ifdef DEBUG_GCDAPROFILING
  printf("llvmgcda: -----\n");
#endif
}
