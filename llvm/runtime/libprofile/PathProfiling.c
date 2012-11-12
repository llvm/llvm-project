/*===-- PathProfiling.c - Support library for path profiling --------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
|*===----------------------------------------------------------------------===*|
|*
|* This file implements the call back routines for the path profiling
|* instrumentation pass.  This should be used with the -insert-path-profiling
|* LLVM pass.
|*
\*===----------------------------------------------------------------------===*/

#include "Profiling.h"
#include "llvm/Analysis/ProfileInfoTypes.h"
#include "llvm/Support/DataTypes.h"
#include <sys/types.h>
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* note that this is used for functions with large path counts,
         but it is unlikely those paths will ALL be executed */
#define ARBITRARY_HASH_BIN_COUNT 100

typedef struct pathHashEntry_s {
  uint32_t pathNumber;
  uint32_t pathCount;
  struct pathHashEntry_s* next;
} pathHashEntry_t;

typedef struct pathHashTable_s {
  pathHashEntry_t* hashBins[ARBITRARY_HASH_BIN_COUNT];
  uint32_t pathCounts;
} pathHashTable_t;

typedef struct {
  enum ProfilingStorageType type;
  uint32_t size;
  void* array;
} ftEntry_t;

/* pointer to the function table allocated in the instrumented program */
ftEntry_t* ft;
uint32_t ftSize;

/* write an array table to file */
void writeArrayTable(uint32_t fNumber, ftEntry_t* ft, uint32_t* funcCount) {
  int outFile = getOutFile();
  uint32_t arrayHeaderLocation = 0;
  uint32_t arrayCurrentLocation = 0;
  uint32_t arrayIterator = 0;
  uint32_t functionUsed = 0;
  uint32_t pathCounts = 0;

  /* look through each entry in the array to determine whether the function
     was executed at all */
  for( arrayIterator = 0; arrayIterator < ft->size; arrayIterator++ ) {
    uint32_t pc = ((uint32_t*)ft->array)[arrayIterator];

    /* was this path executed? */
    if( pc ) {
      PathProfileTableEntry pte;
      pte.pathNumber = arrayIterator;
      pte.pathCounter = pc;
      pathCounts++;

      /* one-time initialization stuff */
      if(!functionUsed) {
        arrayHeaderLocation = lseek(outFile, 0, SEEK_CUR);
        lseek(outFile, sizeof(PathProfileHeader), SEEK_CUR);
        functionUsed = 1;
        (*funcCount)++;
      }

      /* write path data */
      if (write(outFile, &pte, sizeof(PathProfileTableEntry)) < 0) {
        fprintf(stderr, "error: unable to write path entry to output file.\n");
        return;
      }
    }
  }

  /* If this function was executed, write the header */
  if( functionUsed ) {
    PathProfileHeader fHeader;
    fHeader.fnNumber = fNumber;
    fHeader.numEntries = pathCounts;

    arrayCurrentLocation = lseek(outFile, 0, SEEK_CUR);
    lseek(outFile, arrayHeaderLocation, SEEK_SET);

    if (write(outFile, &fHeader, sizeof(PathProfileHeader)) < 0) {
      fprintf(stderr,
              "error: unable to write function header to output file.\n");
      return;
    }

    lseek(outFile, arrayCurrentLocation, SEEK_SET);
  }
}

static uint32_t hash (uint32_t key) {
  /* this may benefit from a proper hash function */
  return key%ARBITRARY_HASH_BIN_COUNT;
}

/* output a specific function's hash table to the profile file */
void writeHashTable(uint32_t functionNumber, pathHashTable_t* hashTable) {
  int outFile = getOutFile();
  PathProfileHeader header;
  uint32_t i;

  header.fnNumber = functionNumber;
  header.numEntries = hashTable->pathCounts;

  if (write(outFile, &header, sizeof(PathProfileHeader)) < 0) {
    fprintf(stderr, "error: unable to write function header to output file.\n");
    return;
  }

  for (i = 0; i < ARBITRARY_HASH_BIN_COUNT; i++) {
    pathHashEntry_t* hashEntry = hashTable->hashBins[i];

    while (hashEntry) {
      pathHashEntry_t* temp;

      PathProfileTableEntry pte;
      pte.pathNumber = hashEntry->pathNumber;
      pte.pathCounter = hashEntry->pathCount;

      if (write(outFile, &pte, sizeof(PathProfileTableEntry)) < 0) {
        fprintf(stderr, "error: unable to write path entry to output file.\n");
        return;
      }

      temp = hashEntry;
      hashEntry = hashEntry->next;
      free (temp);

    }
  }
}

/* Return a pointer to this path's specific path counter */
static uint32_t* getPathCounter(uint32_t functionNumber,
                                       uint32_t pathNumber) {
  pathHashTable_t* hashTable;
  pathHashEntry_t* hashEntry;
  uint32_t index = hash(pathNumber);

  if( ft[functionNumber-1].array == 0)
    ft[functionNumber-1].array = calloc(sizeof(pathHashTable_t), 1);

  hashTable = (pathHashTable_t*)((ftEntry_t*)ft)[functionNumber-1].array;
  hashEntry = hashTable->hashBins[index];

  while (hashEntry) {
    if (hashEntry->pathNumber == pathNumber) {
      return &hashEntry->pathCount;
    }

    hashEntry = hashEntry->next;
  }

  hashEntry = malloc(sizeof(pathHashEntry_t));
  hashEntry->pathNumber = pathNumber;
  hashEntry->pathCount = 0;
  hashEntry->next = hashTable->hashBins[index];
  hashTable->hashBins[index] = hashEntry;
  hashTable->pathCounts++;
  return &hashEntry->pathCount;
}

/* Increment a specific path's count */
void llvm_increment_path_count (uint32_t functionNumber, uint32_t pathNumber) {
  uint32_t* pathCounter = getPathCounter(functionNumber, pathNumber);
  if( *pathCounter < 0xffffffff )
    (*pathCounter)++;
}

/* Increment a specific path's count */
void llvm_decrement_path_count (uint32_t functionNumber, uint32_t pathNumber) {
  uint32_t* pathCounter = getPathCounter(functionNumber, pathNumber);
  (*pathCounter)--;
}

/*
 * Writes out a path profile given a function table, in the following format.
 *
 *
 *      | <-- 32 bits --> |
 *      +-----------------+-----------------+
 * 0x00 | profileType     | functionCount   |
 *      +-----------------+-----------------+
 * 0x08 | functionNum     | profileEntries  |  // function 1
 *      +-----------------+-----------------+
 * 0x10 | pathNumber      | pathCounter     |  // entry 1.1
 *      +-----------------+-----------------+
 * 0x18 | pathNumber      | pathCounter     |  // entry 1.2
 *      +-----------------+-----------------+
 *  ... |       ...       |       ...       |  // entry 1.n
 *      +-----------------+-----------------+
 *  ... | functionNum     | profileEntries  |  // function 2
 *      +-----------------+-----------------+
 *  ... | pathNumber      | pathCounter     |  // entry 2.1
 *      +-----------------+-----------------+
 *  ... | pathNumber      | pathCounter     |  // entry 2.2
 *      +-----------------+-----------------+
 *  ... |       ...       |       ...       |  // entry 2.n
 *      +-----------------+-----------------+
 *
 */
static void pathProfAtExitHandler(void) {
  int outFile = getOutFile();
  uint32_t i;
  uint32_t header[2] = { PathInfo, 0 };
  uint32_t headerLocation;
  uint32_t currentLocation;

  /* skip over the header for now */
  headerLocation = lseek(outFile, 0, SEEK_CUR);
  lseek(outFile, 2*sizeof(uint32_t), SEEK_CUR);

  /* Iterate through each function */
  for( i = 0; i < ftSize; i++ ) {
    if( ft[i].type == ProfilingArray ) {
      writeArrayTable(i+1,&ft[i],header + 1);

    } else if( ft[i].type == ProfilingHash ) {
      /* If the hash exists, write it to file */
      if( ft[i].array ) {
        writeHashTable(i+1,ft[i].array);
        header[1]++;
        free(ft[i].array);
      }
    }
  }

  /* Setup and write the path profile header */
  currentLocation = lseek(outFile, 0, SEEK_CUR);
  lseek(outFile, headerLocation, SEEK_SET);

  if (write(outFile, header, sizeof(header)) < 0) {
    fprintf(stderr,
            "error: unable to write path profile header to output file.\n");
    return;
  }

  lseek(outFile, currentLocation, SEEK_SET);
}
/* llvm_start_path_profiling - This is the main entry point of the path
 * profiling library.  It is responsible for setting up the atexit handler.
 */
int llvm_start_path_profiling(int argc, const char** argv,
                              void* functionTable, uint32_t numElements) {
  int Ret = save_arguments(argc, argv);
  ft = functionTable;
  ftSize = numElements;
  atexit(pathProfAtExitHandler);

  return Ret;
}
