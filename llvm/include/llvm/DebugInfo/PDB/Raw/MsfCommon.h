//===- MsfCommon.h - Common types and functions for MSF files ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_MSFCOMMON_H
#define LLVM_DEBUGINFO_PDB_RAW_MSFCOMMON_H

#include "llvm/ADT/ArrayRef.h"

#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <vector>

namespace llvm {
namespace pdb {
namespace msf {
static const char Magic[] = {'M',  'i',  'c',    'r', 'o', 's',  'o',  'f',
                             't',  ' ',  'C',    '/', 'C', '+',  '+',  ' ',
                             'M',  'S',  'F',    ' ', '7', '.',  '0',  '0',
                             '\r', '\n', '\x1a', 'D', 'S', '\0', '\0', '\0'};

// The superblock is overlaid at the beginning of the file (offset 0).
// It starts with a magic header and is followed by information which
// describes the layout of the file system.
struct SuperBlock {
  char MagicBytes[sizeof(Magic)];
  // The file system is split into a variable number of fixed size elements.
  // These elements are referred to as blocks.  The size of a block may vary
  // from system to system.
  support::ulittle32_t BlockSize;
  // The index of the free block map.
  support::ulittle32_t FreeBlockMapBlock;
  // This contains the number of blocks resident in the file system.  In
  // practice, NumBlocks * BlockSize is equivalent to the size of the PDB
  // file.
  support::ulittle32_t NumBlocks;
  // This contains the number of bytes which make up the directory.
  support::ulittle32_t NumDirectoryBytes;
  // This field's purpose is not yet known.
  support::ulittle32_t Unknown1;
  // This contains the block # of the block map.
  support::ulittle32_t BlockMapAddr;
};

struct Layout {
  SuperBlock *SB;
  ArrayRef<support::ulittle32_t> DirectoryBlocks;
  ArrayRef<support::ulittle32_t> StreamSizes;
  std::vector<ArrayRef<support::ulittle32_t>> StreamMap;
};

inline bool isValidBlockSize(uint32_t Size) {
  switch (Size) {
  case 512:
  case 1024:
  case 2048:
  case 4096:
    return true;
  }
  return false;
}

// Super Block, Fpm0, Fpm1, and Block Map
inline uint32_t getMinimumBlockCount() { return 4; }

// Super Block, Fpm0, and Fpm1 are reserved.  The Block Map, although required
// need not be at block 3.
inline uint32_t getFirstUnreservedBlock() { return 3; }

inline uint64_t bytesToBlocks(uint64_t NumBytes, uint64_t BlockSize) {
  return alignTo(NumBytes, BlockSize) / BlockSize;
}

inline uint64_t blockToOffset(uint64_t BlockNumber, uint64_t BlockSize) {
  return BlockNumber * BlockSize;
}

Error validateSuperBlock(const SuperBlock &SB);
}
}
}

#endif
