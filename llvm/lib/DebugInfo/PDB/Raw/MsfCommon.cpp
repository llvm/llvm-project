//===- MsfCommon.cpp - Common types and functions for MSF files -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/MsfCommon.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::pdb::msf;

Error llvm::pdb::msf::validateSuperBlock(const SuperBlock &SB) {
  // Check the magic bytes.
  if (std::memcmp(SB.MagicBytes, Magic, sizeof(Magic)) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "MSF magic header doesn't match");

  if (!isValidBlockSize(SB.BlockSize))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unsupported block size.");

  // We don't support directories whose sizes aren't a multiple of four bytes.
  if (SB.NumDirectoryBytes % sizeof(support::ulittle32_t) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Directory size is not multiple of 4.");

  // The number of blocks which comprise the directory is a simple function of
  // the number of bytes it contains.
  uint64_t NumDirectoryBlocks =
      bytesToBlocks(SB.NumDirectoryBytes, SB.BlockSize);

  // The directory, as we understand it, is a block which consists of a list of
  // block numbers.  It is unclear what would happen if the number of blocks
  // couldn't fit on a single block.
  if (NumDirectoryBlocks > SB.BlockSize / sizeof(support::ulittle32_t))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Too many directory blocks.");

  if (SB.BlockMapAddr == 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Block 0 is reserved");

  return Error::success();
}
