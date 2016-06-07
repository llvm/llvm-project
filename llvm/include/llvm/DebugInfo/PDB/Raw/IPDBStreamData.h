//===- IPDBStreamData.h - Base interface for PDB Stream Data ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_IPDBSTREAMDATA_H
#define LLVM_DEBUGINFO_PDB_RAW_IPDBSTREAMDATA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace pdb {
/// IPDBStream abstracts the notion of PDB stream data.  Although we already
/// have another stream abstraction (namely in the form of StreamInterface
/// and MappedBlockStream), they assume that the stream data is referenced
/// the same way.  Namely, by looking in the directory to get the list of
/// stream blocks, and by looking in the array of stream lengths to get the
/// length.  This breaks down for the directory itself, however, since its
/// length and list of blocks are stored elsewhere.  By abstracting the
/// notion of stream data further, we can use a MappedBlockStream to read
/// from the directory itself, or from an indexed stream which references
/// the directory.
class IPDBStreamData {
public:
  virtual ~IPDBStreamData() {}

  virtual uint32_t getLength() = 0;
  virtual ArrayRef<support::ulittle32_t> getStreamBlocks() = 0;
};
}
}

#endif
