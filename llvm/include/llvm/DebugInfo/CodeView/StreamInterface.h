//===- StreamInterface.h - Base interface for a stream of data --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_STREAMINTERFACE_H
#define LLVM_DEBUGINFO_CODEVIEW_STREAMINTERFACE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
namespace codeview {

/// StreamInterface abstracts the notion of a data stream.  This way, an
/// implementation could implement trivial reading from a contiguous memory
/// buffer or, as in the case of PDB files, reading from a set of possibly
/// discontiguous blocks.  The implementation is required to return references
/// to stable memory, so if this is not possible (for example in the case of
/// a PDB file with discontiguous blocks, it must keep its own pool of temp
/// storage.
class StreamInterface {
public:
  virtual ~StreamInterface() {}

  virtual Error readBytes(uint32_t Offset, uint32_t Size,
                          ArrayRef<uint8_t> &Buffer) const = 0;

  virtual uint32_t getLength() const = 0;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_STREAMINTERFACE_H
