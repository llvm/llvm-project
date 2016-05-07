//===- StreamInterface.h - Base interface for a PDB stream ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_STREAMINTERFACE_H
#define LLVM_DEBUGINFO_PDB_RAW_STREAMINTERFACE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

#include <stdint.h>

namespace llvm {
namespace pdb {
class StreamInterface {
public:
  virtual ~StreamInterface() {}

  virtual Error readBytes(uint32_t Offset,
                          MutableArrayRef<uint8_t> Buffer) const = 0;
  virtual Error getArrayRef(uint32_t Offset, ArrayRef<uint8_t> &Buffer,
                            uint32_t Length) const = 0;

  virtual uint32_t getLength() const = 0;
};
}
}

#endif
