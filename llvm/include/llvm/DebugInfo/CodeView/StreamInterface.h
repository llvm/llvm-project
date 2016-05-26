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

class StreamInterface {
public:
  virtual ~StreamInterface() {}

  virtual Error readBytes(uint32_t Offset,
                          MutableArrayRef<uint8_t> Buffer) const = 0;
  virtual Error getArrayRef(uint32_t Offset, ArrayRef<uint8_t> &Buffer,
                            uint32_t Length) const = 0;

  virtual uint32_t getLength() const = 0;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_STREAMINTERFACE_H
