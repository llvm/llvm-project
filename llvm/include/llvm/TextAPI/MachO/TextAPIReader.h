//===--- TextAPIReader.h - Text API Reader ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_MACHO_READER_H
#define LLVM_TEXTAPI_MACHO_READER_H

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace MachO {

class InterfaceFile;

class TextAPIReader {
public:
  static Expected<std::unique_ptr<InterfaceFile>>
  get(std::unique_ptr<MemoryBuffer> InputBuffer);

  static Expected<std::unique_ptr<InterfaceFile>>
  getUnmanaged(llvm::MemoryBuffer *InputBuffer);

  TextAPIReader() = delete;
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_MACHO_READER_H
