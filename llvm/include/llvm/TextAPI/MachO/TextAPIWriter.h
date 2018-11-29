//===--- TextAPIWriter.h - Text API Writer ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_MACHO_WRITER_H
#define LLVM_TEXTAPI_MACHO_WRITER_H

#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace MachO {

class InterfaceFile;

class TextAPIWriter {
public:
  TextAPIWriter() = delete;

  static Error writeToStream(raw_ostream &os, const InterfaceFile &);
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_MACHO_WRITER_H
