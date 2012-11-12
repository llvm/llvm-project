//===------ utils/obj2yaml.hpp - obj2yaml conversion tool -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// This file declares some helper routines, and also the format-specific
// writers. To add a new format, add the declaration here, and, in a separate
// source file, implement it.
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_OBJ2YAML_H
#define LLVM_UTILS_OBJ2YAML_H

#include "llvm/ADT/ArrayRef.h"

#include "llvm/Support/system_error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"

namespace yaml {  // routines for writing YAML
// Write a hex stream:
//    <Prefix> !hex: "<hex digits>" #|<ASCII chars>\n
  llvm::raw_ostream &writeHexStream
    (llvm::raw_ostream &Out, const llvm::ArrayRef<uint8_t> arr);

// Writes a number in hex; prefix it by 0x if it is >= 10
  llvm::raw_ostream &writeHexNumber
    (llvm::raw_ostream &Out, unsigned long long N);
}

llvm::error_code coff2yaml(llvm::raw_ostream &Out, llvm::MemoryBuffer *TheObj);

#endif
