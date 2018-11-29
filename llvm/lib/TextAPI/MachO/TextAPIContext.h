//===- llvm/TextAPI/YAMLContext.h - YAML Context ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the YAML Context for the TextAPI Reader/Writer
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_MACHO_CONTEXT_H
#define LLVM_TEXTAPI_MACHO_CONTEXT_H

#include "llvm/Support/MemoryBuffer.h"
#include <string>

namespace llvm {
namespace MachO {

enum FileType : unsigned;

struct TextAPIContext {
  std::string ErrorMessage;
  std::string Path;
  FileType FileKind;
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_MACHO_CONTEXT_H
