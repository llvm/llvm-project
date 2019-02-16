//===- ELFObjHandler.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-----------------------------------------------------------------------===/
///
/// This supports reading and writing of elf dynamic shared objects.
///
//===-----------------------------------------------------------------------===/

#ifndef LLVM_TOOLS_ELFABI_ELFOBJHANDLER_H
#define LLVM_TOOLS_ELFABI_ELFOBJHANDLER_H

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/TextAPI/ELF/ELFStub.h"

namespace llvm {

class MemoryBuffer;

namespace elfabi {

/// Attempt to read a binary ELF file from a MemoryBuffer.
Expected<std::unique_ptr<ELFStub>> readELFFile(MemoryBufferRef Buf);

} // end namespace elfabi
} // end namespace llvm

#endif // LLVM_TOOLS_ELFABI_ELFOBJHANDLER_H
