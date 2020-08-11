//===- ELFObjHandler.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
