//===- llvm/DWP/ELFWriter.h - ELF structure writer -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for writing ELF header and section header structures.
// Used by both the MC ELFObjectWriter and the DWP direct ELF writer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWP_ELFWRITER_H
#define LLVM_DWP_ELFWRITER_H

#include "llvm/Support/EndianStream.h"
#include <cstdint>

namespace llvm {
namespace ELF {

/// Write an ELF file header (Elf32_Ehdr or Elf64_Ehdr) for an ET_REL object.
void writeHeader(support::endian::Writer &W, bool Is64Bit, uint8_t OSABI,
                 uint8_t ABIVersion, uint16_t EMachine, uint32_t EFlags,
                 uint64_t SHOff, uint16_t SHNum, uint16_t SHStrNdx);

/// Write a single ELF section header entry (Elf32_Shdr or Elf64_Shdr).
void writeSectionHeader(support::endian::Writer &W, bool Is64Bit, uint32_t Name,
                        uint32_t Type, uint64_t Flags, uint64_t Address,
                        uint64_t Offset, uint64_t Size, uint32_t Link,
                        uint32_t Info, uint64_t Alignment, uint64_t EntrySize);

} // namespace ELF
} // namespace llvm

#endif // LLVM_DWP_ELFWRITER_H
