//===-- Utils/ELF.h - Common ELF functionality ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common ELF functionality for target plugins.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGIN_COMMON_UTILS_ELF_H
#define OFFLOAD_PLUGIN_COMMON_UTILS_ELF_H

#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"

namespace utils {
namespace elf {

/// Returns true or false if the \p Buffer is an ELF file.
bool isELF(llvm::StringRef Buffer);

/// Returns the ELF e_machine value of the current compilation target.
uint16_t getTargetMachine();

/// Checks if the given \p Object is a valid ELF matching the e_machine value.
llvm::Expected<bool> checkMachine(llvm::StringRef Object, uint16_t EMachine);

/// Returns a pointer to the given \p Symbol inside of an ELF object. Returns
/// a null pointer if the symbol exists but has no contents in the file (its
/// section is SHT_NOBITS, e.g. `.bss`).
llvm::Expected<const void *>
getSymbolAddress(const llvm::object::ELFSymbolRef &Symbol);

/// Returns the symbol associated with the \p Name in the \p ELFObj. It will
/// first search for the hash sections to identify symbols from the hash table.
/// If that fails it will fall back to a linear search in the case of an
/// executable file without a hash table.
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
getSymbol(const llvm::object::ObjectFile &ELFObj, llvm::StringRef Name);

} // namespace elf
} // namespace utils

#endif // OFFLOAD_PLUGIN_COMMON_UTILS_ELF_H