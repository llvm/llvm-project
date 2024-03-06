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

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_ELF_UTILS_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_ELF_UTILS_H

#include "Shared/PluginAPI.h"

#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"

namespace utils {
namespace elf {

/// Returns true or false if the \p Buffer is an ELF file.
bool isELF(llvm::StringRef Buffer);

/// Checks if the given \p Object is a valid ELF matching the e_machine value.
llvm::Expected<bool> checkMachine(llvm::StringRef Object, uint16_t EMachine);

/// Returns the symbol associated with the \p Name in the \p Obj. It will
/// first search for the hash sections to identify symbols from the hash table.
/// If that fails it will fall back to a linear search in the case of an
/// executable file without a hash table.  If the symbol is found, it returns
/// a StringRef covering the symbol's data in the Obj buffer, based on its
/// address and size; otherwise, it returns std::nullopt.
llvm::Expected<std::optional<llvm::StringRef>>
findSymbolInImage(const llvm::StringRef Obj, llvm::StringRef Name);

} // namespace elf
} // namespace utils

#endif // LLVM_OPENMP_LIBOMPTARGET_PLUGINS_ELF_UTILS_H
