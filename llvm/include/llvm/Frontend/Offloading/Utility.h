//===- Utility.h - Collection of geneirc offloading utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "llvm/Object/OffloadBinary.h"

namespace llvm {
namespace offloading {

/// Create an offloading section struct used to register this global at
/// runtime.
///
/// Type struct __tgt_offload_entry{
///   void    *addr;      // Pointer to the offload entry info.
///                       // (function or global)
///   char    *name;      // Name of the function or global.
///   size_t  size;       // Size of the entry info (0 if it a function).
///   int32_t flags;
///   int32_t reserved;
/// };
///
/// \param M The module to be used
/// \param Addr The pointer to the global being registered.
/// \param Name The symbol name associated with the global.
/// \param Size The size in bytes of the global (0 for functions).
/// \param Flags Flags associated with the entry.
/// \param SectionName The section this entry will be placed at.
void emitOffloadingEntry(Module &M, Constant *Addr, StringRef Name,
                         uint64_t Size, int32_t Flags, StringRef SectionName);

} // namespace offloading
} // namespace llvm
