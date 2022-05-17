//===-- Procfs.h ---------------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// source/Plugins/Process/Linux/Procfs.h defines the symbols we need from
// sys/procfs.h on Android/Linux for all supported architectures.

#include <sys/ptrace.h>

#include "lldb/lldb-types.h"

#include "llvm/Support/Error.h"

#include <vector>

#ifdef __ANDROID__
#if defined(__arm64__) || defined(__aarch64__)
typedef unsigned long elf_greg_t;
typedef elf_greg_t
    elf_gregset_t[(sizeof(struct user_pt_regs) / sizeof(elf_greg_t))];
typedef struct user_fpsimd_state elf_fpregset_t;
#ifndef NT_FPREGSET
#define NT_FPREGSET NT_PRFPREG
#endif // NT_FPREGSET
#elif defined(__mips__)
#ifndef NT_FPREGSET
#define NT_FPREGSET NT_PRFPREG
#endif // NT_FPREGSET
#endif
#else // __ANDROID__
#include <sys/procfs.h>
#endif // __ANDROID__

namespace lldb_private {
namespace process_linux {

/// \return
///     The content of /proc/cpuinfo and cache it if errors didn't happen.
llvm::Expected<llvm::ArrayRef<uint8_t>> GetProcfsCpuInfo();

/// \return
///     A list of available logical core ids given the contents of
///     /proc/cpuinfo.
llvm::Expected<std::vector<lldb::core_id_t>>
GetAvailableLogicalCoreIDs(llvm::StringRef cpuinfo);

/// \return
///     A list with all the logical cores available in the system and cache it
///     if errors didn't happen.
llvm::Expected<llvm::ArrayRef<lldb::core_id_t>> GetAvailableLogicalCoreIDs();

} // namespace process_linux
} // namespace lldb_private
