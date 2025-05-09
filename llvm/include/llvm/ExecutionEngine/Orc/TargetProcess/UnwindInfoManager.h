//===--- UnwindInfoManager.h -- Register unwind info sections ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for managing eh-frame and compact-unwind registration and lookup
// through libunwind's find_dynamic_unwind_sections mechanism.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_UNWINDINFOMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_UNWINDINFOMANAGER_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Error.h"
#include <map>
#include <mutex>

namespace llvm::orc {

class UnwindInfoManager {
public:
  // This struct's layout should match the unw_dynamic_unwind_sections struct
  // from libunwind/src/libunwid_ext.h.
  struct UnwindSections {
    uintptr_t dso_base;
    uintptr_t dwarf_section;
    size_t dwarf_section_length;
    uintptr_t compact_unwind_section;
    size_t compact_unwind_section_length;
  };

  UnwindInfoManager(UnwindInfoManager &&) = delete;
  UnwindInfoManager &operator=(UnwindInfoManager &&) = delete;
  ~UnwindInfoManager();

  /// If the libunwind find-dynamic-unwind-info callback registration APIs are
  /// available then this method will instantiate a global UnwindInfoManager
  /// instance suitable for the process and return true. Otherwise it will
  /// return false.
  static bool TryEnable();

  static void addBootstrapSymbols(StringMap<ExecutorAddr> &M);

  static Error registerSections(ArrayRef<orc::ExecutorAddrRange> CodeRanges,
                                orc::ExecutorAddr DSOBase,
                                orc::ExecutorAddrRange DWARFEHFrame,
                                orc::ExecutorAddrRange CompactUnwind);

  static Error deregisterSections(ArrayRef<orc::ExecutorAddrRange> CodeRanges);

private:
  UnwindInfoManager() = default;

  int findSectionsImpl(uintptr_t Addr, UnwindSections *Info);
  static int findSections(uintptr_t Addr, UnwindSections *Info);

  Error registerSectionsImpl(ArrayRef<orc::ExecutorAddrRange> CodeRanges,
                             orc::ExecutorAddr DSOBase,
                             orc::ExecutorAddrRange DWARFEHFrame,
                             orc::ExecutorAddrRange CompactUnwind);

  Error deregisterSectionsImpl(ArrayRef<orc::ExecutorAddrRange> CodeRanges);

  std::mutex M;
  std::map<uintptr_t, UnwindSections> UWSecs;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_UNWINDINFOMANAGER_H
