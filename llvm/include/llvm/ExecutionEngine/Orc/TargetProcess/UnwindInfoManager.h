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
#include "llvm/ExecutionEngine/Orc/TargetProcess/ExecutorBootstrapService.h"
#include "llvm/Support/Error.h"
#include <map>
#include <mutex>

namespace llvm::orc {

class UnwindInfoManager : public ExecutorBootstrapService {
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

  /// If the libunwind find-dynamic-unwind-info callback registration APIs are
  /// available then this method will return an UnwindInfoManager instance,
  /// otherwise it will return nullptr.
  static std::unique_ptr<UnwindInfoManager> TryCreate();

  Error shutdown() override;
  void addBootstrapSymbols(StringMap<ExecutorAddr> &M) override;

  Error enable(void *FindDynamicUnwindSections);
  Error disable(void);

  Error registerSections(ArrayRef<orc::ExecutorAddrRange> CodeRanges,
                         orc::ExecutorAddr DSOBase,
                         orc::ExecutorAddrRange DWARFEHFrame,
                         orc::ExecutorAddrRange CompactUnwind);

  Error deregisterSections(ArrayRef<orc::ExecutorAddrRange> CodeRanges);

  int findSections(uintptr_t Addr, UnwindSections *Info);

private:
  UnwindInfoManager(int (*AddFindDynamicUnwindSections)(void *),
                    int (*RemoveFindDynamicUnwindSections)(void *))
      : AddFindDynamicUnwindSections(AddFindDynamicUnwindSections),
        RemoveFindDynamicUnwindSections(RemoveFindDynamicUnwindSections) {}

  static int findSectionsHelper(UnwindInfoManager *Instance, uintptr_t Addr,
                                UnwindSections *Info);

  std::mutex M;
  std::map<uintptr_t, UnwindSections> UWSecs;

  int (*AddFindDynamicUnwindSections)(void *) = nullptr;
  int (*RemoveFindDynamicUnwindSections)(void *) = nullptr;
  void *FindDynamicUnwindSections = nullptr;

  static const char *AddFnName, *RemoveFnName;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_UNWINDINFOMANAGER_H
