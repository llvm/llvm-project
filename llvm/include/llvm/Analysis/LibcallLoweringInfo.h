//===- LibcallLoweringInfo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tracks which library function implementations to use for depending on a
// calling context (e.g., which library function a particular subtarget should
// use).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LIBCALLLOWERINGINFO_H
#define LLVM_ANALYSIS_LIBCALLLOWERINGINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/Pass.h"

namespace llvm {

/// Tracks which library functions to use for a particular subtarget or
/// function.
class LibcallLoweringInfo {
private:
  const RTLIB::RuntimeLibcallsInfo &RTLCI;
  /// Stores the implementation choice for each libcall.
  RTLIB::LibcallImpl LibcallImpls[RTLIB::UNKNOWN_LIBCALL + 1] = {
      RTLIB::Unsupported};

public:
  /// Callback applying the caller's context-specific (e.g. subtarget) libcall
  /// rules on top of the module-level defaults.
  using ApplyContextRulesFn = function_ref<void(LibcallLoweringInfo &)>;

  /// Construct the lowering info from the module-level \p RTLCI, seeding the
  /// default implementation for every available libcall, then applying the
  /// optional \p ApplyContextRules callback for the caller's context-specific
  /// libcall rules.
  LLVM_ABI LibcallLoweringInfo(const RTLIB::RuntimeLibcallsInfo &RTLCI,
                               ApplyContextRulesFn ApplyContextRules = {});

  const RTLIB::RuntimeLibcallsInfo &getRuntimeLibcallsInfo() const {
    return RTLCI;
  }

  /// Get the libcall routine name for the specified libcall.
  // FIXME: This should be removed. Only LibcallImpl should have a name.
  const char *getLibcallName(RTLIB::Libcall Call) const {
    // FIXME: Return StringRef
    return RTLIB::RuntimeLibcallsInfo::getLibcallImplName(LibcallImpls[Call])
        .data();
  }

  /// Return the lowering's selection of implementation call for \p Call
  RTLIB::LibcallImpl getLibcallImpl(RTLIB::Libcall Call) const {
    return LibcallImpls[Call];
  }

  /// Remap the default libcall routine name for the specified libcall.
  void setLibcallImpl(RTLIB::Libcall Call, RTLIB::LibcallImpl Impl) {
    LibcallImpls[Call] = Impl;
  }

  // FIXME: Remove this wrapper in favor of directly using
  // getLibcallImplCallingConv
  CallingConv::ID getLibcallCallingConv(RTLIB::Libcall Call) const {
    return RTLCI.LibcallImplCallingConvs[LibcallImpls[Call]];
  }

  /// Get the CallingConv that should be used for the specified libcall.
  CallingConv::ID getLibcallImplCallingConv(RTLIB::LibcallImpl Call) const {
    return RTLCI.LibcallImplCallingConvs[Call];
  }

  /// Return a function impl compatible with RTLIB::MEMCPY, or
  /// RTLIB::Unsupported if fully unsupported.
  RTLIB::LibcallImpl getMemcpyImpl() const {
    RTLIB::LibcallImpl Memcpy = getLibcallImpl(RTLIB::MEMCPY);
    if (Memcpy == RTLIB::Unsupported) {
      // Fallback to memmove if memcpy isn't available.
      return getLibcallImpl(RTLIB::MEMMOVE);
    }

    return Memcpy;
  }
};

/// Records a mapping from an opaque lowering context to its
/// LibcallLoweringInfo.
///
/// The context is identified by an opaque key (which should be
/// TargetSubtargetInfo*)
class ModuleLibcallLoweringInfo {
private:
  using LibcallLoweringMap = DenseMap<const void *, LibcallLoweringInfo>;
  mutable LibcallLoweringMap LoweringMap;
  const RTLIB::RuntimeLibcallsInfo *RTLCI = nullptr;

public:
  ModuleLibcallLoweringInfo() = default;
  ModuleLibcallLoweringInfo(RTLIB::RuntimeLibcallsInfo &RTLCI)
      : RTLCI(&RTLCI) {}

  void init(const RTLIB::RuntimeLibcallsInfo *RT) { RTLCI = RT; }

  void clear() {
    RTLCI = nullptr;
    LoweringMap.clear();
  }

  operator bool() const { return RTLCI != nullptr; }

  LLVM_ABI bool invalidate(Module &, const PreservedAnalyses &,
                           ModuleAnalysisManager::Invalidator &);

  /// Return the LibcallLoweringInfo for the context identified by \p Key,
  /// creating it (via \p ApplyContextRules) on first request. \p Key should be
  /// TargetSubtargetInfo*.
  template <typename KeyT>
  const LibcallLoweringInfo &getLibcallLowering(
      const KeyT *Key,
      LibcallLoweringInfo::ApplyContextRulesFn ApplyContextRules = {}) const {
    return getLibcallLoweringForKey(static_cast<const void *>(Key),
                                    ApplyContextRules);
  }

private:
  const LibcallLoweringInfo &getLibcallLoweringForKey(
      const void *Key,
      LibcallLoweringInfo::ApplyContextRulesFn ApplyContextRules) const {
    auto It = LoweringMap.find(Key);
    if (It != LoweringMap.end())
      return It->second;
    return LoweringMap.try_emplace(Key, *RTLCI, ApplyContextRules)
        .first->second;
  }
};

class LibcallLoweringModuleAnalysis
    : public AnalysisInfoMixin<LibcallLoweringModuleAnalysis> {
private:
  friend AnalysisInfoMixin<LibcallLoweringModuleAnalysis>;
  LLVM_ABI static AnalysisKey Key;

  ModuleLibcallLoweringInfo LibcallLoweringMap;

public:
  using Result = ModuleLibcallLoweringInfo;

  LLVM_ABI Result run(Module &M, ModuleAnalysisManager &);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_LIBCALLLOWERINGINFO_H
