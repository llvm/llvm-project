//===- LibcallLoweringInfo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIBCALLLOWERINGINFO_H
#define LLVM_CODEGEN_LIBCALLLOWERINGINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/Pass.h"

namespace llvm {
class RuntimeLibraryInfoWrapper;
class TargetSubtargetInfo;
class TargetMachine;

/// Tracks which library functions to use for a particular subtarget.
class LibcallLoweringInfo {
private:
  const RTLIB::RuntimeLibcallsInfo &RTLCI;
  /// Stores the implementation choice for each each libcall.
  RTLIB::LibcallImpl LibcallImpls[RTLIB::UNKNOWN_LIBCALL + 1] = {
      RTLIB::Unsupported};

public:
  LLVM_ABI LibcallLoweringInfo(const RTLIB::RuntimeLibcallsInfo &RTLCI,
                               const TargetSubtargetInfo &Subtarget);

  const RTLIB::RuntimeLibcallsInfo &getRuntimeLibcallsInfo() const {
    return RTLCI;
  }

  /// Get the libcall routine name for the specified libcall.
  // FIXME: This should be removed. Only LibcallImpl should have a name.
  LLVM_ABI const char *getLibcallName(RTLIB::Libcall Call) const {
    // FIXME: Return StringRef
    return RTLIB::RuntimeLibcallsInfo::getLibcallImplName(LibcallImpls[Call])
        .data();
  }

  /// Return the lowering's selection of implementation call for \p Call
  LLVM_ABI RTLIB::LibcallImpl getLibcallImpl(RTLIB::Libcall Call) const {
    return LibcallImpls[Call];
  }

  /// Rename the default libcall routine name for the specified libcall.
  LLVM_ABI void setLibcallImpl(RTLIB::Libcall Call, RTLIB::LibcallImpl Impl) {
    LibcallImpls[Call] = Impl;
  }

  // FIXME: Remove this wrapper in favor of directly using
  // getLibcallImplCallingConv
  LLVM_ABI CallingConv::ID getLibcallCallingConv(RTLIB::Libcall Call) const {
    return RTLCI.LibcallImplCallingConvs[LibcallImpls[Call]];
  }

  /// Get the CallingConv that should be used for the specified libcall.
  LLVM_ABI CallingConv::ID
  getLibcallImplCallingConv(RTLIB::LibcallImpl Call) const {
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

/// Record a mapping from subtarget to LibcallLoweringInfo.
class LibcallLoweringModuleAnalysisResult {
private:
  using LibcallLoweringMap =
      DenseMap<const TargetSubtargetInfo *, LibcallLoweringInfo>;
  mutable LibcallLoweringMap LoweringMap;
  const RTLIB::RuntimeLibcallsInfo *RTLCI = nullptr;

public:
  LibcallLoweringModuleAnalysisResult() = default;
  LibcallLoweringModuleAnalysisResult(RTLIB::RuntimeLibcallsInfo &RTLCI)
      : RTLCI(&RTLCI) {}

  void init(const RTLIB::RuntimeLibcallsInfo *RT) { RTLCI = RT; }

  void clear() {
    RTLCI = nullptr;
    LoweringMap.clear();
  }

  operator bool() const { return RTLCI != nullptr; }

  LLVM_ABI bool invalidate(Module &, const PreservedAnalyses &,
                           ModuleAnalysisManager::Invalidator &);

  const LibcallLoweringInfo &
  getLibcallLowering(const TargetSubtargetInfo &Subtarget) const {
    return LoweringMap.try_emplace(&Subtarget, *RTLCI, Subtarget).first->second;
  }
};

class LibcallLoweringModuleAnalysis
    : public AnalysisInfoMixin<LibcallLoweringModuleAnalysis> {
private:
  friend AnalysisInfoMixin<LibcallLoweringModuleAnalysis>;
  LLVM_ABI static AnalysisKey Key;

  LibcallLoweringModuleAnalysisResult LibcallLoweringMap;

public:
  using Result = LibcallLoweringModuleAnalysisResult;

  LLVM_ABI Result run(Module &M, ModuleAnalysisManager &);
};

class LLVM_ABI LibcallLoweringInfoWrapper : public ImmutablePass {
  LibcallLoweringModuleAnalysisResult Result;
  RuntimeLibraryInfoWrapper *RuntimeLibcallsWrapper = nullptr;

public:
  static char ID;
  LibcallLoweringInfoWrapper();

  const LibcallLoweringInfo &
  getLibcallLowering(const Module &M, const TargetSubtargetInfo &Subtarget) {
    return getResult(M).getLibcallLowering(Subtarget);
  }

  const LibcallLoweringModuleAnalysisResult &getResult(const Module &M) {
    if (!Result)
      Result.init(&RuntimeLibcallsWrapper->getRTLCI(M));
    return Result;
  }

  void initializePass() override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_LIBCALLLOWERINGINFO_H
