//===-- RuntimeLibcallInfo.h - Runtime library information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_RUNTIMELIBCALLINFO_H
#define LLVM_ANALYSIS_RUNTIMELIBCALLINFO_H

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/Pass.h"

namespace llvm {

class LLVM_ABI RuntimeLibraryAnalysis
    : public AnalysisInfoMixin<RuntimeLibraryAnalysis> {
public:
  using Result = RTLIB::RuntimeLibcallsInfo;

  RuntimeLibraryAnalysis() = default;
  RuntimeLibraryAnalysis(RTLIB::RuntimeLibcallsInfo &&BaselineInfoImpl)
      : LibcallsInfo(std::move(BaselineInfoImpl)) {}
  RuntimeLibraryAnalysis(
      const Triple &TT,
      ExceptionHandling ExceptionModel = ExceptionHandling::None,
      FloatABI::ABIType FloatABI = FloatABI::Default,
      EABI EABIVersion = EABI::Default, StringRef ABIName = "",
      VectorLibrary VecLib = VectorLibrary::NoLibrary);

  RTLIB::RuntimeLibcallsInfo run(const Module &M, ModuleAnalysisManager &);

  operator bool() const { return LibcallsInfo.has_value(); }

private:
  friend AnalysisInfoMixin<RuntimeLibraryAnalysis>;
  static AnalysisKey Key;

  std::optional<RTLIB::RuntimeLibcallsInfo> LibcallsInfo;
};

class LLVM_ABI RuntimeLibraryInfoWrapper : public ImmutablePass {
  RuntimeLibraryAnalysis RTLA;
  std::optional<RTLIB::RuntimeLibcallsInfo> RTLCI;

public:
  static char ID;
  RuntimeLibraryInfoWrapper();
  RuntimeLibraryInfoWrapper(
      const Triple &TT,
      ExceptionHandling ExceptionModel = ExceptionHandling::None,
      FloatABI::ABIType FloatABI = FloatABI::Default,
      EABI EABIVersion = EABI::Default, StringRef ABIName = "",
      VectorLibrary VecLib = VectorLibrary::NoLibrary);

  const RTLIB::RuntimeLibcallsInfo &getRTLCI(const Module &M) {
    if (!RTLCI) {
      ModuleAnalysisManager DummyMAM;
      RTLCI = RTLA.run(M, DummyMAM);
    }

    return *RTLCI;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

LLVM_ABI ModulePass *createRuntimeLibraryInfoWrapperPass();

} // namespace llvm

#endif
