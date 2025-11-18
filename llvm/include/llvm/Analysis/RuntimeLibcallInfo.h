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
  explicit RuntimeLibraryAnalysis(const Triple &T) : LibcallsInfo(T) {}

  LLVM_ABI RTLIB::RuntimeLibcallsInfo run(const Module &M,
                                          ModuleAnalysisManager &);

private:
  friend AnalysisInfoMixin<RuntimeLibraryAnalysis>;
  LLVM_ABI static AnalysisKey Key;

  std::optional<RTLIB::RuntimeLibcallsInfo> LibcallsInfo;
};

class LLVM_ABI RuntimeLibraryInfoWrapper : public ImmutablePass {
  RuntimeLibraryAnalysis RTLA;
  std::optional<RTLIB::RuntimeLibcallsInfo> RTLCI;

public:
  static char ID;
  RuntimeLibraryInfoWrapper();
  explicit RuntimeLibraryInfoWrapper(const Triple &T);
  explicit RuntimeLibraryInfoWrapper(const RTLIB::RuntimeLibcallsInfo &RTLCI);

  const RTLIB::RuntimeLibcallsInfo &getRTLCI(const Module &M) {
    ModuleAnalysisManager DummyMAM;
    RTLCI = RTLA.run(M, DummyMAM);
    return *RTLCI;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

LLVM_ABI ModulePass *createRuntimeLibraryInfoWrapperPass();

} // namespace llvm

#endif
