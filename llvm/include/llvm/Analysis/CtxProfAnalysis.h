//===- CtxProfAnalysis.h - maintain contextual profile info   -*- C++ ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_ANALYSIS_CTXPROFANALYSIS_H
#define LLVM_ANALYSIS_CTXPROFANALYSIS_H

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include <map>

namespace llvm {
class CtxProfAnalysis : public AnalysisInfoMixin<CtxProfAnalysis> {
  StringRef Profile;
public:
  static AnalysisKey Key;
  explicit CtxProfAnalysis(StringRef Profile) : Profile(Profile) {};

  class Result {
    std::optional<PGOContextualProfile::CallTargetMapTy> Profiles;
    public:
      explicit Result(PGOContextualProfile::CallTargetMapTy &&Profiles)
          : Profiles(std::move(Profiles)) {}
      Result() = default;
      Result(const Result&) = delete;
      Result(Result &&) = default;

      operator bool() const { return !!Profiles; }
      const PGOContextualProfile::CallTargetMapTy &profiles() const {
        return *Profiles;
      }
  };

  Result run(Module &M, ModuleAnalysisManager &MAM);
};

class CtxProfAnalysisPrinterPass
    : public PassInfoMixin<CtxProfAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit CtxProfAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }
};
} // namespace llvm
#endif // LLVM_ANALYSIS_CTXPROFANALYSIS_H
