//===- RepeatedPass.h - Runs another pass multiple times --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_REPEATEDPASS_H
#define LLVM_IR_REPEATEDPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// A utility pass template that simply runs another pass multiple times.
///
/// This can be useful when debugging or testing passes. It also serves as an
/// example of how to extend the pass manager in ways beyond composition.
template <typename PassT>
class RepeatedPass : public PassInfoMixin<RepeatedPass<PassT>> {
public:
  RepeatedPass(int Count, PassT &&P)
      : Count(Count), P(std::forward<PassT>(P)) {}

  template <typename IRUnitT, typename AnalysisManagerT, typename... Ts>
  PreservedAnalyses run(IRUnitT &IR, AnalysisManagerT &AM, Ts &&... Args) {

    // Request PassInstrumentation from analysis manager, will use it to run
    // instrumenting callbacks for the passes later.
    // Here we use std::tuple wrapper over getResult which helps to extract
    // AnalysisManager's arguments out of the whole Args set.
    PassInstrumentation PI =
        detail::getAnalysisResult<PassInstrumentationAnalysis>(
            AM, IR, std::tuple<Ts...>(Args...));

    auto PA = PreservedAnalyses::all();
    for (int i = 0; i < Count; ++i) {
      // Check the PassInstrumentation's BeforePass callbacks before running the
      // pass, skip its execution completely if asked to (callback returns
      // false).
      if (!PI.runBeforePass<IRUnitT>(P, IR))
        continue;
      PreservedAnalyses IterPA = P.run(IR, AM, std::forward<Ts>(Args)...);
      PA.intersect(IterPA);
      PI.runAfterPass(P, IR, IterPA);
    }
    return PA;
  }

  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName) {
    OS << "repeat<" << Count << ">(";
    P.printPipeline(OS, MapClassName2PassName);
    OS << ')';
  }

private:
  int Count;
  PassT P;
};

template <typename PassT>
RepeatedPass<PassT> createRepeatedPass(int Count, PassT &&P) {
  return RepeatedPass<PassT>(Count, std::forward<PassT>(P));
}

} // end namespace llvm

#endif // LLVM_IR_REPEATEDPASS_H
