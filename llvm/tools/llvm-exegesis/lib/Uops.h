//===-- Uops.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A BenchmarkRunner implementation to measure uop decomposition.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_UOPS_H
#define LLVM_TOOLS_LLVM_EXEGESIS_UOPS_H

#include "BenchmarkRunner.h"

namespace exegesis {

class UopsBenchmarkRunner : public BenchmarkRunner {
public:
  using BenchmarkRunner::BenchmarkRunner;
  ~UopsBenchmarkRunner() override;

private:
  const char *getDisplayName() const override;

  llvm::Expected<std::vector<llvm::MCInst>>
  createSnippet(RegisterAliasingTrackerCache &RATC, unsigned Opcode,
                llvm::raw_ostream &Info) const override;

  std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF,
                  const unsigned NumRepetitions) const override;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_UOPS_H
