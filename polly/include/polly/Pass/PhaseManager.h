//===------ PhaseManager.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the sequence of operations on SCoPs, called phases. It is itelf
// not a pass in either pass manager, but used from PollyFunctionPass or
// PollyModulePass.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_PASS_PHASEMANAGER_H_
#define POLLY_PASS_PHASEMANAGER_H_

#include "polly/DependenceInfo.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/IR/PassManager.h"
#include <stddef.h>

namespace llvm {
template <typename EnumT> struct enum_iteration_traits;
} // namespace llvm

namespace polly {
using llvm::Function;
using llvm::StringRef;

/// Phases (in execution order) within the Polly pass.
enum class PassPhase {
  None,

  Prepare,

  Detection,
  PrintDetect,
  DotScops,
  DotScopsOnly,
  ViewScops,
  ViewScopsOnly,

  ScopInfo,
  PrintScopInfo,

  Flatten,

  Dependences,
  PrintDependences,

  ImportJScop,
  Simplify0,
  Optree,
  DeLICM,
  Simplify1,
  DeadCodeElimination,
  MaximumStaticExtension,
  PruneUnprofitable,
  Optimization,
  ExportJScop,
  AstGen,
  CodeGen,

  PassPhaseFirst = Prepare,
  PassPhaseLast = CodeGen
};

StringRef getPhaseName(PassPhase Phase);
PassPhase parsePhase(StringRef Name);
bool dependsOnDependenceInfo(PassPhase Phase);

/// Options for the Polly pass.
class PollyPassOptions {
  /// For each Polly phase, whether it should be executed.
  /// Since PassPhase::None is unused, bit positions are shifted by one.
  llvm::Bitset<static_cast<size_t>(PassPhase::PassPhaseLast) -
               static_cast<size_t>(PassPhase::PassPhaseFirst) + 1>
      PhaseEnabled;

public:
  bool ViewAll = false;
  std::string ViewFilter;
  Dependences::AnalysisLevel PrintDepsAnalysisLevel = Dependences::AL_Statement;

  bool isPhaseEnabled(PassPhase Phase) const {
    assert(Phase != PassPhase::None);
    unsigned BitPos = static_cast<size_t>(Phase) -
                      static_cast<size_t>(PassPhase::PassPhaseFirst);
    return PhaseEnabled[BitPos];
  }

  void setPhaseEnabled(PassPhase Phase, bool Enabled = true) {
    assert(Phase != PassPhase::None);
    unsigned BitPos = static_cast<size_t>(Phase) -
                      static_cast<size_t>(PassPhase::PassPhaseFirst);
    if (Enabled)
      PhaseEnabled.set(BitPos);
    else
      PhaseEnabled.reset(BitPos);
  }

  /// Enable all phases that are necessary for a roundtrip from LLVM-IR back to
  /// LLVM-IR.
  void enableEnd2End();

  /// Enabled the default optimization phases.
  void enableDefaultOpts();

  /// Disable all phases following \p Phase.
  /// Useful when regression testing that particular phase and everything after
  /// it is not of interest.
  void disableAfter(PassPhase Phase);

  /// Check whether the options are coherent relative to each other.
  llvm::Error checkConsistency() const;
};

/// Run Polly and its phases on \p F.
bool runPollyPass(Function &F, llvm::FunctionAnalysisManager &FAM,
                  PollyPassOptions Opts);
} // namespace polly

/// Make llvm::enum_seq<PassPhase> work.
template <> struct llvm::enum_iteration_traits<polly::PassPhase> {
  static constexpr bool is_iterable = true;
};

#endif /* POLLY_PASS_PHASEMANAGER_H_ */
