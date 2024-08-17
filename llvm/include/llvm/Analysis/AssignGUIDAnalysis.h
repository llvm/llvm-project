//===- AssignGUIDAnalysis.h - assign a GUID to each GV   ------*- C++ ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_ANALYSIS_ASSIGNGUIDANALYSIS_H
#define LLVM_ANALYSIS_ASSIGNGUIDANALYSIS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
/// Assign a GUID to functions as metadata. GUID calculation takes linkage into
/// account, which may change especially through and after thinlto. By
/// pre-computing and assigning as metadata, this mechanism is resilient to such
/// changes (as well as name changes e.g. suffix ".llvm." additions).
class AssignGUIDAnalysis : public AnalysisInfoMixin<AssignGUIDAnalysis> {
public:
  explicit AssignGUIDAnalysis() = default;
  static AnalysisKey Key;

  class Result {
    friend class AssignGUIDAnalysis;
    Module &M;
    Result(Module &M);

  public:
    void generateGuidTable();
  };
  /// Assign a GUID *if* one is not already assign, as a function metadata named
  /// `GUIDMetadataName`.
  Result run(Module &M, ModuleAnalysisManager &MAM);
};
} // namespace llvm
#endif