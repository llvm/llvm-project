//===-------- EdgeBundles.h - Bundles of CFG edges --------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The EdgeBundles analysis forms equivalence classes of CFG edges such that all
// edges leaving a machine basic block are in the same bundle, and all edges
// entering a machine basic block are in the same bundle.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EDGEBUNDLES_H
#define LLVM_CODEGEN_EDGEBUNDLES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class EdgeBundlesWrapperLegacy;
class EdgeBundlesAnalysis;

class EdgeBundles {
  friend class EdgeBundlesWrapperLegacy;
  friend class EdgeBundlesAnalysis;

  const MachineFunction *MF = nullptr;

  /// EC - Each edge bundle is an equivalence class. The keys are:
  ///   2*BB->getNumber()   -> Ingoing bundle.
  ///   2*BB->getNumber()+1 -> Outgoing bundle.
  IntEqClasses EC;

  /// Blocks - Map each bundle to a list of basic block numbers.
  SmallVector<SmallVector<unsigned, 8>, 4> Blocks;

  void init();
  EdgeBundles(MachineFunction &MF);

public:
  /// getBundle - Return the ingoing (Out = false) or outgoing (Out = true)
  /// bundle number for basic block #N
  unsigned getBundle(unsigned N, bool Out) const { return EC[2 * N + Out]; }

  /// getNumBundles - Return the total number of bundles in the CFG.
  unsigned getNumBundles() const { return EC.getNumClasses(); }

  /// getBlocks - Return an array of blocks that are connected to Bundle.
  ArrayRef<unsigned> getBlocks(unsigned Bundle) const { return Blocks[Bundle]; }

  /// getMachineFunction - Return the last machine function computed.
  const MachineFunction *getMachineFunction() const { return MF; }

  /// view - Visualize the annotated bipartite CFG with Graphviz.
  void view() const;

  // Handle invalidation for the new pass manager
  bool invalidate(MachineFunction &MF, const PreservedAnalyses &PA,
                  MachineFunctionAnalysisManager::Invalidator &Inv);
};

class EdgeBundlesWrapperLegacy : public MachineFunctionPass {
public:
  static char ID;
  EdgeBundlesWrapperLegacy() : MachineFunctionPass(ID) {}

  EdgeBundles &getEdgeBundles() { return *Impl; }
  const EdgeBundles &getEdgeBundles() const { return *Impl; }

private:
  std::unique_ptr<EdgeBundles> Impl;
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage&) const override;
};

class EdgeBundlesAnalysis : public AnalysisInfoMixin<EdgeBundlesAnalysis> {
  friend AnalysisInfoMixin<EdgeBundlesAnalysis>;
  static AnalysisKey Key;

public:
  using Result = EdgeBundles;
  EdgeBundles run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif
