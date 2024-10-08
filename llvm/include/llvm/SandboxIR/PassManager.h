//===- PassManager.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers and executes the Sandbox IR passes.
//
// The pass manager contains an ordered sequence of passes that it runs in
// order. The passes are owned by the PassRegistry, not by the PassManager.
//
// Note that in this design a pass manager is also a pass. So a pass manager
// runs when it is it's turn to run in its parent pass-manager pass pipeline.
//

#ifndef LLVM_SANDBOXIR_PASSMANAGER_H
#define LLVM_SANDBOXIR_PASSMANAGER_H

#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/Support/Debug.h"

namespace llvm::sandboxir {

class Value;

/// Base class.
template <typename ParentPass, typename ContainedPass>
class PassManager : public ParentPass {
protected:
  /// The list of passes that this pass manager will run.
  SmallVector<std::unique_ptr<ContainedPass>> Passes;

  PassManager(StringRef Name) : ParentPass(Name) {}
  PassManager(const PassManager &) = delete;
  PassManager(PassManager &&) = default;
  virtual ~PassManager() = default;
  PassManager &operator=(const PassManager &) = delete;

public:
  /// Adds \p Pass to the pass pipeline.
  void addPass(std::unique_ptr<ContainedPass> Pass) {
    // TODO: Check that Pass's class type works with this PassManager type.
    Passes.push_back(std::move(Pass));
  }
#ifndef NDEBUG
  void print(raw_ostream &OS) const override {
    OS << this->getName();
    OS << "(";
    // TODO: This should call Pass->print(OS) because Pass may be a PM.
    interleave(Passes, OS, [&OS](auto &Pass) { OS << Pass->getName(); }, ",");
    OS << ")";
  }
  LLVM_DUMP_METHOD void dump() const override {
    print(dbgs());
    dbgs() << "\n";
  }
#endif
  /// Similar to print() but prints one pass per line. Used for testing.
  void printPipeline(raw_ostream &OS) const {
    OS << this->getName() << "\n";
    for (const auto &PassPtr : Passes)
      PassPtr->printPipeline(OS);
  }
};

class FunctionPassManager final
    : public PassManager<FunctionPass, FunctionPass> {
public:
  FunctionPassManager(StringRef Name) : PassManager(Name) {}
  bool runOnFunction(Function &F) final;
};

class RegionPassManager final : public PassManager<RegionPass, RegionPass> {
public:
  RegionPassManager(StringRef Name) : PassManager(Name) {}
  bool runOnRegion(Region &R) final;
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_PASSMANAGER_H
