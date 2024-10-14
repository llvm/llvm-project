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

  using CreatePassFunc =
      std::function<std::unique_ptr<ContainedPass>(StringRef)>;

  /// Parses \p Pipeline as a comma-separated sequence of pass names and sets
  /// the pass pipeline, using \p CreatePass to instantiate passes by name.
  ///
  /// After calling this function, the PassManager contains only the specified
  /// pipeline, any previously added passes are cleared.
  void setPassPipeline(StringRef Pipeline, CreatePassFunc CreatePass) {
    static constexpr const char EndToken = '\0';
    static constexpr const char PassDelimToken = ',';

    assert(Passes.empty() &&
           "setPassPipeline called on a non-empty sandboxir::PassManager");
    // Add EndToken to the end to ease parsing.
    std::string PipelineStr = std::string(Pipeline) + EndToken;
    int FlagBeginIdx = 0;

    for (auto [Idx, C] : enumerate(PipelineStr)) {
      // Keep moving Idx until we find the end of the pass name.
      bool FoundDelim = C == EndToken || C == PassDelimToken;
      if (!FoundDelim)
        continue;
      unsigned Sz = Idx - FlagBeginIdx;
      std::string PassName(&PipelineStr[FlagBeginIdx], Sz);
      FlagBeginIdx = Idx + 1;

      // Get the pass that corresponds to PassName and add it to the pass
      // manager.
      auto Pass = CreatePass(PassName);
      if (Pass == nullptr) {
        errs() << "Pass '" << PassName << "' not registered!\n";
        exit(1);
      }
      addPass(std::move(Pass));
    }
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
