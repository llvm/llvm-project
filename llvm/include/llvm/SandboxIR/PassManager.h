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
  SmallVector<ContainedPass *> Passes;

  PassManager(StringRef Name) : ParentPass(Name) {}
  PassManager(const PassManager &) = delete;
  virtual ~PassManager() = default;
  PassManager &operator=(const PassManager &) = delete;

public:
  /// Adds \p Pass to the pass pipeline.
  void addPass(ContainedPass *Pass) {
    // TODO: Check that Pass's class type works with this PassManager type.
    Passes.push_back(Pass);
  }
#ifndef NDEBUG
  void print(raw_ostream &OS) const override {
    OS << this->getName();
    OS << "(";
    // TODO: This should call Pass->print(OS) because Pass may be a PM.
    interleave(Passes, OS, [&OS](auto *Pass) { OS << Pass->getName(); }, ",");
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

/// Owns the passes and provides an API to get a pass by its name.
class PassRegistry {
  SmallVector<std::unique_ptr<Pass>, 8> Passes;
  DenseMap<StringRef, Pass *> NameToPassMap;

public:
  static constexpr const char PassDelimToken = ',';
  PassRegistry() = default;
  /// Registers \p PassPtr and takes ownership.
  Pass &registerPass(std::unique_ptr<Pass> &&PassPtr) {
    auto &PassRef = *PassPtr.get();
    NameToPassMap[PassRef.getName()] = &PassRef;
    Passes.push_back(std::move(PassPtr));
    return PassRef;
  }
  /// \Returns the pass with name \p Name, or null if not registered.
  Pass *getPassByName(StringRef Name) const {
    auto It = NameToPassMap.find(Name);
    return It != NameToPassMap.end() ? It->second : nullptr;
  }
  /// Creates a pass pipeline and returns the first pass manager.
  FunctionPassManager &parseAndCreatePassPipeline(StringRef Pipeline);

#ifndef NDEBUG
  void print(raw_ostream &OS) const {
    for (const auto &PassPtr : Passes)
      OS << PassPtr->getName() << "\n";
  }
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_PASSMANAGER_H
