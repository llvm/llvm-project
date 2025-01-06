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
public:
  // CreatePassFunc(StringRef PassName, StringRef PassArgs).
  using CreatePassFunc =
      std::function<std::unique_ptr<ContainedPass>(StringRef, StringRef)>;

protected:
  /// The list of passes that this pass manager will run.
  SmallVector<std::unique_ptr<ContainedPass>> Passes;

  PassManager(StringRef Name) : ParentPass(Name) {}
  PassManager(StringRef Name, StringRef Pipeline, CreatePassFunc CreatePass)
      : ParentPass(Name) {
    setPassPipeline(Pipeline, CreatePass);
  }
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

  /// Parses \p Pipeline as a comma-separated sequence of pass names and sets
  /// the pass pipeline, using \p CreatePass to instantiate passes by name.
  ///
  /// Passes can have arguments, for example:
  ///   "pass1<arg1,arg2>,pass2,pass3<arg3,arg4>"
  ///
  /// The arguments between angle brackets are treated as a mostly opaque string
  /// and each pass is responsible for parsing its arguments. The exception to
  /// this are nested angle brackets, which must match pair-wise to allow
  /// arguments to contain nested pipelines, like:
  ///
  ///   "pass1<subpass1,subpass2<arg1,arg2>,subpass3>"
  ///
  /// An empty args string is treated the same as no args, so "pass" and
  /// "pass<>" are equivalent.
  void setPassPipeline(StringRef Pipeline, CreatePassFunc CreatePass) {
    static constexpr const char EndToken = '\0';
    static constexpr const char BeginArgsToken = '<';
    static constexpr const char EndArgsToken = '>';
    static constexpr const char PassDelimToken = ',';

    assert(Passes.empty() &&
           "setPassPipeline called on a non-empty sandboxir::PassManager");

    // Accept an empty pipeline as a special case. This can be useful, for
    // example, to test conversion to SandboxIR without running any passes on
    // it.
    if (Pipeline.empty())
      return;

    // Add EndToken to the end to ease parsing.
    std::string PipelineStr = std::string(Pipeline) + EndToken;
    Pipeline = StringRef(PipelineStr);

    auto AddPass = [this, CreatePass](StringRef PassName, StringRef PassArgs) {
      if (PassName.empty()) {
        errs() << "Found empty pass name.\n";
        exit(1);
      }
      // Get the pass that corresponds to PassName and add it to the pass
      // manager.
      auto Pass = CreatePass(PassName, PassArgs);
      if (Pass == nullptr) {
        errs() << "Pass '" << PassName << "' not registered!\n";
        exit(1);
      }
      addPass(std::move(Pass));
    };

    enum class State {
      ScanName,  // reading a pass name
      ScanArgs,  // reading a list of args
      ArgsEnded, // read the last '>' in an args list, must read delimiter next
    } CurrentState = State::ScanName;
    int PassBeginIdx = 0;
    int ArgsBeginIdx;
    StringRef PassName;
    int NestedArgs = 0;
    for (auto [Idx, C] : enumerate(Pipeline)) {
      switch (CurrentState) {
      case State::ScanName:
        if (C == BeginArgsToken) {
          // Save pass name for later and begin scanning args.
          PassName = Pipeline.slice(PassBeginIdx, Idx);
          ArgsBeginIdx = Idx + 1;
          ++NestedArgs;
          CurrentState = State::ScanArgs;
          break;
        }
        if (C == EndArgsToken) {
          errs() << "Unexpected '>' in pass pipeline.\n";
          exit(1);
        }
        if (C == EndToken || C == PassDelimToken) {
          // Delimiter found, add the pass (with empty args), stay in the
          // ScanName state.
          AddPass(Pipeline.slice(PassBeginIdx, Idx), StringRef());
          PassBeginIdx = Idx + 1;
        }
        break;
      case State::ScanArgs:
        // While scanning args, we only care about making sure nesting of angle
        // brackets is correct.
        if (C == BeginArgsToken) {
          ++NestedArgs;
          break;
        }
        if (C == EndArgsToken) {
          --NestedArgs;
          if (NestedArgs == 0) {
            // Done scanning args.
            AddPass(PassName, Pipeline.slice(ArgsBeginIdx, Idx));
            CurrentState = State::ArgsEnded;
          } else if (NestedArgs < 0) {
            errs() << "Unexpected '>' in pass pipeline.\n";
            exit(1);
          }
          break;
        }
        if (C == EndToken) {
          errs() << "Missing '>' in pass pipeline. End-of-string reached while "
                    "reading arguments for pass '"
                 << PassName << "'.\n";
          exit(1);
        }
        break;
      case State::ArgsEnded:
        // Once we're done scanning args, only a delimiter is valid. This avoids
        // accepting strings like "foo<args><more-args>" or "foo<args>bar".
        if (C == EndToken || C == PassDelimToken) {
          PassBeginIdx = Idx + 1;
          CurrentState = State::ScanName;
        } else {
          errs() << "Expected delimiter or end-of-string after pass "
                    "arguments.\n";
          exit(1);
        }
        break;
      }
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
  void printPipeline(raw_ostream &OS) const override {
    OS << this->getName() << "\n";
    for (const auto &PassPtr : Passes)
      PassPtr->printPipeline(OS);
  }
};

class FunctionPassManager final
    : public PassManager<FunctionPass, FunctionPass> {
public:
  FunctionPassManager(StringRef Name) : PassManager(Name) {}
  FunctionPassManager(StringRef Name, StringRef Pipeline,
                      CreatePassFunc CreatePass)
      : PassManager(Name, Pipeline, CreatePass) {}
  bool runOnFunction(Function &F, const Analyses &A) final;
};

class RegionPassManager final : public PassManager<RegionPass, RegionPass> {
public:
  RegionPassManager(StringRef Name) : PassManager(Name) {}
  RegionPassManager(StringRef Name, StringRef Pipeline,
                    CreatePassFunc CreatePass)
      : PassManager(Name, Pipeline, CreatePass) {}
  bool runOnRegion(Region &R, const Analyses &A) final;
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_PASSMANAGER_H
