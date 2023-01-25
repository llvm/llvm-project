//===- CFGBuilder.h - CFG building and updating utility ----------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// CFGBuilders provides utilities fo building and updating CFG for testing
/// purposes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_CFG_BUILDER_H
#define LLVM_UNITTESTS_CFG_BUILDER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace llvm {

class LLVMContext;
class Module;
class Function;
class BasicBlock;
class raw_ostream;

struct CFGHolder {
  std::unique_ptr<LLVMContext> Context;
  std::unique_ptr<Module> M;
  Function *F;

  CFGHolder(StringRef ModuleName = "m", StringRef FunctionName = "foo");
  ~CFGHolder(); // Defined in the .cpp file so we can use forward declarations.
};

/// \brief
/// CFGBuilder builds IR with specific CFG, based on the supplied list of arcs.
/// It's able to apply the provided updates and automatically modify the IR.
///
/// Internally it makes every basic block end with either SwitchInst or with
/// UnreachableInst. When all arc to a BB are deleted, the BB remains in the
/// function and doesn't get deleted.
///
class CFGBuilder {
public:
  struct Arc {
    StringRef From;
    StringRef To;

    friend bool operator<(const Arc &LHS, const Arc &RHS) {
      return std::tie(LHS.From, LHS.To) <
             std::tie(RHS.From, RHS.To);
    }
  };

  enum class ActionKind { Insert, Delete };
  struct Update {
    ActionKind Action;
    Arc Edge;
  };

  CFGBuilder(Function *F, const std::vector<Arc> &InitialArcs,
             std::vector<Update> Updates);

  BasicBlock *getOrAddBlock(StringRef BlockName);
  std::optional<Update> getNextUpdate() const;
  std::optional<Update> applyUpdate();
  void dump(raw_ostream &OS = dbgs()) const;

private:
  void buildCFG(const std::vector<Arc> &Arcs);
  bool connect(const Arc &A);
  bool disconnect(const Arc &A);

  Function *F;
  unsigned UpdateIdx = 0;
  StringMap<BasicBlock *> NameToBlock;
  std::set<Arc> Arcs;
  std::vector<Update> Updates;
};

} // namespace llvm

#endif
