//=== GuardedLoadHardening.cpp -Lightweight spectre v1 mitigation *- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a form of load hardening as a mitigation against Spectre v1.
// Unlike the other [LLVM mitigations](/llvm/docs/SpeculativeLoadHardening.md)
// this mitigation more like MSVC's /Qspectre flag where it provides less
// comprehensive coverage but is also cheap enough that it can be widely
// applied.
//
// Specifically this mitigation is trying to identify the pattern outlined in
// <https://devblogs.microsoft.com/cppblog/spectre-mitigations-in-msvc>
// that is, an offsetted load that is used to offset another load, both of which
// are guarded by a bounds check. For example:
// ```cpp
// if (untrusted_index < array1_length) {
//     unsigned char value = array1[untrusted_index];
//     unsigned char value2 = array2[value * 64];
// }
// ```
//
// The other case that this mitigation looks for is an indirect call from an
// offsetted load that is protected by a bounds check. For example:
// ```cpp
// if (index < funcs_len) {
//   return funcs[index * 4]();
// }
// ```
//
// This mitigation will insert the `speculative_data_barrier` intrinsic into the
// block with the second load or the indirect call.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/GuardedLoadHardening.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "guarded-load-hardening"

static cl::opt<bool>
    EnableGuardedLoadHardening("guarded-load-hardening",
                               cl::desc("Enable guarded load hardening"),
                               cl::init(false), cl::Hidden);

STATISTIC(NumIntrInserted, "Intrinsics inserted");
STATISTIC(CandidateBlocks, "Candidate blocks discovered");
STATISTIC(OffsettedLoads, "Offsetted loads discovered");
STATISTIC(DownstreamInstr, "Downstream loads or calls discovered");
STATISTIC(OffsettedLoadsRemoved, "Candidate offsetted loads removed");

namespace {

class GuardedLoadHardening : public FunctionPass {
public:
  static char ID;

  // Default constructor required for the INITIALIZE_PASS macro.
  GuardedLoadHardening() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;
};

} // end anonymous namespace

/// Visits the given value and all of its operands recursively, if they are of a
/// type that is interesting to this analysis.
bool visitDependencies(const Value &Start,
                       const std::function<bool(const Value &)> &Visitor) {
  SmallVector<const Value *, 4> Worklist{&Start};
  while (!Worklist.empty()) {
    auto *Item = Worklist.pop_back_val();
    if (isa<Argument>(Item)) {
      if (Visitor(*Item)) {
        return true;
      }
    } else if (auto *Inst = dyn_cast<Instruction>(Item)) {
      // Only visit the operands of unary, binary, and cast instructions. There
      // are many other instructions that could be unwrapped here (e.g., Phi
      // nodes, SelectInst), but they make the analysis too expensive.
      if (Inst->isUnaryOp() || Inst->isBinaryOp() || Inst->isCast()) {
        Worklist.append(Inst->value_op_begin(), Inst->value_op_end());
      } else if (isa<CallInst>(Inst) || isa<LoadInst>(Inst) ||
                 isa<AllocaInst>(Inst)) {
        if (Visitor(*Item)) {
          return true;
        }
      }
    }
  }

  return false;
}

/// Gathers the given value and all of its operands recursively, if they are of
/// a type that is interesting to this analysis.
void gatherDependencies(const Value &Start,
                        std::vector<const Value *> &Dependencies) {
  visitDependencies(Start, [&](const Value &V) {
    Dependencies.push_back(&V);
    return false;
  });
}

/// Checks if the given instruction is an offsetted load and returns the indices
/// used to offset that load.
std::optional<iterator_range<User::const_op_iterator>>
tryGetIndicesIfOffsettedLoad(const Value &I) {
  if (auto *Load = dyn_cast<LoadInst>(&I)) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Load->getPointerOperand())) {
      if (GEP->hasIndices() && !GEP->hasAllConstantIndices()) {
        return GEP->indices();
      }
    }
  }
  return std::nullopt;
}

/// Tries to get the comparison instruction if the given block is guarded by a
/// relative integer comparison.
std::optional<const ICmpInst *>
tryGetComparisonIfGuarded(const BasicBlock &BB) {
  if (auto *PredBB = BB.getSinglePredecessor()) {
    if (auto *CondBranch = dyn_cast<BranchInst>(PredBB->getTerminator())) {
      if (CondBranch->isConditional()) {
        if (auto *Comparison = dyn_cast<ICmpInst>(CondBranch->getCondition())) {
          if (Comparison->isRelational()) {
            return Comparison;
          }
        }
      }
    }
  }

  return std::nullopt;
}

/// Does the given value use an offsetted load that requires protection?
bool useRequiresProtection(const Value &MightUseIndex,
                           const ICmpInst &Comparison,
                           SmallVector<std::pair<const Value *, const Value *>,
                                       4> &OffsettedLoadAndUses) {

  SmallVector<const Value *, 4> OffsettedLoadIndexesToRemove;
  for (auto &LoadAndUse : OffsettedLoadAndUses) {
    if ((&MightUseIndex == LoadAndUse.second) &&
        !is_contained(OffsettedLoadIndexesToRemove, LoadAndUse.first)) {
      ++DownstreamInstr;

      // If we've found a use of one of the offsetted loads, then we need to
      // check if that offsetted load uses a value that is also used in the
      // comparison.
      std::vector<const Value *> ComparisonDependencies;
      gatherDependencies(*Comparison.getOperand(0), ComparisonDependencies);
      gatherDependencies(*Comparison.getOperand(1), ComparisonDependencies);

      for (auto &Index : *tryGetIndicesIfOffsettedLoad(*LoadAndUse.first)) {
        if (!isa<Constant>(&Index) &&
            visitDependencies(*Index, [&](const Value &V) {
              return is_contained(ComparisonDependencies, &V);
            })) {
          return true;
        }
      }

      // The offsetted load doesn't use any of the values in the comparison, so
      // remove it from the list since we never need to check it again.
      OffsettedLoadIndexesToRemove.push_back(LoadAndUse.first);
      ++OffsettedLoadsRemoved;
    }
  }

  for (auto *IndexToRemove : OffsettedLoadIndexesToRemove) {
    OffsettedLoadAndUses.erase(
        std::remove_if(
            OffsettedLoadAndUses.begin(), OffsettedLoadAndUses.end(),
            [&](const auto &Pair) { return Pair.first == IndexToRemove; }),
        OffsettedLoadAndUses.end());
  }
  return false;
}

bool runOnFunctionImpl(Function &F) {
  SmallVector<BasicBlock *, 4> BlocksToProtect;
  for (auto &BB : F) {
    // Check for guarded loads that need to be protected.
    if (auto Comparison = tryGetComparisonIfGuarded(BB)) {
      ++CandidateBlocks;
      SmallVector<std::pair<const Value *, const Value *>, 4>
          OffsettedLoadAndUses;
      for (auto &I : BB) {
        if (OffsettedLoadAndUses.empty()) {
          if (tryGetIndicesIfOffsettedLoad(I)) {
            OffsettedLoadAndUses.emplace_back(&I, &I);
            ++OffsettedLoads;
          }
        } else {
          // Case 1: Look for an indirect call where the target is an offsetted
          // load.
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            if (Call->isIndirectCall() &&
                useRequiresProtection(*Call->getCalledOperand(), **Comparison,
                                      OffsettedLoadAndUses)) {
              BlocksToProtect.push_back(&BB);
              break;
            }

            // Case 2: Look for an offsetted load that is used as an index.
          } else if (auto DependentIndexOp = tryGetIndicesIfOffsettedLoad(I)) {
            for (auto &Op : *DependentIndexOp) {
              if (!isa<Constant>(&Op) &&
                  useRequiresProtection(*Op, **Comparison,
                                        OffsettedLoadAndUses)) {
                BlocksToProtect.push_back(&BB);
                break;
              }
            }

            OffsettedLoadAndUses.emplace_back(&I, &I);
            ++OffsettedLoads;

            // Otherwise, check if this value uses something from an offsetted
            // load or one of its downstreams.
          } else if (auto *Instr = dyn_cast<Instruction>(&I)) {
            if (Instr->isUnaryOp() || Instr->isBinaryOp() || Instr->isCast()) {
              for (auto &Op : Instr->operands()) {
                // If any use of an offsetted load is used by this instruction,
                // then add this instruction as a use of that offsetted load as
                // well.
                for (auto &LoadAndUse : OffsettedLoadAndUses) {
                  if (Op.get() == LoadAndUse.second) {
                    OffsettedLoadAndUses.emplace_back(LoadAndUse.first, Instr);
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if (BlocksToProtect.empty()) {
    return false;
  }

  // Add a barrier to each block that requires protection.
  for (auto *BB : BlocksToProtect) {
    IRBuilder<> Builder(&BB->front());
    Builder.CreateIntrinsic(Intrinsic::speculative_data_barrier, {}, {});
    ++NumIntrInserted;
  }

  return true;
}

char GuardedLoadHardening::ID = 0;
INITIALIZE_PASS(GuardedLoadHardening, "GuardedLoadHardening",
                "GuardedLoadHardening", false, false)

bool GuardedLoadHardening::runOnFunction(Function &F) {
  if (EnableGuardedLoadHardening) {
    return runOnFunctionImpl(F);
  }
  return false;
}

PreservedAnalyses GuardedLoadHardeningPass::run(Function &F,
                                                FunctionAnalysisManager &FAM) {
  bool Changed = false;
  if (EnableGuardedLoadHardening) {
    Changed = runOnFunctionImpl(F);
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

FunctionPass *llvm::createGuardedLoadHardeningPass() {
  return new GuardedLoadHardening();
}