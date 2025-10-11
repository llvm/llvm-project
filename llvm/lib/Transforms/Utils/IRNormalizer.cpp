//===--------------- IRNormalizer.cpp - IR Normalizer ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the IRNormalizer class which aims to transform LLVM
/// Modules into a normal form by reordering and renaming instructions while
/// preserving the same semantics. The normalizer makes it easier to spot
/// semantic differences while diffing two modules which have undergone
/// different passes.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/IRNormalizer.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"
#include <stack>

#define DEBUG_TYPE "normalize"

using namespace llvm;

namespace {
/// IRNormalizer aims to transform LLVM IR into normal form.
class IRNormalizer {
public:
  bool runOnFunction(Function &F);

  IRNormalizer(IRNormalizerOptions Options) : Options(Options) {}

private:
  const IRNormalizerOptions Options;

  // Random constant for hashing, so the state isn't zero.
  const uint64_t MagicHashConstant = 0x6acaa36bef8325c5ULL;
  DenseSet<const Instruction *> NamedInstructions;

  SmallVector<Instruction *, 16> Outputs;

  /// \name Naming.
  /// @{
  void nameFunctionArguments(Function &F) const;
  void nameBasicBlocks(Function &F) const;
  void nameInstruction(Instruction *I);
  void nameAsInitialInstruction(Instruction *I) const;
  void nameAsRegularInstruction(Instruction *I);
  void foldInstructionName(Instruction *I) const;
  /// @}

  /// \name Reordering.
  /// @{
  void reorderInstructions(Function &F) const;
  void reorderDefinition(Instruction *Definition,
                         std::stack<Instruction *> &TopologicalSort,
                         SmallPtrSet<const Instruction *, 32> &Visited) const;
  void reorderInstructionOperandsByNames(Instruction *I) const;
  void reorderPHIIncomingValues(PHINode *Phi) const;
  /// @}

  /// \name Utility methods.
  /// @{
  template <typename T>
  void sortCommutativeOperands(Instruction *I, T &Operands) const;
  SmallVector<Instruction *, 16> collectOutputInstructions(Function &F) const;
  bool isOutput(const Instruction *I) const;
  bool isInitialInstruction(const Instruction *I) const;
  bool hasOnlyImmediateOperands(const Instruction *I) const;
  SetVector<int>
  getOutputFootprint(Instruction *I,
                     SmallPtrSet<const Instruction *, 32> &Visited) const;
  /// @}
};
} // namespace

/// Entry method to the IRNormalizer.
///
/// \param F Function to normalize.
bool IRNormalizer::runOnFunction(Function &F) {
  nameFunctionArguments(F);
  nameBasicBlocks(F);

  Outputs = collectOutputInstructions(F);

  if (!Options.PreserveOrder)
    reorderInstructions(F);

  // TODO: Reorder basic blocks via a topological sort.

  for (auto &I : Outputs)
    nameInstruction(I);

  for (auto &I : instructions(F)) {
    if (!Options.PreserveOrder) {
      if (Options.ReorderOperands)
        reorderInstructionOperandsByNames(&I);

      if (auto *Phi = dyn_cast<PHINode>(&I))
        reorderPHIIncomingValues(Phi);
    }
    foldInstructionName(&I);
  }

  return true;
}

/// Numbers arguments.
///
/// \param F Function whose arguments will be renamed.
void IRNormalizer::nameFunctionArguments(Function &F) const {
  int ArgumentCounter = 0;
  for (auto &A : F.args()) {
    if (Options.RenameAll || A.getName().empty()) {
      A.setName("a" + Twine(ArgumentCounter));
      ArgumentCounter += 1;
    }
  }
}

/// Names basic blocks using a generated hash for each basic block in
/// a function considering the opcode and the order of output instructions.
///
/// \param F Function containing basic blocks to rename.
void IRNormalizer::nameBasicBlocks(Function &F) const {
  for (auto &B : F) {
    // Initialize to a magic constant, so the state isn't zero.
    uint64_t Hash = MagicHashConstant;

    // Hash considering output instruction opcodes.
    for (auto &I : B)
      if (isOutput(&I))
        Hash = hashing::detail::hash_16_bytes(Hash, I.getOpcode());

    if (Options.RenameAll || B.getName().empty()) {
      // Name basic block. Substring hash to make diffs more readable.
      B.setName("bb" + std::to_string(Hash).substr(0, 5));
    }
  }
}

/// Names instructions graphically (recursive) in accordance with the
/// def-use tree, starting from the initial instructions (defs), finishing at
/// the output (top-most user) instructions (depth-first).
///
/// \param I Instruction to be renamed.
void IRNormalizer::nameInstruction(Instruction *I) {
  // Ensure instructions are not renamed. This is done
  // to prevent situation where instructions are used
  // before their definition (in phi nodes)
  if (NamedInstructions.contains(I))
    return;
  NamedInstructions.insert(I);
  if (isInitialInstruction(I)) {
    nameAsInitialInstruction(I);
  } else {
    // This must be a regular instruction.
    nameAsRegularInstruction(I);
  }
}

template <typename T>
void IRNormalizer::sortCommutativeOperands(Instruction *I, T &Operands) const {
  if (!(I->isCommutative() && Operands.size() >= 2))
    return;
  auto CommutativeEnd = Operands.begin();
  std::advance(CommutativeEnd, 2);
  llvm::sort(Operands.begin(), CommutativeEnd);
}

/// Names instruction following the scheme:
/// vl00000Callee(Operands)
///
/// Where 00000 is a hash calculated considering instruction's opcode and output
/// footprint. Callee's name is only included when instruction's type is
/// CallInst. In cases where instruction is commutative, operands list is also
/// sorted.
///
/// Renames instruction only when RenameAll flag is raised or instruction is
/// unnamed.
///
/// \see getOutputFootprint()
/// \param I Instruction to be renamed.
void IRNormalizer::nameAsInitialInstruction(Instruction *I) const {
  if (I->getType()->isVoidTy())
    return;
  if (!(I->getName().empty() || Options.RenameAll))
    return;
  LLVM_DEBUG(dbgs() << "Naming initial instruction: " << *I << "\n");

  // Instruction operands for further sorting.
  SmallVector<SmallString<64>, 4> Operands;

  // Collect operands.
  for (auto &Op : I->operands()) {
    if (!isa<Function>(Op)) {
      std::string TextRepresentation;
      raw_string_ostream Stream(TextRepresentation);
      Op->printAsOperand(Stream, false);
      Operands.push_back(StringRef(Stream.str()));
    }
  }

  sortCommutativeOperands(I, Operands);

  // Initialize to a magic constant, so the state isn't zero.
  uint64_t Hash = MagicHashConstant;

  // Consider instruction's opcode in the hash.
  Hash = hashing::detail::hash_16_bytes(Hash, I->getOpcode());

  SmallPtrSet<const Instruction *, 32> Visited;
  // Get output footprint for I.
  SetVector<int> OutputFootprint = getOutputFootprint(I, Visited);

  // Consider output footprint in the hash.
  for (const int &Output : OutputFootprint)
    Hash = hashing::detail::hash_16_bytes(Hash, Output);

  // Base instruction name.
  SmallString<256> Name;
  Name.append("vl" + std::to_string(Hash).substr(0, 5));

  // In case of CallInst, consider callee in the instruction name.
  if (const auto *CI = dyn_cast<CallInst>(I)) {
    Function *F = CI->getCalledFunction();

    if (F != nullptr)
      Name.append(F->getName());
  }

  Name.append("(");
  for (size_t i = 0; i < Operands.size(); ++i) {
    Name.append(Operands[i]);

    if (i < Operands.size() - 1)
      Name.append(", ");
  }
  Name.append(")");

  I->setName(Name);
}

/// Names instruction following the scheme:
/// op00000Callee(Operands)
///
/// Where 00000 is a hash calculated considering instruction's opcode, its
/// operands' opcodes and order. Callee's name is only included when
/// instruction's type is CallInst. In cases where instruction is commutative,
/// operand list is also sorted.
///
/// Names instructions recursively in accordance with the def-use tree,
/// starting from the initial instructions (defs), finishing at
/// the output (top-most user) instructions (depth-first).
///
/// Renames instruction only when RenameAll flag is raised or instruction is
/// unnamed.
///
/// \see getOutputFootprint()
/// \param I Instruction to be renamed.
void IRNormalizer::nameAsRegularInstruction(Instruction *I) {
  LLVM_DEBUG(dbgs() << "Naming regular instruction: " << *I << "\n");

  // Instruction operands for further sorting.
  SmallVector<SmallString<128>, 4> Operands;

  // The name of a regular instruction depends
  // on the names of its operands. Hence, all
  // operands must be named first in the use-def
  // walk.

  // Collect operands.
  for (auto &Op : I->operands()) {
    if (auto *I = dyn_cast<Instruction>(Op)) {
      // Walk down the use-def chain.
      nameInstruction(I);
      Operands.push_back(I->getName());
    } else if (!isa<Function>(Op)) {
      // This must be an immediate value.
      std::string TextRepresentation;
      raw_string_ostream Stream(TextRepresentation);
      Op->printAsOperand(Stream, false);
      Operands.push_back(StringRef(Stream.str()));
    }
  }

  sortCommutativeOperands(I, Operands);

  // Initialize to a magic constant, so the state isn't zero.
  uint64_t Hash = MagicHashConstant;

  // Consider instruction opcode in the hash.
  Hash = hashing::detail::hash_16_bytes(Hash, I->getOpcode());

  // Operand opcodes for further sorting (commutative).
  SmallVector<int, 4> OperandsOpcodes;

  // Collect operand opcodes for hashing.
  for (auto &Op : I->operands())
    if (auto *I = dyn_cast<Instruction>(Op))
      OperandsOpcodes.push_back(I->getOpcode());

  sortCommutativeOperands(I, OperandsOpcodes);

  // Consider operand opcodes in the hash.
  for (const int Code : OperandsOpcodes)
    Hash = hashing::detail::hash_16_bytes(Hash, Code);

  // Base instruction name.
  SmallString<512> Name;
  Name.append("op" + std::to_string(Hash).substr(0, 5));

  // In case of CallInst, consider callee in the instruction name.
  if (const auto *CI = dyn_cast<CallInst>(I))
    if (const Function *F = CI->getCalledFunction())
      Name.append(F->getName());

  Name.append("(");
  for (size_t i = 0; i < Operands.size(); ++i) {
    Name.append(Operands[i]);

    if (i < Operands.size() - 1)
      Name.append(", ");
  }
  Name.append(")");

  if ((I->getName().empty() || Options.RenameAll) && !I->getType()->isVoidTy())
    I->setName(Name);
}

/// Shortens instruction's name. This method removes called function name from
/// the instruction name and substitutes the call chain with a corresponding
/// list of operands.
///
/// Examples:
/// op00000Callee(op00001Callee(...), vl00000Callee(1, 2), ...)  ->
/// op00000(op00001, vl00000, ...) vl00000Callee(1, 2)  ->  vl00000(1, 2)
///
/// This method omits output instructions and pre-output (instructions directly
/// used by an output instruction) instructions (by default). By default it also
/// does not affect user named instructions.
///
/// \param I Instruction whose name will be folded.
void IRNormalizer::foldInstructionName(Instruction *I) const {
  // If this flag is raised, fold all regular
  // instructions (including pre-outputs).
  if (!Options.FoldPreOutputs) {
    // Don't fold if one of the users is an output instruction.
    for (auto *U : I->users())
      if (auto *IU = dyn_cast<Instruction>(U))
        if (isOutput(IU))
          return;
  }

  // Don't fold if it is an output instruction or has no op prefix.
  if (isOutput(I) || !I->getName().starts_with("op"))
    return;

  // Instruction operands.
  SmallVector<SmallString<64>, 4> Operands;

  for (auto &Op : I->operands()) {
    if (const auto *I = dyn_cast<Instruction>(Op)) {
      bool HasNormalName =
          I->getName().starts_with("op") || I->getName().starts_with("vl");

      Operands.push_back(HasNormalName ? I->getName().substr(0, 7)
                                       : I->getName());
    }
  }

  sortCommutativeOperands(I, Operands);

  SmallString<256> Name;
  Name.append(I->getName().substr(0, 7));

  Name.append("(");
  for (size_t i = 0; i < Operands.size(); ++i) {
    Name.append(Operands[i]);

    if (i < Operands.size() - 1)
      Name.append(", ");
  }
  Name.append(")");

  I->setName(Name);
}

/// Reorders instructions by walking up the tree from each operand of an output
/// instruction and reducing the def-use distance.
/// This method assumes that output instructions were collected top-down,
/// otherwise the def-use chain may be broken.
/// This method is a wrapper for recursive reorderInstruction().
///
/// \see reorderInstruction()
void IRNormalizer::reorderInstructions(Function &F) const {
  for (auto &BB : F) {
    LLVM_DEBUG(dbgs() << "Reordering instructions in basic block: "
                      << BB.getName() << "\n");
    // Find the source nodes of the DAG of instructions in this basic block.
    // Source nodes are instructions that have side effects, are terminators, or
    // don't have a parent in the DAG of instructions.
    //
    // We must iterate from the first to the last instruction otherwise side
    // effecting instructions could be reordered.

    std::stack<Instruction *> TopologicalSort;
    SmallPtrSet<const Instruction *, 32> Visited;
    for (auto &I : BB) {
      // First process side effecting and terminating instructions.
      if (!(isOutput(&I) || I.isTerminator()))
        continue;
      LLVM_DEBUG(dbgs() << "\tReordering from source effecting instruction: ";
                 I.dump());
      reorderDefinition(&I, TopologicalSort, Visited);
    }

    for (auto &I : BB) {
      // Process the remaining instructions.
      //
      // TODO: Do more a intelligent sorting of these instructions. For example,
      // separate between dead instructinos and instructions used in another
      // block. Use properties of the CFG the order instructions that are used
      // in another block.
      if (Visited.contains(&I))
        continue;
      LLVM_DEBUG(dbgs() << "\tReordering from source instruction: "; I.dump());
      reorderDefinition(&I, TopologicalSort, Visited);
    }

    LLVM_DEBUG(dbgs() << "Inserting instructions into: " << BB.getName()
                      << "\n");
    // Reorder based on the topological sort.
    while (!TopologicalSort.empty()) {
      auto *Instruction = TopologicalSort.top();
      auto FirstNonPHIOrDbgOrAlloca = BB.getFirstNonPHIOrDbgOrAlloca();
      if (auto *Call = dyn_cast<CallInst>(&*FirstNonPHIOrDbgOrAlloca)) {
        if (Call->getIntrinsicID() ==
                Intrinsic::experimental_convergence_entry ||
            Call->getIntrinsicID() == Intrinsic::experimental_convergence_loop)
          FirstNonPHIOrDbgOrAlloca++;
      }
      Instruction->moveBefore(FirstNonPHIOrDbgOrAlloca);
      TopologicalSort.pop();
    }
  }
}

void IRNormalizer::reorderDefinition(
    Instruction *Definition, std::stack<Instruction *> &TopologicalSort,
    SmallPtrSet<const Instruction *, 32> &Visited) const {
  if (Visited.contains(Definition))
    return;
  Visited.insert(Definition);

  {
    const auto *BasicBlock = Definition->getParent();
    const auto FirstNonPHIOrDbgOrAlloca =
        BasicBlock->getFirstNonPHIOrDbgOrAlloca();
    if (FirstNonPHIOrDbgOrAlloca == BasicBlock->end())
      return; // TODO: Is this necessary?
    if (Definition->comesBefore(&*FirstNonPHIOrDbgOrAlloca))
      return; // TODO: Do some kind of ordering for these instructions.
  }

  for (auto &Operand : Definition->operands()) {
    if (auto *Op = dyn_cast<Instruction>(Operand)) {
      if (Op->getParent() != Definition->getParent())
        continue; // Only reorder instruction within the same basic block
      reorderDefinition(Op, TopologicalSort, Visited);
    }
  }

  LLVM_DEBUG(dbgs() << "\t\tNext in topological sort: "; Definition->dump());
  if (Definition->isTerminator())
    return;
  if (auto *Call = dyn_cast<CallInst>(Definition)) {
    if (Call->isMustTailCall())
      return;
    if (Call->getIntrinsicID() == Intrinsic::experimental_deoptimize)
      return;
    if (Call->getIntrinsicID() == Intrinsic::experimental_convergence_entry)
      return;
    if (Call->getIntrinsicID() == Intrinsic::experimental_convergence_loop)
      return;
  }
  if (auto *BitCast = dyn_cast<BitCastInst>(Definition)) {
    if (auto *Call = dyn_cast<CallInst>(BitCast->getOperand(0))) {
      if (Call->isMustTailCall())
        return;
    }
  }

  TopologicalSort.emplace(Definition);
}

/// Reorders instruction's operands alphabetically. This method assumes
/// that passed instruction is commutative. Changing the operand order
/// in other instructions may change the semantics.
///
/// \param I Instruction whose operands will be reordered.
void IRNormalizer::reorderInstructionOperandsByNames(Instruction *I) const {
  // This method assumes that passed I is commutative,
  // changing the order of operands in other instructions
  // may change the semantics.

  // Instruction operands for further sorting.
  SmallVector<std::pair<std::string, Value *>, 4> Operands;

  // Collect operands.
  for (auto &Op : I->operands()) {
    if (auto *V = dyn_cast<Value>(Op)) {
      if (isa<Instruction>(V)) {
        // This is an an instruction.
        Operands.push_back(std::pair<std::string, Value *>(V->getName(), V));
      } else {
        std::string TextRepresentation;
        raw_string_ostream Stream(TextRepresentation);
        Op->printAsOperand(Stream, false);
        Operands.push_back(std::pair<std::string, Value *>(Stream.str(), V));
      }
    }
  }

  // Sort operands.
  sortCommutativeOperands(I, Operands);

  // Reorder operands.
  unsigned Position = 0;
  for (auto &Op : I->operands()) {
    Op.set(Operands[Position].second);
    Position += 1;
  }
}

/// Reorders PHI node's values according to the names of corresponding basic
/// blocks.
///
/// \param Phi PHI node to normalize.
void IRNormalizer::reorderPHIIncomingValues(PHINode *Phi) const {
  // Values for further sorting.
  SmallVector<std::pair<Value *, BasicBlock *>, 2> Values;

  // Collect blocks and corresponding values.
  for (auto &BB : Phi->blocks()) {
    Value *V = Phi->getIncomingValueForBlock(BB);
    Values.push_back(std::pair<Value *, BasicBlock *>(V, BB));
  }

  // Sort values according to the name of a basic block.
  llvm::sort(Values, [](const std::pair<Value *, BasicBlock *> &LHS,
                        const std::pair<Value *, BasicBlock *> &RHS) {
    return LHS.second->getName() < RHS.second->getName();
  });

  // Swap.
  for (unsigned i = 0; i < Values.size(); ++i) {
    Phi->setIncomingBlock(i, Values[i].second);
    Phi->setIncomingValue(i, Values[i].first);
  }
}

/// Returns a vector of output instructions. An output is an instruction which
/// has side-effects or is ReturnInst. Uses isOutput().
///
/// \see isOutput()
/// \param F Function to collect outputs from.
SmallVector<Instruction *, 16>
IRNormalizer::collectOutputInstructions(Function &F) const {
  // Output instructions are collected top-down in each function,
  // any change may break the def-use chain in reordering methods.
  SmallVector<Instruction *, 16> Outputs;
  for (auto &I : instructions(F))
    if (isOutput(&I))
      Outputs.push_back(&I);
  return Outputs;
}

/// Helper method checking whether the instruction may have side effects or is
/// ReturnInst.
///
/// \param I Considered instruction.
bool IRNormalizer::isOutput(const Instruction *I) const {
  // Outputs are such instructions which may have side effects or is ReturnInst.
  return I->mayHaveSideEffects() || isa<ReturnInst>(I);
}

/// Helper method checking whether the instruction has users and only
/// immediate operands.
///
/// \param I Considered instruction.
bool IRNormalizer::isInitialInstruction(const Instruction *I) const {
  // Initial instructions are such instructions whose values are used by
  // other instructions, yet they only depend on immediate values.
  return !I->user_empty() && hasOnlyImmediateOperands(I);
}

/// Helper method checking whether the instruction has only immediate operands.
///
/// \param I Considered instruction.
bool IRNormalizer::hasOnlyImmediateOperands(const Instruction *I) const {
  for (const auto &Op : I->operands())
    if (isa<Instruction>(Op))
      return false; // Found non-immediate operand (instruction).
  return true;
}

/// Helper method returning indices (distance from the beginning of the basic
/// block) of outputs using the \p I (eliminates repetitions). Walks down the
/// def-use tree recursively.
///
/// \param I Considered instruction.
/// \param Visited Set of visited instructions.
SetVector<int> IRNormalizer::getOutputFootprint(
    Instruction *I, SmallPtrSet<const Instruction *, 32> &Visited) const {

  // Vector containing indexes of outputs (no repetitions),
  // which use I in the order of walking down the def-use tree.
  SetVector<int> Outputs;

  if (!Visited.count(I)) {
    Visited.insert(I);

    if (isOutput(I)) {
      // Gets output instruction's parent function.
      Function *Func = I->getParent()->getParent();

      // Finds and inserts the index of the output to the vector.
      unsigned Count = 0;
      for (const auto &B : *Func) {
        for (const auto &E : B) {
          if (&E == I)
            Outputs.insert(Count);
          Count += 1;
        }
      }

      // Returns to the used instruction.
      return Outputs;
    }

    for (auto *U : I->users()) {
      if (auto *UI = dyn_cast<Instruction>(U)) {
        // Vector for outputs which use UI.
        SetVector<int> OutputsUsingUI = getOutputFootprint(UI, Visited);
        // Insert the indexes of outputs using UI.
        Outputs.insert_range(OutputsUsingUI);
      }
    }
  }

  // Return to the used instruction.
  return Outputs;
}

PreservedAnalyses IRNormalizerPass::run(Function &F,
                                        FunctionAnalysisManager &AM) const {
  IRNormalizer(Options).runOnFunction(F);
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
