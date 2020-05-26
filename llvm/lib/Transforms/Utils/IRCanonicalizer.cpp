//===--------------- IRCanonicalizer.cpp - IR Canonicalizer ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the IRCanonicalizer class which aims to transform LLVM
/// Modules into a canonical form by reordering and renaming instructions while
/// preserving the same semantics. The canonicalizer makes it easier to spot
/// semantic differences while diffing two modules which have undergone
/// different passes.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "ir-canonicalizer"

using namespace llvm;

namespace {
/// IRCanonicalizer aims to transform LLVM IR into canonical form.
class IRCanonicalizer : public FunctionPass {
public:
  static char ID;

  /// \name Canonicalizer flags.
  /// @{
  /// Preserves original order of instructions.
  static cl::opt<bool> PreserveOrder;
  /// Renames all instructions (including user-named).
  static cl::opt<bool> RenameAll;
  /// Folds all regular instructions (including pre-outputs).
  static cl::opt<bool> FoldPreoutputs;
  /// Sorts and reorders operands in commutative instructions.
  static cl::opt<bool> ReorderOperands;
  /// @}

  /// Constructor for the IRCanonicalizer.
  IRCanonicalizer() : FunctionPass(ID) {
    initializeIRCanonicalizerPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

private:
  // Random constant for hashing, so the state isn't zero.
  const uint64_t MagicHashConstant = 0x6acaa36bef8325c5ULL;

  /// \name Naming.
  /// @{
  void nameFunctionArguments(Function &F);
  void nameBasicBlocks(Function &F);
  void nameInstruction(Instruction *I);
  void nameAsInitialInstruction(Instruction *I);
  void nameAsRegularInstruction(Instruction *I);
  void foldInstructionName(Instruction *I);
  /// @}

  /// \name Reordering.
  /// @{
  void reorderInstructions(SmallVector<Instruction *, 16> &Outputs);
  void reorderInstruction(Instruction *Used, Instruction *User,
                          SmallPtrSet<const Instruction *, 32> &Visited);
  void reorderInstructionOperandsByNames(Instruction *I);
  void reorderPHIIncomingValues(PHINode *PN);
  /// @}

  /// \name Utility methods.
  /// @{
  SmallVector<Instruction *, 16> collectOutputInstructions(Function &F);
  bool isOutput(const Instruction *I);
  bool isInitialInstruction(const Instruction *I);
  bool hasOnlyImmediateOperands(const Instruction *I);
  SetVector<int>
  getOutputFootprint(Instruction *I,
                     SmallPtrSet<const Instruction *, 32> &Visited);
  /// @}
};
} // namespace

char IRCanonicalizer::ID = 0;

cl::opt<bool> IRCanonicalizer::PreserveOrder(
    "preserve-order", cl::Hidden,
    cl::desc("Preserves original instruction order"));
cl::opt<bool> IRCanonicalizer::RenameAll(
    "rename-all", cl::Hidden,
    cl::desc("Renames all instructions (including user-named)"));
cl::opt<bool> IRCanonicalizer::FoldPreoutputs(
    "fold-all", cl::Hidden,
    cl::desc("Folds all regular instructions (including pre-outputs)"));
cl::opt<bool> IRCanonicalizer::ReorderOperands(
    "reorder-operands", cl::Hidden,
    cl::desc("Sorts and reorders operands in commutative instructions"));

INITIALIZE_PASS(IRCanonicalizer, "ir-canonicalizer",
                "Transforms IR into canonical form", false, false)

Pass *llvm::createIRCanonicalizerPass() { return new IRCanonicalizer(); }

/// Entry method to the IRCanonicalizer.
///
/// \param M Module to canonicalize.
bool IRCanonicalizer::runOnFunction(Function &F) {
  nameFunctionArguments(F);
  nameBasicBlocks(F);

  SmallVector<Instruction *, 16> Outputs = collectOutputInstructions(F);

  if (!PreserveOrder)
    reorderInstructions(Outputs);

  for (auto &I : Outputs)
    nameInstruction(I);

  for (auto &I : instructions(F)) {
    if (!PreserveOrder) {
      if (ReorderOperands && I.isCommutative())
        reorderInstructionOperandsByNames(&I);

      if (auto *PN = dyn_cast<PHINode>(&I))
        reorderPHIIncomingValues(PN);
    }

    foldInstructionName(&I);
  }

  return true;
}

/// Numbers arguments.
///
/// \param F Function whose arguments will be renamed.
void IRCanonicalizer::nameFunctionArguments(Function &F) {
  int ArgumentCounter = 0;
  for (auto &A : F.args()) {
    if (RenameAll || A.getName().empty()) {
      A.setName("a" + Twine(ArgumentCounter));
      ++ArgumentCounter;
    }
  }
}

/// Names basic blocks using a generated hash for each basic block in
/// a function considering the opcode and the order of output instructions.
///
/// \param F Function containing basic blocks to rename.
void IRCanonicalizer::nameBasicBlocks(Function &F) {
  for (auto &B : F) {
    // Initialize to a magic constant, so the state isn't zero.
    uint64_t Hash = MagicHashConstant;

    // Hash considering output instruction opcodes.
    for (auto &I : B)
      if (isOutput(&I))
        Hash = hashing::detail::hash_16_bytes(Hash, I.getOpcode());

    if (RenameAll || B.getName().empty()) {
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
void IRCanonicalizer::nameInstruction(Instruction *I) {
  // Determine the type of instruction to name.
  if (isInitialInstruction(I)) {
    // This is an initial instruction.
    nameAsInitialInstruction(I);
  } else {
    // This must be a regular instruction.
    nameAsRegularInstruction(I);
  }
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
void IRCanonicalizer::nameAsInitialInstruction(Instruction *I) {
  if (I->getType()->isVoidTy() || (!I->getName().empty() && !RenameAll))
    return;

  // Instruction operands for further sorting.
  SmallVector<SmallString<64>, 4> Operands;

  // Collect operands.
  for (auto &OP : I->operands()) {
    if (!isa<Function>(OP)) {
      std::string TextRepresentation;
      raw_string_ostream Stream(TextRepresentation);
      OP->printAsOperand(Stream, false);
      Operands.push_back(StringRef(Stream.str()));
    }
  }

  if (I->isCommutative())
    llvm::sort(Operands);

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

    if (F != nullptr) {
      Name.append(F->getName());
    }
  }

  Name.append("(");
  for (unsigned long i = 0; i < Operands.size(); ++i) {
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
void IRCanonicalizer::nameAsRegularInstruction(Instruction *I) {
  // Instruction operands for further sorting.
  SmallVector<SmallString<128>, 4> Operands;

  // The name of a regular instruction depends
  // on the names of its operands. Hence, all
  // operands must be named first in the use-def
  // walk.

  // Collect operands.
  for (auto &OP : I->operands()) {
    if (auto *IOP = dyn_cast<Instruction>(OP)) {
      // Walk down the use-def chain.
      nameInstruction(IOP);
      Operands.push_back(IOP->getName());
    } else if (isa<Value>(OP) && !isa<Function>(OP)) {
      // This must be an immediate value.
      std::string TextRepresentation;
      raw_string_ostream Stream(TextRepresentation);
      OP->printAsOperand(Stream, false);
      Operands.push_back(StringRef(Stream.str()));
    }
  }

  if (I->isCommutative())
    llvm::sort(Operands.begin(), Operands.end());

  // Initialize to a magic constant, so the state isn't zero.
  uint64_t Hash = MagicHashConstant;

  // Consider instruction opcode in the hash.
  Hash = hashing::detail::hash_16_bytes(Hash, I->getOpcode());

  // Operand opcodes for further sorting (commutative).
  SmallVector<int, 4> OperandsOpcodes;

  // Collect operand opcodes for hashing.
  for (auto &OP : I->operands())
    if (auto *IOP = dyn_cast<Instruction>(OP))
      OperandsOpcodes.push_back(IOP->getOpcode());

  if (I->isCommutative())
    llvm::sort(OperandsOpcodes.begin(), OperandsOpcodes.end());

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
  for (unsigned long i = 0; i < Operands.size(); ++i) {
    Name.append(Operands[i]);

    if (i < Operands.size() - 1)
      Name.append(", ");
  }
  Name.append(")");

  if ((I->getName().empty() || RenameAll) && !I->getType()->isVoidTy())
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
void IRCanonicalizer::foldInstructionName(Instruction *I) {
  // If this flag is raised, fold all regular
  // instructions (including pre-outputs).
  if (!FoldPreoutputs) {
    // Don't fold if one of the users is an output instruction.
    for (auto *U : I->users())
      if (auto *IU = dyn_cast<Instruction>(U))
        if (isOutput(IU))
          return;
  }

  // Don't fold if it is an output instruction or has no op prefix.
  if (isOutput(I) || I->getName().substr(0, 2) != "op")
    return;

  // Instruction operands.
  SmallVector<SmallString<64>, 4> Operands;

  for (auto &OP : I->operands()) {
    if (const Instruction *IOP = dyn_cast<Instruction>(OP)) {
      bool HasCanonicalName = I->getName().substr(0, 2) == "op" ||
                              I->getName().substr(0, 2) == "vl";

      Operands.push_back(HasCanonicalName ? IOP->getName().substr(0, 7)
                                          : IOP->getName());
    }
  }

  if (I->isCommutative())
    llvm::sort(Operands.begin(), Operands.end());

  SmallString<256> Name;
  Name.append(I->getName().substr(0, 7));

  Name.append("(");
  for (unsigned long i = 0; i < Operands.size(); ++i) {
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
/// \param Outputs Vector of pointers to output instructions collected top-down.
void IRCanonicalizer::reorderInstructions(
    SmallVector<Instruction *, 16> &Outputs) {
  // This method assumes output instructions were collected top-down,
  // otherwise the def-use chain may be broken.

  SmallPtrSet<const Instruction *, 32> Visited;

  // Walk up the tree.
  for (auto &I : Outputs)
    for (auto &OP : I->operands())
      if (auto *IOP = dyn_cast<Instruction>(OP))
        reorderInstruction(IOP, I, Visited);
}

/// Reduces def-use distance or places instruction at the end of the basic
/// block. Continues to walk up the def-use tree recursively. Used by
/// reorderInstructions().
///
/// \see reorderInstructions()
/// \param Used Pointer to the instruction whose value is used by the \p User.
/// \param User Pointer to the instruction which uses the \p Used.
/// \param Visited Set of visited instructions.
void IRCanonicalizer::reorderInstruction(
    Instruction *Used, Instruction *User,
    SmallPtrSet<const Instruction *, 32> &Visited) {

  if (!Visited.count(Used)) {
    Visited.insert(Used);

    if (Used->getParent() == User->getParent()) {
      // If Used and User share the same basic block move Used just before User.
      Used->moveBefore(User);
    } else {
      // Otherwise move Used to the very end of its basic block.
      Used->moveBefore(&Used->getParent()->back());
    }

    for (auto &OP : Used->operands()) {
      if (auto *IOP = dyn_cast<Instruction>(OP)) {
        // Walk up the def-use tree.
        reorderInstruction(IOP, Used, Visited);
      }
    }
  }
}

/// Reorders instruction's operands alphabetically. This method assumes
/// that passed instruction is commutative. Changing the operand order
/// in other instructions may change the semantics.
///
/// \param I Instruction whose operands will be reordered.
void IRCanonicalizer::reorderInstructionOperandsByNames(Instruction *I) {
  // This method assumes that passed I is commutative,
  // changing the order of operands in other instructions
  // may change the semantics.

  // Instruction operands for further sorting.
  SmallVector<std::pair<std::string, Value *>, 4> Operands;

  // Collect operands.
  for (auto &OP : I->operands()) {
    if (auto *VOP = dyn_cast<Value>(OP)) {
      if (isa<Instruction>(VOP)) {
        // This is an an instruction.
        Operands.push_back(
            std::pair<std::string, Value *>(VOP->getName(), VOP));
      } else {
        std::string TextRepresentation;
        raw_string_ostream Stream(TextRepresentation);
        OP->printAsOperand(Stream, false);
        Operands.push_back(std::pair<std::string, Value *>(Stream.str(), VOP));
      }
    }
  }

  // Sort operands.
  llvm::sort(Operands.begin(), Operands.end(), llvm::less_first());

  // Reorder operands.
  unsigned Position = 0;
  for (auto &OP : I->operands()) {
    OP.set(Operands[Position].second);
    Position++;
  }
}

/// Reorders PHI node's values according to the names of corresponding basic
/// blocks.
///
/// \param PN PHI node to canonicalize.
void IRCanonicalizer::reorderPHIIncomingValues(PHINode *PN) {
  // Values for further sorting.
  SmallVector<std::pair<Value *, BasicBlock *>, 2> Values;

  // Collect blocks and corresponding values.
  for (auto &BB : PN->blocks()) {
    Value *V = PN->getIncomingValueForBlock(BB);
    Values.push_back(std::pair<Value *, BasicBlock *>(V, BB));
  }

  // Sort values according to the name of a basic block.
  llvm::sort(Values, [](const std::pair<Value *, BasicBlock *> &LHS,
                        const std::pair<Value *, BasicBlock *> &RHS) {
    return LHS.second->getName() < RHS.second->getName();
  });

  // Swap.
  for (unsigned i = 0; i < Values.size(); ++i) {
    PN->setIncomingBlock(i, Values[i].second);
    PN->setIncomingValue(i, Values[i].first);
  }
}

/// Returns a vector of output instructions. An output is an instruction which
/// has side-effects or is ReturnInst. Uses isOutput().
///
/// \see isOutput()
/// \param F Function to collect outputs from.
SmallVector<Instruction *, 16>
IRCanonicalizer::collectOutputInstructions(Function &F) {
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
bool IRCanonicalizer::isOutput(const Instruction *I) {
  // Outputs are such instructions which may have side effects or is ReturnInst.
  if (I->mayHaveSideEffects() || isa<ReturnInst>(I))
    return true;

  return false;
}

/// Helper method checking whether the instruction has users and only
/// immediate operands.
///
/// \param I Considered instruction.
bool IRCanonicalizer::isInitialInstruction(const Instruction *I) {
  // Initial instructions are such instructions whose values are used by
  // other instructions, yet they only depend on immediate values.
  return !I->user_empty() && hasOnlyImmediateOperands(I);
}

/// Helper method checking whether the instruction has only immediate operands.
///
/// \param I Considered instruction.
bool IRCanonicalizer::hasOnlyImmediateOperands(const Instruction *I) {
  for (const auto &OP : I->operands())
    if (isa<Instruction>(OP))
      return false; // Found non-immediate operand (instruction).

  return true;
}

/// Helper method returning indices (distance from the beginning of the basic
/// block) of outputs using the \p I (eliminates repetitions). Walks down the
/// def-use tree recursively.
///
/// \param I Considered instruction.
/// \param Visited Set of visited instructions.
SetVector<int> IRCanonicalizer::getOutputFootprint(
    Instruction *I, SmallPtrSet<const Instruction *, 32> &Visited) {

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
          Count++;
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
        Outputs.insert(OutputsUsingUI.begin(), OutputsUsingUI.end());
      }
    }
  }

  // Return to the used instruction.
  return Outputs;
}