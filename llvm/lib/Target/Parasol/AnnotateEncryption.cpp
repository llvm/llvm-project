//===----------------------------------------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#include "AnnotateEncryption.h"
#include "Parasol.h"
#include "llvm/Analysis/EncryptionColoring.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

using ColorMap = MapVector<const Value *, EncryptionColor>;

static EncryptionColor mapPtrToRegColoring(EncryptionColor c) {
  switch (c) {
  case EncryptionColor::EncryptedPtrEncryptedIndex:
  case EncryptionColor::EncryptedPtrEncryptedPlainIndex:
    return EncryptionColor::Encrypted;
  case EncryptionColor::PlainPtr:
    return EncryptionColor::Plaintext;
  default:
    llvm_unreachable("Unexpected color");
  }
}

void colorInstruction(const Instruction &Inst, ColorMap &Colors) {
  if (Inst.isTerminator()) {
    return;
  }

  switch (Inst.getOpcode()) {
  case Instruction::Load: {
    auto ptrIdent = Inst.getOperand(0);
    auto ptrColor = Colors.find(ptrIdent)->second;
    auto regColor = mapPtrToRegColoring(ptrColor);

    Colors.insert(std::pair(&Inst, regColor));
    break;
  }
  case Instruction::Alloca: {
    Colors.insert(std::pair(&Inst, EncryptionColor::Plaintext));
    break;
  }
  case Instruction::Ret: {
    break;
  }
  case Instruction::Store: {
    break;
  }
  default: {
    auto kind = EncryptionColor::Plaintext;

    for (auto &op : Inst.operands()) {
      if (Colors.find(op)->second == EncryptionColor::Encrypted) {
        kind = EncryptionColor::Encrypted;
      }
    }

    Colors.insert(std::pair(&Inst, kind));
  }
  }
}

PreservedAnalyses AnnotateEncryptionPass::run(Function &F,
                                              FunctionAnalysisManager &FM) {
  if (!F.hasFnAttribute(Attribute::FheCircuit)) {
    return PreservedAnalyses::all();
  }

  ColorMap Colors;

  // First, capture the color for the function arguments.
  for (const auto &A : F.args()) {
    if (!A.getType()->isPointerTy()) {
      report_fatal_error("An fhe_circuit function had a non-pointer argument");
    }

    if (A.hasEncryptedAttr()) {
      Colors.insert(std::pair(&A, EncryptedPtrEncryptedPlainIndex));
    } else {
      Colors.insert(std::pair(&A, PlainPtr));
    }
  }

  // Next, color each identifier for each statement.
  for (const auto &inst : instructions(F)) {
    colorInstruction(inst, Colors);
  }

  for (auto &inst : instructions(F)) {
    auto Color = Colors.find(&inst);

    if (Color != Colors.end()) {
      MDBuilder MDHelper(F.getParent()->getContext());
      MDNode *EncNode = MDHelper.createEncryption(Color->second);

      inst.setMetadata(LLVMContext::MD_encryption, EncNode);
    }
  }

  return PreservedAnalyses::none();
}

namespace {
class AnnotateEncryptionLegacy : public FunctionPass {
private:
  AnnotateEncryptionPass Impl;

public:
  static char ID;

  AnnotateEncryptionLegacy() : FunctionPass(ID), Impl() {
    initializeAnnotateEncryptionLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    // Don't skip optnone functions; atomics still need to be lowered.
    FunctionAnalysisManager DummyFAM;
    auto PA = Impl.run(F, DummyFAM);
    return !PA.areAllPreserved();
  }
};
} // namespace

char AnnotateEncryptionLegacy::ID = 0;
INITIALIZE_PASS(
    AnnotateEncryptionLegacy, "annotate-encryption",
    "Annotate each llvm::Instruction as having an encrypted result or not",
    false, false)

FunctionPass *llvm::createAnnotateEncryptionPass() {
  return new AnnotateEncryptionLegacy();
}
