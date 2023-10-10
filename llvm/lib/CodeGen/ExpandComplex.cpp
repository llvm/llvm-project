//===-- ExpandComplex.cpp - Expand experimental complex intrinsics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR expansion for complex intrinsics, allowing targets
// to enable the intrinsics until just before codegen.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandComplex.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

bool expandComplexInstruction(IntrinsicInst *CI, const TargetLowering *TLI,
                              const DataLayout &DL) {
  Intrinsic::ID Opcode = CI->getIntrinsicID();
  assert((Opcode == Intrinsic::experimental_complex_fmul ||
          Opcode == Intrinsic::experimental_complex_fdiv) &&
         "Expected a complex instruction");

  // Break the input values up into real and imaginary pieces.
  Type *ComplexVectorTy = CI->getArgOperand(0)->getType();
  Type *FloatTy = ComplexVectorTy->getScalarType();
  IRBuilder<> Builder(CI);
  Builder.setFastMathFlags(CI->getFastMathFlags());
  Value *LhsR = Builder.CreateExtractElement(CI->getArgOperand(0), uint64_t(0));
  Value *LhsI = Builder.CreateExtractElement(CI->getArgOperand(0), uint64_t(1));
  Value *RhsR = nullptr, *RhsI = nullptr;
  RhsR = Builder.CreateExtractElement(CI->getArgOperand(1), uint64_t(0));
  RhsI = Builder.CreateExtractElement(CI->getArgOperand(1), uint64_t(1));

  // The expansion has three pieces: the naive arithmetic, a possible prescaling
  // (not relevant for multiplication), and a step to convert NaN output values
  // to infinity values in certain situations (see Annex G of the C
  // specification for more details). The "complex-range" attribute determines
  // how many we need: "limited" has just the first one, "no-nan" the first two,
  // and "full" for all three.

  // Get the "complex-range" attribute, setting a default based on the presence
  // of fast-math flags.
  StringRef Range = CI->getFnAttr("complex-range").getValueAsString();
  if (Range.empty()) {
    Range = CI->getFastMathFlags().noNaNs() || CI->getFastMathFlags().noInfs()
                ? "no-nan"
                : "full";
  }

  // We can expand to naive arithmetic code if we only need the first piece. For
  // multiplication, we can also accept "no-nan", since there is no semantic
  // difference between "limited" and "no-nan" in that case.
  bool CanExpand =
      Range == "limited" ||
      (Range == "no-nan" && Opcode == Intrinsic::experimental_complex_fmul);

  Value *OutReal, *OutImag;
  if (!CanExpand) {
    // Do a call directly to the compiler-rt library here.
    const char *Name = nullptr;
    if (Opcode == Intrinsic::experimental_complex_fmul) {
      if (FloatTy->isHalfTy())
        Name = "__mulhc3";
      else if (FloatTy->isFloatTy())
        Name = "__mulsc3";
      else if (FloatTy->isDoubleTy())
        Name = "__muldc3";
      else if (FloatTy->isX86_FP80Ty())
        Name = "__mulxc3";
      else if (FloatTy->isFP128Ty() || FloatTy->isPPC_FP128Ty())
        Name = "__multc3";
    } else if (Opcode == Intrinsic::experimental_complex_fdiv) {
      if (FloatTy->isHalfTy())
        Name = "__divhc3";
      else if (FloatTy->isFloatTy())
        Name = "__divsc3";
      else if (FloatTy->isDoubleTy())
        Name = "__divdc3";
      else if (FloatTy->isX86_FP80Ty())
        Name = "__divxc3";
      else if (FloatTy->isFP128Ty() || FloatTy->isPPC_FP128Ty())
        Name = "__divtc3";
    }

    if (!Name)
      report_fatal_error("Cannot find libcall for intrinsic");

    // The function we are to call is T complex __name(T, T, T, T) in C terms.
    // Use TLI to figure out what the appropriate actual ABI for this function.
    StructType *ComplexStructTy = StructType::get(FloatTy, FloatTy);
    switch (TLI->getComplexReturnABI(FloatTy)) {
    case TargetLowering::ComplexABI::Vector: {
      // When the result is a vector type directly, we can replace the intrinsic
      // with the call to the underlying function without any other munging.
      FunctionCallee Func = CI->getModule()->getOrInsertFunction(
          Name, ComplexVectorTy, FloatTy, FloatTy, FloatTy, FloatTy);
      Value *NewResult = Builder.CreateCall(Func, {LhsR, LhsI, RhsR, RhsI});
      CI->replaceAllUsesWith(NewResult);
      CI->eraseFromParent();
      return true;
    }
    case TargetLowering::ComplexABI::Integer: {
      // This ABI form packs the type as a small struct in an integer register.
      // All we need to do is move the integer to a vector register, without any
      // other munging.
      uint64_t Width =
        ComplexVectorTy->getPrimitiveSizeInBits().getFixedValue();
      Type *IntegerTy = Builder.getIntNTy(Width);
      FunctionCallee Func = CI->getModule()->getOrInsertFunction(
          Name, IntegerTy, FloatTy, FloatTy, FloatTy, FloatTy);
      Value *NewResult = Builder.CreateBitCast(
          Builder.CreateCall(Func, {LhsR, LhsI, RhsR, RhsI}), ComplexVectorTy);
      CI->replaceAllUsesWith(NewResult);
      CI->eraseFromParent();
      return true;
    }
    case TargetLowering::ComplexABI::Memory: {
      // Allocate a struct for the return type in the entry block. Stack slot
      // coloring should remove duplicate allocations.
      unsigned AllocaAS = DL.getAllocaAddrSpace();
      Value *Alloca;
      {
        IRBuilderBase::InsertPointGuard Guard(Builder);
        BasicBlock *EntryBB = &CI->getParent()->getParent()->getEntryBlock();
        Builder.SetInsertPoint(EntryBB, EntryBB->begin());
        Alloca = Builder.CreateAlloca(ComplexStructTy, AllocaAS);
      }

      AttributeList Attrs;
      Attrs = Attrs.addParamAttribute(
          CI->getContext(), 0,
          Attribute::getWithStructRetType(CI->getContext(), ComplexStructTy));
      FunctionCallee Func = CI->getModule()->getOrInsertFunction(
          Name, std::move(Attrs), Type::getVoidTy(CI->getContext()),
          PointerType::get(ComplexStructTy, AllocaAS), FloatTy, FloatTy,
          FloatTy, FloatTy);

      Builder.CreateCall(Func, {Alloca, LhsR, LhsI, RhsR, RhsI});
      OutReal = Builder.CreateLoad(
          FloatTy, Builder.CreateStructGEP(ComplexStructTy, Alloca, 0));
      OutImag = Builder.CreateLoad(
          FloatTy, Builder.CreateStructGEP(ComplexStructTy, Alloca, 1));
      break;
    }
    case TargetLowering::ComplexABI::Struct: {
      FunctionCallee Func = CI->getModule()->getOrInsertFunction(
          Name, ComplexStructTy, FloatTy, FloatTy, FloatTy, FloatTy);
      Value *ComplexStructRes =
          Builder.CreateCall(Func, {LhsR, LhsI, RhsR, RhsI});
      OutReal = Builder.CreateExtractValue(ComplexStructRes, 0);
      OutImag = Builder.CreateExtractValue(ComplexStructRes, 1);
      break;
    }
    }
  } else {
    switch (Opcode) {
    case Intrinsic::experimental_complex_fmul: {
      // If the target has a complex_fmul expansion and the fast-math flag
      // set, use that instead of expanding.
      if (TLI->CustomLowerComplexMultiply(ComplexVectorTy)) {
        return false;
      }

      OutReal = Builder.CreateFSub(Builder.CreateFMul(LhsR, RhsR),
                                   Builder.CreateFMul(LhsI, RhsI));
      OutImag = Builder.CreateFAdd(Builder.CreateFMul(LhsI, RhsR),
                                   Builder.CreateFMul(LhsR, RhsI));
      break;
    }
    case Intrinsic::experimental_complex_fdiv: {
      Value *Scale = Builder.CreateFAdd(Builder.CreateFMul(RhsR, RhsR),
                                        Builder.CreateFMul(RhsI, RhsI));
      OutReal =
          Builder.CreateFDiv(Builder.CreateFAdd(Builder.CreateFMul(LhsR, RhsR),
                                                Builder.CreateFMul(LhsI, RhsI)),
                             Scale);
      OutImag =
          Builder.CreateFDiv(Builder.CreateFSub(Builder.CreateFMul(LhsI, RhsR),
                                                Builder.CreateFMul(LhsR, RhsI)),
                             Scale);
      break;
    }
    }
  }

  // Replace all of the uses of the intrinsic with OutReal/OutImag. We avoid
  // creating the vector unless we have to.
  bool HasVectorUse = false;
  for (User *U : CI->users()) {
    uint64_t Index;
    if (match(U, m_ExtractElt(m_Value(), m_ConstantInt(Index)))) {
      assert((Index == 0 || Index == 1) && "Extract element too small");
      U->replaceAllUsesWith(Index == 0 ? OutReal : OutImag);
    } else {
      HasVectorUse = true;
    }
  }

  if (HasVectorUse) {
    Value *OutComplex = Builder.CreateInsertElement(
        Builder.CreateInsertElement(UndefValue::get(ComplexVectorTy), OutReal,
                                    uint64_t(0)),
        OutImag, uint64_t(1));
    CI->replaceAllUsesWith(OutComplex);
  } else {
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
  }

  CI->eraseFromParent();
  return true;
}

bool expandComplexIntrinsics(Function &F, const TargetLowering *TLI) {
  bool Changed = false;
  SmallVector<IntrinsicInst *, 4> Worklist;
  for (auto &I : instructions(F)) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      default:
        break;
      case Intrinsic::experimental_complex_fmul:
      case Intrinsic::experimental_complex_fdiv:
        Worklist.push_back(II);
        break;
      }
    }
  }

  const DataLayout &DL = F.getParent()->getDataLayout();
  for (auto *II : Worklist) {
    Changed |= expandComplexInstruction(II, TLI, DL);
  }
  return Changed;
}

class ExpandComplex : public FunctionPass {
public:
  static char ID;
  ExpandComplex() : FunctionPass(ID) {
    initializeExpandComplexPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    const TargetMachine *TM =
        &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    const TargetSubtargetInfo *SubtargetInfo = TM->getSubtargetImpl(F);
    const TargetLowering *TLI = SubtargetInfo->getTargetLowering();
    return expandComplexIntrinsics(F, TLI);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }
};
} // namespace

char ExpandComplex::ID;
INITIALIZE_PASS_BEGIN(ExpandComplex, "expand-complex",
                      "Expand complex intrinsics", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(ExpandComplex, "expand-complex",
                    "Expand complex intrinsics", false, false)

FunctionPass *llvm::createExpandComplexPass() { return new ExpandComplex(); }

PreservedAnalyses ExpandComplexPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  /*const auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  if (!expandReductions(F, &TTI))
    return PreservedAnalyses::all();*/
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
