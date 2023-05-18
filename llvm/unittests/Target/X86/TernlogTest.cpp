//===- LICMTest.cpp - LICM unit tests -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

#include "gtest/gtest.h"

#include <random>

namespace llvm {
static std::unique_ptr<LLVMTargetMachine> initTM() {
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();

  auto TT(Triple::normalize("x86_64--"));
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
      TheTarget->createTargetMachine(TT, "", "", TargetOptions(), std::nullopt,
                                     std::nullopt, CodeGenOptLevel::Default)));
}

struct TernTester {
  unsigned NElem;
  unsigned ElemWidth;
  std::mt19937_64 Rng;
  unsigned ImmVal;
  SmallVector<uint64_t, 16> VecElems[3];

  void updateImm(uint8_t NewImmVal) { ImmVal = NewImmVal; }
  void updateNElem(unsigned NewNElem) {
    NElem = NewNElem;
    for (unsigned I = 0; I < 3; ++I) {
      VecElems[I].resize(NElem);
    }
  }
  void updateElemWidth(unsigned NewElemWidth) {
    ElemWidth = NewElemWidth;
    assert(ElemWidth == 32 || ElemWidth == 64);
  }

  uint64_t getElemMask() const {
    return (~uint64_t(0)) >> ((ElemWidth - 0) % 64);
  }

  void RandomizeVecArgs() {
    uint64_t ElemMask = getElemMask();
    for (unsigned I = 0; I < 3; ++I) {
      for (unsigned J = 0; J < NElem; ++J) {
        VecElems[I][J] = Rng() & ElemMask;
      }
    }
  }

  std::pair<std::string, std::string> getScalarInfo() const {
    switch (ElemWidth) {
    case 32:
      return {"i32", "d"};
    case 64:
      return {"i64", "q"};
    default:
      llvm_unreachable("Invalid ElemWidth");
    }
  }
  std::string getScalarType() const { return getScalarInfo().first; }
  std::string getScalarExt() const { return getScalarInfo().second; }
  std::string getVecType() const {
    return "<" + Twine(NElem).str() + " x " + getScalarType() + ">";
  };

  std::string getVecWidth() const { return Twine(NElem * ElemWidth).str(); }
  std::string getFunctionName() const {
    return "@llvm.x86.avx512.pternlog." + getScalarExt() + "." + getVecWidth();
  }
  std::string getFunctionDecl() const {
    return "declare " + getVecType() + getFunctionName() + "(" + getVecType() +
           ", " + getVecType() + ", " + getVecType() + ", " + "i32 immarg)";
  }

  std::string getVecN(unsigned N) const {
    assert(N < 3);
    std::string VecStr = getVecType() + " <";
    for (unsigned I = 0; I < VecElems[N].size(); ++I) {
      if (I != 0)
        VecStr += ", ";
      VecStr += getScalarType() + " " + Twine(VecElems[N][I]).str();
    }
    return VecStr + ">";
  }
  std::string getFunctionCall() const {
    return "tail call " + getVecType() + " " + getFunctionName() + "(" +
           getVecN(0) + ", " + getVecN(1) + ", " + getVecN(2) + ", " + "i32 " +
           Twine(ImmVal).str() + ")";
  }

  std::string getTestText() const {
    return getFunctionDecl() + "\ndefine " + getVecType() +
           "@foo() {\n%r = " + getFunctionCall() + "\nret " + getVecType() +
           " %r\n}\n";
  }

  void checkResult(const Value *V) {
    auto GetValElem = [&](unsigned Idx) -> uint64_t {
      if (auto *CV = dyn_cast<ConstantDataVector>(V))
        return CV->getElementAsInteger(Idx);

      auto *C = dyn_cast<Constant>(V);
      assert(C);
      if (C->isNullValue())
        return 0;
      if (C->isAllOnesValue())
        return ((~uint64_t(0)) >> (ElemWidth % 64));
      if (C->isOneValue())
        return 1;

      llvm_unreachable("Unknown constant type");
    };

    auto ComputeBit = [&](uint64_t A, uint64_t B, uint64_t C) -> uint64_t {
      unsigned BitIdx = ((A & 1) << 2) | ((B & 1) << 1) | (C & 1);
      return (ImmVal >> BitIdx) & 1;
    };

    for (unsigned I = 0; I < NElem; ++I) {
      uint64_t Expec = 0;
      uint64_t AEle = VecElems[0][I];
      uint64_t BEle = VecElems[1][I];
      uint64_t CEle = VecElems[2][I];
      for (unsigned J = 0; J < ElemWidth; ++J) {
        Expec |= ComputeBit(AEle >> J, BEle >> J, CEle >> J) << J;
      }

      ASSERT_EQ(Expec, GetValElem(I));
    }
  }

  void check(LLVMContext &Ctx, FunctionPassManager &FPM,
             FunctionAnalysisManager &FAM) {
    SMDiagnostic Error;
    std::unique_ptr<Module> M = parseAssemblyString(getTestText(), Error, Ctx);
    ASSERT_TRUE(M);
    Function *F = M->getFunction("foo");
    ASSERT_TRUE(F);
    ASSERT_EQ(F->getInstructionCount(), 2u);
    FPM.run(*F, FAM);
    ASSERT_EQ(F->getInstructionCount(), 1u);
    ASSERT_EQ(F->size(), 1u);
    const Instruction *I = F->begin()->getTerminator();
    ASSERT_TRUE(I);
    ASSERT_EQ(I->getNumOperands(), 1u);
    checkResult(I->getOperand(0));
  }
};

TEST(TernlogTest, TestConstantFolding) {
  LLVMContext Ctx;
  FunctionAnalysisManager FAM;
  FunctionPassManager FPM;
  PassBuilder PB;
  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  TargetIRAnalysis TIRA = TargetIRAnalysis(
      [&](const Function &F) { return initTM()->getTargetTransformInfo(F); });

  FAM.registerPass([&] { return TIRA; });
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  FPM.addPass(InstCombinePass());
  TernTester TT;
  for (unsigned NElem = 2; NElem < 16; NElem += NElem) {
    TT.updateNElem(NElem);
    for (unsigned ElemWidth = 32; ElemWidth <= 64; ElemWidth += ElemWidth) {
      if (ElemWidth * NElem > 512 || ElemWidth * NElem < 128)
        continue;
      TT.updateElemWidth(ElemWidth);
      TT.RandomizeVecArgs();
      for (unsigned Imm = 0; Imm < 256; ++Imm) {
        TT.updateImm(Imm);
        TT.check(Ctx, FPM, FAM);
      }
    }
  }
}
} // namespace llvm
