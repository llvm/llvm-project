//===- llvm/unittests/MC/AMDGPU/AMDGPUMCExprTest.cpp
//---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AMDGPUMCExpr.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "gtest/gtest.h"

using namespace llvm;

class AMDGPUMCExprTest : public AMDGPUTestBase {

protected:
  std::unique_ptr<GCNTargetMachine> TM;
  std::unique_ptr<LLVMContext> LLVMCtx;
  std::unique_ptr<GCNSubtarget> ST;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<Module> M;

  AMDGPUMCExprTest() {

    TM = createAMDGPUTargetMachine("amdgpu10.10--amdpal", "", "");

    LLVMCtx = std::make_unique<LLVMContext>();
    M = std::make_unique<Module>("Module", *LLVMCtx);
    M->setDataLayout(TM->createDataLayout());
    auto *FType = FunctionType::get(Type::getVoidTy(*LLVMCtx), false);
    auto *F = Function::Create(FType, GlobalValue::ExternalLinkage, "Test", *M);
    MMI = std::make_unique<MachineModuleInfo>(TM.get());

    ST = std::make_unique<GCNSubtarget>(TM->getTargetTriple(),
                                        TM->getTargetCPU(),
                                        TM->getTargetFeatureString(), *TM);

    MF = std::make_unique<MachineFunction>(*F, *TM, *ST, MMI->getContext(), 1);
  }
};
// Next two tests showcases the folding of foldAMDGPUMCExpr function that uses
// KnownBits to simplify expressions as much as possible
//  max(((external_unknown & 0) | 0x8000000000000000), 5)
TEST_F(AMDGPUMCExprTest, MaxFoldKnownBitsSignedness) {
  MCContext &Ctx = MF->getContext();

  // Set up the unknown/undefined part of the expression
  MCSymbol *Sym = Ctx.getOrCreateSymbol("external_unknown");
  const MCExpr *SymRef = MCSymbolRefExpr::create(Sym, Ctx);

  // Set up the rest of the expression
  const MCExpr *Masked =
      MCBinaryExpr::createAnd(SymRef, MCConstantExpr::create(0, Ctx), Ctx);
  const MCExpr *SignBit = AMDGPUMCExpr::createOr(
      {Masked, MCConstantExpr::create(INT64_MIN, Ctx)}, Ctx);
  const MCExpr *MaxExpr =
      AMDGPUMCExpr::createMax({SignBit, MCConstantExpr::create(5, Ctx)}, Ctx);

  // running evaluateAsAbsolute will fail since external_unknown is undefined.
  int64_t RuntimeRst;
  EXPECT_FALSE(MaxExpr->evaluateAsAbsolute(RuntimeRst));

  // fold expression and then running evaluateAsAbsolute should now succeed with
  // the correct result
  const MCExpr *Folded = AMDGPU::foldAMDGPUMCExpr(MaxExpr, Ctx);
  int64_t FoldedVal;
  ASSERT_TRUE(Folded->evaluateAsAbsolute(FoldedVal));
  EXPECT_EQ(FoldedVal, int64_t{5});
}

// min(((external_unknown & 0) | 0x8000000000000000), 5)
TEST_F(AMDGPUMCExprTest, MinFoldKnownBitsSignedness) {
  MCContext &Ctx = MF->getContext();

  // Set up the unknown/undefined part of the expression
  MCSymbol *Sym = Ctx.getOrCreateSymbol("external_unknown");
  const MCExpr *SymRef = MCSymbolRefExpr::create(Sym, Ctx);

  // Set up the rest of the expression
  const MCExpr *Masked =
      MCBinaryExpr::createAnd(SymRef, MCConstantExpr::create(0, Ctx), Ctx);
  const MCExpr *SignBit = AMDGPUMCExpr::createOr(
      {Masked, MCConstantExpr::create(INT64_MIN, Ctx)}, Ctx);
  const MCExpr *MinExpr =
      AMDGPUMCExpr::createMin({SignBit, MCConstantExpr::create(5, Ctx)}, Ctx);

  // running evaluateAsAbsolute will fail since external_unknown is undefined.
  int64_t AbsoluteRst;
  EXPECT_FALSE(MinExpr->evaluateAsAbsolute(AbsoluteRst));

  // fold expression and then running evaluateAsAbsolute should now succeed with
  // the correct result
  const MCExpr *Folded = AMDGPU::foldAMDGPUMCExpr(MinExpr, Ctx);
  int64_t FoldedVal;
  ASSERT_TRUE(Folded->evaluateAsAbsolute(FoldedVal));
  EXPECT_EQ(FoldedVal, INT64_MIN);
}