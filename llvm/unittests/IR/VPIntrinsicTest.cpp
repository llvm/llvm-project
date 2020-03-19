//===- VPIntrinsicTest.cpp - VPIntrinsic unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class VPIntrinsicTest : public testing::Test {
protected:
  LLVMContext Context;

  VPIntrinsicTest() : Context() {}

  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> CreateVPDeclarationModule() {
      return parseAssemblyString(
" declare <8 x i32> @llvm.vp.add.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.sub.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.mul.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.srem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.urem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.and.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.xor.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.or.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) "
" declare <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)  "
" declare <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)  "
" declare <8 x i32> @llvm.vp.shl.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) ",
          Err, C);
  }
};

/// Check that VPIntrinsic:canIgnoreVectorLengthParam() returns true
/// if the vector length parameter does not mask off any lanes.
TEST_F(VPIntrinsicTest, CanIgnoreVectorLength) {
  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> M =
      parseAssemblyString(
"declare <256 x i64> @llvm.vp.mul.v256i64(<256 x i64>, <256 x i64>, <256 x i1>, i32)"
"declare <vscale x 2 x i64> @llvm.vp.mul.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i1>, i32)"
"declare i32 @llvm.vscale.i32()"
"define void @test_static_vlen( "
"      <256 x i64> %i0, <vscale x 2 x i64> %si0,"
"      <256 x i64> %i1, <vscale x 2 x i64> %si1,"
"      <256 x i1> %m, <vscale x 2 x i1> %sm, i32 %vl) { "
"  %r0 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 %vl)"
"  %r1 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 256)"
"  %r2 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 0)"
"  %r3 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 7)"
"  %r4 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 123)"
"  %vs = call i32 @llvm.vscale.i32()"
"  %vs.i64 = mul i32 %vs, 2"
"  %r5 = call <vscale x 2 x i64> @llvm.vp.mul.nxv2i64(<vscale x 2 x i64> %si0, <vscale x 2 x i64> %si1, <vscale x 2 x i1> %sm, i32 %vs.i64)"
"  %r6 = call <vscale x 2 x i64> @llvm.vp.mul.nxv2i64(<vscale x 2 x i64> %si0, <vscale x 2 x i64> %si1, <vscale x 2 x i1> %sm, i32 99999)"
"  ret void "
"}",
          Err, C);

  auto *F = M->getFunction("test_static_vlen");
  assert(F);

  const int NumExpected = 7;
  const bool Expected[] = {false, true, false, false, false, true, false};
  int i = 0;
  for (auto &I : F->getEntryBlock()) {
    VPIntrinsic *VPI = dyn_cast<VPIntrinsic>(&I);
    if (!VPI)
      continue;

    ASSERT_LT(i, NumExpected);
    ASSERT_EQ(Expected[i], VPI->canIgnoreVectorLengthParam());
    ++i;
  }
}

/// Check that the argument returned by
/// VPIntrinsic::Get<X>ParamPos(Intrinsic::ID) has the expected type.
TEST_F(VPIntrinsicTest, GetParamPos) {
  std::unique_ptr<Module> M = CreateVPDeclarationModule();
  assert(M);

  for (Function &F : *M) {
    ASSERT_TRUE(F.isIntrinsic());
    Optional<int> MaskParamPos =
        VPIntrinsic::GetMaskParamPos(F.getIntrinsicID());
    if (MaskParamPos.hasValue()) {
      Type *MaskParamType = F.getArg(MaskParamPos.getValue())->getType();
      ASSERT_TRUE(MaskParamType->isVectorTy());
      ASSERT_TRUE(MaskParamType->getVectorElementType()->isIntegerTy(1));
    }

    Optional<int> VecLenParamPos =
        VPIntrinsic::GetVectorLengthParamPos(F.getIntrinsicID());
    if (VecLenParamPos.hasValue()) {
      Type *VecLenParamType = F.getArg(VecLenParamPos.getValue())->getType();
      ASSERT_TRUE(VecLenParamType->isIntegerTy(32));
    }
  }
}

/// Check that going from Opcode to VP intrinsic and back results in the same
/// Opcode.
TEST_F(VPIntrinsicTest, OpcodeRoundTrip) {
  std::vector<unsigned> Opcodes;
  Opcodes.reserve(100);

  {
#define HANDLE_INST(OCNum, OCName, Class) Opcodes.push_back(OCNum);
#include "llvm/IR/Instruction.def"
  }

  unsigned FullTripCounts = 0;
  for (unsigned OC : Opcodes) {
    Intrinsic::ID VPID = VPIntrinsic::GetForOpcode(OC);
    // no equivalent VP intrinsic available
    if (VPID == Intrinsic::not_intrinsic)
      continue;

    unsigned RoundTripOC = VPIntrinsic::GetFunctionalOpcodeForVP(VPID);
    // no equivalent Opcode available
    if (RoundTripOC == Instruction::Call)
      continue;

    ASSERT_EQ(RoundTripOC, OC);
    ++FullTripCounts;
  }
  ASSERT_NE(FullTripCounts, 0u);
}

} // end anonymous namespace
