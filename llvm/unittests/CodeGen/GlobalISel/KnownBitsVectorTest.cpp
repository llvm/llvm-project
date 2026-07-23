//===- KnownBitsTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

// Vector KnownBits track bits that are common for all vector scalar elements.
// For tests below KnownBits analysis is same as for scalar/pointer types, tests
// are mostly copied from KnownBitsTest.cpp using splat vectors and have the
// same result.

TEST_F(AArch64GISelMITest, TestVectorSignBitIsZero) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  const LLT V2S32 = LLT::fixed_vector(2, 32);
  // Vector buildConstant makes splat G_BUILD_VECTOR instruction.
  auto SignBit = B.buildConstant(V2S32, 0x80000000);
  auto Zero = B.buildConstant(V2S32, 0);

  const LLT S32 = LLT::scalar(32);
  auto NonSplat =
      B.buildBuildVector(V2S32, {B.buildConstant(S32, 1).getReg(0),
                                 B.buildConstant(S32, 2).getReg(0)});
  auto NonSplat2 =
      B.buildBuildVector(V2S32, {B.buildConstant(S32, 0x80000000).getReg(0),
                                 B.buildConstant(S32, 0x80000004).getReg(0)});
  // signBitIsZero is true for elt 0 and false for elt 1 GISelValueTracking
  // takes common bits so this is false.
  auto NonSplat3 =
      B.buildBuildVector(V2S32, {B.buildConstant(S32, 0x80000000).getReg(0),
                                 B.buildConstant(S32, 0x8).getReg(0)});
  GISelValueTracking KnownBits(*MF);

  EXPECT_TRUE(KnownBits.signBitIsZero(Zero.getReg(0)));
  EXPECT_FALSE(KnownBits.signBitIsZero(SignBit.getReg(0)));
  EXPECT_TRUE(KnownBits.signBitIsZero(NonSplat.getReg(0)));
  EXPECT_FALSE(KnownBits.signBitIsZero(NonSplat2.getReg(0)));
  EXPECT_FALSE(KnownBits.signBitIsZero(NonSplat3.getReg(0)));
}

TEST_F(AMDGPUGISelMITest, TestVectorIsKnownToBeAPowerOfTwo) {

  StringRef MIRString = R"(
  %zero:_(s32) = G_CONSTANT i32 0
  %zero_splat:_(<2 x s32>) = G_BUILD_VECTOR %zero:_(s32), %zero:_(s32)
  %one:_(s32) = G_CONSTANT i32 1
  %one_splat:_(<2 x s32>) = G_BUILD_VECTOR %one:_(s32), %one:_(s32)
  %two:_(s32) = G_CONSTANT i32 2
  %two_splat:_(<2 x s32>) = G_BUILD_VECTOR %two:_(s32), %two:_(s32)
  %three:_(s32) = G_CONSTANT i32 3
  %three_splat:_(<2 x s32>) = G_BUILD_VECTOR %three:_(s32), %three:_(s32)
  %five:_(s32) = G_CONSTANT i32 5
  %five_splat:_(<2 x s32>) = G_BUILD_VECTOR %five:_(s32), %five:_(s32)
  %copy_zero_splat:_(<2 x s32>) = COPY %zero_splat
  %copy_one_splat:_(<2 x s32>) = COPY %one_splat
  %copy_two_splat:_(<2 x s32>) = COPY %two_splat
  %copy_three_splat:_(<2 x s32>) = COPY %three_splat

  %trunc_two_splat:_(<2 x s1>) = G_TRUNC %two_splat
  %trunc_three_splat:_(<2 x s1>) = G_TRUNC %three_splat
  %trunc_five_splat:_(<2 x s1>) = G_TRUNC %five_splat

  %copy_trunc_two_splat:_(<2 x s1>) = COPY %trunc_two_splat
  %copy_trunc_three_splat:_(<2 x s1>) = COPY %trunc_three_splat
  %copy_trunc_five_splat:_(<2 x s1>) = COPY %trunc_five_splat

  %ptr:_(p1) = G_IMPLICIT_DEF
  %shift_amt:_(<2 x s32>) = G_LOAD %ptr :: (load (<2 x s32>), addrspace 1)

  %shl_1:_(<2 x s32>) = G_SHL %one_splat, %shift_amt
  %copy_shl_1:_(<2 x s32>) = COPY %shl_1

  %shl_2:_(<2 x s32>) = G_SHL %two_splat, %shift_amt
  %copy_shl_2:_(<2 x s32>) = COPY %shl_2

  %not_sign_mask:_(<2 x s32>) = G_LOAD %ptr :: (load (<2 x s32>), addrspace 1)
  %sign_mask:_(s32) = G_CONSTANT i32 -2147483648
  %sign_mask_splat:_(<2 x s32>) = G_BUILD_VECTOR %sign_mask:_(s32), %sign_mask:_(s32)

  %lshr_not_sign_mask:_(<2 x s32>) = G_LSHR %not_sign_mask, %shift_amt
  %copy_lshr_not_sign_mask:_(<2 x s32>) = COPY %lshr_not_sign_mask

  %lshr_sign_mask:_(<2 x s32>) = G_LSHR %sign_mask_splat, %shift_amt
  %copy_lshr_sign_mask:_(<2 x s32>) = COPY %lshr_sign_mask

  %or_pow2:_(<2 x s32>) = G_OR %zero_splat, %two_splat
  %copy_or_pow2:_(<2 x s32>) = COPY %or_pow2
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  GISelValueTracking VT(*MF);

  Register CopyZero = Copies[Copies.size() - 12];
  Register CopyOne = Copies[Copies.size() - 11];
  Register CopyTwo = Copies[Copies.size() - 10];
  Register CopyThree = Copies[Copies.size() - 9];
  Register CopyTruncTwo = Copies[Copies.size() - 8];
  Register CopyTruncThree = Copies[Copies.size() - 7];
  Register CopyTruncFive = Copies[Copies.size() - 6];

  Register CopyShl1 = Copies[Copies.size() - 5];
  Register CopyShl2 = Copies[Copies.size() - 4];

  Register CopyLShrNotSignMask = Copies[Copies.size() - 3];
  Register CopyLShrSignMask = Copies[Copies.size() - 2];
  Register CopyOrPow2 = Copies[Copies.size() - 1];

  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyZero, *MRI, &VT));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyOne, *MRI, &VT));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyTwo, *MRI, &VT));
  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyThree, *MRI, &VT));

  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyTruncTwo, *MRI, &VT));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyTruncThree, *MRI, &VT));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyTruncFive, *MRI, &VT));
  // TODO: check for vector(splat) shift amount.
  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyShl1, *MRI, &VT));
  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyShl2, *MRI, &VT));

  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyLShrNotSignMask, *MRI, &VT));
  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyLShrSignMask, *MRI, &VT));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyOrPow2, *MRI, &VT));
}

TEST_F(AArch64GISelMITest, TestVectorMetadata) {
  StringRef MIRString = R"(
   %imp:_(p0) = G_IMPLICIT_DEF
   %load:_(<2 x s8>) = G_LOAD %imp(p0) :: (load (<2 x s8>))
   %ext:_(<2 x s32>) = G_ZEXT %load(<2 x s8>)
   %cst_elt:_(s32) = G_CONSTANT i32 1
   %cst:_(<2 x s32>) = G_BUILD_VECTOR %cst_elt:_(s32), %cst_elt:_(s32)
   %and:_(<2 x s32>) = G_AND %ext, %cst
   %copy:_(<2 x s32>) = COPY %and(<2 x s32>)
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  MachineInstr *And = MRI->getVRegDef(SrcReg);
  MachineInstr *Ext = MRI->getVRegDef(And->getOperand(1).getReg());
  MachineInstr *Load = MRI->getVRegDef(Ext->getOperand(1).getReg());
  IntegerType *Int8Ty = Type::getInt8Ty(Context);

  Metadata *LowAndHigh[] = {
      ConstantAsMetadata::get(ConstantInt::get(Int8Ty, 0)),
      ConstantAsMetadata::get(ConstantInt::get(Int8Ty, 2))};
  auto *NewMDNode = MDNode::get(Context, LowAndHigh);
  const MachineMemOperand *OldMMO = *Load->memoperands_begin();
  MachineMemOperand NewMMO(OldMMO->getPointerInfo(), OldMMO->getFlags(),
                           OldMMO->getMemoryType(), OldMMO->getAlign(),
                           OldMMO->getAAInfo(), NewMDNode);
  MachineIRBuilder MIB(*Load);
  MIB.buildLoad(Load->getOperand(0), Load->getOperand(1), NewMMO);
  Load->eraseFromParent();

  GISelValueTracking Info(*MF);
  KnownBits Res = Info.getKnownBits(And->getOperand(1).getReg());

  EXPECT_TRUE(Res.One.isZero());

  APInt Mask(Res.getBitWidth(), 1);
  Mask.flipAllBits();
  EXPECT_EQ(Mask.getZExtValue(), Res.Zero.getZExtValue());
}
