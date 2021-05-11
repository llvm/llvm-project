//===- VPIntrinsicTest.cpp - VPIntrinsic unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <sstream>

using namespace llvm;

namespace {

class VPIntrinsicTest : public testing::Test {
protected:
  LLVMContext Context;

  VPIntrinsicTest() : Context() {}

  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> CreateVPDeclarationModule() {
    const char *BinaryIntOpcodes[] = {"add",  "sub",  "mul", "sdiv", "srem",
                                      "udiv", "urem", "and", "xor",  "or",
                                      "ashr", "lshr", "shl"};
    std::stringstream Str;
    for (const char *BinaryIntOpcode : BinaryIntOpcodes)
      Str << " declare <8 x i32> @llvm.vp." << BinaryIntOpcode
          << ".v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) ";

    return parseAssemblyString(Str.str(), Err, C);
  }
};

/// Check that the property scopes include/llvm/IR/VPIntrinsics.def are closed.
TEST_F(VPIntrinsicTest, VPIntrinsicsDefScopes) {
  Optional<Intrinsic::ID> ScopeVPID;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...)                                 \
  ASSERT_FALSE(ScopeVPID.hasValue());                                          \
  ScopeVPID = Intrinsic::VPID;
#define END_REGISTER_VP_INTRINSIC(VPID)                                        \
  ASSERT_TRUE(ScopeVPID.hasValue());                                           \
  ASSERT_EQ(ScopeVPID.getValue(), Intrinsic::VPID);                            \
  ScopeVPID = None;

  Optional<ISD::NodeType> ScopeOPC;
#define BEGIN_REGISTER_VP_SDNODE(SDOPC, ...)                                   \
  ASSERT_FALSE(ScopeOPC.hasValue());                                           \
  ScopeOPC = ISD::SDOPC;
#define END_REGISTER_VP_SDNODE(SDOPC)                                          \
  ASSERT_TRUE(ScopeOPC.hasValue());                                            \
  ASSERT_EQ(ScopeOPC.getValue(), ISD::SDOPC);                                  \
  ScopeOPC = None;
#include "llvm/IR/VPIntrinsics.def"

  ASSERT_FALSE(ScopeVPID.hasValue());
  ASSERT_FALSE(ScopeOPC.hasValue());
}

/// Check that every VP intrinsic in the test module is recognized as a VP
/// intrinsic.
TEST_F(VPIntrinsicTest, VPModuleComplete) {
  std::unique_ptr<Module> M = CreateVPDeclarationModule();
  assert(M);

  // Check that all @llvm.vp.* functions in the module are recognized vp
  // intrinsics.
  std::set<Intrinsic::ID> SeenIDs;
  for (const auto &VPDecl : *M) {
    ASSERT_TRUE(VPDecl.isIntrinsic());
    ASSERT_TRUE(VPIntrinsic::IsVPIntrinsic(VPDecl.getIntrinsicID()));
    SeenIDs.insert(VPDecl.getIntrinsicID());
  }

  // Check that every registered VP intrinsic has an instance in the test
  // module.
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...)                                 \
  ASSERT_TRUE(SeenIDs.count(Intrinsic::VPID));
#include "llvm/IR/VPIntrinsics.def"
}

/// Check that VPIntrinsic:canIgnoreVectorLengthParam() returns true
/// if the vector length parameter does not mask off any lanes.
TEST_F(VPIntrinsicTest, CanIgnoreVectorLength) {
  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> M =
      parseAssemblyString(
"declare <256 x i64> @llvm.vp.mul.v256i64(<256 x i64>, <256 x i64>, <256 x i1>, i32)"
"declare <vscale x 2 x i64> @llvm.vp.mul.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i1>, i32)"
"declare <vscale x 1 x i64> @llvm.vp.mul.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, <vscale x 1 x i1>, i32)"
"declare i32 @llvm.vscale.i32()"
"define void @test_static_vlen( "
"      <256 x i64> %i0, <vscale x 2 x i64> %si0x2, <vscale x 1 x i64> %si0x1,"
"      <256 x i64> %i1, <vscale x 2 x i64> %si1x2, <vscale x 1 x i64> %si1x1,"
"      <256 x i1> %m, <vscale x 2 x i1> %smx2, <vscale x 1 x i1> %smx1, i32 %vl) { "
"  %r0 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 %vl)"
"  %r1 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 256)"
"  %r2 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 0)"
"  %r3 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 7)"
"  %r4 = call <256 x i64> @llvm.vp.mul.v256i64(<256 x i64> %i0, <256 x i64> %i1, <256 x i1> %m, i32 123)"
"  %vs = call i32 @llvm.vscale.i32()"
"  %vs.x2 = mul i32 %vs, 2"
"  %r5 = call <vscale x 2 x i64> @llvm.vp.mul.nxv2i64(<vscale x 2 x i64> %si0x2, <vscale x 2 x i64> %si1x2, <vscale x 2 x i1> %smx2, i32 %vs.x2)"
"  %r6 = call <vscale x 2 x i64> @llvm.vp.mul.nxv2i64(<vscale x 2 x i64> %si0x2, <vscale x 2 x i64> %si1x2, <vscale x 2 x i1> %smx2, i32 %vs)"
"  %r7 = call <vscale x 2 x i64> @llvm.vp.mul.nxv2i64(<vscale x 2 x i64> %si0x2, <vscale x 2 x i64> %si1x2, <vscale x 2 x i1> %smx2, i32 99999)"
"  %r8 = call <vscale x 1 x i64> @llvm.vp.mul.nxv1i64(<vscale x 1 x i64> %si0x1, <vscale x 1 x i64> %si1x1, <vscale x 1 x i1> %smx1, i32 %vs)"
"  %r9 = call <vscale x 1 x i64> @llvm.vp.mul.nxv1i64(<vscale x 1 x i64> %si0x1, <vscale x 1 x i64> %si1x1, <vscale x 1 x i1> %smx1, i32 1)"
"  %r10 = call <vscale x 1 x i64> @llvm.vp.mul.nxv1i64(<vscale x 1 x i64> %si0x1, <vscale x 1 x i64> %si1x1, <vscale x 1 x i1> %smx1, i32 %vs.x2)"
"  %vs.wat = add i32 %vs, 2"
"  %r11 = call <vscale x 2 x i64> @llvm.vp.mul.nxv2i64(<vscale x 2 x i64> %si0x2, <vscale x 2 x i64> %si1x2, <vscale x 2 x i1> %smx2, i32 %vs.wat)"
"  ret void "
"}",
          Err, C);

  auto *F = M->getFunction("test_static_vlen");
  assert(F);

  const int NumExpected = 12;
  const bool Expected[] = {false, true, false, false, false, true, false, false, true, false, true, false};
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
      ASSERT_TRUE(cast<VectorType>(MaskParamType)->getElementType()->isIntegerTy(1));
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

/// Check that going from VP intrinsic to Opcode and back results in the same
/// intrinsic id.
TEST_F(VPIntrinsicTest, IntrinsicIDRoundTrip) {
  std::unique_ptr<Module> M = CreateVPDeclarationModule();
  assert(M);

  unsigned FullTripCounts = 0;
  for (const auto &VPDecl : *M) {
    auto VPID = VPDecl.getIntrinsicID();
    unsigned OC = VPIntrinsic::GetFunctionalOpcodeForVP(VPID);

    // no equivalent Opcode available
    if (OC == Instruction::Call)
      continue;

    Intrinsic::ID RoundTripVPID = VPIntrinsic::GetForOpcode(OC);

    ASSERT_EQ(RoundTripVPID, VPID);
    ++FullTripCounts;
  }
  ASSERT_NE(FullTripCounts, 0u);
}

} // end anonymous namespace
