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

static const char *ReductionIntOpcodes[] = {
    "add", "mul", "and", "or", "xor", "smin", "smax", "umin", "umax"};

static const char *ReductionFPOpcodes[] = {"fadd", "fmul", "fmin", "fmax"};

class VPIntrinsicTest : public testing::Test {
protected:
  LLVMContext Context;

  VPIntrinsicTest() : Context() {}

  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> createVPDeclarationModule() {
    const char *BinaryIntOpcodes[] = {"add",  "sub",  "mul", "sdiv", "srem",
                                      "udiv", "urem", "and", "xor",  "or",
                                      "ashr", "lshr", "shl"};
    std::stringstream Str;
    for (const char *BinaryIntOpcode : BinaryIntOpcodes)
      Str << " declare <8 x i32> @llvm.vp." << BinaryIntOpcode
          << ".v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32) ";

    const char *BinaryFPOpcodes[] = {"fadd", "fsub",   "fmul",  "fdiv",
                                     "frem", "minnum", "maxnum"};
    for (const char *BinaryFPOpcode : BinaryFPOpcodes)
      Str << " declare <8 x float> @llvm.vp." << BinaryFPOpcode
          << ".v8f32(<8 x float>, <8 x float>, <8 x i1>, i32) ";

    Str << " declare <8 x float> @llvm.vp.floor.v8f32(<8 x float>, <8 x i1>, "
           "i32)";
    Str << " declare <8 x float> @llvm.vp.round.v8f32(<8 x float>, <8 x i1>, "
           "i32)";
    Str << " declare <8 x float> @llvm.vp.roundeven.v8f32(<8 x float>, <8 x "
           "i1>, "
           "i32)";
    Str << " declare <8 x float> @llvm.vp.ceil.v8f32(<8 x float>, <8 x i1>, "
           "i32)";
    Str << " declare <8 x float> @llvm.vp.fneg.v8f32(<8 x float>, <8 x i1>, "
           "i32)";
    Str << " declare <8 x float> @llvm.vp.fabs.v8f32(<8 x float>, <8 x i1>, "
           "i32)";
    Str << " declare <8 x float> @llvm.vp.sqrt.v8f32(<8 x float>, <8 x i1>, "
           "i32)";
    Str << " declare <8 x float> @llvm.vp.fma.v8f32(<8 x float>, <8 x float>, "
           "<8 x float>, <8 x i1>, i32) ";
    Str << " declare <8 x float> @llvm.vp.fmuladd.v8f32(<8 x float>, "
           "<8 x float>, <8 x float>, <8 x i1>, i32) ";

    Str << " declare void @llvm.vp.store.v8i32.p0v8i32(<8 x i32>, <8 x i32>*, "
           "<8 x i1>, i32) ";
    Str << "declare void "
           "@llvm.experimental.vp.strided.store.v8i32.i32(<8 x i32>, "
           "i32*, i32, <8 x i1>, i32) ";
    Str << "declare void "
           "@llvm.experimental.vp.strided.store.v8i32.p1i32.i32(<8 x i32>, "
           "i32 addrspace(1)*, i32, <8 x i1>, i32) ";
    Str << " declare void @llvm.vp.scatter.v8i32.v8p0i32(<8 x i32>, <8 x "
           "i32*>, <8 x i1>, i32) ";
    Str << " declare <8 x i32> @llvm.vp.load.v8i32.p0v8i32(<8 x i32>*, <8 x "
           "i1>, i32) ";
    Str << "declare <8 x i32> "
           "@llvm.experimental.vp.strided.load.v8i32.i32(i32*, i32, <8 "
           "x i1>, i32) ";
    Str << "declare <8 x i32> "
           "@llvm.experimental.vp.strided.load.v8i32.p1i32.i32(i32 "
           "addrspace(1)*, i32, <8 x i1>, i32) ";
    Str << " declare <8 x i32> @llvm.vp.gather.v8i32.v8p0i32(<8 x i32*>, <8 x "
           "i1>, i32) ";

    for (const char *ReductionOpcode : ReductionIntOpcodes)
      Str << " declare i32 @llvm.vp.reduce." << ReductionOpcode
          << ".v8i32(i32, <8 x i32>, <8 x i1>, i32) ";

    for (const char *ReductionOpcode : ReductionFPOpcodes)
      Str << " declare float @llvm.vp.reduce." << ReductionOpcode
          << ".v8f32(float, <8 x float>, <8 x i1>, i32) ";

    Str << " declare <8 x i32> @llvm.vp.merge.v8i32(<8 x i1>, <8 x i32>, <8 x "
           "i32>, i32)";
    Str << " declare <8 x i32> @llvm.vp.select.v8i32(<8 x i1>, <8 x i32>, <8 x "
           "i32>, i32)";
    Str << " declare <8 x i32> @llvm.experimental.vp.splice.v8i32(<8 x "
           "i32>, <8 x i32>, i32, <8 x i1>, i32, i32) ";

    Str << " declare <8 x i32> @llvm.vp.fptoui.v8i32"
        << ".v8f32(<8 x float>, <8 x i1>, i32) ";
    Str << " declare <8 x i32> @llvm.vp.fptosi.v8i32"
        << ".v8f32(<8 x float>, <8 x i1>, i32) ";
    Str << " declare <8 x float> @llvm.vp.uitofp.v8f32"
        << ".v8i32(<8 x i32>, <8 x i1>, i32) ";
    Str << " declare <8 x float> @llvm.vp.sitofp.v8f32"
        << ".v8i32(<8 x i32>, <8 x i1>, i32) ";
    Str << " declare <8 x float> @llvm.vp.fptrunc.v8f32"
        << ".v8f64(<8 x double>, <8 x i1>, i32) ";
    Str << " declare <8 x double> @llvm.vp.fpext.v8f64"
        << ".v8f32(<8 x float>, <8 x i1>, i32) ";
    Str << " declare <8 x i32> @llvm.vp.trunc.v8i32"
        << ".v8i64(<8 x i64>, <8 x i1>, i32) ";
    Str << " declare <8 x i64> @llvm.vp.zext.v8i64"
        << ".v8i32(<8 x i32>, <8 x i1>, i32) ";
    Str << " declare <8 x i64> @llvm.vp.sext.v8i64"
        << ".v8i32(<8 x i32>, <8 x i1>, i32) ";
    Str << " declare <8 x i32> @llvm.vp.ptrtoint.v8i32"
        << ".v8p0i32(<8 x i32*>, <8 x i1>, i32) ";
    Str << " declare <8 x i32*> @llvm.vp.inttoptr.v8p0i32"
        << ".v8i32(<8 x i32>, <8 x i1>, i32) ";

    Str << " declare <8 x i1> @llvm.vp.fcmp.v8f32"
        << "(<8 x float>, <8 x float>, metadata, <8 x i1>, i32) ";
    Str << " declare <8 x i1> @llvm.vp.icmp.v8i16"
        << "(<8 x i16>, <8 x i16>, metadata, <8 x i1>, i32) ";

    return parseAssemblyString(Str.str(), Err, C);
  }
};

/// Check that the property scopes include/llvm/IR/VPIntrinsics.def are closed.
TEST_F(VPIntrinsicTest, VPIntrinsicsDefScopes) {
  Optional<Intrinsic::ID> ScopeVPID;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...)                                 \
  ASSERT_FALSE(ScopeVPID.has_value());                                         \
  ScopeVPID = Intrinsic::VPID;
#define END_REGISTER_VP_INTRINSIC(VPID)                                        \
  ASSERT_TRUE(ScopeVPID.has_value());                                          \
  ASSERT_EQ(ScopeVPID.value(), Intrinsic::VPID);                               \
  ScopeVPID = None;

  Optional<ISD::NodeType> ScopeOPC;
#define BEGIN_REGISTER_VP_SDNODE(SDOPC, ...)                                   \
  ASSERT_FALSE(ScopeOPC.has_value());                                          \
  ScopeOPC = ISD::SDOPC;
#define END_REGISTER_VP_SDNODE(SDOPC)                                          \
  ASSERT_TRUE(ScopeOPC.has_value());                                           \
  ASSERT_EQ(ScopeOPC.value(), ISD::SDOPC);                                     \
  ScopeOPC = None;
#include "llvm/IR/VPIntrinsics.def"

  ASSERT_FALSE(ScopeVPID.has_value());
  ASSERT_FALSE(ScopeOPC.has_value());
}

/// Check that every VP intrinsic in the test module is recognized as a VP
/// intrinsic.
TEST_F(VPIntrinsicTest, VPModuleComplete) {
  std::unique_ptr<Module> M = createVPDeclarationModule();
  assert(M);

  // Check that all @llvm.vp.* functions in the module are recognized vp
  // intrinsics.
  std::set<Intrinsic::ID> SeenIDs;
  for (const auto &VPDecl : *M) {
    ASSERT_TRUE(VPDecl.isIntrinsic());
    ASSERT_TRUE(VPIntrinsic::isVPIntrinsic(VPDecl.getIntrinsicID()));
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

  const bool Expected[] = {false, true,  false, false, false, true,
                           false, false, true,  false, true,  false};
  const auto *ExpectedIt = std::begin(Expected);
  for (auto &I : F->getEntryBlock()) {
    VPIntrinsic *VPI = dyn_cast<VPIntrinsic>(&I);
    if (!VPI)
      continue;

    ASSERT_NE(ExpectedIt, std::end(Expected));
    ASSERT_EQ(*ExpectedIt, VPI->canIgnoreVectorLengthParam());
    ++ExpectedIt;
  }
}

/// Check that the argument returned by
/// VPIntrinsic::get<X>ParamPos(Intrinsic::ID) has the expected type.
TEST_F(VPIntrinsicTest, GetParamPos) {
  std::unique_ptr<Module> M = createVPDeclarationModule();
  assert(M);

  for (Function &F : *M) {
    ASSERT_TRUE(F.isIntrinsic());
    Optional<unsigned> MaskParamPos =
        VPIntrinsic::getMaskParamPos(F.getIntrinsicID());
    if (MaskParamPos) {
      Type *MaskParamType = F.getArg(MaskParamPos.value())->getType();
      ASSERT_TRUE(MaskParamType->isVectorTy());
      ASSERT_TRUE(
          cast<VectorType>(MaskParamType)->getElementType()->isIntegerTy(1));
    }

    Optional<unsigned> VecLenParamPos =
        VPIntrinsic::getVectorLengthParamPos(F.getIntrinsicID());
    if (VecLenParamPos) {
      Type *VecLenParamType = F.getArg(VecLenParamPos.value())->getType();
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
    Intrinsic::ID VPID = VPIntrinsic::getForOpcode(OC);
    // No equivalent VP intrinsic available.
    if (VPID == Intrinsic::not_intrinsic)
      continue;

    Optional<unsigned> RoundTripOC =
        VPIntrinsic::getFunctionalOpcodeForVP(VPID);
    // No equivalent Opcode available.
    if (!RoundTripOC)
      continue;

    ASSERT_EQ(*RoundTripOC, OC);
    ++FullTripCounts;
  }
  ASSERT_NE(FullTripCounts, 0u);
}

/// Check that going from VP intrinsic to Opcode and back results in the same
/// intrinsic id.
TEST_F(VPIntrinsicTest, IntrinsicIDRoundTrip) {
  std::unique_ptr<Module> M = createVPDeclarationModule();
  assert(M);

  unsigned FullTripCounts = 0;
  for (const auto &VPDecl : *M) {
    auto VPID = VPDecl.getIntrinsicID();
    Optional<unsigned> OC = VPIntrinsic::getFunctionalOpcodeForVP(VPID);

    // no equivalent Opcode available
    if (!OC)
      continue;

    Intrinsic::ID RoundTripVPID = VPIntrinsic::getForOpcode(*OC);

    ASSERT_EQ(RoundTripVPID, VPID);
    ++FullTripCounts;
  }
  ASSERT_NE(FullTripCounts, 0u);
}

/// Check that VPIntrinsic::getDeclarationForParams works.
TEST_F(VPIntrinsicTest, VPIntrinsicDeclarationForParams) {
  std::unique_ptr<Module> M = createVPDeclarationModule();
  assert(M);

  auto OutM = std::make_unique<Module>("", M->getContext());

  for (auto &F : *M) {
    auto *FuncTy = F.getFunctionType();

    // Declare intrinsic anew with explicit types.
    std::vector<Value *> Values;
    for (auto *ParamTy : FuncTy->params())
      Values.push_back(UndefValue::get(ParamTy));

    ASSERT_NE(F.getIntrinsicID(), Intrinsic::not_intrinsic);
    auto *NewDecl = VPIntrinsic::getDeclarationForParams(
        OutM.get(), F.getIntrinsicID(), FuncTy->getReturnType(), Values);
    ASSERT_TRUE(NewDecl);

    // Check that 'old decl' == 'new decl'.
    ASSERT_EQ(F.getIntrinsicID(), NewDecl->getIntrinsicID());
    FunctionType::param_iterator ItNewParams =
        NewDecl->getFunctionType()->param_begin();
    FunctionType::param_iterator EndItNewParams =
        NewDecl->getFunctionType()->param_end();
    for (auto *ParamTy : FuncTy->params()) {
      ASSERT_NE(ItNewParams, EndItNewParams);
      ASSERT_EQ(*ItNewParams, ParamTy);
      ++ItNewParams;
    }
  }
}

/// Check that the HANDLE_VP_TO_CONSTRAINEDFP maps to an existing intrinsic with
/// the right amount of constrained-fp metadata args.
TEST_F(VPIntrinsicTest, HandleToConstrainedFP) {
#define VP_PROPERTY_CONSTRAINEDFP(HASROUND, HASEXCEPT, CFPID)                  \
  {                                                                            \
    SmallVector<Intrinsic::IITDescriptor, 5> T;                                \
    Intrinsic::getIntrinsicInfoTableEntries(Intrinsic::CFPID, T);              \
    unsigned NumMetadataArgs = 0;                                              \
    for (auto TD : T)                                                          \
      NumMetadataArgs += (TD.Kind == Intrinsic::IITDescriptor::Metadata);      \
    bool IsCmp = Intrinsic::CFPID == Intrinsic::experimental_constrained_fcmp; \
    ASSERT_EQ(NumMetadataArgs, (unsigned)(IsCmp + HASROUND + HASEXCEPT));      \
  }
#include "llvm/IR/VPIntrinsics.def"
}

} // end anonymous namespace

/// Check various properties of VPReductionIntrinsics
TEST_F(VPIntrinsicTest, VPReductions) {
  LLVMContext C;
  SMDiagnostic Err;

  std::stringstream Str;
  Str << "declare <8 x i32> @llvm.vp.mul.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, "
         "i32)";
  for (const char *ReductionOpcode : ReductionIntOpcodes)
    Str << " declare i32 @llvm.vp.reduce." << ReductionOpcode
        << ".v8i32(i32, <8 x i32>, <8 x i1>, i32) ";

  for (const char *ReductionOpcode : ReductionFPOpcodes)
    Str << " declare float @llvm.vp.reduce." << ReductionOpcode
        << ".v8f32(float, <8 x float>, <8 x i1>, i32) ";

  Str << "define void @test_reductions(i32 %start, <8 x i32> %val, float "
         "%fpstart, <8 x float> %fpval, <8 x i1> %m, i32 %vl) {";

  // Mix in a regular non-reduction intrinsic to check that the
  // VPReductionIntrinsic subclass works as intended.
  Str << "  %r0 = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %val, <8 x i32> "
         "%val, <8 x i1> %m, i32 %vl)";

  unsigned Idx = 1;
  for (const char *ReductionOpcode : ReductionIntOpcodes)
    Str << "  %r" << Idx++ << " = call i32 @llvm.vp.reduce." << ReductionOpcode
        << ".v8i32(i32 %start, <8 x i32> %val, <8 x i1> %m, i32 %vl)";
  for (const char *ReductionOpcode : ReductionFPOpcodes)
    Str << "  %r" << Idx++ << " = call float @llvm.vp.reduce."
        << ReductionOpcode
        << ".v8f32(float %fpstart, <8 x float> %fpval, <8 x i1> %m, i32 %vl)";

  Str << "  ret void"
         "}";

  std::unique_ptr<Module> M = parseAssemblyString(Str.str(), Err, C);
  assert(M);

  auto *F = M->getFunction("test_reductions");
  assert(F);

  for (const auto &I : F->getEntryBlock()) {
    const VPIntrinsic *VPI = dyn_cast<VPIntrinsic>(&I);
    if (!VPI)
      continue;

    Intrinsic::ID ID = VPI->getIntrinsicID();
    const auto *VPRedI = dyn_cast<VPReductionIntrinsic>(&I);

    if (!VPReductionIntrinsic::isVPReduction(ID)) {
      EXPECT_EQ(VPRedI, nullptr);
      EXPECT_EQ(VPReductionIntrinsic::getStartParamPos(ID).has_value(), false);
      EXPECT_EQ(VPReductionIntrinsic::getVectorParamPos(ID).has_value(), false);
      continue;
    }

    EXPECT_EQ(VPReductionIntrinsic::getStartParamPos(ID).has_value(), true);
    EXPECT_EQ(VPReductionIntrinsic::getVectorParamPos(ID).has_value(), true);
    ASSERT_NE(VPRedI, nullptr);
    EXPECT_EQ(VPReductionIntrinsic::getStartParamPos(ID),
              VPRedI->getStartParamPos());
    EXPECT_EQ(VPReductionIntrinsic::getVectorParamPos(ID),
              VPRedI->getVectorParamPos());
    EXPECT_EQ(VPRedI->getStartParamPos(), 0u);
    EXPECT_EQ(VPRedI->getVectorParamPos(), 1u);
  }
}
