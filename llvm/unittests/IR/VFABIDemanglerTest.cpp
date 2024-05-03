//===------- VFABIDemanglerTest.cpp - VFABI unit tests  -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/VFABIDemangler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;

namespace {

static LLVMContext Ctx;

/// Perform tests against VFABI Rules. `invokeParser` creates a VFInfo object
/// and a scalar FunctionType, which are used by tests to check that:
/// 1. The scalar and vector names are correct.
/// 2. The number of parameters from the parsed mangled name matches the number
///    of arguments in the scalar function passed as FunctionType string.
/// 3. The number of vector parameters and their types match the values
///    specified in the test.
///    On masked functions it also checks that the last parameter is a mask (ie,
///    GlobalPredicate).
/// 4. The vector function is correctly found to have a mask.
///
class VFABIParserTest : public ::testing::Test {
private:
  // Parser output.
  VFInfo Info;
  /// Reset the data needed for the test.
  void reset(const StringRef ScalarFTyStr) {
    M = parseAssemblyString("declare void @dummy()", Err, Ctx);
    EXPECT_NE(M.get(), nullptr)
        << "Loading an invalid module.\n " << Err.getMessage() << "\n";
    Type *Ty = parseType(ScalarFTyStr, Err, *(M.get()));
    ScalarFTy = dyn_cast<FunctionType>(Ty);
    EXPECT_NE(ScalarFTy, nullptr)
        << "Invalid function type string: " << ScalarFTyStr << "\n"
        << Err.getMessage() << "\n";
    // Reset the VFInfo
    Info = VFInfo();
  }

  // Data needed to load the optional IR passed to invokeParser
  SMDiagnostic Err;
  std::unique_ptr<Module> M;
  FunctionType *ScalarFTy = nullptr;

protected:
  // References to the parser output field.
  ElementCount &VF = Info.Shape.VF;
  VFISAKind &ISA = Info.ISA;
  /// Parameters for the vectorized function
  SmallVector<VFParameter, 8> &Parameters = Info.Shape.Parameters;
  std::string &ScalarName = Info.ScalarName;
  std::string &VectorName = Info.VectorName;

  /// Invoke the parser. Every time this method is invoked the state of the test
  /// is reset.
  ///
  /// \p MangledName string the parser has to demangle.
  ///
  /// \p ScalarFTyStr FunctionType string to get the signature of the scalar
  /// function, which is used by `tryDemangleForVFABI` to check for the number
  /// of arguments on scalable vectors, and by `matchParameters` to perform some
  /// additional checking in the tests in this file.
  bool invokeParser(const StringRef MangledName,
                    const StringRef ScalarFTyStr = "void()") {
    // Reset the VFInfo to be able to call `invokeParser` multiple times in
    // the same test.
    reset(ScalarFTyStr);

    const auto OptInfo = VFABI::tryDemangleForVFABI(MangledName, ScalarFTy);
    if (OptInfo)
      Info = *OptInfo;

    return OptInfo.has_value();
  }

  /// Returns whether the parsed function contains a mask.
  bool isMasked() const { return Info.isMasked(); }

  FunctionType *getFunctionType() {
    return VFABI::createFunctionType(Info, ScalarFTy);
  }
};
} // unnamed namespace

// Function Types commonly used in tests
FunctionType *FTyMaskVLen2_i32 = FunctionType::get(
    Type::getVoidTy(Ctx),
    {
        VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getFixed(2)),
        VectorType::get(Type::getInt1Ty(Ctx), ElementCount::getFixed(2)),
    },
    false);

FunctionType *FTyNoMaskVLen2_i32 = FunctionType::get(
    Type::getVoidTy(Ctx),
    {
        VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getFixed(2)),
    },
    false);

FunctionType *FTyMaskedVLA_i32 = FunctionType::get(
    Type::getVoidTy(Ctx),
    {
        VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getScalable(4)),
        VectorType::get(Type::getInt1Ty(Ctx), ElementCount::getScalable(4)),
    },
    false);

// This test makes sure that the demangling method succeeds only on
// valid values of the string.
TEST_F(VFABIParserTest, OnlyValidNames) {
  // Incomplete string.
  EXPECT_FALSE(invokeParser(""));
  EXPECT_FALSE(invokeParser("_ZGV"));
  EXPECT_FALSE(invokeParser("_ZGVn"));
  EXPECT_FALSE(invokeParser("_ZGVnN"));
  EXPECT_FALSE(invokeParser("_ZGVnN2"));
  EXPECT_FALSE(invokeParser("_ZGVnN2v"));
  EXPECT_FALSE(invokeParser("_ZGVnN2v_"));
  // Missing parameters.
  EXPECT_FALSE(invokeParser("_ZGVnN2_foo"));
  // Missing _ZGV prefix.
  EXPECT_FALSE(invokeParser("_ZVnN2v_foo"));
  // Missing <isa>.
  EXPECT_FALSE(invokeParser("_ZGVN2v_foo"));
  // Missing <mask>.
  EXPECT_FALSE(invokeParser("_ZGVn2v_foo"));
  // Missing <vlen>.
  EXPECT_FALSE(invokeParser("_ZGVnNv_foo"));
  // Missing <scalarname>.
  EXPECT_FALSE(invokeParser("_ZGVnN2v_"));
  // Missing _ separator.
  EXPECT_FALSE(invokeParser("_ZGVnN2vfoo"));
  // Missing <vectorname>.
  EXPECT_FALSE(invokeParser("_ZGVnN2v_foo()"));
  // Unterminated name.
  EXPECT_FALSE(invokeParser("_ZGVnN2v_foo(bar"));
}

TEST_F(VFABIParserTest, ParamListParsing) {
  EXPECT_TRUE(
      invokeParser("_ZGVnN2vl16Ls32R3l_foo", "void(i32, i32, i32, ptr, i32)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(false, isMasked());
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx),
      {VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getFixed(2)),
       Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx),
       Type::getInt32Ty(Ctx)->getPointerTo(), Type::getInt32Ty(Ctx)},
      false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(Parameters.size(), (unsigned)5);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector, 0}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_Linear, 16}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_LinearValPos, 32}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::OMP_LinearRef, 3}));
  EXPECT_EQ(Parameters[4], VFParameter({4, VFParamKind::OMP_Linear, 1}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "_ZGVnN2vl16Ls32R3l_foo");
}

TEST_F(VFABIParserTest, ScalarNameAndVectorName_01) {
  EXPECT_TRUE(invokeParser("_ZGVnM2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(true, isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ScalarNameAndVectorName_02) {
  EXPECT_TRUE(invokeParser("_ZGVnM2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(true, isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ScalarNameAndVectorName_03) {
  EXPECT_TRUE(
      invokeParser("_ZGVnM2v___foo_bar_abc(fooBarAbcVec)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(true, isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(ScalarName, "__foo_bar_abc");
  EXPECT_EQ(VectorName, "fooBarAbcVec");
}

TEST_F(VFABIParserTest, ScalarNameOnly) {
  EXPECT_TRUE(invokeParser("_ZGVnM2v___foo_bar_abc", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(true, isMasked());
  EXPECT_EQ(ScalarName, "__foo_bar_abc");
  // no vector name specified (as it's optional), so it should have the entire
  // mangled name.
  EXPECT_EQ(VectorName, "_ZGVnM2v___foo_bar_abc");
}

TEST_F(VFABIParserTest, Parse) {
  EXPECT_TRUE(
      invokeParser("_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000_foo",
                   "void(i32, i32, i32, i32, ptr, i32, i32, i32, ptr)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_FALSE(isMasked());
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx),
      {
          VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getFixed(2)),
          Type::getInt32Ty(Ctx),
          Type::getInt32Ty(Ctx),
          Type::getInt32Ty(Ctx),
          Type::getInt32Ty(Ctx)->getPointerTo(),
          Type::getInt32Ty(Ctx),
          Type::getInt32Ty(Ctx),
          Type::getInt32Ty(Ctx),
          Type::getInt32Ty(Ctx)->getPointerTo(),
      },
      false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)9);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector, 0}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_LinearPos, 2}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_LinearValPos, 27}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::OMP_LinearUValPos, 4}));
  EXPECT_EQ(Parameters[4], VFParameter({4, VFParamKind::OMP_LinearRefPos, 5}));
  EXPECT_EQ(Parameters[5], VFParameter({5, VFParamKind::OMP_Linear, 1}));
  EXPECT_EQ(Parameters[6], VFParameter({6, VFParamKind::OMP_LinearVal, 10}));
  EXPECT_EQ(Parameters[7], VFParameter({7, VFParamKind::OMP_LinearUVal, 100}));
  EXPECT_EQ(Parameters[8], VFParameter({8, VFParamKind::OMP_LinearRef, 1000}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000_foo");
}

TEST_F(VFABIParserTest, ParseVectorName) {
  EXPECT_TRUE(invokeParser("_ZGVnN2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_FALSE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyNoMaskVLen2_i32);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector, 0}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, LinearWithCompileTimeNegativeStep) {
  EXPECT_TRUE(invokeParser("_ZGVnN2ln1Ln10Un100Rn1000_foo(vector_foo)",
                           "void(i32, i32, i32, ptr)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_FALSE(isMasked());
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx),
      {Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx),
       Type::getInt32Ty(Ctx)->getPointerTo()},
      false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)4);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::OMP_Linear, -1}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_LinearVal, -10}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_LinearUVal, -100}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::OMP_LinearRef, -1000}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseScalableSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsMxv_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskedVLA_i32);
  EXPECT_EQ(VF, ElementCount::getScalable(4));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseFixedWidthSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsM2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, NotAVectorFunctionABIName) {
  // Vector names should start with `_ZGV`.
  EXPECT_FALSE(invokeParser("ZGVnN2v_foo"));
}

TEST_F(VFABIParserTest, LinearWithRuntimeStep) {
  EXPECT_FALSE(invokeParser("_ZGVnN2ls_foo"))
      << "A number should be present after \"ls\".";
  EXPECT_TRUE(invokeParser("_ZGVnN2ls2_foo", "void(i32)"));
  EXPECT_FALSE(invokeParser("_ZGVnN2Rs_foo"))
      << "A number should be present after \"Rs\".";
  EXPECT_TRUE(invokeParser("_ZGVnN2Rs4_foo", "void(i32)"));
  EXPECT_FALSE(invokeParser("_ZGVnN2Ls_foo"))
      << "A number should be present after \"Ls\".";
  EXPECT_TRUE(invokeParser("_ZGVnN2Ls6_foo", "void(i32)"));
  EXPECT_FALSE(invokeParser("_ZGVnN2Us_foo"))
      << "A number should be present after \"Us\".";
  EXPECT_TRUE(invokeParser("_ZGVnN2Us8_foo", "void(i32)"));
}

TEST_F(VFABIParserTest, LinearWithoutCompileTime) {
  EXPECT_TRUE(invokeParser("_ZGVnN3lLRUlnLnRnUn_foo(vector_foo)",
                           "void(i32, i32, ptr, i32, i32, i32, ptr, i32)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_FALSE(isMasked());
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx),
      {Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx),
       Type::getInt32Ty(Ctx)->getPointerTo(), Type::getInt32Ty(Ctx),
       Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx),
       Type::getInt32Ty(Ctx)->getPointerTo(), Type::getInt32Ty(Ctx)},
      false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(Parameters.size(), (unsigned)8);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::OMP_Linear, 1}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_LinearVal, 1}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_LinearRef, 1}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::OMP_LinearUVal, 1}));
  EXPECT_EQ(Parameters[4], VFParameter({4, VFParamKind::OMP_Linear, -1}));
  EXPECT_EQ(Parameters[5], VFParameter({5, VFParamKind::OMP_LinearVal, -1}));
  EXPECT_EQ(Parameters[6], VFParameter({6, VFParamKind::OMP_LinearRef, -1}));
  EXPECT_EQ(Parameters[7], VFParameter({7, VFParamKind::OMP_LinearUVal, -1}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, LLVM_ISA) {
  EXPECT_FALSE(invokeParser("_ZGV_LLVM_N2v_foo"));
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_N2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_FALSE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyNoMaskVLen2_i32);
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, InvalidMask) {
  EXPECT_FALSE(invokeParser("_ZGVsK2v_foo"));
}

TEST_F(VFABIParserTest, InvalidParameter) {
  EXPECT_FALSE(invokeParser("_ZGVsM2vX_foo"));
}

TEST_F(VFABIParserTest, Align) {
  EXPECT_TRUE(invokeParser("_ZGVsN2l2a2_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_FALSE(isMasked());
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0].Alignment, Align(2));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx)}, false);
  EXPECT_EQ(getFunctionType(), FTy);
  // Missing alignment value.
  EXPECT_FALSE(invokeParser("_ZGVsM2l2a_foo"));
  // Invalid alignment token "x".
  EXPECT_FALSE(invokeParser("_ZGVsM2l2ax_foo"));
  // Alignment MUST be associated to a paramater.
  EXPECT_FALSE(invokeParser("_ZGVsM2a2_foo"));
  // Alignment must be a power of 2.
  EXPECT_FALSE(invokeParser("_ZGVsN2l2a0_foo"));
  EXPECT_TRUE(invokeParser("_ZGVsN2l2a1_foo", "void(i32)"));
  EXPECT_FALSE(invokeParser("_ZGVsN2l2a3_foo"));
  EXPECT_FALSE(invokeParser("_ZGVsN2l2a6_foo"));
}

TEST_F(VFABIParserTest, ParseUniform) {
  EXPECT_TRUE(invokeParser("_ZGVnN2u_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_FALSE(isMasked());
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx)}, false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::OMP_Uniform, 0}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");

  // Uniform doesn't expect extra data.
  EXPECT_FALSE(invokeParser("_ZGVnN2u0_foo"));
}

TEST_F(VFABIParserTest, ISAIndependentMangling) {
  // This test makes sure that the mangling of the parameters in
  // independent on the <isa> token.
  const StringRef IRTy =
      "void(i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32)";
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx),
      {VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getFixed(2)),
       Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx),
       Type::getInt32Ty(Ctx)->getPointerTo(), Type::getInt32Ty(Ctx),
       Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx),
       Type::getInt32Ty(Ctx)},
      false);

  const SmallVector<VFParameter, 8> ExpectedParams = {
      VFParameter({0, VFParamKind::Vector, 0}),
      VFParameter({1, VFParamKind::OMP_LinearPos, 2}),
      VFParameter({2, VFParamKind::OMP_LinearValPos, 27}),
      VFParameter({3, VFParamKind::OMP_LinearUValPos, 4}),
      VFParameter({4, VFParamKind::OMP_LinearRefPos, 5}),
      VFParameter({5, VFParamKind::OMP_Linear, 1}),
      VFParameter({6, VFParamKind::OMP_LinearVal, 10}),
      VFParameter({7, VFParamKind::OMP_LinearUVal, 100}),
      VFParameter({8, VFParamKind::OMP_LinearRef, 1000}),
      VFParameter({9, VFParamKind::OMP_Uniform, 0}),
  };

#define __COMMON_CHECKS                                                        \
  do {                                                                         \
    EXPECT_EQ(VF, ElementCount::getFixed(2));                                  \
    EXPECT_FALSE(isMasked());                                                  \
    EXPECT_EQ(getFunctionType(), FTy);                                         \
    EXPECT_EQ(Parameters.size(), (unsigned)10);                                \
    EXPECT_EQ(Parameters, ExpectedParams);                                     \
    EXPECT_EQ(ScalarName, "foo");                                              \
    EXPECT_EQ(VectorName, "vector_foo");                                       \
  } while (0)

  // Advanced SIMD: <isa> = "n"
  EXPECT_TRUE(invokeParser(
      "_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000u_foo(vector_foo)", IRTy));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  __COMMON_CHECKS;

  // SVE: <isa> = "s"
  EXPECT_TRUE(invokeParser(
      "_ZGVsN2vls2Ls27Us4Rs5l1L10U100R1000u_foo(vector_foo)", IRTy));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  __COMMON_CHECKS;

  // SSE: <isa> = "b"
  EXPECT_TRUE(invokeParser(
      "_ZGVbN2vls2Ls27Us4Rs5l1L10U100R1000u_foo(vector_foo)", IRTy));
  EXPECT_EQ(ISA, VFISAKind::SSE);
  __COMMON_CHECKS;

  // AVX: <isa> = "c"
  EXPECT_TRUE(invokeParser(
      "_ZGVcN2vls2Ls27Us4Rs5l1L10U100R1000u_foo(vector_foo)", IRTy));
  EXPECT_EQ(ISA, VFISAKind::AVX);
  __COMMON_CHECKS;

  // AVX2: <isa> = "d"
  EXPECT_TRUE(invokeParser(
      "_ZGVdN2vls2Ls27Us4Rs5l1L10U100R1000u_foo(vector_foo)", IRTy));
  EXPECT_EQ(ISA, VFISAKind::AVX2);
  __COMMON_CHECKS;

  // AVX512: <isa> = "e"
  EXPECT_TRUE(invokeParser(
      "_ZGVeN2vls2Ls27Us4Rs5l1L10U100R1000u_foo(vector_foo)", IRTy));
  EXPECT_EQ(ISA, VFISAKind::AVX512);
  __COMMON_CHECKS;

  // LLVM: <isa> = "_LLVM_" internal vector function.
  EXPECT_TRUE(invokeParser(
      "_ZGV_LLVM_N2vls2Ls27Us4Rs5l1L10U100R1000u_foo(vector_foo)", IRTy));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  __COMMON_CHECKS;

  // Unknown ISA (randomly using "q"). This test will need update if
  // some targets decide to use "q" as their ISA token.
  EXPECT_TRUE(invokeParser(
      "_ZGVqN2vls2Ls27Us4Rs5l1L10U100R1000u_foo(vector_foo)", IRTy));
  EXPECT_EQ(ISA, VFISAKind::Unknown);
  __COMMON_CHECKS;

#undef __COMMON_CHECKS
}

TEST_F(VFABIParserTest, MissingScalarName) {
  EXPECT_FALSE(invokeParser("_ZGVnN2v_"));
}

TEST_F(VFABIParserTest, MissingVectorName) {
  EXPECT_FALSE(invokeParser("_ZGVnN2v_foo()"));
}

TEST_F(VFABIParserTest, MissingVectorNameTermination) {
  EXPECT_FALSE(invokeParser("_ZGVnN2v_foo(bar"));
}

TEST_F(VFABIParserTest, ParseMaskingNEON) {
  EXPECT_TRUE(invokeParser("_ZGVnM2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseMaskingSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsM2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseMaskingSSE) {
  EXPECT_TRUE(invokeParser("_ZGVbM2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::SSE);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseMaskingAVX) {
  EXPECT_TRUE(invokeParser("_ZGVcM2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AVX);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseMaskingAVX2) {
  EXPECT_TRUE(invokeParser("_ZGVdM2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AVX2);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseMaskingAVX512) {
  EXPECT_TRUE(invokeParser("_ZGVeM2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::AVX512);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseMaskingLLVM) {
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_M2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskVLen2_i32);
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseScalableMaskingLLVM) {
  EXPECT_FALSE(invokeParser("_ZGV_LLVM_Mxv_foo(vector_foo)"));
}

TEST_F(VFABIParserTest, LLVM_InternalISA) {
  EXPECT_FALSE(invokeParser("_ZGV_LLVM_N2v_foo"));
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_N2v_foo(vector_foo)", "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_FALSE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyNoMaskVLen2_i32);
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, LLVM_Intrinsics) {
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_N4vv_llvm.pow.f32(__svml_powf4)",
                           "void(float, float)"));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_FALSE(isMasked());
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx),
      {
          VectorType::get(Type::getFloatTy(Ctx), ElementCount::getFixed(4)),
          VectorType::get(Type::getFloatTy(Ctx), ElementCount::getFixed(4)),
      },
      false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(VF, ElementCount::getFixed(4));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::Vector}));
  EXPECT_EQ(ScalarName, "llvm.pow.f32");
  EXPECT_EQ(VectorName, "__svml_powf4");
}

TEST_F(VFABIParserTest, ParseScalableRequiresDeclaration) {
  const char *MangledName = "_ZGVsMxv_sin(custom_vg)";
  EXPECT_FALSE(invokeParser(MangledName));
  EXPECT_TRUE(invokeParser(MangledName, "void(i32)"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_TRUE(isMasked());
  EXPECT_EQ(getFunctionType(), FTyMaskedVLA_i32);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "custom_vg");
}

TEST_F(VFABIParserTest, ZeroIsInvalidVLEN) {
  EXPECT_FALSE(invokeParser("_ZGVeM0v_foo"));
  EXPECT_FALSE(invokeParser("_ZGVeN0v_foo"));
  EXPECT_FALSE(invokeParser("_ZGVsM0v_foo"));
  EXPECT_FALSE(invokeParser("_ZGVsN0v_foo"));
}

TEST_F(VFABIParserTest, ParseScalableMaskingSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsMxv_foo(vector_foo)", "i32(i32)"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_TRUE(isMasked());
  FunctionType *FTy = FunctionType::get(
      VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getScalable(4)),
      {VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getScalable(4)),
       VectorType::get(Type::getInt1Ty(Ctx), ElementCount::getScalable(4))},
      false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(VF, ElementCount::getScalable(4));
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

TEST_F(VFABIParserTest, ParseScalableMaskingSVESincos) {
  EXPECT_TRUE(invokeParser("_ZGVsMxvl8l8_sincos(custom_vector_sincos)",
                           "void(double, ptr, ptr)"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_TRUE(isMasked());
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx),
      {
          VectorType::get(Type::getDoubleTy(Ctx), ElementCount::getScalable(2)),
          Type::getInt32Ty(Ctx)->getPointerTo(),
          Type::getInt32Ty(Ctx)->getPointerTo(),
          VectorType::get(Type::getInt1Ty(Ctx), ElementCount::getScalable(2)),
      },
      false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(VF, ElementCount::getScalable(2));
  EXPECT_EQ(Parameters.size(), (unsigned)4);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_Linear, 8}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_Linear, 8}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sincos");
  EXPECT_EQ(VectorName, "custom_vector_sincos");
}

// Make sure that we get the correct VF if the return type is wider than any
// parameter type.
TEST_F(VFABIParserTest, ParseWiderReturnTypeSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsMxvv_foo(vector_foo)", "i64(i32, i32)"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_TRUE(isMasked());
  FunctionType *FTy = FunctionType::get(
      VectorType::get(Type::getInt64Ty(Ctx), ElementCount::getScalable(2)),
      {
          VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getScalable(2)),
          VectorType::get(Type::getInt32Ty(Ctx), ElementCount::getScalable(2)),
          VectorType::get(Type::getInt1Ty(Ctx), ElementCount::getScalable(2)),
      },
      false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(Parameters.size(), (unsigned)3);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(VF, ElementCount::getScalable(2));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

// Make sure we handle void return types.
TEST_F(VFABIParserTest, ParseVoidReturnTypeSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsMxv_foo(vector_foo)", "void(i16)"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_TRUE(isMasked());
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx),
      {
          VectorType::get(Type::getInt16Ty(Ctx), ElementCount::getScalable(8)),
          VectorType::get(Type::getInt1Ty(Ctx), ElementCount::getScalable(8)),
      },
      false);
  EXPECT_EQ(getFunctionType(), FTy);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(VF, ElementCount::getScalable(8));
  EXPECT_EQ(ScalarName, "foo");
  EXPECT_EQ(VectorName, "vector_foo");
}

// Make sure we reject unsupported parameter types.
TEST_F(VFABIParserTest, ParseUnsupportedElementTypeSVE) {
  EXPECT_FALSE(invokeParser("_ZGVsMxv_foo(vector_foo)", "void(i128)"));
}

// Make sure we reject unsupported return types
TEST_F(VFABIParserTest, ParseUnsupportedReturnTypeSVE) {
  EXPECT_FALSE(invokeParser("_ZGVsMxv_foo(vector_foo)", "fp128(float)"));
}

class VFABIAttrTest : public testing::Test {
protected:
  void SetUp() override {
    M = parseAssemblyString(IR, Err, Ctx);
    // Get the only call instruction in the block, which is the first
    // instruction.
    CI = dyn_cast<CallInst>(&*(instructions(M->getFunction("f")).begin()));
  }
  const char *IR = "define i32 @f(i32 %a) {\n"
                   " %1 = call i32 @g(i32 %a) #0\n"
                   "  ret i32 %1\n"
                   "}\n"
                   "declare i32 @g(i32)\n"
                   "declare <2 x i32> @custom_vg(<2 x i32>)"
                   "declare <4 x i32> @_ZGVnN4v_g(<4 x i32>)"
                   "declare <8 x i32> @_ZGVnN8v_g(<8 x i32>)"
                   "attributes #0 = { "
                   "\"vector-function-abi-variant\"=\""
                   "_ZGVnN2v_g(custom_vg),_ZGVnN4v_g\" }";
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M;
  CallInst *CI;
  SmallVector<std::string, 8> Mappings;
};

TEST_F(VFABIAttrTest, Read) {
  VFABI::getVectorVariantNames(*CI, Mappings);
  SmallVector<std::string, 8> Exp;
  Exp.push_back("_ZGVnN2v_g(custom_vg)");
  Exp.push_back("_ZGVnN4v_g");
  EXPECT_EQ(Mappings, Exp);
}

TEST_F(VFABIAttrTest, Write) {
  Mappings.push_back("_ZGVnN8v_g");
  Mappings.push_back("_ZGVnN2v_g(custom_vg)");
  VFABI::setVectorVariantNames(CI, Mappings);
  const StringRef S =
      CI->getFnAttr("vector-function-abi-variant").getValueAsString();
  EXPECT_EQ(S, "_ZGVnN8v_g,_ZGVnN2v_g(custom_vg)");
}

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("VectorFunctionABITests", errs());
  return Mod;
}

TEST(VFABIGetMappingsTest, IndirectCallInst) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @call(void () * %f) {
entry:
  call void %f()
  ret void
}
)IR");
  auto *F = dyn_cast_or_null<Function>(M->getNamedValue("call"));
  ASSERT_TRUE(F);
  auto *CI = dyn_cast<CallInst>(&F->front().front());
  ASSERT_TRUE(CI);
  ASSERT_TRUE(CI->isIndirectCall());
  auto Mappings = VFDatabase::getMappings(*CI);
  EXPECT_EQ(Mappings.size(), (unsigned)0);
}
