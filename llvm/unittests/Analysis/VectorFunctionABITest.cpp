//===------- VectorFunctionABITest.cpp - VFABI Unittests  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/VectorUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/InstIterator.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
// Test fixture needed that holds the veariables needed by the parser.
class VFABIParserTest : public ::testing::Test {
private:
  // Parser output.
  VFInfo Info;
  // Reset the data needed for the test.
  void reset(const StringRef Name, const StringRef IRType) {
    M = parseAssemblyString("declare void @dummy()", Err, Ctx);
    EXPECT_NE(M.get(), nullptr) << "Loading an invalid module.\n "
                                << Err.getMessage() << "\n";
    Type *Ty = parseType(IRType, Err, *(M.get()));
    FunctionType *FTy = dyn_cast<FunctionType>(Ty);
    EXPECT_NE(FTy, nullptr) << "Invalid function type string: " << IRType
                            << "\n"
                            << Err.getMessage() << "\n";
    FunctionCallee F = M->getOrInsertFunction(Name, FTy);
    EXPECT_NE(F.getCallee(), nullptr)
        << "The function must be present in the module\n";
    // Reset the VFInfo
    Info = VFInfo();
  }

  // Data needed to load the optional IR passed to invokeParser
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M;
  //  CallInst *CI;
protected:
  // Referencies to the parser output field.
  ElementCount &VF = Info.Shape.VF;
  VFISAKind &ISA = Info.ISA;
  SmallVector<VFParameter, 8> &Parameters = Info.Shape.Parameters;
  std::string &ScalarName = Info.ScalarName;
  std::string &VectorName = Info.VectorName;
  // Invoke the parser. We need to make sure that a function exist in
  // the module because the parser fails if such function don't
  // exists. Every time this method is invoked the state of the test
  // is reset.
  //
  // \p MangledName -> the string the parser has to demangle.
  //
  // \p VectorName -> optional vector name that the method needs to
  // use to create the function in the module if it differs from the
  // standard mangled name.
  //
  // \p IRType -> FunctionType string to be used for the signature of
  // the vector function.  The correct signature is needed by the
  // parser only for scalable functions. For the sake of testing, the
  // generic fixed-length case can use as signature `void()`.
  //
  bool invokeParser(const StringRef MangledName,
                    const StringRef VectorName = "",
                    const StringRef IRType = "void()") {
    StringRef Name = MangledName;
    if (!VectorName.empty())
      Name = VectorName;
    // Reset the VFInfo and the Module to be able to invoke
    // `invokeParser` multiple times in the same test.
    reset(Name, IRType);

    const auto OptInfo = VFABI::tryDemangleForVFABI(MangledName, *(M.get()));
    if (OptInfo.hasValue()) {
      Info = OptInfo.getValue();
      return true;
    }

    return false;
  }

  // Checks that 1. the last Parameter in the Shape is of type
  // VFParamKind::GlobalPredicate and 2. it is the only one of such
  // type.
  bool IsMasked() const {
    const auto NGlobalPreds =
        std::count_if(Info.Shape.Parameters.begin(),
                      Info.Shape.Parameters.end(), [](const VFParameter PK) {
                        return PK.ParamKind == VFParamKind::GlobalPredicate;
                      });
    return NGlobalPreds == 1 && Info.Shape.Parameters.back().ParamKind ==
                                    VFParamKind::GlobalPredicate;
  }
};
} // unnamed namespace

// This test makes sure correct mangling occurs for given string.
TEST_F(VFABIParserTest, ManglingVectorTLINames) {
  EXPECT_EQ(
      VFABI::mangleTLIVectorName("vec", "scalar", 3, ElementCount::getFixed(4)),
      "_ZGV_LLVM_N4vvv_scalar(vec)");
  EXPECT_EQ(VFABI::mangleTLIVectorName("vec", "scalar", 3,
                                       ElementCount::getScalable(4)),
            "_ZGV_LLVM_Nxvvv_scalar(vec)");
  EXPECT_EQ(VFABI::mangleTLIVectorName("custom.call.v5", "custom.call", 1,
                                       ElementCount::getFixed(5)),
            "_ZGV_LLVM_N5v_custom.call(custom.call.v5)");
}

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
  // Missing <vectorname>. Using `fakename` because the string being
  // parsed is not a valid function name that `invokeParser` can add.
  EXPECT_FALSE(invokeParser("_ZGVnN2v_foo()", "fakename"));
  // Unterminated name. Using `fakename` because the string being
  // parsed is not a valid function name that `invokeParser` can add.
  EXPECT_FALSE(invokeParser("_ZGVnN2v_foo(bar", "fakename"));
}

TEST_F(VFABIParserTest, ParamListParsing) {
  EXPECT_TRUE(invokeParser("_ZGVnN2vl16Ls32R3l_foo"));
  EXPECT_EQ(Parameters.size(), (unsigned)5);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector, 0}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_Linear, 16}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_LinearValPos, 32}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::OMP_LinearRef, 3}));
  EXPECT_EQ(Parameters[4], VFParameter({4, VFParamKind::OMP_Linear, 1}));
}

TEST_F(VFABIParserTest, ScalarNameAndVectorName_01) {
  EXPECT_TRUE(invokeParser("_ZGVnM2v_sin"));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "_ZGVnM2v_sin");
}

TEST_F(VFABIParserTest, ScalarNameAndVectorName_02) {
  EXPECT_TRUE(invokeParser("_ZGVnM2v_sin(UserFunc)", "UserFunc"));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "UserFunc");
}

TEST_F(VFABIParserTest, ScalarNameAndVectorName_03) {
  EXPECT_TRUE(invokeParser("_ZGVnM2v___sin_sin_sin"));
  EXPECT_EQ(ScalarName, "__sin_sin_sin");
  EXPECT_EQ(VectorName, "_ZGVnM2v___sin_sin_sin");
}

TEST_F(VFABIParserTest, Parse) {
  EXPECT_TRUE(invokeParser("_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_FALSE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
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
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000_sin");
}

TEST_F(VFABIParserTest, ParseVectorName) {
  EXPECT_TRUE(invokeParser("_ZGVnN2v_sin(my_v_sin)", "my_v_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_FALSE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector, 0}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "my_v_sin");
}

TEST_F(VFABIParserTest, LinearWithCompileTimeNegativeStep) {
  EXPECT_TRUE(invokeParser("_ZGVnN2ln1Ln10Un100Rn1000_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_FALSE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(Parameters.size(), (unsigned)4);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::OMP_Linear, -1}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_LinearVal, -10}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_LinearUVal, -100}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::OMP_LinearRef, -1000}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "_ZGVnN2ln1Ln10Un100Rn1000_sin");
}

TEST_F(VFABIParserTest, ParseScalableSVE) {
  EXPECT_TRUE(invokeParser(
      "_ZGVsMxv_sin(custom_vg)", "custom_vg",
      "<vscale x 2 x i32>(<vscale x 2 x i32>, <vscale x 2 x i1>)"));
  EXPECT_EQ(VF, ElementCount::getScalable(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "custom_vg");
}

TEST_F(VFABIParserTest, ParseFixedWidthSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsM2v_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "_ZGVsM2v_sin");
}

TEST_F(VFABIParserTest, NotAVectorFunctionABIName) {
  // Vector names should start with `_ZGV`.
  EXPECT_FALSE(invokeParser("ZGVnN2v_sin"));
}

TEST_F(VFABIParserTest, LinearWithRuntimeStep) {
  EXPECT_FALSE(invokeParser("_ZGVnN2ls_sin"))
      << "A number should be present after \"ls\".";
  EXPECT_TRUE(invokeParser("_ZGVnN2ls2_sin"));
  EXPECT_FALSE(invokeParser("_ZGVnN2Rs_sin"))
      << "A number should be present after \"Rs\".";
  EXPECT_TRUE(invokeParser("_ZGVnN2Rs4_sin"));
  EXPECT_FALSE(invokeParser("_ZGVnN2Ls_sin"))
      << "A number should be present after \"Ls\".";
  EXPECT_TRUE(invokeParser("_ZGVnN2Ls6_sin"));
  EXPECT_FALSE(invokeParser("_ZGVnN2Us_sin"))
      << "A number should be present after \"Us\".";
  EXPECT_TRUE(invokeParser("_ZGVnN2Us8_sin"));
}

TEST_F(VFABIParserTest, LinearWithoutCompileTime) {
  EXPECT_TRUE(invokeParser("_ZGVnN3lLRUlnLnRnUn_sin"));
  EXPECT_EQ(Parameters.size(), (unsigned)8);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::OMP_Linear, 1}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_LinearVal, 1}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_LinearRef, 1}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::OMP_LinearUVal, 1}));
  EXPECT_EQ(Parameters[4], VFParameter({4, VFParamKind::OMP_Linear, -1}));
  EXPECT_EQ(Parameters[5], VFParameter({5, VFParamKind::OMP_LinearVal, -1}));
  EXPECT_EQ(Parameters[6], VFParameter({6, VFParamKind::OMP_LinearRef, -1}));
  EXPECT_EQ(Parameters[7], VFParameter({7, VFParamKind::OMP_LinearUVal, -1}));
}

TEST_F(VFABIParserTest, ISA) {
  EXPECT_TRUE(invokeParser("_ZGVqN2v_sin"));
  EXPECT_EQ(ISA, VFISAKind::Unknown);

  EXPECT_TRUE(invokeParser("_ZGVnN2v_sin"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);

  EXPECT_TRUE(invokeParser("_ZGVsN2v_sin"));
  EXPECT_EQ(ISA, VFISAKind::SVE);

  EXPECT_TRUE(invokeParser("_ZGVbN2v_sin"));
  EXPECT_EQ(ISA, VFISAKind::SSE);

  EXPECT_TRUE(invokeParser("_ZGVcN2v_sin"));
  EXPECT_EQ(ISA, VFISAKind::AVX);

  EXPECT_TRUE(invokeParser("_ZGVdN2v_sin"));
  EXPECT_EQ(ISA, VFISAKind::AVX2);

  EXPECT_TRUE(invokeParser("_ZGVeN2v_sin"));
  EXPECT_EQ(ISA, VFISAKind::AVX512);
}

TEST_F(VFABIParserTest, LLVM_ISA) {
  EXPECT_FALSE(invokeParser("_ZGV_LLVM_N2v_sin"));
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_N2v_sin_(vector_name)", "vector_name"));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
}

TEST_F(VFABIParserTest, InvalidMask) {
  EXPECT_FALSE(invokeParser("_ZGVsK2v_sin"));
}

TEST_F(VFABIParserTest, InvalidParameter) {
  EXPECT_FALSE(invokeParser("_ZGVsM2vX_sin"));
}

TEST_F(VFABIParserTest, Align) {
  EXPECT_TRUE(invokeParser("_ZGVsN2l2a2_sin"));
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0].Alignment, Align(2));

  // Missing alignment value.
  EXPECT_FALSE(invokeParser("_ZGVsM2l2a_sin"));
  // Invalid alignment token "x".
  EXPECT_FALSE(invokeParser("_ZGVsM2l2ax_sin"));
  // Alignment MUST be associated to a paramater.
  EXPECT_FALSE(invokeParser("_ZGVsM2a2_sin"));
  // Alignment must be a power of 2.
  EXPECT_FALSE(invokeParser("_ZGVsN2l2a0_sin"));
  EXPECT_TRUE(invokeParser("_ZGVsN2l2a1_sin"));
  EXPECT_FALSE(invokeParser("_ZGVsN2l2a3_sin"));
  EXPECT_FALSE(invokeParser("_ZGVsN2l2a6_sin"));
}

TEST_F(VFABIParserTest, ParseUniform) {
  EXPECT_TRUE(invokeParser("_ZGVnN2u_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_FALSE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::OMP_Uniform, 0}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "_ZGVnN2u_sin");

  // Uniform doesn't expect extra data.
  EXPECT_FALSE(invokeParser("_ZGVnN2u0_sin"));
}

TEST_F(VFABIParserTest, ISAIndependentMangling) {
  // This test makes sure that the mangling of the parameters in
  // independent on the <isa> token.
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
    EXPECT_FALSE(IsMasked());                                                  \
    EXPECT_EQ(Parameters.size(), (unsigned)10);                                \
    EXPECT_EQ(Parameters, ExpectedParams);                                     \
    EXPECT_EQ(ScalarName, "sin");                                              \
  } while (0)

  // Advanced SIMD: <isa> = "n"
  EXPECT_TRUE(invokeParser("_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000u_sin"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000u_sin");

  // SVE: <isa> = "s"
  EXPECT_TRUE(invokeParser("_ZGVsN2vls2Ls27Us4Rs5l1L10U100R1000u_sin"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVsN2vls2Ls27Us4Rs5l1L10U100R1000u_sin");

  // SSE: <isa> = "b"
  EXPECT_TRUE(invokeParser("_ZGVbN2vls2Ls27Us4Rs5l1L10U100R1000u_sin"));
  EXPECT_EQ(ISA, VFISAKind::SSE);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVbN2vls2Ls27Us4Rs5l1L10U100R1000u_sin");

  // AVX: <isa> = "c"
  EXPECT_TRUE(invokeParser("_ZGVcN2vls2Ls27Us4Rs5l1L10U100R1000u_sin"));
  EXPECT_EQ(ISA, VFISAKind::AVX);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVcN2vls2Ls27Us4Rs5l1L10U100R1000u_sin");

  // AVX2: <isa> = "d"
  EXPECT_TRUE(invokeParser("_ZGVdN2vls2Ls27Us4Rs5l1L10U100R1000u_sin"));
  EXPECT_EQ(ISA, VFISAKind::AVX2);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVdN2vls2Ls27Us4Rs5l1L10U100R1000u_sin");

  // AVX512: <isa> = "e"
  EXPECT_TRUE(invokeParser("_ZGVeN2vls2Ls27Us4Rs5l1L10U100R1000u_sin"));
  EXPECT_EQ(ISA, VFISAKind::AVX512);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVeN2vls2Ls27Us4Rs5l1L10U100R1000u_sin");

  // LLVM: <isa> = "_LLVM_" internal vector function.
  EXPECT_TRUE(invokeParser(
      "_ZGV_LLVM_N2vls2Ls27Us4Rs5l1L10U100R1000u_sin(vectorf)", "vectorf"));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "vectorf");

  // Unknown ISA (randomly using "q"). This test will need update if
  // some targets decide to use "q" as their ISA token.
  EXPECT_TRUE(invokeParser("_ZGVqN2vls2Ls27Us4Rs5l1L10U100R1000u_sin"));
  EXPECT_EQ(ISA, VFISAKind::Unknown);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVqN2vls2Ls27Us4Rs5l1L10U100R1000u_sin");

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
  EXPECT_TRUE(invokeParser("_ZGVnM2v_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsM2v_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingSSE) {
  EXPECT_TRUE(invokeParser("_ZGVbM2v_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::SSE);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingAVX) {
  EXPECT_TRUE(invokeParser("_ZGVcM2v_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AVX);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingAVX2) {
  EXPECT_TRUE(invokeParser("_ZGVdM2v_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AVX2);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingAVX512) {
  EXPECT_TRUE(invokeParser("_ZGVeM2v_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AVX512);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingLLVM) {
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_M2v_sin(custom_vector_sin)",
                           "custom_vector_sin"));
  EXPECT_EQ(VF, ElementCount::getFixed(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "custom_vector_sin");
}

TEST_F(VFABIParserTest, ParseScalableMaskingLLVM) {
  EXPECT_TRUE(invokeParser(
      "_ZGV_LLVM_Mxv_sin(custom_vector_sin)", "custom_vector_sin",
      "<vscale x 2 x i32> (<vscale x 2 x i32>, <vscale x 2 x i1>)"));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(VF, ElementCount::getScalable(2));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "custom_vector_sin");
}

TEST_F(VFABIParserTest, ParseScalableMaskingLLVMSincos) {
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_Mxvl8l8_sincos(custom_vector_sincos)",
                           "custom_vector_sincos",
                           "void(<vscale x 2 x double>, double *, double *)"));
  EXPECT_EQ(VF, ElementCount::getScalable(2));
  EXPECT_TRUE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_EQ(Parameters.size(), (unsigned)4);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_Linear, 8}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_Linear, 8}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sincos");
  EXPECT_EQ(VectorName, "custom_vector_sincos");
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

TEST_F(VFABIParserTest, LLVM_InternalISA) {
  EXPECT_FALSE(invokeParser("_ZGV_LLVM_N2v_sin"));
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_N2v_sin_(vector_name)", "vector_name"));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
}

TEST_F(VFABIParserTest, IntrinsicsInLLVMIsa) {
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_N4vv_llvm.pow.f32(__svml_powf4)",
                           "__svml_powf4"));
  EXPECT_EQ(VF, ElementCount::getFixed(4));
  EXPECT_FALSE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::Vector}));
  EXPECT_EQ(ScalarName, "llvm.pow.f32");
}

TEST_F(VFABIParserTest, ParseScalableRequiresDeclaration) {
  const char *MangledName = "_ZGVsMxv_sin(custom_vg)";
  // The parser succeds only when the correct function definition of
  // `custom_vg` is added to the module.
  EXPECT_FALSE(invokeParser(MangledName));
  EXPECT_TRUE(invokeParser(
      MangledName, "custom_vg",
      "<vscale x 4 x double>(<vscale x 4 x double>, <vscale x 4 x i1>)"));
}

TEST_F(VFABIParserTest, ZeroIsInvalidVLEN) {
  EXPECT_FALSE(invokeParser("_ZGVeM0v_sin"));
  EXPECT_FALSE(invokeParser("_ZGVeN0v_sin"));
  EXPECT_FALSE(invokeParser("_ZGVsM0v_sin"));
  EXPECT_FALSE(invokeParser("_ZGVsN0v_sin"));
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
  auto F = dyn_cast_or_null<Function>(M->getNamedValue("call"));
  ASSERT_TRUE(F);
  auto CI = dyn_cast<CallInst>(&F->front().front());
  ASSERT_TRUE(CI);
  ASSERT_TRUE(CI->isIndirectCall());
  auto Mappings = VFDatabase::getMappings(*CI);
  EXPECT_EQ(Mappings.size(), (unsigned)0);
}
