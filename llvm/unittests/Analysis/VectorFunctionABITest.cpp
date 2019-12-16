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

// This test makes sure that the getFromVFABI method succeeds only on
// valid values of the string.
TEST(VectorFunctionABITests, OnlyValidNames) {
  // Incomplete string.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("").hasValue());
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGV").hasValue());
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVn").hasValue());
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnN").hasValue());
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnN2").hasValue());
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnN2v").hasValue());
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnN2v_").hasValue());
  // Missing parameters.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnN2_foo").hasValue());
  // Missing _ZGV prefix.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZVnN2v_foo").hasValue());
  // Missing <isa>.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVN2v_foo").hasValue());
  // Missing <mask>.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVn2v_foo").hasValue());
  // Missing <vlen>.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnNv_foo").hasValue());
  // Missing <scalarname>.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnN2v_").hasValue());
  // Missing _ separator.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnN2vfoo").hasValue());
  // Missing <vectorname>.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnN2v_foo()").hasValue());
  // Unterminated name.
  EXPECT_FALSE(VFABI::tryDemangleForVFABI("_ZGVnN2v_foo(bar").hasValue());
}

TEST(VectorFunctionABITests, ParamListParsing) {
  // Testing "vl16Ls32R3l"
  const auto OptVFS = VFABI::tryDemangleForVFABI("_ZGVnN2vl16Ls32R3l_foo");
  EXPECT_TRUE(OptVFS.hasValue());
  const VFInfo VFS = OptVFS.getValue();
  EXPECT_EQ(VFS.Shape.Parameters.size(), (unsigned)5);
  EXPECT_EQ(VFS.Shape.Parameters[0], VFParameter({0, VFParamKind::Vector, 0}));
  EXPECT_EQ(VFS.Shape.Parameters[1],
            VFParameter({1, VFParamKind::OMP_Linear, 16}));
  EXPECT_EQ(VFS.Shape.Parameters[2],
            VFParameter({2, VFParamKind::OMP_LinearValPos, 32}));
  EXPECT_EQ(VFS.Shape.Parameters[3],
            VFParameter({3, VFParamKind::OMP_LinearRef, 3}));
  EXPECT_EQ(VFS.Shape.Parameters[4],
            VFParameter({4, VFParamKind::OMP_Linear, 1}));
}

TEST(VectorFunctionABITests, ScalarNameAndVectorName) {
  // Parse Scalar Name
  const auto A = VFABI::tryDemangleForVFABI("_ZGVnM2v_sin");
  const auto B = VFABI::tryDemangleForVFABI("_ZGVnM2v_sin(UserFunc)");
  const auto C = VFABI::tryDemangleForVFABI("_ZGVnM2v___sin_sin_sin");
  EXPECT_TRUE(A.hasValue());
  EXPECT_TRUE(B.hasValue());
  EXPECT_TRUE(C.hasValue());
  EXPECT_EQ(A.getValue().ScalarName, "sin");
  EXPECT_EQ(B.getValue().ScalarName, "sin");
  EXPECT_EQ(C.getValue().ScalarName, "__sin_sin_sin");
  EXPECT_EQ(A.getValue().VectorName, "_ZGVnM2v_sin");
  EXPECT_EQ(B.getValue().VectorName, "UserFunc");
  EXPECT_EQ(C.getValue().VectorName, "_ZGVnM2v___sin_sin_sin");
}

namespace {
// Test fixture needed that holds the veariables needed by the parser.
class VFABIParserTest : public ::testing::Test {
private:
  // Parser output.
  VFInfo Info;
  // Reset the parser output references.
  void reset() { Info = VFInfo(); }

protected:
  // Referencies to the parser output field.
  unsigned &VF = Info.Shape.VF;
  VFISAKind &ISA = Info.ISA;
  SmallVector<VFParameter, 8> &Parameters = Info.Shape.Parameters;
  StringRef &ScalarName = Info.ScalarName;
  StringRef &VectorName = Info.VectorName;
  bool &IsScalable = Info.Shape.IsScalable;
  // Invoke the parser.
  bool invokeParser(const StringRef MangledName) {
    reset();
    const auto OptInfo = VFABI::tryDemangleForVFABI(MangledName);
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

TEST_F(VFABIParserTest, Parse) {
  EXPECT_TRUE(invokeParser("_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000_sin"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_FALSE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_FALSE(IsScalable);
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
  EXPECT_TRUE(invokeParser("_ZGVnN2v_sin(my_v_sin)"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_FALSE(IsMasked());
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector, 0}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "my_v_sin");
}

TEST_F(VFABIParserTest, LinearWithCompileTimeNegativeStep) {
  EXPECT_TRUE(invokeParser("_ZGVnN2ln1Ln10Un100Rn1000_sin"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_FALSE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(Parameters.size(), (unsigned)4);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::OMP_Linear, -1}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::OMP_LinearVal, -10}));
  EXPECT_EQ(Parameters[2], VFParameter({2, VFParamKind::OMP_LinearUVal, -100}));
  EXPECT_EQ(Parameters[3], VFParameter({3, VFParamKind::OMP_LinearRef, -1000}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "_ZGVnN2ln1Ln10Un100Rn1000_sin");
}

TEST_F(VFABIParserTest, ParseScalableSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsMxv_sin"));
  EXPECT_EQ(VF, (unsigned)0);
  EXPECT_TRUE(IsMasked());
  EXPECT_TRUE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "_ZGVsMxv_sin");
}

TEST_F(VFABIParserTest, ParseFixedWidthSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsM2v_sin"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_TRUE(IsMasked());
  EXPECT_FALSE(IsScalable);
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

  // Missing alignement value.
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
  EXPECT_TRUE(invokeParser("_ZGVnN2u0_sin"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_FALSE(IsMasked());
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(Parameters.size(), (unsigned)1);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::OMP_Uniform, 0}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "_ZGVnN2u0_sin");

  EXPECT_FALSE(invokeParser("_ZGVnN2u_sin"));
  EXPECT_FALSE(invokeParser("_ZGVnN2ul_sin"));
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
      VFParameter({9, VFParamKind::OMP_Uniform, 2}),
  };

#define __COMMON_CHECKS                                                        \
  do {                                                                         \
    EXPECT_EQ(VF, (unsigned)2);                                                \
    EXPECT_FALSE(IsMasked());                                                  \
    EXPECT_FALSE(IsScalable);                                                  \
    EXPECT_EQ(Parameters.size(), (unsigned)10);                                \
    EXPECT_EQ(Parameters, ExpectedParams);                                     \
    EXPECT_EQ(ScalarName, "sin");                                              \
  } while (0)

  // Advanced SIMD: <isa> = "n"
  EXPECT_TRUE(invokeParser("_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin"));
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVnN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin");

  // SVE: <isa> = "s"
  EXPECT_TRUE(invokeParser("_ZGVsN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin"));
  EXPECT_EQ(ISA, VFISAKind::SVE);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVsN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin");

  // SSE: <isa> = "b"
  EXPECT_TRUE(invokeParser("_ZGVbN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin"));
  EXPECT_EQ(ISA, VFISAKind::SSE);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVbN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin");

  // AVX: <isa> = "c"
  EXPECT_TRUE(invokeParser("_ZGVcN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin"));
  EXPECT_EQ(ISA, VFISAKind::AVX);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVcN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin");

  // AVX2: <isa> = "d"
  EXPECT_TRUE(invokeParser("_ZGVdN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin"));
  EXPECT_EQ(ISA, VFISAKind::AVX2);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVdN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin");

  // AVX512: <isa> = "e"
  EXPECT_TRUE(invokeParser("_ZGVeN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin"));
  EXPECT_EQ(ISA, VFISAKind::AVX512);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVeN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin");

  // LLVM: <isa> = "_LLVM_" internal vector function.
  EXPECT_TRUE(
      invokeParser("_ZGV_LLVM_N2vls2Ls27Us4Rs5l1L10U100R1000u2_sin(vectorf)"));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "vectorf");

  // Unknown ISA (randomly using "q"). This test will need update if
  // some targets decide to use "q" as their ISA token.
  EXPECT_TRUE(invokeParser("_ZGVqN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin"));
  EXPECT_EQ(ISA, VFISAKind::Unknown);
  __COMMON_CHECKS;
  EXPECT_EQ(VectorName, "_ZGVqN2vls2Ls27Us4Rs5l1L10U100R1000u2_sin");

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
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_TRUE(IsMasked());
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::AdvancedSIMD);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingSVE) {
  EXPECT_TRUE(invokeParser("_ZGVsM2v_sin"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_TRUE(IsMasked());
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::SVE);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingSSE) {
  EXPECT_TRUE(invokeParser("_ZGVbM2v_sin"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_TRUE(IsMasked());
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::SSE);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingAVX) {
  EXPECT_TRUE(invokeParser("_ZGVcM2v_sin"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_TRUE(IsMasked());
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::AVX);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingAVX2) {
  EXPECT_TRUE(invokeParser("_ZGVdM2v_sin"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_TRUE(IsMasked());
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::AVX2);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingAVX512) {
  EXPECT_TRUE(invokeParser("_ZGVeM2v_sin"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_TRUE(IsMasked());
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::AVX512);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
}

TEST_F(VFABIParserTest, ParseMaskingLLVM) {
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_M2v_sin(custom_vector_sin)"));
  EXPECT_EQ(VF, (unsigned)2);
  EXPECT_TRUE(IsMasked());
  EXPECT_FALSE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "custom_vector_sin");
}

TEST_F(VFABIParserTest, ParseScalableMaskingLLVM) {
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_Mxv_sin(custom_vector_sin)"));
  EXPECT_TRUE(IsMasked());
  EXPECT_TRUE(IsScalable);
  EXPECT_EQ(ISA, VFISAKind::LLVM);
  EXPECT_EQ(Parameters.size(), (unsigned)2);
  EXPECT_EQ(Parameters[0], VFParameter({0, VFParamKind::Vector}));
  EXPECT_EQ(Parameters[1], VFParameter({1, VFParamKind::GlobalPredicate}));
  EXPECT_EQ(ScalarName, "sin");
  EXPECT_EQ(VectorName, "custom_vector_sin");
}

TEST_F(VFABIParserTest, ParseScalableMaskingLLVMSincos) {
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_Mxvl8l8_sincos(custom_vector_sincos)"));
  EXPECT_TRUE(IsMasked());
  EXPECT_TRUE(IsScalable);
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
  EXPECT_TRUE(invokeParser("_ZGV_LLVM_N2v_sin_(vector_name)"));
  EXPECT_EQ(ISA, VFISAKind::LLVM);
}
