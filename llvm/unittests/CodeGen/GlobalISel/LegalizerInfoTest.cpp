//===- llvm/unittest/CodeGen/GlobalISel/LegalizerInfoTest.cpp -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/Target/TargetOpcodes.h"
#include "gtest/gtest.h"

using namespace llvm;

// Define a couple of pretty printers to help debugging when things go wrong.
namespace llvm {
std::ostream &
operator<<(std::ostream &OS, const llvm::LegalizerInfo::LegalizeAction Act) {
  switch (Act) {
  case LegalizerInfo::Lower: OS << "Lower"; break;
  case LegalizerInfo::Legal: OS << "Legal"; break;
  case LegalizerInfo::NarrowScalar: OS << "NarrowScalar"; break;
  case LegalizerInfo::WidenScalar:  OS << "WidenScalar"; break;
  case LegalizerInfo::FewerElements:  OS << "FewerElements"; break;
  case LegalizerInfo::MoreElements:  OS << "MoreElements"; break;
  case LegalizerInfo::Libcall: OS << "Libcall"; break;
  case LegalizerInfo::Custom: OS << "Custom"; break;
  case LegalizerInfo::Unsupported: OS << "Unsupported"; break;
  case LegalizerInfo::NotFound: OS << "NotFound";
  }
  return OS;
}

std::ostream &
operator<<(std::ostream &OS, const llvm::LLT Ty) {
  std::string Repr;
  raw_string_ostream SS{Repr};
  Ty.print(SS);
  OS << SS.str();
  return OS;
}
}

namespace {


TEST(LegalizerInfoTest, ScalarRISC) {
  using namespace TargetOpcode;
  LegalizerInfo L;
  // Typical RISCy set of operations based on AArch64.
  for (auto Op : {G_ADD, G_SUB}) {
    for (unsigned Size : {32, 64})
      L.setAction({Op, 0, LLT::scalar(Size)}, LegalizerInfo::Legal);
    L.setLegalizeScalarToDifferentSizeStrategy(
        Op, 0, LegalizerInfo::widenToLargerTypesAndNarrowToLargest);
  }

  L.computeTables();

  for (auto &opcode : {G_ADD, G_SUB}) {
    // Check we infer the correct types and actually do what we're told.
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(8)}),
              std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(32)));
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(16)}),
              std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(32)));
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(32)}),
              std::make_pair(LegalizerInfo::Legal, LLT::scalar(32)));
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(64)}),
              std::make_pair(LegalizerInfo::Legal, LLT::scalar(64)));

    // Make sure the default for over-sized types applies.
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(128)}),
              std::make_pair(LegalizerInfo::NarrowScalar, LLT::scalar(64)));
    // Make sure we also handle unusual sizes
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(1)}),
              std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(32)));
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(31)}),
              std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(32)));
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(33)}),
              std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(64)));
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(63)}),
              std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(64)));
    ASSERT_EQ(L.getAction({opcode, LLT::scalar(65)}),
              std::make_pair(LegalizerInfo::NarrowScalar, LLT::scalar(64)));
  }
}

TEST(LegalizerInfoTest, VectorRISC) {
  using namespace TargetOpcode;
  LegalizerInfo L;
  // Typical RISCy set of operations based on ARM.
  L.setAction({G_ADD, LLT::vector(8, 8)}, LegalizerInfo::Legal);
  L.setAction({G_ADD, LLT::vector(16, 8)}, LegalizerInfo::Legal);
  L.setAction({G_ADD, LLT::vector(4, 16)}, LegalizerInfo::Legal);
  L.setAction({G_ADD, LLT::vector(8, 16)}, LegalizerInfo::Legal);
  L.setAction({G_ADD, LLT::vector(2, 32)}, LegalizerInfo::Legal);
  L.setAction({G_ADD, LLT::vector(4, 32)}, LegalizerInfo::Legal);

  L.setLegalizeVectorElementToDifferentSizeStrategy(
      G_ADD, 0, LegalizerInfo::widenToLargerTypesUnsupportedOtherwise);

  L.setAction({G_ADD, 0, LLT::scalar(32)}, LegalizerInfo::Legal);

  L.computeTables();

  // Check we infer the correct types and actually do what we're told for some
  // simple cases.
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(8, 8)}),
            std::make_pair(LegalizerInfo::Legal, LLT::vector(8, 8)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(8, 7)}),
            std::make_pair(LegalizerInfo::WidenScalar, LLT::vector(8, 8)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(2, 8)}),
            std::make_pair(LegalizerInfo::MoreElements, LLT::vector(8, 8)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(8, 32)}),
            std::make_pair(LegalizerInfo::FewerElements, LLT::vector(4, 32)));
  // Check a few non-power-of-2 sizes:
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(3, 3)}),
            std::make_pair(LegalizerInfo::WidenScalar, LLT::vector(3, 8)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(3, 8)}),
            std::make_pair(LegalizerInfo::MoreElements, LLT::vector(8, 8)));
}

TEST(LegalizerInfoTest, MultipleTypes) {
  using namespace TargetOpcode;
  LegalizerInfo L;
  LLT p0 = LLT::pointer(0, 64);
  LLT s64 = LLT::scalar(64);

  // Typical RISCy set of operations based on AArch64.
  L.setAction({G_PTRTOINT, 0, s64}, LegalizerInfo::Legal);
  L.setAction({G_PTRTOINT, 1, p0}, LegalizerInfo::Legal);

  L.setLegalizeScalarToDifferentSizeStrategy(
      G_PTRTOINT, 0, LegalizerInfo::widenToLargerTypesAndNarrowToLargest);

  L.computeTables();

  // Check we infer the correct types and actually do what we're told.
  ASSERT_EQ(L.getAction({G_PTRTOINT, 0, s64}),
            std::make_pair(LegalizerInfo::Legal, s64));
  ASSERT_EQ(L.getAction({G_PTRTOINT, 1, p0}),
            std::make_pair(LegalizerInfo::Legal, p0));
  // Make sure we also handle unusual sizes
  ASSERT_EQ(L.getAction({G_PTRTOINT, 0, LLT::scalar(65)}),
            std::make_pair(LegalizerInfo::NarrowScalar, s64));
  ASSERT_EQ(L.getAction({G_PTRTOINT, 1, LLT::pointer(0, 32)}),
            std::make_pair(LegalizerInfo::Unsupported, LLT::pointer(0, 32)));
}

TEST(LegalizerInfoTest, MultipleSteps) {
  using namespace TargetOpcode;
  LegalizerInfo L;
  LLT s32 = LLT::scalar(32);
  LLT s64 = LLT::scalar(64);

  L.setLegalizeScalarToDifferentSizeStrategy(
      G_UREM, 0, LegalizerInfo::widenToLargerTypesUnsupportedOtherwise);
  L.setAction({G_UREM, 0, s32}, LegalizerInfo::Lower);
  L.setAction({G_UREM, 0, s64}, LegalizerInfo::Lower);

  L.computeTables();

  ASSERT_EQ(L.getAction({G_UREM, LLT::scalar(16)}),
            std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(32)));
  ASSERT_EQ(L.getAction({G_UREM, LLT::scalar(32)}),
            std::make_pair(LegalizerInfo::Lower, LLT::scalar(32)));
}

TEST(LegalizerInfoTest, SizeChangeStrategy) {
  using namespace TargetOpcode;
  LegalizerInfo L;
  for (unsigned Size : {1, 8, 16, 32})
    L.setAction({G_UREM, 0, LLT::scalar(Size)}, LegalizerInfo::Legal);

  L.setLegalizeScalarToDifferentSizeStrategy(
      G_UREM, 0, LegalizerInfo::widenToLargerTypesUnsupportedOtherwise);
  L.computeTables();

  // Check we infer the correct types and actually do what we're told.
  for (unsigned Size : {1, 8, 16, 32}) {
    ASSERT_EQ(L.getAction({G_UREM, LLT::scalar(Size)}),
              std::make_pair(LegalizerInfo::Legal, LLT::scalar(Size)));
  }
  ASSERT_EQ(L.getAction({G_UREM, LLT::scalar(2)}),
            std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(8)));
  ASSERT_EQ(L.getAction({G_UREM, LLT::scalar(7)}),
            std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(8)));
  ASSERT_EQ(L.getAction({G_UREM, LLT::scalar(9)}),
            std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(16)));
  ASSERT_EQ(L.getAction({G_UREM, LLT::scalar(17)}),
            std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(32)));
  ASSERT_EQ(L.getAction({G_UREM, LLT::scalar(31)}),
            std::make_pair(LegalizerInfo::WidenScalar, LLT::scalar(32)));
  ASSERT_EQ(L.getAction({G_UREM, LLT::scalar(33)}),
            std::make_pair(LegalizerInfo::Unsupported, LLT::scalar(33)));
}
}
