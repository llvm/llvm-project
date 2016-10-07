//===- llvm/unittest/CodeGen/GlobalISel/MachineLegalizerTest.cpp ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/MachineLegalizer.h"
#include "llvm/Target/TargetOpcodes.h"
#include "gtest/gtest.h"

using namespace llvm;

// Define a couple of pretty printers to help debugging when things go wrong.
namespace llvm {
std::ostream &
operator<<(std::ostream &OS, const llvm::MachineLegalizer::LegalizeAction Act) {
  switch (Act) {
  case MachineLegalizer::Lower: OS << "Lower"; break;
  case MachineLegalizer::Legal: OS << "Legal"; break;
  case MachineLegalizer::NarrowScalar: OS << "NarrowScalar"; break;
  case MachineLegalizer::WidenScalar:  OS << "WidenScalar"; break;
  case MachineLegalizer::FewerElements:  OS << "FewerElements"; break;
  case MachineLegalizer::MoreElements:  OS << "MoreElements"; break;
  case MachineLegalizer::Libcall: OS << "Libcall"; break;
  case MachineLegalizer::Custom: OS << "Custom"; break;
  case MachineLegalizer::Unsupported: OS << "Unsupported"; break;
  case MachineLegalizer::NotFound: OS << "NotFound";
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


TEST(MachineLegalizerTest, ScalarRISC) {
  using namespace TargetOpcode;
  MachineLegalizer L;
  // Typical RISCy set of operations based on AArch64.
  L.setAction({G_ADD, LLT::scalar(8)}, MachineLegalizer::WidenScalar);
  L.setAction({G_ADD, LLT::scalar(16)}, MachineLegalizer::WidenScalar);
  L.setAction({G_ADD, LLT::scalar(32)}, MachineLegalizer::Legal);
  L.setAction({G_ADD, LLT::scalar(64)}, MachineLegalizer::Legal);
  L.computeTables();

  // Check we infer the correct types and actually do what we're told.
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(8)}),
            std::make_pair(MachineLegalizer::WidenScalar, LLT::scalar(32)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(16)}),
            std::make_pair(MachineLegalizer::WidenScalar, LLT::scalar(32)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(32)}),
            std::make_pair(MachineLegalizer::Legal, LLT::scalar(32)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(64)}),
            std::make_pair(MachineLegalizer::Legal, LLT::scalar(64)));

  // Make sure the default for over-sized types applies.
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(128)}),
            std::make_pair(MachineLegalizer::NarrowScalar, LLT::scalar(64)));
}

TEST(MachineLegalizerTest, VectorRISC) {
  using namespace TargetOpcode;
  MachineLegalizer L;
  // Typical RISCy set of operations based on ARM.
  L.setScalarInVectorAction(G_ADD, LLT::scalar(8), MachineLegalizer::Legal);
  L.setScalarInVectorAction(G_ADD, LLT::scalar(16), MachineLegalizer::Legal);
  L.setScalarInVectorAction(G_ADD, LLT::scalar(32), MachineLegalizer::Legal);

  L.setAction({G_ADD, LLT::vector(8, 8)}, MachineLegalizer::Legal);
  L.setAction({G_ADD, LLT::vector(16, 8)}, MachineLegalizer::Legal);
  L.setAction({G_ADD, LLT::vector(4, 16)}, MachineLegalizer::Legal);
  L.setAction({G_ADD, LLT::vector(8, 16)}, MachineLegalizer::Legal);
  L.setAction({G_ADD, LLT::vector(2, 32)}, MachineLegalizer::Legal);
  L.setAction({G_ADD, LLT::vector(4, 32)}, MachineLegalizer::Legal);
  L.computeTables();

  // Check we infer the correct types and actually do what we're told for some
  // simple cases.
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(2, 8)}),
            std::make_pair(MachineLegalizer::MoreElements, LLT::vector(8, 8)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(8, 8)}),
            std::make_pair(MachineLegalizer::Legal, LLT::vector(8, 8)));
  ASSERT_EQ(
      L.getAction({G_ADD, LLT::vector(8, 32)}),
      std::make_pair(MachineLegalizer::FewerElements, LLT::vector(4, 32)));
}

TEST(MachineLegalizerTest, MultipleTypes) {
  using namespace TargetOpcode;
  MachineLegalizer L;
  LLT p0 = LLT::pointer(0, 64);
  LLT s32 = LLT::scalar(32);
  LLT s64 = LLT::scalar(64);

  // Typical RISCy set of operations based on AArch64.
  L.setAction({G_PTRTOINT, 0, s64}, MachineLegalizer::Legal);
  L.setAction({G_PTRTOINT, 1, p0}, MachineLegalizer::Legal);

  L.setAction({G_PTRTOINT, 0, s32}, MachineLegalizer::WidenScalar);
  L.computeTables();

  // Check we infer the correct types and actually do what we're told.
  ASSERT_EQ(L.getAction({G_PTRTOINT, 0, s64}),
            std::make_pair(MachineLegalizer::Legal, s64));
  ASSERT_EQ(L.getAction({G_PTRTOINT, 1, p0}),
            std::make_pair(MachineLegalizer::Legal, p0));
}
}
