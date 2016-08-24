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
using llvm::MachineLegalizer::LegalizeAction::Legal;
using llvm::MachineLegalizer::LegalizeAction::NarrowScalar;
using llvm::MachineLegalizer::LegalizeAction::WidenScalar;
using llvm::MachineLegalizer::LegalizeAction::FewerElements;
using llvm::MachineLegalizer::LegalizeAction::MoreElements;
using llvm::MachineLegalizer::LegalizeAction::Libcall;
using llvm::MachineLegalizer::LegalizeAction::Custom;
using llvm::MachineLegalizer::LegalizeAction::Unsupported;

// Define a couple of pretty printers to help debugging when things go wrong.
namespace llvm {
std::ostream &
operator<<(std::ostream &OS, const llvm::MachineLegalizer::LegalizeAction Act) {
  switch (Act) {
  case Legal: OS << "Legal"; break;
  case NarrowScalar: OS << "NarrowScalar"; break;
  case WidenScalar:  OS << "WidenScalar"; break;
  case FewerElements:  OS << "FewerElements"; break;
  case MoreElements:  OS << "MoreElements"; break;
  case Libcall: OS << "Libcall"; break;
  case Custom: OS << "Custom"; break;
  case Unsupported: OS << "Unsupported"; break;
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
  L.setAction({G_ADD, LLT::scalar(8)}, WidenScalar);
  L.setAction({G_ADD, LLT::scalar(16)}, WidenScalar);
  L.setAction({G_ADD, LLT::scalar(32)}, Legal);
  L.setAction({G_ADD, LLT::scalar(64)}, Legal);
  L.computeTables();

  // Check we infer the correct types and actually do what we're told.
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(8)}),
                        std::make_pair(WidenScalar, LLT::scalar(32)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(16)}),
                        std::make_pair(WidenScalar, LLT::scalar(32)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(32)}),
                        std::make_pair(Legal, LLT::scalar(32)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(64)}),
                        std::make_pair(Legal, LLT::scalar(64)));

  // Make sure the default for over-sized types applies.
  ASSERT_EQ(L.getAction({G_ADD, LLT::scalar(128)}),
                        std::make_pair(NarrowScalar, LLT::scalar(64)));
}

TEST(MachineLegalizerTest, VectorRISC) {
  using namespace TargetOpcode;
  MachineLegalizer L;
  // Typical RISCy set of operations based on ARM.
  L.setScalarInVectorAction(G_ADD, LLT::scalar(8), Legal);
  L.setScalarInVectorAction(G_ADD, LLT::scalar(16), Legal);
  L.setScalarInVectorAction(G_ADD, LLT::scalar(32), Legal);

  L.setAction({G_ADD, LLT::vector(8, 8)}, Legal);
  L.setAction({G_ADD, LLT::vector(16, 8)}, Legal);
  L.setAction({G_ADD, LLT::vector(4, 16)}, Legal);
  L.setAction({G_ADD, LLT::vector(8, 16)}, Legal);
  L.setAction({G_ADD, LLT::vector(2, 32)}, Legal);
  L.setAction({G_ADD, LLT::vector(4, 32)}, Legal);
  L.computeTables();

  // Check we infer the correct types and actually do what we're told for some
  // simple cases.
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(2, 8)}),
            std::make_pair(MoreElements, LLT::vector(8, 8)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(8, 8)}),
            std::make_pair(Legal, LLT::vector(8, 8)));
  ASSERT_EQ(L.getAction({G_ADD, LLT::vector(8, 32)}),
            std::make_pair(FewerElements, LLT::vector(4, 32)));
}

TEST(MachineLegalizerTest, MultipleTypes) {
  using namespace TargetOpcode;
  MachineLegalizer L;

  // Typical RISCy set of operations based on AArch64.
  L.setAction({G_PTRTOINT, 0, LLT::scalar(64)}, Legal);
  L.setAction({G_PTRTOINT, 1, LLT::pointer(0)}, Legal);

  L.setAction({G_PTRTOINT, 0, LLT::scalar(32)}, WidenScalar);
  L.computeTables();

  // Check we infer the correct types and actually do what we're told.
  ASSERT_EQ(L.getAction({G_PTRTOINT, 0, LLT::scalar(64)}),
                        std::make_pair(Legal, LLT::scalar(64)));
  ASSERT_EQ(L.getAction({G_PTRTOINT, 1, LLT::pointer(0)}),
                        std::make_pair(Legal, LLT::pointer(0)));
}
}
