//===-- AnalysisTest.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"

#include <cassert>
#include <memory>

#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {
namespace {

using testing::Pair;
using testing::UnorderedElementsAre;

class AnalysisTest : public ::testing::Test {
protected:
  AnalysisTest() {
    const std::string TT = "x86_64-unknown-linux";
    std::string error;
    const llvm::Target *const TheTarget =
        llvm::TargetRegistry::lookupTarget(TT, error);
    if (!TheTarget) {
      llvm::errs() << error << "\n";
      return;
    }
    STI.reset(TheTarget->createMCSubtargetInfo(TT, "haswell", ""));

    // Compute the ProxResIdx of ports uses in tests.
    const auto &SM = STI->getSchedModel();
    for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
      const std::string Name = SM.getProcResource(I)->Name;
      if (Name == "HWPort0") {
        P0Idx = I;
      } else if (Name == "HWPort1") {
        P1Idx = I;
      } else if (Name == "HWPort5") {
        P5Idx = I;
      } else if (Name == "HWPort6") {
        P6Idx = I;
      } else if (Name == "HWPort05") {
        P05Idx = I;
      } else if (Name == "HWPort0156") {
        P0156Idx = I;
      }
    }
    EXPECT_NE(P0Idx, 0);
    EXPECT_NE(P1Idx, 0);
    EXPECT_NE(P5Idx, 0);
    EXPECT_NE(P6Idx, 0);
    EXPECT_NE(P05Idx, 0);
    EXPECT_NE(P0156Idx, 0);
  }

  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
  }

protected:
  std::unique_ptr<const llvm::MCSubtargetInfo> STI;
  uint16_t P0Idx = 0;
  uint16_t P1Idx = 0;
  uint16_t P5Idx = 0;
  uint16_t P6Idx = 0;
  uint16_t P05Idx = 0;
  uint16_t P0156Idx = 0;
};

TEST_F(AnalysisTest, ComputeIdealizedProcResPressure_2P0) {
  const auto Pressure =
      computeIdealizedProcResPressure(STI->getSchedModel(), {{P0Idx, 2}});
  EXPECT_THAT(Pressure, UnorderedElementsAre(Pair(P0Idx, 2.0)));
}

TEST_F(AnalysisTest, ComputeIdealizedProcResPressure_2P05) {
  const auto Pressure =
      computeIdealizedProcResPressure(STI->getSchedModel(), {{P05Idx, 2}});
  EXPECT_THAT(Pressure,
              UnorderedElementsAre(Pair(P0Idx, 1.0), Pair(P5Idx, 1.0)));
}

TEST_F(AnalysisTest, ComputeIdealizedProcResPressure_2P05_2P0156) {
  const auto Pressure = computeIdealizedProcResPressure(
      STI->getSchedModel(), {{P05Idx, 2}, {P0156Idx, 2}});
  EXPECT_THAT(Pressure,
              UnorderedElementsAre(Pair(P0Idx, 1.0), Pair(P1Idx, 1.0),
                                   Pair(P5Idx, 1.0), Pair(P6Idx, 1.0)));
}

TEST_F(AnalysisTest, ComputeIdealizedProcResPressure_1P1_1P05_2P0156) {
  const auto Pressure = computeIdealizedProcResPressure(
      STI->getSchedModel(), {{P1Idx, 1}, {P05Idx, 1}, {P0156Idx, 2}});
  EXPECT_THAT(Pressure,
              UnorderedElementsAre(Pair(P0Idx, 1.0), Pair(P1Idx, 1.0),
                                   Pair(P5Idx, 1.0), Pair(P6Idx, 1.0)));
}

} // namespace
} // namespace exegesis
} // namespace llvm
