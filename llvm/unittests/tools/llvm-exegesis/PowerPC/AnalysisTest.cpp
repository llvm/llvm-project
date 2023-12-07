//===-- PPCAnalysisTest.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"

#include <cassert>
#include <memory>

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm{
namespace exegesis {
namespace {

using testing::Pair;
using testing::UnorderedElementsAre;

class PPCAnalysisTest : public ::testing::Test {
protected:
  PPCAnalysisTest() {
    const std::string TT = "powerpc64le-unknown-linux";
    std::string error;
    const Target *const TheTarget = TargetRegistry::lookupTarget(TT, error);
    if (!TheTarget) {
      errs() << error << "\n";
      return;
    }
    STI.reset(TheTarget->createMCSubtargetInfo(TT, "pwr9", ""));

    // Compute the ProxResIdx of ports uses in tests.
    const auto &SM = STI->getSchedModel();
    for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
      const std::string Name = SM.getProcResource(I)->Name;
      if (Name == "ALU") {
        ALUIdx = I;
      } else if (Name == "ALUE") {
        ALUEIdx = I;
      } else if (Name == "ALUO") {
        ALUOIdx = I;
      } else if (Name == "IP_AGEN") {
        IPAGENIdx = I;
      }
    }
    EXPECT_NE(ALUIdx, 0);
    EXPECT_NE(ALUEIdx, 0);
    EXPECT_NE(ALUOIdx, 0);
    EXPECT_NE(IPAGENIdx, 0);
  }

  static void SetUpTestCase() {
    LLVMInitializePowerPCTargetInfo();
    LLVMInitializePowerPCTarget();
    LLVMInitializePowerPCTargetMC();
  }

protected:
  std::unique_ptr<const MCSubtargetInfo> STI;
  uint16_t ALUIdx = 0;
  uint16_t ALUEIdx = 0;
  uint16_t ALUOIdx = 0;
  uint16_t IPAGENIdx = 0;
};

TEST_F(PPCAnalysisTest, ComputeIdealizedProcResPressure_2ALU) {
  const auto Pressure =
      computeIdealizedProcResPressure(STI->getSchedModel(), {{ALUIdx, 2, 0}});
  EXPECT_THAT(Pressure, UnorderedElementsAre(Pair(ALUIdx, 2.0)));
}

TEST_F(PPCAnalysisTest, ComputeIdealizedProcResPressure_1ALUE) {
  const auto Pressure =
      computeIdealizedProcResPressure(STI->getSchedModel(), {{ALUEIdx, 2, 0}});
  EXPECT_THAT(Pressure, UnorderedElementsAre(Pair(ALUEIdx, 2.0)));
}

TEST_F(PPCAnalysisTest, ComputeIdealizedProcResPressure_1ALU1IPAGEN) {
  const auto Pressure = computeIdealizedProcResPressure(
      STI->getSchedModel(), {{ALUIdx, 1, 0}, {IPAGENIdx, 1, 0}});
  EXPECT_THAT(Pressure, UnorderedElementsAre(Pair(ALUIdx, 1.0),Pair(IPAGENIdx, 1)));
}
} // namespace
} // namespace exegesis
} // namespace llvm
