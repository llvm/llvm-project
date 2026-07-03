//===- SPIRVModuleAnalysisTests.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVModuleAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

// A capability pruned by removeCapabilityIf must be dropped from the minimal
// capabilities too, since those are the ones the AsmPrinter emits as
// OpCapability.
TEST(SPIRVModuleAnalysisTest, RemoveCapabilityIfPrunesMinimalCaps) {
  SPIRV::RequirementHandler Reqs;
  Reqs.addCapability(SPIRV::Capability::BitInstructions);
  Reqs.addCapability(SPIRV::Capability::Shader);

  Reqs.removeCapabilityIf(SPIRV::Capability::BitInstructions,
                          SPIRV::Capability::Shader);

  EXPECT_FALSE(llvm::is_contained(Reqs.getMinimalCapabilities(),
                                  SPIRV::Capability::BitInstructions));
  EXPECT_TRUE(llvm::is_contained(Reqs.getMinimalCapabilities(),
                                 SPIRV::Capability::Shader));
}

// When the guarding capability is absent, nothing is removed.
TEST(SPIRVModuleAnalysisTest, RemoveCapabilityIfNoopWhenGuardAbsent) {
  SPIRV::RequirementHandler Reqs;
  Reqs.addCapability(SPIRV::Capability::BitInstructions);

  Reqs.removeCapabilityIf(SPIRV::Capability::BitInstructions,
                          SPIRV::Capability::Shader);

  EXPECT_TRUE(llvm::is_contained(Reqs.getMinimalCapabilities(),
                                 SPIRV::Capability::BitInstructions));
}
