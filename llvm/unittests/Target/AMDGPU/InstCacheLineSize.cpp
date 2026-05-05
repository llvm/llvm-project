//===- llvm/unittests/Target/AMDGPU/InstCacheLineSize.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

// Verify that AMDGPU::IsaInfo::getInstCacheLineSize() returns the correct
// value for targets with different cache line sizes. This catches bugs where
// the feature-bit based query in AMDGPUBaseInfo.cpp returns the wrong value
// (e.g., FeatureInstCacheLineSize64 mistakenly returning 128).
TEST(AMDGPU, GetInstCacheLineSize) {
  // GFX9 has 64-byte instruction cache lines.
  {
    auto TM = createAMDGPUTargetMachine("amdgcn-amd-amdhsa", "gfx908", "");
    if (TM) {
      GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetFeatureString()), *TM);
      EXPECT_EQ(64u, AMDGPU::IsaInfo::getInstCacheLineSize(&ST));
      EXPECT_EQ(64u, ST.getInstCacheLineSize());
    }
  }
  // GFX10 has 64-byte instruction cache lines.
  {
    auto TM = createAMDGPUTargetMachine("amdgcn-amd-amdhsa", "gfx1030", "");
    if (TM) {
      GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetFeatureString()), *TM);
      EXPECT_EQ(64u, AMDGPU::IsaInfo::getInstCacheLineSize(&ST));
      EXPECT_EQ(64u, ST.getInstCacheLineSize());
    }
  }
  // GFX11 has 128-byte instruction cache lines.
  {
    auto TM = createAMDGPUTargetMachine("amdgcn-amd-amdhsa", "gfx1100", "");
    if (TM) {
      GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetFeatureString()), *TM);
      EXPECT_EQ(128u, AMDGPU::IsaInfo::getInstCacheLineSize(&ST));
      EXPECT_EQ(128u, ST.getInstCacheLineSize());
    }
  }
  // GFX12 has 128-byte instruction cache lines.
  {
    auto TM = createAMDGPUTargetMachine("amdgcn-amd-amdhsa", "gfx1200", "");
    if (TM) {
      GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetFeatureString()), *TM);
      EXPECT_EQ(128u, AMDGPU::IsaInfo::getInstCacheLineSize(&ST));
      EXPECT_EQ(128u, ST.getInstCacheLineSize());
    }
  }
}
