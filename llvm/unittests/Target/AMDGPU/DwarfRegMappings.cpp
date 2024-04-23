//===- llvm/unittests/Target/AMDGPU/DwarfRegMappings.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(AMDGPU, TestWave64DwarfRegMapping) {
  for (auto Triple :
       {"amdgcn-amd-", "amdgcn-amd-amdhsa", "amdgcn-amd-amdpal"}) {
    auto TM = createAMDGPUTargetMachine(Triple, "gfx1010", "+wavefrontsize64");
    if (TM) {
      GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetFeatureString()), *TM);
      auto MRI = ST.getRegisterInfo();
      if (MRI) {
        // Wave64 Dwarf register mapping test numbers
        // PC_64 => 16, EXEC_MASK_64 => 17, S0 => 32, S63 => 95,
        // S64 => 1088, S105 => 1129, V0 => 2560, V255 => 2815,
        // A0 => 3072, A255 => 3327
        for (int llvmReg :
             {16, 17, 32, 95, 1088, 1129, 2560, 2815, 3072, 3327}) {
          MCRegister PCReg(*MRI->getLLVMRegNum(llvmReg, false));
          EXPECT_EQ(llvmReg, MRI->getDwarfRegNum(PCReg, false));
          EXPECT_EQ(llvmReg, MRI->getDwarfRegNum(PCReg, true));
        }
      }
    }
  }
}

TEST(AMDGPU, TestWave32DwarfRegMapping) {
  for (auto Triple :
       {"amdgcn-amd-", "amdgcn-amd-amdhsa", "amdgcn-amd-amdpal"}) {
    auto TM = createAMDGPUTargetMachine(Triple, "gfx1010", "+wavefrontsize32");
    if (TM) {
      GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetFeatureString()), *TM);
      auto MRI = ST.getRegisterInfo();
      if (MRI) {
        // Wave32 Dwarf register mapping test numbers
        // PC_64 => 16, EXEC_MASK_32 => 1, S0 => 32, S63 => 95,
        // S64 => 1088, S105 => 1129, V0 => 1536, V255 => 1791,
        // A0 => 2048, A255 => 2303
        for (int llvmReg :
             {16, 1, 32, 95, 1088, 1129, 1536, 1791, 2048, 2303}) {
          MCRegister PCReg(*MRI->getLLVMRegNum(llvmReg, false));
          EXPECT_EQ(llvmReg, MRI->getDwarfRegNum(PCReg, false));
          EXPECT_EQ(llvmReg, MRI->getDwarfRegNum(PCReg, true));
        }
      }
    }
  }
}
