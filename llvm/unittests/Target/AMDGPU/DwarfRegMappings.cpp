//===- llvm/unittests/Target/AMDGPU/DwarfRegMappings.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"
#include <thread>

using namespace llvm;

std::once_flag flag;

void InitializeAMDGPUTarget() {
  std::call_once(flag, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
  });
}

std::unique_ptr<const GCNTargetMachine>
createTargetMachine(std::string TStr, StringRef CPU, StringRef FS) {
  InitializeAMDGPUTarget();

  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TStr, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  return std::unique_ptr<GCNTargetMachine>(static_cast<GCNTargetMachine *>(
      T->createTargetMachine(TStr, CPU, FS, Options, None, None)));
}

TEST(AMDGPUDwarfRegMappingTests, TestWave64DwarfRegMapping) {
  for (auto Triple :
       {"amdgcn-amd-", "amdgcn-amd-amdhsa", "amdgcn-amd-amdpal"}) {
    auto TM = createTargetMachine(Triple, "gfx1010", "+wavefrontsize64");
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
        }
      }
    }
  }
}

TEST(AMDGPUDwarfRegMappingTests, TestWave32DwarfRegMapping) {
  for (auto Triple :
       {"amdgcn-amd-", "amdgcn-amd-amdhsa", "amdgcn-amd-amdpal"}) {
    auto TM = createTargetMachine(Triple, "gfx1010", "+wavefrontsize32");
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
        }
      }
    }
  }
}
