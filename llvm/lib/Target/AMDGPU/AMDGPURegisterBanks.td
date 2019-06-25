//=- AMDGPURegisterBank.td - Describe the AMDGPU Banks -------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def SGPRRegBank : RegisterBank<"SGPR",
  [SReg_32, SReg_64, SReg_128, SReg_256, SReg_512]
>;

def VGPRRegBank : RegisterBank<"VGPR",
  [VGPR_32, VReg_64, VReg_96, VReg_128, VReg_256, VReg_512]
>;

def SCCRegBank : RegisterBank <"SCC", [SReg_32, SCC_CLASS]>;

// It is helpful to distinguish conditions from ordinary SGPRs.
def VCCRegBank : RegisterBank <"VCC", [SReg_64]>;
