//=-- RISCVRegisterBank.td - Describe the RISC-V Banks -------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

/// General Purpose Registers: X.
def GPRBRegBank : RegisterBank<"GPRB", [GPR]>;

/// Floating Point Registers: F.
def FPRBRegBank : RegisterBank<"FPRB", [FPR64]>;

/// Vector Register Banks:
def VRRegBank : RegisterBank<"VRB", [VR]>;
def VRNoV0RegBank : RegisterBank<"VRNoV0B", [VRNoV0]>;
def VRM2RegBank : RegisterBank<"VRM2B", [VRM2]>;
def VRM2NoV0RegBank : RegisterBank<"VRM2NoV0B", [VRM2NoV0]>;
def VRM4RegBank : RegisterBank<"VRM4B", [VRM4]>;
def VRM4NoV0RegBank : RegisterBank<"VRM4NoV0B", [VRM4NoV0]>;
def VRM8RegBank : RegisterBank<"VRM8B", [VRM8]>;
def VRM8NoV0RegBank : RegisterBank<"VRM8NoV0B", [VRM8NoV0]>;
def VMRegBank : RegisterBank<"VMB", [VM]>;
def VMV0RegBank : RegisterBank<"VMNoV0B", [VMV0]>;
