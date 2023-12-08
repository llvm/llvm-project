//===-- SPIRVRegisterBanks.td - Describe SPIR-V RegBanks ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Although RegisterBankSelection is disabled we need to distinct the banks
// as InstructionSelector RegClass checking code relies on them
def IDRegBank : RegisterBank<"IDBank", [ID]>;
def fIDRegBank : RegisterBank<"fIDBank", [fID]>;
def vIDRegBank : RegisterBank<"vIDBank", [vID]>;
def vfIDRegBank : RegisterBank<"vfIDBank", [vfID]>;
def TYPERegBank : RegisterBank<"TYPEBank", [TYPE]>;
