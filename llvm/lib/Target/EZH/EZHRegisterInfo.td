//===-- EZHRegisterInfo.td - EZH Register defs ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class EZHReg<bits<4> num, string n> : Register<n> {
  field bits<4> Num;
  let Num = num;
  let Namespace = "EZH";
  let HWEncoding{3-0} = num;
}

def sub_even : SubRegIndex<32>;
def sub_odd  : SubRegIndex<32, 32>;

// General Purpose Registers
foreach i = 0-7 in {
  def R#i : EZHReg<i, "r"#i>, DwarfRegNum<[i]>;
}

def R0_R1 : RegisterTuples<[sub_even, sub_odd], [(add R0), (add R1)]>;
def R2_R3 : RegisterTuples<[sub_even, sub_odd], [(add R2), (add R3)]>;
def R4_R5 : RegisterTuples<[sub_even, sub_odd], [(add R4), (add R5)]>;
def R6_R7 : RegisterTuples<[sub_even, sub_odd], [(add R6), (add R7)]>;

def GPO : EZHReg<8,  "gpo">, DwarfRegNum<[8]>;
def GPD : EZHReg<9,  "gpd">, DwarfRegNum<[9]>;
def CFS : EZHReg<10, "cfs">, DwarfRegNum<[10]>;
def CFM : EZHReg<11, "cfm">, DwarfRegNum<[11]>;
def SP  : EZHReg<12, "sp">, DwarfRegNum<[12]>;
def PC  : EZHReg<13, "pc">, DwarfRegNum<[13]>;
def GPI : EZHReg<14, "gpi">, DwarfRegNum<[14]>;
def RA  : EZHReg<15, "ra">, DwarfRegNum<[15]>;

// Register classes
def GPR : RegisterClass<"EZH", [i32], 32, (add
  R0, R1, R2, R3, R4, R5, R6, R7, RA, SP, PC, GPO, GPD, CFS, CFM, GPI
)>;

def GPRPair : RegisterClass<"EZH", [i64], 64, (add
  R0_R1, R2_R3, R4_R5, R6_R7
)>;

def tcGPR : RegisterClass<"EZH", [i32], 32, (add R0, R1, R2, R3, R4, R5)>;

def GPRSP : RegisterClass<"EZH", [i32], 32, (add
  GPR, SP
)>;

def GPRAll : RegisterClass<"EZH", [i32], 32, (add
  GPR, SP, PC, RA, GPO, GPD, CFS, CFM, GPI
)>;

def SPR : RegisterClass<"EZH", [i32], 32, (add SP)>;
def PCR : RegisterClass<"EZH", [i32], 32, (add PC)>;

// Condition code register
def SR : EZHReg<0, "sw">;
def CCR : RegisterClass<"EZH", [i32], 32, (add SR)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}
