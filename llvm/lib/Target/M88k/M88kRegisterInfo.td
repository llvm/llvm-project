//===-- M88kRegisterInfo.td - M88k Register definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class M88kReg<bits<5> Enc, string n> : Register<n> {
  let HWEncoding{15-5} = 0;
  let HWEncoding{4-0} = Enc;
  let Namespace = "M88k";
}

class M88kRegWithSubregs<string n, list<Register> subregs>
  : RegisterWithSubRegs<n, subregs> {
  let Namespace = "M88k";
}

class M88kRegisterClass<list<ValueType> regTypes, int alignment,
                        bits<1> regFile, dag regList>
  : RegisterClass<"M88k", regTypes, alignment, regList> {
  bits<1> HWRegFile = regFile; // HW register file: 0 = GPR, 1 = XPR    
}

let Namespace = "M88k" in {
  def sub_hi : SubRegIndex<32, 0>;
  def sub_lo : SubRegIndex<32, 32>;
}

// General-purpose registers
foreach I = 0-31 in {
  def R#I : M88kReg<I, "r"#I>;
}

// Register pairs.
// TODO Check let AsmName = "", isArtificial = 1
let SubRegIndices = [sub_hi, sub_lo], CoveredBySubRegs = true in {
  foreach I = 0-15 in {
    defvar Rx = "R" # !mul(2,I);
    defvar Ry = "R" # !add(!mul(2,I),1);
    def Rx#_#Ry : M88kRegWithSubregs<Rx#_#Ry,
                                    [!cast<Register>(Rx), !cast<Register>(Ry)]>;
  }
}

def GPR : M88kRegisterClass<[i32], 32, 0,
                            (add (sequence "R%u", 0, 31))>;
def GPR64 : M88kRegisterClass<[i64], 32, 0,
                              (add R0_R1, R2_R3, R4_R5, R6_R7, R8_R9,
                               R10_R11, R12_R13, R14_R15, R16_R17, R18_R19,
                               R20_R21, R22_R23, R24_R25, R26_R27, R28_R29,
                               R30_R31)>;

def GPROpnd : RegisterOperand<GPR>;
def GPR64Opnd : RegisterOperand<GPR64>;

// Experimental: GPR with floating point types.
def GPRF32 : M88kRegisterClass<[f32], 32, 0,
                            (add GPR)>;
def GPRF64 : M88kRegisterClass<[f64], 32, 0,
                            (add GPR64)>;

def GPRF32Opnd : RegisterOperand<GPRF32>;
def GPRF64Opnd : RegisterOperand<GPRF64>;

// Extended (floating point) registers.
foreach I = 0-31 in {
  def X#I : M88kReg<I, "x"#I>;
}

def XR : M88kRegisterClass<[f32, f64, f80], 32, 1,
                            (add (sequence "X%u", 0, 31))>;

def XROpnd : RegisterOperand<XR>;
