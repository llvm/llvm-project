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

class M88kRegisterClass<list<ValueType> regTypes, int alignment,
                        bits<1> regFile, dag regList>
  : RegisterClass<"M88k", regTypes, alignment, regList> {
  bits<1> HWRegFile = regFile; // HW register file: 0 = GPR, 1 = XPR    
}

let Namespace = "M88k" in {
  def sub_hi : SubRegIndex<32, 0>;
  def sub_lo : SubRegIndex<32, 1>;
}

// General register.
class GRi<bits<5> Enc, string n> : M88kReg<Enc, n>;

// General-purpose registers
foreach I = 0-31 in {
  def R#I : GRi<I, "r"#I>;
}

// Register pairs 
def GPRTuple : RegisterTuples<
          [sub_hi, sub_lo],
          [(add (decimate (sequence "R%u", 2, 30), 2)),
           (add (decimate (sequence "R%u", 3, 30), 2))
          ]>;

def GPR : M88kRegisterClass<[i32], 32, 0,
                            (add (sequence "R%u", 0, 31))>;
def GPR64 : RegisterClass<"M88k", [i64], 32, (add GPRTuple)>;

def GPROpnd : RegisterOperand<GPR>;
def GPR64Opnd : RegisterOperand<GPR64>;

// Experimental: GPR with floating point types.
def GPRF32 : M88kRegisterClass<[f32], 32, 0,
                            (add GPR)>;
// TODO Only even registers.
def GPRF64 : M88kRegisterClass<[f64], 32, 0,
                            (add GPR64)>;

def GPRF32Opnd : RegisterOperand<GPRF32>;
def GPRF64Opnd : RegisterOperand<GPRF64>;

// Extended (floating point) register.
class XRi<bits<5> Enc, string n> : M88kReg<Enc, n>;

// Extended (floating point) registers.
foreach I = 0-31 in {
  def X#I : XRi<I, "x"#I>;
}

def XR : M88kRegisterClass<[f32, f64, f80], 32, 1,
                            (add (sequence "X%u", 0, 31))>;

def XROpnd : RegisterOperand<XR>;
