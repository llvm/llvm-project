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

// General register
class GRi<bits<5> Enc, string n> : M88kReg<Enc, n>;

// General-purpose registers
foreach I = 0-31 in {
  def R#I : GRi<I, "r"#I>;
}

def GPR : RegisterClass<"M88k", [i32], 32,
                            (add (sequence "R%u", 0, 31))>;

def GPROpnd : RegisterOperand<GPR>;

// Experimental: GPR with floating point types.
def GPRF32 : RegisterClass<"M88k", [f32], 32,
                            (add (sequence "R%u", 0, 31))>;
// TODO Only even registers.
def GPRF64 : RegisterClass<"M88k", [f64], 32,
                            (add (sequence "R%u", 0, 31))>;

def GPRF32Opnd : RegisterOperand<GPRF32>;
def GPRF64Opnd : RegisterOperand<GPRF64>;

// Extended (floating point) register.
class XRi<bits<5> Enc, string n> : M88kReg<Enc, n>;

// Extended (floating point) registers.
foreach I = 0-31 in {
  def X#I : XRi<I, "x"#I>;
}

def XR : RegisterClass<"M88k", [f32, f64], 32,
                            (add (sequence "X%u", 0, 31))>;

def XROpnd : RegisterOperand<XR>;
