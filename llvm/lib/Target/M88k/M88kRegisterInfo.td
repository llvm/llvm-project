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