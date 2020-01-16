//===-- VERegisterInfo.td - VE Register defs ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the VE register file
//===----------------------------------------------------------------------===//

class VEReg<bits<7> Enc, string n> : Register<n> {
  let HWEncoding{15-7} = 0;
  let HWEncoding{6-0} = Enc;
  let Namespace = "VE";
}

// Registers are identified with 7-bit ID numbers.
// R - 64-bit integer or floating-point registers
class R<bits<7> Enc, string n, list<Register> subregs = [],
        list<Register> aliases = []>: VEReg<Enc, n> {
  let SubRegs = subregs;
  let Aliases = aliases;
}

// Generic integer registers - 64 bits wide
foreach I = 0-63 in
  def SX#I : R<I, "S"#I, []>,
             DwarfRegNum<[I]>;

// Register classes.
//
// The register order is defined in terms of the preferred
// allocation order.
def I64 : RegisterClass<"VE", [i64], 64,
                        (sequence "SX%u", 0, 63)>;
