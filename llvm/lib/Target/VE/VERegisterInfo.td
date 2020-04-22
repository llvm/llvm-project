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

class VEReg<bits<7> enc, string n, list<Register> subregs = [],
            list<string> altNames = [], list<Register> aliases = []>
        : Register<n, altNames> {
  let HWEncoding{15-7} = 0;
  let HWEncoding{6-0} = enc;
  let Namespace = "VE";
  let SubRegs = subregs;
  let Aliases = aliases;
}

let Namespace = "VE" in {
  def sub_i8      : SubRegIndex<8, 56>;         // Low 8 bit (56..63)
  def sub_i16     : SubRegIndex<16, 48>;        // Low 16 bit (48..63)
  def sub_i32     : SubRegIndex<32, 32>;        // Low 32 bit (32..63)
  def sub_f32     : SubRegIndex<32>;            // High 32 bit (0..31)
  def AsmName     : RegAltNameIndex;
}

//-----------------------------------------------------------------------------
// Gneric Registers
//-----------------------------------------------------------------------------

let RegAltNameIndices = [AsmName] in {

// Generic integer registers - 8 bits wide
foreach I = 0-63 in
  def SB#I : VEReg<I, "sb"#I, [], ["s"#I]>, DwarfRegNum<[I]>;

// Generic integer registers - 16 bits wide
let SubRegIndices = [sub_i8] in
foreach I = 0-63 in
  def SH#I : VEReg<I, "sh"#I, [!cast<VEReg>("SB"#I)], ["s"#I]>,
                   DwarfRegNum<[I]>;

// Generic integer registers - 32 bits wide
let SubRegIndices = [sub_i16] in
foreach I = 0-63 in
  def SW#I : VEReg<I, "sw"#I, [!cast<VEReg>("SH"#I)], ["s"#I]>,
                   DwarfRegNum<[I]>;

// Generic floating point registers - 32 bits wide
//   NOTE: Mark SF#I as alias of SW#I temporary to avoid register allocation
//         problem.
foreach I = 0-63 in
  def SF#I : VEReg<I, "sf"#I, [], ["s"#I], [!cast<VEReg>("SW"#I)]>,
                   DwarfRegNum<[I]>;

// Generic integer registers - 64 bits wide
let SubRegIndices = [sub_i32, sub_f32], CoveredBySubRegs = 1 in
foreach I = 0-63 in
  def SX#I : VEReg<I, "s"#I, [!cast<VEReg>("SW"#I), !cast<VEReg>("SF"#I)],
                   ["s"#I]>, DwarfRegNum<[I]>;

} // RegAltNameIndices = [AsmName]

// Register classes.
//
// The register order is defined in terms of the preferred
// allocation order.
def I8  : RegisterClass<"VE", [i8], 8,
                        (add (sequence "SB%u", 0, 7),
                             (sequence "SB%u", 34, 63),
                             (sequence "SB%u", 8, 33))>;
def I16 : RegisterClass<"VE", [i16], 16,
                        (add (sequence "SH%u", 0, 7),
                             (sequence "SH%u", 34, 63),
                             (sequence "SH%u", 8, 33))>;
def I32 : RegisterClass<"VE", [i32], 32,
                        (add (sequence "SW%u", 0, 7),
                             (sequence "SW%u", 34, 63),
                             (sequence "SW%u", 8, 33))>;
def I64 : RegisterClass<"VE", [i64, f64], 64,
                        (add (sequence "SX%u", 0, 7),
                             (sequence "SX%u", 34, 63),
                             (sequence "SX%u", 8, 33))>;
def F32 : RegisterClass<"VE", [f32], 32,
                        (add (sequence "SF%u", 0, 7),
                             (sequence "SF%u", 34, 63),
                             (sequence "SF%u", 8, 33))>;
