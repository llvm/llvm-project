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

let Namespace = "VE" in {
  def sub_i8      : SubRegIndex<8, 56>;         // Low 8 bit (56..63)
  def sub_i16     : SubRegIndex<16, 48>;        // Low 16 bit (48..63)
  def sub_i32     : SubRegIndex<32, 32>;        // Low 32 bit (32..63)
  def sub_f32     : SubRegIndex<32>;            // High 32 bit (0..31)
}

// Registers are identified with 7-bit ID numbers.
// R - 64-bit integer or floating-point registers
class R<bits<7> Enc, string n, list<Register> subregs = [],
        list<Register> aliases = []>: VEReg<Enc, n> {
  let SubRegs = subregs;
  let Aliases = aliases;
}

// Generic integer registers - 8 bits wide
foreach I = 0-63 in
  def SB#I : R<I, "S"#I>, DwarfRegNum<[I]>;

// Generic integer registers - 16 bits wide
let SubRegIndices = [sub_i8] in
foreach I = 0-63 in
  def SH#I : R<I, "S"#I, [!cast<R>("SB"#I)]>, DwarfRegNum<[I]>;

// Generic integer registers - 32 bits wide
let SubRegIndices = [sub_i16] in
foreach I = 0-63 in
  def SW#I : R<I, "S"#I, [!cast<R>("SH"#I)]>, DwarfRegNum<[I]>;

// Generic floating point registers - 32 bits wide
//   NOTE: Mark SF#I as alias of SW#I temporary to avoid register allocation
//         problem.
foreach I = 0-63 in
  def SF#I : R<I, "S"#I, [], [!cast<R>("SW"#I)]>, DwarfRegNum<[I]>;

// Generic integer registers - 64 bits wide
let SubRegIndices = [sub_i32, sub_f32], CoveredBySubRegs = 1 in
foreach I = 0-63 in
  def SX#I : R<I, "S"#I, [!cast<R>("SW"#I), !cast<R>("SF"#I)]>,
             DwarfRegNum<[I]>;

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
