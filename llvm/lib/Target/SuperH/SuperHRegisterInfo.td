//==-- SuperHRegisterInfo.td - SuperH register definitions ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the SuperH Register file, defining the registers
/// aliases between the registers, and the register classes built out of the
/// registers.
///
//===----------------------------------------------------------------------===//

// SuperH Registers
class SHRegister<string N, bits<16> ENC, 
            list<Register> SUBREGS = [], list<SubRegIndex> SUBIDX,
            list<string> ALTNAMES = []>
            : Register<N, ALTNAMES> {
  let Namespace     = "SH";
  let HWEncoding    = ENC;
  let SubRegs       = SUBREGS;
  let SubRegIndices = SUBIDX;
}

let Namespace = "SH" in {

  // MAC registers
  def sub_16bit_lo : SubRegIndex<16>;
  def sub_16bit_hi : SubRegIndex<16, 16>;

  // FP Registers
  def sub_32bit_lo : SubRegIndex<32>;
  def sub_32bit_hi : SubRegIndex<32, 32>;
}

//===----------------------------------------------------------------------===//
// Registers
//===----------------------------------------------------------------------===//

let Namespace = "SH" in {

  // GP Registers
  def R0 : SHRegister<0, "r0>, DwarfRegNum<[ 0 ]>;
  def R1 : SHRegister<1, "r1>, DwarfRegNum<[ 1 ]>;
  def R2 : SHRegister<2, "r2>, DwarfRegNum<[ 2 ]>;
  def R3 : SHRegister<3, "r3>, DwarfRegNum<[ 3 ]>;
  def R4 : SHRegister<4, "r4>, DwarfRegNum<[ 4 ]>;
  def R5 : SHRegister<5, "r5>, DwarfRegNum<[ 5 ]>;
  def R6 : SHRegister<6, "r6>, DwarfRegNum<[ 6 ]>;
  def R7 : SHRegister<7, "r7>, DwarfRegNum<[ 7 ]>;
  def R8 : SHRegister<8, "r8>, DwarfRegNum<[ 8 ]>;
  def R9 : SHRegister<9, "r9>, DwarfRegNum<[ 9 ]>;
  def R10 : SHRegister<10, "r10>, DwarfRegNum<[ 10 ]>;
  def R11 : SHRegister<11, "r11>, DwarfRegNum<[ 11 ]>;
  def R12 : SHRegister<12, "r12>, DwarfRegNum<[ 12 ]>;
  def R13 : SHRegister<13, "r13>, DwarfRegNum<[ 13 ]>;
  def R14 : SHRegister<14, "r14, [ "fp" ]>, DwarfRegNum<[ 14 ]>;
  def R15 : SHRegister<15, "r15, [ "sp" ]>, DwarfRegNum<[ 15 ]>;

  // Control Registers
  def SR : SHRegister<16, "sr">, DwarfRegNum<[ 22 ]>;
  def GBR : SHRegister<17, "gbr">, DwarfRegNum<[ 18 ]>;
  def VBR : SHRegister<18, "vbr">, DwarfRegNum<[ 19 ]>;

  // System Registers
  def MACH : SHRegister<19, "mach">, DwarfRegNum<[ 21 ]>;
  def MACL : SHRegister<20, "macl">, DwarfRegNum<[ 21 ]>;
  let SubRegIndices = [sub_16bit_lo, sub_16bit_hi] in {
    def MAC : SHRegister<21, "mac", [], [MACL, MACH]>;
  }
  def PC : SHRegister<22, "pc">, DwarfRegNum<[ 16 ]>;
  def PR : SHRegister<23, "pr">, DwarfRegNum<[ 17 ]>;

  // FR Registers

  // TODO: Add float registers
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

def GPR : RegisterClass<"SH", [i32], 32, (
  add(
    R0, R1, R2,  R3,  R4,  R5,  R6,  R7,  // Banked memory
    R8, R9, R10, R11, R12, R13, R14, R15, // Non-banked memory.
  ) 
)>;

// Control registers
def CR : RegisterClass<"SH", [i32], 32, (add SR, GBR, VBR>) {
  let CopyCost = -1;
  let isAllocatable = 0;
}

// System registers.
def SYSR : RegisterClass<"SH", [ i32 ], 32, (add MACH, MACL, PR, PC)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

// MAC register.
def MACR : RegisterClass<"SH", [ i64 ], 64, (add MAC)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

def PCR : RegisterClass<"SH", [ i32 ], 32, (add PC)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}