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

//===----------------------------------------------------------------------===//
//  Declarations that describe the SuperH register file
//===----------------------------------------------------------------------===//

let Namespace = "SH" in {
  def sub_64_lo  : SubRegIndex<64>;
  def sub_64_hi  : SubRegIndex<64, 64>;
  def sub_32_lo  : SubRegIndex<32>;
  def sub_32_hi  : SubRegIndex<32, 32>;

  // Vector Registers
  def sub_fv_x   : SubRegIndex<32>;
  def sub_fv_y   : SubRegIndex<32, 32>;
  def sub_fv_z   : SubRegIndex<32, 64>;
  def sub_fv_w   : SubRegIndex<32, 96>;

  // Matrix Register
}

class Unallocatable {
  int CopyCost = -1;
  bit isAllocatable = false;
}

class SHReg<bits<16> Enc, string n> : Register<n> {
  let HWEncoding = Enc;
  let Namespace = "SH";
}

class SHRegWithSubRegs<bits<16> Enc, string n, list<Register> subregs>
  : RegisterWithSubRegs<n, subregs> {
  let HWEncoding = Enc;
  let Namespace = "SH";
}

// SH General Purpose Registers.
class GPRReg<bits<16> Enc, string n> : SHReg<Enc, n>;

// Control Registers
class CtrlReg<bits<16> Enc, string n> : SHReg<Enc, n>;

// SH System Registers.
class SysReg<bits<16> Enc, string n> : SHReg<Enc, n>;

// 32-bit floating point registers
class FRReg<bits<16> Enc, string n> : SHReg<Enc, n>;

// 64-bit (alias) FPU registers
class DRReg<bits<16> Enc, string n, list<Register> subregs>
  : SHRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_32_lo, sub_32_hi];
  let CoveredBySubRegs = 1;
}

// 32-bit (alias) VFPU registers
class FVReg<bits<16> Enc, string n, list<Register> subregs>
  : SHRegWithSubRegs<Enc, n, subregs> {
  let SubRegIndices = [sub_fv_x, sub_fv_y, sub_fv_z, sub_fv_w];
  let CoveredBySubRegs = 1;
}

// 512-bit Matrix Register
class MTRXReg<bits<16> Enc, string n> : SHReg<Enc, n>;

//===----------------------------------------------------------------------===//
// Registers
//===----------------------------------------------------------------------===//

let Namespace = "SH" in {

  // General Purpose Registers
  def R0  : GPRReg<0,  "r0">,   DwarfRegNum<[0]>;
  def R1  : GPRReg<1,  "r1">,   DwarfRegNum<[1]>;
  def R2  : GPRReg<2,  "r2">,   DwarfRegNum<[2]>;
  def R3  : GPRReg<3,  "r3">,   DwarfRegNum<[3]>;
  def R4  : GPRReg<4,  "r4">,   DwarfRegNum<[4]>;
  def R5  : GPRReg<5,  "r5">,   DwarfRegNum<[5]>;
  def R6  : GPRReg<6,  "r6">,   DwarfRegNum<[6]>;
  def R7  : GPRReg<7,  "r7">,   DwarfRegNum<[7]>;
  def R8  : GPRReg<8,  "r8">,   DwarfRegNum<[8]>;
  def R9  : GPRReg<9,  "r9">,   DwarfRegNum<[9]>;
  def R10 : GPRReg<10, "r10">,  DwarfRegNum<[10]>;
  def R11 : GPRReg<11, "r11">,  DwarfRegNum<[11]>;
  def R12 : GPRReg<12, "r12">,  DwarfRegNum<[12]>;
  def R13 : GPRReg<13, "r13">,  DwarfRegNum<[13]>;
  def R14 : GPRReg<14, "r14">,  DwarfRegNum<[14]>;
  def R15 : GPRReg<15, "r15">,  DwarfRegNum<[15]>;

  // Control Registers
  def SR  : CtrlReg<0, "sr">;   // Status Register
  def GBR : CtrlReg<0, "gbr">;  // Global Base Register
  def VBR : CtrlReg<0, "vbr">;  // Vector Base Register
  def SGR : CtrlReg<0, "sgr">;  // Saved General Register
  def DBR : CtrlReg<0, "dbr">;  // Debug Base Register
  def SSR : CtrlReg<0, "ssr">;  // Saved Status Register
  def SPC : CtrlReg<0, "spc">;  // Saved Program Counter

  // System Registers
  def PR    : SysReg<0, "pr">;    // Procedure Register
  def PC    : SysReg<0, "pc">;    // Program Counter
  def MACL  : SysReg<0, "macl">;  // Mult & Accum Low
  def MACH  : SysReg<1, "mach">;  // Mult & Accum Hi
  def FPSCR : SysReg<0, "fpscr">; // FPU Status/Control Register
  def FPUL  : SysReg<0, "fpul">;  // FPU Comms Register

  // 32-bit floating point registers
  foreach I = 0-15 in
    def FR#I : FRReg<I, "fr"#I>, DwarfRegNum<[!add(I, 16)]>;
  foreach I = 0-15 in
    def XF#I : FRReg<!add(I, 16), "xf"#I>, DwarfRegNum<[!add(I, 32)]>;

  // 64-bit floating point registers
  foreach I = 0-7 in
    def DR#!shl(I, 1) : DRReg<!shl(I, 1), "dr"#!shl(I, 1),
    [!cast<FRReg>("FR"#!shl(I, 1)),
     !cast<FRReg>("FR"#!add(!shl(I, 1), 1))]>;
  foreach I = 0-7 in
    def XD#!shl(I, 1) : DRReg<!shl(I, 1), "xd"#!shl(I, 1),
    [!cast<FRReg>("XF"#!shl(I, 1)),
     !cast<FRReg>("XF"#!add(!shl(I, 1), 1))]>;

  // 128-bit floating point vector registers
  foreach I = 0-3 in
    def FV#!shl(I, 2) : FVReg<!shl(I, 2), "fv"#!shl(I, 2),
    [!cast<FRReg>("FR"#!shl(I, 2)),
     !cast<FRReg>("FR"#!add(!shl(I, 2), 1)),
     !cast<FRReg>("FR"#!add(!shl(I, 2), 2)),
     !cast<FRReg>("XF"#!add(!shl(I, 2), 3))]>;
  
  // 512-bit matrix register (shadows XF registers)
  // TODO: Add proper subregisters here.
  def XMTRX : MTRXReg<0, "xmtrx">;
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

def GPR : RegisterClass<"SH", [ i32 ], 32, (add
  R0, R1, R2,  R3,  R4,  R5,  R6,  R7,  // Banked memory
  R8, R9, R10, R11, R12, R13, R14, R15  // Non-banked memory.
)>;

// 32-bit floating point registers.
def FR32 : RegisterClass<"SH", [f32], 32, (add
  // FR
  FR0,  FR1,  FR2,  FR3,  FR4,  FR5,  FR6,  FR7,
  FR8,  FR9, FR10, FR11, FR12, FR13, FR14, FR15,

  // XF
  XF0,  XF1,  XF2,  XF3,  XF4,  XF5,  XF6,  XF7,
  XF8,  XF9, XF10, XF11, XF12, XF13, XF14, XF15
)>;

// 64-bit floating point registers.
def FR64 : RegisterClass<"SH", [f64], 64, (add
  // DR
  DR0, DR2, DR4, DR6, DR8, DR10, DR12, DR14,

  // XD
  XD0, XD2, XD4, XD6, XD8, XD10, XD12, XD14
)>;

// Vector registers
def VEC128 : RegisterClass<"SH", [v4f32], 128, (add FV0, FV4, FV8, FV12)>;

// Matrix register
def XMTRX512 : RegisterClass<"SH", [v16f32], 512, (add (sequence "FR%u", 0, 15), (sequence "XF%u", 0, 15))>;

// Control registers
def CTRL : RegisterClass<"SH", [ i32 ], 32, 
  (add SR, GBR, VBR, SGR, DBR, SSR, SPC)>, Unallocatable;

// System registers.
def SYS : RegisterClass<"SH", [ i32 ], 32, 
  (add PR, PC, MACL, MACH, FPSCR, FPUL)>, Unallocatable;