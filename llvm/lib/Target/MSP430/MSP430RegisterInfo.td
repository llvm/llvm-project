//===-- MSP430RegisterInfo.td - MSP430 Register defs -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the MSP430 register file
//===----------------------------------------------------------------------===//

class MSP430Reg<bits<4> num, string n> : Register<n> {
  field bits<4> Num = num;
  let Namespace = "MSP430";
}

class MSP430RegWithSubregs<bits<4> num, string n, list<Register> subregs> 
  : RegisterWithSubRegs<n, subregs> {
  field bits<4> Num = num;
  let Namespace = "MSP430";
}

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

def PCB  : MSP430Reg<0,  "r0">;
def SPB  : MSP430Reg<1,  "r1">;
def SRB  : MSP430Reg<2,  "r2">;
def CGB  : MSP430Reg<3,  "r3">;
def FPB  : MSP430Reg<4,  "r4">;
def R5B  : MSP430Reg<5,  "r5">;
def R6B  : MSP430Reg<6,  "r6">;
def R7B  : MSP430Reg<7,  "r7">;
def R8B  : MSP430Reg<8,  "r8">;
def R9B  : MSP430Reg<9,  "r9">;
def R10B : MSP430Reg<10, "r10">;
def R11B : MSP430Reg<11, "r11">;
def R12B : MSP430Reg<12, "r12">;
def R13B : MSP430Reg<13, "r13">;
def R14B : MSP430Reg<14, "r14">;
def R15B : MSP430Reg<15, "r15">;

def subreg_8bit : SubRegIndex { let Namespace = "MSP430"; }

let SubRegIndices = [subreg_8bit] in {
def PCW  : MSP430RegWithSubregs<0,  "r0",  [PCB]>;
def SPW  : MSP430RegWithSubregs<1,  "r1",  [SPB]>;
def SRW  : MSP430RegWithSubregs<2,  "r2",  [SRB]>;
def CGW  : MSP430RegWithSubregs<3,  "r3",  [CGB]>;
def FPW  : MSP430RegWithSubregs<4,  "r4",  [FPB]>;
def R5W  : MSP430RegWithSubregs<5,  "r5",  [R5B]>;
def R6W  : MSP430RegWithSubregs<6,  "r6",  [R6B]>;
def R7W  : MSP430RegWithSubregs<7,  "r7",  [R7B]>;
def R8W  : MSP430RegWithSubregs<8,  "r8",  [R8B]>;
def R9W  : MSP430RegWithSubregs<9,  "r9",  [R9B]>;
def R10W : MSP430RegWithSubregs<10, "r10", [R10B]>;
def R11W : MSP430RegWithSubregs<11, "r11", [R11B]>;
def R12W : MSP430RegWithSubregs<12, "r12", [R12B]>;
def R13W : MSP430RegWithSubregs<13, "r13", [R13B]>;
def R14W : MSP430RegWithSubregs<14, "r14", [R14B]>;
def R15W : MSP430RegWithSubregs<15, "r15", [R15B]>;
}

def GR8 : RegisterClass<"MSP430", [i8], 8,
   // Volatile registers
  (add R12B, R13B, R14B, R15B, R11B, R10B, R9B, R8B, R7B, R6B, R5B,
   // Frame pointer, sometimes allocable
   FPB,
   // Volatile, but not allocable
   PCB, SPB, SRB, CGB)>;

def GR16 : RegisterClass<"MSP430", [i16], 16,
   // Volatile registers
  (add R12W, R13W, R14W, R15W, R11W, R10W, R9W, R8W, R7W, R6W, R5W,
   // Frame pointer, sometimes allocable
   FPW,
   // Volatile, but not allocable
   PCW, SPW, SRW, CGW)>;
