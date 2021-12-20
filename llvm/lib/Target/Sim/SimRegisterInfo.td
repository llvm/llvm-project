//===-- SimRegisterInfo.td - Sim Register defs -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the Sim register file
//===----------------------------------------------------------------------===//

// For tablegen(... -gen-emitter)  in CMakeLists.txt
class SimReg<bits<16> Enc, string n> : Register<n> {
  let HWEncoding = Enc;
}

// Sim CPU Registers
class SimGPRReg<bits<16> Enc, string n> : SimReg<Enc, n>;

//===----------------------------------------------------------------------===//
//@Registers
//===----------------------------------------------------------------------===//
//@ All registers definition
let Namespace = "Sim" in {
  //@ General Purpose Registers
  def R0    : SimGPRReg<0,  "r0">,    DwarfRegNum<[0]>;
  def R1    : SimGPRReg<1,  "r1">,    DwarfRegNum<[1]>;
  def R2    : SimGPRReg<2,  "r2">,    DwarfRegNum<[2]>;
  def R3    : SimGPRReg<3,  "r3">,    DwarfRegNum<[3]>;
  def R4    : SimGPRReg<4,  "r4">,    DwarfRegNum<[4]>;
  def R5    : SimGPRReg<5,  "r5">,    DwarfRegNum<[5]>;
  def R6    : SimGPRReg<6,  "r6">,    DwarfRegNum<[6]>;
  def R7    : SimGPRReg<7,  "r7">,    DwarfRegNum<[7]>;
  def R8    : SimGPRReg<8,  "r8">,    DwarfRegNum<[8]>;
  def R9    : SimGPRReg<9,  "r9">,    DwarfRegNum<[9]>;
  def R10   : SimGPRReg<10, "r10">,   DwarfRegNum<[10]>;
  def R11   : SimGPRReg<11, "r11">,   DwarfRegNum<[11]>;
  def R12   : SimGPRReg<12, "r12">,   DwarfRegNum<[12]>;
  def R13   : SimGPRReg<13, "r13">,   DwarfRegNum<[13]>;
  def R14   : SimGPRReg<14, "r14">,   DwarfRegNum<[14]>;
  def R15   : SimGPRReg<15, "r15">,   DwarfRegNum<[15]>;
}

//===----------------------------------------------------------------------===//
//@Register Classes
//===----------------------------------------------------------------------===//

def CPURegs : RegisterClass<"Sim", [i32], 32, (add
  R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15)>;