//===-- Nios2RegisterInfo.td - Nios2 Register defs ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// We have bank of 32 registers.
class Nios2Reg<string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "Nios2";
}

// Nios2 CPU Registers
class Nios2GPRReg<bits<5> num, string n> : Nios2Reg<n> {
  let Num = num;
}

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

let Namespace = "Nios2" in {
  // General Purpose Registers
  def ZERO : Nios2GPRReg<0, "zero">, DwarfRegNum<[ 0 ]>;
  def AT : Nios2GPRReg<1, "at">, DwarfRegNum<[ 1 ]>;
  foreach RegNum = 2 - 23 in {
    def R #RegNum : Nios2GPRReg<RegNum, "r" #RegNum>, DwarfRegNum<[ RegNum ]>;
  }
  def ET : Nios2GPRReg<24, "et">, DwarfRegNum<[ 24 ]>;
  def BT : Nios2GPRReg<25, "bt">, DwarfRegNum<[ 25 ]>;
  def GP : Nios2GPRReg<26, "gp">, DwarfRegNum<[ 26 ]>;
  def SP : Nios2GPRReg<27, "sp">, DwarfRegNum<[ 27 ]>;
  def FP : Nios2GPRReg<28, "fp">, DwarfRegNum<[ 28 ]>;
  def EA : Nios2GPRReg<29, "ea">, DwarfRegNum<[ 29 ]>;
  def BA : Nios2GPRReg<30, "ba">, DwarfRegNum<[ 30 ]>;
  def RA : Nios2GPRReg<31, "ra">, DwarfRegNum<[ 31 ]>;
  def PC : Nios2Reg<"pc">, DwarfRegNum<[ 32 ]>;
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

def CPURegs : RegisterClass<"Nios2", [ i32 ], 32,
                            (add
                            // Reserved
                            ZERO,
                            AT,
                            // Return Values and Arguments
                            (sequence "R%u", 2, 7),
                            // Not preserved across procedure calls
                            // Caller saved
                            (sequence "R%u", 8, 15),
                            // Callee saved
                            (sequence "R%u", 16, 23),
                            // Reserved
                            ET, BT, GP, SP, FP, EA, BA, RA, PC)>;
