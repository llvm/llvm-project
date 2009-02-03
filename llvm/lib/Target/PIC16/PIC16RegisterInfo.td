//===- PIC16RegisterInfo.td - PIC16 Register defs ------------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the PIC16 register file
//===----------------------------------------------------------------------===//

class PIC16Reg<string n> : Register<n> {
  let Namespace = "PIC16";
}

// PIC16 Registers.
def W   : PIC16Reg<"W">;
def FSR0   : PIC16Reg<"FSR0">;
def FSR1   : PIC16Reg<"FSR1">;
def BS     : PIC16Reg<"BS">;

def STATUS : PIC16Reg<"STATUS">;

// PIC16 Register classes.
def GPR    : RegisterClass<"PIC16", [i8], 8, [W]>;
def FSR16  : RegisterClass<"PIC16", [i16], 8, [FSR0, FSR1]>;
def BSR  : RegisterClass<"PIC16", [i8], 8, [BS]>;

def STATUSR: RegisterClass<"PIC16", [i8], 8, [STATUS]>;

