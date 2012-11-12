//===-- PPCBaseInfo.h - Top level definitions for PPC -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the PPC target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef PPCBASEINFO_H
#define PPCBASEINFO_H

#include "PPCMCTargetDesc.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// getPPCRegisterNumbering - Given the enum value for some register, e.g.
/// PPC::F14, return the number that it corresponds to (e.g. 14).
inline static unsigned getPPCRegisterNumbering(unsigned RegEnum) {
  using namespace PPC;
  switch (RegEnum) {
  case 0: return 0;
  case R0 :  case X0 :  case F0 :  case V0 : case CR0:  case CR0LT: return  0;
  case R1 :  case X1 :  case F1 :  case V1 : case CR1:  case CR0GT: return  1;
  case R2 :  case X2 :  case F2 :  case V2 : case CR2:  case CR0EQ: return  2;
  case R3 :  case X3 :  case F3 :  case V3 : case CR3:  case CR0UN: return  3;
  case R4 :  case X4 :  case F4 :  case V4 : case CR4:  case CR1LT: return  4;
  case R5 :  case X5 :  case F5 :  case V5 : case CR5:  case CR1GT: return  5;
  case R6 :  case X6 :  case F6 :  case V6 : case CR6:  case CR1EQ: return  6;
  case R7 :  case X7 :  case F7 :  case V7 : case CR7:  case CR1UN: return  7;
  case R8 :  case X8 :  case F8 :  case V8 : case CR2LT: return  8;
  case R9 :  case X9 :  case F9 :  case V9 : case CR2GT: return  9;
  case R10:  case X10:  case F10:  case V10: case CR2EQ: return 10;
  case R11:  case X11:  case F11:  case V11: case CR2UN: return 11;
  case R12:  case X12:  case F12:  case V12: case CR3LT: return 12;
  case R13:  case X13:  case F13:  case V13: case CR3GT: return 13;
  case R14:  case X14:  case F14:  case V14: case CR3EQ: return 14;
  case R15:  case X15:  case F15:  case V15: case CR3UN: return 15;
  case R16:  case X16:  case F16:  case V16: case CR4LT: return 16;
  case R17:  case X17:  case F17:  case V17: case CR4GT: return 17;
  case R18:  case X18:  case F18:  case V18: case CR4EQ: return 18;
  case R19:  case X19:  case F19:  case V19: case CR4UN: return 19;
  case R20:  case X20:  case F20:  case V20: case CR5LT: return 20;
  case R21:  case X21:  case F21:  case V21: case CR5GT: return 21;
  case R22:  case X22:  case F22:  case V22: case CR5EQ: return 22;
  case R23:  case X23:  case F23:  case V23: case CR5UN: return 23;
  case R24:  case X24:  case F24:  case V24: case CR6LT: return 24;
  case R25:  case X25:  case F25:  case V25: case CR6GT: return 25;
  case R26:  case X26:  case F26:  case V26: case CR6EQ: return 26;
  case R27:  case X27:  case F27:  case V27: case CR6UN: return 27;
  case R28:  case X28:  case F28:  case V28: case CR7LT: return 28;
  case R29:  case X29:  case F29:  case V29: case CR7GT: return 29;
  case R30:  case X30:  case F30:  case V30: case CR7EQ: return 30;
  case R31:  case X31:  case F31:  case V31: case CR7UN: return 31;
  default:
    llvm_unreachable("Unhandled reg in PPCRegisterInfo::getRegisterNumbering!");
  }
}

} // end namespace llvm;

#endif
