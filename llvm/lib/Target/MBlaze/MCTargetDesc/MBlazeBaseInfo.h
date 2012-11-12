//===-- MBlazeBaseInfo.h - Top level definitions for MBlaze -- --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the MBlaze target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef MBlazeBASEINFO_H
#define MBlazeBASEINFO_H

#include "MBlazeMCTargetDesc.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// MBlazeII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace MBlazeII {
  enum {
    // PseudoFrm - This represents an instruction that is a pseudo instruction
    // or one that has not been implemented yet.  It is illegal to code generate
    // it, but tolerated for intermediate implementation stages.
    FPseudo = 0,
    FRRR,
    FRRI,
    FCRR,
    FCRI,
    FRCR,
    FRCI,
    FCCR,
    FCCI,
    FRRCI,
    FRRC,
    FRCX,
    FRCS,
    FCRCS,
    FCRCX,
    FCX,
    FCR,
    FRIR,
    FRRRR,
    FRI,
    FC,
    FRR,
    FormMask = 63

    //===------------------------------------------------------------------===//
    // MBlaze Specific MachineOperand flags.
    // MO_NO_FLAG,

    /// MO_GOT - Represents the offset into the global offset table at which
    /// the address the relocation entry symbol resides during execution.
    // MO_GOT,

    /// MO_GOT_CALL - Represents the offset into the global offset table at
    /// which the address of a call site relocation entry symbol resides
    /// during execution. This is different from the above since this flag
    /// can only be present in call instructions.
    // MO_GOT_CALL,

    /// MO_GPREL - Represents the offset from the current gp value to be used
    /// for the relocatable object file being produced.
    // MO_GPREL,

    /// MO_ABS_HILO - Represents the hi or low part of an absolute symbol
    /// address.
    // MO_ABS_HILO

  };
}

static inline bool isMBlazeRegister(unsigned Reg) {
  return Reg <= 31;
}

static inline bool isSpecialMBlazeRegister(unsigned Reg) {
  switch (Reg) {
    case 0x0000 : case 0x0001 : case 0x0003 : case 0x0005 : 
    case 0x0007 : case 0x000B : case 0x000D : case 0x1000 : 
    case 0x1001 : case 0x1002 : case 0x1003 : case 0x1004 : 
    case 0x2000 : case 0x2001 : case 0x2002 : case 0x2003 : 
    case 0x2004 : case 0x2005 : case 0x2006 : case 0x2007 : 
    case 0x2008 : case 0x2009 : case 0x200A : case 0x200B : 
      return true;

    default:
      return false;
  }
}

/// getMBlazeRegisterNumbering - Given the enum value for some register, e.g.
/// MBlaze::R0, return the number that it corresponds to (e.g. 0).
static inline unsigned getMBlazeRegisterNumbering(unsigned RegEnum) {
  switch (RegEnum) {
    case MBlaze::R0     : return 0;
    case MBlaze::R1     : return 1;
    case MBlaze::R2     : return 2;
    case MBlaze::R3     : return 3;
    case MBlaze::R4     : return 4;
    case MBlaze::R5     : return 5;
    case MBlaze::R6     : return 6;
    case MBlaze::R7     : return 7;
    case MBlaze::R8     : return 8;
    case MBlaze::R9     : return 9;
    case MBlaze::R10    : return 10;
    case MBlaze::R11    : return 11;
    case MBlaze::R12    : return 12;
    case MBlaze::R13    : return 13;
    case MBlaze::R14    : return 14;
    case MBlaze::R15    : return 15;
    case MBlaze::R16    : return 16;
    case MBlaze::R17    : return 17;
    case MBlaze::R18    : return 18;
    case MBlaze::R19    : return 19;
    case MBlaze::R20    : return 20;
    case MBlaze::R21    : return 21;
    case MBlaze::R22    : return 22;
    case MBlaze::R23    : return 23;
    case MBlaze::R24    : return 24;
    case MBlaze::R25    : return 25;
    case MBlaze::R26    : return 26;
    case MBlaze::R27    : return 27;
    case MBlaze::R28    : return 28;
    case MBlaze::R29    : return 29;
    case MBlaze::R30    : return 30;
    case MBlaze::R31    : return 31;
    case MBlaze::RPC    : return 0x0000;
    case MBlaze::RMSR   : return 0x0001;
    case MBlaze::REAR   : return 0x0003;
    case MBlaze::RESR   : return 0x0005;
    case MBlaze::RFSR   : return 0x0007;
    case MBlaze::RBTR   : return 0x000B;
    case MBlaze::REDR   : return 0x000D;
    case MBlaze::RPID   : return 0x1000;
    case MBlaze::RZPR   : return 0x1001;
    case MBlaze::RTLBX  : return 0x1002;
    case MBlaze::RTLBLO : return 0x1003;
    case MBlaze::RTLBHI : return 0x1004;
    case MBlaze::RPVR0  : return 0x2000;
    case MBlaze::RPVR1  : return 0x2001;
    case MBlaze::RPVR2  : return 0x2002;
    case MBlaze::RPVR3  : return 0x2003;
    case MBlaze::RPVR4  : return 0x2004;
    case MBlaze::RPVR5  : return 0x2005;
    case MBlaze::RPVR6  : return 0x2006;
    case MBlaze::RPVR7  : return 0x2007;
    case MBlaze::RPVR8  : return 0x2008;
    case MBlaze::RPVR9  : return 0x2009;
    case MBlaze::RPVR10 : return 0x200A;
    case MBlaze::RPVR11 : return 0x200B;
    default: llvm_unreachable("Unknown register number!");
  }
}

/// getRegisterFromNumbering - Given the enum value for some register, e.g.
/// MBlaze::R0, return the number that it corresponds to (e.g. 0).
static inline unsigned getMBlazeRegisterFromNumbering(unsigned Reg) {
  switch (Reg) {
    case 0  : return MBlaze::R0;
    case 1  : return MBlaze::R1;
    case 2  : return MBlaze::R2;
    case 3  : return MBlaze::R3;
    case 4  : return MBlaze::R4;
    case 5  : return MBlaze::R5;
    case 6  : return MBlaze::R6;
    case 7  : return MBlaze::R7;
    case 8  : return MBlaze::R8;
    case 9  : return MBlaze::R9;
    case 10 : return MBlaze::R10;
    case 11 : return MBlaze::R11;
    case 12 : return MBlaze::R12;
    case 13 : return MBlaze::R13;
    case 14 : return MBlaze::R14;
    case 15 : return MBlaze::R15;
    case 16 : return MBlaze::R16;
    case 17 : return MBlaze::R17;
    case 18 : return MBlaze::R18;
    case 19 : return MBlaze::R19;
    case 20 : return MBlaze::R20;
    case 21 : return MBlaze::R21;
    case 22 : return MBlaze::R22;
    case 23 : return MBlaze::R23;
    case 24 : return MBlaze::R24;
    case 25 : return MBlaze::R25;
    case 26 : return MBlaze::R26;
    case 27 : return MBlaze::R27;
    case 28 : return MBlaze::R28;
    case 29 : return MBlaze::R29;
    case 30 : return MBlaze::R30;
    case 31 : return MBlaze::R31;
    default: llvm_unreachable("Unknown register number!");
  }
}

static inline unsigned getSpecialMBlazeRegisterFromNumbering(unsigned Reg) {
  switch (Reg) {
    case 0x0000 : return MBlaze::RPC;
    case 0x0001 : return MBlaze::RMSR;
    case 0x0003 : return MBlaze::REAR;
    case 0x0005 : return MBlaze::RESR;
    case 0x0007 : return MBlaze::RFSR;
    case 0x000B : return MBlaze::RBTR;
    case 0x000D : return MBlaze::REDR;
    case 0x1000 : return MBlaze::RPID;
    case 0x1001 : return MBlaze::RZPR;
    case 0x1002 : return MBlaze::RTLBX;
    case 0x1003 : return MBlaze::RTLBLO;
    case 0x1004 : return MBlaze::RTLBHI;
    case 0x2000 : return MBlaze::RPVR0;
    case 0x2001 : return MBlaze::RPVR1;
    case 0x2002 : return MBlaze::RPVR2;
    case 0x2003 : return MBlaze::RPVR3;
    case 0x2004 : return MBlaze::RPVR4;
    case 0x2005 : return MBlaze::RPVR5;
    case 0x2006 : return MBlaze::RPVR6;
    case 0x2007 : return MBlaze::RPVR7;
    case 0x2008 : return MBlaze::RPVR8;
    case 0x2009 : return MBlaze::RPVR9;
    case 0x200A : return MBlaze::RPVR10;
    case 0x200B : return MBlaze::RPVR11;
    default: llvm_unreachable("Unknown register number!");
  }
}

} // end namespace llvm;

#endif
