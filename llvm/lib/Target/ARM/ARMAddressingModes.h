//===- ARMAddressingModes.h - ARM Addressing Modes --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM addressing mode implementation stuff.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ARM_ARMADDRESSINGMODES_H
#define LLVM_TARGET_ARM_ARMADDRESSINGMODES_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

namespace llvm {
  
/// ARM_AM - ARM Addressing Mode Stuff
namespace ARM_AM {
  enum ShiftOpc {
    no_shift = 0,
    asr,
    lsl,
    lsr,
    ror,
    rrx
  };
  
  enum AddrOpc {
    add = '+', sub = '-'
  };
  
  static inline const char *getShiftOpcStr(ShiftOpc Op) {
    switch (Op) {
    default: assert(0 && "Unknown shift opc!");
    case ARM_AM::asr: return "asr";
    case ARM_AM::lsl: return "lsl";
    case ARM_AM::lsr: return "lsr";
    case ARM_AM::ror: return "ror";
    case ARM_AM::rrx: return "rrx";
    }
  }
  
  static inline ShiftOpc getShiftOpcForNode(SDValue N) {
    switch (N.getOpcode()) {
    default:          return ARM_AM::no_shift;
    case ISD::SHL:    return ARM_AM::lsl;
    case ISD::SRL:    return ARM_AM::lsr;
    case ISD::SRA:    return ARM_AM::asr;
    case ISD::ROTR:   return ARM_AM::ror;
    //case ISD::ROTL:  // Only if imm -> turn into ROTR.
    // Can't handle RRX here, because it would require folding a flag into
    // the addressing mode.  :(  This causes us to miss certain things.
    //case ARMISD::RRX: return ARM_AM::rrx;
    }
  }

  enum AMSubMode {
    bad_am_submode = 0,
    ia,
    ib,
    da,
    db
  };

  static inline const char *getAMSubModeStr(AMSubMode Mode) {
    switch (Mode) {
    default: assert(0 && "Unknown addressing sub-mode!");
    case ARM_AM::ia: return "ia";
    case ARM_AM::ib: return "ib";
    case ARM_AM::da: return "da";
    case ARM_AM::db: return "db";
    }
  }

  static inline const char *getAMSubModeAltStr(AMSubMode Mode, bool isLD) {
    switch (Mode) {
    default: assert(0 && "Unknown addressing sub-mode!");
    case ARM_AM::ia: return isLD ? "fd" : "ea";
    case ARM_AM::ib: return isLD ? "ed" : "fa";
    case ARM_AM::da: return isLD ? "fa" : "ed";
    case ARM_AM::db: return isLD ? "ea" : "fd";
    }
  }

  /// rotr32 - Rotate a 32-bit unsigned value right by a specified # bits.
  ///
  static inline unsigned rotr32(unsigned Val, unsigned Amt) {
    assert(Amt < 32 && "Invalid rotate amount");
    return (Val >> Amt) | (Val << ((32-Amt)&31));
  }
  
  /// rotl32 - Rotate a 32-bit unsigned value left by a specified # bits.
  ///
  static inline unsigned rotl32(unsigned Val, unsigned Amt) {
    assert(Amt < 32 && "Invalid rotate amount");
    return (Val << Amt) | (Val >> ((32-Amt)&31));
  }
  
  //===--------------------------------------------------------------------===//
  // Addressing Mode #1: shift_operand with registers
  //===--------------------------------------------------------------------===//
  //
  // This 'addressing mode' is used for arithmetic instructions.  It can
  // represent things like:
  //   reg
  //   reg [asr|lsl|lsr|ror|rrx] reg
  //   reg [asr|lsl|lsr|ror|rrx] imm
  //
  // This is stored three operands [rega, regb, opc].  The first is the base
  // reg, the second is the shift amount (or reg0 if not present or imm).  The
  // third operand encodes the shift opcode and the imm if a reg isn't present.
  //
  static inline unsigned getSORegOpc(ShiftOpc ShOp, unsigned Imm) {
    return ShOp | (Imm << 3);
  }
  static inline unsigned getSORegOffset(unsigned Op) {
    return Op >> 3;
  }
  static inline ShiftOpc getSORegShOp(unsigned Op) {
    return (ShiftOpc)(Op & 7);
  }

  /// getSOImmValImm - Given an encoded imm field for the reg/imm form, return
  /// the 8-bit imm value.
  static inline unsigned getSOImmValImm(unsigned Imm) {
    return Imm & 0xFF;
  }
  /// getSOImmValRotate - Given an encoded imm field for the reg/imm form, return
  /// the rotate amount.
  static inline unsigned getSOImmValRot(unsigned Imm) {
    return (Imm >> 8) * 2;
  }
  
  /// getSOImmValRotate - Try to handle Imm with an immediate shifter operand,
  /// computing the rotate amount to use.  If this immediate value cannot be
  /// handled with a single shifter-op, determine a good rotate amount that will
  /// take a maximal chunk of bits out of the immediate.
  static inline unsigned getSOImmValRotate(unsigned Imm) {
    // 8-bit (or less) immediates are trivially shifter_operands with a rotate
    // of zero.
    if ((Imm & ~255U) == 0) return 0;
    
    // Use CTZ to compute the rotate amount.
    unsigned TZ = CountTrailingZeros_32(Imm);
    
    // Rotate amount must be even.  Something like 0x200 must be rotated 8 bits,
    // not 9.
    unsigned RotAmt = TZ & ~1;
    
    // If we can handle this spread, return it.
    if ((rotr32(Imm, RotAmt) & ~255U) == 0)
      return (32-RotAmt)&31;  // HW rotates right, not left.

    // For values like 0xF000000F, we should skip the first run of ones, then
    // retry the hunt.
    if (Imm & 1) {
      unsigned TrailingOnes = CountTrailingZeros_32(~Imm);
      if (TrailingOnes != 32) {  // Avoid overflow on 0xFFFFFFFF
        // Restart the search for a high-order bit after the initial seconds of
        // ones.
        unsigned TZ2 = CountTrailingZeros_32(Imm & ~((1 << TrailingOnes)-1));
      
        // Rotate amount must be even.
        unsigned RotAmt2 = TZ2 & ~1;
        
        // If this fits, use it.
        if (RotAmt2 != 32 && (rotr32(Imm, RotAmt2) & ~255U) == 0)
          return (32-RotAmt2)&31;  // HW rotates right, not left.
      }
    }
    
    // Otherwise, we have no way to cover this span of bits with a single
    // shifter_op immediate.  Return a chunk of bits that will be useful to
    // handle.
    return (32-RotAmt)&31;  // HW rotates right, not left.
  }

  /// getSOImmVal - Given a 32-bit immediate, if it is something that can fit
  /// into an shifter_operand immediate operand, return the 12-bit encoding for
  /// it.  If not, return -1.
  static inline int getSOImmVal(unsigned Arg) {
    // 8-bit (or less) immediates are trivially shifter_operands with a rotate
    // of zero.
    if ((Arg & ~255U) == 0) return Arg;
    
    unsigned RotAmt = getSOImmValRotate(Arg);

    // If this cannot be handled with a single shifter_op, bail out.
    if (rotr32(~255U, RotAmt) & Arg)
      return -1;
      
    // Encode this correctly.
    return rotl32(Arg, RotAmt) | ((RotAmt>>1) << 8);
  }
  
  /// isSOImmTwoPartVal - Return true if the specified value can be obtained by
  /// or'ing together two SOImmVal's.
  static inline bool isSOImmTwoPartVal(unsigned V) {
    // If this can be handled with a single shifter_op, bail out.
    V = rotr32(~255U, getSOImmValRotate(V)) & V;
    if (V == 0)
      return false;
    
    // If this can be handled with two shifter_op's, accept.
    V = rotr32(~255U, getSOImmValRotate(V)) & V;
    return V == 0;
  }
  
  /// getSOImmTwoPartFirst - If V is a value that satisfies isSOImmTwoPartVal,
  /// return the first chunk of it.
  static inline unsigned getSOImmTwoPartFirst(unsigned V) {
    return rotr32(255U, getSOImmValRotate(V)) & V;
  }

  /// getSOImmTwoPartSecond - If V is a value that satisfies isSOImmTwoPartVal,
  /// return the second chunk of it.
  static inline unsigned getSOImmTwoPartSecond(unsigned V) {
    // Mask out the first hunk.  
    V = rotr32(~255U, getSOImmValRotate(V)) & V;
    
    // Take what's left.
    assert(V == (rotr32(255U, getSOImmValRotate(V)) & V));
    return V;
  }
  
  /// getThumbImmValShift - Try to handle Imm with a 8-bit immediate followed
  /// by a left shift. Returns the shift amount to use.
  static inline unsigned getThumbImmValShift(unsigned Imm) {
    // 8-bit (or less) immediates are trivially immediate operand with a shift
    // of zero.
    if ((Imm & ~255U) == 0) return 0;

    // Use CTZ to compute the shift amount.
    return CountTrailingZeros_32(Imm);
  }

  /// isThumbImmShiftedVal - Return true if the specified value can be obtained
  /// by left shifting a 8-bit immediate.
  static inline bool isThumbImmShiftedVal(unsigned V) {
    // If this can be handled with 
    V = (~255U << getThumbImmValShift(V)) & V;
    return V == 0;
  }

  /// getThumbImmNonShiftedVal - If V is a value that satisfies
  /// isThumbImmShiftedVal, return the non-shiftd value.
  static inline unsigned getThumbImmNonShiftedVal(unsigned V) {
    return V >> getThumbImmValShift(V);
  }

  //===--------------------------------------------------------------------===//
  // Addressing Mode #2
  //===--------------------------------------------------------------------===//
  //
  // This is used for most simple load/store instructions.
  //
  // addrmode2 := reg +/- reg shop imm
  // addrmode2 := reg +/- imm12
  //
  // The first operand is always a Reg.  The second operand is a reg if in
  // reg/reg form, otherwise it's reg#0.  The third field encodes the operation
  // in bit 12, the immediate in bits 0-11, and the shift op in 13-15.
  //
  // If this addressing mode is a frame index (before prolog/epilog insertion
  // and code rewriting), this operand will have the form:  FI#, reg0, <offs>
  // with no shift amount for the frame offset.
  // 
  static inline unsigned getAM2Opc(AddrOpc Opc, unsigned Imm12, ShiftOpc SO) {
    assert(Imm12 < (1 << 12) && "Imm too large!");
    bool isSub = Opc == sub;
    return Imm12 | ((int)isSub << 12) | (SO << 13);
  }
  static inline unsigned getAM2Offset(unsigned AM2Opc) {
    return AM2Opc & ((1 << 12)-1);
  }
  static inline AddrOpc getAM2Op(unsigned AM2Opc) {
    return ((AM2Opc >> 12) & 1) ? sub : add;
  }
  static inline ShiftOpc getAM2ShiftOpc(unsigned AM2Opc) {
    return (ShiftOpc)(AM2Opc >> 13);
  }
  
  
  //===--------------------------------------------------------------------===//
  // Addressing Mode #3
  //===--------------------------------------------------------------------===//
  //
  // This is used for sign-extending loads, and load/store-pair instructions.
  //
  // addrmode3 := reg +/- reg
  // addrmode3 := reg +/- imm8
  //
  // The first operand is always a Reg.  The second operand is a reg if in
  // reg/reg form, otherwise it's reg#0.  The third field encodes the operation
  // in bit 8, the immediate in bits 0-7.
  
  /// getAM3Opc - This function encodes the addrmode3 opc field.
  static inline unsigned getAM3Opc(AddrOpc Opc, unsigned char Offset) {
    bool isSub = Opc == sub;
    return ((int)isSub << 8) | Offset;
  }
  static inline unsigned char getAM3Offset(unsigned AM3Opc) {
    return AM3Opc & 0xFF;
  }
  static inline AddrOpc getAM3Op(unsigned AM3Opc) {
    return ((AM3Opc >> 8) & 1) ? sub : add;
  }
  
  //===--------------------------------------------------------------------===//
  // Addressing Mode #4
  //===--------------------------------------------------------------------===//
  //
  // This is used for load / store multiple instructions.
  //
  // addrmode4 := reg, <mode>
  //
  // The four modes are:
  //    IA - Increment after
  //    IB - Increment before
  //    DA - Decrement after
  //    DB - Decrement before
  //
  // If the 4th bit (writeback)is set, then the base register is updated after
  // the memory transfer.

  static inline AMSubMode getAM4SubMode(unsigned Mode) {
    return (AMSubMode)(Mode & 0x7);
  }

  static inline unsigned getAM4ModeImm(AMSubMode SubMode, bool WB = false) {
    return (int)SubMode | ((int)WB << 3);
  }

  static inline bool getAM4WBFlag(unsigned Mode) {
    return (Mode >> 3) & 1;
  }

  //===--------------------------------------------------------------------===//
  // Addressing Mode #5
  //===--------------------------------------------------------------------===//
  //
  // This is used for coprocessor instructions, such as FP load/stores.
  //
  // addrmode5 := reg +/- imm8*4
  //
  // The first operand is always a Reg.  The third field encodes the operation
  // in bit 8, the immediate in bits 0-7.
  //
  // This can also be used for FP load/store multiple ops. The third field encodes
  // writeback mode in bit 8, the number of registers (or 2 times the number of
  // registers for DPR ops) in bits 0-7. In addition, bit 9-11 encodes one of the
  // following two sub-modes:
  //
  //    IA - Increment after
  //    DB - Decrement before
  
  /// getAM5Opc - This function encodes the addrmode5 opc field.
  static inline unsigned getAM5Opc(AddrOpc Opc, unsigned char Offset) {
    bool isSub = Opc == sub;
    return ((int)isSub << 8) | Offset;
  }
  static inline unsigned char getAM5Offset(unsigned AM5Opc) {
    return AM5Opc & 0xFF;
  }
  static inline AddrOpc getAM5Op(unsigned AM5Opc) {
    return ((AM5Opc >> 8) & 1) ? sub : add;
  }

  /// getAM5Opc - This function encodes the addrmode5 opc field for FLDM and
  /// FSTM instructions.
  static inline unsigned getAM5Opc(AMSubMode SubMode, bool WB,
                                   unsigned char Offset) {
    assert((SubMode == ia || SubMode == db) &&
           "Illegal addressing mode 5 sub-mode!");
    return ((int)SubMode << 9) | ((int)WB << 8) | Offset;
  }
  static inline AMSubMode getAM5SubMode(unsigned AM5Opc) {
    return (AMSubMode)((AM5Opc >> 9) & 0x7);
  }
  static inline bool getAM5WBFlag(unsigned AM5Opc) {
    return ((AM5Opc >> 8) & 1);
  }
  
} // end namespace ARM_AM
} // end namespace llvm

#endif

