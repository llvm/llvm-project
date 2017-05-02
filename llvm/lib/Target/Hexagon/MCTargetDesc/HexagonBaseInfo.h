//===-- HexagonBaseInfo.h - Top level definitions for Hexagon --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Hexagon target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONBASEINFO_H
#define LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONBASEINFO_H

#include "HexagonDepITypes.h"
#include "HexagonMCTargetDesc.h"
#include "llvm/Support/ErrorHandling.h"
#include <stdint.h>

namespace llvm {

/// HexagonII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace HexagonII {
  unsigned const TypeCVI_FIRST = TypeCVI_HIST;
  unsigned const TypeCVI_LAST = TypeCVI_VX_DV;

  enum SubTarget {
    HasV4SubT     = 0x3f,
    HasV5SubT     = 0x3e,
    HasV55SubT    = 0x3c,
    HasV60SubT    = 0x38,
  };

  enum AddrMode {
    NoAddrMode     = 0,  // No addressing mode
    Absolute       = 1,  // Absolute addressing mode
    AbsoluteSet    = 2,  // Absolute set addressing mode
    BaseImmOffset  = 3,  // Indirect with offset
    BaseLongOffset = 4,  // Indirect with long offset
    BaseRegOffset  = 5,  // Indirect with register offset
    PostInc        = 6   // Post increment addressing mode
  };

  // MemAccessSize is represented as 1+log2(N) where N is size in bits.
  enum class MemAccessSize {
    NoMemAccess = 0,            // Not a memory access instruction.
    ByteAccess = 1,             // Byte access instruction (memb).
    HalfWordAccess = 2,         // Half word access instruction (memh).
    WordAccess = 3,             // Word access instruction (memw).
    DoubleWordAccess = 4,       // Double word access instruction (memd)
                    // 5,       // We do not have a 16 byte vector access.
    Vector64Access = 7,         // 64 Byte vector access instruction (vmem).
    Vector128Access = 8         // 128 Byte vector access instruction (vmem).
  };

  // MCInstrDesc TSFlags
  // *** Must match HexagonInstrFormat*.td ***
  enum {
    // This 5-bit field describes the insn type.
    TypePos  = 0,
    TypeMask = 0x3f,

    // Solo instructions.
    SoloPos  = 6,
    SoloMask = 0x1,
    // Packed only with A or X-type instructions.
    SoloAXPos  = 7,
    SoloAXMask = 0x1,
    // Only A-type instruction in first slot or nothing.
    SoloAin1Pos  = 8,
    SoloAin1Mask = 0x1,

    // Predicated instructions.
    PredicatedPos  = 9,
    PredicatedMask = 0x1,
    PredicatedFalsePos  = 10,
    PredicatedFalseMask = 0x1,
    PredicatedNewPos  = 11,
    PredicatedNewMask = 0x1,
    PredicateLatePos  = 12,
    PredicateLateMask = 0x1,

    // New-Value consumer instructions.
    NewValuePos  = 13,
    NewValueMask = 0x1,
    // New-Value producer instructions.
    hasNewValuePos  = 14,
    hasNewValueMask = 0x1,
    // Which operand consumes or produces a new value.
    NewValueOpPos  = 15,
    NewValueOpMask = 0x7,
    // Stores that can become new-value stores.
    mayNVStorePos  = 18,
    mayNVStoreMask = 0x1,
    // New-value store instructions.
    NVStorePos  = 19,
    NVStoreMask = 0x1,
    // Loads that can become current-value loads.
    mayCVLoadPos  = 20,
    mayCVLoadMask = 0x1,
    // Current-value load instructions.
    CVLoadPos  = 21,
    CVLoadMask = 0x1,

    // Extendable insns.
    ExtendablePos  = 22,
    ExtendableMask = 0x1,
    // Insns must be extended.
    ExtendedPos  = 23,
    ExtendedMask = 0x1,
    // Which operand may be extended.
    ExtendableOpPos  = 24,
    ExtendableOpMask = 0x7,
    // Signed or unsigned range.
    ExtentSignedPos  = 27,
    ExtentSignedMask = 0x1,
    // Number of bits of range before extending operand.
    ExtentBitsPos  = 28,
    ExtentBitsMask = 0x1f,
    // Alignment power-of-two before extending operand.
    ExtentAlignPos  = 33,
    ExtentAlignMask = 0x3,

    // Addressing mode for load/store instructions.
    AddrModePos  = 41,
    AddrModeMask = 0x7,
    // Access size for load/store instructions.
    MemAccessSizePos = 44,
    MemAccesSizeMask = 0xf,

    // Branch predicted taken.
    TakenPos = 48,
    TakenMask = 0x1,

    // Floating-point instructions.
    FPPos  = 49,
    FPMask = 0x1,

    // New-Value producer-2 instructions.
    hasNewValuePos2  = 51,
    hasNewValueMask2 = 0x1,
    // Which operand consumes or produces a new value.
    NewValueOpPos2  = 52,
    NewValueOpMask2 = 0x7,

    // Accumulator instructions.
    AccumulatorPos = 55,
    AccumulatorMask = 0x1,

    // Complex XU, prevent xu competition by preferring slot3
    PrefersSlot3Pos = 56,
    PrefersSlot3Mask = 0x1,

    CofMax1Pos = 60,
    CofMax1Mask = 0x1,

    CVINewPos = 61,
    CVINewMask = 0x1
  };

  // *** The code above must match HexagonInstrFormat*.td *** //

  // Hexagon specific MO operand flag mask.
  enum HexagonMOTargetFlagVal {
    //===------------------------------------------------------------------===//
    // Hexagon Specific MachineOperand flags.
    MO_NO_FLAG,

    /// MO_PCREL - On a symbol operand, indicates a PC-relative relocation
    /// Used for computing a global address for PIC compilations
    MO_PCREL,

    /// MO_GOT - Indicates a GOT-relative relocation
    MO_GOT,

    // Low or high part of a symbol.
    MO_LO16, MO_HI16,

    // Offset from the base of the SDA.
    MO_GPREL,

    // MO_GDGOT - indicates GOT relative relocation for TLS
    // GeneralDynamic method
    MO_GDGOT,

    // MO_GDPLT - indicates PLT relative relocation for TLS
    // GeneralDynamic method
    MO_GDPLT,

    // MO_IE - indicates non PIC relocation for TLS
    // Initial Executable method
    MO_IE,

    // MO_IEGOT - indicates PIC relocation for TLS
    // Initial Executable method
    MO_IEGOT,

    // MO_TPREL - indicates relocation for TLS
    // local Executable method
    MO_TPREL,

    // HMOTF_ConstExtended
    // Addendum to abovem, indicates a const extended op
    // Can be used as a mask.
    HMOTF_ConstExtended = 0x80

  };

  // Hexagon Sub-instruction classes.
  enum SubInstructionGroup {
    HSIG_None = 0,
    HSIG_L1,
    HSIG_L2,
    HSIG_S1,
    HSIG_S2,
    HSIG_A,
    HSIG_Compound
  };

  // Hexagon Compound classes.
  enum CompoundGroup {
    HCG_None = 0,
    HCG_A,
    HCG_B,
    HCG_C
  };

  enum InstParseBits {
    INST_PARSE_MASK       = 0x0000c000,
    INST_PARSE_PACKET_END = 0x0000c000,
    INST_PARSE_LOOP_END   = 0x00008000,
    INST_PARSE_NOT_END    = 0x00004000,
    INST_PARSE_DUPLEX     = 0x00000000,
    INST_PARSE_EXTENDER   = 0x00000000
  };

  enum InstIClassBits : unsigned {
    INST_ICLASS_MASK      = 0xf0000000,
    INST_ICLASS_EXTENDER  = 0x00000000,
    INST_ICLASS_J_1       = 0x10000000,
    INST_ICLASS_J_2       = 0x20000000,
    INST_ICLASS_LD_ST_1   = 0x30000000,
    INST_ICLASS_LD_ST_2   = 0x40000000,
    INST_ICLASS_J_3       = 0x50000000,
    INST_ICLASS_CR        = 0x60000000,
    INST_ICLASS_ALU32_1   = 0x70000000,
    INST_ICLASS_XTYPE_1   = 0x80000000,
    INST_ICLASS_LD        = 0x90000000,
    INST_ICLASS_ST        = 0xa0000000,
    INST_ICLASS_ALU32_2   = 0xb0000000,
    INST_ICLASS_XTYPE_2   = 0xc0000000,
    INST_ICLASS_XTYPE_3   = 0xd0000000,
    INST_ICLASS_XTYPE_4   = 0xe0000000,
    INST_ICLASS_ALU32_3   = 0xf0000000
  };

} // End namespace HexagonII.

} // End namespace llvm.

#endif
