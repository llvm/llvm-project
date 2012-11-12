//===-- X86BaseInfo.h - Top level definitions for X86 -------- --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the X86 target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef X86BASEINFO_H
#define X86BASEINFO_H

#include "X86MCTargetDesc.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

namespace X86 {
  // Enums for memory operand decoding.  Each memory operand is represented with
  // a 5 operand sequence in the form:
  //   [BaseReg, ScaleAmt, IndexReg, Disp, Segment]
  // These enums help decode this.
  enum {
    AddrBaseReg = 0,
    AddrScaleAmt = 1,
    AddrIndexReg = 2,
    AddrDisp = 3,

    /// AddrSegmentReg - The operand # of the segment in the memory operand.
    AddrSegmentReg = 4,

    /// AddrNumOperands - Total number of operands in a memory reference.
    AddrNumOperands = 5
  };
} // end namespace X86;
 

/// X86II - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace X86II {
  /// Target Operand Flag enum.
  enum TOF {
    //===------------------------------------------------------------------===//
    // X86 Specific MachineOperand flags.

    MO_NO_FLAG,

    /// MO_GOT_ABSOLUTE_ADDRESS - On a symbol operand, this represents a
    /// relocation of:
    ///    SYMBOL_LABEL + [. - PICBASELABEL]
    MO_GOT_ABSOLUTE_ADDRESS,

    /// MO_PIC_BASE_OFFSET - On a symbol operand this indicates that the
    /// immediate should get the value of the symbol minus the PIC base label:
    ///    SYMBOL_LABEL - PICBASELABEL
    MO_PIC_BASE_OFFSET,

    /// MO_GOT - On a symbol operand this indicates that the immediate is the
    /// offset to the GOT entry for the symbol name from the base of the GOT.
    ///
    /// See the X86-64 ELF ABI supplement for more details.
    ///    SYMBOL_LABEL @GOT
    MO_GOT,

    /// MO_GOTOFF - On a symbol operand this indicates that the immediate is
    /// the offset to the location of the symbol name from the base of the GOT.
    ///
    /// See the X86-64 ELF ABI supplement for more details.
    ///    SYMBOL_LABEL @GOTOFF
    MO_GOTOFF,

    /// MO_GOTPCREL - On a symbol operand this indicates that the immediate is
    /// offset to the GOT entry for the symbol name from the current code
    /// location.
    ///
    /// See the X86-64 ELF ABI supplement for more details.
    ///    SYMBOL_LABEL @GOTPCREL
    MO_GOTPCREL,

    /// MO_PLT - On a symbol operand this indicates that the immediate is
    /// offset to the PLT entry of symbol name from the current code location.
    ///
    /// See the X86-64 ELF ABI supplement for more details.
    ///    SYMBOL_LABEL @PLT
    MO_PLT,

    /// MO_TLSGD - On a symbol operand this indicates that the immediate is
    /// the offset of the GOT entry with the TLS index structure that contains
    /// the module number and variable offset for the symbol. Used in the
    /// general dynamic TLS access model.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details.
    ///    SYMBOL_LABEL @TLSGD
    MO_TLSGD,

    /// MO_TLSLD - On a symbol operand this indicates that the immediate is
    /// the offset of the GOT entry with the TLS index for the module that
    /// contains the symbol. When this index is passed to a call to to
    /// __tls_get_addr, the function will return the base address of the TLS
    /// block for the symbol. Used in the x86-64 local dynamic TLS access model.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details.
    ///    SYMBOL_LABEL @TLSLD
    MO_TLSLD,

    /// MO_TLSLDM - On a symbol operand this indicates that the immediate is
    /// the offset of the GOT entry with the TLS index for the module that
    /// contains the symbol. When this index is passed to a call to to
    /// ___tls_get_addr, the function will return the base address of the TLS
    /// block for the symbol. Used in the IA32 local dynamic TLS access model.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details.
    ///    SYMBOL_LABEL @TLSLDM
    MO_TLSLDM,

    /// MO_GOTTPOFF - On a symbol operand this indicates that the immediate is
    /// the offset of the GOT entry with the thread-pointer offset for the
    /// symbol. Used in the x86-64 initial exec TLS access model.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details.
    ///    SYMBOL_LABEL @GOTTPOFF
    MO_GOTTPOFF,

    /// MO_INDNTPOFF - On a symbol operand this indicates that the immediate is
    /// the absolute address of the GOT entry with the negative thread-pointer
    /// offset for the symbol. Used in the non-PIC IA32 initial exec TLS access
    /// model.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details.
    ///    SYMBOL_LABEL @INDNTPOFF
    MO_INDNTPOFF,

    /// MO_TPOFF - On a symbol operand this indicates that the immediate is
    /// the thread-pointer offset for the symbol. Used in the x86-64 local
    /// exec TLS access model.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details.
    ///    SYMBOL_LABEL @TPOFF
    MO_TPOFF,

    /// MO_DTPOFF - On a symbol operand this indicates that the immediate is
    /// the offset of the GOT entry with the TLS offset of the symbol. Used
    /// in the local dynamic TLS access model.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details.
    ///    SYMBOL_LABEL @DTPOFF
    MO_DTPOFF,

    /// MO_NTPOFF - On a symbol operand this indicates that the immediate is
    /// the negative thread-pointer offset for the symbol. Used in the IA32
    /// local exec TLS access model.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details.
    ///    SYMBOL_LABEL @NTPOFF
    MO_NTPOFF,

    /// MO_GOTNTPOFF - On a symbol operand this indicates that the immediate is
    /// the offset of the GOT entry with the negative thread-pointer offset for
    /// the symbol. Used in the PIC IA32 initial exec TLS access model.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details.
    ///    SYMBOL_LABEL @GOTNTPOFF
    MO_GOTNTPOFF,

    /// MO_DLLIMPORT - On a symbol operand "FOO", this indicates that the
    /// reference is actually to the "__imp_FOO" symbol.  This is used for
    /// dllimport linkage on windows.
    MO_DLLIMPORT,

    /// MO_DARWIN_STUB - On a symbol operand "FOO", this indicates that the
    /// reference is actually to the "FOO$stub" symbol.  This is used for calls
    /// and jumps to external functions on Tiger and earlier.
    MO_DARWIN_STUB,

    /// MO_DARWIN_NONLAZY - On a symbol operand "FOO", this indicates that the
    /// reference is actually to the "FOO$non_lazy_ptr" symbol, which is a
    /// non-PIC-base-relative reference to a non-hidden dyld lazy pointer stub.
    MO_DARWIN_NONLAZY,

    /// MO_DARWIN_NONLAZY_PIC_BASE - On a symbol operand "FOO", this indicates
    /// that the reference is actually to "FOO$non_lazy_ptr - PICBASE", which is
    /// a PIC-base-relative reference to a non-hidden dyld lazy pointer stub.
    MO_DARWIN_NONLAZY_PIC_BASE,

    /// MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE - On a symbol operand "FOO", this
    /// indicates that the reference is actually to "FOO$non_lazy_ptr -PICBASE",
    /// which is a PIC-base-relative reference to a hidden dyld lazy pointer
    /// stub.
    MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE,

    /// MO_TLVP - On a symbol operand this indicates that the immediate is
    /// some TLS offset.
    ///
    /// This is the TLS offset for the Darwin TLS mechanism.
    MO_TLVP,

    /// MO_TLVP_PIC_BASE - On a symbol operand this indicates that the immediate
    /// is some TLS offset from the picbase.
    ///
    /// This is the 32-bit TLS offset for Darwin TLS in PIC mode.
    MO_TLVP_PIC_BASE,

    /// MO_SECREL - On a symbol operand this indicates that the immediate is
    /// the offset from beginning of section.
    ///
    /// This is the TLS offset for the COFF/Windows TLS mechanism.
    MO_SECREL
  };

  enum {
    //===------------------------------------------------------------------===//
    // Instruction encodings.  These are the standard/most common forms for X86
    // instructions.
    //

    // PseudoFrm - This represents an instruction that is a pseudo instruction
    // or one that has not been implemented yet.  It is illegal to code generate
    // it, but tolerated for intermediate implementation stages.
    Pseudo         = 0,

    /// Raw - This form is for instructions that don't have any operands, so
    /// they are just a fixed opcode value, like 'leave'.
    RawFrm         = 1,

    /// AddRegFrm - This form is used for instructions like 'push r32' that have
    /// their one register operand added to their opcode.
    AddRegFrm      = 2,

    /// MRMDestReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is a register.
    ///
    MRMDestReg     = 3,

    /// MRMDestMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is memory.
    ///
    MRMDestMem     = 4,

    /// MRMSrcReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is a register.
    ///
    MRMSrcReg      = 5,

    /// MRMSrcMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is memory.
    ///
    MRMSrcMem      = 6,

    /// MRM[0-7][rm] - These forms are used to represent instructions that use
    /// a Mod/RM byte, and use the middle field to hold extended opcode
    /// information.  In the intel manual these are represented as /0, /1, ...
    ///

    // First, instructions that operate on a register r/m operand...
    MRM0r = 16,  MRM1r = 17,  MRM2r = 18,  MRM3r = 19, // Format /0 /1 /2 /3
    MRM4r = 20,  MRM5r = 21,  MRM6r = 22,  MRM7r = 23, // Format /4 /5 /6 /7

    // Next, instructions that operate on a memory r/m operand...
    MRM0m = 24,  MRM1m = 25,  MRM2m = 26,  MRM3m = 27, // Format /0 /1 /2 /3
    MRM4m = 28,  MRM5m = 29,  MRM6m = 30,  MRM7m = 31, // Format /4 /5 /6 /7

    // MRMInitReg - This form is used for instructions whose source and
    // destinations are the same register.
    MRMInitReg = 32,

    //// MRM_XX - A mod/rm byte of exactly 0xXX.
    MRM_C1 = 33, MRM_C2 = 34, MRM_C3 = 35, MRM_C4 = 36,
    MRM_C8 = 37, MRM_C9 = 38, MRM_E8 = 39, MRM_F0 = 40,
    MRM_F8 = 41, MRM_F9 = 42, MRM_D0 = 45, MRM_D1 = 46,
    MRM_D4 = 47, MRM_D5 = 48, MRM_D8 = 49, MRM_D9 = 50,
    MRM_DA = 51, MRM_DB = 52, MRM_DC = 53, MRM_DD = 54,
    MRM_DE = 55, MRM_DF = 56,

    /// RawFrmImm8 - This is used for the ENTER instruction, which has two
    /// immediates, the first of which is a 16-bit immediate (specified by
    /// the imm encoding) and the second is a 8-bit fixed value.
    RawFrmImm8 = 43,

    /// RawFrmImm16 - This is used for CALL FAR instructions, which have two
    /// immediates, the first of which is a 16 or 32-bit immediate (specified by
    /// the imm encoding) and the second is a 16-bit fixed value.  In the AMD
    /// manual, this operand is described as pntr16:32 and pntr16:16
    RawFrmImm16 = 44,

    FormMask       = 63,

    //===------------------------------------------------------------------===//
    // Actual flags...

    // OpSize - Set if this instruction requires an operand size prefix (0x66),
    // which most often indicates that the instruction operates on 16 bit data
    // instead of 32 bit data.
    OpSize      = 1 << 6,

    // AsSize - Set if this instruction requires an operand size prefix (0x67),
    // which most often indicates that the instruction address 16 bit address
    // instead of 32 bit address (or 32 bit address in 64 bit mode).
    AdSize      = 1 << 7,

    //===------------------------------------------------------------------===//
    // Op0Mask - There are several prefix bytes that are used to form two byte
    // opcodes.  These are currently 0x0F, 0xF3, and 0xD8-0xDF.  This mask is
    // used to obtain the setting of this field.  If no bits in this field is
    // set, there is no prefix byte for obtaining a multibyte opcode.
    //
    Op0Shift    = 8,
    Op0Mask     = 0x1F << Op0Shift,

    // TB - TwoByte - Set if this instruction has a two byte opcode, which
    // starts with a 0x0F byte before the real opcode.
    TB          = 1 << Op0Shift,

    // REP - The 0xF3 prefix byte indicating repetition of the following
    // instruction.
    REP         = 2 << Op0Shift,

    // D8-DF - These escape opcodes are used by the floating point unit.  These
    // values must remain sequential.
    D8 = 3 << Op0Shift,   D9 = 4 << Op0Shift,
    DA = 5 << Op0Shift,   DB = 6 << Op0Shift,
    DC = 7 << Op0Shift,   DD = 8 << Op0Shift,
    DE = 9 << Op0Shift,   DF = 10 << Op0Shift,

    // XS, XD - These prefix codes are for single and double precision scalar
    // floating point operations performed in the SSE registers.
    XD = 11 << Op0Shift,  XS = 12 << Op0Shift,

    // T8, TA, A6, A7 - Prefix after the 0x0F prefix.
    T8 = 13 << Op0Shift,  TA = 14 << Op0Shift,
    A6 = 15 << Op0Shift,  A7 = 16 << Op0Shift,

    // T8XD - Prefix before and after 0x0F. Combination of T8 and XD.
    T8XD = 17 << Op0Shift,

    // T8XS - Prefix before and after 0x0F. Combination of T8 and XS.
    T8XS = 18 << Op0Shift,

    // TAXD - Prefix before and after 0x0F. Combination of TA and XD.
    TAXD = 19 << Op0Shift,

    // XOP8 - Prefix to include use of imm byte.
    XOP8 = 20 << Op0Shift,

    // XOP9 - Prefix to exclude use of imm byte.
    XOP9 = 21 << Op0Shift,

    //===------------------------------------------------------------------===//
    // REX_W - REX prefixes are instruction prefixes used in 64-bit mode.
    // They are used to specify GPRs and SSE registers, 64-bit operand size,
    // etc. We only cares about REX.W and REX.R bits and only the former is
    // statically determined.
    //
    REXShift    = Op0Shift + 5,
    REX_W       = 1 << REXShift,

    //===------------------------------------------------------------------===//
    // This three-bit field describes the size of an immediate operand.  Zero is
    // unused so that we can tell if we forgot to set a value.
    ImmShift = REXShift + 1,
    ImmMask    = 7 << ImmShift,
    Imm8       = 1 << ImmShift,
    Imm8PCRel  = 2 << ImmShift,
    Imm16      = 3 << ImmShift,
    Imm16PCRel = 4 << ImmShift,
    Imm32      = 5 << ImmShift,
    Imm32PCRel = 6 << ImmShift,
    Imm64      = 7 << ImmShift,

    //===------------------------------------------------------------------===//
    // FP Instruction Classification...  Zero is non-fp instruction.

    // FPTypeMask - Mask for all of the FP types...
    FPTypeShift = ImmShift + 3,
    FPTypeMask  = 7 << FPTypeShift,

    // NotFP - The default, set for instructions that do not use FP registers.
    NotFP      = 0 << FPTypeShift,

    // ZeroArgFP - 0 arg FP instruction which implicitly pushes ST(0), f.e. fld0
    ZeroArgFP  = 1 << FPTypeShift,

    // OneArgFP - 1 arg FP instructions which implicitly read ST(0), such as fst
    OneArgFP   = 2 << FPTypeShift,

    // OneArgFPRW - 1 arg FP instruction which implicitly read ST(0) and write a
    // result back to ST(0).  For example, fcos, fsqrt, etc.
    //
    OneArgFPRW = 3 << FPTypeShift,

    // TwoArgFP - 2 arg FP instructions which implicitly read ST(0), and an
    // explicit argument, storing the result to either ST(0) or the implicit
    // argument.  For example: fadd, fsub, fmul, etc...
    TwoArgFP   = 4 << FPTypeShift,

    // CompareFP - 2 arg FP instructions which implicitly read ST(0) and an
    // explicit argument, but have no destination.  Example: fucom, fucomi, ...
    CompareFP  = 5 << FPTypeShift,

    // CondMovFP - "2 operand" floating point conditional move instructions.
    CondMovFP  = 6 << FPTypeShift,

    // SpecialFP - Special instruction forms.  Dispatch by opcode explicitly.
    SpecialFP  = 7 << FPTypeShift,

    // Lock prefix
    LOCKShift = FPTypeShift + 3,
    LOCK = 1 << LOCKShift,

    // Segment override prefixes. Currently we just need ability to address
    // stuff in gs and fs segments.
    SegOvrShift = LOCKShift + 1,
    SegOvrMask  = 3 << SegOvrShift,
    FS          = 1 << SegOvrShift,
    GS          = 2 << SegOvrShift,

    // Execution domain for SSE instructions in bits 23, 24.
    // 0 in bits 23-24 means normal, non-SSE instruction.
    SSEDomainShift = SegOvrShift + 2,

    OpcodeShift   = SSEDomainShift + 2,

    //===------------------------------------------------------------------===//
    /// VEX - The opcode prefix used by AVX instructions
    VEXShift = OpcodeShift + 8,
    VEX         = 1U << 0,

    /// VEX_W - Has a opcode specific functionality, but is used in the same
    /// way as REX_W is for regular SSE instructions.
    VEX_W       = 1U << 1,

    /// VEX_4V - Used to specify an additional AVX/SSE register. Several 2
    /// address instructions in SSE are represented as 3 address ones in AVX
    /// and the additional register is encoded in VEX_VVVV prefix.
    VEX_4V      = 1U << 2,

    /// VEX_4VOp3 - Similar to VEX_4V, but used on instructions that encode
    /// operand 3 with VEX.vvvv.
    VEX_4VOp3   = 1U << 3,

    /// VEX_I8IMM - Specifies that the last register used in a AVX instruction,
    /// must be encoded in the i8 immediate field. This usually happens in
    /// instructions with 4 operands.
    VEX_I8IMM   = 1U << 4,

    /// VEX_L - Stands for a bit in the VEX opcode prefix meaning the current
    /// instruction uses 256-bit wide registers. This is usually auto detected
    /// if a VR256 register is used, but some AVX instructions also have this
    /// field marked when using a f256 memory references.
    VEX_L       = 1U << 5,

    // VEX_LIG - Specifies that this instruction ignores the L-bit in the VEX
    // prefix. Usually used for scalar instructions. Needed by disassembler.
    VEX_LIG     = 1U << 6,

    /// Has3DNow0F0FOpcode - This flag indicates that the instruction uses the
    /// wacky 0x0F 0x0F prefix for 3DNow! instructions.  The manual documents
    /// this as having a 0x0F prefix with a 0x0F opcode, and each instruction
    /// storing a classifier in the imm8 field.  To simplify our implementation,
    /// we handle this by storeing the classifier in the opcode field and using
    /// this flag to indicate that the encoder should do the wacky 3DNow! thing.
    Has3DNow0F0FOpcode = 1U << 7,

    /// MemOp4 - Used to indicate swapping of operand 3 and 4 to be encoded in
    /// ModRM or I8IMM. This is used for FMA4 and XOP instructions.
    MemOp4 = 1U << 8,

    /// XOP - Opcode prefix used by XOP instructions.
    XOP = 1U << 9

  };

  // getBaseOpcodeFor - This function returns the "base" X86 opcode for the
  // specified machine instruction.
  //
  inline unsigned char getBaseOpcodeFor(uint64_t TSFlags) {
    return TSFlags >> X86II::OpcodeShift;
  }

  inline bool hasImm(uint64_t TSFlags) {
    return (TSFlags & X86II::ImmMask) != 0;
  }

  /// getSizeOfImm - Decode the "size of immediate" field from the TSFlags field
  /// of the specified instruction.
  inline unsigned getSizeOfImm(uint64_t TSFlags) {
    switch (TSFlags & X86II::ImmMask) {
    default: llvm_unreachable("Unknown immediate size");
    case X86II::Imm8:
    case X86II::Imm8PCRel:  return 1;
    case X86II::Imm16:
    case X86II::Imm16PCRel: return 2;
    case X86II::Imm32:
    case X86II::Imm32PCRel: return 4;
    case X86II::Imm64:      return 8;
    }
  }

  /// isImmPCRel - Return true if the immediate of the specified instruction's
  /// TSFlags indicates that it is pc relative.
  inline unsigned isImmPCRel(uint64_t TSFlags) {
    switch (TSFlags & X86II::ImmMask) {
    default: llvm_unreachable("Unknown immediate size");
    case X86II::Imm8PCRel:
    case X86II::Imm16PCRel:
    case X86II::Imm32PCRel:
      return true;
    case X86II::Imm8:
    case X86II::Imm16:
    case X86II::Imm32:
    case X86II::Imm64:
      return false;
    }
  }

  /// getMemoryOperandNo - The function returns the MCInst operand # for the
  /// first field of the memory operand.  If the instruction doesn't have a
  /// memory operand, this returns -1.
  ///
  /// Note that this ignores tied operands.  If there is a tied register which
  /// is duplicated in the MCInst (e.g. "EAX = addl EAX, [mem]") it is only
  /// counted as one operand.
  ///
  inline int getMemoryOperandNo(uint64_t TSFlags, unsigned Opcode) {
    switch (TSFlags & X86II::FormMask) {
    case X86II::MRMInitReg:
        // FIXME: Remove this form.
        return -1;
    default: llvm_unreachable("Unknown FormMask value in getMemoryOperandNo!");
    case X86II::Pseudo:
    case X86II::RawFrm:
    case X86II::AddRegFrm:
    case X86II::MRMDestReg:
    case X86II::MRMSrcReg:
    case X86II::RawFrmImm8:
    case X86II::RawFrmImm16:
       return -1;
    case X86II::MRMDestMem:
      return 0;
    case X86II::MRMSrcMem: {
      bool HasVEX_4V = (TSFlags >> X86II::VEXShift) & X86II::VEX_4V;
      bool HasMemOp4 = (TSFlags >> X86II::VEXShift) & X86II::MemOp4;
      unsigned FirstMemOp = 1;
      if (HasVEX_4V)
        ++FirstMemOp;// Skip the register source (which is encoded in VEX_VVVV).
      if (HasMemOp4)
        ++FirstMemOp;// Skip the register source (which is encoded in I8IMM).

      // FIXME: Maybe lea should have its own form?  This is a horrible hack.
      //if (Opcode == X86::LEA64r || Opcode == X86::LEA64_32r ||
      //    Opcode == X86::LEA16r || Opcode == X86::LEA32r)
      return FirstMemOp;
    }
    case X86II::MRM0r: case X86II::MRM1r:
    case X86II::MRM2r: case X86II::MRM3r:
    case X86II::MRM4r: case X86II::MRM5r:
    case X86II::MRM6r: case X86II::MRM7r:
      return -1;
    case X86II::MRM0m: case X86II::MRM1m:
    case X86II::MRM2m: case X86II::MRM3m:
    case X86II::MRM4m: case X86II::MRM5m:
    case X86II::MRM6m: case X86II::MRM7m: {
      bool HasVEX_4V = (TSFlags >> X86II::VEXShift) & X86II::VEX_4V;
      unsigned FirstMemOp = 0;
      if (HasVEX_4V)
        ++FirstMemOp;// Skip the register dest (which is encoded in VEX_VVVV).
      return FirstMemOp;
    }
    case X86II::MRM_C1: case X86II::MRM_C2:
    case X86II::MRM_C3: case X86II::MRM_C4:
    case X86II::MRM_C8: case X86II::MRM_C9:
    case X86II::MRM_E8: case X86II::MRM_F0:
    case X86II::MRM_F8: case X86II::MRM_F9:
    case X86II::MRM_D0: case X86II::MRM_D1:
    case X86II::MRM_D4: case X86II::MRM_D5:
    case X86II::MRM_D8: case X86II::MRM_D9:
    case X86II::MRM_DA: case X86II::MRM_DB:
    case X86II::MRM_DC: case X86II::MRM_DD:
    case X86II::MRM_DE: case X86II::MRM_DF:
      return -1;
    }
  }

  /// isX86_64ExtendedReg - Is the MachineOperand a x86-64 extended (r8 or
  /// higher) register?  e.g. r8, xmm8, xmm13, etc.
  inline bool isX86_64ExtendedReg(unsigned RegNo) {
    switch (RegNo) {
    default: break;
    case X86::R8:    case X86::R9:    case X86::R10:   case X86::R11:
    case X86::R12:   case X86::R13:   case X86::R14:   case X86::R15:
    case X86::R8D:   case X86::R9D:   case X86::R10D:  case X86::R11D:
    case X86::R12D:  case X86::R13D:  case X86::R14D:  case X86::R15D:
    case X86::R8W:   case X86::R9W:   case X86::R10W:  case X86::R11W:
    case X86::R12W:  case X86::R13W:  case X86::R14W:  case X86::R15W:
    case X86::R8B:   case X86::R9B:   case X86::R10B:  case X86::R11B:
    case X86::R12B:  case X86::R13B:  case X86::R14B:  case X86::R15B:
    case X86::XMM8:  case X86::XMM9:  case X86::XMM10: case X86::XMM11:
    case X86::XMM12: case X86::XMM13: case X86::XMM14: case X86::XMM15:
    case X86::YMM8:  case X86::YMM9:  case X86::YMM10: case X86::YMM11:
    case X86::YMM12: case X86::YMM13: case X86::YMM14: case X86::YMM15:
    case X86::CR8:   case X86::CR9:   case X86::CR10:  case X86::CR11:
    case X86::CR12:  case X86::CR13:  case X86::CR14:  case X86::CR15:
        return true;
    }
    return false;
  }
  
  inline bool isX86_64NonExtLowByteReg(unsigned reg) {
    return (reg == X86::SPL || reg == X86::BPL ||
            reg == X86::SIL || reg == X86::DIL);
  }
}

} // end namespace llvm;

#endif
