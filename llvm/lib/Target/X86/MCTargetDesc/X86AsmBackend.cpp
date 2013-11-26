//===-- X86AsmBackend.cpp - X86 Assembler Backend -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86FixupKinds.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// Option to allow disabling arithmetic relaxation to workaround PR9807, which
// is useful when running bitwise comparison experiments on Darwin. We should be
// able to remove this once PR9807 is resolved.
static cl::opt<bool>
MCDisableArithRelaxation("mc-x86-disable-arith-relaxation",
         cl::desc("Disable relaxation of arithmetic instruction for X86"));

static unsigned getFixupKindLog2Size(unsigned Kind) {
  switch (Kind) {
  default: llvm_unreachable("invalid fixup kind!");
  case FK_PCRel_1:
  case FK_SecRel_1:
  case FK_Data_1: return 0;
  case FK_PCRel_2:
  case FK_SecRel_2:
  case FK_Data_2: return 1;
  case FK_PCRel_4:
  case X86::reloc_riprel_4byte:
  case X86::reloc_riprel_4byte_movq_load:
  case X86::reloc_signed_4byte:
  case X86::reloc_global_offset_table:
  case FK_SecRel_4:
  case FK_Data_4: return 2;
  case FK_PCRel_8:
  case FK_SecRel_8:
  case FK_Data_8: return 3;
  }
}

namespace {

class X86ELFObjectWriter : public MCELFObjectTargetWriter {
public:
  X86ELFObjectWriter(bool is64Bit, uint8_t OSABI, uint16_t EMachine,
                     bool HasRelocationAddend, bool foobar)
    : MCELFObjectTargetWriter(is64Bit, OSABI, EMachine, HasRelocationAddend) {}
};

class X86AsmBackend : public MCAsmBackend {
  StringRef CPU;
  bool HasNopl;
public:
  X86AsmBackend(const Target &T, StringRef _CPU)
    : MCAsmBackend(), CPU(_CPU) {
    HasNopl = CPU != "generic" && CPU != "i386" && CPU != "i486" &&
              CPU != "i586" && CPU != "pentium" && CPU != "pentium-mmx" &&
              CPU != "i686" && CPU != "k6" && CPU != "k6-2" && CPU != "k6-3" &&
              CPU != "geode" && CPU != "winchip-c6" && CPU != "winchip2" &&
              CPU != "c3" && CPU != "c3-2";
  }

  unsigned getNumFixupKinds() const {
    return X86::NumTargetFixupKinds;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    const static MCFixupKindInfo Infos[X86::NumTargetFixupKinds] = {
      { "reloc_riprel_4byte", 0, 4 * 8, MCFixupKindInfo::FKF_IsPCRel },
      { "reloc_riprel_4byte_movq_load", 0, 4 * 8, MCFixupKindInfo::FKF_IsPCRel},
      { "reloc_signed_4byte", 0, 4 * 8, 0},
      { "reloc_global_offset_table", 0, 4 * 8, 0}
    };

    if (Kind < FirstTargetFixupKind)
      return MCAsmBackend::getFixupKindInfo(Kind);

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }

  void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value) const {
    unsigned Size = 1 << getFixupKindLog2Size(Fixup.getKind());

    assert(Fixup.getOffset() + Size <= DataSize &&
           "Invalid fixup offset!");

    // Check that uppper bits are either all zeros or all ones.
    // Specifically ignore overflow/underflow as long as the leakage is
    // limited to the lower bits. This is to remain compatible with
    // other assemblers.
    assert(isIntN(Size * 8 + 1, Value) &&
           "Value does not fit in the Fixup field");

    for (unsigned i = 0; i != Size; ++i)
      Data[Fixup.getOffset() + i] = uint8_t(Value >> (i * 8));
  }

  bool mayNeedRelaxation(const MCInst &Inst) const;

  bool fixupNeedsRelaxation(const MCFixup &Fixup,
                            uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const;

  void relaxInstruction(const MCInst &Inst, MCInst &Res) const;

  bool writeNopData(uint64_t Count, MCObjectWriter *OW) const;
};
} // end anonymous namespace

static unsigned getRelaxedOpcodeBranch(unsigned Op) {
  switch (Op) {
  default:
    return Op;

  case X86::JAE_1: return X86::JAE_4;
  case X86::JA_1:  return X86::JA_4;
  case X86::JBE_1: return X86::JBE_4;
  case X86::JB_1:  return X86::JB_4;
  case X86::JE_1:  return X86::JE_4;
  case X86::JGE_1: return X86::JGE_4;
  case X86::JG_1:  return X86::JG_4;
  case X86::JLE_1: return X86::JLE_4;
  case X86::JL_1:  return X86::JL_4;
  case X86::JMP_1: return X86::JMP_4;
  case X86::JNE_1: return X86::JNE_4;
  case X86::JNO_1: return X86::JNO_4;
  case X86::JNP_1: return X86::JNP_4;
  case X86::JNS_1: return X86::JNS_4;
  case X86::JO_1:  return X86::JO_4;
  case X86::JP_1:  return X86::JP_4;
  case X86::JS_1:  return X86::JS_4;
  }
}

static unsigned getRelaxedOpcodeArith(unsigned Op) {
  switch (Op) {
  default:
    return Op;

    // IMUL
  case X86::IMUL16rri8: return X86::IMUL16rri;
  case X86::IMUL16rmi8: return X86::IMUL16rmi;
  case X86::IMUL32rri8: return X86::IMUL32rri;
  case X86::IMUL32rmi8: return X86::IMUL32rmi;
  case X86::IMUL64rri8: return X86::IMUL64rri32;
  case X86::IMUL64rmi8: return X86::IMUL64rmi32;

    // AND
  case X86::AND16ri8: return X86::AND16ri;
  case X86::AND16mi8: return X86::AND16mi;
  case X86::AND32ri8: return X86::AND32ri;
  case X86::AND32mi8: return X86::AND32mi;
  case X86::AND64ri8: return X86::AND64ri32;
  case X86::AND64mi8: return X86::AND64mi32;

    // OR
  case X86::OR16ri8: return X86::OR16ri;
  case X86::OR16mi8: return X86::OR16mi;
  case X86::OR32ri8: return X86::OR32ri;
  case X86::OR32mi8: return X86::OR32mi;
  case X86::OR64ri8: return X86::OR64ri32;
  case X86::OR64mi8: return X86::OR64mi32;

    // XOR
  case X86::XOR16ri8: return X86::XOR16ri;
  case X86::XOR16mi8: return X86::XOR16mi;
  case X86::XOR32ri8: return X86::XOR32ri;
  case X86::XOR32mi8: return X86::XOR32mi;
  case X86::XOR64ri8: return X86::XOR64ri32;
  case X86::XOR64mi8: return X86::XOR64mi32;

    // ADD
  case X86::ADD16ri8: return X86::ADD16ri;
  case X86::ADD16mi8: return X86::ADD16mi;
  case X86::ADD32ri8: return X86::ADD32ri;
  case X86::ADD32mi8: return X86::ADD32mi;
  case X86::ADD64ri8: return X86::ADD64ri32;
  case X86::ADD64mi8: return X86::ADD64mi32;

    // SUB
  case X86::SUB16ri8: return X86::SUB16ri;
  case X86::SUB16mi8: return X86::SUB16mi;
  case X86::SUB32ri8: return X86::SUB32ri;
  case X86::SUB32mi8: return X86::SUB32mi;
  case X86::SUB64ri8: return X86::SUB64ri32;
  case X86::SUB64mi8: return X86::SUB64mi32;

    // CMP
  case X86::CMP16ri8: return X86::CMP16ri;
  case X86::CMP16mi8: return X86::CMP16mi;
  case X86::CMP32ri8: return X86::CMP32ri;
  case X86::CMP32mi8: return X86::CMP32mi;
  case X86::CMP64ri8: return X86::CMP64ri32;
  case X86::CMP64mi8: return X86::CMP64mi32;

    // PUSH
  case X86::PUSHi8: return X86::PUSHi32;
  case X86::PUSHi16: return X86::PUSHi32;
  case X86::PUSH64i8: return X86::PUSH64i32;
  case X86::PUSH64i16: return X86::PUSH64i32;
  }
}

static unsigned getRelaxedOpcode(unsigned Op) {
  unsigned R = getRelaxedOpcodeArith(Op);
  if (R != Op)
    return R;
  return getRelaxedOpcodeBranch(Op);
}

bool X86AsmBackend::mayNeedRelaxation(const MCInst &Inst) const {
  // Branches can always be relaxed.
  if (getRelaxedOpcodeBranch(Inst.getOpcode()) != Inst.getOpcode())
    return true;

  if (MCDisableArithRelaxation)
    return false;

  // Check if this instruction is ever relaxable.
  if (getRelaxedOpcodeArith(Inst.getOpcode()) == Inst.getOpcode())
    return false;


  // Check if it has an expression and is not RIP relative.
  bool hasExp = false;
  bool hasRIP = false;
  for (unsigned i = 0; i < Inst.getNumOperands(); ++i) {
    const MCOperand &Op = Inst.getOperand(i);
    if (Op.isExpr())
      hasExp = true;

    if (Op.isReg() && Op.getReg() == X86::RIP)
      hasRIP = true;
  }

  // FIXME: Why exactly do we need the !hasRIP? Is it just a limitation on
  // how we do relaxations?
  return hasExp && !hasRIP;
}

bool X86AsmBackend::fixupNeedsRelaxation(const MCFixup &Fixup,
                                         uint64_t Value,
                                         const MCRelaxableFragment *DF,
                                         const MCAsmLayout &Layout) const {
  // Relax if the value is too big for a (signed) i8.
  return int64_t(Value) != int64_t(int8_t(Value));
}

// FIXME: Can tblgen help at all here to verify there aren't other instructions
// we can relax?
void X86AsmBackend::relaxInstruction(const MCInst &Inst, MCInst &Res) const {
  // The only relaxations X86 does is from a 1byte pcrel to a 4byte pcrel.
  unsigned RelaxedOp = getRelaxedOpcode(Inst.getOpcode());

  if (RelaxedOp == Inst.getOpcode()) {
    SmallString<256> Tmp;
    raw_svector_ostream OS(Tmp);
    Inst.dump_pretty(OS);
    OS << "\n";
    report_fatal_error("unexpected instruction to relax: " + OS.str());
  }

  Res = Inst;
  Res.setOpcode(RelaxedOp);
}

/// \brief Write a sequence of optimal nops to the output, covering \p Count
/// bytes.
/// \return - true on success, false on failure
bool X86AsmBackend::writeNopData(uint64_t Count, MCObjectWriter *OW) const {
  static const uint8_t Nops[10][10] = {
    // nop
    {0x90},
    // xchg %ax,%ax
    {0x66, 0x90},
    // nopl (%[re]ax)
    {0x0f, 0x1f, 0x00},
    // nopl 0(%[re]ax)
    {0x0f, 0x1f, 0x40, 0x00},
    // nopl 0(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopw 0(%[re]ax,%[re]ax,1)
    {0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopl 0L(%[re]ax)
    {0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00},
    // nopl 0L(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    // nopw 0L(%[re]ax,%[re]ax,1)
    {0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    // nopw %cs:0L(%[re]ax,%[re]ax,1)
    {0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
  };

  // This CPU doesnt support long nops. If needed add more.
  // FIXME: Can we get this from the subtarget somehow?
  // FIXME: We could generated something better than plain 0x90.
  if (!HasNopl) {
    for (uint64_t i = 0; i < Count; ++i)
      OW->Write8(0x90);
    return true;
  }

  // 15 is the longest single nop instruction.  Emit as many 15-byte nops as
  // needed, then emit a nop of the remaining length.
  do {
    const uint8_t ThisNopLength = (uint8_t) std::min(Count, (uint64_t) 15);
    const uint8_t Prefixes = ThisNopLength <= 10 ? 0 : ThisNopLength - 10;
    for (uint8_t i = 0; i < Prefixes; i++)
      OW->Write8(0x66);
    const uint8_t Rest = ThisNopLength - Prefixes;
    for (uint8_t i = 0; i < Rest; i++)
      OW->Write8(Nops[Rest - 1][i]);
    Count -= ThisNopLength;
  } while (Count != 0);

  return true;
}

/* *** */

namespace {

class ELFX86AsmBackend : public X86AsmBackend {
public:
  uint8_t OSABI;
  ELFX86AsmBackend(const Target &T, uint8_t _OSABI, StringRef CPU)
    : X86AsmBackend(T, CPU), OSABI(_OSABI) {
    HasReliableSymbolDifference = true;
  }

  virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
    const MCSectionELF &ES = static_cast<const MCSectionELF&>(Section);
    return ES.getFlags() & ELF::SHF_MERGE;
  }
};

class ELFX86_32AsmBackend : public ELFX86AsmBackend {
public:
  ELFX86_32AsmBackend(const Target &T, uint8_t OSABI, StringRef CPU)
    : ELFX86AsmBackend(T, OSABI, CPU) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createX86ELFObjectWriter(OS, /*IsELF64*/ false, OSABI, ELF::EM_386);
  }
};

class ELFX86_64AsmBackend : public ELFX86AsmBackend {
public:
  ELFX86_64AsmBackend(const Target &T, uint8_t OSABI, StringRef CPU)
    : ELFX86AsmBackend(T, OSABI, CPU) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createX86ELFObjectWriter(OS, /*IsELF64*/ true, OSABI, ELF::EM_X86_64);
  }
};

class WindowsX86AsmBackend : public X86AsmBackend {
  bool Is64Bit;

public:
  WindowsX86AsmBackend(const Target &T, bool is64Bit, StringRef CPU)
    : X86AsmBackend(T, CPU)
    , Is64Bit(is64Bit) {
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createX86WinCOFFObjectWriter(OS, Is64Bit);
  }
};

namespace CU {

  /// Compact unwind encoding values.
  enum CompactUnwindEncodings {
    /// [RE]BP based frame where [RE]BP is pused on the stack immediately after
    /// the return address, then [RE]SP is moved to [RE]BP.
    UNWIND_MODE_BP_FRAME                   = 0x01000000,

    /// A frameless function with a small constant stack size.
    UNWIND_MODE_STACK_IMMD                 = 0x02000000,

    /// A frameless function with a large constant stack size.
    UNWIND_MODE_STACK_IND                  = 0x03000000,

    /// No compact unwind encoding is available.
    UNWIND_MODE_DWARF                      = 0x04000000,

    /// Mask for encoding the frame registers.
    UNWIND_BP_FRAME_REGISTERS              = 0x00007FFF,

    /// Mask for encoding the frameless registers.
    UNWIND_FRAMELESS_STACK_REG_PERMUTATION = 0x000003FF
  };

} // end CU namespace

class DarwinX86AsmBackend : public X86AsmBackend {
  const MCRegisterInfo &MRI;

  /// \brief Number of registers that can be saved in a compact unwind encoding.
  enum { CU_NUM_SAVED_REGS = 6 };

  mutable unsigned SavedRegs[CU_NUM_SAVED_REGS];
  bool Is64Bit;

  unsigned OffsetSize;                   ///< Offset of a "push" instruction.
  unsigned PushInstrSize;                ///< Size of a "push" instruction.
  unsigned MoveInstrSize;                ///< Size of a "move" instruction.
  unsigned StackDivide;                  ///< Amount to adjust stack stize by.
protected:
  /// \brief Implementation of algorithm to generate the compact unwind encoding
  /// for the CFI instructions.
  uint32_t
  generateCompactUnwindEncodingImpl(ArrayRef<MCCFIInstruction> Instrs) const {
    if (Instrs.empty()) return 0;

    // Reset the saved registers.
    unsigned SavedRegIdx = 0;
    memset(SavedRegs, 0, sizeof(SavedRegs));

    bool HasFP = false;

    // Encode that we are using EBP/RBP as the frame pointer.
    uint32_t CompactUnwindEncoding = 0;

    unsigned SubtractInstrIdx = Is64Bit ? 3 : 2;
    unsigned InstrOffset = 0;
    unsigned StackAdjust = 0;
    unsigned StackSize = 0;
    unsigned PrevStackSize = 0;
    unsigned NumDefCFAOffsets = 0;

    for (unsigned i = 0, e = Instrs.size(); i != e; ++i) {
      const MCCFIInstruction &Inst = Instrs[i];

      switch (Inst.getOperation()) {
      default:
        // Any other CFI directives indicate a frame that we aren't prepared
        // to represent via compact unwind, so just bail out.
        return 0;
      case MCCFIInstruction::OpDefCfaRegister: {
        // Defines a frame pointer. E.g.
        //
        //     movq %rsp, %rbp
        //  L0:
        //     .cfi_def_cfa_register %rbp
        //
        HasFP = true;
        assert(MRI.getLLVMRegNum(Inst.getRegister(), true) ==
               (Is64Bit ? X86::RBP : X86::EBP) && "Invalid frame pointer!");

        // Reset the counts.
        memset(SavedRegs, 0, sizeof(SavedRegs));
        StackAdjust = 0;
        SavedRegIdx = 0;
        InstrOffset += MoveInstrSize;
        break;
      }
      case MCCFIInstruction::OpDefCfaOffset: {
        // Defines a new offset for the CFA. E.g.
        //
        //  With frame:
        //  
        //     pushq %rbp
        //  L0:
        //     .cfi_def_cfa_offset 16
        //
        //  Without frame:
        //
        //     subq $72, %rsp
        //  L0:
        //     .cfi_def_cfa_offset 80
        //
        PrevStackSize = StackSize;
        StackSize = std::abs(Inst.getOffset()) / StackDivide;
        ++NumDefCFAOffsets;
        break;
      }
      case MCCFIInstruction::OpOffset: {
        // Defines a "push" of a callee-saved register. E.g.
        //
        //     pushq %r15
        //     pushq %r14
        //     pushq %rbx
        //  L0:
        //     subq $120, %rsp
        //  L1:
        //     .cfi_offset %rbx, -40
        //     .cfi_offset %r14, -32
        //     .cfi_offset %r15, -24
        //
        if (SavedRegIdx == CU_NUM_SAVED_REGS)
          // If there are too many saved registers, we cannot use a compact
          // unwind encoding.
          return CU::UNWIND_MODE_DWARF;

        unsigned Reg = MRI.getLLVMRegNum(Inst.getRegister(), true);
        SavedRegs[SavedRegIdx++] = Reg;
        StackAdjust += OffsetSize;
        InstrOffset += PushInstrSize;
        break;
      }
      }
    }

    StackAdjust /= StackDivide;

    if (HasFP) {
      if ((StackAdjust & 0xFF) != StackAdjust)
        // Offset was too big for a compact unwind encoding.
        return CU::UNWIND_MODE_DWARF;

      // Get the encoding of the saved registers when we have a frame pointer.
      uint32_t RegEnc = encodeCompactUnwindRegistersWithFrame();
      if (RegEnc == ~0U) return CU::UNWIND_MODE_DWARF;

      CompactUnwindEncoding |= CU::UNWIND_MODE_BP_FRAME;
      CompactUnwindEncoding |= (StackAdjust & 0xFF) << 16;
      CompactUnwindEncoding |= RegEnc & CU::UNWIND_BP_FRAME_REGISTERS;
    } else {
      // If the amount of the stack allocation is the size of a register, then
      // we "push" the RAX/EAX register onto the stack instead of adjusting the
      // stack pointer with a SUB instruction. We don't support the push of the
      // RAX/EAX register with compact unwind. So we check for that situation
      // here.
      if ((NumDefCFAOffsets == SavedRegIdx + 1 &&
           StackSize - PrevStackSize == 1) ||
          (Instrs.size() == 1 && NumDefCFAOffsets == 1 && StackSize == 2))
        return CU::UNWIND_MODE_DWARF;

      SubtractInstrIdx += InstrOffset;
      ++StackAdjust;

      if ((StackSize & 0xFF) == StackSize) {
        // Frameless stack with a small stack size.
        CompactUnwindEncoding |= CU::UNWIND_MODE_STACK_IMMD;

        // Encode the stack size.
        CompactUnwindEncoding |= (StackSize & 0xFF) << 16;
      } else {
        if ((StackAdjust & 0x7) != StackAdjust)
          // The extra stack adjustments are too big for us to handle.
          return CU::UNWIND_MODE_DWARF;

        // Frameless stack with an offset too large for us to encode compactly.
        CompactUnwindEncoding |= CU::UNWIND_MODE_STACK_IND;

        // Encode the offset to the nnnnnn value in the 'subl $nnnnnn, ESP'
        // instruction.
        CompactUnwindEncoding |= (SubtractInstrIdx & 0xFF) << 16;

        // Encode any extra stack stack adjustments (done via push
        // instructions).
        CompactUnwindEncoding |= (StackAdjust & 0x7) << 13;
      }

      // Encode the number of registers saved. (Reverse the list first.)
      std::reverse(&SavedRegs[0], &SavedRegs[SavedRegIdx]);
      CompactUnwindEncoding |= (SavedRegIdx & 0x7) << 10;

      // Get the encoding of the saved registers when we don't have a frame
      // pointer.
      uint32_t RegEnc = encodeCompactUnwindRegistersWithoutFrame(SavedRegIdx);
      if (RegEnc == ~0U) return CU::UNWIND_MODE_DWARF;

      // Encode the register encoding.
      CompactUnwindEncoding |=
        RegEnc & CU::UNWIND_FRAMELESS_STACK_REG_PERMUTATION;
    }

    return CompactUnwindEncoding;
  }

private:
  /// \brief Get the compact unwind number for a given register. The number
  /// corresponds to the enum lists in compact_unwind_encoding.h.
  int getCompactUnwindRegNum(unsigned Reg) const {
    static const uint16_t CU32BitRegs[7] = {
      X86::EBX, X86::ECX, X86::EDX, X86::EDI, X86::ESI, X86::EBP, 0
    };
    static const uint16_t CU64BitRegs[] = {
      X86::RBX, X86::R12, X86::R13, X86::R14, X86::R15, X86::RBP, 0
    };
    const uint16_t *CURegs = Is64Bit ? CU64BitRegs : CU32BitRegs;
    for (int Idx = 1; *CURegs; ++CURegs, ++Idx)
      if (*CURegs == Reg)
        return Idx;

    return -1;
  }

  /// \brief Return the registers encoded for a compact encoding with a frame
  /// pointer.
  uint32_t encodeCompactUnwindRegistersWithFrame() const {
    // Encode the registers in the order they were saved --- 3-bits per
    // register. The list of saved registers is assumed to be in reverse
    // order. The registers are numbered from 1 to CU_NUM_SAVED_REGS.
    uint32_t RegEnc = 0;
    for (int i = 0, Idx = 0; i != CU_NUM_SAVED_REGS; ++i) {
      unsigned Reg = SavedRegs[i];
      if (Reg == 0) break;

      int CURegNum = getCompactUnwindRegNum(Reg);
      if (CURegNum == -1) return ~0U;

      // Encode the 3-bit register number in order, skipping over 3-bits for
      // each register.
      RegEnc |= (CURegNum & 0x7) << (Idx++ * 3);
    }

    assert((RegEnc & 0x3FFFF) == RegEnc &&
           "Invalid compact register encoding!");
    return RegEnc;
  }

  /// \brief Create the permutation encoding used with frameless stacks. It is
  /// passed the number of registers to be saved and an array of the registers
  /// saved.
  uint32_t encodeCompactUnwindRegistersWithoutFrame(unsigned RegCount) const {
    // The saved registers are numbered from 1 to 6. In order to encode the
    // order in which they were saved, we re-number them according to their
    // place in the register order. The re-numbering is relative to the last
    // re-numbered register. E.g., if we have registers {6, 2, 4, 5} saved in
    // that order:
    //
    //    Orig  Re-Num
    //    ----  ------
    //     6       6
    //     2       2
    //     4       3
    //     5       3
    //
    for (unsigned i = 0; i != CU_NUM_SAVED_REGS; ++i) {
      int CUReg = getCompactUnwindRegNum(SavedRegs[i]);
      if (CUReg == -1) return ~0U;
      SavedRegs[i] = CUReg;
    }

    // Reverse the list.
    std::reverse(&SavedRegs[0], &SavedRegs[CU_NUM_SAVED_REGS]);

    uint32_t RenumRegs[CU_NUM_SAVED_REGS];
    for (unsigned i = CU_NUM_SAVED_REGS - RegCount; i < CU_NUM_SAVED_REGS; ++i){
      unsigned Countless = 0;
      for (unsigned j = CU_NUM_SAVED_REGS - RegCount; j < i; ++j)
        if (SavedRegs[j] < SavedRegs[i])
          ++Countless;

      RenumRegs[i] = SavedRegs[i] - Countless - 1;
    }

    // Take the renumbered values and encode them into a 10-bit number.
    uint32_t permutationEncoding = 0;
    switch (RegCount) {
    case 6:
      permutationEncoding |= 120 * RenumRegs[0] + 24 * RenumRegs[1]
                             + 6 * RenumRegs[2] +  2 * RenumRegs[3]
                             +     RenumRegs[4];
      break;
    case 5:
      permutationEncoding |= 120 * RenumRegs[1] + 24 * RenumRegs[2]
                             + 6 * RenumRegs[3] +  2 * RenumRegs[4]
                             +     RenumRegs[5];
      break;
    case 4:
      permutationEncoding |=  60 * RenumRegs[2] + 12 * RenumRegs[3]
                             + 3 * RenumRegs[4] +      RenumRegs[5];
      break;
    case 3:
      permutationEncoding |=  20 * RenumRegs[3] +  4 * RenumRegs[4]
                             +     RenumRegs[5];
      break;
    case 2:
      permutationEncoding |=   5 * RenumRegs[4] +      RenumRegs[5];
      break;
    case 1:
      permutationEncoding |=       RenumRegs[5];
      break;
    }

    assert((permutationEncoding & 0x3FF) == permutationEncoding &&
           "Invalid compact register encoding!");
    return permutationEncoding;
  }

public:
  DarwinX86AsmBackend(const Target &T, const MCRegisterInfo &MRI, StringRef CPU,
                      bool Is64Bit)
    : X86AsmBackend(T, CPU), MRI(MRI), Is64Bit(Is64Bit) {
    memset(SavedRegs, 0, sizeof(SavedRegs));
    OffsetSize = Is64Bit ? 8 : 4;
    MoveInstrSize = Is64Bit ? 3 : 2;
    StackDivide = Is64Bit ? 8 : 4;
    PushInstrSize = 1;
  }
};

class DarwinX86_32AsmBackend : public DarwinX86AsmBackend {
  bool SupportsCU;
public:
  DarwinX86_32AsmBackend(const Target &T, const MCRegisterInfo &MRI,
                         StringRef CPU, bool SupportsCU)
    : DarwinX86AsmBackend(T, MRI, CPU, false), SupportsCU(SupportsCU) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createX86MachObjectWriter(OS, /*Is64Bit=*/false,
                                     MachO::CPU_TYPE_I386,
                                     MachO::CPU_SUBTYPE_I386_ALL);
  }

  /// \brief Generate the compact unwind encoding for the CFI instructions.
  virtual uint32_t
  generateCompactUnwindEncoding(ArrayRef<MCCFIInstruction> Instrs) const {
    return SupportsCU ? generateCompactUnwindEncodingImpl(Instrs) : 0;
  }
};

class DarwinX86_64AsmBackend : public DarwinX86AsmBackend {
  bool SupportsCU;
  const MachO::CPUSubTypeX86 Subtype;
public:
  DarwinX86_64AsmBackend(const Target &T, const MCRegisterInfo &MRI,
                         StringRef CPU, bool SupportsCU,
                         MachO::CPUSubTypeX86 st)
    : DarwinX86AsmBackend(T, MRI, CPU, true), SupportsCU(SupportsCU),
      Subtype(st) {
    HasReliableSymbolDifference = true;
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createX86MachObjectWriter(OS, /*Is64Bit=*/true,
                                     MachO::CPU_TYPE_X86_64, Subtype);
  }

  virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
    // Temporary labels in the string literals sections require symbols. The
    // issue is that the x86_64 relocation format does not allow symbol +
    // offset, and so the linker does not have enough information to resolve the
    // access to the appropriate atom unless an external relocation is used. For
    // non-cstring sections, we expect the compiler to use a non-temporary label
    // for anything that could have an addend pointing outside the symbol.
    //
    // See <rdar://problem/4765733>.
    const MCSectionMachO &SMO = static_cast<const MCSectionMachO&>(Section);
    return SMO.getType() == MCSectionMachO::S_CSTRING_LITERALS;
  }

  virtual bool isSectionAtomizable(const MCSection &Section) const {
    const MCSectionMachO &SMO = static_cast<const MCSectionMachO&>(Section);
    // Fixed sized data sections are uniqued, they cannot be diced into atoms.
    switch (SMO.getType()) {
    default:
      return true;

    case MCSectionMachO::S_4BYTE_LITERALS:
    case MCSectionMachO::S_8BYTE_LITERALS:
    case MCSectionMachO::S_16BYTE_LITERALS:
    case MCSectionMachO::S_LITERAL_POINTERS:
    case MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS:
    case MCSectionMachO::S_LAZY_SYMBOL_POINTERS:
    case MCSectionMachO::S_MOD_INIT_FUNC_POINTERS:
    case MCSectionMachO::S_MOD_TERM_FUNC_POINTERS:
    case MCSectionMachO::S_INTERPOSING:
      return false;
    }
  }

  /// \brief Generate the compact unwind encoding for the CFI instructions.
  virtual uint32_t
  generateCompactUnwindEncoding(ArrayRef<MCCFIInstruction> Instrs) const {
    return SupportsCU ? generateCompactUnwindEncodingImpl(Instrs) : 0;
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createX86_32AsmBackend(const Target &T,
                                           const MCRegisterInfo &MRI,
                                           StringRef TT,
                                           StringRef CPU) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin() || TheTriple.getEnvironment() == Triple::MachO)
    return new DarwinX86_32AsmBackend(T, MRI, CPU,
                                      TheTriple.isMacOSX() &&
                                      !TheTriple.isMacOSXVersionLT(10, 7));

  if (TheTriple.isOSWindows() && TheTriple.getEnvironment() != Triple::ELF)
    return new WindowsX86AsmBackend(T, false, CPU);

  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TheTriple.getOS());
  return new ELFX86_32AsmBackend(T, OSABI, CPU);
}

MCAsmBackend *llvm::createX86_64AsmBackend(const Target &T,
                                           const MCRegisterInfo &MRI,
                                           StringRef TT,
                                           StringRef CPU) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin() || TheTriple.getEnvironment() == Triple::MachO) {
    MachO::CPUSubTypeX86 CS =
        StringSwitch<MachO::CPUSubTypeX86>(TheTriple.getArchName())
            .Case("x86_64h", MachO::CPU_SUBTYPE_X86_64_H)
            .Default(MachO::CPU_SUBTYPE_X86_64_ALL);
    return new DarwinX86_64AsmBackend(T, MRI, CPU,
                                      TheTriple.isMacOSX() &&
                                      !TheTriple.isMacOSXVersionLT(10, 7), CS);
  }

  if (TheTriple.isOSWindows() && TheTriple.getEnvironment() != Triple::ELF)
    return new WindowsX86AsmBackend(T, true, CPU);

  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TheTriple.getOS());
  return new ELFX86_64AsmBackend(T, OSABI, CPU);
}
