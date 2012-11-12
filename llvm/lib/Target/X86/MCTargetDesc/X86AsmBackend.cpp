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
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
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
public:
  X86AsmBackend(const Target &T, StringRef _CPU)
    : MCAsmBackend(), CPU(_CPU) {}

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
                            const MCInstFragment *DF,
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
                                         const MCInstFragment *DF,
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

/// writeNopData - Write optimal nops to the output file for the \p Count
/// bytes.  This returns the number of bytes written.  It may return 0 if
/// the \p Count is more than the maximum optimal nops.
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
  if (CPU == "generic" || CPU == "i386" || CPU == "i486" || CPU == "i586" ||
      CPU == "pentium" || CPU == "pentium-mmx" || CPU == "geode") {
    for (uint64_t i = 0; i < Count; ++i)
      OW->Write8(0x90);
    return true;
  }

  // Write an optimal sequence for the first 15 bytes.
  const uint64_t OptimalCount = (Count < 16) ? Count : 15;
  const uint64_t Prefixes = OptimalCount <= 10 ? 0 : OptimalCount - 10;
  for (uint64_t i = 0, e = Prefixes; i != e; i++)
    OW->Write8(0x66);
  const uint64_t Rest = OptimalCount - Prefixes;
  for (uint64_t i = 0, e = Rest; i != e; i++)
    OW->Write8(Nops[Rest - 1][i]);

  // Finish with single byte nops.
  for (uint64_t i = OptimalCount, e = Count; i != e; ++i)
   OW->Write8(0x90);

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

class DarwinX86AsmBackend : public X86AsmBackend {
public:
  DarwinX86AsmBackend(const Target &T, StringRef CPU)
    : X86AsmBackend(T, CPU) { }
};

class DarwinX86_32AsmBackend : public DarwinX86AsmBackend {
public:
  DarwinX86_32AsmBackend(const Target &T, StringRef CPU)
    : DarwinX86AsmBackend(T, CPU) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createX86MachObjectWriter(OS, /*Is64Bit=*/false,
                                     object::mach::CTM_i386,
                                     object::mach::CSX86_ALL);
  }
};

class DarwinX86_64AsmBackend : public DarwinX86AsmBackend {
public:
  DarwinX86_64AsmBackend(const Target &T, StringRef CPU)
    : DarwinX86AsmBackend(T, CPU) {
    HasReliableSymbolDifference = true;
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createX86MachObjectWriter(OS, /*Is64Bit=*/true,
                                     object::mach::CTM_x86_64,
                                     object::mach::CSX86_ALL);
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
};

} // end anonymous namespace

MCAsmBackend *llvm::createX86_32AsmBackend(const Target &T, StringRef TT, StringRef CPU) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin() || TheTriple.getEnvironment() == Triple::MachO)
    return new DarwinX86_32AsmBackend(T, CPU);

  if (TheTriple.isOSWindows() && TheTriple.getEnvironment() != Triple::ELF)
    return new WindowsX86AsmBackend(T, false, CPU);

  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TheTriple.getOS());
  return new ELFX86_32AsmBackend(T, OSABI, CPU);
}

MCAsmBackend *llvm::createX86_64AsmBackend(const Target &T, StringRef TT, StringRef CPU) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin() || TheTriple.getEnvironment() == Triple::MachO)
    return new DarwinX86_64AsmBackend(T, CPU);

  if (TheTriple.isOSWindows() && TheTriple.getEnvironment() != Triple::ELF)
    return new WindowsX86AsmBackend(T, true, CPU);

  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TheTriple.getOS());
  return new ELFX86_64AsmBackend(T, OSABI, CPU);
}
