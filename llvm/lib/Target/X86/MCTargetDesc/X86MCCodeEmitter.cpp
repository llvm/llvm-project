//===-- X86MCCodeEmitter.cpp - Convert X86 code to machine code -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86MCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mccodeemitter"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86FixupKinds.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class X86MCCodeEmitter : public MCCodeEmitter {
  X86MCCodeEmitter(const X86MCCodeEmitter &) LLVM_DELETED_FUNCTION;
  void operator=(const X86MCCodeEmitter &) LLVM_DELETED_FUNCTION;
  const MCInstrInfo &MCII;
  const MCSubtargetInfo &STI;
  MCContext &Ctx;
public:
  X86MCCodeEmitter(const MCInstrInfo &mcii, const MCSubtargetInfo &sti,
                   MCContext &ctx)
    : MCII(mcii), STI(sti), Ctx(ctx) {
  }

  ~X86MCCodeEmitter() {}

  bool is64BitMode() const {
    // FIXME: Can tablegen auto-generate this?
    return (STI.getFeatureBits() & X86::Mode64Bit) != 0;
  }

  bool is32BitMode() const {
    // FIXME: Can tablegen auto-generate this?
    return (STI.getFeatureBits() & X86::Mode64Bit) == 0;
  }

  unsigned GetX86RegNum(const MCOperand &MO) const {
    return Ctx.getRegisterInfo().getEncodingValue(MO.getReg()) & 0x7;
  }

  // On regular x86, both XMM0-XMM7 and XMM8-XMM15 are encoded in the range
  // 0-7 and the difference between the 2 groups is given by the REX prefix.
  // In the VEX prefix, registers are seen sequencially from 0-15 and encoded
  // in 1's complement form, example:
  //
  //  ModRM field => XMM9 => 1
  //  VEX.VVVV    => XMM9 => ~9
  //
  // See table 4-35 of Intel AVX Programming Reference for details.
  unsigned char getVEXRegisterEncoding(const MCInst &MI,
                                       unsigned OpNum) const {
    unsigned SrcReg = MI.getOperand(OpNum).getReg();
    unsigned SrcRegNum = GetX86RegNum(MI.getOperand(OpNum));
    if (X86II::isX86_64ExtendedReg(SrcReg))
      SrcRegNum |= 8;

    // The registers represented through VEX_VVVV should
    // be encoded in 1's complement form.
    return (~SrcRegNum) & 0xf;
  }

  void EmitByte(unsigned char C, unsigned &CurByte, raw_ostream &OS) const {
    OS << (char)C;
    ++CurByte;
  }

  void EmitConstant(uint64_t Val, unsigned Size, unsigned &CurByte,
                    raw_ostream &OS) const {
    // Output the constant in little endian byte order.
    for (unsigned i = 0; i != Size; ++i) {
      EmitByte(Val & 255, CurByte, OS);
      Val >>= 8;
    }
  }

  void EmitImmediate(const MCOperand &Disp, SMLoc Loc,
                     unsigned ImmSize, MCFixupKind FixupKind,
                     unsigned &CurByte, raw_ostream &OS,
                     SmallVectorImpl<MCFixup> &Fixups,
                     int ImmOffset = 0) const;

  inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                        unsigned RM) {
    assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
    return RM | (RegOpcode << 3) | (Mod << 6);
  }

  void EmitRegModRMByte(const MCOperand &ModRMReg, unsigned RegOpcodeFld,
                        unsigned &CurByte, raw_ostream &OS) const {
    EmitByte(ModRMByte(3, RegOpcodeFld, GetX86RegNum(ModRMReg)), CurByte, OS);
  }

  void EmitSIBByte(unsigned SS, unsigned Index, unsigned Base,
                   unsigned &CurByte, raw_ostream &OS) const {
    // SIB byte is in the same format as the ModRMByte.
    EmitByte(ModRMByte(SS, Index, Base), CurByte, OS);
  }


  void EmitMemModRMByte(const MCInst &MI, unsigned Op,
                        unsigned RegOpcodeField,
                        uint64_t TSFlags, unsigned &CurByte, raw_ostream &OS,
                        SmallVectorImpl<MCFixup> &Fixups) const;

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;

  void EmitVEXOpcodePrefix(uint64_t TSFlags, unsigned &CurByte, int MemOperand,
                           const MCInst &MI, const MCInstrDesc &Desc,
                           raw_ostream &OS) const;

  void EmitSegmentOverridePrefix(uint64_t TSFlags, unsigned &CurByte,
                                 int MemOperand, const MCInst &MI,
                                 raw_ostream &OS) const;

  void EmitOpcodePrefix(uint64_t TSFlags, unsigned &CurByte, int MemOperand,
                        const MCInst &MI, const MCInstrDesc &Desc,
                        raw_ostream &OS) const;
};

} // end anonymous namespace


MCCodeEmitter *llvm::createX86MCCodeEmitter(const MCInstrInfo &MCII,
                                            const MCRegisterInfo &MRI,
                                            const MCSubtargetInfo &STI,
                                            MCContext &Ctx) {
  return new X86MCCodeEmitter(MCII, STI, Ctx);
}

/// isDisp8 - Return true if this signed displacement fits in a 8-bit
/// sign-extended field.
static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

/// getImmFixupKind - Return the appropriate fixup kind to use for an immediate
/// in an instruction with the specified TSFlags.
static MCFixupKind getImmFixupKind(uint64_t TSFlags) {
  unsigned Size = X86II::getSizeOfImm(TSFlags);
  bool isPCRel = X86II::isImmPCRel(TSFlags);

  return MCFixup::getKindForSize(Size, isPCRel);
}

/// Is32BitMemOperand - Return true if the specified instruction has
/// a 32-bit memory operand. Op specifies the operand # of the memoperand.
static bool Is32BitMemOperand(const MCInst &MI, unsigned Op) {
  const MCOperand &BaseReg  = MI.getOperand(Op+X86::AddrBaseReg);
  const MCOperand &IndexReg = MI.getOperand(Op+X86::AddrIndexReg);

  if ((BaseReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR32RegClassID].contains(BaseReg.getReg())) ||
      (IndexReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR32RegClassID].contains(IndexReg.getReg())))
    return true;
  return false;
}

/// Is64BitMemOperand - Return true if the specified instruction has
/// a 64-bit memory operand. Op specifies the operand # of the memoperand.
#ifndef NDEBUG
static bool Is64BitMemOperand(const MCInst &MI, unsigned Op) {
  const MCOperand &BaseReg  = MI.getOperand(Op+X86::AddrBaseReg);
  const MCOperand &IndexReg = MI.getOperand(Op+X86::AddrIndexReg);

  if ((BaseReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR64RegClassID].contains(BaseReg.getReg())) ||
      (IndexReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR64RegClassID].contains(IndexReg.getReg())))
    return true;
  return false;
}
#endif

/// Is16BitMemOperand - Return true if the specified instruction has
/// a 16-bit memory operand. Op specifies the operand # of the memoperand.
static bool Is16BitMemOperand(const MCInst &MI, unsigned Op) {
  const MCOperand &BaseReg  = MI.getOperand(Op+X86::AddrBaseReg);
  const MCOperand &IndexReg = MI.getOperand(Op+X86::AddrIndexReg);

  if ((BaseReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR16RegClassID].contains(BaseReg.getReg())) ||
      (IndexReg.getReg() != 0 &&
       X86MCRegisterClasses[X86::GR16RegClassID].contains(IndexReg.getReg())))
    return true;
  return false;
}

/// StartsWithGlobalOffsetTable - Check if this expression starts with
///  _GLOBAL_OFFSET_TABLE_ and if it is of the form
///  _GLOBAL_OFFSET_TABLE_-symbol. This is needed to support PIC on ELF
/// i386 as _GLOBAL_OFFSET_TABLE_ is magical. We check only simple case that
/// are know to be used: _GLOBAL_OFFSET_TABLE_ by itself or at the start
/// of a binary expression.
enum GlobalOffsetTableExprKind {
  GOT_None,
  GOT_Normal,
  GOT_SymDiff
};
static GlobalOffsetTableExprKind
StartsWithGlobalOffsetTable(const MCExpr *Expr) {
  const MCExpr *RHS = 0;
  if (Expr->getKind() == MCExpr::Binary) {
    const MCBinaryExpr *BE = static_cast<const MCBinaryExpr *>(Expr);
    Expr = BE->getLHS();
    RHS = BE->getRHS();
  }

  if (Expr->getKind() != MCExpr::SymbolRef)
    return GOT_None;

  const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr*>(Expr);
  const MCSymbol &S = Ref->getSymbol();
  if (S.getName() != "_GLOBAL_OFFSET_TABLE_")
    return GOT_None;
  if (RHS && RHS->getKind() == MCExpr::SymbolRef)
    return GOT_SymDiff;
  return GOT_Normal;
}

void X86MCCodeEmitter::
EmitImmediate(const MCOperand &DispOp, SMLoc Loc, unsigned Size,
              MCFixupKind FixupKind, unsigned &CurByte, raw_ostream &OS,
              SmallVectorImpl<MCFixup> &Fixups, int ImmOffset) const {
  const MCExpr *Expr = NULL;
  if (DispOp.isImm()) {
    // If this is a simple integer displacement that doesn't require a
    // relocation, emit it now.
    if (FixupKind != FK_PCRel_1 &&
        FixupKind != FK_PCRel_2 &&
        FixupKind != FK_PCRel_4) {
      EmitConstant(DispOp.getImm()+ImmOffset, Size, CurByte, OS);
      return;
    }
    Expr = MCConstantExpr::Create(DispOp.getImm(), Ctx);
  } else {
    Expr = DispOp.getExpr();
  }

  // If we have an immoffset, add it to the expression.
  if ((FixupKind == FK_Data_4 ||
       FixupKind == FK_Data_8 ||
       FixupKind == MCFixupKind(X86::reloc_signed_4byte))) {
    GlobalOffsetTableExprKind Kind = StartsWithGlobalOffsetTable(Expr);
    if (Kind != GOT_None) {
      assert(ImmOffset == 0);

      FixupKind = MCFixupKind(X86::reloc_global_offset_table);
      if (Kind == GOT_Normal)
        ImmOffset = CurByte;
    } else if (Expr->getKind() == MCExpr::SymbolRef) {
      const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr*>(Expr);
      if (Ref->getKind() == MCSymbolRefExpr::VK_SECREL) {
        FixupKind = MCFixupKind(FK_SecRel_4);
      }
    }
  }

  // If the fixup is pc-relative, we need to bias the value to be relative to
  // the start of the field, not the end of the field.
  if (FixupKind == FK_PCRel_4 ||
      FixupKind == MCFixupKind(X86::reloc_riprel_4byte) ||
      FixupKind == MCFixupKind(X86::reloc_riprel_4byte_movq_load))
    ImmOffset -= 4;
  if (FixupKind == FK_PCRel_2)
    ImmOffset -= 2;
  if (FixupKind == FK_PCRel_1)
    ImmOffset -= 1;

  if (ImmOffset)
    Expr = MCBinaryExpr::CreateAdd(Expr, MCConstantExpr::Create(ImmOffset, Ctx),
                                   Ctx);

  // Emit a symbolic constant as a fixup and 4 zeros.
  Fixups.push_back(MCFixup::Create(CurByte, Expr, FixupKind, Loc));
  EmitConstant(0, Size, CurByte, OS);
}

void X86MCCodeEmitter::EmitMemModRMByte(const MCInst &MI, unsigned Op,
                                        unsigned RegOpcodeField,
                                        uint64_t TSFlags, unsigned &CurByte,
                                        raw_ostream &OS,
                                        SmallVectorImpl<MCFixup> &Fixups) const{
  const MCOperand &Disp     = MI.getOperand(Op+X86::AddrDisp);
  const MCOperand &Base     = MI.getOperand(Op+X86::AddrBaseReg);
  const MCOperand &Scale    = MI.getOperand(Op+X86::AddrScaleAmt);
  const MCOperand &IndexReg = MI.getOperand(Op+X86::AddrIndexReg);
  unsigned BaseReg = Base.getReg();

  // Handle %rip relative addressing.
  if (BaseReg == X86::RIP) {    // [disp32+RIP] in X86-64 mode
    assert(is64BitMode() && "Rip-relative addressing requires 64-bit mode");
    assert(IndexReg.getReg() == 0 && "Invalid rip-relative address");
    EmitByte(ModRMByte(0, RegOpcodeField, 5), CurByte, OS);

    unsigned FixupKind = X86::reloc_riprel_4byte;

    // movq loads are handled with a special relocation form which allows the
    // linker to eliminate some loads for GOT references which end up in the
    // same linkage unit.
    if (MI.getOpcode() == X86::MOV64rm)
      FixupKind = X86::reloc_riprel_4byte_movq_load;

    // rip-relative addressing is actually relative to the *next* instruction.
    // Since an immediate can follow the mod/rm byte for an instruction, this
    // means that we need to bias the immediate field of the instruction with
    // the size of the immediate field.  If we have this case, add it into the
    // expression to emit.
    int ImmSize = X86II::hasImm(TSFlags) ? X86II::getSizeOfImm(TSFlags) : 0;

    EmitImmediate(Disp, MI.getLoc(), 4, MCFixupKind(FixupKind),
                  CurByte, OS, Fixups, -ImmSize);
    return;
  }

  unsigned BaseRegNo = BaseReg ? GetX86RegNum(Base) : -1U;

  // Determine whether a SIB byte is needed.
  // If no BaseReg, issue a RIP relative instruction only if the MCE can
  // resolve addresses on-the-fly, otherwise use SIB (Intel Manual 2A, table
  // 2-7) and absolute references.

  if (// The SIB byte must be used if there is an index register.
      IndexReg.getReg() == 0 &&
      // The SIB byte must be used if the base is ESP/RSP/R12, all of which
      // encode to an R/M value of 4, which indicates that a SIB byte is
      // present.
      BaseRegNo != N86::ESP &&
      // If there is no base register and we're in 64-bit mode, we need a SIB
      // byte to emit an addr that is just 'disp32' (the non-RIP relative form).
      (!is64BitMode() || BaseReg != 0)) {

    if (BaseReg == 0) {          // [disp32]     in X86-32 mode
      EmitByte(ModRMByte(0, RegOpcodeField, 5), CurByte, OS);
      EmitImmediate(Disp, MI.getLoc(), 4, FK_Data_4, CurByte, OS, Fixups);
      return;
    }

    // If the base is not EBP/ESP and there is no displacement, use simple
    // indirect register encoding, this handles addresses like [EAX].  The
    // encoding for [EBP] with no displacement means [disp32] so we handle it
    // by emitting a displacement of 0 below.
    if (Disp.isImm() && Disp.getImm() == 0 && BaseRegNo != N86::EBP) {
      EmitByte(ModRMByte(0, RegOpcodeField, BaseRegNo), CurByte, OS);
      return;
    }

    // Otherwise, if the displacement fits in a byte, encode as [REG+disp8].
    if (Disp.isImm() && isDisp8(Disp.getImm())) {
      EmitByte(ModRMByte(1, RegOpcodeField, BaseRegNo), CurByte, OS);
      EmitImmediate(Disp, MI.getLoc(), 1, FK_Data_1, CurByte, OS, Fixups);
      return;
    }

    // Otherwise, emit the most general non-SIB encoding: [REG+disp32]
    EmitByte(ModRMByte(2, RegOpcodeField, BaseRegNo), CurByte, OS);
    EmitImmediate(Disp, MI.getLoc(), 4, MCFixupKind(X86::reloc_signed_4byte), CurByte, OS,
                  Fixups);
    return;
  }

  // We need a SIB byte, so start by outputting the ModR/M byte first
  assert(IndexReg.getReg() != X86::ESP &&
         IndexReg.getReg() != X86::RSP && "Cannot use ESP as index reg!");

  bool ForceDisp32 = false;
  bool ForceDisp8  = false;
  if (BaseReg == 0) {
    // If there is no base register, we emit the special case SIB byte with
    // MOD=0, BASE=5, to JUST get the index, scale, and displacement.
    EmitByte(ModRMByte(0, RegOpcodeField, 4), CurByte, OS);
    ForceDisp32 = true;
  } else if (!Disp.isImm()) {
    // Emit the normal disp32 encoding.
    EmitByte(ModRMByte(2, RegOpcodeField, 4), CurByte, OS);
    ForceDisp32 = true;
  } else if (Disp.getImm() == 0 &&
             // Base reg can't be anything that ends up with '5' as the base
             // reg, it is the magic [*] nomenclature that indicates no base.
             BaseRegNo != N86::EBP) {
    // Emit no displacement ModR/M byte
    EmitByte(ModRMByte(0, RegOpcodeField, 4), CurByte, OS);
  } else if (isDisp8(Disp.getImm())) {
    // Emit the disp8 encoding.
    EmitByte(ModRMByte(1, RegOpcodeField, 4), CurByte, OS);
    ForceDisp8 = true;           // Make sure to force 8 bit disp if Base=EBP
  } else {
    // Emit the normal disp32 encoding.
    EmitByte(ModRMByte(2, RegOpcodeField, 4), CurByte, OS);
  }

  // Calculate what the SS field value should be...
  static const unsigned SSTable[] = { ~0U, 0, 1, ~0U, 2, ~0U, ~0U, ~0U, 3 };
  unsigned SS = SSTable[Scale.getImm()];

  if (BaseReg == 0) {
    // Handle the SIB byte for the case where there is no base, see Intel
    // Manual 2A, table 2-7. The displacement has already been output.
    unsigned IndexRegNo;
    if (IndexReg.getReg())
      IndexRegNo = GetX86RegNum(IndexReg);
    else // Examples: [ESP+1*<noreg>+4] or [scaled idx]+disp32 (MOD=0,BASE=5)
      IndexRegNo = 4;
    EmitSIBByte(SS, IndexRegNo, 5, CurByte, OS);
  } else {
    unsigned IndexRegNo;
    if (IndexReg.getReg())
      IndexRegNo = GetX86RegNum(IndexReg);
    else
      IndexRegNo = 4;   // For example [ESP+1*<noreg>+4]
    EmitSIBByte(SS, IndexRegNo, GetX86RegNum(Base), CurByte, OS);
  }

  // Do we need to output a displacement?
  if (ForceDisp8)
    EmitImmediate(Disp, MI.getLoc(), 1, FK_Data_1, CurByte, OS, Fixups);
  else if (ForceDisp32 || Disp.getImm() != 0)
    EmitImmediate(Disp, MI.getLoc(), 4, MCFixupKind(X86::reloc_signed_4byte),
                  CurByte, OS, Fixups);
}

/// EmitVEXOpcodePrefix - AVX instructions are encoded using a opcode prefix
/// called VEX.
void X86MCCodeEmitter::EmitVEXOpcodePrefix(uint64_t TSFlags, unsigned &CurByte,
                                           int MemOperand, const MCInst &MI,
                                           const MCInstrDesc &Desc,
                                           raw_ostream &OS) const {
  bool HasVEX_4V = (TSFlags >> X86II::VEXShift) & X86II::VEX_4V;
  bool HasVEX_4VOp3 = (TSFlags >> X86II::VEXShift) & X86II::VEX_4VOp3;

  // VEX_R: opcode externsion equivalent to REX.R in
  // 1's complement (inverted) form
  //
  //  1: Same as REX_R=0 (must be 1 in 32-bit mode)
  //  0: Same as REX_R=1 (64 bit mode only)
  //
  unsigned char VEX_R = 0x1;

  // VEX_X: equivalent to REX.X, only used when a
  // register is used for index in SIB Byte.
  //
  //  1: Same as REX.X=0 (must be 1 in 32-bit mode)
  //  0: Same as REX.X=1 (64-bit mode only)
  unsigned char VEX_X = 0x1;

  // VEX_B:
  //
  //  1: Same as REX_B=0 (ignored in 32-bit mode)
  //  0: Same as REX_B=1 (64 bit mode only)
  //
  unsigned char VEX_B = 0x1;

  // VEX_W: opcode specific (use like REX.W, or used for
  // opcode extension, or ignored, depending on the opcode byte)
  unsigned char VEX_W = 0;

  // XOP: Use XOP prefix byte 0x8f instead of VEX.
  unsigned char XOP = 0;

  // VEX_5M (VEX m-mmmmm field):
  //
  //  0b00000: Reserved for future use
  //  0b00001: implied 0F leading opcode
  //  0b00010: implied 0F 38 leading opcode bytes
  //  0b00011: implied 0F 3A leading opcode bytes
  //  0b00100-0b11111: Reserved for future use
  //  0b01000: XOP map select - 08h instructions with imm byte
  //  0b10001: XOP map select - 09h instructions with no imm byte
  unsigned char VEX_5M = 0x1;

  // VEX_4V (VEX vvvv field): a register specifier
  // (in 1's complement form) or 1111 if unused.
  unsigned char VEX_4V = 0xf;

  // VEX_L (Vector Length):
  //
  //  0: scalar or 128-bit vector
  //  1: 256-bit vector
  //
  unsigned char VEX_L = 0;

  // VEX_PP: opcode extension providing equivalent
  // functionality of a SIMD prefix
  //
  //  0b00: None
  //  0b01: 66
  //  0b10: F3
  //  0b11: F2
  //
  unsigned char VEX_PP = 0;

  // Encode the operand size opcode prefix as needed.
  if (TSFlags & X86II::OpSize)
    VEX_PP = 0x01;

  if ((TSFlags >> X86II::VEXShift) & X86II::VEX_W)
    VEX_W = 1;

  if ((TSFlags >> X86II::VEXShift) & X86II::XOP)
    XOP = 1;

  if ((TSFlags >> X86II::VEXShift) & X86II::VEX_L)
    VEX_L = 1;

  switch (TSFlags & X86II::Op0Mask) {
  default: llvm_unreachable("Invalid prefix!");
  case X86II::T8:  // 0F 38
    VEX_5M = 0x2;
    break;
  case X86II::TA:  // 0F 3A
    VEX_5M = 0x3;
    break;
  case X86II::T8XS: // F3 0F 38
    VEX_PP = 0x2;
    VEX_5M = 0x2;
    break;
  case X86II::T8XD: // F2 0F 38
    VEX_PP = 0x3;
    VEX_5M = 0x2;
    break;
  case X86II::TAXD: // F2 0F 3A
    VEX_PP = 0x3;
    VEX_5M = 0x3;
    break;
  case X86II::XS:  // F3 0F
    VEX_PP = 0x2;
    break;
  case X86II::XD:  // F2 0F
    VEX_PP = 0x3;
    break;
  case X86II::XOP8:
    VEX_5M = 0x8;
    break;
  case X86II::XOP9:
    VEX_5M = 0x9;
    break;
  case X86II::A6:  // Bypass: Not used by VEX
  case X86II::A7:  // Bypass: Not used by VEX
  case X86II::TB:  // Bypass: Not used by VEX
  case 0:
    break;  // No prefix!
  }


  // Classify VEX_B, VEX_4V, VEX_R, VEX_X
  unsigned NumOps = Desc.getNumOperands();
  unsigned CurOp = 0;
  if (NumOps > 1 && Desc.getOperandConstraint(1, MCOI::TIED_TO) == 0)
    ++CurOp;
  else if (NumOps > 3 && Desc.getOperandConstraint(2, MCOI::TIED_TO) == 0) {
    assert(Desc.getOperandConstraint(NumOps - 1, MCOI::TIED_TO) == 1);
    // Special case for GATHER with 2 TIED_TO operands
    // Skip the first 2 operands: dst, mask_wb
    CurOp += 2;
  }

  switch (TSFlags & X86II::FormMask) {
  case X86II::MRMInitReg: llvm_unreachable("FIXME: Remove this!");
  case X86II::MRMDestMem: {
    // MRMDestMem instructions forms:
    //  MemAddr, src1(ModR/M)
    //  MemAddr, src1(VEX_4V), src2(ModR/M)
    //  MemAddr, src1(ModR/M), imm8
    //
    if (X86II::isX86_64ExtendedReg(MI.getOperand(X86::AddrBaseReg).getReg()))
      VEX_B = 0x0;
    if (X86II::isX86_64ExtendedReg(MI.getOperand(X86::AddrIndexReg).getReg()))
      VEX_X = 0x0;

    CurOp = X86::AddrNumOperands;
    if (HasVEX_4V)
      VEX_4V = getVEXRegisterEncoding(MI, CurOp++);

    const MCOperand &MO = MI.getOperand(CurOp);
    if (MO.isReg() && X86II::isX86_64ExtendedReg(MO.getReg()))
      VEX_R = 0x0;
    break;
  }
  case X86II::MRMSrcMem:
    // MRMSrcMem instructions forms:
    //  src1(ModR/M), MemAddr
    //  src1(ModR/M), src2(VEX_4V), MemAddr
    //  src1(ModR/M), MemAddr, imm8
    //  src1(ModR/M), MemAddr, src2(VEX_I8IMM)
    //
    //  FMA4:
    //  dst(ModR/M.reg), src1(VEX_4V), src2(ModR/M), src3(VEX_I8IMM)
    //  dst(ModR/M.reg), src1(VEX_4V), src2(VEX_I8IMM), src3(ModR/M),
    if (X86II::isX86_64ExtendedReg(MI.getOperand(CurOp++).getReg()))
      VEX_R = 0x0;

    if (HasVEX_4V)
      VEX_4V = getVEXRegisterEncoding(MI, CurOp);

    if (X86II::isX86_64ExtendedReg(
               MI.getOperand(MemOperand+X86::AddrBaseReg).getReg()))
      VEX_B = 0x0;
    if (X86II::isX86_64ExtendedReg(
               MI.getOperand(MemOperand+X86::AddrIndexReg).getReg()))
      VEX_X = 0x0;

    if (HasVEX_4VOp3)
      // Instruction format for 4VOp3:
      //   src1(ModR/M), MemAddr, src3(VEX_4V)
      // CurOp points to start of the MemoryOperand,
      //   it skips TIED_TO operands if exist, then increments past src1.
      // CurOp + X86::AddrNumOperands will point to src3.
      VEX_4V = getVEXRegisterEncoding(MI, CurOp+X86::AddrNumOperands);
    break;
  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m: {
    // MRM[0-9]m instructions forms:
    //  MemAddr
    //  src1(VEX_4V), MemAddr
    if (HasVEX_4V)
      VEX_4V = getVEXRegisterEncoding(MI, 0);

    if (X86II::isX86_64ExtendedReg(
               MI.getOperand(MemOperand+X86::AddrBaseReg).getReg()))
      VEX_B = 0x0;
    if (X86II::isX86_64ExtendedReg(
               MI.getOperand(MemOperand+X86::AddrIndexReg).getReg()))
      VEX_X = 0x0;
    break;
  }
  case X86II::MRMSrcReg:
    // MRMSrcReg instructions forms:
    //  dst(ModR/M), src1(VEX_4V), src2(ModR/M), src3(VEX_I8IMM)
    //  dst(ModR/M), src1(ModR/M)
    //  dst(ModR/M), src1(ModR/M), imm8
    //
    if (X86II::isX86_64ExtendedReg(MI.getOperand(CurOp).getReg()))
      VEX_R = 0x0;
    CurOp++;

    if (HasVEX_4V)
      VEX_4V = getVEXRegisterEncoding(MI, CurOp++);
    if (X86II::isX86_64ExtendedReg(MI.getOperand(CurOp).getReg()))
      VEX_B = 0x0;
    CurOp++;
    if (HasVEX_4VOp3)
      VEX_4V = getVEXRegisterEncoding(MI, CurOp);
    break;
  case X86II::MRMDestReg:
    // MRMDestReg instructions forms:
    //  dst(ModR/M), src(ModR/M)
    //  dst(ModR/M), src(ModR/M), imm8
    if (X86II::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      VEX_B = 0x0;
    if (X86II::isX86_64ExtendedReg(MI.getOperand(1).getReg()))
      VEX_R = 0x0;
    break;
  case X86II::MRM0r: case X86II::MRM1r:
  case X86II::MRM2r: case X86II::MRM3r:
  case X86II::MRM4r: case X86II::MRM5r:
  case X86II::MRM6r: case X86II::MRM7r:
    // MRM0r-MRM7r instructions forms:
    //  dst(VEX_4V), src(ModR/M), imm8
    VEX_4V = getVEXRegisterEncoding(MI, 0);
    if (X86II::isX86_64ExtendedReg(MI.getOperand(1).getReg()))
      VEX_B = 0x0;
    break;
  default: // RawFrm
    break;
  }

  // Emit segment override opcode prefix as needed.
  EmitSegmentOverridePrefix(TSFlags, CurByte, MemOperand, MI, OS);

  // VEX opcode prefix can have 2 or 3 bytes
  //
  //  3 bytes:
  //    +-----+ +--------------+ +-------------------+
  //    | C4h | | RXB | m-mmmm | | W | vvvv | L | pp |
  //    +-----+ +--------------+ +-------------------+
  //  2 bytes:
  //    +-----+ +-------------------+
  //    | C5h | | R | vvvv | L | pp |
  //    +-----+ +-------------------+
  //
  unsigned char LastByte = VEX_PP | (VEX_L << 2) | (VEX_4V << 3);

  if (VEX_B && VEX_X && !VEX_W && !XOP && (VEX_5M == 1)) { // 2 byte VEX prefix
    EmitByte(0xC5, CurByte, OS);
    EmitByte(LastByte | (VEX_R << 7), CurByte, OS);
    return;
  }

  // 3 byte VEX prefix
  EmitByte(XOP ? 0x8F : 0xC4, CurByte, OS);
  EmitByte(VEX_R << 7 | VEX_X << 6 | VEX_B << 5 | VEX_5M, CurByte, OS);
  EmitByte(LastByte | (VEX_W << 7), CurByte, OS);
}

/// DetermineREXPrefix - Determine if the MCInst has to be encoded with a X86-64
/// REX prefix which specifies 1) 64-bit instructions, 2) non-default operand
/// size, and 3) use of X86-64 extended registers.
static unsigned DetermineREXPrefix(const MCInst &MI, uint64_t TSFlags,
                                   const MCInstrDesc &Desc) {
  unsigned REX = 0;
  if (TSFlags & X86II::REX_W)
    REX |= 1 << 3; // set REX.W

  if (MI.getNumOperands() == 0) return REX;

  unsigned NumOps = MI.getNumOperands();
  // FIXME: MCInst should explicitize the two-addrness.
  bool isTwoAddr = NumOps > 1 &&
                      Desc.getOperandConstraint(1, MCOI::TIED_TO) != -1;

  // If it accesses SPL, BPL, SIL, or DIL, then it requires a 0x40 REX prefix.
  unsigned i = isTwoAddr ? 1 : 0;
  for (; i != NumOps; ++i) {
    const MCOperand &MO = MI.getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (!X86II::isX86_64NonExtLowByteReg(Reg)) continue;
    // FIXME: The caller of DetermineREXPrefix slaps this prefix onto anything
    // that returns non-zero.
    REX |= 0x40; // REX fixed encoding prefix
    break;
  }

  switch (TSFlags & X86II::FormMask) {
  case X86II::MRMInitReg: llvm_unreachable("FIXME: Remove this!");
  case X86II::MRMSrcReg:
    if (MI.getOperand(0).isReg() &&
        X86II::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      REX |= 1 << 2; // set REX.R
    i = isTwoAddr ? 2 : 1;
    for (; i != NumOps; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg() && X86II::isX86_64ExtendedReg(MO.getReg()))
        REX |= 1 << 0; // set REX.B
    }
    break;
  case X86II::MRMSrcMem: {
    if (MI.getOperand(0).isReg() &&
        X86II::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      REX |= 1 << 2; // set REX.R
    unsigned Bit = 0;
    i = isTwoAddr ? 2 : 1;
    for (; i != NumOps; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg()) {
        if (X86II::isX86_64ExtendedReg(MO.getReg()))
          REX |= 1 << Bit; // set REX.B (Bit=0) and REX.X (Bit=1)
        Bit++;
      }
    }
    break;
  }
  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m:
  case X86II::MRMDestMem: {
    unsigned e = (isTwoAddr ? X86::AddrNumOperands+1 : X86::AddrNumOperands);
    i = isTwoAddr ? 1 : 0;
    if (NumOps > e && MI.getOperand(e).isReg() &&
        X86II::isX86_64ExtendedReg(MI.getOperand(e).getReg()))
      REX |= 1 << 2; // set REX.R
    unsigned Bit = 0;
    for (; i != e; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg()) {
        if (X86II::isX86_64ExtendedReg(MO.getReg()))
          REX |= 1 << Bit; // REX.B (Bit=0) and REX.X (Bit=1)
        Bit++;
      }
    }
    break;
  }
  default:
    if (MI.getOperand(0).isReg() &&
        X86II::isX86_64ExtendedReg(MI.getOperand(0).getReg()))
      REX |= 1 << 0; // set REX.B
    i = isTwoAddr ? 2 : 1;
    for (unsigned e = NumOps; i != e; ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg() && X86II::isX86_64ExtendedReg(MO.getReg()))
        REX |= 1 << 2; // set REX.R
    }
    break;
  }
  return REX;
}

/// EmitSegmentOverridePrefix - Emit segment override opcode prefix as needed
void X86MCCodeEmitter::EmitSegmentOverridePrefix(uint64_t TSFlags,
                                        unsigned &CurByte, int MemOperand,
                                        const MCInst &MI,
                                        raw_ostream &OS) const {
  switch (TSFlags & X86II::SegOvrMask) {
  default: llvm_unreachable("Invalid segment!");
  case 0:
    // No segment override, check for explicit one on memory operand.
    if (MemOperand != -1) {   // If the instruction has a memory operand.
      switch (MI.getOperand(MemOperand+X86::AddrSegmentReg).getReg()) {
      default: llvm_unreachable("Unknown segment register!");
      case 0: break;
      case X86::CS: EmitByte(0x2E, CurByte, OS); break;
      case X86::SS: EmitByte(0x36, CurByte, OS); break;
      case X86::DS: EmitByte(0x3E, CurByte, OS); break;
      case X86::ES: EmitByte(0x26, CurByte, OS); break;
      case X86::FS: EmitByte(0x64, CurByte, OS); break;
      case X86::GS: EmitByte(0x65, CurByte, OS); break;
      }
    }
    break;
  case X86II::FS:
    EmitByte(0x64, CurByte, OS);
    break;
  case X86II::GS:
    EmitByte(0x65, CurByte, OS);
    break;
  }
}

/// EmitOpcodePrefix - Emit all instruction prefixes prior to the opcode.
///
/// MemOperand is the operand # of the start of a memory operand if present.  If
/// Not present, it is -1.
void X86MCCodeEmitter::EmitOpcodePrefix(uint64_t TSFlags, unsigned &CurByte,
                                        int MemOperand, const MCInst &MI,
                                        const MCInstrDesc &Desc,
                                        raw_ostream &OS) const {

  // Emit the lock opcode prefix as needed.
  if (TSFlags & X86II::LOCK)
    EmitByte(0xF0, CurByte, OS);

  // Emit segment override opcode prefix as needed.
  EmitSegmentOverridePrefix(TSFlags, CurByte, MemOperand, MI, OS);

  // Emit the repeat opcode prefix as needed.
  if ((TSFlags & X86II::Op0Mask) == X86II::REP)
    EmitByte(0xF3, CurByte, OS);

  // Emit the address size opcode prefix as needed.
  bool need_address_override;
  if (TSFlags & X86II::AdSize) {
    need_address_override = true;
  } else if (MemOperand == -1) {
    need_address_override = false;
  } else if (is64BitMode()) {
    assert(!Is16BitMemOperand(MI, MemOperand));
    need_address_override = Is32BitMemOperand(MI, MemOperand);
  } else if (is32BitMode()) {
    assert(!Is64BitMemOperand(MI, MemOperand));
    need_address_override = Is16BitMemOperand(MI, MemOperand);
  } else {
    need_address_override = false;
  }

  if (need_address_override)
    EmitByte(0x67, CurByte, OS);

  // Emit the operand size opcode prefix as needed.
  if (TSFlags & X86II::OpSize)
    EmitByte(0x66, CurByte, OS);

  bool Need0FPrefix = false;
  switch (TSFlags & X86II::Op0Mask) {
  default: llvm_unreachable("Invalid prefix!");
  case 0: break;  // No prefix!
  case X86II::REP: break; // already handled.
  case X86II::TB:  // Two-byte opcode prefix
  case X86II::T8:  // 0F 38
  case X86II::TA:  // 0F 3A
  case X86II::A6:  // 0F A6
  case X86II::A7:  // 0F A7
    Need0FPrefix = true;
    break;
  case X86II::T8XS: // F3 0F 38
    EmitByte(0xF3, CurByte, OS);
    Need0FPrefix = true;
    break;
  case X86II::T8XD: // F2 0F 38
    EmitByte(0xF2, CurByte, OS);
    Need0FPrefix = true;
    break;
  case X86II::TAXD: // F2 0F 3A
    EmitByte(0xF2, CurByte, OS);
    Need0FPrefix = true;
    break;
  case X86II::XS:   // F3 0F
    EmitByte(0xF3, CurByte, OS);
    Need0FPrefix = true;
    break;
  case X86II::XD:   // F2 0F
    EmitByte(0xF2, CurByte, OS);
    Need0FPrefix = true;
    break;
  case X86II::D8: EmitByte(0xD8, CurByte, OS); break;
  case X86II::D9: EmitByte(0xD9, CurByte, OS); break;
  case X86II::DA: EmitByte(0xDA, CurByte, OS); break;
  case X86II::DB: EmitByte(0xDB, CurByte, OS); break;
  case X86II::DC: EmitByte(0xDC, CurByte, OS); break;
  case X86II::DD: EmitByte(0xDD, CurByte, OS); break;
  case X86II::DE: EmitByte(0xDE, CurByte, OS); break;
  case X86II::DF: EmitByte(0xDF, CurByte, OS); break;
  }

  // Handle REX prefix.
  // FIXME: Can this come before F2 etc to simplify emission?
  if (is64BitMode()) {
    if (unsigned REX = DetermineREXPrefix(MI, TSFlags, Desc))
      EmitByte(0x40 | REX, CurByte, OS);
  }

  // 0x0F escape code must be emitted just before the opcode.
  if (Need0FPrefix)
    EmitByte(0x0F, CurByte, OS);

  // FIXME: Pull this up into previous switch if REX can be moved earlier.
  switch (TSFlags & X86II::Op0Mask) {
  case X86II::T8XS:  // F3 0F 38
  case X86II::T8XD:  // F2 0F 38
  case X86II::T8:    // 0F 38
    EmitByte(0x38, CurByte, OS);
    break;
  case X86II::TAXD:  // F2 0F 3A
  case X86II::TA:    // 0F 3A
    EmitByte(0x3A, CurByte, OS);
    break;
  case X86II::A6:    // 0F A6
    EmitByte(0xA6, CurByte, OS);
    break;
  case X86II::A7:    // 0F A7
    EmitByte(0xA7, CurByte, OS);
    break;
  }
}

void X86MCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  unsigned Opcode = MI.getOpcode();
  const MCInstrDesc &Desc = MCII.get(Opcode);
  uint64_t TSFlags = Desc.TSFlags;

  // Pseudo instructions don't get encoded.
  if ((TSFlags & X86II::FormMask) == X86II::Pseudo)
    return;

  // If this is a two-address instruction, skip one of the register operands.
  // FIXME: This should be handled during MCInst lowering.
  unsigned NumOps = Desc.getNumOperands();
  unsigned CurOp = 0;
  if (NumOps > 1 && Desc.getOperandConstraint(1, MCOI::TIED_TO) == 0)
    ++CurOp;
  else if (NumOps > 3 && Desc.getOperandConstraint(2, MCOI::TIED_TO) == 0) {
    assert(Desc.getOperandConstraint(NumOps - 1, MCOI::TIED_TO) == 1);
    // Special case for GATHER with 2 TIED_TO operands
    // Skip the first 2 operands: dst, mask_wb
    CurOp += 2;
  }

  // Keep track of the current byte being emitted.
  unsigned CurByte = 0;

  // Is this instruction encoded using the AVX VEX prefix?
  bool HasVEXPrefix = (TSFlags >> X86II::VEXShift) & X86II::VEX;

  // It uses the VEX.VVVV field?
  bool HasVEX_4V = (TSFlags >> X86II::VEXShift) & X86II::VEX_4V;
  bool HasVEX_4VOp3 = (TSFlags >> X86II::VEXShift) & X86II::VEX_4VOp3;
  bool HasMemOp4 = (TSFlags >> X86II::VEXShift) & X86II::MemOp4;
  const unsigned MemOp4_I8IMMOperand = 2;

  // Determine where the memory operand starts, if present.
  int MemoryOperand = X86II::getMemoryOperandNo(TSFlags, Opcode);
  if (MemoryOperand != -1) MemoryOperand += CurOp;

  if (!HasVEXPrefix)
    EmitOpcodePrefix(TSFlags, CurByte, MemoryOperand, MI, Desc, OS);
  else
    EmitVEXOpcodePrefix(TSFlags, CurByte, MemoryOperand, MI, Desc, OS);

  unsigned char BaseOpcode = X86II::getBaseOpcodeFor(TSFlags);

  if ((TSFlags >> X86II::VEXShift) & X86II::Has3DNow0F0FOpcode)
    BaseOpcode = 0x0F;   // Weird 3DNow! encoding.

  unsigned SrcRegNum = 0;
  switch (TSFlags & X86II::FormMask) {
  case X86II::MRMInitReg:
    llvm_unreachable("FIXME: Remove this form when the JIT moves to MCCodeEmitter!");
  default: errs() << "FORM: " << (TSFlags & X86II::FormMask) << "\n";
    llvm_unreachable("Unknown FormMask value in X86MCCodeEmitter!");
  case X86II::Pseudo:
    llvm_unreachable("Pseudo instruction shouldn't be emitted");
  case X86II::RawFrm:
    EmitByte(BaseOpcode, CurByte, OS);
    break;
  case X86II::RawFrmImm8:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitImmediate(MI.getOperand(CurOp++), MI.getLoc(),
                  X86II::getSizeOfImm(TSFlags), getImmFixupKind(TSFlags),
                  CurByte, OS, Fixups);
    EmitImmediate(MI.getOperand(CurOp++), MI.getLoc(), 1, FK_Data_1, CurByte,
                  OS, Fixups);
    break;
  case X86II::RawFrmImm16:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitImmediate(MI.getOperand(CurOp++), MI.getLoc(),
                  X86II::getSizeOfImm(TSFlags), getImmFixupKind(TSFlags),
                  CurByte, OS, Fixups);
    EmitImmediate(MI.getOperand(CurOp++), MI.getLoc(), 2, FK_Data_2, CurByte,
                  OS, Fixups);
    break;

  case X86II::AddRegFrm:
    EmitByte(BaseOpcode + GetX86RegNum(MI.getOperand(CurOp++)), CurByte, OS);
    break;

  case X86II::MRMDestReg:
    EmitByte(BaseOpcode, CurByte, OS);
    EmitRegModRMByte(MI.getOperand(CurOp),
                     GetX86RegNum(MI.getOperand(CurOp+1)), CurByte, OS);
    CurOp += 2;
    break;

  case X86II::MRMDestMem:
    EmitByte(BaseOpcode, CurByte, OS);
    SrcRegNum = CurOp + X86::AddrNumOperands;

    if (HasVEX_4V) // Skip 1st src (which is encoded in VEX_VVVV)
      ++SrcRegNum;

    EmitMemModRMByte(MI, CurOp,
                     GetX86RegNum(MI.getOperand(SrcRegNum)),
                     TSFlags, CurByte, OS, Fixups);
    CurOp = SrcRegNum + 1;
    break;

  case X86II::MRMSrcReg:
    EmitByte(BaseOpcode, CurByte, OS);
    SrcRegNum = CurOp + 1;

    if (HasVEX_4V) // Skip 1st src (which is encoded in VEX_VVVV)
      ++SrcRegNum;

    if (HasMemOp4) // Skip 2nd src (which is encoded in I8IMM)
      ++SrcRegNum;

    EmitRegModRMByte(MI.getOperand(SrcRegNum),
                     GetX86RegNum(MI.getOperand(CurOp)), CurByte, OS);

    // 2 operands skipped with HasMemOp4, compensate accordingly
    CurOp = HasMemOp4 ? SrcRegNum : SrcRegNum + 1;
    if (HasVEX_4VOp3)
      ++CurOp;
    break;

  case X86II::MRMSrcMem: {
    int AddrOperands = X86::AddrNumOperands;
    unsigned FirstMemOp = CurOp+1;
    if (HasVEX_4V) {
      ++AddrOperands;
      ++FirstMemOp;  // Skip the register source (which is encoded in VEX_VVVV).
    }
    if (HasMemOp4) // Skip second register source (encoded in I8IMM)
      ++FirstMemOp;

    EmitByte(BaseOpcode, CurByte, OS);

    EmitMemModRMByte(MI, FirstMemOp, GetX86RegNum(MI.getOperand(CurOp)),
                     TSFlags, CurByte, OS, Fixups);
    CurOp += AddrOperands + 1;
    if (HasVEX_4VOp3)
      ++CurOp;
    break;
  }

  case X86II::MRM0r: case X86II::MRM1r:
  case X86II::MRM2r: case X86II::MRM3r:
  case X86II::MRM4r: case X86II::MRM5r:
  case X86II::MRM6r: case X86II::MRM7r:
    if (HasVEX_4V) // Skip the register dst (which is encoded in VEX_VVVV).
      ++CurOp;
    EmitByte(BaseOpcode, CurByte, OS);
    EmitRegModRMByte(MI.getOperand(CurOp++),
                     (TSFlags & X86II::FormMask)-X86II::MRM0r,
                     CurByte, OS);
    break;
  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m:
    if (HasVEX_4V) // Skip the register dst (which is encoded in VEX_VVVV).
      ++CurOp;
    EmitByte(BaseOpcode, CurByte, OS);
    EmitMemModRMByte(MI, CurOp, (TSFlags & X86II::FormMask)-X86II::MRM0m,
                     TSFlags, CurByte, OS, Fixups);
    CurOp += X86::AddrNumOperands;
    break;
  case X86II::MRM_C1: case X86II::MRM_C2:
  case X86II::MRM_C3: case X86II::MRM_C4:
  case X86II::MRM_C8: case X86II::MRM_C9:
  case X86II::MRM_D0: case X86II::MRM_D1:
  case X86II::MRM_D4: case X86II::MRM_D5:
  case X86II::MRM_D8: case X86II::MRM_D9:
  case X86II::MRM_DA: case X86II::MRM_DB:
  case X86II::MRM_DC: case X86II::MRM_DD:
  case X86II::MRM_DE: case X86II::MRM_DF:
  case X86II::MRM_E8: case X86II::MRM_F0:
  case X86II::MRM_F8: case X86II::MRM_F9:
    EmitByte(BaseOpcode, CurByte, OS);

    unsigned char MRM;
    switch (TSFlags & X86II::FormMask) {
    default: llvm_unreachable("Invalid Form");
    case X86II::MRM_C1: MRM = 0xC1; break;
    case X86II::MRM_C2: MRM = 0xC2; break;
    case X86II::MRM_C3: MRM = 0xC3; break;
    case X86II::MRM_C4: MRM = 0xC4; break;
    case X86II::MRM_C8: MRM = 0xC8; break;
    case X86II::MRM_C9: MRM = 0xC9; break;
    case X86II::MRM_D0: MRM = 0xD0; break;
    case X86II::MRM_D1: MRM = 0xD1; break;
    case X86II::MRM_D4: MRM = 0xD4; break;
    case X86II::MRM_D5: MRM = 0xD5; break;
    case X86II::MRM_D8: MRM = 0xD8; break;
    case X86II::MRM_D9: MRM = 0xD9; break;
    case X86II::MRM_DA: MRM = 0xDA; break;
    case X86II::MRM_DB: MRM = 0xDB; break;
    case X86II::MRM_DC: MRM = 0xDC; break;
    case X86II::MRM_DD: MRM = 0xDD; break;
    case X86II::MRM_DE: MRM = 0xDE; break;
    case X86II::MRM_DF: MRM = 0xDF; break;
    case X86II::MRM_E8: MRM = 0xE8; break;
    case X86II::MRM_F0: MRM = 0xF0; break;
    case X86II::MRM_F8: MRM = 0xF8; break;
    case X86II::MRM_F9: MRM = 0xF9; break;
    }
    EmitByte(MRM, CurByte, OS);
    break;
  }

  // If there is a remaining operand, it must be a trailing immediate.  Emit it
  // according to the right size for the instruction. Some instructions
  // (SSE4a extrq and insertq) have two trailing immediates.
  while (CurOp != NumOps && NumOps - CurOp <= 2) {
    // The last source register of a 4 operand instruction in AVX is encoded
    // in bits[7:4] of a immediate byte.
    if ((TSFlags >> X86II::VEXShift) & X86II::VEX_I8IMM) {
      const MCOperand &MO = MI.getOperand(HasMemOp4 ? MemOp4_I8IMMOperand
                                                    : CurOp);
      ++CurOp;
      unsigned RegNum = GetX86RegNum(MO) << 4;
      if (X86II::isX86_64ExtendedReg(MO.getReg()))
        RegNum |= 1 << 7;
      // If there is an additional 5th operand it must be an immediate, which
      // is encoded in bits[3:0]
      if (CurOp != NumOps) {
        const MCOperand &MIMM = MI.getOperand(CurOp++);
        if (MIMM.isImm()) {
          unsigned Val = MIMM.getImm();
          assert(Val < 16 && "Immediate operand value out of range");
          RegNum |= Val;
        }
      }
      EmitImmediate(MCOperand::CreateImm(RegNum), MI.getLoc(), 1, FK_Data_1,
                    CurByte, OS, Fixups);
    } else {
      unsigned FixupKind;
      // FIXME: Is there a better way to know that we need a signed relocation?
      if (MI.getOpcode() == X86::ADD64ri32 ||
          MI.getOpcode() == X86::MOV64ri32 ||
          MI.getOpcode() == X86::MOV64mi32 ||
          MI.getOpcode() == X86::PUSH64i32)
        FixupKind = X86::reloc_signed_4byte;
      else
        FixupKind = getImmFixupKind(TSFlags);
      EmitImmediate(MI.getOperand(CurOp++), MI.getLoc(),
                    X86II::getSizeOfImm(TSFlags), MCFixupKind(FixupKind),
                    CurByte, OS, Fixups);
    }
  }

  if ((TSFlags >> X86II::VEXShift) & X86II::Has3DNow0F0FOpcode)
    EmitByte(X86II::getBaseOpcodeFor(TSFlags), CurByte, OS);

#ifndef NDEBUG
  // FIXME: Verify.
  if (/*!Desc.isVariadic() &&*/ CurOp != NumOps) {
    errs() << "Cannot encode all operands of: ";
    MI.dump();
    errs() << '\n';
    abort();
  }
#endif
}
