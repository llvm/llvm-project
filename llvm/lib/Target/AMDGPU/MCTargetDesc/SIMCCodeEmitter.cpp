//===-- SIMCCodeEmitter.cpp - SI Code Emitter -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief The SI code emitter produces machine code that can be executed
/// directly on the GPU device.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "MCTargetDesc/AMDGPUFixupKinds.h"
#include "MCTargetDesc/AMDGPUMCCodeEmitter.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class SIMCCodeEmitter : public  AMDGPUMCCodeEmitter {
  SIMCCodeEmitter(const SIMCCodeEmitter &) = delete;
  void operator=(const SIMCCodeEmitter &) = delete;
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  MCContext &Ctx;

  /// \brief Can this operand also contain immediate values?
  bool isSrcOperand(const MCInstrDesc &Desc, unsigned OpNo) const;

  /// \brief Encode an fp or int literal
  uint32_t getLitEncoding(const MCOperand &MO, unsigned OpSize) const;

public:
  SIMCCodeEmitter(const MCInstrInfo &mcii, const MCRegisterInfo &mri,
                  MCContext &ctx)
    : MCII(mcii), MRI(mri), Ctx(ctx) { }

  ~SIMCCodeEmitter() override {}

  /// \brief Encode the instruction and write it to the OS.
  void encodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  /// \returns the encoding for an MCOperand.
  uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const override;

  /// \brief Use a fixup to encode the simm16 field for SOPP branch
  ///        instructions.
  unsigned getSOPPBrEncoding(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const override;
};

} // End anonymous namespace

MCCodeEmitter *llvm::createSIMCCodeEmitter(const MCInstrInfo &MCII,
                                           const MCRegisterInfo &MRI,
                                           MCContext &Ctx) {
  return new SIMCCodeEmitter(MCII, MRI, Ctx);
}

bool SIMCCodeEmitter::isSrcOperand(const MCInstrDesc &Desc,
                                   unsigned OpNo) const {
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;

  return OpType == AMDGPU::OPERAND_REG_IMM32 ||
         OpType == AMDGPU::OPERAND_REG_INLINE_C;
}

// Returns the encoding value to use if the given integer is an integer inline
// immediate value, or 0 if it is not.
template <typename IntTy>
static uint32_t getIntInlineImmEncoding(IntTy Imm) {
  if (Imm >= 0 && Imm <= 64)
    return 128 + Imm;

  if (Imm >= -16 && Imm <= -1)
    return 192 + std::abs(Imm);

  return 0;
}

static uint32_t getLit32Encoding(uint32_t Val) {
  uint32_t IntImm = getIntInlineImmEncoding(static_cast<int32_t>(Val));
  if (IntImm != 0)
    return IntImm;

  if (Val == FloatToBits(0.5f))
    return 240;

  if (Val == FloatToBits(-0.5f))
    return 241;

  if (Val == FloatToBits(1.0f))
    return 242;

  if (Val == FloatToBits(-1.0f))
    return 243;

  if (Val == FloatToBits(2.0f))
    return 244;

  if (Val == FloatToBits(-2.0f))
    return 245;

  if (Val == FloatToBits(4.0f))
    return 246;

  if (Val == FloatToBits(-4.0f))
    return 247;

  return 255;
}

static uint32_t getLit64Encoding(uint64_t Val) {
  uint32_t IntImm = getIntInlineImmEncoding(static_cast<int64_t>(Val));
  if (IntImm != 0)
    return IntImm;

  if (Val == DoubleToBits(0.5))
    return 240;

  if (Val == DoubleToBits(-0.5))
    return 241;

  if (Val == DoubleToBits(1.0))
    return 242;

  if (Val == DoubleToBits(-1.0))
    return 243;

  if (Val == DoubleToBits(2.0))
    return 244;

  if (Val == DoubleToBits(-2.0))
    return 245;

  if (Val == DoubleToBits(4.0))
    return 246;

  if (Val == DoubleToBits(-4.0))
    return 247;

  return 255;
}

uint32_t SIMCCodeEmitter::getLitEncoding(const MCOperand &MO,
                                         unsigned OpSize) const {
  if (MO.isExpr())
    return 255;

  assert(!MO.isFPImm());

  if (!MO.isImm())
    return ~0;

  if (OpSize == 4)
    return getLit32Encoding(static_cast<uint32_t>(MO.getImm()));

  assert(OpSize == 8);

  return getLit64Encoding(static_cast<uint64_t>(MO.getImm()));
}

void SIMCCodeEmitter::encodeInstruction(const MCInst &MI, raw_ostream &OS,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {

  uint64_t Encoding = getBinaryCodeForInstr(MI, Fixups, STI);
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  unsigned bytes = Desc.getSize();

  for (unsigned i = 0; i < bytes; i++) {
    OS.write((uint8_t) ((Encoding >> (8 * i)) & 0xff));
  }

  if (bytes > 4)
    return;

  // Check for additional literals in SRC0/1/2 (Op 1/2/3)
  for (unsigned i = 0, e = MI.getNumOperands(); i < e; ++i) {

    // Check if this operand should be encoded as [SV]Src
    if (!isSrcOperand(Desc, i))
      continue;

    int RCID = Desc.OpInfo[i].RegClass;
    const MCRegisterClass &RC = MRI.getRegClass(RCID);

    // Is this operand a literal immediate?
    const MCOperand &Op = MI.getOperand(i);
    if (getLitEncoding(Op, RC.getSize()) != 255)
      continue;

    // Yes! Encode it
    int64_t Imm = 0;

    if (Op.isImm())
      Imm = Op.getImm();
    else if (!Op.isExpr()) // Exprs will be replaced with a fixup value.
      llvm_unreachable("Must be immediate or expr");

    for (unsigned j = 0; j < 4; j++) {
      OS.write((uint8_t) ((Imm >> (8 * j)) & 0xff));
    }

    // Only one literal value allowed
    break;
  }
}

unsigned SIMCCodeEmitter::getSOPPBrEncoding(const MCInst &MI, unsigned OpNo,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isExpr()) {
    const MCExpr *Expr = MO.getExpr();
    MCFixupKind Kind = (MCFixupKind)AMDGPU::fixup_si_sopp_br;
    Fixups.push_back(MCFixup::create(0, Expr, Kind, MI.getLoc()));
    return 0;
  }

  return getMachineOpValue(MI, MO, Fixups, STI);
}

uint64_t SIMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                            const MCOperand &MO,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return MRI.getEncodingValue(MO.getReg());

  if (MO.isExpr()) {
    const MCSymbolRefExpr *Expr = cast<MCSymbolRefExpr>(MO.getExpr());
    MCFixupKind Kind;
    const MCSymbol *Sym =
        Ctx.getOrCreateSymbol(StringRef(END_OF_TEXT_LABEL_NAME));

    if (&Expr->getSymbol() == Sym) {
      // Add the offset to the beginning of the constant values.
      Kind = (MCFixupKind)AMDGPU::fixup_si_end_of_text;
    } else {
      // This is used for constant data stored in .rodata.
     Kind = (MCFixupKind)AMDGPU::fixup_si_rodata;
    }
    Fixups.push_back(MCFixup::create(4, Expr, Kind, MI.getLoc()));
  }

  // Figure out the operand number, needed for isSrcOperand check
  unsigned OpNo = 0;
  for (unsigned e = MI.getNumOperands(); OpNo < e; ++OpNo) {
    if (&MO == &MI.getOperand(OpNo))
      break;
  }

  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  if (isSrcOperand(Desc, OpNo)) {
    int RCID = Desc.OpInfo[OpNo].RegClass;
    const MCRegisterClass &RC = MRI.getRegClass(RCID);

    uint32_t Enc = getLitEncoding(MO, RC.getSize());
    if (Enc != ~0U && (Enc != 255 || Desc.getSize() == 4))
      return Enc;

  } else if (MO.isImm())
    return MO.getImm();

  llvm_unreachable("Encoding of this operand type is not supported yet.");
  return 0;
}

