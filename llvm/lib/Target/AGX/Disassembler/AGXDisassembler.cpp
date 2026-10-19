//===-- AGXDisassembler.cpp - Disassembler for AGX ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX instructions are 2 to 12 bytes long. The TableGen'erated decoder has one
// table per long form size. Tables are tried from the shortest; an
// instruction with a short form is re-decoded with the bytes beyond the short
// form cleared when its length bit is clear.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "TargetInfo/AGXTargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoder.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include <bitset>

using namespace llvm;
using namespace llvm::MCD;

#define DEBUG_TYPE "agx-disassembler"

using DecodeStatus = MCDisassembler::DecodeStatus;

namespace {

class AGXDisassembler : public MCDisassembler {
  std::unique_ptr<const MCInstrInfo> MCII;

public:
  AGXDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx,
                  const MCInstrInfo *MCII)
      : MCDisassembler(STI, Ctx), MCII(MCII) {}

  DecodeStatus getInstruction(MCInst &MI, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;

  const MCRegisterInfo &getRegInfo() const {
    return *getContext().getRegisterInfo();
  }

private:
  DecodeStatus tryDecode(MCInst &MI, uint64_t &Size, ArrayRef<uint8_t> Bytes,
                         unsigned TableBytes, unsigned ValidBytes,
                         uint64_t Address) const;
};

} // namespace

static const MCRegisterInfo &getMRI(const MCDisassembler *D) {
  return static_cast<const AGXDisassembler *>(D)->getRegInfo();
}

static DecodeStatus addReg(MCInst &MI, AGX::RegKind Kind, unsigned HalfIndex,
                           const MCDisassembler *D) {
  MCRegister R = AGX::getRegister(Kind, HalfIndex, getMRI(D));
  if (!R)
    return MCDisassembler::Fail;
  MI.addOperand(MCOperand::createReg(R));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm(MCInst &MI, uint64_t Imm, uint64_t,
                              const MCDisassembler *) {
  MI.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

/// Decode an ALU source in the common layout ([5:0] value, [9:6] type,
/// [11:10] extension) into a modifier and a value operand.
static DecodeStatus decodeALUSrc(MCInst &MI, uint64_t M, unsigned ExtraMods,
                                 unsigned MaxSize, const MCDisassembler *D,
                                 bool Allow32BitUniform = false) {
  unsigned Type = (M >> 6) & 0xf;
  unsigned Value = (M & 63) | (((M >> 10) & 3) << 6);
  if (Type == 0) {
    MI.addOperand(MCOperand::createImm(ExtraMods));
    MI.addOperand(MCOperand::createImm(Value));
    return MCDisassembler::Success;
  }
  if ((Type & 0b1100) == 0b0100) {
    unsigned Idx = Value | ((Type & 1) << 8);
    bool Is32 = Type & 2;
    if (Is32 && MaxSize < 32 && !Allow32BitUniform)
      return MCDisassembler::Fail;
    MI.addOperand(MCOperand::createImm(ExtraMods));
    return addReg(MI, Is32 ? AGX::RegKind::UGPR32 : AGX::RegKind::UGPR16, Idx,
                  D);
  }
  unsigned Hint = Type & 3;
  if (Hint == 0)
    return MCDisassembler::Fail;
  unsigned Mods = ExtraMods | (Hint == 2   ? AGX::SRC_MOD_CACHE
                               : Hint == 3 ? AGX::SRC_MOD_DISCARD
                                           : 0);
  AGX::RegKind Kind;
  switch (Type >> 2) {
  case 0:
    Kind = AGX::RegKind::GPR16;
    break;
  case 2:
    if (MaxSize < 32)
      return MCDisassembler::Fail;
    Kind = AGX::RegKind::GPR32;
    break;
  case 3:
    if (MaxSize < 64)
      return MCDisassembler::Fail;
    Kind = AGX::RegKind::GPR64;
    break;
  default:
    return MCDisassembler::Fail;
  }
  MI.addOperand(MCOperand::createImm(Mods));
  return addReg(MI, Kind, Value, D);
}

static unsigned decodeFPMods(unsigned FM) {
  return ((FM & 1) ? AGX::SRC_MOD_ABS : 0) | ((FM & 2) ? AGX::SRC_MOD_NEG : 0);
}

static DecodeStatus decodeSrc(MCInst &MI, uint64_t V, uint64_t,
                              const MCDisassembler *D) {
  return decodeALUSrc(MI, V, 0, 32, D);
}

[[maybe_unused]] static DecodeStatus decodeSrc64(MCInst &MI, uint64_t V, uint64_t,
                                const MCDisassembler *D) {
  return decodeALUSrc(MI, V, 0, 64, D);
}

static DecodeStatus decodeSrc16(MCInst &MI, uint64_t V, uint64_t,
                                const MCDisassembler *D) {
  return decodeALUSrc(MI, V, 0, 16, D, /*Allow32BitUniform=*/true);
}

static DecodeStatus decodeSrcSX(MCInst &MI, uint64_t V, uint64_t,
                                const MCDisassembler *D) {
  uint64_t M = (V & 0x3ff) | (((V >> 11) & 3) << 10);
  return decodeALUSrc(MI, M, (V & (1u << 10)) ? AGX::SRC_MOD_SX : 0, 32, D);
}

static DecodeStatus decodeSrcSX64(MCInst &MI, uint64_t V, uint64_t,
                                  const MCDisassembler *D) {
  uint64_t M = (V & 0x3ff) | (((V >> 11) & 3) << 10);
  return decodeALUSrc(MI, M, (V & (1u << 10)) ? AGX::SRC_MOD_SX : 0, 64, D);
}

static DecodeStatus decodeSrcF(MCInst &MI, uint64_t V, uint64_t,
                               const MCDisassembler *D) {
  uint64_t M = (V & 0x3ff) | (((V >> 12) & 3) << 10);
  return decodeALUSrc(MI, M, decodeFPMods((V >> 10) & 3), 32, D);
}

static DecodeStatus decodeSrcF16(MCInst &MI, uint64_t V, uint64_t,
                                 const MCDisassembler *D) {
  uint64_t M = (V & 0x1ff) | (((V >> 11) & 3) << 10);
  // The 32-bit uniform flag is below the omitted GPR size bit.
  return decodeALUSrc(MI, M, decodeFPMods((V >> 9) & 3), 16, D,
                      /*Allow32BitUniform=*/true);
}

static DecodeStatus decodeCmpselSrc(MCInst &MI, uint64_t V, uint64_t,
                                    const MCDisassembler *D) {
  unsigned Type = (V >> 6) & 7;
  unsigned Value = (V & 63) | (((V >> 9) & 3) << 6);
  // The access size follows the destination, operand 0.
  AGX::RegInfo Dst = AGX::getRegInfo(MI.getOperand(0).getReg(), getMRI(D));
  bool Is32 = Dst.getSizeInBits() >= 32;
  if (Type == 0b100) {
    MI.addOperand(MCOperand::createImm(0));
    MI.addOperand(MCOperand::createImm(Value));
    return MCDisassembler::Success;
  }
  if ((Type >> 1) == 0b11) {
    MI.addOperand(MCOperand::createImm(0));
    return addReg(MI, Is32 ? AGX::RegKind::UGPR32 : AGX::RegKind::UGPR16,
                  Value | ((Type & 1) << 8), D);
  }
  if (Type == 0 || (Type >> 2))
    return MCDisassembler::Fail;
  unsigned Mods = Type == 2   ? AGX::SRC_MOD_CACHE
                  : Type == 3 ? AGX::SRC_MOD_DISCARD
                              : 0;
  MI.addOperand(MCOperand::createImm(Mods));
  return addReg(MI, Is32 ? AGX::RegKind::GPR32 : AGX::RegKind::GPR16, Value, D);
}

static DecodeStatus decodeDstImpl(MCInst &MI, uint64_t V, bool Allow64,
                                  int ForcedSize, const MCDisassembler *D) {
  bool Cache = V & 1;
  bool Is32 = ForcedSize ? ForcedSize == 32 : (V >> 1) & 1;
  unsigned Value = (V >> 2) & 0xff;
  AGX::RegKind Kind = AGX::RegKind::GPR16;
  if (Is32) {
    if ((Value & 1) && Allow64) {
      Kind = AGX::RegKind::GPR64;
      Value &= ~1u;
    } else {
      Kind = AGX::RegKind::GPR32;
    }
  }
  if (addReg(MI, Kind, Value, D) == MCDisassembler::Fail)
    return MCDisassembler::Fail;
  MI.addOperand(MCOperand::createImm(Cache));
  return MCDisassembler::Success;
}

static DecodeStatus decodeDst(MCInst &MI, uint64_t V, uint64_t,
                              const MCDisassembler *D) {
  return decodeDstImpl(MI, V, false, 0, D);
}

static DecodeStatus decodeDst64(MCInst &MI, uint64_t V, uint64_t,
                                const MCDisassembler *D) {
  return decodeDstImpl(MI, V, true, 0, D);
}

static DecodeStatus decodeDst16(MCInst &MI, uint64_t V, uint64_t,
                                const MCDisassembler *D) {
  return decodeDstImpl(MI, V, false, 16, D);
}

static DecodeStatus decodeDst32(MCInst &MI, uint64_t V, uint64_t,
                                const MCDisassembler *D) {
  return decodeDstImpl(MI, V, false, 32, D);
}

static DecodeStatus decodeMemReg(MCInst &MI, uint64_t V, uint64_t,
                                 const MCDisassembler *D) {
  bool Is32 = (V >> 8) & 1;
  return addReg(MI, Is32 ? AGX::RegKind::GPR32 : AGX::RegKind::GPR16, V & 0xff,
                D);
}

static DecodeStatus decodeAtomicDst(MCInst &MI, uint64_t V, uint64_t,
                                    const MCDisassembler *D) {
  if (!((V >> 8) & 1)) {
    if (V & 0xff)
      return MCDisassembler::Fail;
    MI.addOperand(MCOperand::createReg(MCRegister()));
    return MCDisassembler::Success;
  }
  return addReg(MI, AGX::RegKind::GPR32, V & 0xff, D);
}

// Threadgroup atomic result: [7:0] register, [8] 32-bit, [9] present.
static DecodeStatus decodeLocalAtomicDst(MCInst &MI, uint64_t V, uint64_t,
                                         const MCDisassembler *D) {
  if (!((V >> 9) & 1)) {
    if (V & 0x1ff)
      return MCDisassembler::Fail;
    MI.addOperand(MCOperand::createReg(MCRegister()));
    return MCDisassembler::Success;
  }
  return addReg(MI, ((V >> 8) & 1) ? AGX::RegKind::GPR32 : AGX::RegKind::GPR16,
                V & 0xff, D);
}

static DecodeStatus decodeAtomicSrc(MCInst &MI, uint64_t V, uint64_t,
                                    const MCDisassembler *D) {
  // Operand 1 is the atomic operation; cmpxchg takes a register pair.
  bool Pair = MI.getNumOperands() > 1 && MI.getOperand(1).isImm() &&
              MI.getOperand(1).getImm() == 3;
  return addReg(MI, Pair ? AGX::RegKind::GPR64 : AGX::RegKind::GPR32, V & 0xff,
                D);
}

static DecodeStatus decodeMemBase(MCInst &MI, uint64_t V, uint64_t,
                                  const MCDisassembler *D) {
  bool Uniform = (V >> 8) & 1;
  return addReg(MI, Uniform ? AGX::RegKind::UGPR64 : AGX::RegKind::GPR64,
                V & 0xff, D);
}

static DecodeStatus decodeMemIndex(MCInst &MI, uint64_t V, uint64_t,
                                   const MCDisassembler *D) {
  if ((V >> 16) & 1) {
    MI.addOperand(MCOperand::createImm(int16_t(V & 0xffff)));
    return MCDisassembler::Success;
  }
  if (V & 0xff00)
    return MCDisassembler::Fail;
  return addReg(MI, AGX::RegKind::GPR32, V & 0xff, D);
}

static DecodeStatus decodeLocalBase(MCInst &MI, uint64_t V, uint64_t,
                                    const MCDisassembler *D) {
  unsigned Flags = (V >> 8) & 3;
  unsigned Value = V & 0xff;
  if (Flags == 0)
    return addReg(MI, AGX::RegKind::GPR16, Value, D);
  if (Flags == 2) {
    if (Value)
      return MCDisassembler::Fail;
    MI.addOperand(MCOperand::createImm(0));
    return MCDisassembler::Success;
  }
  return addReg(MI, AGX::RegKind::UGPR16, Value | ((Flags >> 1) << 8), D);
}

static DecodeStatus decodeLocalIndex(MCInst &MI, uint64_t V, uint64_t,
                                     const MCDisassembler *D) {
  if ((V >> 16) & 1) {
    MI.addOperand(MCOperand::createImm(int16_t(V & 0xffff)));
    return MCDisassembler::Success;
  }
  if (V & 0xff00)
    return MCDisassembler::Fail;
  return addReg(MI, AGX::RegKind::GPR16, V & 0xff, D);
}

static DecodeStatus decodeBrTarget(MCInst &MI, uint64_t V, uint64_t,
                                   const MCDisassembler *) {
  MI.addOperand(MCOperand::createImm(SignExtend64<32>(V)));
  return MCDisassembler::Success;
}

#include "AGXGenDisassemblerTables.inc"


static const uint8_t *getTable(unsigned Bytes) {
  switch (Bytes) {
  case 2:
    return DecoderTableG1316;
  case 4:
    return DecoderTableG1332;
  case 6:
    return DecoderTableG1348;
  case 8:
    return DecoderTableG1364;
  case 10:
    return DecoderTableG1380;
  default:
    return nullptr;
  }
}

static std::bitset<96> readBytes(ArrayRef<uint8_t> Bytes, unsigned N) {
  std::bitset<96> Insn;
  for (unsigned I = 0; I < N && I < Bytes.size(); ++I)
    Insn |= std::bitset<96>(Bytes[I]) << (8 * I);
  return Insn;
}

DecodeStatus AGXDisassembler::tryDecode(MCInst &MI, uint64_t &Size,
                                        ArrayRef<uint8_t> Bytes,
                                        unsigned TableBytes,
                                        unsigned ValidBytes,
                                        uint64_t Address) const {
  const uint8_t *Table = getTable(TableBytes);
  std::bitset<96> Insn = readBytes(Bytes, std::min(TableBytes, ValidBytes));
  MI.clear();
  if (decodeInstruction(Table, MI, Insn, Address, this, STI) !=
      MCDisassembler::Success)
    return MCDisassembler::Fail;

  const MCInstrDesc &Desc = MCII->get(MI.getOpcode());
  unsigned Short = AGX::TSFlags::getShortSize(Desc.TSFlags);
  unsigned Long = AGX::TSFlags::getLongSize(Desc.TSFlags);
  if (Long != TableBytes)
    return MCDisassembler::Fail;
  if (Short == Long || Insn[AGX::TSFlags::getLengthBit(Desc.TSFlags)]) {
    if (Long > ValidBytes)
      return MCDisassembler::Fail;
    Size = Long;
    return MCDisassembler::Success;
  }
  if (ValidBytes < Short)
    return MCDisassembler::Fail;
  // Short form: the remaining bytes belong to the next instruction.
  if (ValidBytes != Short) {
    Insn = readBytes(Bytes, Short);
    MI.clear();
    if (decodeInstruction(Table, MI, Insn, Address, this, STI) !=
        MCDisassembler::Success)
      return MCDisassembler::Fail;
    const MCInstrDesc &ShortDesc = MCII->get(MI.getOpcode());
    if (AGX::TSFlags::getShortSize(ShortDesc.TSFlags) != Short)
      return MCDisassembler::Fail;
  }
  Size = Short;
  return MCDisassembler::Success;
}

DecodeStatus AGXDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                             ArrayRef<uint8_t> Bytes,
                                             uint64_t Address,
                                             raw_ostream &CStream) const {
  Size = 0;
  if (Bytes.size() < 2)
    return MCDisassembler::Fail;
  unsigned Avail = std::min<size_t>(Bytes.size(), 12);
  for (unsigned TableBytes : {2u, 4u, 6u, 8u, 10u}) {
    // First as if all bytes up to the long form belonged to the instruction,
    // then with only the bytes of each possible short form.
    if (tryDecode(MI, Size, Bytes, TableBytes, Avail, Address) ==
        MCDisassembler::Success)
      return MCDisassembler::Success;
    for (unsigned ShortBytes : {TableBytes - 2, TableBytes - 4}) {
      if (ShortBytes < 2 || ShortBytes > TableBytes)
        continue;
      if (tryDecode(MI, Size, Bytes, TableBytes, std::min(ShortBytes, Avail),
                    Address) == MCDisassembler::Success)
        return MCDisassembler::Success;
    }
  }
  Size = 2;
  return MCDisassembler::Fail;
}

static MCDisassembler *createAGXDisassembler(const Target &T,
                                             const MCSubtargetInfo &STI,
                                             MCContext &Ctx) {
  return new AGXDisassembler(STI, Ctx, T.createMCInstrInfo());
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeAGXDisassembler() {
  TargetRegistry::RegisterMCDisassembler(getTheAGXTarget(),
                                         createAGXDisassembler);
}
