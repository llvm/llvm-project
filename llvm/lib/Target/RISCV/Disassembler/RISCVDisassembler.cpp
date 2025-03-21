//===-- RISCVDisassembler.cpp - Disassembler for RISC-V -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISCVDisassembler class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "TargetInfo/RISCVTargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Endian.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {
class RISCVDisassembler : public MCDisassembler {
  std::unique_ptr<MCInstrInfo const> const MCII;

public:
  RISCVDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx,
                    MCInstrInfo const *MCII)
      : MCDisassembler(STI, Ctx), MCII(MCII) {}

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;

private:
  void addSPOperands(MCInst &MI) const;

  DecodeStatus getInstruction48(MCInst &Instr, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &CStream) const;

  DecodeStatus getInstruction32(MCInst &Instr, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &CStream) const;
  DecodeStatus getInstruction16(MCInst &Instr, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &CStream) const;
};
} // end anonymous namespace

static MCDisassembler *createRISCVDisassembler(const Target &T,
                                               const MCSubtargetInfo &STI,
                                               MCContext &Ctx) {
  return new RISCVDisassembler(STI, Ctx, T.createMCInstrInfo());
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRISCVDisassembler() {
  // Register the disassembler for each target.
  TargetRegistry::RegisterMCDisassembler(getTheRISCV32Target(),
                                         createRISCVDisassembler);
  TargetRegistry::RegisterMCDisassembler(getTheRISCV64Target(),
                                         createRISCVDisassembler);
}

static DecodeStatus DecodeGPRRegisterClass(MCInst &Inst, uint32_t RegNo,
                                           uint64_t Address,
                                           const MCDisassembler *Decoder) {
  bool IsRVE = Decoder->getSubtargetInfo().hasFeature(RISCV::FeatureStdExtE);

  if (RegNo >= 32 || (IsRVE && RegNo >= 16))
    return MCDisassembler::Fail;

  MCRegister Reg = RISCV::X0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRF16RegisterClass(MCInst &Inst, uint32_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  bool IsRVE = Decoder->getSubtargetInfo().hasFeature(RISCV::FeatureStdExtE);

  if (RegNo >= 32 || (IsRVE && RegNo >= 16))
    return MCDisassembler::Fail;

  MCRegister Reg = RISCV::X0_H + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRF32RegisterClass(MCInst &Inst, uint32_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  bool IsRVE = Decoder->getSubtargetInfo().hasFeature(RISCV::FeatureStdExtE);

  if (RegNo >= 32 || (IsRVE && RegNo >= 16))
    return MCDisassembler::Fail;

  MCRegister Reg = RISCV::X0_W + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRX1X5RegisterClass(MCInst &Inst, uint32_t RegNo,
                                               uint64_t Address,
                                               const MCDisassembler *Decoder) {
  MCRegister Reg = RISCV::X0 + RegNo;
  if (Reg != RISCV::X1 && Reg != RISCV::X5)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR16RegisterClass(MCInst &Inst, uint32_t RegNo,
                                             uint64_t Address,
                                             const MCDisassembler *Decoder) {
  if (RegNo >= 32)
    return MCDisassembler::Fail;

  MCRegister Reg = RISCV::F0_H + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR32RegisterClass(MCInst &Inst, uint32_t RegNo,
                                             uint64_t Address,
                                             const MCDisassembler *Decoder) {
  if (RegNo >= 32)
    return MCDisassembler::Fail;

  MCRegister Reg = RISCV::F0_F + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR32CRegisterClass(MCInst &Inst, uint32_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  if (RegNo >= 8) {
    return MCDisassembler::Fail;
  }
  MCRegister Reg = RISCV::F8_F + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR64RegisterClass(MCInst &Inst, uint32_t RegNo,
                                             uint64_t Address,
                                             const MCDisassembler *Decoder) {
  if (RegNo >= 32)
    return MCDisassembler::Fail;

  MCRegister Reg = RISCV::F0_D + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPR64CRegisterClass(MCInst &Inst, uint32_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  if (RegNo >= 8) {
    return MCDisassembler::Fail;
  }
  MCRegister Reg = RISCV::F8_D + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRNoX0RegisterClass(MCInst &Inst, uint32_t RegNo,
                                               uint64_t Address,
                                               const MCDisassembler *Decoder) {
  if (RegNo == 0) {
    return MCDisassembler::Fail;
  }

  return DecodeGPRRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus
DecodeGPRNoX0X2RegisterClass(MCInst &Inst, uint64_t RegNo, uint32_t Address,
                             const MCDisassembler *Decoder) {
  if (RegNo == 2) {
    return MCDisassembler::Fail;
  }

  return DecodeGPRNoX0RegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeGPRCRegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo >= 8)
    return MCDisassembler::Fail;

  MCRegister Reg = RISCV::X8 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRPairRegisterClass(MCInst &Inst, uint32_t RegNo,
                                               uint64_t Address,
                                               const MCDisassembler *Decoder) {
  if (RegNo >= 32 || RegNo % 2)
    return MCDisassembler::Fail;

  const RISCVDisassembler *Dis =
      static_cast<const RISCVDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg = RI->getMatchingSuperReg(
      RISCV::X0 + RegNo, RISCV::sub_gpr_even,
      &RISCVMCRegisterClasses[RISCV::GPRPairRegClassID]);
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRPairCRegisterClass(MCInst &Inst, uint32_t RegNo,
                                                uint64_t Address,
                                                const MCDisassembler *Decoder) {
  if (RegNo >= 8 || RegNo % 2)
    return MCDisassembler::Fail;

  const RISCVDisassembler *Dis =
      static_cast<const RISCVDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg = RI->getMatchingSuperReg(
      RISCV::X8 + RegNo, RISCV::sub_gpr_even,
      &RISCVMCRegisterClasses[RISCV::GPRPairCRegClassID]);
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeSR07RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  if (RegNo >= 8)
    return MCDisassembler::Fail;

  MCRegister Reg = (RegNo < 2) ? (RegNo + RISCV::X8) : (RegNo - 2 + RISCV::X18);
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVRRegisterClass(MCInst &Inst, uint32_t RegNo,
                                          uint64_t Address,
                                          const MCDisassembler *Decoder) {
  if (RegNo >= 32)
    return MCDisassembler::Fail;

  MCRegister Reg = RISCV::V0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVRM2RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo >= 32 || RegNo % 2)
    return MCDisassembler::Fail;

  const RISCVDisassembler *Dis =
      static_cast<const RISCVDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg =
      RI->getMatchingSuperReg(RISCV::V0 + RegNo, RISCV::sub_vrm1_0,
                              &RISCVMCRegisterClasses[RISCV::VRM2RegClassID]);

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVRM4RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo >= 32 || RegNo % 4)
    return MCDisassembler::Fail;

  const RISCVDisassembler *Dis =
      static_cast<const RISCVDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg =
      RI->getMatchingSuperReg(RISCV::V0 + RegNo, RISCV::sub_vrm1_0,
                              &RISCVMCRegisterClasses[RISCV::VRM4RegClassID]);

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVRM8RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo >= 32 || RegNo % 8)
    return MCDisassembler::Fail;

  const RISCVDisassembler *Dis =
      static_cast<const RISCVDisassembler *>(Decoder);
  const MCRegisterInfo *RI = Dis->getContext().getRegisterInfo();
  MCRegister Reg =
      RI->getMatchingSuperReg(RISCV::V0 + RegNo, RISCV::sub_vrm1_0,
                              &RISCVMCRegisterClasses[RISCV::VRM8RegClassID]);

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVMV0RegisterClass(MCInst &Inst, uint32_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createReg(RISCV::V0));
  return MCDisassembler::Success;
}

static DecodeStatus decodeVMaskReg(MCInst &Inst, uint32_t RegNo,
                                   uint64_t Address,
                                   const MCDisassembler *Decoder) {
  if (RegNo >= 2)
    return MCDisassembler::Fail;

  MCRegister Reg = (RegNo == 0) ? RISCV::V0 : RISCV::NoRegister;

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

template <unsigned N>
static DecodeStatus decodeUImmOperand(MCInst &Inst, uint32_t Imm,
                                      int64_t Address,
                                      const MCDisassembler *Decoder) {
  assert(isUInt<N>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

template <unsigned Width, unsigned LowerBound>
static DecodeStatus decodeUImmOperandGE(MCInst &Inst, uint32_t Imm,
                                        int64_t Address,
                                        const MCDisassembler *Decoder) {
  assert(isUInt<Width>(Imm) && "Invalid immediate");

  if (Imm < LowerBound)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

template <unsigned Width, unsigned LowerBound>
static DecodeStatus decodeUImmPlus1OperandGE(MCInst &Inst, uint32_t Imm,
                                             int64_t Address,
                                             const MCDisassembler *Decoder) {
  assert(isUInt<Width>(Imm) && "Invalid immediate");

  if ((Imm + 1) < LowerBound)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm + 1));
  return MCDisassembler::Success;
}

static DecodeStatus decodeUImmLog2XLenOperand(MCInst &Inst, uint32_t Imm,
                                              int64_t Address,
                                              const MCDisassembler *Decoder) {
  assert(isUInt<6>(Imm) && "Invalid immediate");

  if (!Decoder->getSubtargetInfo().hasFeature(RISCV::Feature64Bit) &&
      !isUInt<5>(Imm))
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

template <unsigned N>
static DecodeStatus decodeUImmNonZeroOperand(MCInst &Inst, uint32_t Imm,
                                             int64_t Address,
                                             const MCDisassembler *Decoder) {
  if (Imm == 0)
    return MCDisassembler::Fail;
  return decodeUImmOperand<N>(Inst, Imm, Address, Decoder);
}

static DecodeStatus
decodeUImmLog2XLenNonZeroOperand(MCInst &Inst, uint32_t Imm, int64_t Address,
                                 const MCDisassembler *Decoder) {
  if (Imm == 0)
    return MCDisassembler::Fail;
  return decodeUImmLog2XLenOperand(Inst, Imm, Address, Decoder);
}

template <unsigned N>
static DecodeStatus decodeUImmPlus1Operand(MCInst &Inst, uint32_t Imm,
                                           int64_t Address,
                                           const MCDisassembler *Decoder) {
  assert(isUInt<N>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm + 1));
  return MCDisassembler::Success;
}

template <unsigned N>
static DecodeStatus decodeSImmOperand(MCInst &Inst, uint32_t Imm,
                                      int64_t Address,
                                      const MCDisassembler *Decoder) {
  assert(isUInt<N>(Imm) && "Invalid immediate");
  // Sign-extend the number in the bottom N bits of Imm
  Inst.addOperand(MCOperand::createImm(SignExtend64<N>(Imm)));
  return MCDisassembler::Success;
}

template <unsigned N>
static DecodeStatus decodeSImmNonZeroOperand(MCInst &Inst, uint32_t Imm,
                                             int64_t Address,
                                             const MCDisassembler *Decoder) {
  if (Imm == 0)
    return MCDisassembler::Fail;
  return decodeSImmOperand<N>(Inst, Imm, Address, Decoder);
}

template <unsigned N>
static DecodeStatus decodeSImmOperandAndLsl1(MCInst &Inst, uint32_t Imm,
                                             int64_t Address,
                                             const MCDisassembler *Decoder) {
  assert(isUInt<N>(Imm) && "Invalid immediate");
  // Sign-extend the number in the bottom N bits of Imm after accounting for
  // the fact that the N bit immediate is stored in N-1 bits (the LSB is
  // always zero)
  Inst.addOperand(MCOperand::createImm(SignExtend64<N>(Imm << 1)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeCLUIImmOperand(MCInst &Inst, uint32_t Imm,
                                         int64_t Address,
                                         const MCDisassembler *Decoder) {
  assert(isUInt<6>(Imm) && "Invalid immediate");
  if (Imm > 31) {
    Imm = (SignExtend64<6>(Imm) & 0xfffff);
  }
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeFRMArg(MCInst &Inst, uint32_t Imm, int64_t Address,
                                 const MCDisassembler *Decoder) {
  assert(isUInt<3>(Imm) && "Invalid immediate");
  if (!llvm::RISCVFPRndMode::isValidRoundingMode(Imm))
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeRTZArg(MCInst &Inst, uint32_t Imm, int64_t Address,
                                 const MCDisassembler *Decoder) {
  assert(isUInt<3>(Imm) && "Invalid immediate");
  if (Imm != RISCVFPRndMode::RTZ)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeRVCInstrRdRs1ImmZero(MCInst &Inst, uint32_t Insn,
                                               uint64_t Address,
                                               const MCDisassembler *Decoder);

static DecodeStatus decodeRVCInstrRdSImm(MCInst &Inst, uint32_t Insn,
                                         uint64_t Address,
                                         const MCDisassembler *Decoder);

static DecodeStatus decodeRVCInstrRdRs1UImm(MCInst &Inst, uint32_t Insn,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder);

static DecodeStatus decodeRVCInstrRdRs2(MCInst &Inst, uint32_t Insn,
                                        uint64_t Address,
                                        const MCDisassembler *Decoder);

static DecodeStatus decodeRVCInstrRdRs1Rs2(MCInst &Inst, uint32_t Insn,
                                           uint64_t Address,
                                           const MCDisassembler *Decoder);

static DecodeStatus decodeXTHeadMemPair(MCInst &Inst, uint32_t Insn,
                                        uint64_t Address,
                                        const MCDisassembler *Decoder);

static DecodeStatus decodeZcmpRlist(MCInst &Inst, uint32_t Imm,
                                    uint64_t Address, const void *Decoder);

static DecodeStatus decodeRegReg(MCInst &Inst, uint32_t Insn, uint64_t Address,
                                 const MCDisassembler *Decoder);

static DecodeStatus decodeZcmpSpimm(MCInst &Inst, uint32_t Imm,
                                    uint64_t Address, const void *Decoder);

static DecodeStatus decodeCSSPushPopchk(MCInst &Inst, uint32_t Insn,
                                        uint64_t Address,
                                        const MCDisassembler *Decoder);

#include "RISCVGenDisassemblerTables.inc"

static DecodeStatus decodeRVCInstrRdRs1ImmZero(MCInst &Inst, uint32_t Insn,
                                               uint64_t Address,
                                               const MCDisassembler *Decoder) {
  uint32_t Rd = fieldFromInstruction(Insn, 7, 5);
  [[maybe_unused]] DecodeStatus Result =
      DecodeGPRNoX0RegisterClass(Inst, Rd, Address, Decoder);
  assert(Result == MCDisassembler::Success && "Invalid register");
  Inst.addOperand(Inst.getOperand(0));
  Inst.addOperand(MCOperand::createImm(0));
  return MCDisassembler::Success;
}

static DecodeStatus decodeCSSPushPopchk(MCInst &Inst, uint32_t Insn,
                                        uint64_t Address,
                                        const MCDisassembler *Decoder) {
  uint32_t Rs1 = fieldFromInstruction(Insn, 7, 5);
  [[maybe_unused]] DecodeStatus Result =
      DecodeGPRX1X5RegisterClass(Inst, Rs1, Address, Decoder);
  assert(Result == MCDisassembler::Success && "Invalid register");
  return MCDisassembler::Success;
}

static DecodeStatus decodeRVCInstrRdSImm(MCInst &Inst, uint32_t Insn,
                                         uint64_t Address,
                                         const MCDisassembler *Decoder) {
  Inst.addOperand(MCOperand::createReg(RISCV::X0));
  uint32_t SImm6 =
      fieldFromInstruction(Insn, 12, 1) << 5 | fieldFromInstruction(Insn, 2, 5);
  [[maybe_unused]] DecodeStatus Result =
      decodeSImmOperand<6>(Inst, SImm6, Address, Decoder);
  assert(Result == MCDisassembler::Success && "Invalid immediate");
  return MCDisassembler::Success;
}

static DecodeStatus decodeRVCInstrRdRs1UImm(MCInst &Inst, uint32_t Insn,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  Inst.addOperand(MCOperand::createReg(RISCV::X0));
  Inst.addOperand(Inst.getOperand(0));
  uint32_t UImm6 =
      fieldFromInstruction(Insn, 12, 1) << 5 | fieldFromInstruction(Insn, 2, 5);
  [[maybe_unused]] DecodeStatus Result =
      decodeUImmOperand<6>(Inst, UImm6, Address, Decoder);
  assert(Result == MCDisassembler::Success && "Invalid immediate");
  return MCDisassembler::Success;
}

static DecodeStatus decodeRVCInstrRdRs2(MCInst &Inst, uint32_t Insn,
                                        uint64_t Address,
                                        const MCDisassembler *Decoder) {
  uint32_t Rd = fieldFromInstruction(Insn, 7, 5);
  uint32_t Rs2 = fieldFromInstruction(Insn, 2, 5);
  DecodeGPRRegisterClass(Inst, Rd, Address, Decoder);
  DecodeGPRRegisterClass(Inst, Rs2, Address, Decoder);
  return MCDisassembler::Success;
}

static DecodeStatus decodeRVCInstrRdRs1Rs2(MCInst &Inst, uint32_t Insn,
                                           uint64_t Address,
                                           const MCDisassembler *Decoder) {
  uint32_t Rd = fieldFromInstruction(Insn, 7, 5);
  uint32_t Rs2 = fieldFromInstruction(Insn, 2, 5);
  DecodeGPRRegisterClass(Inst, Rd, Address, Decoder);
  Inst.addOperand(Inst.getOperand(0));
  DecodeGPRRegisterClass(Inst, Rs2, Address, Decoder);
  return MCDisassembler::Success;
}

static DecodeStatus decodeXTHeadMemPair(MCInst &Inst, uint32_t Insn,
                                        uint64_t Address,
                                        const MCDisassembler *Decoder) {
  uint32_t Rd1 = fieldFromInstruction(Insn, 7, 5);
  uint32_t Rs1 = fieldFromInstruction(Insn, 15, 5);
  uint32_t Rd2 = fieldFromInstruction(Insn, 20, 5);
  uint32_t UImm2 = fieldFromInstruction(Insn, 25, 2);
  DecodeGPRRegisterClass(Inst, Rd1, Address, Decoder);
  DecodeGPRRegisterClass(Inst, Rd2, Address, Decoder);
  DecodeGPRRegisterClass(Inst, Rs1, Address, Decoder);
  [[maybe_unused]] DecodeStatus Result =
      decodeUImmOperand<2>(Inst, UImm2, Address, Decoder);
  assert(Result == MCDisassembler::Success && "Invalid immediate");

  // Disassemble the final operand which is implicit.
  unsigned Opcode = Inst.getOpcode();
  bool IsWordOp = (Opcode == RISCV::TH_LWD || Opcode == RISCV::TH_LWUD ||
                   Opcode == RISCV::TH_SWD);
  if (IsWordOp)
    Inst.addOperand(MCOperand::createImm(3));
  else
    Inst.addOperand(MCOperand::createImm(4));

  return MCDisassembler::Success;
}

static DecodeStatus decodeZcmpRlist(MCInst &Inst, uint32_t Imm,
                                    uint64_t Address, const void *Decoder) {
  if (Imm <= 3)
    return MCDisassembler::Fail;
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeRegReg(MCInst &Inst, uint32_t Insn, uint64_t Address,
                                 const MCDisassembler *Decoder) {
  uint32_t Rs1 = fieldFromInstruction(Insn, 0, 5);
  uint32_t Rs2 = fieldFromInstruction(Insn, 5, 5);
  DecodeGPRRegisterClass(Inst, Rs1, Address, Decoder);
  DecodeGPRRegisterClass(Inst, Rs2, Address, Decoder);
  return MCDisassembler::Success;
}

static DecodeStatus decodeZcmpSpimm(MCInst &Inst, uint32_t Imm,
                                    uint64_t Address, const void *Decoder) {
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

// Add implied SP operand for C.*SP compressed instructions. The SP operand
// isn't explicitly encoded in the instruction.
void RISCVDisassembler::addSPOperands(MCInst &MI) const {
  const MCInstrDesc &MCID = MCII->get(MI.getOpcode());
  for (unsigned i = 0; i < MCID.getNumOperands(); i++)
    if (MCID.operands()[i].RegClass == RISCV::SPRegClassID)
      MI.insert(MI.begin() + i, MCOperand::createReg(RISCV::X2));
}

namespace {

struct DecoderListEntry {
  const uint8_t *Table;
  FeatureBitset ContainedFeatures;
  const char *Desc;

  bool haveContainedFeatures(const FeatureBitset &ActiveFeatures) const {
    return ContainedFeatures.none() ||
           (ContainedFeatures & ActiveFeatures).any();
  }
};

} // end anonymous namespace

static constexpr FeatureBitset XCVFeatureGroup = {
    RISCV::FeatureVendorXCVbitmanip, RISCV::FeatureVendorXCVelw,
    RISCV::FeatureVendorXCVmac,      RISCV::FeatureVendorXCVmem,
    RISCV::FeatureVendorXCValu,      RISCV::FeatureVendorXCVsimd,
    RISCV::FeatureVendorXCVbi};

static constexpr FeatureBitset XRivosFeatureGroup = {
    RISCV::FeatureVendorXRivosVisni,
    RISCV::FeatureVendorXRivosVizip,
};

static constexpr FeatureBitset XqciFeatureGroup = {
    RISCV::FeatureVendorXqcia,   RISCV::FeatureVendorXqciac,
    RISCV::FeatureVendorXqcibi,  RISCV::FeatureVendorXqcibm,
    RISCV::FeatureVendorXqcicli, RISCV::FeatureVendorXqcicm,
    RISCV::FeatureVendorXqcics,  RISCV::FeatureVendorXqcicsr,
    RISCV::FeatureVendorXqciint, RISCV::FeatureVendorXqcilb,
    RISCV::FeatureVendorXqcili,  RISCV::FeatureVendorXqcilia,
    RISCV::FeatureVendorXqcilo,  RISCV::FeatureVendorXqcilsm,
    RISCV::FeatureVendorXqcisim, RISCV::FeatureVendorXqcisls,
};

static constexpr FeatureBitset XSfVectorGroup = {
    RISCV::FeatureVendorXSfvcp, RISCV::FeatureVendorXSfvqmaccdod,
    RISCV::FeatureVendorXSfvqmaccqoq, RISCV::FeatureVendorXSfvfwmaccqqq,
    RISCV::FeatureVendorXSfvfnrclipxfqf};
static constexpr FeatureBitset XSfSystemGroup = {
    RISCV::FeatureVendorXSiFivecdiscarddlone,
    RISCV::FeatureVendorXSiFivecflushdlone,
};

static constexpr FeatureBitset XTHeadGroup = {
    RISCV::FeatureVendorXTHeadBa,      RISCV::FeatureVendorXTHeadBb,
    RISCV::FeatureVendorXTHeadBs,      RISCV::FeatureVendorXTHeadCondMov,
    RISCV::FeatureVendorXTHeadCmo,     RISCV::FeatureVendorXTHeadFMemIdx,
    RISCV::FeatureVendorXTHeadMac,     RISCV::FeatureVendorXTHeadMemIdx,
    RISCV::FeatureVendorXTHeadMemPair, RISCV::FeatureVendorXTHeadSync,
    RISCV::FeatureVendorXTHeadVdot};

static constexpr DecoderListEntry DecoderList32[]{
    // Vendor Extensions
    {DecoderTableXVentana32,
     {RISCV::FeatureVendorXVentanaCondOps},
     "XVentanaCondOps"},
    {DecoderTableXTHead32, XTHeadGroup, "T-Head extensions"},
    {DecoderTableXSfvector32, XSfVectorGroup, "SiFive vector extensions"},
    {DecoderTableXSfsystem32, XSfSystemGroup, "SiFive system extensions"},
    {DecoderTableXSfcease32, {RISCV::FeatureVendorXSfcease}, "SiFive sf.cease"},
    {DecoderTableXmipslsp32, {RISCV::FeatureVendorXMIPSLSP}, "MIPS mips.lsp"},
    {DecoderTableXmipscmove32,
     {RISCV::FeatureVendorXMIPSCMove},
     "MIPS mips.ccmov"},
    // Standard Extensions
    {DecoderTableXCV32, XCVFeatureGroup, "CORE-V extensions"},
    {DecoderTableXqci32, XqciFeatureGroup, "Qualcomm uC Extensions"},
    {DecoderTableXRivos32, XRivosFeatureGroup, "Rivos"},
    {DecoderTable32, {}, "standard 32-bit instructions"},
    {DecoderTableRV32Only32, {}, "RV32-only standard 32-bit instructions"},
    {DecoderTableZfinx32, {}, "Zfinx (Float in Integer)"},
    {DecoderTableZdinxRV32Only32, {}, "RV32-only Zdinx (Double in Integer)"},
};

DecodeStatus RISCVDisassembler::getInstruction32(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &CS) const {
  if (Bytes.size() < 4) {
    Size = 0;
    return MCDisassembler::Fail;
  }
  Size = 4;

  uint32_t Insn = support::endian::read32le(Bytes.data());

  for (const DecoderListEntry &Entry : DecoderList32) {
    if (!Entry.haveContainedFeatures(STI.getFeatureBits()))
      continue;

    LLVM_DEBUG(dbgs() << "Trying " << Entry.Desc << " table:\n");
    DecodeStatus Result =
        decodeInstruction(Entry.Table, MI, Insn, Address, this, STI);
    if (Result == MCDisassembler::Fail)
      continue;

    return Result;
  }

  return MCDisassembler::Fail;
}

static constexpr DecoderListEntry DecoderList16[]{
    // Vendor Extensions
    {DecoderTableXqci16, XqciFeatureGroup, "Qualcomm uC 16-bit"},
    {DecoderTableXqccmp16,
     {RISCV::FeatureVendorXqccmp},
     "Xqccmp (Qualcomm 16-bit Push/Pop & Double Move Instructions)"},
    {DecoderTableXwchc16, {RISCV::FeatureVendorXwchc}, "WCH QingKe XW"},
    // Standard Extensions
    // DecoderTableZicfiss16 must be checked before DecoderTable16.
    {DecoderTableZicfiss16, {}, "Zicfiss (Shadow Stack 16-bit)"},
    {DecoderTable16, {}, "standard 16-bit instructions"},
    {DecoderTableRV32Only16, {}, "RV32-only 16-bit instructions"},
    // Zc* instructions incompatible with Zcf or Zcd
    {DecoderTableZcOverlap16,
     {},
     "ZcOverlap (16-bit Instructions overlapping with Zcf/Zcd)"},
};

DecodeStatus RISCVDisassembler::getInstruction16(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &CS) const {
  if (Bytes.size() < 2) {
    Size = 0;
    return MCDisassembler::Fail;
  }
  Size = 2;

  uint32_t Insn = support::endian::read16le(Bytes.data());

  for (const DecoderListEntry &Entry : DecoderList16) {
    if (!Entry.haveContainedFeatures(STI.getFeatureBits()))
      continue;

    LLVM_DEBUG(dbgs() << "Trying " << Entry.Desc << "table:\n");
    DecodeStatus Result =
        decodeInstruction(Entry.Table, MI, Insn, Address, this, STI);
    if (Result == MCDisassembler::Fail)
      continue;

    addSPOperands(MI);

    return Result;
  }

  return MCDisassembler::Fail;
}

static constexpr DecoderListEntry DecoderList48[]{
    {DecoderTableXqci48, XqciFeatureGroup, "Qualcomm uC 48bit"},
};

DecodeStatus RISCVDisassembler::getInstruction48(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &CS) const {
  if (Bytes.size() < 6) {
    Size = 0;
    return MCDisassembler::Fail;
  }
  Size = 6;

  uint64_t Insn = 0;
  for (size_t i = Size; i-- != 0;)
    Insn += (static_cast<uint64_t>(Bytes[i]) << 8 * i);

  for (const DecoderListEntry &Entry : DecoderList48) {
    if (!Entry.haveContainedFeatures(STI.getFeatureBits()))
      continue;

    LLVM_DEBUG(dbgs() << "Trying " << Entry.Desc << "table:\n");
    DecodeStatus Result =
        decodeInstruction(Entry.Table, MI, Insn, Address, this, STI);
    if (Result == MCDisassembler::Fail)
      continue;

    return Result;
  }

  return MCDisassembler::Fail;
}

DecodeStatus RISCVDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                               ArrayRef<uint8_t> Bytes,
                                               uint64_t Address,
                                               raw_ostream &CS) const {
  CommentStream = &CS;
  // It's a 16 bit instruction if bit 0 and 1 are not 0b11.
  if ((Bytes[0] & 0b11) != 0b11)
    return getInstruction16(MI, Size, Bytes, Address, CS);

  // It's a 32 bit instruction if bit 1:0 are 0b11(checked above) and bits 4:2
  // are not 0b111.
  if ((Bytes[0] & 0b1'1100) != 0b1'1100)
    return getInstruction32(MI, Size, Bytes, Address, CS);

  // 48-bit instructions are encoded as 0bxx011111.
  if ((Bytes[0] & 0b11'1111) == 0b01'1111) {
    return getInstruction48(MI, Size, Bytes, Address, CS);
  }

  // 64-bit instructions are encoded as 0x0111111.
  if ((Bytes[0] & 0b111'1111) == 0b011'1111) {
    Size = Bytes.size() >= 8 ? 8 : 0;
    return MCDisassembler::Fail;
  }

  // Remaining cases need to check a second byte.
  if (Bytes.size() < 2) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  // 80-bit through 176-bit instructions are encoded as 0bxnnnxxxx_x1111111.
  // Where the number of bits is (80 + (nnn * 16)) for nnn != 0b111.
  unsigned nnn = (Bytes[1] >> 4) & 0b111;
  if (nnn != 0b111) {
    Size = 10 + (nnn * 2);
    if (Bytes.size() < Size)
      Size = 0;
    return MCDisassembler::Fail;
  }

  // Remaining encodings are reserved for > 176-bit instructions.
  Size = 0;
  return MCDisassembler::Fail;
}
