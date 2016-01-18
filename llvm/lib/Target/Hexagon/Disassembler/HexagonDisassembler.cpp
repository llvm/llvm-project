//===-- HexagonDisassembler.cpp - Disassembler for Hexagon ISA ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <vector>

using namespace llvm;
using namespace Hexagon;

#define DEBUG_TYPE "hexagon-disassembler"

// Pull DecodeStatus and its enum values into the global namespace.
typedef llvm::MCDisassembler::DecodeStatus DecodeStatus;

namespace {
/// \brief Hexagon disassembler for all Hexagon platforms.
class HexagonDisassembler : public MCDisassembler {
public:
  std::unique_ptr<MCInst *> CurrentBundle;
  HexagonDisassembler(MCSubtargetInfo const &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx), CurrentBundle(new MCInst *) {}

  DecodeStatus getSingleInstruction(MCInst &Instr, MCInst &MCB,
                                    ArrayRef<uint8_t> Bytes, uint64_t Address,
                                    raw_ostream &VStream, raw_ostream &CStream,
                                    bool &Complete) const;
  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &VStream,
                              raw_ostream &CStream) const override;
};
}

static DecodeStatus DecodeModRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                               uint64_t Address,
                                               const void *Decoder);
static DecodeStatus DecodeIntRegsLow8RegisterClass(MCInst &Inst, unsigned RegNo,
                                                   uint64_t Address,
                                                   const void *Decoder);
static DecodeStatus DecodeVectorRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                                  uint64_t Address,
                                                  const void *Decoder);
static DecodeStatus DecodeVecDblRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                                  uint64_t Address,
                                                  const void *Decoder);
static DecodeStatus DecodeVecPredRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                                   uint64_t Address,
                                                   const void *Decoder);
static DecodeStatus DecodeCtrRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                               uint64_t Address,
                                               const void *Decoder);
static DecodeStatus DecodeCtrRegs64RegisterClass(MCInst &Inst, unsigned RegNo,
                                                 uint64_t Address,
                                                 void const *Decoder);

static unsigned GetSubinstOpcode(unsigned IClass, unsigned inst, unsigned &op,
                                 raw_ostream &os);
static void AddSubinstOperands(MCInst *MI, unsigned opcode, unsigned inst);

static DecodeStatus s16ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                  const void *Decoder);
static DecodeStatus s12ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                  const void *Decoder);
static DecodeStatus s11_0ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                    const void *Decoder);
static DecodeStatus s11_1ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                    const void *Decoder);
static DecodeStatus s11_2ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                    const void *Decoder);
static DecodeStatus s11_3ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                    const void *Decoder);
static DecodeStatus s10ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                  const void *Decoder);
static DecodeStatus s8ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                 const void *Decoder);
static DecodeStatus s6_0ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                   const void *Decoder);
static DecodeStatus s4_0ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                   const void *Decoder);
static DecodeStatus s4_1ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                   const void *Decoder);
static DecodeStatus s4_2ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                   const void *Decoder);
static DecodeStatus s4_3ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                   const void *Decoder);
static DecodeStatus s4_6ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                   const void *Decoder);
static DecodeStatus s3_6ImmDecoder(MCInst &MI, unsigned tmp, uint64_t Address,
                                   const void *Decoder);

static const uint16_t IntRegDecoderTable[] = {
    Hexagon::R0,  Hexagon::R1,  Hexagon::R2,  Hexagon::R3,  Hexagon::R4,
    Hexagon::R5,  Hexagon::R6,  Hexagon::R7,  Hexagon::R8,  Hexagon::R9,
    Hexagon::R10, Hexagon::R11, Hexagon::R12, Hexagon::R13, Hexagon::R14,
    Hexagon::R15, Hexagon::R16, Hexagon::R17, Hexagon::R18, Hexagon::R19,
    Hexagon::R20, Hexagon::R21, Hexagon::R22, Hexagon::R23, Hexagon::R24,
    Hexagon::R25, Hexagon::R26, Hexagon::R27, Hexagon::R28, Hexagon::R29,
    Hexagon::R30, Hexagon::R31};

static const uint16_t PredRegDecoderTable[] = {Hexagon::P0, Hexagon::P1,
                                               Hexagon::P2, Hexagon::P3};

static DecodeStatus DecodeRegisterClass(MCInst &Inst, unsigned RegNo,
                                        const uint16_t Table[], size_t Size) {
  if (RegNo < Size) {
    Inst.addOperand(MCOperand::createReg(Table[RegNo]));
    return MCDisassembler::Success;
  } else
    return MCDisassembler::Fail;
}

static DecodeStatus DecodeIntRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                               uint64_t /*Address*/,
                                               void const *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  unsigned Register = IntRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeCtrRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                               uint64_t /*Address*/,
                                               const void *Decoder) {
  static const uint16_t CtrlRegDecoderTable[] = {
      Hexagon::SA0,  Hexagon::LC0,        Hexagon::SA1,  Hexagon::LC1,
      Hexagon::P3_0, Hexagon::NoRegister, Hexagon::C6,   Hexagon::C7,
      Hexagon::USR,  Hexagon::PC,         Hexagon::UGP,  Hexagon::GP,
      Hexagon::CS0,  Hexagon::CS1,        Hexagon::UPCL, Hexagon::UPCH};

  if (RegNo >= sizeof(CtrlRegDecoderTable) / sizeof(CtrlRegDecoderTable[0]))
    return MCDisassembler::Fail;

  if (CtrlRegDecoderTable[RegNo] == Hexagon::NoRegister)
    return MCDisassembler::Fail;

  unsigned Register = CtrlRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeCtrRegs64RegisterClass(MCInst &Inst, unsigned RegNo,
                                                 uint64_t /*Address*/,
                                                 void const *Decoder) {
  static const uint16_t CtrlReg64DecoderTable[] = {
      Hexagon::C1_0,       Hexagon::NoRegister, Hexagon::C3_2,
      Hexagon::NoRegister, Hexagon::NoRegister, Hexagon::NoRegister,
      Hexagon::C7_6,       Hexagon::NoRegister, Hexagon::C9_8,
      Hexagon::NoRegister, Hexagon::C11_10,     Hexagon::NoRegister,
      Hexagon::CS,         Hexagon::NoRegister, Hexagon::UPC,
      Hexagon::NoRegister};

  if (RegNo >= sizeof(CtrlReg64DecoderTable) / sizeof(CtrlReg64DecoderTable[0]))
    return MCDisassembler::Fail;

  if (CtrlReg64DecoderTable[RegNo] == Hexagon::NoRegister)
    return MCDisassembler::Fail;

  unsigned Register = CtrlReg64DecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeModRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                               uint64_t /*Address*/,
                                               const void *Decoder) {
  unsigned Register = 0;
  switch (RegNo) {
  case 0:
    Register = Hexagon::M0;
    break;
  case 1:
    Register = Hexagon::M1;
    break;
  default:
    return MCDisassembler::Fail;
  }
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeDoubleRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                                  uint64_t /*Address*/,
                                                  const void *Decoder) {
  static const uint16_t DoubleRegDecoderTable[] = {
      Hexagon::D0,  Hexagon::D1,  Hexagon::D2,  Hexagon::D3,
      Hexagon::D4,  Hexagon::D5,  Hexagon::D6,  Hexagon::D7,
      Hexagon::D8,  Hexagon::D9,  Hexagon::D10, Hexagon::D11,
      Hexagon::D12, Hexagon::D13, Hexagon::D14, Hexagon::D15};

  return (DecodeRegisterClass(Inst, RegNo >> 1, DoubleRegDecoderTable,
                              sizeof(DoubleRegDecoderTable)));
}

static DecodeStatus DecodePredRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                                uint64_t /*Address*/,
                                                void const *Decoder) {
  if (RegNo > 3)
    return MCDisassembler::Fail;

  unsigned Register = PredRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

#include "HexagonGenDisassemblerTables.inc"

static MCDisassembler *createHexagonDisassembler(Target const &T,
                                                 MCSubtargetInfo const &STI,
                                                 MCContext &Ctx) {
  return new HexagonDisassembler(STI, Ctx);
}

extern "C" void LLVMInitializeHexagonDisassembler() {
  TargetRegistry::RegisterMCDisassembler(TheHexagonTarget,
                                         createHexagonDisassembler);
}

DecodeStatus HexagonDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &os,
                                                 raw_ostream &cs) const {
  DecodeStatus Result = DecodeStatus::Success;
  bool Complete = false;
  Size = 0;

  *CurrentBundle = &MI;
  MI.clear();
  MI.setOpcode(Hexagon::BUNDLE);
  MI.addOperand(MCOperand::createImm(0));
  while (Result == Success && Complete == false) {
    if (Bytes.size() < HEXAGON_INSTR_SIZE)
      return MCDisassembler::Fail;
    MCInst *Inst = new (getContext()) MCInst;
    Result = getSingleInstruction(*Inst, MI, Bytes, Address, os, cs, Complete);
    MI.addOperand(MCOperand::createInst(Inst));
    Size += HEXAGON_INSTR_SIZE;
    Bytes = Bytes.slice(HEXAGON_INSTR_SIZE);
  }
  return Result;
}

DecodeStatus HexagonDisassembler::getSingleInstruction(
    MCInst &MI, MCInst &MCB, ArrayRef<uint8_t> Bytes, uint64_t Address,
    raw_ostream &os, raw_ostream &cs, bool &Complete) const {
  assert(Bytes.size() >= HEXAGON_INSTR_SIZE);

  uint32_t Instruction =
      llvm::support::endian::read<uint32_t, llvm::support::little,
                                  llvm::support::unaligned>(Bytes.data());

  auto BundleSize = HexagonMCInstrInfo::bundleSize(MCB);
  if ((Instruction & HexagonII::INST_PARSE_MASK) ==
      HexagonII::INST_PARSE_LOOP_END) {
    if (BundleSize == 0)
      HexagonMCInstrInfo::setInnerLoop(MCB);
    else if (BundleSize == 1)
      HexagonMCInstrInfo::setOuterLoop(MCB);
    else
      return DecodeStatus::Fail;
  }

  DecodeStatus Result = DecodeStatus::Success;
  if ((Instruction & HexagonII::INST_PARSE_MASK) ==
      HexagonII::INST_PARSE_DUPLEX) {
    // Determine the instruction class of each instruction in the duplex.
    unsigned duplexIClass, IClassLow, IClassHigh;

    duplexIClass = ((Instruction >> 28) & 0xe) | ((Instruction >> 13) & 0x1);
    switch (duplexIClass) {
    default:
      return MCDisassembler::Fail;
    case 0:
      IClassLow = HexagonII::HSIG_L1;
      IClassHigh = HexagonII::HSIG_L1;
      break;
    case 1:
      IClassLow = HexagonII::HSIG_L2;
      IClassHigh = HexagonII::HSIG_L1;
      break;
    case 2:
      IClassLow = HexagonII::HSIG_L2;
      IClassHigh = HexagonII::HSIG_L2;
      break;
    case 3:
      IClassLow = HexagonII::HSIG_A;
      IClassHigh = HexagonII::HSIG_A;
      break;
    case 4:
      IClassLow = HexagonII::HSIG_L1;
      IClassHigh = HexagonII::HSIG_A;
      break;
    case 5:
      IClassLow = HexagonII::HSIG_L2;
      IClassHigh = HexagonII::HSIG_A;
      break;
    case 6:
      IClassLow = HexagonII::HSIG_S1;
      IClassHigh = HexagonII::HSIG_A;
      break;
    case 7:
      IClassLow = HexagonII::HSIG_S2;
      IClassHigh = HexagonII::HSIG_A;
      break;
    case 8:
      IClassLow = HexagonII::HSIG_S1;
      IClassHigh = HexagonII::HSIG_L1;
      break;
    case 9:
      IClassLow = HexagonII::HSIG_S1;
      IClassHigh = HexagonII::HSIG_L2;
      break;
    case 10:
      IClassLow = HexagonII::HSIG_S1;
      IClassHigh = HexagonII::HSIG_S1;
      break;
    case 11:
      IClassLow = HexagonII::HSIG_S2;
      IClassHigh = HexagonII::HSIG_S1;
      break;
    case 12:
      IClassLow = HexagonII::HSIG_S2;
      IClassHigh = HexagonII::HSIG_L1;
      break;
    case 13:
      IClassLow = HexagonII::HSIG_S2;
      IClassHigh = HexagonII::HSIG_L2;
      break;
    case 14:
      IClassLow = HexagonII::HSIG_S2;
      IClassHigh = HexagonII::HSIG_S2;
      break;
    }

    // Set the MCInst to be a duplex instruction. Which one doesn't matter.
    MI.setOpcode(Hexagon::DuplexIClass0);

    // Decode each instruction in the duplex.
    // Create an MCInst for each instruction.
    unsigned instLow = Instruction & 0x1fff;
    unsigned instHigh = (Instruction >> 16) & 0x1fff;
    unsigned opLow;
    if (GetSubinstOpcode(IClassLow, instLow, opLow, os) !=
        MCDisassembler::Success)
      return MCDisassembler::Fail;
    unsigned opHigh;
    if (GetSubinstOpcode(IClassHigh, instHigh, opHigh, os) !=
        MCDisassembler::Success)
      return MCDisassembler::Fail;
    MCInst *MILow = new (getContext()) MCInst;
    MILow->setOpcode(opLow);
    MCInst *MIHigh = new (getContext()) MCInst;
    MIHigh->setOpcode(opHigh);
    AddSubinstOperands(MILow, opLow, instLow);
    AddSubinstOperands(MIHigh, opHigh, instHigh);
    // see ConvertToSubInst() in
    // lib/Target/Hexagon/MCTargetDesc/HexagonMCDuplexInfo.cpp

    // Add the duplex instruction MCInsts as operands to the passed in MCInst.
    MCOperand OPLow = MCOperand::createInst(MILow);
    MCOperand OPHigh = MCOperand::createInst(MIHigh);
    MI.addOperand(OPLow);
    MI.addOperand(OPHigh);
    Complete = true;
  } else {
    if ((Instruction & HexagonII::INST_PARSE_MASK) ==
        HexagonII::INST_PARSE_PACKET_END)
      Complete = true;
    // Calling the auto-generated decoder function.
    Result =
        decodeInstruction(DecoderTable32, MI, Instruction, Address, this, STI);
  }

  return Result;
}
static DecodeStatus DecodeIntRegsLow8RegisterClass(MCInst &Inst, unsigned RegNo,
                                                   uint64_t Address,
                                                   const void *Decoder) {
  return DecodeIntRegsRegisterClass(Inst, RegNo, Address, Decoder);
}


static DecodeStatus DecodeVectorRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                                  uint64_t /*Address*/,
                                                  const void *Decoder) {
  static const uint16_t VecRegDecoderTable[] = {
      Hexagon::V0,  Hexagon::V1,  Hexagon::V2,  Hexagon::V3,  Hexagon::V4,
      Hexagon::V5,  Hexagon::V6,  Hexagon::V7,  Hexagon::V8,  Hexagon::V9,
      Hexagon::V10, Hexagon::V11, Hexagon::V12, Hexagon::V13, Hexagon::V14,
      Hexagon::V15, Hexagon::V16, Hexagon::V17, Hexagon::V18, Hexagon::V19,
      Hexagon::V20, Hexagon::V21, Hexagon::V22, Hexagon::V23, Hexagon::V24,
      Hexagon::V25, Hexagon::V26, Hexagon::V27, Hexagon::V28, Hexagon::V29,
      Hexagon::V30, Hexagon::V31};

  return (DecodeRegisterClass(Inst, RegNo, VecRegDecoderTable,
                              sizeof(VecRegDecoderTable)));
}

static DecodeStatus DecodeVecDblRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                                  uint64_t /*Address*/,
                                                  const void *Decoder) {
  static const uint16_t VecDblRegDecoderTable[] = {
      Hexagon::W0,  Hexagon::W1,  Hexagon::W2,  Hexagon::W3,
      Hexagon::W4,  Hexagon::W5,  Hexagon::W6,  Hexagon::W7,
      Hexagon::W8,  Hexagon::W9,  Hexagon::W10, Hexagon::W11,
      Hexagon::W12, Hexagon::W13, Hexagon::W14, Hexagon::W15};

  return (DecodeRegisterClass(Inst, RegNo >> 1, VecDblRegDecoderTable,
                              sizeof(VecDblRegDecoderTable)));
}

static DecodeStatus DecodeVecPredRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                                   uint64_t /*Address*/,
                                                   const void *Decoder) {
  static const uint16_t VecPredRegDecoderTable[] = {Hexagon::Q0, Hexagon::Q1,
                                                    Hexagon::Q2, Hexagon::Q3};

  return (DecodeRegisterClass(Inst, RegNo, VecPredRegDecoderTable,
                              sizeof(VecPredRegDecoderTable)));
}

static DecodeStatus s16ImmDecoder(MCInst &MI, unsigned tmp,
                                  uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<16>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s12ImmDecoder(MCInst &MI, unsigned tmp,
                                  uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<12>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s11_0ImmDecoder(MCInst &MI, unsigned tmp,
                                    uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<11>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s11_1ImmDecoder(MCInst &MI, unsigned tmp,
                                    uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<12>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s11_2ImmDecoder(MCInst &MI, unsigned tmp,
                                    uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<13>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s11_3ImmDecoder(MCInst &MI, unsigned tmp,
                                    uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<14>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s10ImmDecoder(MCInst &MI, unsigned tmp,
                                  uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<10>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s8ImmDecoder(MCInst &MI, unsigned tmp, uint64_t /*Address*/,
                                 const void *Decoder) {
  uint64_t imm = SignExtend64<8>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s6_0ImmDecoder(MCInst &MI, unsigned tmp,
                                   uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<6>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s4_0ImmDecoder(MCInst &MI, unsigned tmp,
                                   uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<4>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s4_1ImmDecoder(MCInst &MI, unsigned tmp,
                                   uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<5>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s4_2ImmDecoder(MCInst &MI, unsigned tmp,
                                   uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<6>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s4_3ImmDecoder(MCInst &MI, unsigned tmp,
                                   uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<7>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s4_6ImmDecoder(MCInst &MI, unsigned tmp,
                                   uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<10>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

static DecodeStatus s3_6ImmDecoder(MCInst &MI, unsigned tmp,
                                   uint64_t /*Address*/, const void *Decoder) {
  uint64_t imm = SignExtend64<9>(tmp);
  MI.addOperand(MCOperand::createImm(imm));
  return MCDisassembler::Success;
}

// These values are from HexagonGenMCCodeEmitter.inc and HexagonIsetDx.td
enum subInstBinaryValues {
  V4_SA1_addi_BITS = 0x0000,
  V4_SA1_addi_MASK = 0x1800,
  V4_SA1_addrx_BITS = 0x1800,
  V4_SA1_addrx_MASK = 0x1f00,
  V4_SA1_addsp_BITS = 0x0c00,
  V4_SA1_addsp_MASK = 0x1c00,
  V4_SA1_and1_BITS = 0x1200,
  V4_SA1_and1_MASK = 0x1f00,
  V4_SA1_clrf_BITS = 0x1a70,
  V4_SA1_clrf_MASK = 0x1e70,
  V4_SA1_clrfnew_BITS = 0x1a50,
  V4_SA1_clrfnew_MASK = 0x1e70,
  V4_SA1_clrt_BITS = 0x1a60,
  V4_SA1_clrt_MASK = 0x1e70,
  V4_SA1_clrtnew_BITS = 0x1a40,
  V4_SA1_clrtnew_MASK = 0x1e70,
  V4_SA1_cmpeqi_BITS = 0x1900,
  V4_SA1_cmpeqi_MASK = 0x1f00,
  V4_SA1_combine0i_BITS = 0x1c00,
  V4_SA1_combine0i_MASK = 0x1d18,
  V4_SA1_combine1i_BITS = 0x1c08,
  V4_SA1_combine1i_MASK = 0x1d18,
  V4_SA1_combine2i_BITS = 0x1c10,
  V4_SA1_combine2i_MASK = 0x1d18,
  V4_SA1_combine3i_BITS = 0x1c18,
  V4_SA1_combine3i_MASK = 0x1d18,
  V4_SA1_combinerz_BITS = 0x1d08,
  V4_SA1_combinerz_MASK = 0x1d08,
  V4_SA1_combinezr_BITS = 0x1d00,
  V4_SA1_combinezr_MASK = 0x1d08,
  V4_SA1_dec_BITS = 0x1300,
  V4_SA1_dec_MASK = 0x1f00,
  V4_SA1_inc_BITS = 0x1100,
  V4_SA1_inc_MASK = 0x1f00,
  V4_SA1_seti_BITS = 0x0800,
  V4_SA1_seti_MASK = 0x1c00,
  V4_SA1_setin1_BITS = 0x1a00,
  V4_SA1_setin1_MASK = 0x1e40,
  V4_SA1_sxtb_BITS = 0x1500,
  V4_SA1_sxtb_MASK = 0x1f00,
  V4_SA1_sxth_BITS = 0x1400,
  V4_SA1_sxth_MASK = 0x1f00,
  V4_SA1_tfr_BITS = 0x1000,
  V4_SA1_tfr_MASK = 0x1f00,
  V4_SA1_zxtb_BITS = 0x1700,
  V4_SA1_zxtb_MASK = 0x1f00,
  V4_SA1_zxth_BITS = 0x1600,
  V4_SA1_zxth_MASK = 0x1f00,
  V4_SL1_loadri_io_BITS = 0x0000,
  V4_SL1_loadri_io_MASK = 0x1000,
  V4_SL1_loadrub_io_BITS = 0x1000,
  V4_SL1_loadrub_io_MASK = 0x1000,
  V4_SL2_deallocframe_BITS = 0x1f00,
  V4_SL2_deallocframe_MASK = 0x1fc0,
  V4_SL2_jumpr31_BITS = 0x1fc0,
  V4_SL2_jumpr31_MASK = 0x1fc4,
  V4_SL2_jumpr31_f_BITS = 0x1fc5,
  V4_SL2_jumpr31_f_MASK = 0x1fc7,
  V4_SL2_jumpr31_fnew_BITS = 0x1fc7,
  V4_SL2_jumpr31_fnew_MASK = 0x1fc7,
  V4_SL2_jumpr31_t_BITS = 0x1fc4,
  V4_SL2_jumpr31_t_MASK = 0x1fc7,
  V4_SL2_jumpr31_tnew_BITS = 0x1fc6,
  V4_SL2_jumpr31_tnew_MASK = 0x1fc7,
  V4_SL2_loadrb_io_BITS = 0x1000,
  V4_SL2_loadrb_io_MASK = 0x1800,
  V4_SL2_loadrd_sp_BITS = 0x1e00,
  V4_SL2_loadrd_sp_MASK = 0x1f00,
  V4_SL2_loadrh_io_BITS = 0x0000,
  V4_SL2_loadrh_io_MASK = 0x1800,
  V4_SL2_loadri_sp_BITS = 0x1c00,
  V4_SL2_loadri_sp_MASK = 0x1e00,
  V4_SL2_loadruh_io_BITS = 0x0800,
  V4_SL2_loadruh_io_MASK = 0x1800,
  V4_SL2_return_BITS = 0x1f40,
  V4_SL2_return_MASK = 0x1fc4,
  V4_SL2_return_f_BITS = 0x1f45,
  V4_SL2_return_f_MASK = 0x1fc7,
  V4_SL2_return_fnew_BITS = 0x1f47,
  V4_SL2_return_fnew_MASK = 0x1fc7,
  V4_SL2_return_t_BITS = 0x1f44,
  V4_SL2_return_t_MASK = 0x1fc7,
  V4_SL2_return_tnew_BITS = 0x1f46,
  V4_SL2_return_tnew_MASK = 0x1fc7,
  V4_SS1_storeb_io_BITS = 0x1000,
  V4_SS1_storeb_io_MASK = 0x1000,
  V4_SS1_storew_io_BITS = 0x0000,
  V4_SS1_storew_io_MASK = 0x1000,
  V4_SS2_allocframe_BITS = 0x1c00,
  V4_SS2_allocframe_MASK = 0x1e00,
  V4_SS2_storebi0_BITS = 0x1200,
  V4_SS2_storebi0_MASK = 0x1f00,
  V4_SS2_storebi1_BITS = 0x1300,
  V4_SS2_storebi1_MASK = 0x1f00,
  V4_SS2_stored_sp_BITS = 0x0a00,
  V4_SS2_stored_sp_MASK = 0x1e00,
  V4_SS2_storeh_io_BITS = 0x0000,
  V4_SS2_storeh_io_MASK = 0x1800,
  V4_SS2_storew_sp_BITS = 0x0800,
  V4_SS2_storew_sp_MASK = 0x1e00,
  V4_SS2_storewi0_BITS = 0x1000,
  V4_SS2_storewi0_MASK = 0x1f00,
  V4_SS2_storewi1_BITS = 0x1100,
  V4_SS2_storewi1_MASK = 0x1f00
};

static unsigned GetSubinstOpcode(unsigned IClass, unsigned inst, unsigned &op,
                                 raw_ostream &os) {
  switch (IClass) {
  case HexagonII::HSIG_L1:
    if ((inst & V4_SL1_loadri_io_MASK) == V4_SL1_loadri_io_BITS)
      op = Hexagon::V4_SL1_loadri_io;
    else if ((inst & V4_SL1_loadrub_io_MASK) == V4_SL1_loadrub_io_BITS)
      op = Hexagon::V4_SL1_loadrub_io;
    else {
      os << "<unknown subinstruction>";
      return MCDisassembler::Fail;
    }
    break;
  case HexagonII::HSIG_L2:
    if ((inst & V4_SL2_deallocframe_MASK) == V4_SL2_deallocframe_BITS)
      op = Hexagon::V4_SL2_deallocframe;
    else if ((inst & V4_SL2_jumpr31_MASK) == V4_SL2_jumpr31_BITS)
      op = Hexagon::V4_SL2_jumpr31;
    else if ((inst & V4_SL2_jumpr31_f_MASK) == V4_SL2_jumpr31_f_BITS)
      op = Hexagon::V4_SL2_jumpr31_f;
    else if ((inst & V4_SL2_jumpr31_fnew_MASK) == V4_SL2_jumpr31_fnew_BITS)
      op = Hexagon::V4_SL2_jumpr31_fnew;
    else if ((inst & V4_SL2_jumpr31_t_MASK) == V4_SL2_jumpr31_t_BITS)
      op = Hexagon::V4_SL2_jumpr31_t;
    else if ((inst & V4_SL2_jumpr31_tnew_MASK) == V4_SL2_jumpr31_tnew_BITS)
      op = Hexagon::V4_SL2_jumpr31_tnew;
    else if ((inst & V4_SL2_loadrb_io_MASK) == V4_SL2_loadrb_io_BITS)
      op = Hexagon::V4_SL2_loadrb_io;
    else if ((inst & V4_SL2_loadrd_sp_MASK) == V4_SL2_loadrd_sp_BITS)
      op = Hexagon::V4_SL2_loadrd_sp;
    else if ((inst & V4_SL2_loadrh_io_MASK) == V4_SL2_loadrh_io_BITS)
      op = Hexagon::V4_SL2_loadrh_io;
    else if ((inst & V4_SL2_loadri_sp_MASK) == V4_SL2_loadri_sp_BITS)
      op = Hexagon::V4_SL2_loadri_sp;
    else if ((inst & V4_SL2_loadruh_io_MASK) == V4_SL2_loadruh_io_BITS)
      op = Hexagon::V4_SL2_loadruh_io;
    else if ((inst & V4_SL2_return_MASK) == V4_SL2_return_BITS)
      op = Hexagon::V4_SL2_return;
    else if ((inst & V4_SL2_return_f_MASK) == V4_SL2_return_f_BITS)
      op = Hexagon::V4_SL2_return_f;
    else if ((inst & V4_SL2_return_fnew_MASK) == V4_SL2_return_fnew_BITS)
      op = Hexagon::V4_SL2_return_fnew;
    else if ((inst & V4_SL2_return_t_MASK) == V4_SL2_return_t_BITS)
      op = Hexagon::V4_SL2_return_t;
    else if ((inst & V4_SL2_return_tnew_MASK) == V4_SL2_return_tnew_BITS)
      op = Hexagon::V4_SL2_return_tnew;
    else {
      os << "<unknown subinstruction>";
      return MCDisassembler::Fail;
    }
    break;
  case HexagonII::HSIG_A:
    if ((inst & V4_SA1_addi_MASK) == V4_SA1_addi_BITS)
      op = Hexagon::V4_SA1_addi;
    else if ((inst & V4_SA1_addrx_MASK) == V4_SA1_addrx_BITS)
      op = Hexagon::V4_SA1_addrx;
    else if ((inst & V4_SA1_addsp_MASK) == V4_SA1_addsp_BITS)
      op = Hexagon::V4_SA1_addsp;
    else if ((inst & V4_SA1_and1_MASK) == V4_SA1_and1_BITS)
      op = Hexagon::V4_SA1_and1;
    else if ((inst & V4_SA1_clrf_MASK) == V4_SA1_clrf_BITS)
      op = Hexagon::V4_SA1_clrf;
    else if ((inst & V4_SA1_clrfnew_MASK) == V4_SA1_clrfnew_BITS)
      op = Hexagon::V4_SA1_clrfnew;
    else if ((inst & V4_SA1_clrt_MASK) == V4_SA1_clrt_BITS)
      op = Hexagon::V4_SA1_clrt;
    else if ((inst & V4_SA1_clrtnew_MASK) == V4_SA1_clrtnew_BITS)
      op = Hexagon::V4_SA1_clrtnew;
    else if ((inst & V4_SA1_cmpeqi_MASK) == V4_SA1_cmpeqi_BITS)
      op = Hexagon::V4_SA1_cmpeqi;
    else if ((inst & V4_SA1_combine0i_MASK) == V4_SA1_combine0i_BITS)
      op = Hexagon::V4_SA1_combine0i;
    else if ((inst & V4_SA1_combine1i_MASK) == V4_SA1_combine1i_BITS)
      op = Hexagon::V4_SA1_combine1i;
    else if ((inst & V4_SA1_combine2i_MASK) == V4_SA1_combine2i_BITS)
      op = Hexagon::V4_SA1_combine2i;
    else if ((inst & V4_SA1_combine3i_MASK) == V4_SA1_combine3i_BITS)
      op = Hexagon::V4_SA1_combine3i;
    else if ((inst & V4_SA1_combinerz_MASK) == V4_SA1_combinerz_BITS)
      op = Hexagon::V4_SA1_combinerz;
    else if ((inst & V4_SA1_combinezr_MASK) == V4_SA1_combinezr_BITS)
      op = Hexagon::V4_SA1_combinezr;
    else if ((inst & V4_SA1_dec_MASK) == V4_SA1_dec_BITS)
      op = Hexagon::V4_SA1_dec;
    else if ((inst & V4_SA1_inc_MASK) == V4_SA1_inc_BITS)
      op = Hexagon::V4_SA1_inc;
    else if ((inst & V4_SA1_seti_MASK) == V4_SA1_seti_BITS)
      op = Hexagon::V4_SA1_seti;
    else if ((inst & V4_SA1_setin1_MASK) == V4_SA1_setin1_BITS)
      op = Hexagon::V4_SA1_setin1;
    else if ((inst & V4_SA1_sxtb_MASK) == V4_SA1_sxtb_BITS)
      op = Hexagon::V4_SA1_sxtb;
    else if ((inst & V4_SA1_sxth_MASK) == V4_SA1_sxth_BITS)
      op = Hexagon::V4_SA1_sxth;
    else if ((inst & V4_SA1_tfr_MASK) == V4_SA1_tfr_BITS)
      op = Hexagon::V4_SA1_tfr;
    else if ((inst & V4_SA1_zxtb_MASK) == V4_SA1_zxtb_BITS)
      op = Hexagon::V4_SA1_zxtb;
    else if ((inst & V4_SA1_zxth_MASK) == V4_SA1_zxth_BITS)
      op = Hexagon::V4_SA1_zxth;
    else {
      os << "<unknown subinstruction>";
      return MCDisassembler::Fail;
    }
    break;
  case HexagonII::HSIG_S1:
    if ((inst & V4_SS1_storeb_io_MASK) == V4_SS1_storeb_io_BITS)
      op = Hexagon::V4_SS1_storeb_io;
    else if ((inst & V4_SS1_storew_io_MASK) == V4_SS1_storew_io_BITS)
      op = Hexagon::V4_SS1_storew_io;
    else {
      os << "<unknown subinstruction>";
      return MCDisassembler::Fail;
    }
    break;
  case HexagonII::HSIG_S2:
    if ((inst & V4_SS2_allocframe_MASK) == V4_SS2_allocframe_BITS)
      op = Hexagon::V4_SS2_allocframe;
    else if ((inst & V4_SS2_storebi0_MASK) == V4_SS2_storebi0_BITS)
      op = Hexagon::V4_SS2_storebi0;
    else if ((inst & V4_SS2_storebi1_MASK) == V4_SS2_storebi1_BITS)
      op = Hexagon::V4_SS2_storebi1;
    else if ((inst & V4_SS2_stored_sp_MASK) == V4_SS2_stored_sp_BITS)
      op = Hexagon::V4_SS2_stored_sp;
    else if ((inst & V4_SS2_storeh_io_MASK) == V4_SS2_storeh_io_BITS)
      op = Hexagon::V4_SS2_storeh_io;
    else if ((inst & V4_SS2_storew_sp_MASK) == V4_SS2_storew_sp_BITS)
      op = Hexagon::V4_SS2_storew_sp;
    else if ((inst & V4_SS2_storewi0_MASK) == V4_SS2_storewi0_BITS)
      op = Hexagon::V4_SS2_storewi0;
    else if ((inst & V4_SS2_storewi1_MASK) == V4_SS2_storewi1_BITS)
      op = Hexagon::V4_SS2_storewi1;
    else {
      os << "<unknown subinstruction>";
      return MCDisassembler::Fail;
    }
    break;
  default:
    os << "<unknown>";
    return MCDisassembler::Fail;
  }
  return MCDisassembler::Success;
}

static unsigned getRegFromSubinstEncoding(unsigned encoded_reg) {
  if (encoded_reg < 8)
    return Hexagon::R0 + encoded_reg;
  else if (encoded_reg < 16)
    return Hexagon::R0 + encoded_reg + 8;
  return Hexagon::NoRegister;
}

static unsigned getDRegFromSubinstEncoding(unsigned encoded_dreg) {
  if (encoded_dreg < 4)
    return Hexagon::D0 + encoded_dreg;
  else if (encoded_dreg < 8)
    return Hexagon::D0 + encoded_dreg + 4;
  return Hexagon::NoRegister;
}

static void AddSubinstOperands(MCInst *MI, unsigned opcode, unsigned inst) {
  int64_t operand;
  MCOperand Op;
  switch (opcode) {
  case Hexagon::V4_SL2_deallocframe:
  case Hexagon::V4_SL2_jumpr31:
  case Hexagon::V4_SL2_jumpr31_f:
  case Hexagon::V4_SL2_jumpr31_fnew:
  case Hexagon::V4_SL2_jumpr31_t:
  case Hexagon::V4_SL2_jumpr31_tnew:
  case Hexagon::V4_SL2_return:
  case Hexagon::V4_SL2_return_f:
  case Hexagon::V4_SL2_return_fnew:
  case Hexagon::V4_SL2_return_t:
  case Hexagon::V4_SL2_return_tnew:
    // no operands for these instructions
    break;
  case Hexagon::V4_SS2_allocframe:
    // u 8-4{5_3}
    operand = ((inst & 0x1f0) >> 4) << 3;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SL1_loadri_io:
    // Rd 3-0, Rs 7-4, u 11-8{4_2}
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = (inst & 0xf00) >> 6;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SL1_loadrub_io:
    // Rd 3-0, Rs 7-4, u 11-8
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = (inst & 0xf00) >> 8;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SL2_loadrb_io:
    // Rd 3-0, Rs 7-4, u 10-8
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = (inst & 0x700) >> 8;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SL2_loadrh_io:
  case Hexagon::V4_SL2_loadruh_io:
    // Rd 3-0, Rs 7-4, u 10-8{3_1}
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = ((inst & 0x700) >> 8) << 1;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SL2_loadrd_sp:
    // Rdd 2-0, u 7-3{5_3}
    operand = getDRegFromSubinstEncoding(inst & 0x7);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = ((inst & 0x0f8) >> 3) << 3;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SL2_loadri_sp:
    // Rd 3-0, u 8-4{5_2}
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = ((inst & 0x1f0) >> 4) << 2;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SA1_addi:
    // Rx 3-0 (x2), s7 10-4
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    MI->addOperand(Op);
    operand = SignExtend64<7>((inst & 0x7f0) >> 4);
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SA1_addrx:
    // Rx 3-0 (x2), Rs 7-4
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
  case Hexagon::V4_SA1_and1:
  case Hexagon::V4_SA1_dec:
  case Hexagon::V4_SA1_inc:
  case Hexagon::V4_SA1_sxtb:
  case Hexagon::V4_SA1_sxth:
  case Hexagon::V4_SA1_tfr:
  case Hexagon::V4_SA1_zxtb:
  case Hexagon::V4_SA1_zxth:
    // Rd 3-0, Rs 7-4
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SA1_addsp:
    // Rd 3-0, u 9-4{6_2}
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = ((inst & 0x3f0) >> 4) << 2;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SA1_seti:
    // Rd 3-0, u 9-4
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = (inst & 0x3f0) >> 4;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SA1_clrf:
  case Hexagon::V4_SA1_clrfnew:
  case Hexagon::V4_SA1_clrt:
  case Hexagon::V4_SA1_clrtnew:
  case Hexagon::V4_SA1_setin1:
    // Rd 3-0
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SA1_cmpeqi:
    // Rs 7-4, u 1-0
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = inst & 0x3;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SA1_combine0i:
  case Hexagon::V4_SA1_combine1i:
  case Hexagon::V4_SA1_combine2i:
  case Hexagon::V4_SA1_combine3i:
    // Rdd 2-0, u 6-5
    operand = getDRegFromSubinstEncoding(inst & 0x7);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = (inst & 0x060) >> 5;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SA1_combinerz:
  case Hexagon::V4_SA1_combinezr:
    // Rdd 2-0, Rs 7-4
    operand = getDRegFromSubinstEncoding(inst & 0x7);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SS1_storeb_io:
    // Rs 7-4, u 11-8, Rt 3-0
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = (inst & 0xf00) >> 8;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SS1_storew_io:
    // Rs 7-4, u 11-8{4_2}, Rt 3-0
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = ((inst & 0xf00) >> 8) << 2;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SS2_storebi0:
  case Hexagon::V4_SS2_storebi1:
    // Rs 7-4, u 3-0
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = inst & 0xf;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SS2_storewi0:
  case Hexagon::V4_SS2_storewi1:
    // Rs 7-4, u 3-0{4_2}
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = (inst & 0xf) << 2;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SS2_stored_sp:
    // s 8-3{6_3}, Rtt 2-0
    operand = SignExtend64<9>(((inst & 0x1f8) >> 3) << 3);
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    operand = getDRegFromSubinstEncoding(inst & 0x7);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SS2_storeh_io:
    // Rs 7-4, u 10-8{3_1}, Rt 3-0
    operand = getRegFromSubinstEncoding((inst & 0xf0) >> 4);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    operand = ((inst & 0x700) >> 8) << 1;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    break;
  case Hexagon::V4_SS2_storew_sp:
    // u 8-4{5_2}, Rd 3-0
    operand = ((inst & 0x1f0) >> 4) << 2;
    Op = MCOperand::createImm(operand);
    MI->addOperand(Op);
    operand = getRegFromSubinstEncoding(inst & 0xf);
    Op = MCOperand::createReg(operand);
    MI->addOperand(Op);
    break;
  default:
    // don't crash with an invalid subinstruction
    // llvm_unreachable("Invalid subinstruction in duplex instruction");
    break;
  }
}
