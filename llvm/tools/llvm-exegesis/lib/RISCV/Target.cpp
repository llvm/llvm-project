//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../Target.h"

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCVInstrInfo.h"

// include computeAvailableFeatures and computeRequiredFeatures.
#define GET_AVAILABLE_OPCODE_CHECKER
#include "RISCVGenInstrInfo.inc"

#include "llvm/CodeGen/MachineInstrBuilder.h"

#include <vector>

namespace llvm {
namespace exegesis {

#include "RISCVGenExegesis.inc"

namespace {

// Stores constant value to a general-purpose (integer) register.
static std::vector<MCInst> loadIntReg(const MCSubtargetInfo &STI,
                                      MCRegister Reg, const APInt &Value) {
  SmallVector<MCInst, 8> MCInstSeq;
  MCRegister DestReg = Reg;

  RISCVMatInt::generateMCInstSeq(Value.getSExtValue(), STI, DestReg, MCInstSeq);

  std::vector<MCInst> MatIntInstrs(MCInstSeq.begin(), MCInstSeq.end());
  return MatIntInstrs;
}

const MCPhysReg ScratchIntReg = RISCV::X30; // t5

// Stores constant bits to a floating-point register.
static std::vector<MCInst> loadFPRegBits(const MCSubtargetInfo &STI,
                                         MCRegister Reg, const APInt &Bits,
                                         unsigned FmvOpcode) {
  std::vector<MCInst> Instrs = loadIntReg(STI, ScratchIntReg, Bits);
  Instrs.push_back(MCInstBuilder(FmvOpcode).addReg(Reg).addReg(ScratchIntReg));
  return Instrs;
}

// main idea is:
// we support APInt only if (represented as double) it has zero fractional
// part: 1.0, 2.0, 3.0, etc... then we can do the trick: write int to tmp reg t5
// and then do FCVT this is only reliable thing in 32-bit mode, otherwise we
// need to use __floatsidf
static std::vector<MCInst> loadFP64RegBits32(const MCSubtargetInfo &STI,
                                             MCRegister Reg,
                                             const APInt &Bits) {
  double D = Bits.bitsToDouble();
  double IPart;
  double FPart = std::modf(D, &IPart);

  if (std::abs(FPart) > std::numeric_limits<double>::epsilon()) {
    errs() << "loadFP64RegBits32 is not implemented for doubles like " << D
           << ", please remove fractional part\n";
    return {};
  }

  std::vector<MCInst> Instrs = loadIntReg(STI, ScratchIntReg, Bits);
  Instrs.push_back(
      MCInstBuilder(RISCV::FCVT_D_W).addReg(Reg).addReg(ScratchIntReg));
  return Instrs;
}

static MCInst nop() {
  // ADDI X0, X0, 0
  return MCInstBuilder(RISCV::ADDI)
      .addReg(RISCV::X0)
      .addReg(RISCV::X0)
      .addImm(0);
}

static bool isVectorRegList(MCRegister Reg) {
  return RISCV::VRM2RegClass.contains(Reg) ||
         RISCV::VRM4RegClass.contains(Reg) ||
         RISCV::VRM8RegClass.contains(Reg) ||
         RISCV::VRN2M1RegClass.contains(Reg) ||
         RISCV::VRN2M2RegClass.contains(Reg) ||
         RISCV::VRN2M4RegClass.contains(Reg) ||
         RISCV::VRN3M1RegClass.contains(Reg) ||
         RISCV::VRN3M2RegClass.contains(Reg) ||
         RISCV::VRN4M1RegClass.contains(Reg) ||
         RISCV::VRN4M2RegClass.contains(Reg) ||
         RISCV::VRN5M1RegClass.contains(Reg) ||
         RISCV::VRN6M1RegClass.contains(Reg) ||
         RISCV::VRN7M1RegClass.contains(Reg) ||
         RISCV::VRN8M1RegClass.contains(Reg);
}

class ExegesisRISCVTarget : public ExegesisTarget {
public:
  ExegesisRISCVTarget();

  bool matchesArch(Triple::ArchType Arch) const override;

  std::vector<MCInst> setRegTo(const MCSubtargetInfo &STI, MCRegister Reg,
                               const APInt &Value) const override;

  MCRegister getDefaultLoopCounterRegister(const Triple &) const override;

  void decrementLoopCounterAndJump(MachineBasicBlock &MBB,
                                   MachineBasicBlock &TargetMBB,
                                   const MCInstrInfo &MII,
                                   MCRegister LoopRegister) const override;

  MCRegister getScratchMemoryRegister(const Triple &TT) const override;

  void fillMemoryOperands(InstructionTemplate &IT, MCRegister Reg,
                          unsigned Offset) const override;

  ArrayRef<MCPhysReg> getUnavailableRegisters() const override;

  bool allowAsBackToBack(const Instruction &Instr) const override {
    return !Instr.Description.isPseudo();
  }

  Error randomizeTargetMCOperand(const Instruction &Instr, const Variable &Var,
                                 MCOperand &AssignedValue,
                                 const BitVector &ForbiddenRegs) const override;

  std::vector<InstructionTemplate>
  generateInstructionVariants(const Instruction &Instr,
                              unsigned MaxConfigsPerOpcode) const override;
};

ExegesisRISCVTarget::ExegesisRISCVTarget()
    : ExegesisTarget(RISCVCpuPfmCounters, RISCV_MC::isOpcodeAvailable) {}

bool ExegesisRISCVTarget::matchesArch(Triple::ArchType Arch) const {
  return Arch == Triple::riscv32 || Arch == Triple::riscv64;
}

std::vector<MCInst> ExegesisRISCVTarget::setRegTo(const MCSubtargetInfo &STI,
                                                  MCRegister Reg,
                                                  const APInt &Value) const {
  if (RISCV::GPRRegClass.contains(Reg))
    return loadIntReg(STI, Reg, Value);
  if (RISCV::FPR16RegClass.contains(Reg))
    return loadFPRegBits(STI, Reg, Value, RISCV::FMV_H_X);
  if (RISCV::FPR32RegClass.contains(Reg))
    return loadFPRegBits(STI, Reg, Value, RISCV::FMV_W_X);
  if (RISCV::FPR64RegClass.contains(Reg)) {
    if (STI.hasFeature(RISCV::Feature64Bit))
      return loadFPRegBits(STI, Reg, Value, RISCV::FMV_D_X);
    return loadFP64RegBits32(STI, Reg, Value);
  }
  if (Reg == RISCV::FRM || Reg == RISCV::VL || Reg == RISCV::VLENB ||
      Reg == RISCV::VTYPE || RISCV::GPRPairRegClass.contains(Reg) ||
      RISCV::VRRegClass.contains(Reg) || isVectorRegList(Reg)) {
    // Don't initialize:
    // - FRM
    // - VL, VLENB, VTYPE
    // - vector registers (and vector register lists)
    // - Zfinx registers
    // Generate 'NOP' so that exegesis treats such registers as initialized
    // (it tries to initialize them with '0' anyway).
    return {nop()};
  }
  errs() << "setRegTo is not implemented for Reg " << Reg
         << ", results will be unreliable\n";
  return {};
}

const MCPhysReg DefaultLoopCounterReg = RISCV::X31; // t6
const MCPhysReg ScratchMemoryReg = RISCV::X10;      // a0

MCRegister
ExegesisRISCVTarget::getDefaultLoopCounterRegister(const Triple &) const {
  return DefaultLoopCounterReg;
}

void ExegesisRISCVTarget::decrementLoopCounterAndJump(
    MachineBasicBlock &MBB, MachineBasicBlock &TargetMBB,
    const MCInstrInfo &MII, MCRegister LoopRegister) const {
  BuildMI(&MBB, DebugLoc(), MII.get(RISCV::ADDI))
      .addDef(LoopRegister)
      .addUse(LoopRegister)
      .addImm(-1);
  BuildMI(&MBB, DebugLoc(), MII.get(RISCV::BNE))
      .addUse(LoopRegister)
      .addUse(RISCV::X0)
      .addMBB(&TargetMBB);
}

MCRegister
ExegesisRISCVTarget::getScratchMemoryRegister(const Triple &TT) const {
  return ScratchMemoryReg; // a0
}

void ExegesisRISCVTarget::fillMemoryOperands(InstructionTemplate &IT,
                                             MCRegister Reg,
                                             unsigned Offset) const {
  // TODO: for now we ignore Offset because have no way
  // to detect it in instruction.
  auto &I = IT.getInstr();

  auto MemOpIt =
      find_if(I.Operands, [](const Operand &Op) { return Op.isMemory(); });
  assert(MemOpIt != I.Operands.end() &&
         "Instruction must have memory operands");

  const Operand &MemOp = *MemOpIt;

  assert(MemOp.isReg() && "Memory operand expected to be register");

  IT.getValueFor(MemOp) = MCOperand::createReg(Reg);
}

const MCPhysReg UnavailableRegisters[4] = {RISCV::X0, DefaultLoopCounterReg,
                                           ScratchIntReg, ScratchMemoryReg};

ArrayRef<MCPhysReg> ExegesisRISCVTarget::getUnavailableRegisters() const {
  return UnavailableRegisters;
}

Error ExegesisRISCVTarget::randomizeTargetMCOperand(
    const Instruction &Instr, const Variable &Var, MCOperand &AssignedValue,
    const BitVector &ForbiddenRegs) const {
  uint8_t OperandType =
      Instr.getPrimaryOperand(Var).getExplicitOperandInfo().OperandType;

  switch (OperandType) {
  case RISCVOp::OPERAND_FRMARG:
    AssignedValue = MCOperand::createImm(RISCVFPRndMode::DYN);
    break;
  case RISCVOp::OPERAND_SIMM10_LSB0000_NONZERO:
    AssignedValue = MCOperand::createImm(0b1 << 4);
    break;
  case RISCVOp::OPERAND_SIMM6_NONZERO:
  case RISCVOp::OPERAND_UIMMLOG2XLEN_NONZERO:
    AssignedValue = MCOperand::createImm(1);
    break;
  default:
    if (OperandType >= RISCVOp::OPERAND_FIRST_RISCV_IMM &&
        OperandType <= RISCVOp::OPERAND_LAST_RISCV_IMM)
      AssignedValue = MCOperand::createImm(0);
  }
  return Error::success();
}

std::vector<InstructionTemplate>
ExegesisRISCVTarget::generateInstructionVariants(
    const Instruction &Instr, unsigned int MaxConfigsPerOpcode) const {
  InstructionTemplate IT{&Instr};
  for (const Operand &Op : Instr.Operands)
    if (Op.isMemory()) {
      IT.getValueFor(Op) = MCOperand::createReg(ScratchMemoryReg);
    }
  return {IT};
}

} // anonymous namespace

static ExegesisTarget *getTheRISCVExegesisTarget() {
  static ExegesisRISCVTarget Target;
  return &Target;
}

void InitializeRISCVExegesisTarget() {
  ExegesisTarget::registerTarget(getTheRISCVExegesisTarget());
}

} // namespace exegesis
} // namespace llvm
