//===-- DPUMergeComboInstrPass.cpp - DPU Merge combo instruction Pass -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "DPUTargetMachine.h"
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <set>

#include "DPU.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

#define GET_INSTRINFO_ENUM

#include "DPUCondCodes.h"
#include "DPUGenInstrInfo.inc"
#include "DPUISelLowering.h"
#include "MCTargetDesc/DPUAsmCondition.h"

#define GET_REGINFO_ENUM
#include "DPUGenRegisterInfo.inc"

#define DEBUG_TYPE "dpu-merge-combo-instr"

using namespace llvm;

namespace {
class DPUMergeComboInstrPass : public MachineFunctionPass {
public:
  static char ID;

  explicit DPUMergeComboInstrPass(DPUTargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  llvm::StringRef getPassName() const override {
    return "DPU Merge Combo Instructions";
  }

private:
  const DPUTargetMachine &TM;
};

char DPUMergeComboInstrPass::ID = 0;
} // namespace

FunctionPass *llvm::createDPUMergeComboInstrPass(DPUTargetMachine &tm) {
  return new DPUMergeComboInstrPass(tm);
}

static bool
canFindUnaryConditionForCondition(int64_t cond, int64_t immediate,
                                  int64_t &actualCond,
                                  std::set<ISD::CondCode> usableConditions) {
  switch (immediate) {
  default:
    return false;
  case 0:
    actualCond = cond;
    break;
  case -1:
    switch ((ISD::CondCode)cond) {
    default:
      return false;
    case ISD::SETOGT:
      actualCond = ISD::SETOGE;
      break;
    case ISD::SETGT:
      actualCond = ISD::SETGE;
      break;
    case ISD::SETOLE:
      actualCond = ISD::SETOLT;
      break;
    case ISD::SETLE:
      actualCond = ISD::SETLT;
      break;
    }
    break;
  }

  return usableConditions.find((ISD::CondCode)actualCond) !=
         usableConditions.end();
}

static int64_t translateToSourceCondition(int64_t cond) {
  switch ((ISD::CondCode)cond) {
  default:
    return cond;
  case ISD::SETOEQ:
  case ISD::SETEQ:
    return DPU::AddedISDCondCode::ISD_COND_SZ;
  case ISD::SETONE:
  case ISD::SETNE:
    return DPU::AddedISDCondCode::ISD_COND_SNZ;
  case ISD::SETOGE:
  case ISD::SETGE:
    return DPU::AddedISDCondCode::ISD_COND_SPL;
  case ISD::SETOLT:
  case ISD::SETLT:
    return DPU::AddedISDCondCode::ISD_COND_SMI;
  }
}

static int64_t translateToUnaryDPUAsmCondition(int64_t cond) {
  switch (cond) {
  default:
    llvm_unreachable("invalid condition");
  case DPU::AddedISDCondCode::ISD_COND_SZ:
    return DPUAsmCondition::SourceZero;
  case DPU::AddedISDCondCode::ISD_COND_SNZ:
    return DPUAsmCondition::SourceNotZero;
  case DPU::AddedISDCondCode::ISD_COND_SPL:
    return DPUAsmCondition::SourcePositiveOrNull;
  case DPU::AddedISDCondCode::ISD_COND_SMI:
    return DPUAsmCondition::SourceNegative;
  case ISD::SETFALSE:
  case ISD::SETFALSE2:
    return DPUAsmCondition::False;
  case ISD::SETTRUE:
  case ISD::SETTRUE2:
    return DPUAsmCondition::True;
  case ISD::SETOEQ:
  case ISD::SETUEQ:
  case ISD::SETEQ:
    return DPUAsmCondition::Zero;
  case ISD::SETONE:
  case ISD::SETUNE:
  case ISD::SETNE:
    return DPUAsmCondition::NotZero;
  case ISD::SETOGE:
  case ISD::SETUGE:
  case ISD::SETGE:
    return DPUAsmCondition::PositiveOrNull;
  case ISD::SETOLT:
  case ISD::SETULT:
  case ISD::SETLT:
    return DPUAsmCondition::Negative;
  }
}

static bool canEncodeImmediateOnNBitsSigned(int64_t value, uint32_t bits) {
  int64_t threshold = (1L << (bits - 1)) - 1;

  return (value <= threshold) && (value >= ~threshold);
}

enum OpPrototype {
  OprrriUnlimited,
  OprriUnlimited,
  OprriLimited,
  OprirLimited,
  Oprrr,
  Oprr,
  OpriLimited
};

static const ISD::CondCode minimalConditions[] = {
    ISD::SETFALSE, ISD::SETTRUE, ISD::SETFALSE2, ISD::SETTRUE2};
static const ISD::CondCode reducedConditions[] = {
    ISD::SETFALSE, ISD::SETOEQ,    ISD::SETONE, ISD::SETUEQ, ISD::SETUNE,
    ISD::SETTRUE,  ISD::SETFALSE2, ISD::SETEQ,  ISD::SETNE,  ISD::SETTRUE2};
static const ISD::CondCode normalConditions[] = {
    ISD::SETFALSE, ISD::SETOEQ,  ISD::SETOGE,    ISD::SETOLT, ISD::SETONE,
    ISD::SETUEQ,   ISD::SETTRUE, ISD::SETFALSE2, ISD::SETEQ,  ISD::SETGE,
    ISD::SETLT,    ISD::SETNE,   ISD::SETTRUE2};
static const ISD::CondCode extendedConditions[] = {
    ISD::SETFALSE, ISD::SETOEQ,  ISD::SETOGT, ISD::SETOGE,  ISD::SETOLT,
    ISD::SETOLE,   ISD::SETONE,  ISD::SETUEQ, ISD::SETUGT,  ISD::SETUGE,
    ISD::SETULT,   ISD::SETULE,  ISD::SETUNE, ISD::SETTRUE, ISD::SETFALSE2,
    ISD::SETEQ,    ISD::SETGT,   ISD::SETGE,  ISD::SETLT,   ISD::SETLE,
    ISD::SETNE,    ISD::SETTRUE2};

static const ISD::CondCode sourceConditions[] = {
    ISD::SETOEQ, ISD::SETOGE, ISD::SETOLT, ISD::SETONE, ISD::SETUEQ,
    ISD::SETEQ,  ISD::SETGE,  ISD::SETLT,  ISD::SETNE};

static bool mergeComboInstructionsInMBB(MachineBasicBlock *MBB,
                                        const DPUInstrInfo &InstrInfo) {
  MachineBasicBlock::reverse_iterator I = MBB->rbegin(), REnd = MBB->rend();
  MachineInstr *LastInst, *SecondLastInst;
  unsigned int LastOpc, SecondLastOpc;
  enum OpPrototype OpPrototype;
  unsigned int OpJumpOpc, OpNullJumpOpc;
  bool ImmCanBeEncodedOn8Bits, ImmCanBeEncodedOn11Bits;
  std::set<ISD::CondCode> usableConditions;
  std::set<ISD::CondCode> usableConditionsTranslatableToSourceConditions;

  std::set<ISD::CondCode> normalConditionsSet = std::set<ISD::CondCode>(
      std::begin(normalConditions), std::end(normalConditions));
  std::set<ISD::CondCode> minimalConditionsSet = std::set<ISD::CondCode>(
      std::begin(minimalConditions), std::end(minimalConditions));
  std::set<ISD::CondCode> reducedConditionsSet = std::set<ISD::CondCode>(
      std::begin(reducedConditions), std::end(reducedConditions));
  std::set<ISD::CondCode> extendedConditionsSet = std::set<ISD::CondCode>(
      std::begin(extendedConditions), std::end(extendedConditions));

  std::set<ISD::CondCode> sourceConditionsSet = std::set<ISD::CondCode>(
      std::begin(sourceConditions), std::end(sourceConditions));

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugValue()) {
    ++I;
  }

  if (I == REnd) {
    return false;
  }

  LastInst = &*I;

  if (++I == REnd) {
    return false;
  }

  SecondLastInst = &*I;

  LastOpc = LastInst->getOpcode();
  SecondLastOpc = SecondLastInst->getOpcode();

  switch (SecondLastOpc) {
  default:
    return false;
  case DPU::MOVEri:
    OpPrototype = OpriLimited;
    OpJumpOpc = DPU::MOVErici;
    OpNullJumpOpc = DPU::MOVErici; // should not be used
    usableConditions = normalConditionsSet;
    break;
  case DPU::MOVErr:
    OpPrototype = Oprr;
    OpJumpOpc = DPU::MOVErrci;
    OpNullJumpOpc = DPU::MOVErrci; // should not be used
    usableConditions = normalConditionsSet;
    break;
  case DPU::SUBrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::SUBrrrci;
    OpNullJumpOpc = DPU::SUBzrrci;
    usableConditions = extendedConditionsSet;
    break;
  case DPU::SUBCrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::SUBCrrrci;
    OpNullJumpOpc = DPU::SUBCzrrci;
    usableConditions = extendedConditionsSet;
    break;
  case DPU::SUBrir:
    OpPrototype = OprirLimited;
    OpJumpOpc = DPU::SUBrirci;
    OpNullJumpOpc = DPU::SUBzirci;
    usableConditions = extendedConditionsSet;
    break;
  case DPU::RSUBrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::RSUBrrrci;
    OpNullJumpOpc = DPU::RSUBzrrci;
    usableConditions = extendedConditionsSet;
    break;
  case DPU::SUBCrir:
    OpPrototype = OprirLimited;
    OpJumpOpc = DPU::SUBCrirci;
    OpNullJumpOpc = DPU::SUBCzirci;
    usableConditions = extendedConditionsSet;
    break;
  case DPU::RSUBCrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::RSUBCrrrci;
    OpNullJumpOpc = DPU::RSUBCzrrci;
    usableConditions = extendedConditionsSet;
    break;
  case DPU::ADDrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::ADDrrici;
    OpNullJumpOpc = DPU::ADDzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ADDrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::ADDrrrci;
    OpNullJumpOpc = DPU::ADDzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ADDCrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::ADDCrrici;
    OpNullJumpOpc = DPU::ADDCzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ADDCrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::ADDCrrrci;
    OpNullJumpOpc = DPU::ADDCzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ANDrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::ANDrrici;
    OpNullJumpOpc = DPU::ANDzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ANDrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::ANDrrrci;
    OpNullJumpOpc = DPU::ANDzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ORrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::ORrrici;
    OpNullJumpOpc = DPU::ORzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ORrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::ORrrrci;
    OpNullJumpOpc = DPU::ORzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::XORrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::XORrrici;
    OpNullJumpOpc = DPU::XORzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::XORrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::XORrrrci;
    OpNullJumpOpc = DPU::XORzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::NXORrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::NXORrrici;
    OpNullJumpOpc = DPU::NXORzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::NXORrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::NXORrrrci;
    OpNullJumpOpc = DPU::NXORzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::NORrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::NORrrici;
    OpNullJumpOpc = DPU::NORzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::NORrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::NORrrrci;
    OpNullJumpOpc = DPU::NORzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::NANDrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::NANDrrici;
    OpNullJumpOpc = DPU::NANDzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::NANDrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::NANDrrrci;
    OpNullJumpOpc = DPU::NANDzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ANDNrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::ANDNrrici;
    OpNullJumpOpc = DPU::ANDNzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ANDNrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::ANDNrrrci;
    OpNullJumpOpc = DPU::ANDNzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ORNrri:
    OpPrototype = OprriLimited;
    OpJumpOpc = DPU::ORNrrici;
    OpNullJumpOpc = DPU::ORNzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ORNrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::ORNrrrci;
    OpNullJumpOpc = DPU::ORNzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSLrri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::LSLrrici;
    OpNullJumpOpc = DPU::LSLzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSLrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::LSLrrrci;
    OpNullJumpOpc = DPU::LSLzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSRrri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::LSRrrici;
    OpNullJumpOpc = DPU::LSRzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSRrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::LSRrrrci;
    OpNullJumpOpc = DPU::LSRzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ASRrri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::ASRrrici;
    OpNullJumpOpc = DPU::ASRzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ASRrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::ASRrrrci;
    OpNullJumpOpc = DPU::ASRzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ROLrri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::ROLrrici;
    OpNullJumpOpc = DPU::ROLzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ROLrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::ROLrrrci;
    OpNullJumpOpc = DPU::ROLzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::RORrri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::RORrrici;
    OpNullJumpOpc = DPU::RORzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::RORrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::RORrrrci;
    OpNullJumpOpc = DPU::RORzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSLXrri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::LSLXrrici;
    OpNullJumpOpc = DPU::LSLXzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSLXrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::LSLXrrrci;
    OpNullJumpOpc = DPU::LSLXzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSL1rri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::LSL1rrici;
    OpNullJumpOpc = DPU::LSL1zrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSL1rrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::LSL1rrrci;
    OpNullJumpOpc = DPU::LSL1zrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSL1Xrri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::LSL1Xrrici;
    OpNullJumpOpc = DPU::LSL1Xzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSL1Xrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::LSL1Xrrrci;
    OpNullJumpOpc = DPU::LSL1Xzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSRXrri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::LSRXrrici;
    OpNullJumpOpc = DPU::LSRXzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSRXrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::LSRXrrrci;
    OpNullJumpOpc = DPU::LSRXzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSR1rri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::LSR1rrici;
    OpNullJumpOpc = DPU::LSR1zrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSR1rrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::LSR1rrrci;
    OpNullJumpOpc = DPU::LSR1zrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSR1Xrri:
    OpPrototype = OprriUnlimited;
    OpJumpOpc = DPU::LSR1Xrrici;
    OpNullJumpOpc = DPU::LSR1Xzrici;
    usableConditions = normalConditionsSet;
    break;
  case DPU::LSR1Xrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::LSR1Xrrrci;
    OpNullJumpOpc = DPU::LSR1Xzrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::ROL_ADDrrri:
    OpPrototype = OprrriUnlimited;
    OpJumpOpc = DPU::ROL_ADDrrrici;
    OpNullJumpOpc = DPU::ROL_ADDzrrici;
    usableConditions = minimalConditionsSet;
    break;
  case DPU::LSR_ADDrrri:
    OpPrototype = OprrriUnlimited;
    OpJumpOpc = DPU::LSR_ADDrrrici;
    OpNullJumpOpc = DPU::LSR_ADDzrrici;
    usableConditions = minimalConditionsSet;
    break;
  case DPU::LSL_ADDrrri:
    OpPrototype = OprrriUnlimited;
    OpJumpOpc = DPU::LSL_ADDrrrici;
    OpNullJumpOpc = DPU::LSL_ADDzrrici;
    usableConditions = minimalConditionsSet;
    break;
  case DPU::LSL_SUBrrri:
    OpPrototype = OprrriUnlimited;
    OpJumpOpc = DPU::LSL_SUBrrrici;
    OpNullJumpOpc = DPU::LSL_SUBzrrici;
    usableConditions = minimalConditionsSet;
    break;
  case DPU::CAOrr:
    OpPrototype = Oprr;
    OpJumpOpc = DPU::CAOrrci;
    OpNullJumpOpc = DPU::CAOzrci;
    usableConditions = reducedConditionsSet;
    break;
  case DPU::CLZrr:
    OpPrototype = Oprr;
    OpJumpOpc = DPU::CLZrrci;
    OpNullJumpOpc = DPU::CLZzrci;
    usableConditions = reducedConditionsSet;
    break;
  case DPU::CLOrr:
    OpPrototype = Oprr;
    OpJumpOpc = DPU::CLOrrci;
    OpNullJumpOpc = DPU::CLOzrci;
    usableConditions = reducedConditionsSet;
    break;
  case DPU::CLSrr:
    OpPrototype = Oprr;
    OpJumpOpc = DPU::CLSrrci;
    OpNullJumpOpc = DPU::CLSzrci;
    usableConditions = reducedConditionsSet;
    break;
  case DPU::MUL_UL_ULrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::MUL_UL_ULrrrci;
    OpNullJumpOpc = DPU::MUL_UL_ULrrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::MUL_SL_ULrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::MUL_SL_ULrrrci;
    OpNullJumpOpc = DPU::MUL_SL_ULrrrci;
    usableConditions = normalConditionsSet;
    break;
  case DPU::MUL_SL_SLrrr:
    OpPrototype = Oprrr;
    OpJumpOpc = DPU::MUL_SL_SLrrrci;
    OpNullJumpOpc = DPU::MUL_SL_SLzrrci;
    usableConditions = normalConditionsSet;
    break;
  }

  usableConditionsTranslatableToSourceConditions = sourceConditionsSet;

  ImmCanBeEncodedOn8Bits = true;
  ImmCanBeEncodedOn11Bits = true;

  if ((OpPrototype == OprriLimited) || (OpPrototype == OprirLimited)) {
    MachineOperand &immOperand = SecondLastInst->getOperand(2);
    if (!immOperand.isImm()) {
      return false;
    }

    int64_t immOpValue = immOperand.getImm();
    ImmCanBeEncodedOn8Bits = canEncodeImmediateOnNBitsSigned(immOpValue, 8);
    ImmCanBeEncodedOn11Bits = canEncodeImmediateOnNBitsSigned(immOpValue, 11);
  } else if (OpPrototype == OpriLimited) {
    MachineOperand &immOperand = SecondLastInst->getOperand(1);
    if (!immOperand.isImm()) {
      return false;
    }

    int64_t immOpValue = immOperand.getImm();
    ImmCanBeEncodedOn8Bits = canEncodeImmediateOnNBitsSigned(immOpValue, 8);
    ImmCanBeEncodedOn11Bits = canEncodeImmediateOnNBitsSigned(immOpValue, 11);
  }

  switch (LastOpc) {
  default:
    return false;
  case DPU::JUMPi: {
    if (!ImmCanBeEncodedOn8Bits) {
      return false;
    }

    int64_t actualCondition = ISD::SETTRUE2;
    MachineInstrBuilder ComboInst =
        BuildMI(MBB, SecondLastInst->getDebugLoc(), InstrInfo.get(OpJumpOpc))
            .add(SecondLastInst->getOperand(0));

    switch (OpPrototype) {
    case OprrriUnlimited:
      ComboInst.add(SecondLastInst->getOperand(1))
          .add(SecondLastInst->getOperand(2))
          .add(SecondLastInst->getOperand(3));
      break;
    case OprriUnlimited:
    case OprriLimited:
    case OprirLimited:
    case Oprrr:
      ComboInst.add(SecondLastInst->getOperand(1))
          .add(SecondLastInst->getOperand(2));
      break;
    case Oprr:
    case OpriLimited:
      ComboInst.add(SecondLastInst->getOperand(1));
      break;
    }

    actualCondition = translateToUnaryDPUAsmCondition(actualCondition);
    auto actualConditionOperand = MachineOperand::CreateImm(actualCondition);
    ComboInst.add(actualConditionOperand).add(LastInst->getOperand(0));

    LastInst->eraseFromParent();
    SecondLastInst->eraseFromParent();

    return true;
  }
  case DPU::TmpJcci:
  case DPU::Jcci: {
    bool isSourceCondition = false;

    if (SecondLastInst->getOperand(0).getReg() !=
        LastInst->getOperand(1).getReg()) {
      switch (OpPrototype) {
      case OprrriUnlimited:
      case OprriUnlimited:
      case OprriLimited:
      case OprirLimited:
      case Oprrr:
      case Oprr:
        if ((SecondLastInst->getOperand(1).getReg() !=
             LastInst->getOperand(1).getReg())) {
          return false;
        }
        isSourceCondition = true;
        usableConditions = sourceConditionsSet;
        break;
      case OpriLimited:
        return false;
      }
    }

    int64_t actualCondition;

    std::set<ISD::CondCode> availableConditions =
        isSourceCondition ? usableConditionsTranslatableToSourceConditions
                          : usableConditions;

    if (!canFindUnaryConditionForCondition(
            LastInst->getOperand(0).getImm(), LastInst->getOperand(2).getImm(),
            actualCondition, availableConditions)) {
      return false;
    }

    if (isSourceCondition) {
      actualCondition = translateToSourceCondition(actualCondition);
    }

    actualCondition = translateToUnaryDPUAsmCondition(actualCondition);

    MachineInstrBuilder ComboInst;

    if (LastInst->getOperand(1).isKill() && !isSourceCondition) {
      if (!ImmCanBeEncodedOn11Bits) {
        return false;
      }
      // todo: this is not optimal. One register has been allocated but not used
      // now. This can become an issue (unnecessary spilling)
      ComboInst = BuildMI(MBB, SecondLastInst->getDebugLoc(),
                          InstrInfo.get(OpNullJumpOpc)).addReg(DPU::ZERO);
    } else {
      if (!ImmCanBeEncodedOn8Bits) {
        return false;
      }
      ComboInst =
          BuildMI(MBB, SecondLastInst->getDebugLoc(), InstrInfo.get(OpJumpOpc))
              .add(SecondLastInst->getOperand(0));
    }

    switch (OpPrototype) {
    case OprrriUnlimited:
      ComboInst.add(SecondLastInst->getOperand(1))
          .add(SecondLastInst->getOperand(2))
          .add(SecondLastInst->getOperand(3));
      break;
    case OprriUnlimited:
    case OprriLimited:
    case OprirLimited:
    case Oprrr:
      ComboInst.add(SecondLastInst->getOperand(1))
          .add(SecondLastInst->getOperand(2));
      break;
    case Oprr:
    case OpriLimited:
      ComboInst.add(SecondLastInst->getOperand(1));
      break;
    }

    LastInst->getOperand(0).setImm(actualCondition);
    ComboInst.add(LastInst->getOperand(0))
        .add(LastInst->getOperand(LastInst->getNumOperands() - 1));

    LastInst->eraseFromParent();
    SecondLastInst->eraseFromParent();

    return true;
  }
  case DPU::Jcc:
    return false;
  }
}

bool DPUMergeComboInstrPass::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** DPU/MergeComboInstrPass: " << MF.getName()
                    << " **********\n\n");

  auto &SubTarget = static_cast<const DPUSubtarget &>(MF.getSubtarget());
  auto &InstrInfo = *SubTarget.getInstrInfo();
  bool changeMade = false;

  for (auto &MFI : MF) {
    MachineBasicBlock *MBB = &MFI;

    changeMade |= mergeComboInstructionsInMBB(MBB, InstrInfo);
  }

  return changeMade;
}
