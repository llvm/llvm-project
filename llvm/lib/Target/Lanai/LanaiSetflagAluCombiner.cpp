//===-- LanaiSetflagAluCombiner.cpp - Pass to combine set flag & ALU ops --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Lanai.h"
#include "LanaiTargetMachine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "lanai-setflag-alu-combiner"

STATISTIC(NumSetflagAluCombined,
          "Number of SET_FLAG and ALU instructions combined");

static llvm::cl::opt<bool> DisableSetflagAluCombiner(
    "disable-lanai-setflag-alu-combiner", llvm::cl::init(false),
    llvm::cl::desc("Do not combine SET_FLAG and ALU operators"),
    llvm::cl::Hidden);

namespace llvm {
void initializeLanaiSetflagAluCombinerPass(PassRegistry &);
} // namespace llvm

namespace {
typedef MachineBasicBlock::iterator MbbIterator;
typedef MachineFunction::iterator MfIterator;

class LanaiSetflagAluCombiner : public MachineFunctionPass {
public:
  static char ID;
  LanaiSetflagAluCombiner() : MachineFunctionPass(ID) {
    initializeLanaiSetflagAluCombinerPass(*PassRegistry::getPassRegistry());
  }

  const char *getPassName() const override {
    return "Lanai SET_FLAG ALU combiner pass";
  }

  bool runOnMachineFunction(MachineFunction &F) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::AllVRegsAllocated);
  }

private:
  bool CombineSetflagAluInBasicBlock(MachineFunction *MF,
                                     MachineBasicBlock *BB);
};
} // namespace

char LanaiSetflagAluCombiner::ID = 0;

INITIALIZE_PASS(LanaiSetflagAluCombiner, DEBUG_TYPE,
                "Lanai SET_FLAG ALU combiner pass", false, false)

namespace {

const unsigned kInvalid = -1;

static unsigned flagSettingOpcodeVariant(unsigned OldOpcode) {
  switch (OldOpcode) {
  case Lanai::ADD_I_HI:
    return Lanai::ADD_F_I_HI;
  case Lanai::ADD_I_LO:
    return Lanai::ADD_F_I_LO;
  case Lanai::ADD_R:
    return Lanai::ADD_F_R;
  case Lanai::ADD_R_CC:
    return Lanai::ADD_F_R_CC;
  case Lanai::ADDC_I_HI:
    return Lanai::ADDC_F_I_HI;
  case Lanai::ADDC_I_LO:
    return Lanai::ADDC_F_I_LO;
  case Lanai::ADDC_R:
    return Lanai::ADDC_F_R;
  case Lanai::ADDC_R_CC:
    return Lanai::ADDC_F_R_CC;
  case Lanai::AND_I_HI:
    return Lanai::AND_F_I_HI;
  case Lanai::AND_I_LO:
    return Lanai::AND_F_I_LO;
  case Lanai::AND_R:
    return Lanai::AND_F_R;
  case Lanai::AND_R_CC:
    return Lanai::AND_F_R_CC;
  case Lanai::OR_I_HI:
    return Lanai::OR_F_I_HI;
  case Lanai::OR_I_LO:
    return Lanai::OR_F_I_LO;
  case Lanai::OR_R:
    return Lanai::OR_F_R;
  case Lanai::OR_R_CC:
    return Lanai::OR_F_R_CC;
  case Lanai::SL_I:
    return Lanai::SL_F_I;
  case Lanai::SRL_R:
    return Lanai::SRL_F_R;
  case Lanai::SA_I:
    return Lanai::SA_F_I;
  case Lanai::SRA_R:
    return Lanai::SRA_F_R;
  case Lanai::SUB_I_HI:
    return Lanai::SUB_F_I_HI;
  case Lanai::SUB_I_LO:
    return Lanai::SUB_F_I_LO;
  case Lanai::SUB_R:
    return Lanai::SUB_F_R;
  case Lanai::SUB_R_CC:
    return Lanai::SUB_F_R_CC;
  case Lanai::SUBB_I_HI:
    return Lanai::SUBB_F_I_HI;
  case Lanai::SUBB_I_LO:
    return Lanai::SUBB_F_I_LO;
  case Lanai::SUBB_R:
    return Lanai::SUBB_F_R;
  case Lanai::SUBB_R_CC:
    return Lanai::SUBB_F_R_CC;
  case Lanai::XOR_I_HI:
    return Lanai::XOR_F_I_HI;
  case Lanai::XOR_I_LO:
    return Lanai::XOR_F_I_LO;
  case Lanai::XOR_R:
    return Lanai::XOR_F_R;
  case Lanai::XOR_R_CC:
    return Lanai::XOR_F_R_CC;
  default:
    return kInvalid;
  }
}

// Returns whether opcode corresponds to instruction that sets flags.
static bool isFlagSettingInstruction(MbbIterator Instruction) {
  return Instruction->killsRegister(Lanai::SR);
}

// Return the Conditional Code operand for a given instruction kind. For
// example, operand at index 1 of a BRIND_CC instruction is the conditional code
// (eq, ne, etc.). Returns -1 if the instruction does not have a conditional
// code.
static int getCCOperandPosition(unsigned Opcode) {
  switch (Opcode) {
  case Lanai::BRIND_CC:
  case Lanai::BRIND_CCA:
  case Lanai::BRR:
  case Lanai::BRCC:
  case Lanai::SCC:
    return 1;
  case Lanai::SELECT:
  case Lanai::ADDC_F_R_CC:
  case Lanai::ADDC_R_CC:
  case Lanai::ADD_F_R_CC:
  case Lanai::ADD_R_CC:
  case Lanai::AND_F_R_CC:
  case Lanai::AND_R_CC:
  case Lanai::OR_F_R_CC:
  case Lanai::OR_R_CC:
  case Lanai::SUBB_F_R_CC:
  case Lanai::SUBB_R_CC:
  case Lanai::SUB_F_R_CC:
  case Lanai::SUB_R_CC:
  case Lanai::XOR_F_R_CC:
  case Lanai::XOR_R_CC:
    return 3;
  default:
    return -1;
  }
}

// Returns true if instruction is a lowered SET_FLAG instruction with 0/R0 as
// the first operand and whose conditional code is such that it can be merged
// (i.e., EQ, NE, PL and MI).
static bool isSuitableSetflag(MbbIterator Instruction, MbbIterator End) {
  unsigned Opcode = Instruction->getOpcode();
  if (Opcode == Lanai::SFSUB_F_RI || Opcode == Lanai::SFSUB_F_RR) {
    const MachineOperand &Operand = Instruction->getOperand(1);
    if (Operand.isReg() && Operand.getReg() != Lanai::R0)
      return false;
    if (Operand.isImm() && Operand.getImm() != 0)
      return false;

    MbbIterator SCCUserIter = Instruction;
    while (SCCUserIter != End) {
      ++SCCUserIter;
      if (SCCUserIter == End)
        break;
      // Skip debug instructions. Debug instructions don't affect codegen.
      if (SCCUserIter->isDebugValue())
        continue;
      // Early exit when encountering flag setting or return instruction.
      if (isFlagSettingInstruction(SCCUserIter))
        // Only return true if flags are set post the flag setting instruction
        // tested or a return is executed.
        return true;
      int CCIndex = getCCOperandPosition(SCCUserIter->getOpcode());
      if (CCIndex != -1) {
        LPCC::CondCode CC = static_cast<LPCC::CondCode>(
            SCCUserIter->getOperand(CCIndex).getImm());
        // Return false if the flag is used outside of a EQ, NE, PL and MI.
        if (CC != LPCC::ICC_EQ && CC != LPCC::ICC_NE && CC != LPCC::ICC_PL &&
            CC != LPCC::ICC_MI)
          return false;
      }
    }
  }

  return false;
}

// Combines a SET_FLAG instruction comparing a register with 0 and an ALU
// operation that sets the same register used in the comparison into a single
// flag setting ALU instruction (both instructions combined are removed and new
// flag setting ALU operation inserted where ALU instruction was).
bool LanaiSetflagAluCombiner::CombineSetflagAluInBasicBlock(
    MachineFunction *MF, MachineBasicBlock *BB) {
  bool Modified = false;
  const TargetInstrInfo *TII =
      MF->getSubtarget<LanaiSubtarget>().getInstrInfo();

  MbbIterator SetflagIter = BB->begin();
  MbbIterator End = BB->end();
  MbbIterator Begin = BB->begin();
  while (SetflagIter != End) {
    bool Replaced = false;
    if (isSuitableSetflag(SetflagIter, End)) {
      MbbIterator AluIter = SetflagIter;
      while (AluIter != Begin) {
        --AluIter;
        // Skip debug instructions. Debug instructions don't affect codegen.
        if (AluIter->isDebugValue())
          continue;
        // Early exit when encountering flag setting instruction.
        if (isFlagSettingInstruction(AluIter))
          break;
        // Check that output of AluIter is equal to input of SetflagIter.
        if (AluIter->getNumOperands() > 1 && AluIter->getOperand(0).isReg() &&
            (AluIter->getOperand(0).getReg() ==
             SetflagIter->getOperand(0).getReg())) {
          unsigned NewOpc = flagSettingOpcodeVariant(AluIter->getOpcode());
          if (NewOpc == kInvalid)
            break;

          // Change the ALU instruction to the flag setting variant.
          AluIter->setDesc(TII->get(NewOpc));
          AluIter->addImplicitDefUseOperands(*MF);

          Replaced = true;
          ++NumSetflagAluCombined;
          break;
        }
      }
      // Erase the setflag instruction if merged.
      if (Replaced)
        BB->erase(SetflagIter++);
    }

    Modified |= Replaced;
    if (!Replaced)
      ++SetflagIter;
  }

  return Modified;
}

// Driver function that iterates over the machine basic building blocks of a
// machine function
bool LanaiSetflagAluCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (DisableSetflagAluCombiner)
    return false;

  bool Modified = false;
  MfIterator End = MF.end();
  for (MfIterator MFI = MF.begin(); MFI != End; ++MFI) {
    Modified |= CombineSetflagAluInBasicBlock(&MF, &*MFI);
  }
  return Modified;
}
} // namespace

FunctionPass *llvm::createLanaiSetflagAluCombinerPass() {
  return new LanaiSetflagAluCombiner();
}
