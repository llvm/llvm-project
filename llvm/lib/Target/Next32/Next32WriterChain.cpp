//===-- Next32WriterChain.cpp - use or replace LEA instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that finds instructions that can be
// re-written as LEA instructions in order to reduce pipeline delays.
// When optimizing for size it replaces suitable LEAs with INC or DEC.
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "Next32InstrInfo.h"
#include "Next32PassTrace.h"
#include "Next32Subtarget.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace llvm {
void initializeNext32WriterChainPassPass(PassRegistry &);
}

#define WRITERCHAIN_DESC "Next32 WriterChain Fixup"
#define WRITERCHAIN_NAME "Next32-writerchain"

#define DEBUG_TYPE WRITERCHAIN_NAME

namespace {
class Next32WriterChainPass : public MachineFunctionPass {

public:
  static char ID;

  StringRef getPassName() const override { return WRITERCHAIN_DESC; }

  Next32WriterChainPass() : MachineFunctionPass(ID) {
    initializeNext32WriterChainPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Func) override;

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
  }

private:
  typedef struct ChainInfo {
    unsigned int Reg;
    Next32Constants::CondCode Condition;
    unsigned int ConditionReg;
    bool IsPriority;
    unsigned int Attribute;
    unsigned int Probability;
  } ChainInfo;

  typedef struct WriterInfo {
    unsigned int DestReg;
    unsigned int SrcReg;
    unsigned int Size;
  } WriterInfo;

  const TargetInstrInfo *TII;
  MachineBasicBlock::iterator Terminator;
  MachineBasicBlock *MBB;
  MachineFunction *MF;
  const MachineLoop *Loop;
  const MachineBranchProbabilityInfo *MBPI;

  typedef std::vector<std::pair<unsigned, unsigned>> FeedersRegsList;

  void AddWriterChain(ChainInfo CI, SmallVectorImpl<WriterInfo> &Registers);
  void AddBranchWriters(MachineBasicBlock *TargetMBB,
                        Next32Constants::CondCode Condition,
                        unsigned int CondReg);
  void AddRetWriters(Next32Constants::CondCode Condition, unsigned int CondReg);
  void AddCallWriters(Next32Constants::CondCode Condition,
                      unsigned int CondReg);
  void AddCallPtrWriters(Next32Constants::CondCode Condition,
                         unsigned int CondReg);

  void getInputFeedersRegs(MachineBasicBlock *TargetMBB, unsigned int RetFid,
                           SmallVectorImpl<WriterInfo> &FeedersRegs) const;
  void CreateMOVWithRelocation(const Twine &RelocSymName, unsigned int Reg);
  void CreateMOVWithRelocation(MCSymbol *RelocSym, unsigned int Reg);
  unsigned GetChainOpcode(bool IsCondBranch, bool IsPriority);
  Next32Constants::CondCode getCondFromTerminator(unsigned OpIdx) const;
  void SetCallAddress(MachineOperand &Callee, unsigned int Reg);
};
} // namespace

char Next32WriterChainPass::ID = 0;

INITIALIZE_PASS(Next32WriterChainPass, WRITERCHAIN_NAME, WRITERCHAIN_DESC,
                false, false)

FunctionPass *llvm::createNext32WriterChains() {
  return new Next32WriterChainPass();
}

bool Next32WriterChainPass::runOnMachineFunction(MachineFunction &Func) {
  Next32PassTrace TFunc(DEBUG_TYPE, Func);
  MachineLoopInfo *LI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MBPI = &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();
  TII = Func.getSubtarget().getInstrInfo();
  MF = &Func;
  std::list<MachineInstr *> EarseFromMI;

  for (auto &MBBI : TFunc) {
    MBB = &MBBI;
    Loop = LI->getLoopFor(MBB);
    // Look for all terminator and replace them with writer-chain
    for (auto &I : MBBI) {
      Terminator = &I;
      if ((Terminator->getDesc().TSFlags & Next32II::IsWriterChain) == 0)
        continue;

      switch (Terminator->getOpcode()) {
      case Next32::BR:
        AddBranchWriters(Terminator->getOperand(0).getMBB(),
                         Next32Constants::NoCondition, 0);
        break;
      case Next32::BR_CC:
        AddBranchWriters(Terminator->getOperand(2).getMBB(),
                         getCondFromTerminator(0),
                         Terminator->getOperand(1).getReg());
        break;
      case Next32::RET:
        AddRetWriters(Next32Constants::NoCondition, 0);
        break;
      case Next32::RETc:
        AddRetWriters(
            getCondFromTerminator(Terminator->getNumOperands() - 2),
            Terminator->getOperand(Terminator->getNumOperands() - 1).getReg());
        break;
      case Next32::CALL:
        AddCallWriters(Next32Constants::NoCondition, 0);
        break;
      case Next32::CALLc:
        AddCallWriters(
            getCondFromTerminator(Terminator->getNumOperands() - 2),
            Terminator->getOperand(Terminator->getNumOperands() - 1).getReg());
        break;
      case Next32::CALLPTRWRAPPER:
        AddCallPtrWriters(Next32Constants::NoCondition, 0);
      }

      EarseFromMI.push_back(&(*Terminator));
    }
  }

  for (auto &i : EarseFromMI)
    i->eraseFromParent();

  return true;
}

void Next32WriterChainPass::AddWriterChain(
    ChainInfo CI, SmallVectorImpl<WriterInfo> &Registers) {
  bool IsCondBranch = CI.Condition != Next32Constants::NoCondition;
  LLVM_DEBUG(dbgs() << "Adding writers to BB#" << MBB->getNumber() << "("
                    << Registers.size() << " Feeders)\n");

  unsigned Opcode = GetChainOpcode(IsCondBranch, CI.IsPriority);
  auto MI = BuildMI(*MBB, Terminator, Terminator->getDebugLoc(),
                    TII->get(Opcode), CI.Reg)
                .addImm(CI.Attribute)
                .addImm(CI.Probability);
  if (IsCondBranch)
    MI.addImm(CI.Condition).addReg(CI.ConditionReg);
  for (auto R : Registers)
    BuildMI(*MBB, Terminator, Terminator->getDebugLoc(),
            TII->get(Next32::WRITER), R.DestReg)
        .addReg(R.DestReg)
        .addReg(R.SrcReg)
        .addImm(R.Size);
}

void Next32WriterChainPass::AddBranchWriters(
    MachineBasicBlock *TargetMBB, Next32Constants::CondCode Condition,
    unsigned int CondReg) {
  if (MBB->succ_size() == 0) {
    assert(MBB->getParent()->getFunction().doesNotReturn() &&
           "Block has no successor but isn't marked noreturn.");
    return;
  }

  CreateMOVWithRelocation(TargetMBB->getSymbol(), Next32::MBB_ADDR);
  SmallVector<WriterInfo, 16> FeedersRegs;
  getInputFeedersRegs(TargetMBB, Next32::MBB_ADDR, FeedersRegs);
  bool IsPriority = false;
  if (Loop) {
    MachineBasicBlock *Header = Loop->getHeader();
    assert(Header && "No header for loop");
    IsPriority = Header->getNumber() == TargetMBB->getNumber();
  }
  ChainInfo CI = {
      Next32::MBB_ADDR,
      Condition,
      CondReg,
      IsPriority,
      0,
      (unsigned int)MBPI->getEdgeProbability(MBB, TargetMBB).scale(100)};
  AddWriterChain(CI, FeedersRegs);
}

void Next32WriterChainPass::AddRetWriters(Next32Constants::CondCode Condition,
                                          unsigned int CondReg) {
  if (MBB->getParent()->getFunction().doesNotReturn())
    return;
  if (MBB->getParent()->getFunction().hasFnAttribute(
          Next32Helpers::GetFunctionAttrName(Next32Constants::ATT_NORETURN)))
    return;
  SmallVector<WriterInfo, 16> ReturnValues;
  unsigned lastValue = Terminator->getNumOperands();
  if (Condition != Next32Constants::NoCondition)
    lastValue -= 2;

  for (unsigned int i = 0; i < lastValue; i += 3)
    ReturnValues.push_back(
        {Terminator->getOperand(i).getReg(),
         Terminator->getOperand(i + 1).getReg(),
         static_cast<unsigned int>(Terminator->getOperand(i + 2).getImm())});

  ChainInfo CI = {
      Terminator->getOperand(0).getReg(), Condition, CondReg, false, 0, 100};
  AddWriterChain(CI, ReturnValues);
}

void Next32WriterChainPass::SetCallAddress(MachineOperand &Callee,
                                           unsigned int Reg) {
  if (Callee.isReg()) {
    BuildMI(*MBB, Terminator, Terminator->getDebugLoc(), TII->get(Next32::DUP),
            Reg)
        .addReg(Callee.getReg())
        .addReg(Callee.getReg());
    return;
  }

  StringRef CalleeName;
  if (Callee.isGlobal())
    CalleeName = Callee.getGlobal()->getName();
  else if (Callee.isSymbol())
    CalleeName = Callee.getSymbolName();
  else
    llvm_unreachable("Invalid call site symbol operand");

  CreateMOVWithRelocation(CalleeName, Reg);
}

void Next32WriterChainPass::AddCallWriters(Next32Constants::CondCode Condition,
                                           unsigned int CondReg) {
  MachineOperand &Callee = Terminator->getOperand(0);
  SetCallAddress(Callee, Next32::CALL_ADDR);

  unsigned LastValue = Terminator->getNumOperands() - 2;
  if (Condition != Next32Constants::NoCondition)
    LastValue -= 2;

  bool HasSuccessor = Terminator->getOperand(LastValue + 1).isMBB();
  if (HasSuccessor)
    LastValue -= 1;

  MCSymbol *RetSymbol = Terminator->getOperand(LastValue).getMCSymbol();
  CreateMOVWithRelocation(RetSymbol, Next32::CALL_RET_FID);

  SmallVector<WriterInfo, 16> Args;
  Args.push_back({Next32::CALL_ADDR, Next32::TID,
                  llvm::Next32Constants::InstructionSize::InstructionSize32});
  Args.push_back({Next32::CALL_ADDR, Next32::CALL_RET_FID,
                  llvm::Next32Constants::InstructionSize::InstructionSize32});

  for (unsigned int i = Next32Helpers::GetNext32VariadicPosition();
       i < LastValue; i += 2)
    Args.push_back(
        {Next32::CALL_ADDR, Terminator->getOperand(i).getReg(),
         static_cast<unsigned int>(Terminator->getOperand(i + 1).getImm())});

  ChainInfo CI = {Next32::CALL_ADDR, Condition, CondReg, false, 0, 100};
  AddWriterChain(CI, Args);
  if (!HasSuccessor)
    return;

  MachineBasicBlock *Successor = Terminator->getOperand(LastValue + 2).getMBB();
  CreateMOVWithRelocation(Successor->getSymbol(), Next32::CALL_RET_BB);

  Args.clear();
  getInputFeedersRegs(Successor, Next32::CALL_RET_BB, Args);
  CI = {
      Next32::CALL_RET_BB,
      Condition,
      CondReg,
      false,
      static_cast<unsigned int>(Terminator->getOperand(LastValue + 1).getImm()),
      100};
  AddWriterChain(CI, Args);
}

void Next32WriterChainPass::AddCallPtrWriters(
    Next32Constants::CondCode Condition, unsigned int CondReg) {
  unsigned int CalleeAddressHigh = Terminator->getOperand(0).getReg();
  unsigned int CalleeAddressLow = Terminator->getOperand(1).getReg();
  unsigned LastValue = Terminator->getNumOperands() - 2;
  bool HasSuccessor = Terminator->getOperand(LastValue + 1).isMBB();

  if (HasSuccessor)
    LastValue -= 1;

  MCSymbol *RetSymbol = Terminator->getOperand(LastValue).getMCSymbol();
  CreateMOVWithRelocation(RetSymbol, Next32::CALL_RET_FID);

  SmallVector<WriterInfo, 16> Args;
  Args.push_back({CalleeAddressHigh, Next32::TID,
                  llvm::Next32Constants::InstructionSize::InstructionSize32});
  Args.push_back({CalleeAddressHigh, Next32::CALL_RET_FID,
                  llvm::Next32Constants::InstructionSize::InstructionSize32});

  for (unsigned int i = Next32Helpers::GetNext32VariadicPosition() + 1;
       i < LastValue; i += 2)
    Args.push_back(
        {CalleeAddressHigh, Terminator->getOperand(i).getReg(),
         static_cast<unsigned int>(Terminator->getOperand(i + 1).getImm())});

  BuildMI(*MBB, Terminator, Terminator->getDebugLoc(),
          TII->get(Next32::CALLPTR), CalleeAddressHigh)
      .addReg(CalleeAddressHigh)
      .addReg(CalleeAddressLow);
  for (auto R : Args)
    BuildMI(*MBB, Terminator, Terminator->getDebugLoc(),
            TII->get(Next32::WRITER), R.DestReg)
        .addReg(R.DestReg)
        .addReg(R.SrcReg)
        .addImm(R.Size);
  if (!HasSuccessor)
    return;

  MachineBasicBlock *Successor = Terminator->getOperand(LastValue + 2).getMBB();
  CreateMOVWithRelocation(Successor->getSymbol(), Next32::CALL_RET_BB);

  Args.clear();
  getInputFeedersRegs(Successor, Next32::CALL_RET_BB, Args);
  ChainInfo CI = {
      Next32::CALL_RET_BB,
      Condition,
      CondReg,
      false,
      static_cast<unsigned int>(Terminator->getOperand(LastValue + 1).getImm()),
      100};
  AddWriterChain(CI, Args);
}

void Next32WriterChainPass::getInputFeedersRegs(
    MachineBasicBlock *TargetMBB, unsigned int RetFid,
    SmallVectorImpl<WriterInfo> &FeedersRegs) const {
  for (auto &MI : *TargetMBB) {
    if (MI.getOpcode() == Next32::SYM_INSTR)
      break;
    if (MI.getOpcode() != Next32::FEEDER && MI.getOpcode() != Next32::FEEDERP) {
      continue;
    }
    FeedersRegs.push_back(
        {RetFid, MI.getOperand(1).getReg(),
         llvm::Next32Constants::InstructionSize::InstructionSize32});
  }
}

void Next32WriterChainPass::CreateMOVWithRelocation(const Twine &RelocSymName,
                                                    unsigned int Reg) {
  MCSymbol *RelocSym = MF->getContext().getOrCreateSymbol(RelocSymName);
  CreateMOVWithRelocation(RelocSym, Reg);
}

void Next32WriterChainPass::CreateMOVWithRelocation(MCSymbol *RelocSym,
                                                    unsigned int Reg) {
  BuildMI(*MBB, Terminator, Terminator->getDebugLoc(), TII->get(Next32::MOVL),
          Reg)
      .addSym(RelocSym);
}

unsigned Next32WriterChainPass::GetChainOpcode(bool IsCondBranch,
                                               bool IsPriority) {
  static const unsigned ChainOpcodeTable[2][2] = { //! IsCondBranch
                                                  {//! IsPriority
                                                   Next32::CHAIN,
                                                   // IsPriority
                                                   Next32::CHAINP},
                                                  // IsCondBranch
                                                  {//! IsPriority
                                                   Next32::CHAINc,
                                                   // IsPriority
                                                   Next32::CHAINPc}};
  return ChainOpcodeTable[IsCondBranch][IsPriority];
}

Next32Constants::CondCode
Next32WriterChainPass::getCondFromTerminator(unsigned OpIdx) const {
  return (Next32Constants::CondCode)Terminator->getOperand(OpIdx).getImm();
}
