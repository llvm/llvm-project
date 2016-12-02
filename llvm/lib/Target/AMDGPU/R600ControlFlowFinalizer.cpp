//===-- R600ControlFlowFinalizer.cpp - Finalize Control Flow Inst----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass compute turns all control flow pseudo instructions into native one
/// computing their address on the fly ; it also sets STACK_SIZE info.
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "R600Defines.h"
#include "R600InstrInfo.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "r600cf"

namespace {

struct CFStack {

  enum StackItem {
    ENTRY = 0,
    SUB_ENTRY = 1,
    FIRST_NON_WQM_PUSH = 2,
    FIRST_NON_WQM_PUSH_W_FULL_ENTRY = 3
  };

  const R600Subtarget *ST;
  std::vector<StackItem> BranchStack;
  std::vector<StackItem> LoopStack;
  unsigned MaxStackSize;
  unsigned CurrentEntries;
  unsigned CurrentSubEntries;

  CFStack(const R600Subtarget *st, CallingConv::ID cc) : ST(st),
      // We need to reserve a stack entry for CALL_FS in vertex shaders.
      MaxStackSize(cc == CallingConv::AMDGPU_VS ? 1 : 0),
      CurrentEntries(0), CurrentSubEntries(0) { }

  unsigned getLoopDepth();
  bool branchStackContains(CFStack::StackItem);
  bool requiresWorkAroundForInst(unsigned Opcode);
  unsigned getSubEntrySize(CFStack::StackItem Item);
  void updateMaxStackSize();
  void pushBranch(unsigned Opcode, bool isWQM = false);
  void pushLoop();
  void popBranch();
  void popLoop();
};

unsigned CFStack::getLoopDepth() {
  return LoopStack.size();
}

bool CFStack::branchStackContains(CFStack::StackItem Item) {
  for (std::vector<CFStack::StackItem>::const_iterator I = BranchStack.begin(),
       E = BranchStack.end(); I != E; ++I) {
    if (*I == Item)
      return true;
  }
  return false;
}

bool CFStack::requiresWorkAroundForInst(unsigned Opcode) {
  if (Opcode == AMDGPU::CF_ALU_PUSH_BEFORE && ST->hasCaymanISA() &&
      getLoopDepth() > 1)
    return true;

  if (!ST->hasCFAluBug())
    return false;

  switch(Opcode) {
  default: return false;
  case AMDGPU::CF_ALU_PUSH_BEFORE:
  case AMDGPU::CF_ALU_ELSE_AFTER:
  case AMDGPU::CF_ALU_BREAK:
  case AMDGPU::CF_ALU_CONTINUE:
    if (CurrentSubEntries == 0)
      return false;
    if (ST->getWavefrontSize() == 64) {
      // We are being conservative here.  We only require this work-around if
      // CurrentSubEntries > 3 &&
      // (CurrentSubEntries % 4 == 3 || CurrentSubEntries % 4 == 0)
      //
      // We have to be conservative, because we don't know for certain that
      // our stack allocation algorithm for Evergreen/NI is correct.  Applying this
      // work-around when CurrentSubEntries > 3 allows us to over-allocate stack
      // resources without any problems.
      return CurrentSubEntries > 3;
    } else {
      assert(ST->getWavefrontSize() == 32);
      // We are being conservative here.  We only require the work-around if
      // CurrentSubEntries > 7 &&
      // (CurrentSubEntries % 8 == 7 || CurrentSubEntries % 8 == 0)
      // See the comment on the wavefront size == 64 case for why we are
      // being conservative.
      return CurrentSubEntries > 7;
    }
  }
}

unsigned CFStack::getSubEntrySize(CFStack::StackItem Item) {
  switch(Item) {
  default:
    return 0;
  case CFStack::FIRST_NON_WQM_PUSH:
  assert(!ST->hasCaymanISA());
  if (ST->getGeneration() <= R600Subtarget::R700) {
    // +1 For the push operation.
    // +2 Extra space required.
    return 3;
  } else {
    // Some documentation says that this is not necessary on Evergreen,
    // but experimentation has show that we need to allocate 1 extra
    // sub-entry for the first non-WQM push.
    // +1 For the push operation.
    // +1 Extra space required.
    return 2;
  }
  case CFStack::FIRST_NON_WQM_PUSH_W_FULL_ENTRY:
    assert(ST->getGeneration() >= R600Subtarget::EVERGREEN);
    // +1 For the push operation.
    // +1 Extra space required.
    return 2;
  case CFStack::SUB_ENTRY:
    return 1;
  }
}

void CFStack::updateMaxStackSize() {
  unsigned CurrentStackSize =
      CurrentEntries + (alignTo(CurrentSubEntries, 4) / 4);
  MaxStackSize = std::max(CurrentStackSize, MaxStackSize);
}

void CFStack::pushBranch(unsigned Opcode, bool isWQM) {
  CFStack::StackItem Item = CFStack::ENTRY;
  switch(Opcode) {
  case AMDGPU::CF_PUSH_EG:
  case AMDGPU::CF_ALU_PUSH_BEFORE:
    if (!isWQM) {
      if (!ST->hasCaymanISA() &&
          !branchStackContains(CFStack::FIRST_NON_WQM_PUSH))
        Item = CFStack::FIRST_NON_WQM_PUSH;  // May not be required on Evergreen/NI
                                             // See comment in
                                             // CFStack::getSubEntrySize()
      else if (CurrentEntries > 0 &&
               ST->getGeneration() > R600Subtarget::EVERGREEN &&
               !ST->hasCaymanISA() &&
               !branchStackContains(CFStack::FIRST_NON_WQM_PUSH_W_FULL_ENTRY))
        Item = CFStack::FIRST_NON_WQM_PUSH_W_FULL_ENTRY;
      else
        Item = CFStack::SUB_ENTRY;
    } else
      Item = CFStack::ENTRY;
    break;
  }
  BranchStack.push_back(Item);
  if (Item == CFStack::ENTRY)
    CurrentEntries++;
  else
    CurrentSubEntries += getSubEntrySize(Item);
  updateMaxStackSize();
}

void CFStack::pushLoop() {
  LoopStack.push_back(CFStack::ENTRY);
  CurrentEntries++;
  updateMaxStackSize();
}

void CFStack::popBranch() {
  CFStack::StackItem Top = BranchStack.back();
  if (Top == CFStack::ENTRY)
    CurrentEntries--;
  else
    CurrentSubEntries-= getSubEntrySize(Top);
  BranchStack.pop_back();
}

void CFStack::popLoop() {
  CurrentEntries--;
  LoopStack.pop_back();
}

class R600ControlFlowFinalizer : public MachineFunctionPass {

private:
  typedef std::pair<MachineInstr *, std::vector<MachineInstr *> > ClauseFile;

  enum ControlFlowInstruction {
    CF_TC,
    CF_VC,
    CF_CALL_FS,
    CF_WHILE_LOOP,
    CF_END_LOOP,
    CF_LOOP_BREAK,
    CF_LOOP_CONTINUE,
    CF_JUMP,
    CF_ELSE,
    CF_POP,
    CF_END
  };

  static char ID;
  const R600InstrInfo *TII;
  const R600RegisterInfo *TRI;
  unsigned MaxFetchInst;
  const R600Subtarget *ST;

  bool IsTrivialInst(MachineInstr &MI) const {
    switch (MI.getOpcode()) {
    case AMDGPU::KILL:
    case AMDGPU::RETURN:
      return true;
    default:
      return false;
    }
  }

  const MCInstrDesc &getHWInstrDesc(ControlFlowInstruction CFI) const {
    unsigned Opcode = 0;
    bool isEg = (ST->getGeneration() >= R600Subtarget::EVERGREEN);
    switch (CFI) {
    case CF_TC:
      Opcode = isEg ? AMDGPU::CF_TC_EG : AMDGPU::CF_TC_R600;
      break;
    case CF_VC:
      Opcode = isEg ? AMDGPU::CF_VC_EG : AMDGPU::CF_VC_R600;
      break;
    case CF_CALL_FS:
      Opcode = isEg ? AMDGPU::CF_CALL_FS_EG : AMDGPU::CF_CALL_FS_R600;
      break;
    case CF_WHILE_LOOP:
      Opcode = isEg ? AMDGPU::WHILE_LOOP_EG : AMDGPU::WHILE_LOOP_R600;
      break;
    case CF_END_LOOP:
      Opcode = isEg ? AMDGPU::END_LOOP_EG : AMDGPU::END_LOOP_R600;
      break;
    case CF_LOOP_BREAK:
      Opcode = isEg ? AMDGPU::LOOP_BREAK_EG : AMDGPU::LOOP_BREAK_R600;
      break;
    case CF_LOOP_CONTINUE:
      Opcode = isEg ? AMDGPU::CF_CONTINUE_EG : AMDGPU::CF_CONTINUE_R600;
      break;
    case CF_JUMP:
      Opcode = isEg ? AMDGPU::CF_JUMP_EG : AMDGPU::CF_JUMP_R600;
      break;
    case CF_ELSE:
      Opcode = isEg ? AMDGPU::CF_ELSE_EG : AMDGPU::CF_ELSE_R600;
      break;
    case CF_POP:
      Opcode = isEg ? AMDGPU::POP_EG : AMDGPU::POP_R600;
      break;
    case CF_END:
      if (ST->hasCaymanISA()) {
        Opcode = AMDGPU::CF_END_CM;
        break;
      }
      Opcode = isEg ? AMDGPU::CF_END_EG : AMDGPU::CF_END_R600;
      break;
    }
    assert (Opcode && "No opcode selected");
    return TII->get(Opcode);
  }

  bool isCompatibleWithClause(const MachineInstr &MI,
                              std::set<unsigned> &DstRegs) const {
    unsigned DstMI, SrcMI;
    for (MachineInstr::const_mop_iterator I = MI.operands_begin(),
                                          E = MI.operands_end();
         I != E; ++I) {
      const MachineOperand &MO = *I;
      if (!MO.isReg())
        continue;
      if (MO.isDef()) {
        unsigned Reg = MO.getReg();
        if (AMDGPU::R600_Reg128RegClass.contains(Reg))
          DstMI = Reg;
        else
          DstMI = TRI->getMatchingSuperReg(Reg,
              TRI->getSubRegFromChannel(TRI->getHWRegChan(Reg)),
              &AMDGPU::R600_Reg128RegClass);
      }
      if (MO.isUse()) {
        unsigned Reg = MO.getReg();
        if (AMDGPU::R600_Reg128RegClass.contains(Reg))
          SrcMI = Reg;
        else
          SrcMI = TRI->getMatchingSuperReg(Reg,
              TRI->getSubRegFromChannel(TRI->getHWRegChan(Reg)),
              &AMDGPU::R600_Reg128RegClass);
      }
    }
    if ((DstRegs.find(SrcMI) == DstRegs.end())) {
      DstRegs.insert(DstMI);
      return true;
    } else
      return false;
  }

  ClauseFile
  MakeFetchClause(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I)
      const {
    MachineBasicBlock::iterator ClauseHead = I;
    std::vector<MachineInstr *> ClauseContent;
    unsigned AluInstCount = 0;
    bool IsTex = TII->usesTextureCache(*ClauseHead);
    std::set<unsigned> DstRegs;
    for (MachineBasicBlock::iterator E = MBB.end(); I != E; ++I) {
      if (IsTrivialInst(*I))
        continue;
      if (AluInstCount >= MaxFetchInst)
        break;
      if ((IsTex && !TII->usesTextureCache(*I)) ||
          (!IsTex && !TII->usesVertexCache(*I)))
        break;
      if (!isCompatibleWithClause(*I, DstRegs))
        break;
      AluInstCount ++;
      ClauseContent.push_back(&*I);
    }
    MachineInstr *MIb = BuildMI(MBB, ClauseHead, MBB.findDebugLoc(ClauseHead),
        getHWInstrDesc(IsTex?CF_TC:CF_VC))
        .addImm(0) // ADDR
        .addImm(AluInstCount - 1); // COUNT
    return ClauseFile(MIb, std::move(ClauseContent));
  }

  void getLiteral(MachineInstr &MI, std::vector<MachineOperand *> &Lits) const {
    static const unsigned LiteralRegs[] = {
      AMDGPU::ALU_LITERAL_X,
      AMDGPU::ALU_LITERAL_Y,
      AMDGPU::ALU_LITERAL_Z,
      AMDGPU::ALU_LITERAL_W
    };
    const SmallVector<std::pair<MachineOperand *, int64_t>, 3> Srcs =
        TII->getSrcs(MI);
    for (const auto &Src:Srcs) {
      if (Src.first->getReg() != AMDGPU::ALU_LITERAL_X)
        continue;
      int64_t Imm = Src.second;
      std::vector<MachineOperand *>::iterator It =
          find_if(Lits, [&](MachineOperand *val) {
            return val->isImm() && (val->getImm() == Imm);
          });

      // Get corresponding Operand
      MachineOperand &Operand = MI.getOperand(
          TII->getOperandIdx(MI.getOpcode(), AMDGPU::OpName::literal));

      if (It != Lits.end()) {
        // Reuse existing literal reg
        unsigned Index = It - Lits.begin();
        Src.first->setReg(LiteralRegs[Index]);
      } else {
        // Allocate new literal reg
        assert(Lits.size() < 4 && "Too many literals in Instruction Group");
        Src.first->setReg(LiteralRegs[Lits.size()]);
        Lits.push_back(&Operand);
      }
    }
  }

  MachineBasicBlock::iterator insertLiterals(
      MachineBasicBlock::iterator InsertPos,
      const std::vector<unsigned> &Literals) const {
    MachineBasicBlock *MBB = InsertPos->getParent();
    for (unsigned i = 0, e = Literals.size(); i < e; i+=2) {
      unsigned LiteralPair0 = Literals[i];
      unsigned LiteralPair1 = (i + 1 < e)?Literals[i + 1]:0;
      InsertPos = BuildMI(MBB, InsertPos->getDebugLoc(),
          TII->get(AMDGPU::LITERALS))
          .addImm(LiteralPair0)
          .addImm(LiteralPair1);
    }
    return InsertPos;
  }

  ClauseFile
  MakeALUClause(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I)
      const {
    MachineInstr &ClauseHead = *I;
    std::vector<MachineInstr *> ClauseContent;
    I++;
    for (MachineBasicBlock::instr_iterator E = MBB.instr_end(); I != E;) {
      if (IsTrivialInst(*I)) {
        ++I;
        continue;
      }
      if (!I->isBundle() && !TII->isALUInstr(I->getOpcode()))
        break;
      std::vector<MachineOperand *>Literals;
      if (I->isBundle()) {
        MachineInstr &DeleteMI = *I;
        MachineBasicBlock::instr_iterator BI = I.getInstrIterator();
        while (++BI != E && BI->isBundledWithPred()) {
          BI->unbundleFromPred();
          for (MachineOperand &MO : BI->operands()) {
            if (MO.isReg() && MO.isInternalRead())
              MO.setIsInternalRead(false);
          }
          getLiteral(*BI, Literals);
          ClauseContent.push_back(&*BI);
        }
        I = BI;
        DeleteMI.eraseFromParent();
      } else {
        getLiteral(*I, Literals);
        ClauseContent.push_back(&*I);
        I++;
      }
      for (unsigned i = 0, e = Literals.size(); i < e; i += 2) {
        MachineInstrBuilder MILit = BuildMI(MBB, I, I->getDebugLoc(),
            TII->get(AMDGPU::LITERALS));
        if (Literals[i]->isImm()) {
            MILit.addImm(Literals[i]->getImm());
        } else {
            MILit.addGlobalAddress(Literals[i]->getGlobal(),
                                   Literals[i]->getOffset());
        }
        if (i + 1 < e) {
          if (Literals[i + 1]->isImm()) {
            MILit.addImm(Literals[i + 1]->getImm());
          } else {
            MILit.addGlobalAddress(Literals[i + 1]->getGlobal(),
                                   Literals[i + 1]->getOffset());
          }
        } else
          MILit.addImm(0);
        ClauseContent.push_back(MILit);
      }
    }
    assert(ClauseContent.size() < 128 && "ALU clause is too big");
    ClauseHead.getOperand(7).setImm(ClauseContent.size() - 1);
    return ClauseFile(&ClauseHead, std::move(ClauseContent));
  }

  void EmitFetchClause(MachineBasicBlock::iterator InsertPos,
                       const DebugLoc &DL, ClauseFile &Clause,
                       unsigned &CfCount) {
    CounterPropagateAddr(*Clause.first, CfCount);
    MachineBasicBlock *BB = Clause.first->getParent();
    BuildMI(BB, DL, TII->get(AMDGPU::FETCH_CLAUSE)).addImm(CfCount);
    for (unsigned i = 0, e = Clause.second.size(); i < e; ++i) {
      BB->splice(InsertPos, BB, Clause.second[i]);
    }
    CfCount += 2 * Clause.second.size();
  }

  void EmitALUClause(MachineBasicBlock::iterator InsertPos, const DebugLoc &DL,
                     ClauseFile &Clause, unsigned &CfCount) {
    Clause.first->getOperand(0).setImm(0);
    CounterPropagateAddr(*Clause.first, CfCount);
    MachineBasicBlock *BB = Clause.first->getParent();
    BuildMI(BB, DL, TII->get(AMDGPU::ALU_CLAUSE)).addImm(CfCount);
    for (unsigned i = 0, e = Clause.second.size(); i < e; ++i) {
      BB->splice(InsertPos, BB, Clause.second[i]);
    }
    CfCount += Clause.second.size();
  }

  void CounterPropagateAddr(MachineInstr &MI, unsigned Addr) const {
    MI.getOperand(0).setImm(Addr + MI.getOperand(0).getImm());
  }
  void CounterPropagateAddr(const std::set<MachineInstr *> &MIs,
                            unsigned Addr) const {
    for (MachineInstr *MI : MIs) {
      CounterPropagateAddr(*MI, Addr);
    }
  }

public:
  R600ControlFlowFinalizer(TargetMachine &tm)
      : MachineFunctionPass(ID), TII(nullptr), TRI(nullptr), ST(nullptr) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    ST = &MF.getSubtarget<R600Subtarget>();
    MaxFetchInst = ST->getTexVTXClauseSize();
    TII = ST->getInstrInfo();
    TRI = ST->getRegisterInfo();

    R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();

    CFStack CFStack(ST, MF.getFunction()->getCallingConv());
    for (MachineFunction::iterator MB = MF.begin(), ME = MF.end(); MB != ME;
        ++MB) {
      MachineBasicBlock &MBB = *MB;
      unsigned CfCount = 0;
      std::vector<std::pair<unsigned, std::set<MachineInstr *> > > LoopStack;
      std::vector<MachineInstr * > IfThenElseStack;
      if (MF.getFunction()->getCallingConv() == CallingConv::AMDGPU_VS) {
        BuildMI(MBB, MBB.begin(), MBB.findDebugLoc(MBB.begin()),
            getHWInstrDesc(CF_CALL_FS));
        CfCount++;
      }
      std::vector<ClauseFile> FetchClauses, AluClauses;
      std::vector<MachineInstr *> LastAlu(1);
      std::vector<MachineInstr *> ToPopAfter;

      for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
          I != E;) {
        if (TII->usesTextureCache(*I) || TII->usesVertexCache(*I)) {
          DEBUG(dbgs() << CfCount << ":"; I->dump(););
          FetchClauses.push_back(MakeFetchClause(MBB, I));
          CfCount++;
          LastAlu.back() = nullptr;
          continue;
        }

        MachineBasicBlock::iterator MI = I;
        if (MI->getOpcode() != AMDGPU::ENDIF)
          LastAlu.back() = nullptr;
        if (MI->getOpcode() == AMDGPU::CF_ALU)
          LastAlu.back() = &*MI;
        I++;
        bool RequiresWorkAround =
            CFStack.requiresWorkAroundForInst(MI->getOpcode());
        switch (MI->getOpcode()) {
        case AMDGPU::CF_ALU_PUSH_BEFORE:
          if (RequiresWorkAround) {
            DEBUG(dbgs() << "Applying bug work-around for ALU_PUSH_BEFORE\n");
            BuildMI(MBB, MI, MBB.findDebugLoc(MI), TII->get(AMDGPU::CF_PUSH_EG))
                .addImm(CfCount + 1)
                .addImm(1);
            MI->setDesc(TII->get(AMDGPU::CF_ALU));
            CfCount++;
            CFStack.pushBranch(AMDGPU::CF_PUSH_EG);
          } else
            CFStack.pushBranch(AMDGPU::CF_ALU_PUSH_BEFORE);

        case AMDGPU::CF_ALU:
          I = MI;
          AluClauses.push_back(MakeALUClause(MBB, I));
          DEBUG(dbgs() << CfCount << ":"; MI->dump(););
          CfCount++;
          break;
        case AMDGPU::WHILELOOP: {
          CFStack.pushLoop();
          MachineInstr *MIb = BuildMI(MBB, MI, MBB.findDebugLoc(MI),
              getHWInstrDesc(CF_WHILE_LOOP))
              .addImm(1);
          std::pair<unsigned, std::set<MachineInstr *> > Pair(CfCount,
              std::set<MachineInstr *>());
          Pair.second.insert(MIb);
          LoopStack.push_back(std::move(Pair));
          MI->eraseFromParent();
          CfCount++;
          break;
        }
        case AMDGPU::ENDLOOP: {
          CFStack.popLoop();
          std::pair<unsigned, std::set<MachineInstr *> > Pair =
              std::move(LoopStack.back());
          LoopStack.pop_back();
          CounterPropagateAddr(Pair.second, CfCount);
          BuildMI(MBB, MI, MBB.findDebugLoc(MI), getHWInstrDesc(CF_END_LOOP))
              .addImm(Pair.first + 1);
          MI->eraseFromParent();
          CfCount++;
          break;
        }
        case AMDGPU::IF_PREDICATE_SET: {
          LastAlu.push_back(nullptr);
          MachineInstr *MIb = BuildMI(MBB, MI, MBB.findDebugLoc(MI),
              getHWInstrDesc(CF_JUMP))
              .addImm(0)
              .addImm(0);
          IfThenElseStack.push_back(MIb);
          DEBUG(dbgs() << CfCount << ":"; MIb->dump(););
          MI->eraseFromParent();
          CfCount++;
          break;
        }
        case AMDGPU::ELSE: {
          MachineInstr * JumpInst = IfThenElseStack.back();
          IfThenElseStack.pop_back();
          CounterPropagateAddr(*JumpInst, CfCount);
          MachineInstr *MIb = BuildMI(MBB, MI, MBB.findDebugLoc(MI),
              getHWInstrDesc(CF_ELSE))
              .addImm(0)
              .addImm(0);
          DEBUG(dbgs() << CfCount << ":"; MIb->dump(););
          IfThenElseStack.push_back(MIb);
          MI->eraseFromParent();
          CfCount++;
          break;
        }
        case AMDGPU::ENDIF: {
          CFStack.popBranch();
          if (LastAlu.back()) {
            ToPopAfter.push_back(LastAlu.back());
          } else {
            MachineInstr *MIb = BuildMI(MBB, MI, MBB.findDebugLoc(MI),
                getHWInstrDesc(CF_POP))
                .addImm(CfCount + 1)
                .addImm(1);
            (void)MIb;
            DEBUG(dbgs() << CfCount << ":"; MIb->dump(););
            CfCount++;
          }

          MachineInstr *IfOrElseInst = IfThenElseStack.back();
          IfThenElseStack.pop_back();
          CounterPropagateAddr(*IfOrElseInst, CfCount);
          IfOrElseInst->getOperand(1).setImm(1);
          LastAlu.pop_back();
          MI->eraseFromParent();
          break;
        }
        case AMDGPU::BREAK: {
          CfCount ++;
          MachineInstr *MIb = BuildMI(MBB, MI, MBB.findDebugLoc(MI),
              getHWInstrDesc(CF_LOOP_BREAK))
              .addImm(0);
          LoopStack.back().second.insert(MIb);
          MI->eraseFromParent();
          break;
        }
        case AMDGPU::CONTINUE: {
          MachineInstr *MIb = BuildMI(MBB, MI, MBB.findDebugLoc(MI),
              getHWInstrDesc(CF_LOOP_CONTINUE))
              .addImm(0);
          LoopStack.back().second.insert(MIb);
          MI->eraseFromParent();
          CfCount++;
          break;
        }
        case AMDGPU::RETURN: {
          DebugLoc DL = MBB.findDebugLoc(MI);
          BuildMI(MBB, MI, DL, getHWInstrDesc(CF_END));
          CfCount++;
          if (CfCount % 2) {
            BuildMI(MBB, I, DL, TII->get(AMDGPU::PAD));
            CfCount++;
          }
          MI->eraseFromParent();
          for (unsigned i = 0, e = FetchClauses.size(); i < e; i++)
            EmitFetchClause(I, DL, FetchClauses[i], CfCount);
          for (unsigned i = 0, e = AluClauses.size(); i < e; i++)
            EmitALUClause(I, DL, AluClauses[i], CfCount);
          break;
        }
        default:
          if (TII->isExport(MI->getOpcode())) {
            DEBUG(dbgs() << CfCount << ":"; MI->dump(););
            CfCount++;
          }
          break;
        }
      }
      for (unsigned i = 0, e = ToPopAfter.size(); i < e; ++i) {
        MachineInstr *Alu = ToPopAfter[i];
        BuildMI(MBB, Alu, MBB.findDebugLoc((MachineBasicBlock::iterator)Alu),
            TII->get(AMDGPU::CF_ALU_POP_AFTER))
            .addImm(Alu->getOperand(0).getImm())
            .addImm(Alu->getOperand(1).getImm())
            .addImm(Alu->getOperand(2).getImm())
            .addImm(Alu->getOperand(3).getImm())
            .addImm(Alu->getOperand(4).getImm())
            .addImm(Alu->getOperand(5).getImm())
            .addImm(Alu->getOperand(6).getImm())
            .addImm(Alu->getOperand(7).getImm())
            .addImm(Alu->getOperand(8).getImm());
        Alu->eraseFromParent();
      }
      MFI->CFStackSize = CFStack.MaxStackSize;
    }

    return false;
  }

  const char *getPassName() const override {
    return "R600 Control Flow Finalizer Pass";
  }
};

char R600ControlFlowFinalizer::ID = 0;

} // end anonymous namespace


llvm::FunctionPass *llvm::createR600ControlFlowFinalizer(TargetMachine &TM) {
  return new R600ControlFlowFinalizer(TM);
}
