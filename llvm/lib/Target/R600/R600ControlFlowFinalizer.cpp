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

#define DEBUG_TYPE "r600cf"
#include "llvm/Support/Debug.h"
#include "AMDGPU.h"
#include "R600Defines.h"
#include "R600InstrInfo.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

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
  const AMDGPUSubtarget &ST;

  bool IsTrivialInst(MachineInstr *MI) const {
    switch (MI->getOpcode()) {
    case AMDGPU::KILL:
    case AMDGPU::RETURN:
      return true;
    default:
      return false;
    }
  }

  const MCInstrDesc &getHWInstrDesc(ControlFlowInstruction CFI) const {
    unsigned Opcode = 0;
    bool isEg = (ST.getGeneration() >= AMDGPUSubtarget::EVERGREEN);
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
      if (ST.hasCaymanISA()) {
        Opcode = AMDGPU::CF_END_CM;
        break;
      }
      Opcode = isEg ? AMDGPU::CF_END_EG : AMDGPU::CF_END_R600;
      break;
    }
    assert (Opcode && "No opcode selected");
    return TII->get(Opcode);
  }

  bool isCompatibleWithClause(const MachineInstr *MI,
      std::set<unsigned> &DstRegs) const {
    unsigned DstMI, SrcMI;
    for (MachineInstr::const_mop_iterator I = MI->operands_begin(),
        E = MI->operands_end(); I != E; ++I) {
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
    bool IsTex = TII->usesTextureCache(ClauseHead);
    std::set<unsigned> DstRegs;
    for (MachineBasicBlock::iterator E = MBB.end(); I != E; ++I) {
      if (IsTrivialInst(I))
        continue;
      if (AluInstCount >= MaxFetchInst)
        break;
      if ((IsTex && !TII->usesTextureCache(I)) ||
          (!IsTex && !TII->usesVertexCache(I)))
        break;
      if (!isCompatibleWithClause(I, DstRegs))
        break;
      AluInstCount ++;
      ClauseContent.push_back(I);
    }
    MachineInstr *MIb = BuildMI(MBB, ClauseHead, MBB.findDebugLoc(ClauseHead),
        getHWInstrDesc(IsTex?CF_TC:CF_VC))
        .addImm(0) // ADDR
        .addImm(AluInstCount - 1); // COUNT
    return ClauseFile(MIb, ClauseContent);
  }

  void getLiteral(MachineInstr *MI, std::vector<int64_t> &Lits) const {
    static const unsigned LiteralRegs[] = {
      AMDGPU::ALU_LITERAL_X,
      AMDGPU::ALU_LITERAL_Y,
      AMDGPU::ALU_LITERAL_Z,
      AMDGPU::ALU_LITERAL_W
    };
    const SmallVector<std::pair<MachineOperand *, int64_t>, 3 > Srcs =
        TII->getSrcs(MI);
    for (unsigned i = 0, e = Srcs.size(); i < e; ++i) {
      if (Srcs[i].first->getReg() != AMDGPU::ALU_LITERAL_X)
        continue;
      int64_t Imm = Srcs[i].second;
      std::vector<int64_t>::iterator It =
          std::find(Lits.begin(), Lits.end(), Imm);
      if (It != Lits.end()) {
        unsigned Index = It - Lits.begin();
        Srcs[i].first->setReg(LiteralRegs[Index]);
      } else {
        assert(Lits.size() < 4 && "Too many literals in Instruction Group");
        Srcs[i].first->setReg(LiteralRegs[Lits.size()]);
        Lits.push_back(Imm);
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
    MachineBasicBlock::iterator ClauseHead = I;
    std::vector<MachineInstr *> ClauseContent;
    I++;
    for (MachineBasicBlock::instr_iterator E = MBB.instr_end(); I != E;) {
      if (IsTrivialInst(I)) {
        ++I;
        continue;
      }
      if (!I->isBundle() && !TII->isALUInstr(I->getOpcode()))
        break;
      std::vector<int64_t> Literals;
      if (I->isBundle()) {
        MachineInstr *DeleteMI = I;
        MachineBasicBlock::instr_iterator BI = I.getInstrIterator();
        while (++BI != E && BI->isBundledWithPred()) {
          BI->unbundleFromPred();
          for (unsigned i = 0, e = BI->getNumOperands(); i != e; ++i) {
            MachineOperand &MO = BI->getOperand(i);
            if (MO.isReg() && MO.isInternalRead())
              MO.setIsInternalRead(false);
          }
          getLiteral(BI, Literals);
          ClauseContent.push_back(BI);
        }
        I = BI;
        DeleteMI->eraseFromParent();
      } else {
        getLiteral(I, Literals);
        ClauseContent.push_back(I);
        I++;
      }
      for (unsigned i = 0, e = Literals.size(); i < e; i+=2) {
        unsigned literal0 = Literals[i];
        unsigned literal2 = (i + 1 < e)?Literals[i + 1]:0;
        MachineInstr *MILit = BuildMI(MBB, I, I->getDebugLoc(),
            TII->get(AMDGPU::LITERALS))
            .addImm(literal0)
            .addImm(literal2);
        ClauseContent.push_back(MILit);
      }
    }
    assert(ClauseContent.size() < 128 && "ALU clause is too big");
    ClauseHead->getOperand(7).setImm(ClauseContent.size() - 1);
    return ClauseFile(ClauseHead, ClauseContent);
  }

  void
  EmitFetchClause(MachineBasicBlock::iterator InsertPos, ClauseFile &Clause,
      unsigned &CfCount) {
    CounterPropagateAddr(Clause.first, CfCount);
    MachineBasicBlock *BB = Clause.first->getParent();
    BuildMI(BB, InsertPos->getDebugLoc(), TII->get(AMDGPU::FETCH_CLAUSE))
        .addImm(CfCount);
    for (unsigned i = 0, e = Clause.second.size(); i < e; ++i) {
      BB->splice(InsertPos, BB, Clause.second[i]);
    }
    CfCount += 2 * Clause.second.size();
  }

  void
  EmitALUClause(MachineBasicBlock::iterator InsertPos, ClauseFile &Clause,
      unsigned &CfCount) {
    Clause.first->getOperand(0).setImm(0);
    CounterPropagateAddr(Clause.first, CfCount);
    MachineBasicBlock *BB = Clause.first->getParent();
    BuildMI(BB, InsertPos->getDebugLoc(), TII->get(AMDGPU::ALU_CLAUSE))
        .addImm(CfCount);
    for (unsigned i = 0, e = Clause.second.size(); i < e; ++i) {
      BB->splice(InsertPos, BB, Clause.second[i]);
    }
    CfCount += Clause.second.size();
  }

  void CounterPropagateAddr(MachineInstr *MI, unsigned Addr) const {
    MI->getOperand(0).setImm(Addr + MI->getOperand(0).getImm());
  }
  void CounterPropagateAddr(std::set<MachineInstr *> MIs, unsigned Addr)
      const {
    for (std::set<MachineInstr *>::iterator It = MIs.begin(), E = MIs.end();
        It != E; ++It) {
      MachineInstr *MI = *It;
      CounterPropagateAddr(MI, Addr);
    }
  }

  unsigned getHWStackSize(unsigned StackSubEntry, bool hasPush) const {
    switch (ST.getGeneration()) {
    case AMDGPUSubtarget::R600:
    case AMDGPUSubtarget::R700:
      if (hasPush)
        StackSubEntry += 2;
      break;
    case AMDGPUSubtarget::EVERGREEN:
      if (hasPush)
        StackSubEntry ++;
    case AMDGPUSubtarget::NORTHERN_ISLANDS:
      StackSubEntry += 2;
      break;
    default: llvm_unreachable("Not a VLIW4/VLIW5 GPU");
    }
    return (StackSubEntry + 3)/4; // Need ceil value of StackSubEntry/4
  }

public:
  R600ControlFlowFinalizer(TargetMachine &tm) : MachineFunctionPass(ID),
    TII (0), TRI(0),
    ST(tm.getSubtarget<AMDGPUSubtarget>()) {
      const AMDGPUSubtarget &ST = tm.getSubtarget<AMDGPUSubtarget>();
      MaxFetchInst = ST.getTexVTXClauseSize();
  }

  virtual bool runOnMachineFunction(MachineFunction &MF) {
    TII=static_cast<const R600InstrInfo *>(MF.getTarget().getInstrInfo());
    TRI=static_cast<const R600RegisterInfo *>(MF.getTarget().getRegisterInfo());

    unsigned MaxStack = 0;
    unsigned CurrentStack = 0;
    bool HasPush = false;
    for (MachineFunction::iterator MB = MF.begin(), ME = MF.end(); MB != ME;
        ++MB) {
      MachineBasicBlock &MBB = *MB;
      unsigned CfCount = 0;
      std::vector<std::pair<unsigned, std::set<MachineInstr *> > > LoopStack;
      std::vector<MachineInstr * > IfThenElseStack;
      R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
      if (MFI->ShaderType == 1) {
        BuildMI(MBB, MBB.begin(), MBB.findDebugLoc(MBB.begin()),
            getHWInstrDesc(CF_CALL_FS));
        CfCount++;
        MaxStack = 1;
      }
      std::vector<ClauseFile> FetchClauses, AluClauses;
      std::vector<MachineInstr *> LastAlu(1);
      std::vector<MachineInstr *> ToPopAfter;
      
      for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
          I != E;) {
        if (TII->usesTextureCache(I) || TII->usesVertexCache(I)) {
          DEBUG(dbgs() << CfCount << ":"; I->dump(););
          FetchClauses.push_back(MakeFetchClause(MBB, I));
          CfCount++;
          LastAlu.back() = 0;
          continue;
        }

        MachineBasicBlock::iterator MI = I;
        if (MI->getOpcode() != AMDGPU::ENDIF)
          LastAlu.back() = 0;
        if (MI->getOpcode() == AMDGPU::CF_ALU)
          LastAlu.back() = MI;
        I++;
        switch (MI->getOpcode()) {
        case AMDGPU::CF_ALU_PUSH_BEFORE:
          CurrentStack++;
          MaxStack = std::max(MaxStack, CurrentStack);
          HasPush = true;
        case AMDGPU::CF_ALU:
          I = MI;
          AluClauses.push_back(MakeALUClause(MBB, I));
          DEBUG(dbgs() << CfCount << ":"; MI->dump(););
          CfCount++;
          break;
        case AMDGPU::WHILELOOP: {
          CurrentStack+=4;
          MaxStack = std::max(MaxStack, CurrentStack);
          MachineInstr *MIb = BuildMI(MBB, MI, MBB.findDebugLoc(MI),
              getHWInstrDesc(CF_WHILE_LOOP))
              .addImm(1);
          std::pair<unsigned, std::set<MachineInstr *> > Pair(CfCount,
              std::set<MachineInstr *>());
          Pair.second.insert(MIb);
          LoopStack.push_back(Pair);
          MI->eraseFromParent();
          CfCount++;
          break;
        }
        case AMDGPU::ENDLOOP: {
          CurrentStack-=4;
          std::pair<unsigned, std::set<MachineInstr *> > Pair =
              LoopStack.back();
          LoopStack.pop_back();
          CounterPropagateAddr(Pair.second, CfCount);
          BuildMI(MBB, MI, MBB.findDebugLoc(MI), getHWInstrDesc(CF_END_LOOP))
              .addImm(Pair.first + 1);
          MI->eraseFromParent();
          CfCount++;
          break;
        }
        case AMDGPU::IF_PREDICATE_SET: {
          LastAlu.push_back(0);
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
          CounterPropagateAddr(JumpInst, CfCount);
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
          CurrentStack--;
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
          CounterPropagateAddr(IfOrElseInst, CfCount);
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
          BuildMI(MBB, MI, MBB.findDebugLoc(MI), getHWInstrDesc(CF_END));
          CfCount++;
          MI->eraseFromParent();
          if (CfCount % 2) {
            BuildMI(MBB, I, MBB.findDebugLoc(MI), TII->get(AMDGPU::PAD));
            CfCount++;
          }
          for (unsigned i = 0, e = FetchClauses.size(); i < e; i++)
            EmitFetchClause(I, FetchClauses[i], CfCount);
          for (unsigned i = 0, e = AluClauses.size(); i < e; i++)
            EmitALUClause(I, AluClauses[i], CfCount);
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
      MFI->StackSize = getHWStackSize(MaxStack, HasPush);
    }

    return false;
  }

  const char *getPassName() const {
    return "R600 Control Flow Finalizer Pass";
  }
};

char R600ControlFlowFinalizer::ID = 0;

} // end anonymous namespace


llvm::FunctionPass *llvm::createR600ControlFlowFinalizer(TargetMachine &TM) {
  return new R600ControlFlowFinalizer(TM);
}
