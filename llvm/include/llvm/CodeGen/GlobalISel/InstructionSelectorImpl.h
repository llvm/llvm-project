//==-- llvm/CodeGen/GlobalISel/InstructionSelectorImpl.h ---------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file declares the API for the instruction selector.
/// This class is responsible for selecting machine instructions.
/// It's implemented by the target. It's used by the InstructionSelect pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTORIMPL_H
#define LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTORIMPL_H

namespace llvm {
template <class TgtInstructionSelector, class PredicateBitset,
          class ComplexMatcherMemFn>
bool InstructionSelector::executeMatchTable(
    TgtInstructionSelector &ISel, NewMIVector &OutMIs, MatcherState &State,
    const MatcherInfoTy<PredicateBitset, ComplexMatcherMemFn> &MatcherInfo,
    const int64_t *MatchTable, const TargetInstrInfo &TII,
    MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
    const RegisterBankInfo &RBI,
    const PredicateBitset &AvailableFeatures) const {
  const int64_t *Command = MatchTable;
  while (true) {
    switch (*Command++) {
    case GIM_RecordInsn: {
      int64_t NewInsnID = *Command++;
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;

      // As an optimisation we require that MIs[0] is always the root. Refuse
      // any attempt to modify it.
      assert(NewInsnID != 0 && "Refusing to modify MIs[0]");
      (void)NewInsnID;

      MachineOperand &MO = State.MIs[InsnID]->getOperand(OpIdx);
      if (!MO.isReg()) {
        DEBUG(dbgs() << "Rejected (not a register)\n");
        return false;
      }
      if (TRI.isPhysicalRegister(MO.getReg())) {
        DEBUG(dbgs() << "Rejected (is a physical register)\n");
        return false;
      }

      assert((size_t)NewInsnID == State.MIs.size() &&
             "Expected to store MIs in order");
      State.MIs.push_back(MRI.getVRegDef(MO.getReg()));
      DEBUG(dbgs() << "MIs[" << NewInsnID << "] = GIM_RecordInsn(" << InsnID
                   << ", " << OpIdx << ")\n");
      break;
    }

    case GIM_CheckFeatures: {
      int64_t ExpectedBitsetID = *Command++;
      DEBUG(dbgs() << "GIM_CheckFeatures(ExpectedBitsetID=" << ExpectedBitsetID
                   << ")\n");
      if ((AvailableFeatures & MatcherInfo.FeatureBitsets[ExpectedBitsetID]) !=
          MatcherInfo.FeatureBitsets[ExpectedBitsetID]) {
        DEBUG(dbgs() << "Rejected\n");
        return false;
      }
      break;
    }

    case GIM_CheckOpcode: {
      int64_t InsnID = *Command++;
      int64_t Expected = *Command++;

      unsigned Opcode = State.MIs[InsnID]->getOpcode();
      DEBUG(dbgs() << "GIM_CheckOpcode(MIs[" << InsnID << "], ExpectedOpcode="
                   << Expected << ") // Got=" << Opcode << "\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      if (Opcode != Expected)
        return false;
      break;
    }
    case GIM_CheckNumOperands: {
      int64_t InsnID = *Command++;
      int64_t Expected = *Command++;
      DEBUG(dbgs() << "GIM_CheckNumOperands(MIs[" << InsnID
                   << "], Expected=" << Expected << ")\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      if (State.MIs[InsnID]->getNumOperands() != Expected)
        return false;
      break;
    }

    case GIM_CheckType: {
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t TypeID = *Command++;
      DEBUG(dbgs() << "GIM_CheckType(MIs[" << InsnID << "]->getOperand("
                   << OpIdx << "), TypeID=" << TypeID << ")\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      if (MRI.getType(State.MIs[InsnID]->getOperand(OpIdx).getReg()) !=
          MatcherInfo.TypeObjects[TypeID])
        return false;
      break;
    }
    case GIM_CheckRegBankForClass: {
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t RCEnum = *Command++;
      DEBUG(dbgs() << "GIM_CheckRegBankForClass(MIs[" << InsnID
                   << "]->getOperand(" << OpIdx << "), RCEnum=" << RCEnum
                   << ")\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      if (&RBI.getRegBankFromRegClass(*TRI.getRegClass(RCEnum)) !=
          RBI.getRegBank(State.MIs[InsnID]->getOperand(OpIdx).getReg(), MRI, TRI))
        return false;
      break;
    }
    case GIM_CheckComplexPattern: {
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t RendererID = *Command++;
      int64_t ComplexPredicateID = *Command++;
      DEBUG(dbgs() << "State.Renderers[" << RendererID
                   << "] = GIM_CheckComplexPattern(MIs[" << InsnID
                   << "]->getOperand(" << OpIdx
                   << "), ComplexPredicateID=" << ComplexPredicateID << ")\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      // FIXME: Use std::invoke() when it's available.
      if (!(State.Renderers[RendererID] =
                (ISel.*MatcherInfo.ComplexPredicates[ComplexPredicateID])(
                    State.MIs[InsnID]->getOperand(OpIdx))))
        return false;
      break;
    }
    case GIM_CheckConstantInt: {
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t Value = *Command++;
      DEBUG(dbgs() << "GIM_CheckConstantInt(MIs[" << InsnID << "]->getOperand("
                   << OpIdx << "), Value=" << Value << ")\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      if (!isOperandImmEqual(State.MIs[InsnID]->getOperand(OpIdx), Value, MRI))
        return false;
      break;
    }
    case GIM_CheckLiteralInt: {
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t Value = *Command++;
      DEBUG(dbgs() << "GIM_CheckLiteralInt(MIs[" << InsnID << "]->getOperand(" << OpIdx
                   << "), Value=" << Value << ")\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      MachineOperand &OM = State.MIs[InsnID]->getOperand(OpIdx);
      if (!OM.isCImm() || !OM.getCImm()->equalsInt(Value))
        return false;
      break;
    }
    case GIM_CheckIntrinsicID: {
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t Value = *Command++;
      DEBUG(dbgs() << "GIM_CheckIntrinsicID(MIs[" << InsnID << "]->getOperand(" << OpIdx
                   << "), Value=" << Value << ")\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      MachineOperand &OM = State.MIs[InsnID]->getOperand(OpIdx);
      if (!OM.isIntrinsicID() || OM.getIntrinsicID() != Value)
        return false;
      break;
    }
    case GIM_CheckIsMBB: {
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;
      DEBUG(dbgs() << "GIM_CheckIsMBB(MIs[" << InsnID << "]->getOperand("
                   << OpIdx << "))\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      if (!State.MIs[InsnID]->getOperand(OpIdx).isMBB())
        return false;
      break;
    }

    case GIM_CheckIsSafeToFold: {
      int64_t InsnID = *Command++;
      DEBUG(dbgs() << "GIM_CheckIsSafeToFold(MIs[" << InsnID << "])\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      if (!isObviouslySafeToFold(*State.MIs[InsnID]))
        return false;
      break;
    }

    case GIR_MutateOpcode: {
      int64_t OldInsnID = *Command++;
      int64_t NewInsnID = *Command++;
      int64_t NewOpcode = *Command++;
      assert((size_t)NewInsnID == OutMIs.size() &&
             "Expected to store MIs in order");
      OutMIs.push_back(
          MachineInstrBuilder(*State.MIs[OldInsnID]->getParent()->getParent(),
                              State.MIs[OldInsnID]));
      OutMIs[NewInsnID]->setDesc(TII.get(NewOpcode));
      DEBUG(dbgs() << "GIR_MutateOpcode(OutMIs[" << NewInsnID << "], MIs["
                   << OldInsnID << "], " << NewOpcode << ")\n");
      break;
    }
    case GIR_BuildMI: {
      int64_t InsnID = *Command++;
      int64_t Opcode = *Command++;
      assert((size_t)InsnID == OutMIs.size() &&
             "Expected to store MIs in order");
      (void)InsnID;
      OutMIs.push_back(BuildMI(*State.MIs[0]->getParent(), State.MIs[0],
                               State.MIs[0]->getDebugLoc(), TII.get(Opcode)));
      DEBUG(dbgs() << "GIR_BuildMI(OutMIs[" << InsnID << "], " << Opcode
                   << ")\n");
      break;
    }

    case GIR_Copy: {
      int64_t NewInsnID = *Command++;
      int64_t OldInsnID = *Command++;
      int64_t OpIdx = *Command++;
      assert(OutMIs[NewInsnID] && "Attempted to add to undefined instruction");
      OutMIs[NewInsnID].add(State.MIs[OldInsnID]->getOperand(OpIdx));
      DEBUG(dbgs() << "GIR_Copy(OutMIs[" << NewInsnID << "], MIs[" << OldInsnID
                   << "], " << OpIdx << ")\n");
      break;
    }
    case GIR_CopySubReg: {
      int64_t NewInsnID = *Command++;
      int64_t OldInsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t SubRegIdx = *Command++;
      assert(OutMIs[NewInsnID] && "Attempted to add to undefined instruction");
      OutMIs[NewInsnID].addReg(State.MIs[OldInsnID]->getOperand(OpIdx).getReg(),
                               0, SubRegIdx);
      DEBUG(dbgs() << "GIR_CopySubReg(OutMIs[" << NewInsnID << "], MIs["
                   << OldInsnID << "], " << OpIdx << ", " << SubRegIdx
                   << ")\n");
      break;
    }
    case GIR_AddImplicitDef: {
      int64_t InsnID = *Command++;
      int64_t RegNum = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      OutMIs[InsnID].addDef(RegNum, RegState::Implicit);
      DEBUG(dbgs() << "GIR_AddImplicitDef(OutMIs[" << InsnID << "], " << RegNum
                   << ")\n");
      break;
    }
    case GIR_AddImplicitUse: {
      int64_t InsnID = *Command++;
      int64_t RegNum = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      OutMIs[InsnID].addUse(RegNum, RegState::Implicit);
      DEBUG(dbgs() << "GIR_AddImplicitUse(OutMIs[" << InsnID << "], " << RegNum
                   << ")\n");
      break;
    }
    case GIR_AddRegister: {
      int64_t InsnID = *Command++;
      int64_t RegNum = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      OutMIs[InsnID].addReg(RegNum);
      DEBUG(dbgs() << "GIR_AddRegister(OutMIs[" << InsnID << "], " << RegNum
                   << ")\n");
      break;
    }
    case GIR_AddImm: {
      int64_t InsnID = *Command++;
      int64_t Imm = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      OutMIs[InsnID].addImm(Imm);
      DEBUG(dbgs() << "GIR_AddImm(OutMIs[" << InsnID << "], " << Imm << ")\n");
      break;
    }
    case GIR_ComplexRenderer: {
      int64_t InsnID = *Command++;
      int64_t RendererID = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      State.Renderers[RendererID](OutMIs[InsnID]);
      DEBUG(dbgs() << "GIR_ComplexRenderer(OutMIs[" << InsnID << "], "
                   << RendererID << ")\n");
      break;
    }

    case GIR_ConstrainOperandRC: {
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t RCEnum = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      constrainOperandRegToRegClass(*OutMIs[InsnID].getInstr(), OpIdx,
                                    *TRI.getRegClass(RCEnum), TII, TRI, RBI);
      DEBUG(dbgs() << "GIR_ConstrainOperandRC(OutMIs[" << InsnID << "], "
                   << OpIdx << ", " << RCEnum << ")\n");
      break;
    }
    case GIR_ConstrainSelectedInstOperands: {
      int64_t InsnID = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      constrainSelectedInstRegOperands(*OutMIs[InsnID].getInstr(), TII, TRI,
                                       RBI);
      DEBUG(dbgs() << "GIR_ConstrainSelectedInstOperands(OutMIs[" << InsnID
                   << "])\n");
      break;
    }
    case GIR_MergeMemOperands: {
      int64_t InsnID = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      for (const auto *FromMI : State.MIs)
        for (const auto &MMO : FromMI->memoperands())
          OutMIs[InsnID].addMemOperand(MMO);
      DEBUG(dbgs() << "GIR_MergeMemOperands(OutMIs[" << InsnID << "])\n");
      break;
    }
    case GIR_EraseFromParent: {
      int64_t InsnID = *Command++;
      assert(State.MIs[InsnID] &&
             "Attempted to erase an undefined instruction");
      State.MIs[InsnID]->eraseFromParent();
      DEBUG(dbgs() << "GIR_EraseFromParent(MIs[" << InsnID << "])\n");
      break;
    }

    case GIR_Done:
      DEBUG(dbgs() << "GIR_Done");
      return true;

    default:
      llvm_unreachable("Unexpected command");
    }
  }
}

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTORIMPL_H
