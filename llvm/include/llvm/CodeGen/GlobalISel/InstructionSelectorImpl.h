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
    TgtInstructionSelector &ISel, MatcherState &State,
    const MatcherInfoTy<PredicateBitset, ComplexMatcherMemFn> &MatcherInfo,
    const int64_t *MatchTable, MachineRegisterInfo &MRI,
    const TargetRegisterInfo &TRI, const RegisterBankInfo &RBI,
    const PredicateBitset &AvailableFeatures) const {
  const int64_t *Command = MatchTable;
  while (true) {
    switch (*Command++) {
    case GIM_RecordInsn: {
      int64_t NewInsnID LLVM_ATTRIBUTE_UNUSED = *Command++;
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;

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
      DEBUG(dbgs() << "GIM_CheckOpcode(MIs[" << InsnID
                   << "], ExpectedOpcode=" << Expected << ")\n");
      assert(State.MIs[InsnID] != nullptr && "Used insn before defined");
      if (State.MIs[InsnID]->getOpcode() != Expected)
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

    case GIM_Accept:
      DEBUG(dbgs() << "GIM_Accept\n");
      return true;
    default:
      llvm_unreachable("Unexpected command");
    }
  }
}

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTORIMPL_H
