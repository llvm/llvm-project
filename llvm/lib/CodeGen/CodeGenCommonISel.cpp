//===-- CodeGenCommonISel.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilies that are shared between SelectionDAG and
// GlobalISel frameworks.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CodeGenCommonISel.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instruction.h"

#define DEBUG_TYPE "codegen-common"

using namespace llvm;

/// Add a successor MBB to ParentMBB< creating a new MachineBB for BB if SuccMBB
/// is 0.
MachineBasicBlock *
StackProtectorDescriptor::addSuccessorMBB(
    const BasicBlock *BB, MachineBasicBlock *ParentMBB, bool IsLikely,
    MachineBasicBlock *SuccMBB) {
  // If SuccBB has not been created yet, create it.
  if (!SuccMBB) {
    MachineFunction *MF = ParentMBB->getParent();
    MachineFunction::iterator BBI(ParentMBB);
    SuccMBB = MF->CreateMachineBasicBlock(BB);
    MF->insert(++BBI, SuccMBB);
  }
  // Add it as a successor of ParentMBB.
  ParentMBB->addSuccessor(
      SuccMBB, BranchProbabilityInfo::getBranchProbStackProtector(IsLikely));
  return SuccMBB;
}

/// Given that the input MI is before a partial terminator sequence TSeq, return
/// true if M + TSeq also a partial terminator sequence.
///
/// A Terminator sequence is a sequence of MachineInstrs which at this point in
/// lowering copy vregs into physical registers, which are then passed into
/// terminator instructors so we can satisfy ABI constraints. A partial
/// terminator sequence is an improper subset of a terminator sequence (i.e. it
/// may be the whole terminator sequence).
static bool MIIsInTerminatorSequence(const MachineInstr &MI) {
  // If we do not have a copy or an implicit def, we return true if and only if
  // MI is a debug value.
  if (!MI.isCopy() && !MI.isImplicitDef()) {
    // Sometimes DBG_VALUE MI sneak in between the copies from the vregs to the
    // physical registers if there is debug info associated with the terminator
    // of our mbb. We want to include said debug info in our terminator
    // sequence, so we return true in that case.
    if (MI.isDebugInstr())
      return true;

    // For GlobalISel, we may have extension instructions for arguments within
    // copy sequences. Allow these.
    switch (MI.getOpcode()) {
    case TargetOpcode::G_TRUNC:
    case TargetOpcode::G_ZEXT:
    case TargetOpcode::G_ANYEXT:
    case TargetOpcode::G_SEXT:
    case TargetOpcode::G_MERGE_VALUES:
    case TargetOpcode::G_UNMERGE_VALUES:
    case TargetOpcode::G_CONCAT_VECTORS:
    case TargetOpcode::G_BUILD_VECTOR:
    case TargetOpcode::G_EXTRACT:
      return true;
    default:
      return false;
    }
  }

  // We have left the terminator sequence if we are not doing one of the
  // following:
  //
  // 1. Copying a vreg into a physical register.
  // 2. Copying a vreg into a vreg.
  // 3. Defining a register via an implicit def.

  // OPI should always be a register definition...
  MachineInstr::const_mop_iterator OPI = MI.operands_begin();
  if (!OPI->isReg() || !OPI->isDef())
    return false;

  // Defining any register via an implicit def is always ok.
  if (MI.isImplicitDef())
    return true;

  // Grab the copy source...
  MachineInstr::const_mop_iterator OPI2 = OPI;
  ++OPI2;
  assert(OPI2 != MI.operands_end()
         && "Should have a copy implying we should have 2 arguments.");

  // Make sure that the copy dest is not a vreg when the copy source is a
  // physical register.
  if (!OPI2->isReg() ||
      (!OPI->getReg().isPhysical() && OPI2->getReg().isPhysical()))
    return false;

  return true;
}

/// Find the split point at which to splice the end of BB into its success stack
/// protector check machine basic block.
///
/// On many platforms, due to ABI constraints, terminators, even before register
/// allocation, use physical registers. This creates an issue for us since
/// physical registers at this point can not travel across basic
/// blocks. Luckily, selectiondag always moves physical registers into vregs
/// when they enter functions and moves them through a sequence of copies back
/// into the physical registers right before the terminator creating a
/// ``Terminator Sequence''. This function is searching for the beginning of the
/// terminator sequence so that we can ensure that we splice off not just the
/// terminator, but additionally the copies that move the vregs into the
/// physical registers.
MachineBasicBlock::iterator
llvm::findSplitPointForStackProtector(MachineBasicBlock *BB,
                                      const TargetInstrInfo &TII) {
  MachineBasicBlock::iterator SplitPoint = BB->getFirstTerminator();
  if (SplitPoint == BB->begin())
    return SplitPoint;

  MachineBasicBlock::iterator Start = BB->begin();
  MachineBasicBlock::iterator Previous = SplitPoint;
  do {
    --Previous;
  } while (Previous != Start && Previous->isDebugInstr());

  if (TII.isTailCall(*SplitPoint) &&
      Previous->getOpcode() == TII.getCallFrameDestroyOpcode()) {
    // Call frames cannot be nested, so if this frame is describing the tail
    // call itself, then we must insert before the sequence even starts. For
    // example:
    //     <split point>
    //     ADJCALLSTACKDOWN ...
    //     <Moves>
    //     ADJCALLSTACKUP ...
    //     TAILJMP somewhere
    // On the other hand, it could be an unrelated call in which case this tail
    // call has no register moves of its own and should be the split point. For
    // example:
    //     ADJCALLSTACKDOWN
    //     CALL something_else
    //     ADJCALLSTACKUP
    //     <split point>
    //     TAILJMP somewhere
    do {
      --Previous;
      if (Previous->isCall())
        return SplitPoint;
    } while(Previous->getOpcode() != TII.getCallFrameSetupOpcode());

    return Previous;
  }

  while (MIIsInTerminatorSequence(*Previous)) {
    SplitPoint = Previous;
    if (Previous == Start)
      break;
    --Previous;
  }

  return SplitPoint;
}

FPClassTest llvm::invertFPClassTestIfSimpler(FPClassTest Test, bool UseFCmp) {
  FPClassTest InvertedTest = ~Test;

  // Pick the direction with fewer tests
  // TODO: Handle more combinations of cases that can be handled together
  switch (static_cast<unsigned>(InvertedTest)) {
  case fcNan:
  case fcSNan:
  case fcQNan:
  case fcInf:
  case fcPosInf:
  case fcNegInf:
  case fcNormal:
  case fcPosNormal:
  case fcNegNormal:
  case fcSubnormal:
  case fcPosSubnormal:
  case fcNegSubnormal:
  case fcZero:
  case fcPosZero:
  case fcNegZero:
  case fcFinite:
  case fcPosFinite:
  case fcNegFinite:
  case fcZero | fcNan:
  case fcSubnormal | fcZero:
  case fcSubnormal | fcZero | fcNan:
    return InvertedTest;
  case fcInf | fcNan:
  case fcPosInf | fcNan:
  case fcNegInf | fcNan:
    // If we're trying to use fcmp, we can take advantage of the nan check
    // behavior of the compare (but this is more instructions in the integer
    // expansion).
    return UseFCmp ? InvertedTest : fcNone;
  default:
    return fcNone;
  }

  llvm_unreachable("covered FPClassTest");
}

static MachineOperand *getSalvageOpsForCopy(const MachineRegisterInfo &MRI,
                                            MachineInstr &Copy) {
  assert((Copy.getOpcode() == TargetOpcode::COPY ||
          Copy.getOpcode() == TargetOpcode::G_BITCAST) &&
         "Must be a COPY or G_BITCAST");

  return &Copy.getOperand(1);
}

static MachineOperand *getSalvageOpsForTrunc(const MachineRegisterInfo &MRI,
                                            MachineInstr &Trunc,
                                            SmallVectorImpl<uint64_t> &Ops) {
  assert(Trunc.getOpcode() == TargetOpcode::G_TRUNC && "Must be a G_TRUNC");

  const auto FromLLT = MRI.getType(Trunc.getOperand(1).getReg());
  const auto ToLLT = MRI.getType(Trunc.defs().begin()->getReg());

  // TODO: Support non-scalar types.
  if (!FromLLT.isScalar()) {
    return nullptr;
  }

  auto ExtOps = DIExpression::getExtOps(FromLLT.getSizeInBits(),
                                        ToLLT.getSizeInBits(), false);
  Ops.append(ExtOps.begin(), ExtOps.end());
  return &Trunc.getOperand(1);
}

static MachineOperand *
getSalvageOpsForAdd(const MachineRegisterInfo &MRI, MachineInstr &Add,
                    uint64_t CurrentLocOps, SmallVectorImpl<uint64_t> &Ops,
                    SmallVectorImpl<Register> &AdditionalRegs) {
  assert((Add.getOpcode() == TargetOpcode::G_ADD ||
          Add.getOpcode() == TargetOpcode::G_PTR_ADD) &&
         "Must be a G_ADD or G_PTR_ADD");

  if (MRI.getType(Add.getOperand(0).getReg()).isVector())
    return nullptr;

  Register RHS = Add.getOperand(2).getReg();
  const MachineInstr *RHSDef = MRI.getVRegDef(RHS);
  if (RHSDef && RHSDef->getOpcode() == TargetOpcode::G_CONSTANT) {
    const APInt &Val = RHSDef->getOperand(1).getCImm()->getValue();
    if (Val.getBitWidth() > 64)
      return nullptr;
    DIExpression::appendOffset(Ops, Val.getSExtValue());
    return &Add.getOperand(1);
  }

  if (!CurrentLocOps) {
    Ops.append({dwarf::DW_OP_LLVM_arg, 0});
    CurrentLocOps = 1;
  }
  Ops.append({dwarf::DW_OP_LLVM_arg, CurrentLocOps, dwarf::DW_OP_plus});
  AdditionalRegs.push_back(RHS);
  return &Add.getOperand(1);
}

static MachineOperand *
salvageDebugInfoImpl(const MachineRegisterInfo &MRI, MachineInstr &MI,
                     uint64_t CurrentLocOps, SmallVectorImpl<uint64_t> &Ops,
                     SmallVectorImpl<Register> &AdditionalRegs) {
  switch (MI.getOpcode()) {
  case TargetOpcode::G_TRUNC:
    return getSalvageOpsForTrunc(MRI, MI, Ops);
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_PTR_ADD:
    return getSalvageOpsForAdd(MRI, MI, CurrentLocOps, Ops, AdditionalRegs);
  case TargetOpcode::COPY:
  case TargetOpcode::G_BITCAST:
    return getSalvageOpsForCopy(MRI, MI);
  default:
    return nullptr;
  }
}

/// Replace \p DbgMI, which describes the result of \p BuildVector, with one
/// fragment debug value per vector element.
static void salvageBuildVector(const MachineRegisterInfo &MRI,
                               MachineInstr &BuildVector, MachineInstr &DbgMI) {
  const DIExpression *Expr = DbgMI.getDebugExpression();
  if (DbgMI.getNumDebugOperands() != 1 || !Expr->isSingleLocationExpression() ||
      Expr->isComplex())
    return;

  unsigned VecSize = MRI.getType(BuildVector.getOperand(0).getReg())
                         .getSizeInBits()
                         .getFixedValue();
  if (auto Fragment = Expr->getFragmentInfo()) {
    if (VecSize > Fragment->SizeInBits)
      return;
  } else if (auto VarSize = DbgMI.getDebugVariable()->getSizeInBits()) {
    if (VecSize > *VarSize)
      return;
  }

  unsigned NumElts = BuildVector.getNumOperands() - 1;
  unsigned EltSize = VecSize / NumElts;
  SmallVector<DIExpression *, 8> FragExprs;
  for (unsigned I = 0; I != NumElts; ++I) {
    auto FragExpr =
        DIExpression::createFragmentExpression(Expr, I * EltSize, EltSize);
    if (!FragExpr)
      return;
    FragExprs.push_back(*FragExpr);
  }

  MachineBasicBlock &MBB = *DbgMI.getParent();
  MachineFunction &MF = *MBB.getParent();
  for (unsigned I = NumElts - 1; I != 0; --I) {
    MachineInstr *NewMI = MF.CloneMachineInstr(&DbgMI);
    NewMI->getDebugOperand(0).setReg(BuildVector.getOperand(I + 1).getReg());
    NewMI->getDebugExpressionOp().setMetadata(FragExprs[I]);
    MBB.insertAfter(DbgMI.getIterator(), NewMI);
    LLVM_DEBUG(dbgs() << "SALVAGE: " << *NewMI << '\n');
  }
  DbgMI.getDebugOperand(0).setReg(BuildVector.getOperand(1).getReg());
  DbgMI.getDebugExpressionOp().setMetadata(FragExprs[0]);
  LLVM_DEBUG(dbgs() << "SALVAGE: " << DbgMI << '\n');
}

void llvm::salvageDebugInfoForDbgValue(const MachineRegisterInfo &MRI,
                                       MachineInstr &MI,
                                       ArrayRef<MachineOperand *> DbgUsers) {
  // These are arbitrary chosen limits on the maximum number of values and the
  // maximum size of a debug expression we can salvage up to, used for
  // performance reasons.
  const unsigned MaxDebugArgs = 16;
  const unsigned MaxExpressionSize = 128;

  if (DbgUsers.empty())
    return;
  Register Reg = DbgUsers.front()->getReg();

  // Collect the users up front: salvaging may add operands to or replace them.
  SmallSetVector<MachineInstr *, 4> DbgMIs;
  for (auto *DefMO : DbgUsers)
    DbgMIs.insert(DefMO->getParent());

  for (MachineInstr *DbgMI : DbgMIs) {
    if (DbgMI->isIndirectDebugValue()) {
      continue;
    }

    assert(DbgMI->hasDebugOperandForReg(Reg) &&
           "Must use salvaged instruction as its location");

    if (MI.getOpcode() == TargetOpcode::G_BUILD_VECTOR) {
      assert(DbgMI->isDebugValue() &&
             "Must be either DBG_VALUE or DBG_VALUE_LIST");
      salvageBuildVector(MRI, MI, *DbgMI);
      continue;
    }

    const DIExpression *SalvagedExpr = DbgMI->getDebugExpression();
    SmallVector<Register, 2> AdditionalRegs;
    MachineOperand *Op0 = nullptr;

    SmallVector<MachineOperand *, 2> LocMOs(
        make_pointer_range(DbgMI->getDebugOperandsForReg(Reg)));
    for (MachineOperand *LocMO : LocMOs) {
      SmallVector<uint64_t, 16> Ops;
      Op0 = salvageDebugInfoImpl(
          MRI, MI, SalvagedExpr->getNumLocationOperands(), Ops, AdditionalRegs);
      if (!Op0)
        break;
      SalvagedExpr = DIExpression::appendOpsToArg(
          SalvagedExpr, Ops, DbgMI->getDebugOperandIndex(LocMO), true);
    }
    if (!Op0)
      continue;

    if (SalvagedExpr->getNumElements() > MaxExpressionSize ||
        DbgMI->getNumDebugOperands() + AdditionalRegs.size() > MaxDebugArgs)
      continue;

    for (MachineOperand *LocMO : LocMOs) {
      LocMO->setReg(Op0->getReg());
      LocMO->setSubReg(Op0->getSubReg());
    }

    if (AdditionalRegs.empty() || DbgMI->isDebugValueList()) {
      DbgMI->getDebugExpressionOp().setMetadata(SalvagedExpr);
      for (Register AddReg : AdditionalRegs)
        DbgMI->addOperand(MachineOperand::CreateReg(AddReg, /*isDef=*/false));
      LLVM_DEBUG(dbgs() << "SALVAGE: " << *DbgMI << '\n');
      continue;
    }

    // Extra location operands require turning the DBG_VALUE into a
    // DBG_VALUE_LIST.
    MachineBasicBlock &MBB = *DbgMI->getParent();
    const TargetInstrInfo &TII =
        *MBB.getParent()->getSubtarget().getInstrInfo();
    auto MIB = BuildMI(MBB, DbgMI, DbgMI->getDebugLoc(),
                       TII.get(TargetOpcode::DBG_VALUE_LIST))
                   .addMetadata(DbgMI->getDebugVariable())
                   .addMetadata(SalvagedExpr)
                   .add(DbgMI->getDebugOperand(0));
    for (Register AddReg : AdditionalRegs)
      MIB.addReg(AddReg);
    LLVM_DEBUG(dbgs() << "SALVAGE: " << *MIB << '\n');
    DbgMI->eraseFromParent();
  }
}
