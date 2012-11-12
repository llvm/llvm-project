//===-- FastISel.cpp - Implementation of the FastISel class ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the FastISel class.
//
// "Fast" instruction selection is designed to emit very poor code quickly.
// Also, it is not designed to be able to do much lowering, so most illegal
// types (e.g. i64 on 32-bit targets) and operations are not supported.  It is
// also not intended to be able to do much optimization, except in a few cases
// where doing optimizations reduces overall compile time.  For example, folding
// constants into immediate fields is often done, because it's cheap and it
// reduces the number of instructions later phases have to examine.
//
// "Fast" instruction selection is able to fail gracefully and transfer
// control to the SelectionDAG selector for operations that it doesn't
// support.  In many cases, this allows us to avoid duplicating a lot of
// the complicated lowering logic that SelectionDAG currently has.
//
// The intended use for "fast" instruction selection is "-O0" mode
// compilation, where the quality of the generated code is irrelevant when
// weighed against the speed at which the code can be generated.  Also,
// at -O0, the LLVM optimizers are not running, and this makes the
// compile time of codegen a much higher portion of the overall compile
// time.  Despite its limitations, "fast" instruction selection is able to
// handle enough code on its own to provide noticeable overall speedups
// in -O0 compiles.
//
// Basic operations are supported in a target-independent way, by reading
// the same instruction descriptions that the SelectionDAG selector reads,
// and identifying simple arithmetic operations that can be directly selected
// from simple operators.  More complicated operations currently require
// target-specific code.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "isel"
#include "llvm/DebugInfo.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Operator.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumFastIselSuccessIndependent, "Number of insts selected by "
          "target-independent selector");
STATISTIC(NumFastIselSuccessTarget, "Number of insts selected by "
          "target-specific selector");
STATISTIC(NumFastIselDead, "Number of dead insts removed on failure");

/// startNewBlock - Set the current block to which generated machine
/// instructions will be appended, and clear the local CSE map.
///
void FastISel::startNewBlock() {
  LocalValueMap.clear();

  EmitStartPt = 0;

  // Advance the emit start point past any EH_LABEL instructions.
  MachineBasicBlock::iterator
    I = FuncInfo.MBB->begin(), E = FuncInfo.MBB->end();
  while (I != E && I->getOpcode() == TargetOpcode::EH_LABEL) {
    EmitStartPt = I;
    ++I;
  }
  LastLocalValue = EmitStartPt;
}

void FastISel::flushLocalValueMap() {
  LocalValueMap.clear();
  LastLocalValue = EmitStartPt;
  recomputeInsertPt();
}

bool FastISel::hasTrivialKill(const Value *V) const {
  // Don't consider constants or arguments to have trivial kills.
  const Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;

  // No-op casts are trivially coalesced by fast-isel.
  if (const CastInst *Cast = dyn_cast<CastInst>(I))
    if (Cast->isNoopCast(TD.getIntPtrType(Cast->getContext())) &&
        !hasTrivialKill(Cast->getOperand(0)))
      return false;

  // GEPs with all zero indices are trivially coalesced by fast-isel.
  if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I))
    if (GEP->hasAllZeroIndices() && !hasTrivialKill(GEP->getOperand(0)))
      return false;

  // Only instructions with a single use in the same basic block are considered
  // to have trivial kills.
  return I->hasOneUse() &&
         !(I->getOpcode() == Instruction::BitCast ||
           I->getOpcode() == Instruction::PtrToInt ||
           I->getOpcode() == Instruction::IntToPtr) &&
         cast<Instruction>(*I->use_begin())->getParent() == I->getParent();
}

unsigned FastISel::getRegForValue(const Value *V) {
  EVT RealVT = TLI.getValueType(V->getType(), /*AllowUnknown=*/true);
  // Don't handle non-simple values in FastISel.
  if (!RealVT.isSimple())
    return 0;

  // Ignore illegal types. We must do this before looking up the value
  // in ValueMap because Arguments are given virtual registers regardless
  // of whether FastISel can handle them.
  MVT VT = RealVT.getSimpleVT();
  if (!TLI.isTypeLegal(VT)) {
    // Handle integer promotions, though, because they're common and easy.
    if (VT == MVT::i1 || VT == MVT::i8 || VT == MVT::i16)
      VT = TLI.getTypeToTransformTo(V->getContext(), VT).getSimpleVT();
    else
      return 0;
  }

  // Look up the value to see if we already have a register for it.
  unsigned Reg = lookUpRegForValue(V);
  if (Reg != 0)
    return Reg;

  // In bottom-up mode, just create the virtual register which will be used
  // to hold the value. It will be materialized later.
  if (isa<Instruction>(V) &&
      (!isa<AllocaInst>(V) ||
       !FuncInfo.StaticAllocaMap.count(cast<AllocaInst>(V))))
    return FuncInfo.InitializeRegForValue(V);

  SavePoint SaveInsertPt = enterLocalValueArea();

  // Materialize the value in a register. Emit any instructions in the
  // local value area.
  Reg = materializeRegForValue(V, VT);

  leaveLocalValueArea(SaveInsertPt);

  return Reg;
}

/// materializeRegForValue - Helper for getRegForValue. This function is
/// called when the value isn't already available in a register and must
/// be materialized with new instructions.
unsigned FastISel::materializeRegForValue(const Value *V, MVT VT) {
  unsigned Reg = 0;

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getValue().getActiveBits() <= 64)
      Reg = FastEmit_i(VT, VT, ISD::Constant, CI->getZExtValue());
  } else if (isa<AllocaInst>(V)) {
    Reg = TargetMaterializeAlloca(cast<AllocaInst>(V));
  } else if (isa<ConstantPointerNull>(V)) {
    // Translate this as an integer zero so that it can be
    // local-CSE'd with actual integer zeros.
    Reg =
      getRegForValue(Constant::getNullValue(TD.getIntPtrType(V->getContext())));
  } else if (const ConstantFP *CF = dyn_cast<ConstantFP>(V)) {
    if (CF->isNullValue()) {
      Reg = TargetMaterializeFloatZero(CF);
    } else {
      // Try to emit the constant directly.
      Reg = FastEmit_f(VT, VT, ISD::ConstantFP, CF);
    }

    if (!Reg) {
      // Try to emit the constant by using an integer constant with a cast.
      const APFloat &Flt = CF->getValueAPF();
      EVT IntVT = TLI.getPointerTy();

      uint64_t x[2];
      uint32_t IntBitWidth = IntVT.getSizeInBits();
      bool isExact;
      (void) Flt.convertToInteger(x, IntBitWidth, /*isSigned=*/true,
                                  APFloat::rmTowardZero, &isExact);
      if (isExact) {
        APInt IntVal(IntBitWidth, x);

        unsigned IntegerReg =
          getRegForValue(ConstantInt::get(V->getContext(), IntVal));
        if (IntegerReg != 0)
          Reg = FastEmit_r(IntVT.getSimpleVT(), VT, ISD::SINT_TO_FP,
                           IntegerReg, /*Kill=*/false);
      }
    }
  } else if (const Operator *Op = dyn_cast<Operator>(V)) {
    if (!SelectOperator(Op, Op->getOpcode()))
      if (!isa<Instruction>(Op) ||
          !TargetSelectInstruction(cast<Instruction>(Op)))
        return 0;
    Reg = lookUpRegForValue(Op);
  } else if (isa<UndefValue>(V)) {
    Reg = createResultReg(TLI.getRegClassFor(VT));
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL,
            TII.get(TargetOpcode::IMPLICIT_DEF), Reg);
  }

  // If target-independent code couldn't handle the value, give target-specific
  // code a try.
  if (!Reg && isa<Constant>(V))
    Reg = TargetMaterializeConstant(cast<Constant>(V));

  // Don't cache constant materializations in the general ValueMap.
  // To do so would require tracking what uses they dominate.
  if (Reg != 0) {
    LocalValueMap[V] = Reg;
    LastLocalValue = MRI.getVRegDef(Reg);
  }
  return Reg;
}

unsigned FastISel::lookUpRegForValue(const Value *V) {
  // Look up the value to see if we already have a register for it. We
  // cache values defined by Instructions across blocks, and other values
  // only locally. This is because Instructions already have the SSA
  // def-dominates-use requirement enforced.
  DenseMap<const Value *, unsigned>::iterator I = FuncInfo.ValueMap.find(V);
  if (I != FuncInfo.ValueMap.end())
    return I->second;
  return LocalValueMap[V];
}

/// UpdateValueMap - Update the value map to include the new mapping for this
/// instruction, or insert an extra copy to get the result in a previous
/// determined register.
/// NOTE: This is only necessary because we might select a block that uses
/// a value before we select the block that defines the value.  It might be
/// possible to fix this by selecting blocks in reverse postorder.
void FastISel::UpdateValueMap(const Value *I, unsigned Reg, unsigned NumRegs) {
  if (!isa<Instruction>(I)) {
    LocalValueMap[I] = Reg;
    return;
  }

  unsigned &AssignedReg = FuncInfo.ValueMap[I];
  if (AssignedReg == 0)
    // Use the new register.
    AssignedReg = Reg;
  else if (Reg != AssignedReg) {
    // Arrange for uses of AssignedReg to be replaced by uses of Reg.
    for (unsigned i = 0; i < NumRegs; i++)
      FuncInfo.RegFixups[AssignedReg+i] = Reg+i;

    AssignedReg = Reg;
  }
}

std::pair<unsigned, bool> FastISel::getRegForGEPIndex(const Value *Idx) {
  unsigned IdxN = getRegForValue(Idx);
  if (IdxN == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return std::pair<unsigned, bool>(0, false);

  bool IdxNIsKill = hasTrivialKill(Idx);

  // If the index is smaller or larger than intptr_t, truncate or extend it.
  MVT PtrVT = TLI.getPointerTy();
  EVT IdxVT = EVT::getEVT(Idx->getType(), /*HandleUnknown=*/false);
  if (IdxVT.bitsLT(PtrVT)) {
    IdxN = FastEmit_r(IdxVT.getSimpleVT(), PtrVT, ISD::SIGN_EXTEND,
                      IdxN, IdxNIsKill);
    IdxNIsKill = true;
  }
  else if (IdxVT.bitsGT(PtrVT)) {
    IdxN = FastEmit_r(IdxVT.getSimpleVT(), PtrVT, ISD::TRUNCATE,
                      IdxN, IdxNIsKill);
    IdxNIsKill = true;
  }
  return std::pair<unsigned, bool>(IdxN, IdxNIsKill);
}

void FastISel::recomputeInsertPt() {
  if (getLastLocalValue()) {
    FuncInfo.InsertPt = getLastLocalValue();
    FuncInfo.MBB = FuncInfo.InsertPt->getParent();
    ++FuncInfo.InsertPt;
  } else
    FuncInfo.InsertPt = FuncInfo.MBB->getFirstNonPHI();

  // Now skip past any EH_LABELs, which must remain at the beginning.
  while (FuncInfo.InsertPt != FuncInfo.MBB->end() &&
         FuncInfo.InsertPt->getOpcode() == TargetOpcode::EH_LABEL)
    ++FuncInfo.InsertPt;
}

void FastISel::removeDeadCode(MachineBasicBlock::iterator I,
                              MachineBasicBlock::iterator E) {
  assert (I && E && std::distance(I, E) > 0 && "Invalid iterator!");
  while (I != E) {
    MachineInstr *Dead = &*I;
    ++I;
    Dead->eraseFromParent();
    ++NumFastIselDead;
  }
  recomputeInsertPt();
}

FastISel::SavePoint FastISel::enterLocalValueArea() {
  MachineBasicBlock::iterator OldInsertPt = FuncInfo.InsertPt;
  DebugLoc OldDL = DL;
  recomputeInsertPt();
  DL = DebugLoc();
  SavePoint SP = { OldInsertPt, OldDL };
  return SP;
}

void FastISel::leaveLocalValueArea(SavePoint OldInsertPt) {
  if (FuncInfo.InsertPt != FuncInfo.MBB->begin())
    LastLocalValue = llvm::prior(FuncInfo.InsertPt);

  // Restore the previous insert position.
  FuncInfo.InsertPt = OldInsertPt.InsertPt;
  DL = OldInsertPt.DL;
}

/// SelectBinaryOp - Select and emit code for a binary operator instruction,
/// which has an opcode which directly corresponds to the given ISD opcode.
///
bool FastISel::SelectBinaryOp(const User *I, unsigned ISDOpcode) {
  EVT VT = EVT::getEVT(I->getType(), /*HandleUnknown=*/true);
  if (VT == MVT::Other || !VT.isSimple())
    // Unhandled type. Halt "fast" selection and bail.
    return false;

  // We only handle legal types. For example, on x86-32 the instruction
  // selector contains all of the 64-bit instructions from x86-64,
  // under the assumption that i64 won't be used if the target doesn't
  // support it.
  if (!TLI.isTypeLegal(VT)) {
    // MVT::i1 is special. Allow AND, OR, or XOR because they
    // don't require additional zeroing, which makes them easy.
    if (VT == MVT::i1 &&
        (ISDOpcode == ISD::AND || ISDOpcode == ISD::OR ||
         ISDOpcode == ISD::XOR))
      VT = TLI.getTypeToTransformTo(I->getContext(), VT);
    else
      return false;
  }

  // Check if the first operand is a constant, and handle it as "ri".  At -O0,
  // we don't have anything that canonicalizes operand order.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(0)))
    if (isa<Instruction>(I) && cast<Instruction>(I)->isCommutative()) {
      unsigned Op1 = getRegForValue(I->getOperand(1));
      if (Op1 == 0) return false;

      bool Op1IsKill = hasTrivialKill(I->getOperand(1));

      unsigned ResultReg = FastEmit_ri_(VT.getSimpleVT(), ISDOpcode, Op1,
                                        Op1IsKill, CI->getZExtValue(),
                                        VT.getSimpleVT());
      if (ResultReg == 0) return false;

      // We successfully emitted code for the given LLVM Instruction.
      UpdateValueMap(I, ResultReg);
      return true;
    }


  unsigned Op0 = getRegForValue(I->getOperand(0));
  if (Op0 == 0)   // Unhandled operand. Halt "fast" selection and bail.
    return false;

  bool Op0IsKill = hasTrivialKill(I->getOperand(0));

  // Check if the second operand is a constant and handle it appropriately.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
    uint64_t Imm = CI->getZExtValue();

    // Transform "sdiv exact X, 8" -> "sra X, 3".
    if (ISDOpcode == ISD::SDIV && isa<BinaryOperator>(I) &&
        cast<BinaryOperator>(I)->isExact() &&
        isPowerOf2_64(Imm)) {
      Imm = Log2_64(Imm);
      ISDOpcode = ISD::SRA;
    }

    // Transform "urem x, pow2" -> "and x, pow2-1".
    if (ISDOpcode == ISD::UREM && isa<BinaryOperator>(I) &&
        isPowerOf2_64(Imm)) {
      --Imm;
      ISDOpcode = ISD::AND;
    }

    unsigned ResultReg = FastEmit_ri_(VT.getSimpleVT(), ISDOpcode, Op0,
                                      Op0IsKill, Imm, VT.getSimpleVT());
    if (ResultReg == 0) return false;

    // We successfully emitted code for the given LLVM Instruction.
    UpdateValueMap(I, ResultReg);
    return true;
  }

  // Check if the second operand is a constant float.
  if (ConstantFP *CF = dyn_cast<ConstantFP>(I->getOperand(1))) {
    unsigned ResultReg = FastEmit_rf(VT.getSimpleVT(), VT.getSimpleVT(),
                                     ISDOpcode, Op0, Op0IsKill, CF);
    if (ResultReg != 0) {
      // We successfully emitted code for the given LLVM Instruction.
      UpdateValueMap(I, ResultReg);
      return true;
    }
  }

  unsigned Op1 = getRegForValue(I->getOperand(1));
  if (Op1 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  bool Op1IsKill = hasTrivialKill(I->getOperand(1));

  // Now we have both operands in registers. Emit the instruction.
  unsigned ResultReg = FastEmit_rr(VT.getSimpleVT(), VT.getSimpleVT(),
                                   ISDOpcode,
                                   Op0, Op0IsKill,
                                   Op1, Op1IsKill);
  if (ResultReg == 0)
    // Target-specific code wasn't able to find a machine opcode for
    // the given ISD opcode and type. Halt "fast" selection and bail.
    return false;

  // We successfully emitted code for the given LLVM Instruction.
  UpdateValueMap(I, ResultReg);
  return true;
}

bool FastISel::SelectGetElementPtr(const User *I) {
  unsigned N = getRegForValue(I->getOperand(0));
  if (N == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  bool NIsKill = hasTrivialKill(I->getOperand(0));

  // Keep a running tab of the total offset to coalesce multiple N = N + Offset
  // into a single N = N + TotalOffset.
  uint64_t TotalOffs = 0;
  // FIXME: What's a good SWAG number for MaxOffs?
  uint64_t MaxOffs = 2048;
  Type *Ty = I->getOperand(0)->getType();
  MVT VT = TLI.getPointerTy();
  for (GetElementPtrInst::const_op_iterator OI = I->op_begin()+1,
       E = I->op_end(); OI != E; ++OI) {
    const Value *Idx = *OI;
    if (StructType *StTy = dyn_cast<StructType>(Ty)) {
      unsigned Field = cast<ConstantInt>(Idx)->getZExtValue();
      if (Field) {
        // N = N + Offset
        TotalOffs += TD.getStructLayout(StTy)->getElementOffset(Field);
        if (TotalOffs >= MaxOffs) {
          N = FastEmit_ri_(VT, ISD::ADD, N, NIsKill, TotalOffs, VT);
          if (N == 0)
            // Unhandled operand. Halt "fast" selection and bail.
            return false;
          NIsKill = true;
          TotalOffs = 0;
        }
      }
      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();

      // If this is a constant subscript, handle it quickly.
      if (const ConstantInt *CI = dyn_cast<ConstantInt>(Idx)) {
        if (CI->isZero()) continue;
        // N = N + Offset
        TotalOffs +=
          TD.getTypeAllocSize(Ty)*cast<ConstantInt>(CI)->getSExtValue();
        if (TotalOffs >= MaxOffs) {
          N = FastEmit_ri_(VT, ISD::ADD, N, NIsKill, TotalOffs, VT);
          if (N == 0)
            // Unhandled operand. Halt "fast" selection and bail.
            return false;
          NIsKill = true;
          TotalOffs = 0;
        }
        continue;
      }
      if (TotalOffs) {
        N = FastEmit_ri_(VT, ISD::ADD, N, NIsKill, TotalOffs, VT);
        if (N == 0)
          // Unhandled operand. Halt "fast" selection and bail.
          return false;
        NIsKill = true;
        TotalOffs = 0;
      }

      // N = N + Idx * ElementSize;
      uint64_t ElementSize = TD.getTypeAllocSize(Ty);
      std::pair<unsigned, bool> Pair = getRegForGEPIndex(Idx);
      unsigned IdxN = Pair.first;
      bool IdxNIsKill = Pair.second;
      if (IdxN == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;

      if (ElementSize != 1) {
        IdxN = FastEmit_ri_(VT, ISD::MUL, IdxN, IdxNIsKill, ElementSize, VT);
        if (IdxN == 0)
          // Unhandled operand. Halt "fast" selection and bail.
          return false;
        IdxNIsKill = true;
      }
      N = FastEmit_rr(VT, VT, ISD::ADD, N, NIsKill, IdxN, IdxNIsKill);
      if (N == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;
    }
  }
  if (TotalOffs) {
    N = FastEmit_ri_(VT, ISD::ADD, N, NIsKill, TotalOffs, VT);
    if (N == 0)
      // Unhandled operand. Halt "fast" selection and bail.
      return false;
  }

  // We successfully emitted code for the given LLVM Instruction.
  UpdateValueMap(I, N);
  return true;
}

bool FastISel::SelectCall(const User *I) {
  const CallInst *Call = cast<CallInst>(I);

  // Handle simple inline asms.
  if (const InlineAsm *IA = dyn_cast<InlineAsm>(Call->getCalledValue())) {
    // Don't attempt to handle constraints.
    if (!IA->getConstraintString().empty())
      return false;

    unsigned ExtraInfo = 0;
    if (IA->hasSideEffects())
      ExtraInfo |= InlineAsm::Extra_HasSideEffects;
    if (IA->isAlignStack())
      ExtraInfo |= InlineAsm::Extra_IsAlignStack;

    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL,
            TII.get(TargetOpcode::INLINEASM))
      .addExternalSymbol(IA->getAsmString().c_str())
      .addImm(ExtraInfo);
    return true;
  }

  MachineModuleInfo &MMI = FuncInfo.MF->getMMI();
  ComputeUsesVAFloatArgument(*Call, &MMI);

  const Function *F = Call->getCalledFunction();
  if (!F) return false;

  // Handle selected intrinsic function calls.
  switch (F->getIntrinsicID()) {
  default: break;
    // At -O0 we don't care about the lifetime intrinsics.
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
    // The donothing intrinsic does, well, nothing.
  case Intrinsic::donothing:
    return true;

  case Intrinsic::dbg_declare: {
    const DbgDeclareInst *DI = cast<DbgDeclareInst>(Call);
    if (!DIVariable(DI->getVariable()).Verify() ||
        !FuncInfo.MF->getMMI().hasDebugInfo()) {
      DEBUG(dbgs() << "Dropping debug info for " << *DI << "\n");
      return true;
    }

    const Value *Address = DI->getAddress();
    if (!Address || isa<UndefValue>(Address)) {
      DEBUG(dbgs() << "Dropping debug info for " << *DI << "\n");
      return true;
    }

    unsigned Reg = 0;
    unsigned Offset = 0;
    if (const Argument *Arg = dyn_cast<Argument>(Address)) {
      // Some arguments' frame index is recorded during argument lowering.
      Offset = FuncInfo.getArgumentFrameIndex(Arg);
      if (Offset)
        Reg = TRI.getFrameRegister(*FuncInfo.MF);
    }
    if (!Reg)
      Reg = lookUpRegForValue(Address);

    // If we have a VLA that has a "use" in a metadata node that's then used
    // here but it has no other uses, then we have a problem. E.g.,
    //
    //   int foo (const int *x) {
    //     char a[*x];
    //     return 0;
    //   }
    //
    // If we assign 'a' a vreg and fast isel later on has to use the selection
    // DAG isel, it will want to copy the value to the vreg. However, there are
    // no uses, which goes counter to what selection DAG isel expects.
    if (!Reg && !Address->use_empty() && isa<Instruction>(Address) &&
        (!isa<AllocaInst>(Address) ||
         !FuncInfo.StaticAllocaMap.count(cast<AllocaInst>(Address))))
      Reg = FuncInfo.InitializeRegForValue(Address);

    if (Reg)
      BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL,
              TII.get(TargetOpcode::DBG_VALUE))
        .addReg(Reg, RegState::Debug).addImm(Offset)
        .addMetadata(DI->getVariable());
    else
      // We can't yet handle anything else here because it would require
      // generating code, thus altering codegen because of debug info.
      DEBUG(dbgs() << "Dropping debug info for " << DI);
    return true;
  }
  case Intrinsic::dbg_value: {
    // This form of DBG_VALUE is target-independent.
    const DbgValueInst *DI = cast<DbgValueInst>(Call);
    const MCInstrDesc &II = TII.get(TargetOpcode::DBG_VALUE);
    const Value *V = DI->getValue();
    if (!V) {
      // Currently the optimizer can produce this; insert an undef to
      // help debugging.  Probably the optimizer should not do this.
      BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
        .addReg(0U).addImm(DI->getOffset())
        .addMetadata(DI->getVariable());
    } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      if (CI->getBitWidth() > 64)
        BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
          .addCImm(CI).addImm(DI->getOffset())
          .addMetadata(DI->getVariable());
      else
        BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
          .addImm(CI->getZExtValue()).addImm(DI->getOffset())
          .addMetadata(DI->getVariable());
    } else if (const ConstantFP *CF = dyn_cast<ConstantFP>(V)) {
      BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
        .addFPImm(CF).addImm(DI->getOffset())
        .addMetadata(DI->getVariable());
    } else if (unsigned Reg = lookUpRegForValue(V)) {
      BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
        .addReg(Reg, RegState::Debug).addImm(DI->getOffset())
        .addMetadata(DI->getVariable());
    } else {
      // We can't yet handle anything else here because it would require
      // generating code, thus altering codegen because of debug info.
      DEBUG(dbgs() << "Dropping debug info for " << DI);
    }
    return true;
  }
  case Intrinsic::objectsize: {
    ConstantInt *CI = cast<ConstantInt>(Call->getArgOperand(1));
    unsigned long long Res = CI->isZero() ? -1ULL : 0;
    Constant *ResCI = ConstantInt::get(Call->getType(), Res);
    unsigned ResultReg = getRegForValue(ResCI);
    if (ResultReg == 0)
      return false;
    UpdateValueMap(Call, ResultReg);
    return true;
  }
  }

  // Usually, it does not make sense to initialize a value,
  // make an unrelated function call and use the value, because
  // it tends to be spilled on the stack. So, we move the pointer
  // to the last local value to the beginning of the block, so that
  // all the values which have already been materialized,
  // appear after the call. It also makes sense to skip intrinsics
  // since they tend to be inlined.
  if (!isa<IntrinsicInst>(F))
    flushLocalValueMap();

  // An arbitrary call. Bail.
  return false;
}

bool FastISel::SelectCast(const User *I, unsigned Opcode) {
  EVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
  EVT DstVT = TLI.getValueType(I->getType());

  if (SrcVT == MVT::Other || !SrcVT.isSimple() ||
      DstVT == MVT::Other || !DstVT.isSimple())
    // Unhandled type. Halt "fast" selection and bail.
    return false;

  // Check if the destination type is legal.
  if (!TLI.isTypeLegal(DstVT))
    return false;

  // Check if the source operand is legal.
  if (!TLI.isTypeLegal(SrcVT))
    return false;

  unsigned InputReg = getRegForValue(I->getOperand(0));
  if (!InputReg)
    // Unhandled operand.  Halt "fast" selection and bail.
    return false;

  bool InputRegIsKill = hasTrivialKill(I->getOperand(0));

  unsigned ResultReg = FastEmit_r(SrcVT.getSimpleVT(),
                                  DstVT.getSimpleVT(),
                                  Opcode,
                                  InputReg, InputRegIsKill);
  if (!ResultReg)
    return false;

  UpdateValueMap(I, ResultReg);
  return true;
}

bool FastISel::SelectBitCast(const User *I) {
  // If the bitcast doesn't change the type, just use the operand value.
  if (I->getType() == I->getOperand(0)->getType()) {
    unsigned Reg = getRegForValue(I->getOperand(0));
    if (Reg == 0)
      return false;
    UpdateValueMap(I, Reg);
    return true;
  }

  // Bitcasts of other values become reg-reg copies or BITCAST operators.
  EVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
  EVT DstVT = TLI.getValueType(I->getType());

  if (SrcVT == MVT::Other || !SrcVT.isSimple() ||
      DstVT == MVT::Other || !DstVT.isSimple() ||
      !TLI.isTypeLegal(SrcVT) || !TLI.isTypeLegal(DstVT))
    // Unhandled type. Halt "fast" selection and bail.
    return false;

  unsigned Op0 = getRegForValue(I->getOperand(0));
  if (Op0 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  bool Op0IsKill = hasTrivialKill(I->getOperand(0));

  // First, try to perform the bitcast by inserting a reg-reg copy.
  unsigned ResultReg = 0;
  if (SrcVT.getSimpleVT() == DstVT.getSimpleVT()) {
    const TargetRegisterClass* SrcClass = TLI.getRegClassFor(SrcVT);
    const TargetRegisterClass* DstClass = TLI.getRegClassFor(DstVT);
    // Don't attempt a cross-class copy. It will likely fail.
    if (SrcClass == DstClass) {
      ResultReg = createResultReg(DstClass);
      BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
              ResultReg).addReg(Op0);
    }
  }

  // If the reg-reg copy failed, select a BITCAST opcode.
  if (!ResultReg)
    ResultReg = FastEmit_r(SrcVT.getSimpleVT(), DstVT.getSimpleVT(),
                           ISD::BITCAST, Op0, Op0IsKill);

  if (!ResultReg)
    return false;

  UpdateValueMap(I, ResultReg);
  return true;
}

bool
FastISel::SelectInstruction(const Instruction *I) {
  // Just before the terminator instruction, insert instructions to
  // feed PHI nodes in successor blocks.
  if (isa<TerminatorInst>(I))
    if (!HandlePHINodesInSuccessorBlocks(I->getParent()))
      return false;

  DL = I->getDebugLoc();

  MachineBasicBlock::iterator SavedInsertPt = FuncInfo.InsertPt;

  // As a special case, don't handle calls to builtin library functions that
  // may be translated directly to target instructions.
  if (const CallInst *Call = dyn_cast<CallInst>(I)) {
    const Function *F = Call->getCalledFunction();
    LibFunc::Func Func;
    if (F && !F->hasLocalLinkage() && F->hasName() &&
        LibInfo->getLibFunc(F->getName(), Func) &&
        LibInfo->hasOptimizedCodeGen(Func))
      return false;
  }

  // First, try doing target-independent selection.
  if (SelectOperator(I, I->getOpcode())) {
    ++NumFastIselSuccessIndependent;
    DL = DebugLoc();
    return true;
  }
  // Remove dead code.  However, ignore call instructions since we've flushed
  // the local value map and recomputed the insert point.
  if (!isa<CallInst>(I)) {
    recomputeInsertPt();
    if (SavedInsertPt != FuncInfo.InsertPt)
      removeDeadCode(FuncInfo.InsertPt, SavedInsertPt);
  }

  // Next, try calling the target to attempt to handle the instruction.
  SavedInsertPt = FuncInfo.InsertPt;
  if (TargetSelectInstruction(I)) {
    ++NumFastIselSuccessTarget;
    DL = DebugLoc();
    return true;
  }
  // Check for dead code and remove as necessary.
  recomputeInsertPt();
  if (SavedInsertPt != FuncInfo.InsertPt)
    removeDeadCode(FuncInfo.InsertPt, SavedInsertPt);

  DL = DebugLoc();
  return false;
}

/// FastEmitBranch - Emit an unconditional branch to the given block,
/// unless it is the immediate (fall-through) successor, and update
/// the CFG.
void
FastISel::FastEmitBranch(MachineBasicBlock *MSucc, DebugLoc DL) {

  if (FuncInfo.MBB->getBasicBlock()->size() > 1 && FuncInfo.MBB->isLayoutSuccessor(MSucc)) {
    // For more accurate line information if this is the only instruction
    // in the block then emit it, otherwise we have the unconditional
    // fall-through case, which needs no instructions.
  } else {
    // The unconditional branch case.
    TII.InsertBranch(*FuncInfo.MBB, MSucc, NULL,
                     SmallVector<MachineOperand, 0>(), DL);
  }
  FuncInfo.MBB->addSuccessor(MSucc);
}

/// SelectFNeg - Emit an FNeg operation.
///
bool
FastISel::SelectFNeg(const User *I) {
  unsigned OpReg = getRegForValue(BinaryOperator::getFNegArgument(I));
  if (OpReg == 0) return false;

  bool OpRegIsKill = hasTrivialKill(I);

  // If the target has ISD::FNEG, use it.
  EVT VT = TLI.getValueType(I->getType());
  unsigned ResultReg = FastEmit_r(VT.getSimpleVT(), VT.getSimpleVT(),
                                  ISD::FNEG, OpReg, OpRegIsKill);
  if (ResultReg != 0) {
    UpdateValueMap(I, ResultReg);
    return true;
  }

  // Bitcast the value to integer, twiddle the sign bit with xor,
  // and then bitcast it back to floating-point.
  if (VT.getSizeInBits() > 64) return false;
  EVT IntVT = EVT::getIntegerVT(I->getContext(), VT.getSizeInBits());
  if (!TLI.isTypeLegal(IntVT))
    return false;

  unsigned IntReg = FastEmit_r(VT.getSimpleVT(), IntVT.getSimpleVT(),
                               ISD::BITCAST, OpReg, OpRegIsKill);
  if (IntReg == 0)
    return false;

  unsigned IntResultReg = FastEmit_ri_(IntVT.getSimpleVT(), ISD::XOR,
                                       IntReg, /*Kill=*/true,
                                       UINT64_C(1) << (VT.getSizeInBits()-1),
                                       IntVT.getSimpleVT());
  if (IntResultReg == 0)
    return false;

  ResultReg = FastEmit_r(IntVT.getSimpleVT(), VT.getSimpleVT(),
                         ISD::BITCAST, IntResultReg, /*Kill=*/true);
  if (ResultReg == 0)
    return false;

  UpdateValueMap(I, ResultReg);
  return true;
}

bool
FastISel::SelectExtractValue(const User *U) {
  const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(U);
  if (!EVI)
    return false;

  // Make sure we only try to handle extracts with a legal result.  But also
  // allow i1 because it's easy.
  EVT RealVT = TLI.getValueType(EVI->getType(), /*AllowUnknown=*/true);
  if (!RealVT.isSimple())
    return false;
  MVT VT = RealVT.getSimpleVT();
  if (!TLI.isTypeLegal(VT) && VT != MVT::i1)
    return false;

  const Value *Op0 = EVI->getOperand(0);
  Type *AggTy = Op0->getType();

  // Get the base result register.
  unsigned ResultReg;
  DenseMap<const Value *, unsigned>::iterator I = FuncInfo.ValueMap.find(Op0);
  if (I != FuncInfo.ValueMap.end())
    ResultReg = I->second;
  else if (isa<Instruction>(Op0))
    ResultReg = FuncInfo.InitializeRegForValue(Op0);
  else
    return false; // fast-isel can't handle aggregate constants at the moment

  // Get the actual result register, which is an offset from the base register.
  unsigned VTIndex = ComputeLinearIndex(AggTy, EVI->getIndices());

  SmallVector<EVT, 4> AggValueVTs;
  ComputeValueVTs(TLI, AggTy, AggValueVTs);

  for (unsigned i = 0; i < VTIndex; i++)
    ResultReg += TLI.getNumRegisters(FuncInfo.Fn->getContext(), AggValueVTs[i]);

  UpdateValueMap(EVI, ResultReg);
  return true;
}

bool
FastISel::SelectOperator(const User *I, unsigned Opcode) {
  switch (Opcode) {
  case Instruction::Add:
    return SelectBinaryOp(I, ISD::ADD);
  case Instruction::FAdd:
    return SelectBinaryOp(I, ISD::FADD);
  case Instruction::Sub:
    return SelectBinaryOp(I, ISD::SUB);
  case Instruction::FSub:
    // FNeg is currently represented in LLVM IR as a special case of FSub.
    if (BinaryOperator::isFNeg(I))
      return SelectFNeg(I);
    return SelectBinaryOp(I, ISD::FSUB);
  case Instruction::Mul:
    return SelectBinaryOp(I, ISD::MUL);
  case Instruction::FMul:
    return SelectBinaryOp(I, ISD::FMUL);
  case Instruction::SDiv:
    return SelectBinaryOp(I, ISD::SDIV);
  case Instruction::UDiv:
    return SelectBinaryOp(I, ISD::UDIV);
  case Instruction::FDiv:
    return SelectBinaryOp(I, ISD::FDIV);
  case Instruction::SRem:
    return SelectBinaryOp(I, ISD::SREM);
  case Instruction::URem:
    return SelectBinaryOp(I, ISD::UREM);
  case Instruction::FRem:
    return SelectBinaryOp(I, ISD::FREM);
  case Instruction::Shl:
    return SelectBinaryOp(I, ISD::SHL);
  case Instruction::LShr:
    return SelectBinaryOp(I, ISD::SRL);
  case Instruction::AShr:
    return SelectBinaryOp(I, ISD::SRA);
  case Instruction::And:
    return SelectBinaryOp(I, ISD::AND);
  case Instruction::Or:
    return SelectBinaryOp(I, ISD::OR);
  case Instruction::Xor:
    return SelectBinaryOp(I, ISD::XOR);

  case Instruction::GetElementPtr:
    return SelectGetElementPtr(I);

  case Instruction::Br: {
    const BranchInst *BI = cast<BranchInst>(I);

    if (BI->isUnconditional()) {
      const BasicBlock *LLVMSucc = BI->getSuccessor(0);
      MachineBasicBlock *MSucc = FuncInfo.MBBMap[LLVMSucc];
      FastEmitBranch(MSucc, BI->getDebugLoc());
      return true;
    }

    // Conditional branches are not handed yet.
    // Halt "fast" selection and bail.
    return false;
  }

  case Instruction::Unreachable:
    // Nothing to emit.
    return true;

  case Instruction::Alloca:
    // FunctionLowering has the static-sized case covered.
    if (FuncInfo.StaticAllocaMap.count(cast<AllocaInst>(I)))
      return true;

    // Dynamic-sized alloca is not handled yet.
    return false;

  case Instruction::Call:
    return SelectCall(I);

  case Instruction::BitCast:
    return SelectBitCast(I);

  case Instruction::FPToSI:
    return SelectCast(I, ISD::FP_TO_SINT);
  case Instruction::ZExt:
    return SelectCast(I, ISD::ZERO_EXTEND);
  case Instruction::SExt:
    return SelectCast(I, ISD::SIGN_EXTEND);
  case Instruction::Trunc:
    return SelectCast(I, ISD::TRUNCATE);
  case Instruction::SIToFP:
    return SelectCast(I, ISD::SINT_TO_FP);

  case Instruction::IntToPtr: // Deliberate fall-through.
  case Instruction::PtrToInt: {
    EVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
    EVT DstVT = TLI.getValueType(I->getType());
    if (DstVT.bitsGT(SrcVT))
      return SelectCast(I, ISD::ZERO_EXTEND);
    if (DstVT.bitsLT(SrcVT))
      return SelectCast(I, ISD::TRUNCATE);
    unsigned Reg = getRegForValue(I->getOperand(0));
    if (Reg == 0) return false;
    UpdateValueMap(I, Reg);
    return true;
  }

  case Instruction::ExtractValue:
    return SelectExtractValue(I);

  case Instruction::PHI:
    llvm_unreachable("FastISel shouldn't visit PHI nodes!");

  default:
    // Unhandled instruction. Halt "fast" selection and bail.
    return false;
  }
}

FastISel::FastISel(FunctionLoweringInfo &funcInfo,
                   const TargetLibraryInfo *libInfo)
  : FuncInfo(funcInfo),
    MRI(FuncInfo.MF->getRegInfo()),
    MFI(*FuncInfo.MF->getFrameInfo()),
    MCP(*FuncInfo.MF->getConstantPool()),
    TM(FuncInfo.MF->getTarget()),
    TD(*TM.getDataLayout()),
    TII(*TM.getInstrInfo()),
    TLI(*TM.getTargetLowering()),
    TRI(*TM.getRegisterInfo()),
    LibInfo(libInfo) {
}

FastISel::~FastISel() {}

unsigned FastISel::FastEmit_(MVT, MVT,
                             unsigned) {
  return 0;
}

unsigned FastISel::FastEmit_r(MVT, MVT,
                              unsigned,
                              unsigned /*Op0*/, bool /*Op0IsKill*/) {
  return 0;
}

unsigned FastISel::FastEmit_rr(MVT, MVT,
                               unsigned,
                               unsigned /*Op0*/, bool /*Op0IsKill*/,
                               unsigned /*Op1*/, bool /*Op1IsKill*/) {
  return 0;
}

unsigned FastISel::FastEmit_i(MVT, MVT, unsigned, uint64_t /*Imm*/) {
  return 0;
}

unsigned FastISel::FastEmit_f(MVT, MVT,
                              unsigned, const ConstantFP * /*FPImm*/) {
  return 0;
}

unsigned FastISel::FastEmit_ri(MVT, MVT,
                               unsigned,
                               unsigned /*Op0*/, bool /*Op0IsKill*/,
                               uint64_t /*Imm*/) {
  return 0;
}

unsigned FastISel::FastEmit_rf(MVT, MVT,
                               unsigned,
                               unsigned /*Op0*/, bool /*Op0IsKill*/,
                               const ConstantFP * /*FPImm*/) {
  return 0;
}

unsigned FastISel::FastEmit_rri(MVT, MVT,
                                unsigned,
                                unsigned /*Op0*/, bool /*Op0IsKill*/,
                                unsigned /*Op1*/, bool /*Op1IsKill*/,
                                uint64_t /*Imm*/) {
  return 0;
}

/// FastEmit_ri_ - This method is a wrapper of FastEmit_ri. It first tries
/// to emit an instruction with an immediate operand using FastEmit_ri.
/// If that fails, it materializes the immediate into a register and try
/// FastEmit_rr instead.
unsigned FastISel::FastEmit_ri_(MVT VT, unsigned Opcode,
                                unsigned Op0, bool Op0IsKill,
                                uint64_t Imm, MVT ImmType) {
  // If this is a multiply by a power of two, emit this as a shift left.
  if (Opcode == ISD::MUL && isPowerOf2_64(Imm)) {
    Opcode = ISD::SHL;
    Imm = Log2_64(Imm);
  } else if (Opcode == ISD::UDIV && isPowerOf2_64(Imm)) {
    // div x, 8 -> srl x, 3
    Opcode = ISD::SRL;
    Imm = Log2_64(Imm);
  }

  // Horrible hack (to be removed), check to make sure shift amounts are
  // in-range.
  if ((Opcode == ISD::SHL || Opcode == ISD::SRA || Opcode == ISD::SRL) &&
      Imm >= VT.getSizeInBits())
    return 0;

  // First check if immediate type is legal. If not, we can't use the ri form.
  unsigned ResultReg = FastEmit_ri(VT, VT, Opcode, Op0, Op0IsKill, Imm);
  if (ResultReg != 0)
    return ResultReg;
  unsigned MaterialReg = FastEmit_i(ImmType, ImmType, ISD::Constant, Imm);
  if (MaterialReg == 0) {
    // This is a bit ugly/slow, but failing here means falling out of
    // fast-isel, which would be very slow.
    IntegerType *ITy = IntegerType::get(FuncInfo.Fn->getContext(),
                                              VT.getSizeInBits());
    MaterialReg = getRegForValue(ConstantInt::get(ITy, Imm));
  }
  return FastEmit_rr(VT, VT, Opcode,
                     Op0, Op0IsKill,
                     MaterialReg, /*Kill=*/true);
}

unsigned FastISel::createResultReg(const TargetRegisterClass* RC) {
  return MRI.createVirtualRegister(RC);
}

unsigned FastISel::FastEmitInst_(unsigned MachineInstOpcode,
                                 const TargetRegisterClass* RC) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_r(unsigned MachineInstOpcode,
                                  const TargetRegisterClass *RC,
                                  unsigned Op0, bool Op0IsKill) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }

  return ResultReg;
}

unsigned FastISel::FastEmitInst_rr(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, bool Op0IsKill,
                                   unsigned Op1, bool Op1IsKill) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rrr(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, bool Op0IsKill,
                                   unsigned Op1, bool Op1IsKill,
                                   unsigned Op2, bool Op2IsKill) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill)
      .addReg(Op2, Op2IsKill * RegState::Kill);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill)
      .addReg(Op2, Op2IsKill * RegState::Kill);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_ri(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, bool Op0IsKill,
                                   uint64_t Imm) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addImm(Imm);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addImm(Imm);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rii(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, bool Op0IsKill,
                                   uint64_t Imm1, uint64_t Imm2) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addImm(Imm1)
      .addImm(Imm2);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addImm(Imm1)
      .addImm(Imm2);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rf(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, bool Op0IsKill,
                                   const ConstantFP *FPImm) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addFPImm(FPImm);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addFPImm(FPImm);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rri(unsigned MachineInstOpcode,
                                    const TargetRegisterClass *RC,
                                    unsigned Op0, bool Op0IsKill,
                                    unsigned Op1, bool Op1IsKill,
                                    uint64_t Imm) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill)
      .addImm(Imm);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill)
      .addImm(Imm);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rrii(unsigned MachineInstOpcode,
                                     const TargetRegisterClass *RC,
                                     unsigned Op0, bool Op0IsKill,
                                     unsigned Op1, bool Op1IsKill,
                                     uint64_t Imm1, uint64_t Imm2) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill)
      .addImm(Imm1).addImm(Imm2);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill)
      .addImm(Imm1).addImm(Imm2);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_i(unsigned MachineInstOpcode,
                                  const TargetRegisterClass *RC,
                                  uint64_t Imm) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg).addImm(Imm);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II).addImm(Imm);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_ii(unsigned MachineInstOpcode,
                                  const TargetRegisterClass *RC,
                                  uint64_t Imm1, uint64_t Imm2) {
  unsigned ResultReg = createResultReg(RC);
  const MCInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II, ResultReg)
      .addImm(Imm1).addImm(Imm2);
  else {
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, II).addImm(Imm1).addImm(Imm2);
    BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, TII.get(TargetOpcode::COPY),
            ResultReg).addReg(II.ImplicitDefs[0]);
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_extractsubreg(MVT RetVT,
                                              unsigned Op0, bool Op0IsKill,
                                              uint32_t Idx) {
  unsigned ResultReg = createResultReg(TLI.getRegClassFor(RetVT));
  assert(TargetRegisterInfo::isVirtualRegister(Op0) &&
         "Cannot yet extract from physregs");
  const TargetRegisterClass *RC = MRI.getRegClass(Op0);
  MRI.constrainRegClass(Op0, TRI.getSubClassWithSubReg(RC, Idx));
  BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt,
          DL, TII.get(TargetOpcode::COPY), ResultReg)
    .addReg(Op0, getKillRegState(Op0IsKill), Idx);
  return ResultReg;
}

/// FastEmitZExtFromI1 - Emit MachineInstrs to compute the value of Op
/// with all but the least significant bit set to zero.
unsigned FastISel::FastEmitZExtFromI1(MVT VT, unsigned Op0, bool Op0IsKill) {
  return FastEmit_ri(VT, VT, ISD::AND, Op0, Op0IsKill, 1);
}

/// HandlePHINodesInSuccessorBlocks - Handle PHI nodes in successor blocks.
/// Emit code to ensure constants are copied into registers when needed.
/// Remember the virtual registers that need to be added to the Machine PHI
/// nodes as input.  We cannot just directly add them, because expansion
/// might result in multiple MBB's for one BB.  As such, the start of the
/// BB might correspond to a different MBB than the end.
bool FastISel::HandlePHINodesInSuccessorBlocks(const BasicBlock *LLVMBB) {
  const TerminatorInst *TI = LLVMBB->getTerminator();

  SmallPtrSet<MachineBasicBlock *, 4> SuccsHandled;
  unsigned OrigNumPHINodesToUpdate = FuncInfo.PHINodesToUpdate.size();

  // Check successor nodes' PHI nodes that expect a constant to be available
  // from this block.
  for (unsigned succ = 0, e = TI->getNumSuccessors(); succ != e; ++succ) {
    const BasicBlock *SuccBB = TI->getSuccessor(succ);
    if (!isa<PHINode>(SuccBB->begin())) continue;
    MachineBasicBlock *SuccMBB = FuncInfo.MBBMap[SuccBB];

    // If this terminator has multiple identical successors (common for
    // switches), only handle each succ once.
    if (!SuccsHandled.insert(SuccMBB)) continue;

    MachineBasicBlock::iterator MBBI = SuccMBB->begin();

    // At this point we know that there is a 1-1 correspondence between LLVM PHI
    // nodes and Machine PHI nodes, but the incoming operands have not been
    // emitted yet.
    for (BasicBlock::const_iterator I = SuccBB->begin();
         const PHINode *PN = dyn_cast<PHINode>(I); ++I) {

      // Ignore dead phi's.
      if (PN->use_empty()) continue;

      // Only handle legal types. Two interesting things to note here. First,
      // by bailing out early, we may leave behind some dead instructions,
      // since SelectionDAG's HandlePHINodesInSuccessorBlocks will insert its
      // own moves. Second, this check is necessary because FastISel doesn't
      // use CreateRegs to create registers, so it always creates
      // exactly one register for each non-void instruction.
      EVT VT = TLI.getValueType(PN->getType(), /*AllowUnknown=*/true);
      if (VT == MVT::Other || !TLI.isTypeLegal(VT)) {
        // Handle integer promotions, though, because they're common and easy.
        if (VT == MVT::i1 || VT == MVT::i8 || VT == MVT::i16)
          VT = TLI.getTypeToTransformTo(LLVMBB->getContext(), VT);
        else {
          FuncInfo.PHINodesToUpdate.resize(OrigNumPHINodesToUpdate);
          return false;
        }
      }

      const Value *PHIOp = PN->getIncomingValueForBlock(LLVMBB);

      // Set the DebugLoc for the copy. Prefer the location of the operand
      // if there is one; use the location of the PHI otherwise.
      DL = PN->getDebugLoc();
      if (const Instruction *Inst = dyn_cast<Instruction>(PHIOp))
        DL = Inst->getDebugLoc();

      unsigned Reg = getRegForValue(PHIOp);
      if (Reg == 0) {
        FuncInfo.PHINodesToUpdate.resize(OrigNumPHINodesToUpdate);
        return false;
      }
      FuncInfo.PHINodesToUpdate.push_back(std::make_pair(MBBI++, Reg));
      DL = DebugLoc();
    }
  }

  return true;
}
