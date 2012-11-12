//===- AddrModeMatcher.cpp - Addressing mode matching facility --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements target addressing mode matcher class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AddrModeMatcher.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Instruction.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/DataLayout.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CallSite.h"

using namespace llvm;
using namespace llvm::PatternMatch;

void ExtAddrMode::print(raw_ostream &OS) const {
  bool NeedPlus = false;
  OS << "[";
  if (BaseGV) {
    OS << (NeedPlus ? " + " : "")
       << "GV:";
    WriteAsOperand(OS, BaseGV, /*PrintType=*/false);
    NeedPlus = true;
  }

  if (BaseOffs)
    OS << (NeedPlus ? " + " : "") << BaseOffs, NeedPlus = true;

  if (BaseReg) {
    OS << (NeedPlus ? " + " : "")
       << "Base:";
    WriteAsOperand(OS, BaseReg, /*PrintType=*/false);
    NeedPlus = true;
  }
  if (Scale) {
    OS << (NeedPlus ? " + " : "")
       << Scale << "*";
    WriteAsOperand(OS, ScaledReg, /*PrintType=*/false);
    NeedPlus = true;
  }

  OS << ']';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void ExtAddrMode::dump() const {
  print(dbgs());
  dbgs() << '\n';
}
#endif


/// MatchScaledValue - Try adding ScaleReg*Scale to the current addressing mode.
/// Return true and update AddrMode if this addr mode is legal for the target,
/// false if not.
bool AddressingModeMatcher::MatchScaledValue(Value *ScaleReg, int64_t Scale,
                                             unsigned Depth) {
  // If Scale is 1, then this is the same as adding ScaleReg to the addressing
  // mode.  Just process that directly.
  if (Scale == 1)
    return MatchAddr(ScaleReg, Depth);
  
  // If the scale is 0, it takes nothing to add this.
  if (Scale == 0)
    return true;
  
  // If we already have a scale of this value, we can add to it, otherwise, we
  // need an available scale field.
  if (AddrMode.Scale != 0 && AddrMode.ScaledReg != ScaleReg)
    return false;

  ExtAddrMode TestAddrMode = AddrMode;

  // Add scale to turn X*4+X*3 -> X*7.  This could also do things like
  // [A+B + A*7] -> [B+A*8].
  TestAddrMode.Scale += Scale;
  TestAddrMode.ScaledReg = ScaleReg;

  // If the new address isn't legal, bail out.
  if (!TLI.isLegalAddressingMode(TestAddrMode, AccessTy))
    return false;

  // It was legal, so commit it.
  AddrMode = TestAddrMode;
  
  // Okay, we decided that we can add ScaleReg+Scale to AddrMode.  Check now
  // to see if ScaleReg is actually X+C.  If so, we can turn this into adding
  // X*Scale + C*Scale to addr mode.
  ConstantInt *CI = 0; Value *AddLHS = 0;
  if (isa<Instruction>(ScaleReg) &&  // not a constant expr.
      match(ScaleReg, m_Add(m_Value(AddLHS), m_ConstantInt(CI)))) {
    TestAddrMode.ScaledReg = AddLHS;
    TestAddrMode.BaseOffs += CI->getSExtValue()*TestAddrMode.Scale;
      
    // If this addressing mode is legal, commit it and remember that we folded
    // this instruction.
    if (TLI.isLegalAddressingMode(TestAddrMode, AccessTy)) {
      AddrModeInsts.push_back(cast<Instruction>(ScaleReg));
      AddrMode = TestAddrMode;
      return true;
    }
  }

  // Otherwise, not (x+c)*scale, just return what we have.
  return true;
}

/// MightBeFoldableInst - This is a little filter, which returns true if an
/// addressing computation involving I might be folded into a load/store
/// accessing it.  This doesn't need to be perfect, but needs to accept at least
/// the set of instructions that MatchOperationAddr can.
static bool MightBeFoldableInst(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::BitCast:
    // Don't touch identity bitcasts.
    if (I->getType() == I->getOperand(0)->getType())
      return false;
    return I->getType()->isPointerTy() || I->getType()->isIntegerTy();
  case Instruction::PtrToInt:
    // PtrToInt is always a noop, as we know that the int type is pointer sized.
    return true;
  case Instruction::IntToPtr:
    // We know the input is intptr_t, so this is foldable.
    return true;
  case Instruction::Add:
    return true;
  case Instruction::Mul:
  case Instruction::Shl:
    // Can only handle X*C and X << C.
    return isa<ConstantInt>(I->getOperand(1));
  case Instruction::GetElementPtr:
    return true;
  default:
    return false;
  }
}


/// MatchOperationAddr - Given an instruction or constant expr, see if we can
/// fold the operation into the addressing mode.  If so, update the addressing
/// mode and return true, otherwise return false without modifying AddrMode.
bool AddressingModeMatcher::MatchOperationAddr(User *AddrInst, unsigned Opcode,
                                               unsigned Depth) {
  // Avoid exponential behavior on extremely deep expression trees.
  if (Depth >= 5) return false;
  
  switch (Opcode) {
  case Instruction::PtrToInt:
    // PtrToInt is always a noop, as we know that the int type is pointer sized.
    return MatchAddr(AddrInst->getOperand(0), Depth);
  case Instruction::IntToPtr:
    // This inttoptr is a no-op if the integer type is pointer sized.
    if (TLI.getValueType(AddrInst->getOperand(0)->getType()) ==
        TLI.getPointerTy())
      return MatchAddr(AddrInst->getOperand(0), Depth);
    return false;
  case Instruction::BitCast:
    // BitCast is always a noop, and we can handle it as long as it is
    // int->int or pointer->pointer (we don't want int<->fp or something).
    if ((AddrInst->getOperand(0)->getType()->isPointerTy() ||
         AddrInst->getOperand(0)->getType()->isIntegerTy()) &&
        // Don't touch identity bitcasts.  These were probably put here by LSR,
        // and we don't want to mess around with them.  Assume it knows what it
        // is doing.
        AddrInst->getOperand(0)->getType() != AddrInst->getType())
      return MatchAddr(AddrInst->getOperand(0), Depth);
    return false;
  case Instruction::Add: {
    // Check to see if we can merge in the RHS then the LHS.  If so, we win.
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();
    if (MatchAddr(AddrInst->getOperand(1), Depth+1) &&
        MatchAddr(AddrInst->getOperand(0), Depth+1))
      return true;
    
    // Restore the old addr mode info.
    AddrMode = BackupAddrMode;
    AddrModeInsts.resize(OldSize);
    
    // Otherwise this was over-aggressive.  Try merging in the LHS then the RHS.
    if (MatchAddr(AddrInst->getOperand(0), Depth+1) &&
        MatchAddr(AddrInst->getOperand(1), Depth+1))
      return true;
    
    // Otherwise we definitely can't merge the ADD in.
    AddrMode = BackupAddrMode;
    AddrModeInsts.resize(OldSize);
    break;
  }
  //case Instruction::Or:
  // TODO: We can handle "Or Val, Imm" iff this OR is equivalent to an ADD.
  //break;
  case Instruction::Mul:
  case Instruction::Shl: {
    // Can only handle X*C and X << C.
    ConstantInt *RHS = dyn_cast<ConstantInt>(AddrInst->getOperand(1));
    if (!RHS) return false;
    int64_t Scale = RHS->getSExtValue();
    if (Opcode == Instruction::Shl)
      Scale = 1LL << Scale;
    
    return MatchScaledValue(AddrInst->getOperand(0), Scale, Depth);
  }
  case Instruction::GetElementPtr: {
    // Scan the GEP.  We check it if it contains constant offsets and at most
    // one variable offset.
    int VariableOperand = -1;
    unsigned VariableScale = 0;
    
    int64_t ConstantOffset = 0;
    const DataLayout *TD = TLI.getDataLayout();
    gep_type_iterator GTI = gep_type_begin(AddrInst);
    for (unsigned i = 1, e = AddrInst->getNumOperands(); i != e; ++i, ++GTI) {
      if (StructType *STy = dyn_cast<StructType>(*GTI)) {
        const StructLayout *SL = TD->getStructLayout(STy);
        unsigned Idx =
          cast<ConstantInt>(AddrInst->getOperand(i))->getZExtValue();
        ConstantOffset += SL->getElementOffset(Idx);
      } else {
        uint64_t TypeSize = TD->getTypeAllocSize(GTI.getIndexedType());
        if (ConstantInt *CI = dyn_cast<ConstantInt>(AddrInst->getOperand(i))) {
          ConstantOffset += CI->getSExtValue()*TypeSize;
        } else if (TypeSize) {  // Scales of zero don't do anything.
          // We only allow one variable index at the moment.
          if (VariableOperand != -1)
            return false;
          
          // Remember the variable index.
          VariableOperand = i;
          VariableScale = TypeSize;
        }
      }
    }
    
    // A common case is for the GEP to only do a constant offset.  In this case,
    // just add it to the disp field and check validity.
    if (VariableOperand == -1) {
      AddrMode.BaseOffs += ConstantOffset;
      if (ConstantOffset == 0 || TLI.isLegalAddressingMode(AddrMode, AccessTy)){
        // Check to see if we can fold the base pointer in too.
        if (MatchAddr(AddrInst->getOperand(0), Depth+1))
          return true;
      }
      AddrMode.BaseOffs -= ConstantOffset;
      return false;
    }

    // Save the valid addressing mode in case we can't match.
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();

    // See if the scale and offset amount is valid for this target.
    AddrMode.BaseOffs += ConstantOffset;

    // Match the base operand of the GEP.
    if (!MatchAddr(AddrInst->getOperand(0), Depth+1)) {
      // If it couldn't be matched, just stuff the value in a register.
      if (AddrMode.HasBaseReg) {
        AddrMode = BackupAddrMode;
        AddrModeInsts.resize(OldSize);
        return false;
      }
      AddrMode.HasBaseReg = true;
      AddrMode.BaseReg = AddrInst->getOperand(0);
    }

    // Match the remaining variable portion of the GEP.
    if (!MatchScaledValue(AddrInst->getOperand(VariableOperand), VariableScale,
                          Depth)) {
      // If it couldn't be matched, try stuffing the base into a register
      // instead of matching it, and retrying the match of the scale.
      AddrMode = BackupAddrMode;
      AddrModeInsts.resize(OldSize);
      if (AddrMode.HasBaseReg)
        return false;
      AddrMode.HasBaseReg = true;
      AddrMode.BaseReg = AddrInst->getOperand(0);
      AddrMode.BaseOffs += ConstantOffset;
      if (!MatchScaledValue(AddrInst->getOperand(VariableOperand),
                            VariableScale, Depth)) {
        // If even that didn't work, bail.
        AddrMode = BackupAddrMode;
        AddrModeInsts.resize(OldSize);
        return false;
      }
    }

    return true;
  }
  }
  return false;
}

/// MatchAddr - If we can, try to add the value of 'Addr' into the current
/// addressing mode.  If Addr can't be added to AddrMode this returns false and
/// leaves AddrMode unmodified.  This assumes that Addr is either a pointer type
/// or intptr_t for the target.
///
bool AddressingModeMatcher::MatchAddr(Value *Addr, unsigned Depth) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Addr)) {
    // Fold in immediates if legal for the target.
    AddrMode.BaseOffs += CI->getSExtValue();
    if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
      return true;
    AddrMode.BaseOffs -= CI->getSExtValue();
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(Addr)) {
    // If this is a global variable, try to fold it into the addressing mode.
    if (AddrMode.BaseGV == 0) {
      AddrMode.BaseGV = GV;
      if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
        return true;
      AddrMode.BaseGV = 0;
    }
  } else if (Instruction *I = dyn_cast<Instruction>(Addr)) {
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();

    // Check to see if it is possible to fold this operation.
    if (MatchOperationAddr(I, I->getOpcode(), Depth)) {
      // Okay, it's possible to fold this.  Check to see if it is actually
      // *profitable* to do so.  We use a simple cost model to avoid increasing
      // register pressure too much.
      if (I->hasOneUse() ||
          IsProfitableToFoldIntoAddressingMode(I, BackupAddrMode, AddrMode)) {
        AddrModeInsts.push_back(I);
        return true;
      }
      
      // It isn't profitable to do this, roll back.
      //cerr << "NOT FOLDING: " << *I;
      AddrMode = BackupAddrMode;
      AddrModeInsts.resize(OldSize);
    }
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Addr)) {
    if (MatchOperationAddr(CE, CE->getOpcode(), Depth))
      return true;
  } else if (isa<ConstantPointerNull>(Addr)) {
    // Null pointer gets folded without affecting the addressing mode.
    return true;
  }

  // Worse case, the target should support [reg] addressing modes. :)
  if (!AddrMode.HasBaseReg) {
    AddrMode.HasBaseReg = true;
    AddrMode.BaseReg = Addr;
    // Still check for legality in case the target supports [imm] but not [i+r].
    if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
      return true;
    AddrMode.HasBaseReg = false;
    AddrMode.BaseReg = 0;
  }

  // If the base register is already taken, see if we can do [r+r].
  if (AddrMode.Scale == 0) {
    AddrMode.Scale = 1;
    AddrMode.ScaledReg = Addr;
    if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
      return true;
    AddrMode.Scale = 0;
    AddrMode.ScaledReg = 0;
  }
  // Couldn't match.
  return false;
}


/// IsOperandAMemoryOperand - Check to see if all uses of OpVal by the specified
/// inline asm call are due to memory operands.  If so, return true, otherwise
/// return false.
static bool IsOperandAMemoryOperand(CallInst *CI, InlineAsm *IA, Value *OpVal,
                                    const TargetLowering &TLI) {
  TargetLowering::AsmOperandInfoVector TargetConstraints = TLI.ParseConstraints(ImmutableCallSite(CI));
  for (unsigned i = 0, e = TargetConstraints.size(); i != e; ++i) {
    TargetLowering::AsmOperandInfo &OpInfo = TargetConstraints[i];
    
    // Compute the constraint code and ConstraintType to use.
    TLI.ComputeConstraintToUse(OpInfo, SDValue());

    // If this asm operand is our Value*, and if it isn't an indirect memory
    // operand, we can't fold it!
    if (OpInfo.CallOperandVal == OpVal &&
        (OpInfo.ConstraintType != TargetLowering::C_Memory ||
         !OpInfo.isIndirect))
      return false;
  }

  return true;
}


/// FindAllMemoryUses - Recursively walk all the uses of I until we find a
/// memory use.  If we find an obviously non-foldable instruction, return true.
/// Add the ultimately found memory instructions to MemoryUses.
static bool FindAllMemoryUses(Instruction *I,
                SmallVectorImpl<std::pair<Instruction*,unsigned> > &MemoryUses,
                              SmallPtrSet<Instruction*, 16> &ConsideredInsts,
                              const TargetLowering &TLI) {
  // If we already considered this instruction, we're done.
  if (!ConsideredInsts.insert(I))
    return false;
  
  // If this is an obviously unfoldable instruction, bail out.
  if (!MightBeFoldableInst(I))
    return true;

  // Loop over all the uses, recursively processing them.
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    User *U = *UI;

    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      MemoryUses.push_back(std::make_pair(LI, UI.getOperandNo()));
      continue;
    }
    
    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      unsigned opNo = UI.getOperandNo();
      if (opNo == 0) return true; // Storing addr, not into addr.
      MemoryUses.push_back(std::make_pair(SI, opNo));
      continue;
    }
    
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      InlineAsm *IA = dyn_cast<InlineAsm>(CI->getCalledValue());
      if (!IA) return true;
      
      // If this is a memory operand, we're cool, otherwise bail out.
      if (!IsOperandAMemoryOperand(CI, IA, I, TLI))
        return true;
      continue;
    }
    
    if (FindAllMemoryUses(cast<Instruction>(U), MemoryUses, ConsideredInsts,
                          TLI))
      return true;
  }

  return false;
}


/// ValueAlreadyLiveAtInst - Retrn true if Val is already known to be live at
/// the use site that we're folding it into.  If so, there is no cost to
/// include it in the addressing mode.  KnownLive1 and KnownLive2 are two values
/// that we know are live at the instruction already.
bool AddressingModeMatcher::ValueAlreadyLiveAtInst(Value *Val,Value *KnownLive1,
                                                   Value *KnownLive2) {
  // If Val is either of the known-live values, we know it is live!
  if (Val == 0 || Val == KnownLive1 || Val == KnownLive2)
    return true;
  
  // All values other than instructions and arguments (e.g. constants) are live.
  if (!isa<Instruction>(Val) && !isa<Argument>(Val)) return true;
  
  // If Val is a constant sized alloca in the entry block, it is live, this is
  // true because it is just a reference to the stack/frame pointer, which is
  // live for the whole function.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Val))
    if (AI->isStaticAlloca())
      return true;
  
  // Check to see if this value is already used in the memory instruction's
  // block.  If so, it's already live into the block at the very least, so we
  // can reasonably fold it.
  return Val->isUsedInBasicBlock(MemoryInst->getParent());
}



/// IsProfitableToFoldIntoAddressingMode - It is possible for the addressing
/// mode of the machine to fold the specified instruction into a load or store
/// that ultimately uses it.  However, the specified instruction has multiple
/// uses.  Given this, it may actually increase register pressure to fold it
/// into the load.  For example, consider this code:
///
///     X = ...
///     Y = X+1
///     use(Y)   -> nonload/store
///     Z = Y+1
///     load Z
///
/// In this case, Y has multiple uses, and can be folded into the load of Z
/// (yielding load [X+2]).  However, doing this will cause both "X" and "X+1" to
/// be live at the use(Y) line.  If we don't fold Y into load Z, we use one
/// fewer register.  Since Y can't be folded into "use(Y)" we don't increase the
/// number of computations either.
///
/// Note that this (like most of CodeGenPrepare) is just a rough heuristic.  If
/// X was live across 'load Z' for other reasons, we actually *would* want to
/// fold the addressing mode in the Z case.  This would make Y die earlier.
bool AddressingModeMatcher::
IsProfitableToFoldIntoAddressingMode(Instruction *I, ExtAddrMode &AMBefore,
                                     ExtAddrMode &AMAfter) {
  if (IgnoreProfitability) return true;
  
  // AMBefore is the addressing mode before this instruction was folded into it,
  // and AMAfter is the addressing mode after the instruction was folded.  Get
  // the set of registers referenced by AMAfter and subtract out those
  // referenced by AMBefore: this is the set of values which folding in this
  // address extends the lifetime of.
  //
  // Note that there are only two potential values being referenced here,
  // BaseReg and ScaleReg (global addresses are always available, as are any
  // folded immediates).
  Value *BaseReg = AMAfter.BaseReg, *ScaledReg = AMAfter.ScaledReg;
  
  // If the BaseReg or ScaledReg was referenced by the previous addrmode, their
  // lifetime wasn't extended by adding this instruction.
  if (ValueAlreadyLiveAtInst(BaseReg, AMBefore.BaseReg, AMBefore.ScaledReg))
    BaseReg = 0;
  if (ValueAlreadyLiveAtInst(ScaledReg, AMBefore.BaseReg, AMBefore.ScaledReg))
    ScaledReg = 0;

  // If folding this instruction (and it's subexprs) didn't extend any live
  // ranges, we're ok with it.
  if (BaseReg == 0 && ScaledReg == 0)
    return true;

  // If all uses of this instruction are ultimately load/store/inlineasm's,
  // check to see if their addressing modes will include this instruction.  If
  // so, we can fold it into all uses, so it doesn't matter if it has multiple
  // uses.
  SmallVector<std::pair<Instruction*,unsigned>, 16> MemoryUses;
  SmallPtrSet<Instruction*, 16> ConsideredInsts;
  if (FindAllMemoryUses(I, MemoryUses, ConsideredInsts, TLI))
    return false;  // Has a non-memory, non-foldable use!
  
  // Now that we know that all uses of this instruction are part of a chain of
  // computation involving only operations that could theoretically be folded
  // into a memory use, loop over each of these uses and see if they could
  // *actually* fold the instruction.
  SmallVector<Instruction*, 32> MatchedAddrModeInsts;
  for (unsigned i = 0, e = MemoryUses.size(); i != e; ++i) {
    Instruction *User = MemoryUses[i].first;
    unsigned OpNo = MemoryUses[i].second;
    
    // Get the access type of this use.  If the use isn't a pointer, we don't
    // know what it accesses.
    Value *Address = User->getOperand(OpNo);
    if (!Address->getType()->isPointerTy())
      return false;
    Type *AddressAccessTy =
      cast<PointerType>(Address->getType())->getElementType();
    
    // Do a match against the root of this address, ignoring profitability. This
    // will tell us if the addressing mode for the memory operation will
    // *actually* cover the shared instruction.
    ExtAddrMode Result;
    AddressingModeMatcher Matcher(MatchedAddrModeInsts, TLI, AddressAccessTy,
                                  MemoryInst, Result);
    Matcher.IgnoreProfitability = true;
    bool Success = Matcher.MatchAddr(Address, 0);
    (void)Success; assert(Success && "Couldn't select *anything*?");

    // If the match didn't cover I, then it won't be shared by it.
    if (std::find(MatchedAddrModeInsts.begin(), MatchedAddrModeInsts.end(),
                  I) == MatchedAddrModeInsts.end())
      return false;
    
    MatchedAddrModeInsts.clear();
  }
  
  return true;
}
