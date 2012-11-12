//===-- FunctionLoweringInfo.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements routines for translating functions from LLVM IR into
// Machine IR.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "function-lowering-info"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/DebugInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
using namespace llvm;

/// isUsedOutsideOfDefiningBlock - Return true if this instruction is used by
/// PHI nodes or outside of the basic block that defines it, or used by a
/// switch or atomic instruction, which may expand to multiple basic blocks.
static bool isUsedOutsideOfDefiningBlock(const Instruction *I) {
  if (I->use_empty()) return false;
  if (isa<PHINode>(I)) return true;
  const BasicBlock *BB = I->getParent();
  for (Value::const_use_iterator UI = I->use_begin(), E = I->use_end();
        UI != E; ++UI) {
    const User *U = *UI;
    if (cast<Instruction>(U)->getParent() != BB || isa<PHINode>(U))
      return true;
  }
  return false;
}

FunctionLoweringInfo::FunctionLoweringInfo(const TargetLowering &tli)
  : TLI(tli) {
}

void FunctionLoweringInfo::set(const Function &fn, MachineFunction &mf) {
  Fn = &fn;
  MF = &mf;
  RegInfo = &MF->getRegInfo();

  // Check whether the function can return without sret-demotion.
  SmallVector<ISD::OutputArg, 4> Outs;
  GetReturnInfo(Fn->getReturnType(),
                Fn->getAttributes().getRetAttributes(), Outs, TLI);
  CanLowerReturn = TLI.CanLowerReturn(Fn->getCallingConv(), *MF,
                                      Fn->isVarArg(),
                                      Outs, Fn->getContext());

  // Initialize the mapping of values to registers.  This is only set up for
  // instruction values that are used outside of the block that defines
  // them.
  Function::const_iterator BB = Fn->begin(), EB = Fn->end();
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (const AllocaInst *AI = dyn_cast<AllocaInst>(I))
      if (const ConstantInt *CUI = dyn_cast<ConstantInt>(AI->getArraySize())) {
        Type *Ty = AI->getAllocatedType();
        uint64_t TySize = TLI.getDataLayout()->getTypeAllocSize(Ty);
        unsigned Align =
          std::max((unsigned)TLI.getDataLayout()->getPrefTypeAlignment(Ty),
                   AI->getAlignment());

        TySize *= CUI->getZExtValue();   // Get total allocated size.
        if (TySize == 0) TySize = 1; // Don't create zero-sized stack objects.

        // The object may need to be placed onto the stack near the stack
        // protector if one exists. Determine here if this object is a suitable
        // candidate. I.e., it would trigger the creation of a stack protector.
        bool MayNeedSP =
          (AI->isArrayAllocation() ||
           (TySize >= 8 && isa<ArrayType>(Ty) &&
            cast<ArrayType>(Ty)->getElementType()->isIntegerTy(8)));
        StaticAllocaMap[AI] =
          MF->getFrameInfo()->CreateStackObject(TySize, Align, false,
                                                MayNeedSP, AI);
      }

  for (; BB != EB; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      // Mark values used outside their block as exported, by allocating
      // a virtual register for them.
      if (isUsedOutsideOfDefiningBlock(I))
        if (!isa<AllocaInst>(I) ||
            !StaticAllocaMap.count(cast<AllocaInst>(I)))
          InitializeRegForValue(I);

      // Collect llvm.dbg.declare information. This is done now instead of
      // during the initial isel pass through the IR so that it is done
      // in a predictable order.
      if (const DbgDeclareInst *DI = dyn_cast<DbgDeclareInst>(I)) {
        MachineModuleInfo &MMI = MF->getMMI();
        if (MMI.hasDebugInfo() &&
            DIVariable(DI->getVariable()).Verify() &&
            !DI->getDebugLoc().isUnknown()) {
          // Don't handle byval struct arguments or VLAs, for example.
          // Non-byval arguments are handled here (they refer to the stack
          // temporary alloca at this point).
          const Value *Address = DI->getAddress();
          if (Address) {
            if (const BitCastInst *BCI = dyn_cast<BitCastInst>(Address))
              Address = BCI->getOperand(0);
            if (const AllocaInst *AI = dyn_cast<AllocaInst>(Address)) {
              DenseMap<const AllocaInst *, int>::iterator SI =
                StaticAllocaMap.find(AI);
              if (SI != StaticAllocaMap.end()) { // Check for VLAs.
                int FI = SI->second;
                MMI.setVariableDbgInfo(DI->getVariable(),
                                       FI, DI->getDebugLoc());
              }
            }
          }
        }
      }
    }

  // Create an initial MachineBasicBlock for each LLVM BasicBlock in F.  This
  // also creates the initial PHI MachineInstrs, though none of the input
  // operands are populated.
  for (BB = Fn->begin(); BB != EB; ++BB) {
    MachineBasicBlock *MBB = mf.CreateMachineBasicBlock(BB);
    MBBMap[BB] = MBB;
    MF->push_back(MBB);

    // Transfer the address-taken flag. This is necessary because there could
    // be multiple MachineBasicBlocks corresponding to one BasicBlock, and only
    // the first one should be marked.
    if (BB->hasAddressTaken())
      MBB->setHasAddressTaken();

    // Create Machine PHI nodes for LLVM PHI nodes, lowering them as
    // appropriate.
    for (BasicBlock::const_iterator I = BB->begin();
         const PHINode *PN = dyn_cast<PHINode>(I); ++I) {
      if (PN->use_empty()) continue;

      // Skip empty types
      if (PN->getType()->isEmptyTy())
        continue;

      DebugLoc DL = PN->getDebugLoc();
      unsigned PHIReg = ValueMap[PN];
      assert(PHIReg && "PHI node does not have an assigned virtual register!");

      SmallVector<EVT, 4> ValueVTs;
      ComputeValueVTs(TLI, PN->getType(), ValueVTs);
      for (unsigned vti = 0, vte = ValueVTs.size(); vti != vte; ++vti) {
        EVT VT = ValueVTs[vti];
        unsigned NumRegisters = TLI.getNumRegisters(Fn->getContext(), VT);
        const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();
        for (unsigned i = 0; i != NumRegisters; ++i)
          BuildMI(MBB, DL, TII->get(TargetOpcode::PHI), PHIReg + i);
        PHIReg += NumRegisters;
      }
    }
  }

  // Mark landing pad blocks.
  for (BB = Fn->begin(); BB != EB; ++BB)
    if (const InvokeInst *Invoke = dyn_cast<InvokeInst>(BB->getTerminator()))
      MBBMap[Invoke->getSuccessor(1)]->setIsLandingPad();
}

/// clear - Clear out all the function-specific state. This returns this
/// FunctionLoweringInfo to an empty state, ready to be used for a
/// different function.
void FunctionLoweringInfo::clear() {
  assert(CatchInfoFound.size() == CatchInfoLost.size() &&
         "Not all catch info was assigned to a landing pad!");

  MBBMap.clear();
  ValueMap.clear();
  StaticAllocaMap.clear();
#ifndef NDEBUG
  CatchInfoLost.clear();
  CatchInfoFound.clear();
#endif
  LiveOutRegInfo.clear();
  VisitedBBs.clear();
  ArgDbgValues.clear();
  ByValArgFrameIndexMap.clear();
  RegFixups.clear();
}

/// CreateReg - Allocate a single virtual register for the given type.
unsigned FunctionLoweringInfo::CreateReg(EVT VT) {
  return RegInfo->createVirtualRegister(TLI.getRegClassFor(VT));
}

/// CreateRegs - Allocate the appropriate number of virtual registers of
/// the correctly promoted or expanded types.  Assign these registers
/// consecutive vreg numbers and return the first assigned number.
///
/// In the case that the given value has struct or array type, this function
/// will assign registers for each member or element.
///
unsigned FunctionLoweringInfo::CreateRegs(Type *Ty) {
  SmallVector<EVT, 4> ValueVTs;
  ComputeValueVTs(TLI, Ty, ValueVTs);

  unsigned FirstReg = 0;
  for (unsigned Value = 0, e = ValueVTs.size(); Value != e; ++Value) {
    EVT ValueVT = ValueVTs[Value];
    EVT RegisterVT = TLI.getRegisterType(Ty->getContext(), ValueVT);

    unsigned NumRegs = TLI.getNumRegisters(Ty->getContext(), ValueVT);
    for (unsigned i = 0; i != NumRegs; ++i) {
      unsigned R = CreateReg(RegisterVT);
      if (!FirstReg) FirstReg = R;
    }
  }
  return FirstReg;
}

/// GetLiveOutRegInfo - Gets LiveOutInfo for a register, returning NULL if the
/// register is a PHI destination and the PHI's LiveOutInfo is not valid. If
/// the register's LiveOutInfo is for a smaller bit width, it is extended to
/// the larger bit width by zero extension. The bit width must be no smaller
/// than the LiveOutInfo's existing bit width.
const FunctionLoweringInfo::LiveOutInfo *
FunctionLoweringInfo::GetLiveOutRegInfo(unsigned Reg, unsigned BitWidth) {
  if (!LiveOutRegInfo.inBounds(Reg))
    return NULL;

  LiveOutInfo *LOI = &LiveOutRegInfo[Reg];
  if (!LOI->IsValid)
    return NULL;

  if (BitWidth > LOI->KnownZero.getBitWidth()) {
    LOI->NumSignBits = 1;
    LOI->KnownZero = LOI->KnownZero.zextOrTrunc(BitWidth);
    LOI->KnownOne = LOI->KnownOne.zextOrTrunc(BitWidth);
  }

  return LOI;
}

/// ComputePHILiveOutRegInfo - Compute LiveOutInfo for a PHI's destination
/// register based on the LiveOutInfo of its operands.
void FunctionLoweringInfo::ComputePHILiveOutRegInfo(const PHINode *PN) {
  Type *Ty = PN->getType();
  if (!Ty->isIntegerTy() || Ty->isVectorTy())
    return;

  SmallVector<EVT, 1> ValueVTs;
  ComputeValueVTs(TLI, Ty, ValueVTs);
  assert(ValueVTs.size() == 1 &&
         "PHIs with non-vector integer types should have a single VT.");
  EVT IntVT = ValueVTs[0];

  if (TLI.getNumRegisters(PN->getContext(), IntVT) != 1)
    return;
  IntVT = TLI.getTypeToTransformTo(PN->getContext(), IntVT);
  unsigned BitWidth = IntVT.getSizeInBits();

  unsigned DestReg = ValueMap[PN];
  if (!TargetRegisterInfo::isVirtualRegister(DestReg))
    return;
  LiveOutRegInfo.grow(DestReg);
  LiveOutInfo &DestLOI = LiveOutRegInfo[DestReg];

  Value *V = PN->getIncomingValue(0);
  if (isa<UndefValue>(V) || isa<ConstantExpr>(V)) {
    DestLOI.NumSignBits = 1;
    APInt Zero(BitWidth, 0);
    DestLOI.KnownZero = Zero;
    DestLOI.KnownOne = Zero;
    return;
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    APInt Val = CI->getValue().zextOrTrunc(BitWidth);
    DestLOI.NumSignBits = Val.getNumSignBits();
    DestLOI.KnownZero = ~Val;
    DestLOI.KnownOne = Val;
  } else {
    assert(ValueMap.count(V) && "V should have been placed in ValueMap when its"
                                "CopyToReg node was created.");
    unsigned SrcReg = ValueMap[V];
    if (!TargetRegisterInfo::isVirtualRegister(SrcReg)) {
      DestLOI.IsValid = false;
      return;
    }
    const LiveOutInfo *SrcLOI = GetLiveOutRegInfo(SrcReg, BitWidth);
    if (!SrcLOI) {
      DestLOI.IsValid = false;
      return;
    }
    DestLOI = *SrcLOI;
  }

  assert(DestLOI.KnownZero.getBitWidth() == BitWidth &&
         DestLOI.KnownOne.getBitWidth() == BitWidth &&
         "Masks should have the same bit width as the type.");

  for (unsigned i = 1, e = PN->getNumIncomingValues(); i != e; ++i) {
    Value *V = PN->getIncomingValue(i);
    if (isa<UndefValue>(V) || isa<ConstantExpr>(V)) {
      DestLOI.NumSignBits = 1;
      APInt Zero(BitWidth, 0);
      DestLOI.KnownZero = Zero;
      DestLOI.KnownOne = Zero;
      return;
    }

    if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      APInt Val = CI->getValue().zextOrTrunc(BitWidth);
      DestLOI.NumSignBits = std::min(DestLOI.NumSignBits, Val.getNumSignBits());
      DestLOI.KnownZero &= ~Val;
      DestLOI.KnownOne &= Val;
      continue;
    }

    assert(ValueMap.count(V) && "V should have been placed in ValueMap when "
                                "its CopyToReg node was created.");
    unsigned SrcReg = ValueMap[V];
    if (!TargetRegisterInfo::isVirtualRegister(SrcReg)) {
      DestLOI.IsValid = false;
      return;
    }
    const LiveOutInfo *SrcLOI = GetLiveOutRegInfo(SrcReg, BitWidth);
    if (!SrcLOI) {
      DestLOI.IsValid = false;
      return;
    }
    DestLOI.NumSignBits = std::min(DestLOI.NumSignBits, SrcLOI->NumSignBits);
    DestLOI.KnownZero &= SrcLOI->KnownZero;
    DestLOI.KnownOne &= SrcLOI->KnownOne;
  }
}

/// setArgumentFrameIndex - Record frame index for the byval
/// argument. This overrides previous frame index entry for this argument,
/// if any.
void FunctionLoweringInfo::setArgumentFrameIndex(const Argument *A,
                                                 int FI) {
  ByValArgFrameIndexMap[A] = FI;
}

/// getArgumentFrameIndex - Get frame index for the byval argument.
/// If the argument does not have any assigned frame index then 0 is
/// returned.
int FunctionLoweringInfo::getArgumentFrameIndex(const Argument *A) {
  DenseMap<const Argument *, int>::iterator I =
    ByValArgFrameIndexMap.find(A);
  if (I != ByValArgFrameIndexMap.end())
    return I->second;
  DEBUG(dbgs() << "Argument does not have assigned frame index!\n");
  return 0;
}

/// ComputeUsesVAFloatArgument - Determine if any floating-point values are
/// being passed to this variadic function, and set the MachineModuleInfo's
/// usesVAFloatArgument flag if so. This flag is used to emit an undefined
/// reference to _fltused on Windows, which will link in MSVCRT's
/// floating-point support.
void llvm::ComputeUsesVAFloatArgument(const CallInst &I,
                                      MachineModuleInfo *MMI)
{
  FunctionType *FT = cast<FunctionType>(
    I.getCalledValue()->getType()->getContainedType(0));
  if (FT->isVarArg() && !MMI->usesVAFloatArgument()) {
    for (unsigned i = 0, e = I.getNumArgOperands(); i != e; ++i) {
      Type* T = I.getArgOperand(i)->getType();
      for (po_iterator<Type*> i = po_begin(T), e = po_end(T);
           i != e; ++i) {
        if (i->isFloatingPointTy()) {
          MMI->setUsesVAFloatArgument(true);
          return;
        }
      }
    }
  }
}

/// AddCatchInfo - Extract the personality and type infos from an eh.selector
/// call, and add them to the specified machine basic block.
void llvm::AddCatchInfo(const CallInst &I, MachineModuleInfo *MMI,
                        MachineBasicBlock *MBB) {
  // Inform the MachineModuleInfo of the personality for this landing pad.
  const ConstantExpr *CE = cast<ConstantExpr>(I.getArgOperand(1));
  assert(CE->getOpcode() == Instruction::BitCast &&
         isa<Function>(CE->getOperand(0)) &&
         "Personality should be a function");
  MMI->addPersonality(MBB, cast<Function>(CE->getOperand(0)));

  // Gather all the type infos for this landing pad and pass them along to
  // MachineModuleInfo.
  std::vector<const GlobalVariable *> TyInfo;
  unsigned N = I.getNumArgOperands();

  for (unsigned i = N - 1; i > 1; --i) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(I.getArgOperand(i))) {
      unsigned FilterLength = CI->getZExtValue();
      unsigned FirstCatch = i + FilterLength + !FilterLength;
      assert(FirstCatch <= N && "Invalid filter length");

      if (FirstCatch < N) {
        TyInfo.reserve(N - FirstCatch);
        for (unsigned j = FirstCatch; j < N; ++j)
          TyInfo.push_back(ExtractTypeInfo(I.getArgOperand(j)));
        MMI->addCatchTypeInfo(MBB, TyInfo);
        TyInfo.clear();
      }

      if (!FilterLength) {
        // Cleanup.
        MMI->addCleanup(MBB);
      } else {
        // Filter.
        TyInfo.reserve(FilterLength - 1);
        for (unsigned j = i + 1; j < FirstCatch; ++j)
          TyInfo.push_back(ExtractTypeInfo(I.getArgOperand(j)));
        MMI->addFilterTypeInfo(MBB, TyInfo);
        TyInfo.clear();
      }

      N = i;
    }
  }

  if (N > 2) {
    TyInfo.reserve(N - 2);
    for (unsigned j = 2; j < N; ++j)
      TyInfo.push_back(ExtractTypeInfo(I.getArgOperand(j)));
    MMI->addCatchTypeInfo(MBB, TyInfo);
  }
}

/// AddLandingPadInfo - Extract the exception handling information from the
/// landingpad instruction and add them to the specified machine module info.
void llvm::AddLandingPadInfo(const LandingPadInst &I, MachineModuleInfo &MMI,
                             MachineBasicBlock *MBB) {
  MMI.addPersonality(MBB,
                     cast<Function>(I.getPersonalityFn()->stripPointerCasts()));

  if (I.isCleanup())
    MMI.addCleanup(MBB);

  // FIXME: New EH - Add the clauses in reverse order. This isn't 100% correct,
  //        but we need to do it this way because of how the DWARF EH emitter
  //        processes the clauses.
  for (unsigned i = I.getNumClauses(); i != 0; --i) {
    Value *Val = I.getClause(i - 1);
    if (I.isCatch(i - 1)) {
      MMI.addCatchTypeInfo(MBB,
                           dyn_cast<GlobalVariable>(Val->stripPointerCasts()));
    } else {
      // Add filters in a list.
      Constant *CVal = cast<Constant>(Val);
      SmallVector<const GlobalVariable*, 4> FilterList;
      for (User::op_iterator
             II = CVal->op_begin(), IE = CVal->op_end(); II != IE; ++II)
        FilterList.push_back(cast<GlobalVariable>((*II)->stripPointerCasts()));

      MMI.addFilterTypeInfo(MBB, FilterList);
    }
  }
}
