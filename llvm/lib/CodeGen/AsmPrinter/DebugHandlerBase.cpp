//===-- llvm/lib/CodeGen/AsmPrinter/DebugHandlerBase.cpp -------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common functionality for different debug information format backends.
// LLVM currently supports DWARF and CodeView.
//
//===----------------------------------------------------------------------===//

#include "DebugHandlerBase.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

DebugHandlerBase::DebugHandlerBase(AsmPrinter *A) : Asm(A), MMI(Asm->MMI) {}

// Each LexicalScope has first instruction and last instruction to mark
// beginning and end of a scope respectively. Create an inverse map that list
// scopes starts (and ends) with an instruction. One instruction may start (or
// end) multiple scopes. Ignore scopes that are not reachable.
void DebugHandlerBase::identifyScopeMarkers() {
  SmallVector<LexicalScope *, 4> WorkList;
  WorkList.push_back(LScopes.getCurrentFunctionScope());
  while (!WorkList.empty()) {
    LexicalScope *S = WorkList.pop_back_val();

    const SmallVectorImpl<LexicalScope *> &Children = S->getChildren();
    if (!Children.empty())
      WorkList.append(Children.begin(), Children.end());

    if (S->isAbstractScope())
      continue;

    for (const InsnRange &R : S->getRanges()) {
      assert(R.first && "InsnRange does not have first instruction!");
      assert(R.second && "InsnRange does not have second instruction!");
      requestLabelBeforeInsn(R.first);
      requestLabelAfterInsn(R.second);
    }
  }
}

// Return Label preceding the instruction.
MCSymbol *DebugHandlerBase::getLabelBeforeInsn(const MachineInstr *MI) {
  MCSymbol *Label = LabelsBeforeInsn.lookup(MI);
  assert(Label && "Didn't insert label before instruction");
  return Label;
}

// Return Label immediately following the instruction.
MCSymbol *DebugHandlerBase::getLabelAfterInsn(const MachineInstr *MI) {
  return LabelsAfterInsn.lookup(MI);
}

int DebugHandlerBase::fragmentCmp(const DIExpression *P1,
                                  const DIExpression *P2) {
  auto Fragment1 = *P1->getFragmentInfo();
  auto Fragment2 = *P2->getFragmentInfo();
  unsigned l1 = Fragment1.OffsetInBits;
  unsigned l2 = Fragment2.OffsetInBits;
  unsigned r1 = l1 + Fragment1.SizeInBits;
  unsigned r2 = l2 + Fragment2.SizeInBits;
  if (r1 <= l2)
    return -1;
  else if (r2 <= l1)
    return 1;
  else
    return 0;
}

bool DebugHandlerBase::fragmentsOverlap(const DIExpression *P1,
                                        const DIExpression *P2) {
  if (!P1->isFragment() || !P2->isFragment())
    return true;
  return fragmentCmp(P1, P2) == 0;
}

/// If this type is derived from a base type then return base type size.
uint64_t DebugHandlerBase::getBaseTypeSize(const DITypeRef TyRef) {
  DIType *Ty = TyRef.resolve();
  assert(Ty);
  DIDerivedType *DDTy = dyn_cast<DIDerivedType>(Ty);
  if (!DDTy)
    return Ty->getSizeInBits();

  unsigned Tag = DDTy->getTag();

  if (Tag != dwarf::DW_TAG_member && Tag != dwarf::DW_TAG_typedef &&
      Tag != dwarf::DW_TAG_const_type && Tag != dwarf::DW_TAG_volatile_type &&
      Tag != dwarf::DW_TAG_restrict_type && Tag != dwarf::DW_TAG_atomic_type)
    return DDTy->getSizeInBits();

  DIType *BaseType = DDTy->getBaseType().resolve();

  assert(BaseType && "Unexpected invalid base type");

  // If this is a derived type, go ahead and get the base type, unless it's a
  // reference then it's just the size of the field. Pointer types have no need
  // of this since they're a different type of qualification on the type.
  if (BaseType->getTag() == dwarf::DW_TAG_reference_type ||
      BaseType->getTag() == dwarf::DW_TAG_rvalue_reference_type)
    return Ty->getSizeInBits();

  return getBaseTypeSize(BaseType);
}

bool hasDebugInfo(const MachineModuleInfo *MMI, const MachineFunction *MF) {
  if (!MMI->hasDebugInfo())
    return false;
  auto *SP = MF->getFunction()->getSubprogram();
  if (!SP)
    return false;
  assert(SP->getUnit());
  auto EK = SP->getUnit()->getEmissionKind();
  if (EK == DICompileUnit::NoDebug)
    return false;
  return true;
}

void DebugHandlerBase::beginFunction(const MachineFunction *MF) {
  PrevInstBB = nullptr;

  if (!Asm || !hasDebugInfo(MMI, MF)) {
    skippedNonDebugFunction();
    return;
  }

  // Grab the lexical scopes for the function, if we don't have any of those
  // then we're not going to be able to do anything.
  LScopes.initialize(*MF);
  if (LScopes.empty()) {
    beginFunctionImpl(MF);
    return;
  }

  // Make sure that each lexical scope will have a begin/end label.
  identifyScopeMarkers();

  // Calculate history for local variables.
  assert(DbgValues.empty() && "DbgValues map wasn't cleaned!");
  calculateDbgValueHistory(MF, Asm->MF->getSubtarget().getRegisterInfo(),
                           DbgValues);

  // Request labels for the full history.
  for (const auto &I : DbgValues) {
    const auto &Ranges = I.second;
    if (Ranges.empty())
      continue;

    // The first mention of a function argument gets the CurrentFnBegin
    // label, so arguments are visible when breaking at function entry.
    const DILocalVariable *DIVar = Ranges.front().first->getDebugVariable();
    if (DIVar->isParameter() &&
        getDISubprogram(DIVar->getScope())->describes(MF->getFunction())) {
      LabelsBeforeInsn[Ranges.front().first] = Asm->getFunctionBegin();
      if (Ranges.front().first->getDebugExpression()->isFragment()) {
        // Mark all non-overlapping initial fragments.
        for (auto I = Ranges.begin(); I != Ranges.end(); ++I) {
          const DIExpression *Fragment = I->first->getDebugExpression();
          if (std::all_of(Ranges.begin(), I,
                          [&](DbgValueHistoryMap::InstrRange Pred) {
                            return !fragmentsOverlap(
                                Fragment, Pred.first->getDebugExpression());
                          }))
            LabelsBeforeInsn[I->first] = Asm->getFunctionBegin();
          else
            break;
        }
      }
    }

    for (const auto &Range : Ranges) {
      requestLabelBeforeInsn(Range.first);
      if (Range.second)
        requestLabelAfterInsn(Range.second);
    }
  }

  PrevInstLoc = DebugLoc();
  PrevLabel = Asm->getFunctionBegin();
  beginFunctionImpl(MF);
}

void DebugHandlerBase::beginInstruction(const MachineInstr *MI) {
  if (!MMI->hasDebugInfo())
    return;

  assert(CurMI == nullptr);
  CurMI = MI;

  // Insert labels where requested.
  DenseMap<const MachineInstr *, MCSymbol *>::iterator I =
      LabelsBeforeInsn.find(MI);

  // No label needed.
  if (I == LabelsBeforeInsn.end())
    return;

  // Label already assigned.
  if (I->second)
    return;

  if (!PrevLabel) {
    PrevLabel = MMI->getContext().createTempSymbol();
    Asm->OutStreamer->EmitLabel(PrevLabel);
  }
  I->second = PrevLabel;
}

void DebugHandlerBase::endInstruction() {
  if (!MMI->hasDebugInfo())
    return;

  assert(CurMI != nullptr);
  // Don't create a new label after DBG_VALUE instructions.
  // They don't generate code.
  if (!CurMI->isDebugValue()) {
    PrevLabel = nullptr;
    PrevInstBB = CurMI->getParent();
  }

  DenseMap<const MachineInstr *, MCSymbol *>::iterator I =
      LabelsAfterInsn.find(CurMI);
  CurMI = nullptr;

  // No label needed.
  if (I == LabelsAfterInsn.end())
    return;

  // Label already assigned.
  if (I->second)
    return;

  // We need a label after this instruction.
  if (!PrevLabel) {
    PrevLabel = MMI->getContext().createTempSymbol();
    Asm->OutStreamer->EmitLabel(PrevLabel);
  }
  I->second = PrevLabel;
}

void DebugHandlerBase::endFunction(const MachineFunction *MF) {
  if (hasDebugInfo(MMI, MF))
    endFunctionImpl(MF);
  DbgValues.clear();
  LabelsBeforeInsn.clear();
  LabelsAfterInsn.clear();
}
