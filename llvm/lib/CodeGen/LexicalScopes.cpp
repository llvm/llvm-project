//===- LexicalScopes.cpp - Collecting lexical scope info ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements LexicalScopes analysis.
//
// This pass collects lexical scope information and maps machine instructions
// to respective lexical scopes.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lexicalscopes"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/DebugInfo.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

LexicalScopes::~LexicalScopes() {
  releaseMemory();
}

/// releaseMemory - release memory.
void LexicalScopes::releaseMemory() {
  MF = NULL;
  CurrentFnLexicalScope = NULL;
  DeleteContainerSeconds(LexicalScopeMap);
  DeleteContainerSeconds(AbstractScopeMap);
  InlinedLexicalScopeMap.clear();
  AbstractScopesList.clear();
}

/// initialize - Scan machine function and constuct lexical scope nest.
void LexicalScopes::initialize(const MachineFunction &Fn) {
  releaseMemory();
  MF = &Fn;
  SmallVector<InsnRange, 4> MIRanges;
  DenseMap<const MachineInstr *, LexicalScope *> MI2ScopeMap;
  extractLexicalScopes(MIRanges, MI2ScopeMap);
  if (CurrentFnLexicalScope) {
    constructScopeNest(CurrentFnLexicalScope);
    assignInstructionRanges(MIRanges, MI2ScopeMap);
  }
}

/// extractLexicalScopes - Extract instruction ranges for each lexical scopes
/// for the given machine function.
void LexicalScopes::
extractLexicalScopes(SmallVectorImpl<InsnRange> &MIRanges,
                  DenseMap<const MachineInstr *, LexicalScope *> &MI2ScopeMap) {

  // Scan each instruction and create scopes. First build working set of scopes.
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    const MachineInstr *RangeBeginMI = NULL;
    const MachineInstr *PrevMI = NULL;
    DebugLoc PrevDL;
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MInsn = II;

      // Check if instruction has valid location information.
      const DebugLoc MIDL = MInsn->getDebugLoc();
      if (MIDL.isUnknown()) {
        PrevMI = MInsn;
        continue;
      }

      // If scope has not changed then skip this instruction.
      if (MIDL == PrevDL) {
        PrevMI = MInsn;
        continue;
      }

      // Ignore DBG_VALUE. It does not contribute to any instruction in output.
      if (MInsn->isDebugValue())
        continue;

      if (RangeBeginMI) {
        // If we have already seen a beginning of an instruction range and
        // current instruction scope does not match scope of first instruction
        // in this range then create a new instruction range.
        InsnRange R(RangeBeginMI, PrevMI);
        MI2ScopeMap[RangeBeginMI] = getOrCreateLexicalScope(PrevDL);
        MIRanges.push_back(R);
      }

      // This is a beginning of a new instruction range.
      RangeBeginMI = MInsn;

      // Reset previous markers.
      PrevMI = MInsn;
      PrevDL = MIDL;
    }

    // Create last instruction range.
    if (RangeBeginMI && PrevMI && !PrevDL.isUnknown()) {
      InsnRange R(RangeBeginMI, PrevMI);
      MIRanges.push_back(R);
      MI2ScopeMap[RangeBeginMI] = getOrCreateLexicalScope(PrevDL);
    }
  }
}

/// findLexicalScope - Find lexical scope, either regular or inlined, for the
/// given DebugLoc. Return NULL if not found.
LexicalScope *LexicalScopes::findLexicalScope(DebugLoc DL) {
  MDNode *Scope = NULL;
  MDNode *IA = NULL;
  DL.getScopeAndInlinedAt(Scope, IA, MF->getFunction()->getContext());
  if (!Scope) return NULL;

  // The scope that we were created with could have an extra file - which
  // isn't what we care about in this case.
  DIDescriptor D = DIDescriptor(Scope);
  if (D.isLexicalBlockFile())
    Scope = DILexicalBlockFile(Scope).getScope();
  
  if (IA)
    return InlinedLexicalScopeMap.lookup(DebugLoc::getFromDILocation(IA));
  return LexicalScopeMap.lookup(Scope);
}

/// getOrCreateLexicalScope - Find lexical scope for the given DebugLoc. If
/// not available then create new lexical scope.
LexicalScope *LexicalScopes::getOrCreateLexicalScope(DebugLoc DL) {
  MDNode *Scope = NULL;
  MDNode *InlinedAt = NULL;
  DL.getScopeAndInlinedAt(Scope, InlinedAt, MF->getFunction()->getContext());

  if (InlinedAt) {
    // Create an abstract scope for inlined function.
    getOrCreateAbstractScope(Scope);
    // Create an inlined scope for inlined function.
    return getOrCreateInlinedScope(Scope, InlinedAt);
  }
   
  return getOrCreateRegularScope(Scope);
}

/// getOrCreateRegularScope - Find or create a regular lexical scope.
LexicalScope *LexicalScopes::getOrCreateRegularScope(MDNode *Scope) {
  DIDescriptor D = DIDescriptor(Scope);
  if (D.isLexicalBlockFile()) {
    Scope = DILexicalBlockFile(Scope).getScope();
    D = DIDescriptor(Scope);
  }
 
  LexicalScope *WScope = LexicalScopeMap.lookup(Scope);
  if (WScope)
    return WScope;

  LexicalScope *Parent = NULL;
  if (D.isLexicalBlock())
    Parent = getOrCreateLexicalScope(DebugLoc::getFromDILexicalBlock(Scope));
  WScope = new LexicalScope(Parent, DIDescriptor(Scope), NULL, false);
  LexicalScopeMap.insert(std::make_pair(Scope, WScope));
  if (!Parent && DIDescriptor(Scope).isSubprogram()
      && DISubprogram(Scope).describes(MF->getFunction()))
    CurrentFnLexicalScope = WScope;
  
  return WScope;
}

/// getOrCreateInlinedScope - Find or create an inlined lexical scope.
LexicalScope *LexicalScopes::getOrCreateInlinedScope(MDNode *Scope, 
                                                     MDNode *InlinedAt) {
  LexicalScope *InlinedScope = LexicalScopeMap.lookup(InlinedAt);
  if (InlinedScope)
    return InlinedScope;

  DebugLoc InlinedLoc = DebugLoc::getFromDILocation(InlinedAt);
  InlinedScope = new LexicalScope(getOrCreateLexicalScope(InlinedLoc),
                                  DIDescriptor(Scope), InlinedAt, false);
  InlinedLexicalScopeMap[InlinedLoc] = InlinedScope;
  LexicalScopeMap[InlinedAt] = InlinedScope;
  return InlinedScope;
}

/// getOrCreateAbstractScope - Find or create an abstract lexical scope.
LexicalScope *LexicalScopes::getOrCreateAbstractScope(const MDNode *N) {
  assert(N && "Invalid Scope encoding!");

  DIDescriptor Scope(N);
  if (Scope.isLexicalBlockFile())
    Scope = DILexicalBlockFile(Scope).getScope();
  LexicalScope *AScope = AbstractScopeMap.lookup(N);
  if (AScope)
    return AScope;

  LexicalScope *Parent = NULL;
  if (Scope.isLexicalBlock()) {
    DILexicalBlock DB(N);
    DIDescriptor ParentDesc = DB.getContext();
    Parent = getOrCreateAbstractScope(ParentDesc);
  }
  AScope = new LexicalScope(Parent, DIDescriptor(N), NULL, true);
  AbstractScopeMap[N] = AScope;
  if (DIDescriptor(N).isSubprogram())
    AbstractScopesList.push_back(AScope);
  return AScope;
}

/// constructScopeNest
void LexicalScopes::constructScopeNest(LexicalScope *Scope) {
  assert (Scope && "Unable to calculate scop edominance graph!");
  SmallVector<LexicalScope *, 4> WorkStack;
  WorkStack.push_back(Scope);
  unsigned Counter = 0;
  while (!WorkStack.empty()) {
    LexicalScope *WS = WorkStack.back();
    const SmallVector<LexicalScope *, 4> &Children = WS->getChildren();
    bool visitedChildren = false;
    for (SmallVector<LexicalScope *, 4>::const_iterator SI = Children.begin(),
           SE = Children.end(); SI != SE; ++SI) {
      LexicalScope *ChildScope = *SI;
      if (!ChildScope->getDFSOut()) {
        WorkStack.push_back(ChildScope);
        visitedChildren = true;
        ChildScope->setDFSIn(++Counter);
        break;
      }
    }
    if (!visitedChildren) {
      WorkStack.pop_back();
      WS->setDFSOut(++Counter);
    }
  }
}

/// assignInstructionRanges - Find ranges of instructions covered by each
/// lexical scope.
void LexicalScopes::
assignInstructionRanges(SmallVectorImpl<InsnRange> &MIRanges,
                    DenseMap<const MachineInstr *, LexicalScope *> &MI2ScopeMap)
{
  
  LexicalScope *PrevLexicalScope = NULL;
  for (SmallVectorImpl<InsnRange>::const_iterator RI = MIRanges.begin(),
         RE = MIRanges.end(); RI != RE; ++RI) {
    const InsnRange &R = *RI;
    LexicalScope *S = MI2ScopeMap.lookup(R.first);
    assert (S && "Lost LexicalScope for a machine instruction!");
    if (PrevLexicalScope && !PrevLexicalScope->dominates(S))
      PrevLexicalScope->closeInsnRange(S);
    S->openInsnRange(R.first);
    S->extendInsnRange(R.second);
    PrevLexicalScope = S;
  }

  if (PrevLexicalScope)
    PrevLexicalScope->closeInsnRange();
}

/// getMachineBasicBlocks - Populate given set using machine basic blocks which
/// have machine instructions that belong to lexical scope identified by 
/// DebugLoc.
void LexicalScopes::
getMachineBasicBlocks(DebugLoc DL, 
                      SmallPtrSet<const MachineBasicBlock*, 4> &MBBs) {
  MBBs.clear();
  LexicalScope *Scope = getOrCreateLexicalScope(DL);
  if (!Scope)
    return;
  
  if (Scope == CurrentFnLexicalScope) {
    for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
         I != E; ++I)
      MBBs.insert(I);
    return;
  }

  SmallVector<InsnRange, 4> &InsnRanges = Scope->getRanges();
  for (SmallVector<InsnRange, 4>::iterator I = InsnRanges.begin(),
         E = InsnRanges.end(); I != E; ++I) {
    InsnRange &R = *I;
    MBBs.insert(R.first->getParent());
  }
}

/// dominates - Return true if DebugLoc's lexical scope dominates at least one
/// machine instruction's lexical scope in a given machine basic block.
bool LexicalScopes::dominates(DebugLoc DL, MachineBasicBlock *MBB) {
  LexicalScope *Scope = getOrCreateLexicalScope(DL);
  if (!Scope)
    return false;

  // Current function scope covers all basic blocks in the function.
  if (Scope == CurrentFnLexicalScope && MBB->getParent() == MF)
    return true;

  bool Result = false;
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
       I != E; ++I) {
    DebugLoc IDL = I->getDebugLoc();
    if (IDL.isUnknown())
      continue;
    if (LexicalScope *IScope = getOrCreateLexicalScope(IDL))
      if (Scope->dominates(IScope))
        return true;
  }
  return Result;
}

void LexicalScope::anchor() { }

/// dump - Print data structures.
void LexicalScope::dump() const {
#ifndef NDEBUG
  raw_ostream &err = dbgs();
  err.indent(IndentLevel);
  err << "DFSIn: " << DFSIn << " DFSOut: " << DFSOut << "\n";
  const MDNode *N = Desc;
  N->dump();
  if (AbstractScope)
    err << "Abstract Scope\n";

  IndentLevel += 2;
  if (!Children.empty())
    err << "Children ...\n";
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    if (Children[i] != this)
      Children[i]->dump();

  IndentLevel -= 2;
#endif
}

