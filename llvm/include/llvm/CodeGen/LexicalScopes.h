//===- LexicalScopes.cpp - Collecting lexical scope info --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements LexicalScopes analysis.
//
// This pass collects lexical scope information and maps machine instructions
// to respective lexical scopes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LEXICALSCOPES_H
#define LLVM_CODEGEN_LEXICALSCOPES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <unordered_map>
#include <utility>

namespace llvm {

class MachineBasicBlock;
class MachineFunction;
class MachineInstr;
class MDNode;

//===----------------------------------------------------------------------===//
/// This is used to track range of instructions with identical lexical scope.
///
using InsnRange = std::pair<const MachineInstr *, const MachineInstr *>;

//===----------------------------------------------------------------------===//
/// This class is used to track scope information.
///
class LexicalScope {
public:
  LexicalScope(LexicalScope *P, const DILocalScope *D, const DILocation *I,
               bool A)
      : Parent(P), Desc(D), InlinedAtLocation(I), AbstractScope(A) {
    assert(D);
    assert(D->getSubprogram()->getUnit()->getEmissionKind() !=
           DICompileUnit::NoDebug &&
           "Don't build lexical scopes for non-debug locations");
    assert(D->isResolved() && "Expected resolved node");
    assert((!I || I->isResolved()) && "Expected resolved node");
    if (Parent)
      Parent->addChild(this);
  }

  // Accessors.
  LexicalScope *getParent() const { return Parent; }
  const MDNode *getDesc() const { return Desc; }
  const DILocation *getInlinedAt() const { return InlinedAtLocation; }
  const DILocalScope *getScopeNode() const { return Desc; }
  bool isAbstractScope() const { return AbstractScope; }
  SmallVectorImpl<LexicalScope *> &getChildren() { return Children; }
  SmallVectorImpl<InsnRange> &getRanges() { return Ranges; }

  /// Add a child scope.
  void addChild(LexicalScope *S) { Children.push_back(S); }

  /// This scope covers instruction range starting from MI.
  void openInsnRange(const MachineInstr *MI) {
    if (!FirstInsn)
      FirstInsn = MI;

    if (Parent)
      Parent->openInsnRange(MI);
  }

  /// Extend the current instruction range covered by this scope.
  void extendInsnRange(const MachineInstr *MI) {
    assert(FirstInsn && "MI Range is not open!");
    LastInsn = MI;
    if (Parent)
      Parent->extendInsnRange(MI);
  }

  /// Create a range based on FirstInsn and LastInsn collected until now.
  /// This is used when a new scope is encountered while walking machine
  /// instructions.
  void closeInsnRange(LexicalScope *NewScope = nullptr) {
    assert(LastInsn && "Last insn missing!");
    Ranges.push_back(InsnRange(FirstInsn, LastInsn));
    FirstInsn = nullptr;
    LastInsn = nullptr;
    // If Parent dominates NewScope then do not close Parent's instruction
    // range.
    if (Parent && (!NewScope || !Parent->dominates(NewScope)))
      Parent->closeInsnRange(NewScope);
  }

  /// Return true if current scope dominates given lexical scope.
  bool dominates(const LexicalScope *S) const {
    if (S == this)
      return true;
    if (DFSIn < S->getDFSIn() && DFSOut > S->getDFSOut())
      return true;
    return false;
  }

  // Depth First Search support to walk and manipulate LexicalScope hierarchy.
  unsigned getDFSOut() const { return DFSOut; }
  void setDFSOut(unsigned O) { DFSOut = O; }
  unsigned getDFSIn() const { return DFSIn; }
  void setDFSIn(unsigned I) { DFSIn = I; }

  /// Print lexical scope.
  LLVM_ABI void dump(unsigned Indent = 0) const;

private:
  LexicalScope *Parent;                        // Parent to this scope.
  const DILocalScope *Desc;                    // Debug info descriptor.
  const DILocation *InlinedAtLocation;         // Location at which this
                                               // scope is inlined.
  bool AbstractScope;                          // Abstract Scope
  SmallVector<LexicalScope *, 4> Children;     // Scopes defined in scope.
                                               // Contents not owned.
  SmallVector<InsnRange, 4> Ranges;

  const MachineInstr *LastInsn = nullptr;  // Last instruction of this scope.
  const MachineInstr *FirstInsn = nullptr; // First instruction of this scope.
  unsigned DFSIn = 0; // In & Out Depth use to determine scope nesting.
  unsigned DFSOut = 0;
};

//===----------------------------------------------------------------------===//
/// This class provides interface to collect and use lexical scoping information
/// from machine instruction.
///
class LexicalScopes {
public:
  LexicalScopes() = default;

  /// Scan module to build subprogram-to-function map.
  LLVM_ABI void initialize(const Module &);

  /// Scan machine function and constuct lexical scope nest, resets
  /// the instance if necessary.
  LLVM_ABI void scanFunction(const MachineFunction &);

  /// Reset the instance so that it's prepared for another module.
  LLVM_ABI void resetModule();

  /// Reset the instance so that it's prepared for another function.
  LLVM_ABI void resetFunction();

  /// Return true if there is any lexical scope information available.
  bool empty() { return CurrentFnLexicalScope == nullptr; }

  /// Return lexical scope for the current function.
  LexicalScope *getCurrentFunctionScope() const {
    return CurrentFnLexicalScope;
  }

  /// Populate given set using machine basic blocks which have machine
  /// instructions that belong to lexical scope identified by DebugLoc.
  LLVM_ABI void
  getMachineBasicBlocks(const DILocation *DL,
                        SmallPtrSetImpl<const MachineBasicBlock *> &MBBs);

  /// Return true if DebugLoc's lexical scope dominates at least one machine
  /// instruction's lexical scope in a given machine basic block.
  LLVM_ABI bool dominates(const DILocation *DL, MachineBasicBlock *MBB);

  /// Find lexical scope, either regular or inlined, for the given DebugLoc.
  /// Return NULL if not found.
  LLVM_ABI LexicalScope *findLexicalScope(const DILocation *DL);

  /// Return a reference to list of abstract scopes.
  ArrayRef<LexicalScope *> getAbstractScopesList() const {
    return AbstractScopesList;
  }

  /// Find an abstract scope or return null.
  LexicalScope *findAbstractScope(const DILocalScope *N) {
    auto I = AbstractScopeMap.find(N);
    return I != AbstractScopeMap.end() ? &I->second : nullptr;
  }

  /// Find an inlined scope for the given scope/inlined-at.
  LexicalScope *findInlinedScope(const DILocalScope *N, const DILocation *IA) {
    auto I = InlinedLexicalScopeMap.find(std::make_pair(N, IA));
    return I != InlinedLexicalScopeMap.end() ? &I->second : nullptr;
  }

  /// Find regular lexical scope or return null.
  LexicalScope *findLexicalScope(const DILocalScope *N) {
    auto I = LexicalScopeMap.find(N);
    return I != LexicalScopeMap.end() ? &I->second : nullptr;
  }

  /// Find or create an abstract lexical scope.
  LLVM_ABI LexicalScope *getOrCreateAbstractScope(const DILocalScope *Scope);

  /// Get function to which the given subprogram is attached, if exists.
  const Function *getFunction(const DISubprogram *SP) const {
    return FunctionMap.lookup(SP);
  }

private:
  /// Find lexical scope for the given Scope/IA. If not available
  /// then create new lexical scope.
  LLVM_ABI LexicalScope *
  getOrCreateLexicalScope(const DILocalScope *Scope,
                          const DILocation *IA = nullptr);
  LexicalScope *getOrCreateLexicalScope(const DILocation *DL) {
    return DL ? getOrCreateLexicalScope(DL->getScope(), DL->getInlinedAt())
              : nullptr;
  }

  /// Find or create a regular lexical scope.
  LexicalScope *getOrCreateRegularScope(const DILocalScope *Scope);

  /// Find or create an inlined lexical scope.
  LexicalScope *getOrCreateInlinedScope(const DILocalScope *Scope,
                                        const DILocation *InlinedAt);

  /// Extract instruction ranges for each lexical scopes
  /// for the given machine function.
  void extractLexicalScopes(SmallVectorImpl<InsnRange> &MIRanges,
                            DenseMap<const MachineInstr *, LexicalScope *> &M);
  void constructScopeNest(LexicalScope *Scope);
  void
  assignInstructionRanges(SmallVectorImpl<InsnRange> &MIRanges,
                          DenseMap<const MachineInstr *, LexicalScope *> &M);

  const MachineFunction *MF = nullptr;

  /// Mapping between DISubprograms and IR functions.
  DenseMap<const DISubprogram *, const Function *> FunctionMap;

  /// Tracks the scopes in the current function.
  // Use an unordered_map to ensure value pointer validity over insertion.
  std::unordered_map<const DILocalScope *, LexicalScope> LexicalScopeMap;

  /// Tracks inlined function scopes in current function.
  std::unordered_map<std::pair<const DILocalScope *, const DILocation *>,
                     LexicalScope,
                     pair_hash<const DILocalScope *, const DILocation *>>
      InlinedLexicalScopeMap;

  /// These scopes are  not included LexicalScopeMap.
  // Use an unordered_map to ensure value pointer validity over insertion.
  std::unordered_map<const DILocalScope *, LexicalScope> AbstractScopeMap;

  /// Tracks abstract scopes constructed while processing a function.
  SmallVector<LexicalScope *, 4> AbstractScopesList;

  /// Top level scope for the current function.
  LexicalScope *CurrentFnLexicalScope = nullptr;

  /// Map a location to the set of basic blocks it dominates. This is a cache
  /// for \ref LexicalScopes::getMachineBasicBlocks results.
  using BlockSetT = SmallPtrSet<const MachineBasicBlock *, 4>;
  DenseMap<const DILocation *, std::unique_ptr<BlockSetT>> DominatedBlocks;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_LEXICALSCOPES_H
