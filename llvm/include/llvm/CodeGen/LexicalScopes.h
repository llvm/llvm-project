//===- LexicalScopes.cpp - Collecting lexical scope info -*- C++ -*--------===//
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

#ifndef LLVM_CODEGEN_LEXICALSCOPES_H
#define LLVM_CODEGEN_LEXICALSCOPES_H

#include "llvm/Metadata.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLoc.h"
#include "llvm/Support/ValueHandle.h"
#include <utility>
namespace llvm {

class MachineInstr;
class MachineBasicBlock;
class MachineFunction;
class LexicalScope;

//===----------------------------------------------------------------------===//
/// InsnRange - This is used to track range of instructions with identical
/// lexical scope.
///
typedef std::pair<const MachineInstr *, const MachineInstr *> InsnRange;

//===----------------------------------------------------------------------===//
/// LexicalScopes -  This class provides interface to collect and use lexical
/// scoping information from machine instruction.
///
class LexicalScopes {
public:
  LexicalScopes() : MF(NULL),  CurrentFnLexicalScope(NULL) { }
  virtual ~LexicalScopes();

  /// initialize - Scan machine function and constuct lexical scope nest.
  virtual void initialize(const MachineFunction &);

  /// releaseMemory - release memory.
  virtual void releaseMemory();
  
  /// empty - Return true if there is any lexical scope information available.
  bool empty() { return CurrentFnLexicalScope == NULL; }

  /// isCurrentFunctionScope - Return true if given lexical scope represents 
  /// current function.
  bool isCurrentFunctionScope(const LexicalScope *LS) { 
    return LS == CurrentFnLexicalScope;
  }

  /// getCurrentFunctionScope - Return lexical scope for the current function.
  LexicalScope *getCurrentFunctionScope() const { return CurrentFnLexicalScope;}

  /// getMachineBasicBlocks - Populate given set using machine basic blocks
  /// which have machine instructions that belong to lexical scope identified by
  /// DebugLoc.
  void getMachineBasicBlocks(DebugLoc DL,
                             SmallPtrSet<const MachineBasicBlock*, 4> &MBBs);

  /// dominates - Return true if DebugLoc's lexical scope dominates at least one
  /// machine instruction's lexical scope in a given machine basic block.
  bool dominates(DebugLoc DL, MachineBasicBlock *MBB);

  /// findLexicalScope - Find lexical scope, either regular or inlined, for the
  /// given DebugLoc. Return NULL if not found.
  LexicalScope *findLexicalScope(DebugLoc DL);

  /// getAbstractScopesList - Return a reference to list of abstract scopes.
  ArrayRef<LexicalScope *> getAbstractScopesList() const {
    return AbstractScopesList;
  }

  /// findAbstractScope - Find an abstract scope or return NULL.
  LexicalScope *findAbstractScope(const MDNode *N) {
    return AbstractScopeMap.lookup(N);
  }

  /// findInlinedScope - Find an inlined scope for the given DebugLoc or return
  /// NULL.
  LexicalScope *findInlinedScope(DebugLoc DL) {
    return InlinedLexicalScopeMap.lookup(DL);
  }

  /// findLexicalScope - Find regular lexical scope or return NULL.
  LexicalScope *findLexicalScope(const MDNode *N) {
    return LexicalScopeMap.lookup(N);
  }

  /// dump - Print data structures to dbgs().
  void dump();

private:

  /// getOrCreateLexicalScope - Find lexical scope for the given DebugLoc. If
  /// not available then create new lexical scope.
  LexicalScope *getOrCreateLexicalScope(DebugLoc DL);

  /// getOrCreateRegularScope - Find or create a regular lexical scope.
  LexicalScope *getOrCreateRegularScope(MDNode *Scope);

  /// getOrCreateInlinedScope - Find or create an inlined lexical scope.
  LexicalScope *getOrCreateInlinedScope(MDNode *Scope, MDNode *InlinedAt);

  /// getOrCreateAbstractScope - Find or create an abstract lexical scope.
  LexicalScope *getOrCreateAbstractScope(const MDNode *N);

  /// extractLexicalScopes - Extract instruction ranges for each lexical scopes
  /// for the given machine function.
  void extractLexicalScopes(SmallVectorImpl<InsnRange> &MIRanges,
                            DenseMap<const MachineInstr *, LexicalScope *> &M);
  void constructScopeNest(LexicalScope *Scope);
  void assignInstructionRanges(SmallVectorImpl<InsnRange> &MIRanges,
                             DenseMap<const MachineInstr *, LexicalScope *> &M);

private:
  const MachineFunction *MF;

  /// LexicalScopeMap - Tracks the scopes in the current function.  Owns the
  /// contained LexicalScope*s.
  DenseMap<const MDNode *, LexicalScope *> LexicalScopeMap;

  /// InlinedLexicalScopeMap - Tracks inlined function scopes in current function.
  DenseMap<DebugLoc, LexicalScope *> InlinedLexicalScopeMap;

  /// AbstractScopeMap - These scopes are  not included LexicalScopeMap.  
  /// AbstractScopes owns its LexicalScope*s.
  DenseMap<const MDNode *, LexicalScope *> AbstractScopeMap;

  /// AbstractScopesList - Tracks abstract scopes constructed while processing
  /// a function. 
  SmallVector<LexicalScope *, 4>AbstractScopesList;

  /// CurrentFnLexicalScope - Top level scope for the current function.
  ///
  LexicalScope *CurrentFnLexicalScope;
};

//===----------------------------------------------------------------------===//
/// LexicalScope - This class is used to track scope information.
///
class LexicalScope {
  virtual void anchor();

public:
  LexicalScope(LexicalScope *P, const MDNode *D, const MDNode *I, bool A)
    : Parent(P), Desc(D), InlinedAtLocation(I), AbstractScope(A),
      LastInsn(0), FirstInsn(0), DFSIn(0), DFSOut(0) {
#ifndef NDEBUG
    IndentLevel = 0;
#endif
    if (Parent)
      Parent->addChild(this);
  }

  virtual ~LexicalScope() {}

  // Accessors.
  LexicalScope *getParent() const               { return Parent; }
  const MDNode *getDesc() const                 { return Desc; }
  const MDNode *getInlinedAt() const            { return InlinedAtLocation; }
  const MDNode *getScopeNode() const            { return Desc; }
  bool isAbstractScope() const                  { return AbstractScope; }
  SmallVector<LexicalScope *, 4> &getChildren() { return Children; }
  SmallVector<InsnRange, 4> &getRanges()        { return Ranges; }

  /// addChild - Add a child scope.
  void addChild(LexicalScope *S) { Children.push_back(S); }

  /// openInsnRange - This scope covers instruction range starting from MI.
  void openInsnRange(const MachineInstr *MI) {
    if (!FirstInsn)
      FirstInsn = MI;

    if (Parent)
      Parent->openInsnRange(MI);
  }

  /// extendInsnRange - Extend the current instruction range covered by
  /// this scope.
  void extendInsnRange(const MachineInstr *MI) {
    assert (FirstInsn && "MI Range is not open!");
    LastInsn = MI;
    if (Parent)
      Parent->extendInsnRange(MI);
  }

  /// closeInsnRange - Create a range based on FirstInsn and LastInsn collected
  /// until now. This is used when a new scope is encountered while walking
  /// machine instructions.
  void closeInsnRange(LexicalScope *NewScope = NULL) {
    assert (LastInsn && "Last insn missing!");
    Ranges.push_back(InsnRange(FirstInsn, LastInsn));
    FirstInsn = NULL;
    LastInsn = NULL;
    // If Parent dominates NewScope then do not close Parent's instruction
    // range.
    if (Parent && (!NewScope || !Parent->dominates(NewScope)))
      Parent->closeInsnRange(NewScope);
  }

  /// dominates - Return true if current scope dominates given lexical scope.
  bool dominates(const LexicalScope *S) const {
    if (S == this)
      return true;
    if (DFSIn < S->getDFSIn() && DFSOut > S->getDFSOut())
      return true;
    return false;
  }

  // Depth First Search support to walk and manipulate LexicalScope hierarchy.
  unsigned getDFSOut() const            { return DFSOut; }
  void setDFSOut(unsigned O)            { DFSOut = O; }
  unsigned getDFSIn() const             { return DFSIn; }
  void setDFSIn(unsigned I)             { DFSIn = I; }

  /// dump - print lexical scope.
  void dump() const;

private:
  LexicalScope *Parent;                          // Parent to this scope.
  AssertingVH<const MDNode> Desc;                // Debug info descriptor.
  AssertingVH<const MDNode> InlinedAtLocation;   // Location at which this 
                                                 // scope is inlined.
  bool AbstractScope;                            // Abstract Scope
  SmallVector<LexicalScope *, 4> Children;       // Scopes defined in scope.  
                                                 // Contents not owned.
  SmallVector<InsnRange, 4> Ranges;

  const MachineInstr *LastInsn;       // Last instruction of this scope.
  const MachineInstr *FirstInsn;      // First instruction of this scope.
  unsigned DFSIn, DFSOut;             // In & Out Depth use to determine
                                      // scope nesting.
#ifndef NDEBUG
  mutable unsigned IndentLevel;       // Private state for dump()
#endif
};

} // end llvm namespace

#endif
