//===- TypeBasedAliasAnalysis.cpp - Type-Based Alias Analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TypeBasedAliasAnalysis pass, which implements
// metadata-based TBAA.
//
// In LLVM IR, memory does not have types, so LLVM's own type system is not
// suitable for doing TBAA. Instead, metadata is added to the IR to describe
// a type system of a higher level language. This can be used to implement
// typical C/C++ TBAA, but it can also be used to implement custom alias
// analysis behavior for other languages.
//
// The current metadata format is very simple. TBAA MDNodes have up to
// three fields, e.g.:
//   !0 = metadata !{ metadata !"an example type tree" }
//   !1 = metadata !{ metadata !"int", metadata !0 }
//   !2 = metadata !{ metadata !"float", metadata !0 }
//   !3 = metadata !{ metadata !"const float", metadata !2, i64 1 }
//
// The first field is an identity field. It can be any value, usually
// an MDString, which uniquely identifies the type. The most important
// name in the tree is the name of the root node. Two trees with
// different root node names are entirely disjoint, even if they
// have leaves with common names.
//
// The second field identifies the type's parent node in the tree, or
// is null or omitted for a root node. A type is considered to alias
// all of its descendants and all of its ancestors in the tree. Also,
// a type is considered to alias all types in other trees, so that
// bitcode produced from multiple front-ends is handled conservatively.
//
// If the third field is present, it's an integer which if equal to 1
// indicates that the type is "constant" (meaning pointsToConstantMemory
// should return true; see
// http://llvm.org/docs/AliasAnalysis.html#OtherItfs).
//
// TODO: The current metadata format doesn't support struct
// fields. For example:
//   struct X {
//     double d;
//     int i;
//   };
//   void foo(struct X *x, struct X *y, double *p) {
//     *x = *y;
//     *p = 0.0;
//   }
// Struct X has a double member, so the store to *x can alias the store to *p.
// Currently it's not possible to precisely describe all the things struct X
// aliases, so struct assignments must use conservative TBAA nodes. There's
// no scheme for attaching metadata to @llvm.memcpy yet either.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

// A handy option for disabling TBAA functionality. The same effect can also be
// achieved by stripping the !tbaa tags from IR, but this option is sometimes
// more convenient.
static cl::opt<bool> EnableTBAA("enable-tbaa", cl::init(true));

namespace {
  /// TBAANode - This is a simple wrapper around an MDNode which provides a
  /// higher-level interface by hiding the details of how alias analysis
  /// information is encoded in its operands.
  class TBAANode {
    const MDNode *Node;

  public:
    TBAANode() : Node(0) {}
    explicit TBAANode(const MDNode *N) : Node(N) {}

    /// getNode - Get the MDNode for this TBAANode.
    const MDNode *getNode() const { return Node; }

    /// getParent - Get this TBAANode's Alias tree parent.
    TBAANode getParent() const {
      if (Node->getNumOperands() < 2)
        return TBAANode();
      MDNode *P = dyn_cast_or_null<MDNode>(Node->getOperand(1));
      if (!P)
        return TBAANode();
      // Ok, this node has a valid parent. Return it.
      return TBAANode(P);
    }

    /// TypeIsImmutable - Test if this TBAANode represents a type for objects
    /// which are not modified (by any means) in the context where this
    /// AliasAnalysis is relevant.
    bool TypeIsImmutable() const {
      if (Node->getNumOperands() < 3)
        return false;
      ConstantInt *CI = dyn_cast<ConstantInt>(Node->getOperand(2));
      if (!CI)
        return false;
      return CI->getValue()[0];
    }
  };
}

namespace {
  /// TypeBasedAliasAnalysis - This is a simple alias analysis
  /// implementation that uses TypeBased to answer queries.
  class TypeBasedAliasAnalysis : public ImmutablePass,
                                 public AliasAnalysis {
  public:
    static char ID; // Class identification, replacement for typeinfo
    TypeBasedAliasAnalysis() : ImmutablePass(ID) {
      initializeTypeBasedAliasAnalysisPass(*PassRegistry::getPassRegistry());
    }

    virtual void initializePass() {
      InitializeAliasAnalysis(this);
    }

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    virtual void *getAdjustedAnalysisPointer(const void *PI) {
      if (PI == &AliasAnalysis::ID)
        return (AliasAnalysis*)this;
      return this;
    }

    bool Aliases(const MDNode *A, const MDNode *B) const;

  private:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual AliasResult alias(const Location &LocA, const Location &LocB);
    virtual bool pointsToConstantMemory(const Location &Loc, bool OrLocal);
    virtual ModRefBehavior getModRefBehavior(ImmutableCallSite CS);
    virtual ModRefBehavior getModRefBehavior(const Function *F);
    virtual ModRefResult getModRefInfo(ImmutableCallSite CS,
                                       const Location &Loc);
    virtual ModRefResult getModRefInfo(ImmutableCallSite CS1,
                                       ImmutableCallSite CS2);
  };
}  // End of anonymous namespace

// Register this pass...
char TypeBasedAliasAnalysis::ID = 0;
INITIALIZE_AG_PASS(TypeBasedAliasAnalysis, AliasAnalysis, "tbaa",
                   "Type-Based Alias Analysis", false, true, false)

ImmutablePass *llvm::createTypeBasedAliasAnalysisPass() {
  return new TypeBasedAliasAnalysis();
}

void
TypeBasedAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AliasAnalysis::getAnalysisUsage(AU);
}

/// Aliases - Test whether the type represented by A may alias the
/// type represented by B.
bool
TypeBasedAliasAnalysis::Aliases(const MDNode *A,
                                const MDNode *B) const {
  // Keep track of the root node for A and B.
  TBAANode RootA, RootB;

  // Climb the tree from A to see if we reach B.
  for (TBAANode T(A); ; ) {
    if (T.getNode() == B)
      // B is an ancestor of A.
      return true;

    RootA = T;
    T = T.getParent();
    if (!T.getNode())
      break;
  }

  // Climb the tree from B to see if we reach A.
  for (TBAANode T(B); ; ) {
    if (T.getNode() == A)
      // A is an ancestor of B.
      return true;

    RootB = T;
    T = T.getParent();
    if (!T.getNode())
      break;
  }

  // Neither node is an ancestor of the other.
  
  // If they have different roots, they're part of different potentially
  // unrelated type systems, so we must be conservative.
  if (RootA.getNode() != RootB.getNode())
    return true;

  // If they have the same root, then we've proved there's no alias.
  return false;
}

AliasAnalysis::AliasResult
TypeBasedAliasAnalysis::alias(const Location &LocA,
                              const Location &LocB) {
  if (!EnableTBAA)
    return AliasAnalysis::alias(LocA, LocB);

  // Get the attached MDNodes. If either value lacks a tbaa MDNode, we must
  // be conservative.
  const MDNode *AM = LocA.TBAATag;
  if (!AM) return AliasAnalysis::alias(LocA, LocB);
  const MDNode *BM = LocB.TBAATag;
  if (!BM) return AliasAnalysis::alias(LocA, LocB);

  // If they may alias, chain to the next AliasAnalysis.
  if (Aliases(AM, BM))
    return AliasAnalysis::alias(LocA, LocB);

  // Otherwise return a definitive result.
  return NoAlias;
}

bool TypeBasedAliasAnalysis::pointsToConstantMemory(const Location &Loc,
                                                    bool OrLocal) {
  if (!EnableTBAA)
    return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);

  const MDNode *M = Loc.TBAATag;
  if (!M) return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);

  // If this is an "immutable" type, we can assume the pointer is pointing
  // to constant memory.
  if (TBAANode(M).TypeIsImmutable())
    return true;

  return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);
}

AliasAnalysis::ModRefBehavior
TypeBasedAliasAnalysis::getModRefBehavior(ImmutableCallSite CS) {
  if (!EnableTBAA)
    return AliasAnalysis::getModRefBehavior(CS);

  ModRefBehavior Min = UnknownModRefBehavior;

  // If this is an "immutable" type, we can assume the call doesn't write
  // to memory.
  if (const MDNode *M = CS.getInstruction()->getMetadata(LLVMContext::MD_tbaa))
    if (TBAANode(M).TypeIsImmutable())
      Min = OnlyReadsMemory;

  return ModRefBehavior(AliasAnalysis::getModRefBehavior(CS) & Min);
}

AliasAnalysis::ModRefBehavior
TypeBasedAliasAnalysis::getModRefBehavior(const Function *F) {
  // Functions don't have metadata. Just chain to the next implementation.
  return AliasAnalysis::getModRefBehavior(F);
}

AliasAnalysis::ModRefResult
TypeBasedAliasAnalysis::getModRefInfo(ImmutableCallSite CS,
                                      const Location &Loc) {
  if (!EnableTBAA)
    return AliasAnalysis::getModRefInfo(CS, Loc);

  if (const MDNode *L = Loc.TBAATag)
    if (const MDNode *M =
          CS.getInstruction()->getMetadata(LLVMContext::MD_tbaa))
      if (!Aliases(L, M))
        return NoModRef;

  return AliasAnalysis::getModRefInfo(CS, Loc);
}

AliasAnalysis::ModRefResult
TypeBasedAliasAnalysis::getModRefInfo(ImmutableCallSite CS1,
                                      ImmutableCallSite CS2) {
  if (!EnableTBAA)
    return AliasAnalysis::getModRefInfo(CS1, CS2);

  if (const MDNode *M1 =
        CS1.getInstruction()->getMetadata(LLVMContext::MD_tbaa))
    if (const MDNode *M2 =
          CS2.getInstruction()->getMetadata(LLVMContext::MD_tbaa))
      if (!Aliases(M1, M2))
        return NoModRef;

  return AliasAnalysis::getModRefInfo(CS1, CS2);
}
