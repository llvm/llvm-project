//===- llvm/Analysis/AliasSetTracker.h - Build Alias Sets -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines two classes: AliasSetTracker and AliasSet.  These interface
// are used to classify a collection of pointer references into a maximal number
// of disjoint sets.  Each AliasSet object constructed by the AliasSetTracker
// object refers to memory disjoint from the other sets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIASSETTRACKER_H
#define LLVM_ANALYSIS_ALIASSETTRACKER_H

#include "llvm/Support/CallSite.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/hash_map.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

namespace llvm {

class AliasAnalysis;
class LoadInst;
class StoreInst;
class FreeInst;
class VAArgInst;
class AliasSetTracker;
class AliasSet;

class AliasSet : public ilist_node<AliasSet> {
  friend class AliasSetTracker;

  class PointerRec;
  typedef std::pair<Value* const, PointerRec> HashNodePair;

  class PointerRec {
    HashNodePair **PrevInList, *NextInList;
    AliasSet *AS;
    unsigned Size;
  public:
    PointerRec() : PrevInList(0), NextInList(0), AS(0), Size(0) {}

    HashNodePair *getNext() const { return NextInList; }
    bool hasAliasSet() const { return AS != 0; }

    HashNodePair** setPrevInList(HashNodePair **PIL) {
      PrevInList = PIL;
      return &NextInList;
    }

    void updateSize(unsigned NewSize) {
      if (NewSize > Size) Size = NewSize;
    }

    unsigned getSize() const { return Size; }

    AliasSet *getAliasSet(AliasSetTracker &AST) {
      assert(AS && "No AliasSet yet!");
      if (AS->Forward) {
        AliasSet *OldAS = AS;
        AS = OldAS->getForwardedTarget(AST);
        AS->addRef();
        OldAS->dropRef(AST);
      }
      return AS;
    }

    void setAliasSet(AliasSet *as) {
      assert(AS == 0 && "Already have an alias set!");
      AS = as;
    }

    void removeFromList() {
      if (NextInList) NextInList->second.PrevInList = PrevInList;
      *PrevInList = NextInList;
      if (AS->PtrListEnd == &NextInList) {
        AS->PtrListEnd = PrevInList;
        assert(*AS->PtrListEnd == 0 && "List not terminated right!");
      }
    }
  };

  HashNodePair *PtrList, **PtrListEnd;  // Doubly linked list of nodes
  AliasSet *Forward;             // Forwarding pointer
  AliasSet *Next, *Prev;         // Doubly linked list of AliasSets

  std::vector<CallSite> CallSites; // All calls & invokes in this node

  // RefCount - Number of nodes pointing to this AliasSet plus the number of
  // AliasSets forwarding to it.
  unsigned RefCount : 28;

  /// AccessType - Keep track of whether this alias set merely refers to the
  /// locations of memory, whether it modifies the memory, or whether it does
  /// both.  The lattice goes from "NoModRef" to either Refs or Mods, then to
  /// ModRef as necessary.
  ///
  enum AccessType {
    NoModRef = 0, Refs = 1,         // Ref = bit 1
    Mods     = 2, ModRef = 3        // Mod = bit 2
  };
  unsigned AccessTy : 2;

  /// AliasType - Keep track the relationships between the pointers in the set.
  /// Lattice goes from MustAlias to MayAlias.
  ///
  enum AliasType {
    MustAlias = 0, MayAlias = 1
  };
  unsigned AliasTy : 1;

  // Volatile - True if this alias set contains volatile loads or stores.
  bool Volatile : 1;

  void addRef() { ++RefCount; }
  void dropRef(AliasSetTracker &AST) {
    assert(RefCount >= 1 && "Invalid reference count detected!");
    if (--RefCount == 0)
      removeFromTracker(AST);
  }

public:
  /// Accessors...
  bool isRef() const { return AccessTy & Refs; }
  bool isMod() const { return AccessTy & Mods; }
  bool isMustAlias() const { return AliasTy == MustAlias; }
  bool isMayAlias()  const { return AliasTy == MayAlias; }

  // isVolatile - Return true if this alias set contains volatile loads or
  // stores.
  bool isVolatile() const { return Volatile; }

  /// isForwardingAliasSet - Return true if this alias set should be ignored as
  /// part of the AliasSetTracker object.
  bool isForwardingAliasSet() const { return Forward; }

  /// mergeSetIn - Merge the specified alias set into this alias set...
  ///
  void mergeSetIn(AliasSet &AS, AliasSetTracker &AST);

  // Alias Set iteration - Allow access to all of the pointer which are part of
  // this alias set...
  class iterator;
  iterator begin() const { return iterator(PtrList); }
  iterator end()   const { return iterator(); }
  bool empty() const { return PtrList == 0; }

  void print(std::ostream &OS) const;
  void print(std::ostream *OS) const { if (OS) print(*OS); }
  void dump() const;

  /// Define an iterator for alias sets... this is just a forward iterator.
  class iterator : public forward_iterator<HashNodePair, ptrdiff_t> {
    HashNodePair *CurNode;
  public:
    explicit iterator(HashNodePair *CN = 0) : CurNode(CN) {}

    bool operator==(const iterator& x) const {
      return CurNode == x.CurNode;
    }
    bool operator!=(const iterator& x) const { return !operator==(x); }

    const iterator &operator=(const iterator &I) {
      CurNode = I.CurNode;
      return *this;
    }

    value_type &operator*() const {
      assert(CurNode && "Dereferencing AliasSet.end()!");
      return *CurNode;
    }
    value_type *operator->() const { return &operator*(); }

    Value *getPointer() const { return CurNode->first; }
    unsigned getSize() const { return CurNode->second.getSize(); }

    iterator& operator++() {                // Preincrement
      assert(CurNode && "Advancing past AliasSet.end()!");
      CurNode = CurNode->second.getNext();
      return *this;
    }
    iterator operator++(int) { // Postincrement
      iterator tmp = *this; ++*this; return tmp;
    }
  };

private:
  // Can only be created by AliasSetTracker. Also, ilist creates one
  // to serve as a sentinel.
  friend struct ilist_sentinel_traits<AliasSet>;
  AliasSet() : PtrList(0), PtrListEnd(&PtrList), Forward(0), RefCount(0),
               AccessTy(NoModRef), AliasTy(MustAlias), Volatile(false) {
  }

  AliasSet(const AliasSet &AS);        // do not implement
  void operator=(const AliasSet &AS);  // do not implement

  HashNodePair *getSomePointer() const {
    return PtrList;
  }

  /// getForwardedTarget - Return the real alias set this represents.  If this
  /// has been merged with another set and is forwarding, return the ultimate
  /// destination set.  This also implements the union-find collapsing as well.
  AliasSet *getForwardedTarget(AliasSetTracker &AST) {
    if (!Forward) return this;

    AliasSet *Dest = Forward->getForwardedTarget(AST);
    if (Dest != Forward) {
      Dest->addRef();
      Forward->dropRef(AST);
      Forward = Dest;
    }
    return Dest;
  }

  void removeFromTracker(AliasSetTracker &AST);

  void addPointer(AliasSetTracker &AST, HashNodePair &Entry, unsigned Size,
                  bool KnownMustAlias = false);
  void addCallSite(CallSite CS, AliasAnalysis &AA);
  void removeCallSite(CallSite CS) {
    for (size_t i = 0, e = CallSites.size(); i != e; ++i)
      if (CallSites[i].getInstruction() == CS.getInstruction()) {
        CallSites[i] = CallSites.back();
        CallSites.pop_back();
      }
  }
  void setVolatile() { Volatile = true; }

  /// aliasesPointer - Return true if the specified pointer "may" (or must)
  /// alias one of the members in the set.
  ///
  bool aliasesPointer(const Value *Ptr, unsigned Size, AliasAnalysis &AA) const;
  bool aliasesCallSite(CallSite CS, AliasAnalysis &AA) const;
};

inline std::ostream& operator<<(std::ostream &OS, const AliasSet &AS) {
  AS.print(OS);
  return OS;
}


class AliasSetTracker {
  AliasAnalysis &AA;
  ilist<AliasSet> AliasSets;

  // Map from pointers to their node
  hash_map<Value*, AliasSet::PointerRec> PointerMap;
public:
  /// AliasSetTracker ctor - Create an empty collection of AliasSets, and use
  /// the specified alias analysis object to disambiguate load and store
  /// addresses.
  explicit AliasSetTracker(AliasAnalysis &aa) : AA(aa) {}
  ~AliasSetTracker() { clear(); }

  /// add methods - These methods are used to add different types of
  /// instructions to the alias sets.  Adding a new instruction can result in
  /// one of three actions happening:
  ///
  ///   1. If the instruction doesn't alias any other sets, create a new set.
  ///   2. If the instruction aliases exactly one set, add it to the set
  ///   3. If the instruction aliases multiple sets, merge the sets, and add
  ///      the instruction to the result.
  ///
  /// These methods return true if inserting the instruction resulted in the
  /// addition of a new alias set (i.e., the pointer did not alias anything).
  ///
  bool add(Value *Ptr, unsigned Size);  // Add a location
  bool add(LoadInst *LI);
  bool add(StoreInst *SI);
  bool add(FreeInst *FI);
  bool add(VAArgInst *VAAI);
  bool add(CallSite CS);          // Call/Invoke instructions
  bool add(CallInst *CI)   { return add(CallSite(CI)); }
  bool add(InvokeInst *II) { return add(CallSite(II)); }
  bool add(Instruction *I);       // Dispatch to one of the other add methods...
  void add(BasicBlock &BB);       // Add all instructions in basic block
  void add(const AliasSetTracker &AST); // Add alias relations from another AST

  /// remove methods - These methods are used to remove all entries that might
  /// be aliased by the specified instruction.  These methods return true if any
  /// alias sets were eliminated.
  bool remove(Value *Ptr, unsigned Size);  // Remove a location
  bool remove(LoadInst *LI);
  bool remove(StoreInst *SI);
  bool remove(FreeInst *FI);
  bool remove(VAArgInst *VAAI);
  bool remove(CallSite CS);
  bool remove(CallInst *CI)   { return remove(CallSite(CI)); }
  bool remove(InvokeInst *II) { return remove(CallSite(II)); }
  bool remove(Instruction *I);
  void remove(AliasSet &AS);
  
  void clear() {
    PointerMap.clear();
    AliasSets.clear();
  }

  /// getAliasSets - Return the alias sets that are active.
  ///
  const ilist<AliasSet> &getAliasSets() const { return AliasSets; }

  /// getAliasSetForPointer - Return the alias set that the specified pointer
  /// lives in.  If the New argument is non-null, this method sets the value to
  /// true if a new alias set is created to contain the pointer (because the
  /// pointer didn't alias anything).
  AliasSet &getAliasSetForPointer(Value *P, unsigned Size, bool *New = 0);

  /// getAliasSetForPointerIfExists - Return the alias set containing the
  /// location specified if one exists, otherwise return null.
  AliasSet *getAliasSetForPointerIfExists(Value *P, unsigned Size) {
    return findAliasSetForPointer(P, Size);
  }

  /// containsPointer - Return true if the specified location is represented by
  /// this alias set, false otherwise.  This does not modify the AST object or
  /// alias sets.
  bool containsPointer(Value *P, unsigned Size) const;

  /// getAliasAnalysis - Return the underlying alias analysis object used by
  /// this tracker.
  AliasAnalysis &getAliasAnalysis() const { return AA; }

  /// deleteValue method - This method is used to remove a pointer value from
  /// the AliasSetTracker entirely.  It should be used when an instruction is
  /// deleted from the program to update the AST.  If you don't use this, you
  /// would have dangling pointers to deleted instructions.
  ///
  void deleteValue(Value *PtrVal);

  /// copyValue - This method should be used whenever a preexisting value in the
  /// program is copied or cloned, introducing a new value.  Note that it is ok
  /// for clients that use this method to introduce the same value multiple
  /// times: if the tracker already knows about a value, it will ignore the
  /// request.
  ///
  void copyValue(Value *From, Value *To);


  typedef ilist<AliasSet>::iterator iterator;
  typedef ilist<AliasSet>::const_iterator const_iterator;

  const_iterator begin() const { return AliasSets.begin(); }
  const_iterator end()   const { return AliasSets.end(); }

  iterator begin() { return AliasSets.begin(); }
  iterator end()   { return AliasSets.end(); }

  void print(std::ostream &OS) const;
  void print(std::ostream *OS) const { if (OS) print(*OS); }
  void dump() const;

private:
  friend class AliasSet;
  void removeAliasSet(AliasSet *AS);

  AliasSet::HashNodePair &getEntryFor(Value *V) {
    // Standard operator[], except that it returns the whole pair, not just
    // ->second.
    return *PointerMap.insert(AliasSet::HashNodePair(V,
                                            AliasSet::PointerRec())).first;
  }

  AliasSet &addPointer(Value *P, unsigned Size, AliasSet::AccessType E,
                       bool &NewSet) {
    NewSet = false;
    AliasSet &AS = getAliasSetForPointer(P, Size, &NewSet);
    AS.AccessTy |= E;
    return AS;
  }
  AliasSet *findAliasSetForPointer(const Value *Ptr, unsigned Size);

  AliasSet *findAliasSetForCallSite(CallSite CS);
};

inline std::ostream& operator<<(std::ostream &OS, const AliasSetTracker &AST) {
  AST.print(OS);
  return OS;
}

} // End llvm namespace

#endif
