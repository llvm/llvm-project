//===- MayAliasSet.h  - May-alias Set for Base Pointers ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines two classes: MayAliasSet and MayAliasSetInfo.
// MayAliasSet contains the base pointers of access functions in SCoP that
// may/must alias each others. And MayAliasSetInfo will compute and hold these
// MayAliasSets in every SCoP in a function.
//
// The difference between MayAliasSet and the original LLVM AliasSet is that
// the LLVM AliasSets are disjoint, but MayAliasSets are not.
//
// Suppose we have the following LLVM IR:
// define void @f(i32* noalias nocapture %a, i32* noalias nocapture %b)nounwind{
// bb.nph:
//   %0 = tail call i32 (...)* @rnd() nounwind
//   %1 = icmp eq i32 %0, 0
//   %ptr0 = select i1 %1, i32* %b, i32* %a
//   %2 = load i32* %ptr0, align 4
//   %3 = load i32* %a, align 4
//   %4 = load i32* %b, align 4
//   ret void
// }
//
// The LLVM AliasSetTracker constructs only one LLVM AliasSet that contains
// ptr0, a and b, but MayAliasSetInfo is supposed to build two MayAliasSets:
// {a, ptr0} and {b, ptr0}.
//
// Take the above LLVM IR for example, the MayAliasSetInfo builds two set:
// A: {a, ptr0} and B: {b, ptr0} and constructs base pointer to MayAliasSet
// mapping like:
// a -> A
// b -> B
// ptr0 -> A, B
// 
// After that, SCoPInfo pass will build a access function for each MayAliasSet,
// so "%2 = load i32* %ptr0, align 4" will be translated to "read A" and
// "read B", while "%3 = load i32* %a, align 4" will be translated to "read A",
// and "%4 = load i32* %b, align 4" will be translated to "read B". This means
// we can treat the MayAliasSet as the identifier of the virtual array of memory
// access in SCoPs.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_MAY_ALIAS_SET_H
#define POLLY_MAY_ALIAS_SET_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Allocator.h"
#include <map>

namespace llvm {
  class Value;
  class AliasAnalysis;
  class raw_ostream;
}

using namespace llvm;

namespace polly {
class MayAliasSetInfo;
class TempScop;

//===----------------------------------------------------------------------===//
/// @brief MayAliasSet of pointers in SCoPs.
///
/// Note: Pointers in MayAliasSet only must-alias with each other now.
class MayAliasSet {
  // DO NOT IMPLEMENT
  MayAliasSet(const MayAliasSet &);
  // DO NOT IMPLEMENT
  const MayAliasSet &operator=(const MayAliasSet &);

  // TODO: Use CallbackVH to update the set when some base pointers are deleted
  // by some pass.
  SmallPtrSet<const Value*, 8> MustAliasPtrs;

  MayAliasSet() {}

  friend class MayAliasSetInfo;
public:

  /// @name Must Alias Pointer Iterators
  ///
  /// These iterators iterate over all must alias pointers in the set.
  //@{
  typedef SmallPtrSetIterator<const Value*> const_iterator;
  const_iterator mustalias_begin() const { return MustAliasPtrs.begin(); }
  const_iterator mustalias_end() const {  return MustAliasPtrs.end(); }
  //@}

  /// @brief Add a must alias pointer to this set.
  ///
  /// @param V The pointer to add.
  void addMustAliasPtr(const Value* V) { MustAliasPtrs.insert(V); }

  void print(raw_ostream &OS) const;
  void dump() const;
};

//===----------------------------------------------------------------------===//
/// @brief Compute and manage the may-alias sets in a TempSCoP or SCoP.
class MayAliasSetInfo {
  // DO NOT IMPLEMENT
  MayAliasSetInfo(const MayAliasSetInfo &);
  // DO NOT IMPLEMENT
  const MayAliasSetInfo &operator=(const MayAliasSetInfo &);

  SpecificBumpPtrAllocator<MayAliasSet> MayASAllocator;

  // Mapping the pointers to their may-alias sets.
  typedef std::multimap<const Value*, MayAliasSet*> MayAliasSetMapType;
  MayAliasSetMapType BasePtrMap;

public:
  MayAliasSetInfo() {}

  /// @name MayAliasSet Iterators
  ///
  /// These iterators iterate over all may-alias sets referring to a base
  /// pointer.
  //@{
  typedef MayAliasSetMapType::iterator alias_iterator;
  typedef MayAliasSetMapType::const_iterator const_alias_iterator;
  
  alias_iterator alias_begin(const Value *BasePtr) {
    return BasePtrMap.lower_bound(BasePtr);
  }

  alias_iterator alias_end(const Value *BasePtr) {
    return BasePtrMap.upper_bound(BasePtr);
  }

  const_alias_iterator alias_begin(const Value *BasePtr) const {
    return BasePtrMap.lower_bound(BasePtr);
  }

  const_alias_iterator alias_end(const Value *BasePtr) const {
    return BasePtrMap.upper_bound(BasePtr);
  }
  //@}


  /// @brief Build MayAliasSets in a SCoP.
  ///
  /// @param Scop The SCoP to build MayAliasSets in.
  /// @param AA   The AliasAnalaysis provides the alias information.
  void buildMayAliasSets(TempScop &Scop, AliasAnalysis &AA);
};
}

#endif
