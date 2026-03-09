//===- DAGISelMatcher.h - Representation of DAG pattern matcher -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_COMMON_DAGISELMATCHER_H
#define LLVM_UTILS_TABLEGEN_COMMON_DAGISELMATCHER_H

#include "Common/InfoByHwMode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace llvm {
class CodeGenRegister;
class CodeGenDAGPatterns;
class CodeGenInstruction;
class Matcher;
class MatcherList;
class PatternToMatch;
class raw_ostream;
class ComplexPattern;
class Record;
class SDNodeInfo;
class TreePredicateFn;
class TreePattern;

MatcherList ConvertPatternToMatcherList(const PatternToMatch &Pattern,
                                        unsigned Variant,
                                        const CodeGenDAGPatterns &CGP);
void OptimizeMatcher(MatcherList &ML, const CodeGenDAGPatterns &CGP);
void EmitMatcherTable(MatcherList &ML, const CodeGenDAGPatterns &CGP,
                      raw_ostream &OS);

/// Base class that holds a pointer to the next entry in the MatcherList.
/// Separated from Matcher so that we can have an instance of it in
/// MatcherList.
class MatcherBase {
  friend class MatcherList;

  Matcher *Next = nullptr;
};

/// Matcher - Base class for all the DAG ISel Matcher representation
/// nodes.
class Matcher : public MatcherBase {
  virtual void anchor();

public:
  enum KindTy {
    // Matcher state manipulation.
    Scope,            // Push a checking scope.
    RecordNode,       // Record the current node.
    RecordChild,      // Record a child of the current node.
    RecordMemRef,     // Record the memref in the current node.
    CaptureGlueInput, // If the current node has an input glue, save it.
    MoveChild,        // Move current node to specified child.
    MoveSibling,      // Move current node to specified sibling.
    MoveParent,       // Move current node to parent.

    // Predicate checking.
    CheckSame,      // Fail if not same as prev match.
    CheckChildSame, // Fail if child not same as prev match.
    CheckPatternPredicate,
    CheckPredicate,      // Fail if node predicate fails.
    CheckOpcode,         // Fail if not opcode.
    SwitchOpcode,        // Dispatch based on opcode.
    CheckType,           // Fail if not correct type.
    SwitchType,          // Dispatch based on type.
    CheckChildType,      // Fail if child has wrong type.
    CheckInteger,        // Fail if wrong val.
    CheckChildInteger,   // Fail if child is wrong val.
    CheckCondCode,       // Fail if not condcode.
    CheckChild2CondCode, // Fail if child is wrong condcode.
    CheckValueType,
    CheckComplexPat,
    CheckAndImm,
    CheckOrImm,
    CheckImmAllOnesV,
    CheckImmAllZerosV,
    CheckFoldableChainNode,

    // Node creation/emisssion.
    EmitInteger,          // Create a TargetConstant
    EmitRegister,         // Create a register.
    EmitConvertToTarget,  // Convert a imm/fpimm to target imm/fpimm
    EmitMergeInputChains, // Merge together a chains for an input.
    EmitCopyToReg,        // Emit a copytoreg into a physreg.
    EmitNode,             // Create a DAG node
    EmitNodeXForm,        // Run a SDNodeXForm
    CompleteMatch,        // Finish a match and update the results.
    MorphNodeTo,          // Build a node, finish a match and update results.

    // Highest enum value; watch out when adding more.
    HighestKind = MorphNodeTo
  };
  const KindTy Kind;

protected:
  Matcher(KindTy K) : Kind(K) {}

public:
  virtual ~Matcher() = default;

  KindTy getKind() const { return Kind; }

  bool isEqual(const Matcher *M) const {
    if (getKind() != M->getKind())
      return false;
    return isEqualImpl(M);
  }

  /// isSimplePredicateNode - Return true if this is a simple predicate that
  /// operates on the node or its children without potential side effects or a
  /// change of the current node.
  bool isSimplePredicateNode() const {
    switch (getKind()) {
    default:
      return false;
    case CheckSame:
    case CheckChildSame:
    case CheckPatternPredicate:
    case CheckPredicate:
    case CheckOpcode:
    case CheckType:
    case CheckChildType:
    case CheckInteger:
    case CheckChildInteger:
    case CheckCondCode:
    case CheckChild2CondCode:
    case CheckValueType:
    case CheckAndImm:
    case CheckOrImm:
    case CheckImmAllOnesV:
    case CheckImmAllZerosV:
    case CheckFoldableChainNode:
      return true;
    }
  }

  /// isSimplePredicateOrRecordNode - Return true if this is a record node or
  /// a simple predicate.
  bool isSimplePredicateOrRecordNode() const {
    return isSimplePredicateNode() || getKind() == RecordNode ||
           getKind() == RecordChild;
  }

  /// canMoveBeforeNode - Return true if it is safe to move the current
  /// matcher across the specified one.
  bool canMoveBeforeNode(const Matcher *Other) const;

  /// isContradictory - Return true of these two matchers could never match on
  /// the same node.
  bool isContradictory(const Matcher *Other) const {
    // Since this predicate is reflexive, we canonicalize the ordering so that
    // we always match a node against nodes with kinds that are greater or
    // equal to them.  For example, we'll pass in a CheckType node as an
    // argument to the CheckOpcode method, not the other way around.
    if (getKind() < Other->getKind())
      return isContradictoryImpl(Other);
    return Other->isContradictoryImpl(this);
  }

  void printOne(raw_ostream &OS, indent Indent = indent(0)) const;
  void dump() const;

protected:
  virtual void printImpl(raw_ostream &OS, indent Indent) const = 0;
  virtual bool isEqualImpl(const Matcher *M) const = 0;
  virtual bool isContradictoryImpl(const Matcher *M) const { return false; }
};

/// Manages a singly linked list of Matcher objects. Interface based on
/// std::forward_list. Once a Matcher is added to a list, it cannot be removed.
/// It can only be erased or spliced to another position in this list or another
/// list.
class MatcherList {
  MatcherBase BeforeBegin;

  // Emitted size of all of the nodes in this list.
  unsigned Size = 0;

public:
  MatcherList() = default;
  MatcherList(const MatcherList &RHS) = delete;
  MatcherList(MatcherList &&RHS) {
    splice_after(before_begin(), RHS);
    Size = RHS.Size;
    RHS.Size = 0;
  }
  ~MatcherList() { clear(); }

  MatcherList &operator=(const MatcherList &) = delete;
  MatcherList &operator=(MatcherList &&RHS) {
    clear();
    splice_after(before_begin(), RHS);
    Size = RHS.Size;
    RHS.Size = 0;
    return *this;
  }

  void clear() {
    for (Matcher *P = BeforeBegin.Next; P != nullptr;) {
      Matcher *Next = P->Next;
      delete P;
      P = Next;
    }
    BeforeBegin.Next = nullptr;
    Size = 0;
  }

  template <bool IsConst> class iterator_impl {
    friend class MatcherList;
    using Base = std::conditional_t<IsConst, const MatcherBase, MatcherBase>;

    Base *Pointer;

    explicit iterator_impl(Base *P) { Pointer = P; }

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::conditional_t<IsConst, const Matcher *, Matcher *>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = value_type &;

    iterator_impl &operator++() {
      Pointer = Pointer->Next;
      return *this;
    }

    iterator_impl operator++(int) {
      iterator Tmp(*this);
      Pointer = Pointer->Next;
      return Tmp;
    }

    value_type operator*() const { return static_cast<value_type>(Pointer); }

    value_type operator->() const { return operator*(); }

    bool operator==(const iterator_impl &X) const {
      return Pointer == X.Pointer;
    }
    bool operator!=(const iterator_impl &X) const { return !operator==(X); }

    // Allow conversion to a const iterator.
    operator iterator_impl<true>() const {
      return iterator_impl<true>(Pointer);
    }
  };

  using iterator = iterator_impl<false>;
  using const_iterator = iterator_impl<true>;

  /// Return an iterator before the first Matcher in the list. This iterator
  /// cannot be dereferenced. Incrementing returns the iterator to begin().
  iterator before_begin() { return iterator(&BeforeBegin); }
  const_iterator before_begin() const { return const_iterator(&BeforeBegin); }

  iterator begin() { return iterator(BeforeBegin.Next); }
  const_iterator begin() const { return const_iterator(BeforeBegin.Next); }

  iterator end() { return iterator(nullptr); }
  const_iterator end() const { return const_iterator(nullptr); }

  Matcher *front() { return *begin(); }
  const Matcher *front() const { return *begin(); }

  bool empty() const { return BeforeBegin.Next == nullptr; }

  void push_front(Matcher *M) { insert_after(before_begin(), M); }

  /// Delete the first Matcher from the list.
  void pop_front() {
    assert(Size == 0 && "Should not modify list once size is set");
    assert(!empty());
    Matcher *N = BeforeBegin.Next;
    BeforeBegin.Next = N->Next;
    delete N;
  }

  /// Insert the matcher \p N into this list after \p Pos.
  iterator insert_after(iterator Pos, Matcher *N) {
    assert(Size == 0 && "Should not modify list once size is set");
    N->Next = Pos.Pointer->Next;
    Pos.Pointer->Next = N;
    return iterator(N);
  }

  /// Insert Matchers in the range [F, L) into this list.
  template <class InIt> iterator insert_after(iterator Pos, InIt F, InIt L) {
    MatcherBase *R = Pos.Pointer;
    if (F != L) {
      Matcher *First = *F;
      Matcher *Last = First;

      // Link the Matchers together.
      for (++F; F != L; ++F, Last = Last->Next)
        Last->Next = *F;

      // Insert them into the list.
      Last->Next = R->Next;
      R->Next = First;
      R = Last;
    }

    return iterator(R);
  }

  /// Insert multiple matchers into this list.
  iterator insert_after(iterator Pos, std::initializer_list<Matcher *> IL) {
    return insert_after(Pos, IL.begin(), IL.end());
  }

  /// Erase the Matcher after \p Pos.
  iterator erase_after(iterator Pos) {
    assert(Size == 0 && "Should not modify list once size is set");
    MatcherBase *P = Pos.Pointer;
    Matcher *N = P->Next;
    P->Next = N->Next;
    delete N;
    return iterator(P->Next);
  }

  iterator erase_after(iterator F, iterator L) {
    Matcher *E = static_cast<Matcher *>(L.Pointer);
    if (F != L) {
      Matcher *N = F.Pointer->Next;
      if (N != E) {
        F.Pointer->Next = E;
        do {
          Matcher *Tmp = N->Next;
          delete N;
          N = Tmp;
        } while (N != E);
      }
    }
    return iterator(E);
  }

  /// Splice the contents of list \p X after \p Pos in this list.
  void splice_after(iterator Pos, MatcherList &X) {
    assert(Size == 0 && "Should not modify list once size is set");
    if (!X.empty()) {
      if (Pos.Pointer->Next != nullptr) {
        auto LM1 = X.before_begin();
        while (LM1.Pointer->Next != nullptr)
          ++LM1;
        LM1.Pointer->Next = Pos.Pointer->Next;
      }
      Pos.Pointer->Next = X.BeforeBegin.Next;
      X.BeforeBegin.Next = nullptr;
    }
  }

  /// Splice the Matcher after \p I into this list after \p Pos.
  void splice_after(iterator Pos, MatcherList &, iterator I) {
    assert(Size == 0 && "Should not modify list once size is set");
    auto LM1 = std::next(I);
    if (Pos != I && Pos != LM1) {
      I.Pointer->Next = LM1.Pointer->Next;
      LM1.Pointer->Next = Pos.Pointer->Next;
      Pos.Pointer->Next = static_cast<Matcher *>(LM1.Pointer);
    }
  }

  /// Splice the Matchers in the range (\p F, \p L) into this list after \p Pos.
  void splice_after(iterator Pos, MatcherList &, iterator F, iterator L) {
    assert(Size == 0 && "Should not modify list once size is set");
    if (F != L && Pos != F) {
      auto LM1 = F;
      while (LM1.Pointer->Next != L.Pointer)
        ++LM1;
      if (F != LM1) {
        LM1.Pointer->Next = Pos.Pointer->Next;
        Pos.Pointer->Next = F.Pointer->Next;
        F.Pointer->Next = static_cast<Matcher *>(L.Pointer);
      }
    }
  }

  void setSize(unsigned Sz) { Size = Sz; }
  unsigned getSize() const { return Size; }

  void print(raw_ostream &OS, indent Indent = indent(0)) const;
  void dump() const;
};

/// ScopeMatcher - This attempts to match each of its children to find the first
/// one that successfully matches.  If one child fails, it tries the next child.
/// If none of the children match then this check fails.  It never has a 'next'.
class ScopeMatcher : public Matcher {
  SmallVector<MatcherList, 4> Children;

public:
  ScopeMatcher(SmallVectorImpl<MatcherList> &&children)
      : Matcher(Scope), Children(std::move(children)) {}

  unsigned getNumChildren() const { return Children.size(); }

  MatcherList &getChild(unsigned i) { return Children[i]; }
  const MatcherList &getChild(unsigned i) const { return Children[i]; }

  SmallVectorImpl<MatcherList> &getChildren() { return Children; }

  static bool classof(const Matcher *N) { return N->getKind() == Scope; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return false; }
};

/// RecordMatcher - Save the current node in the operand list.
class RecordMatcher : public Matcher {
  /// WhatFor - This is a string indicating why we're recording this.  This
  /// should only be used for comment generation not anything semantic.
  std::string WhatFor;

  /// ResultNo - The slot number in the RecordedNodes vector that this will be,
  /// just printed as a comment.
  unsigned ResultNo;

public:
  RecordMatcher(const std::string &whatfor, unsigned resultNo)
      : Matcher(RecordNode), WhatFor(whatfor), ResultNo(resultNo) {}

  const std::string &getWhatFor() const { return WhatFor; }
  unsigned getResultNo() const { return ResultNo; }

  static bool classof(const Matcher *N) { return N->getKind() == RecordNode; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return true; }
};

/// RecordChildMatcher - Save a numbered child of the current node, or fail
/// the match if it doesn't exist.  This is logically equivalent to:
///    MoveChild N + RecordNode + MoveParent.
class RecordChildMatcher : public Matcher {
  unsigned ChildNo;

  /// WhatFor - This is a string indicating why we're recording this.  This
  /// should only be used for comment generation not anything semantic.
  std::string WhatFor;

  /// ResultNo - The slot number in the RecordedNodes vector that this will be,
  /// just printed as a comment.
  unsigned ResultNo;

public:
  RecordChildMatcher(unsigned childno, const std::string &whatfor,
                     unsigned resultNo)
      : Matcher(RecordChild), ChildNo(childno), WhatFor(whatfor),
        ResultNo(resultNo) {}

  unsigned getChildNo() const { return ChildNo; }
  const std::string &getWhatFor() const { return WhatFor; }
  unsigned getResultNo() const { return ResultNo; }

  static bool classof(const Matcher *N) { return N->getKind() == RecordChild; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<RecordChildMatcher>(M)->getChildNo() == getChildNo();
  }
};

/// RecordMemRefMatcher - Save the current node's memref.
class RecordMemRefMatcher : public Matcher {
public:
  RecordMemRefMatcher() : Matcher(RecordMemRef) {}

  static bool classof(const Matcher *N) { return N->getKind() == RecordMemRef; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return true; }
};

/// CaptureGlueInputMatcher - If the current record has a glue input, record
/// it so that it is used as an input to the generated code.
class CaptureGlueInputMatcher : public Matcher {
public:
  CaptureGlueInputMatcher() : Matcher(CaptureGlueInput) {}

  static bool classof(const Matcher *N) {
    return N->getKind() == CaptureGlueInput;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return true; }
};

/// MoveChildMatcher - This tells the interpreter to move into the
/// specified child node.
class MoveChildMatcher : public Matcher {
  unsigned ChildNo;

public:
  MoveChildMatcher(unsigned childNo) : Matcher(MoveChild), ChildNo(childNo) {}

  unsigned getChildNo() const { return ChildNo; }

  static bool classof(const Matcher *N) { return N->getKind() == MoveChild; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<MoveChildMatcher>(M)->getChildNo() == getChildNo();
  }
};

/// MoveSiblingMatcher - This tells the interpreter to move into the
/// specified sibling node.
class MoveSiblingMatcher : public Matcher {
  unsigned SiblingNo;

public:
  MoveSiblingMatcher(unsigned SiblingNo)
      : Matcher(MoveSibling), SiblingNo(SiblingNo) {}

  unsigned getSiblingNo() const { return SiblingNo; }

  static bool classof(const Matcher *N) { return N->getKind() == MoveSibling; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<MoveSiblingMatcher>(M)->getSiblingNo() == getSiblingNo();
  }
};

/// MoveParentMatcher - This tells the interpreter to move to the parent
/// of the current node.
class MoveParentMatcher : public Matcher {
public:
  MoveParentMatcher() : Matcher(MoveParent) {}

  static bool classof(const Matcher *N) { return N->getKind() == MoveParent; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return true; }
};

/// CheckSameMatcher - This checks to see if this node is exactly the same
/// node as the specified match that was recorded with 'Record'.  This is used
/// when patterns have the same name in them, like '(mul GPR:$in, GPR:$in)'.
class CheckSameMatcher : public Matcher {
  unsigned MatchNumber;

public:
  CheckSameMatcher(unsigned matchnumber)
      : Matcher(CheckSame), MatchNumber(matchnumber) {}

  unsigned getMatchNumber() const { return MatchNumber; }

  static bool classof(const Matcher *N) { return N->getKind() == CheckSame; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckSameMatcher>(M)->getMatchNumber() == getMatchNumber();
  }
};

/// CheckChildSameMatcher - This checks to see if child node is exactly the same
/// node as the specified match that was recorded with 'Record'.  This is used
/// when patterns have the same name in them, like '(mul GPR:$in, GPR:$in)'.
class CheckChildSameMatcher : public Matcher {
  unsigned ChildNo;
  unsigned MatchNumber;

public:
  CheckChildSameMatcher(unsigned childno, unsigned matchnumber)
      : Matcher(CheckChildSame), ChildNo(childno), MatchNumber(matchnumber) {}

  unsigned getChildNo() const { return ChildNo; }
  unsigned getMatchNumber() const { return MatchNumber; }

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckChildSame;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckChildSameMatcher>(M)->ChildNo == ChildNo &&
           cast<CheckChildSameMatcher>(M)->MatchNumber == MatchNumber;
  }
};

/// CheckPatternPredicateMatcher - This checks the target-specific predicate
/// to see if the entire pattern is capable of matching.  This predicate does
/// not take a node as input.  This is used for subtarget feature checks etc.
class CheckPatternPredicateMatcher : public Matcher {
  std::string Predicate;

public:
  CheckPatternPredicateMatcher(StringRef predicate)
      : Matcher(CheckPatternPredicate), Predicate(predicate) {}

  StringRef getPredicate() const { return Predicate; }

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckPatternPredicate;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckPatternPredicateMatcher>(M)->getPredicate() == Predicate;
  }
};

/// CheckPredicateMatcher - This checks the target-specific predicate to
/// see if the node is acceptable.
class CheckPredicateMatcher : public Matcher {
  TreePattern *Pred;
  const SmallVector<unsigned, 4> Operands;

public:
  CheckPredicateMatcher(const TreePredicateFn &pred,
                        ArrayRef<unsigned> Operands);

  TreePredicateFn getPredicate() const;
  unsigned getNumOperands() const;
  unsigned getOperandNo(unsigned i) const;

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckPredicate;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckPredicateMatcher>(M)->Pred == Pred;
  }
};

/// CheckOpcodeMatcher - This checks to see if the current node has the
/// specified opcode, if not it fails to match.
class CheckOpcodeMatcher : public Matcher {
  const SDNodeInfo &Opcode;

public:
  CheckOpcodeMatcher(const SDNodeInfo &opcode)
      : Matcher(CheckOpcode), Opcode(opcode) {}

  const SDNodeInfo &getOpcode() const { return Opcode; }

  static bool classof(const Matcher *N) { return N->getKind() == CheckOpcode; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override;
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// SwitchOpcodeMatcher - Switch based on the current node's opcode, dispatching
/// to one matcher per opcode.  If the opcode doesn't match any of the cases,
/// then the match fails.  This is semantically equivalent to a Scope node where
/// every child does a CheckOpcode, but is much faster.
class SwitchOpcodeMatcher : public Matcher {
  SmallVector<std::pair<const SDNodeInfo *, MatcherList>, 8> Cases;

public:
  SwitchOpcodeMatcher(
      SmallVectorImpl<std::pair<const SDNodeInfo *, MatcherList>> &&cases)
      : Matcher(SwitchOpcode), Cases(std::move(cases)) {}

  static bool classof(const Matcher *N) { return N->getKind() == SwitchOpcode; }

  unsigned getNumCases() const { return Cases.size(); }

  const SDNodeInfo &getCaseOpcode(unsigned i) const { return *Cases[i].first; }
  MatcherList &getCaseMatcher(unsigned i) { return Cases[i].second; }
  const MatcherList &getCaseMatcher(unsigned i) const {
    return Cases[i].second;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return false; }
};

/// CheckTypeMatcher - This checks to see if the current node has the
/// specified type at the specified result, if not it fails to match.
class CheckTypeMatcher : public Matcher {
  ValueTypeByHwMode Type;
  unsigned ResNo;

public:
  CheckTypeMatcher(ValueTypeByHwMode type, unsigned resno)
      : Matcher(CheckType), Type(std::move(type)), ResNo(resno) {}

  const ValueTypeByHwMode &getType() const { return Type; }
  unsigned getResNo() const { return ResNo; }

  static bool classof(const Matcher *N) { return N->getKind() == CheckType; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckTypeMatcher>(M)->Type == Type;
  }
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// SwitchTypeMatcher - Switch based on the current node's type, dispatching
/// to one matcher per case.  If the type doesn't match any of the cases,
/// then the match fails.  This is semantically equivalent to a Scope node where
/// every child does a CheckType, but is much faster.
class SwitchTypeMatcher : public Matcher {
  SmallVector<std::pair<MVT, MatcherList>, 8> Cases;

public:
  SwitchTypeMatcher(SmallVectorImpl<std::pair<MVT, MatcherList>> &&cases)
      : Matcher(SwitchType), Cases(std::move(cases)) {}

  static bool classof(const Matcher *N) { return N->getKind() == SwitchType; }

  unsigned getNumCases() const { return Cases.size(); }

  MVT getCaseType(unsigned i) const { return Cases[i].first; }
  MatcherList &getCaseMatcher(unsigned i) { return Cases[i].second; }
  const MatcherList &getCaseMatcher(unsigned i) const {
    return Cases[i].second;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return false; }
};

/// CheckChildTypeMatcher - This checks to see if a child node has the
/// specified type, if not it fails to match.
class CheckChildTypeMatcher : public Matcher {
  unsigned ChildNo;
  ValueTypeByHwMode Type;

public:
  CheckChildTypeMatcher(unsigned childno, ValueTypeByHwMode type)
      : Matcher(CheckChildType), ChildNo(childno), Type(std::move(type)) {}

  unsigned getChildNo() const { return ChildNo; }
  const ValueTypeByHwMode &getType() const { return Type; }

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckChildType;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckChildTypeMatcher>(M)->ChildNo == ChildNo &&
           cast<CheckChildTypeMatcher>(M)->Type == Type;
  }
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// CheckIntegerMatcher - This checks to see if the current node is a
/// ConstantSDNode with the specified integer value, if not it fails to match.
class CheckIntegerMatcher : public Matcher {
  int64_t Value;

public:
  CheckIntegerMatcher(int64_t value) : Matcher(CheckInteger), Value(value) {}

  int64_t getValue() const { return Value; }

  static bool classof(const Matcher *N) { return N->getKind() == CheckInteger; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckIntegerMatcher>(M)->Value == Value;
  }
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// CheckChildIntegerMatcher - This checks to see if the child node is a
/// ConstantSDNode with a specified integer value, if not it fails to match.
class CheckChildIntegerMatcher : public Matcher {
  unsigned ChildNo;
  int64_t Value;

public:
  CheckChildIntegerMatcher(unsigned childno, int64_t value)
      : Matcher(CheckChildInteger), ChildNo(childno), Value(value) {}

  unsigned getChildNo() const { return ChildNo; }
  int64_t getValue() const { return Value; }

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckChildInteger;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckChildIntegerMatcher>(M)->ChildNo == ChildNo &&
           cast<CheckChildIntegerMatcher>(M)->Value == Value;
  }
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// CheckCondCodeMatcher - This checks to see if the current node is a
/// CondCodeSDNode with the specified condition, if not it fails to match.
class CheckCondCodeMatcher : public Matcher {
  StringRef CondCodeName;

public:
  CheckCondCodeMatcher(StringRef condcodename)
      : Matcher(CheckCondCode), CondCodeName(condcodename) {}

  StringRef getCondCodeName() const { return CondCodeName; }

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckCondCode;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckCondCodeMatcher>(M)->CondCodeName == CondCodeName;
  }
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// CheckChild2CondCodeMatcher - This checks to see if child 2 node is a
/// CondCodeSDNode with the specified condition, if not it fails to match.
class CheckChild2CondCodeMatcher : public Matcher {
  StringRef CondCodeName;

public:
  CheckChild2CondCodeMatcher(StringRef condcodename)
      : Matcher(CheckChild2CondCode), CondCodeName(condcodename) {}

  StringRef getCondCodeName() const { return CondCodeName; }

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckChild2CondCode;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckChild2CondCodeMatcher>(M)->CondCodeName == CondCodeName;
  }
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// CheckValueTypeMatcher - This checks to see if the current node is a
/// VTSDNode with the specified type, if not it fails to match.
class CheckValueTypeMatcher : public Matcher {
  MVT VT;

public:
  CheckValueTypeMatcher(MVT SimpleVT) : Matcher(CheckValueType), VT(SimpleVT) {}

  MVT getVT() const { return VT; }

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckValueType;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckValueTypeMatcher>(M)->VT == VT;
  }
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// CheckComplexPatMatcher - This node runs the specified ComplexPattern on
/// the current node.
class CheckComplexPatMatcher : public Matcher {
  const ComplexPattern &Pattern;

  /// MatchNumber - This is the recorded nodes slot that contains the node we
  /// want to match against.
  unsigned MatchNumber;

  /// Name - The name of the node we're matching, for comment emission.
  StringRef Name;

  /// FirstResult - This is the first slot in the RecordedNodes list that the
  /// result of the match populates.
  unsigned FirstResult;

public:
  CheckComplexPatMatcher(const ComplexPattern &pattern, unsigned matchnumber,
                         StringRef name, unsigned firstresult)
      : Matcher(CheckComplexPat), Pattern(pattern), MatchNumber(matchnumber),
        Name(name), FirstResult(firstresult) {}

  const ComplexPattern &getPattern() const { return Pattern; }
  unsigned getMatchNumber() const { return MatchNumber; }

  StringRef getName() const { return Name; }
  unsigned getFirstResult() const { return FirstResult; }

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckComplexPat;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return &cast<CheckComplexPatMatcher>(M)->Pattern == &Pattern &&
           cast<CheckComplexPatMatcher>(M)->MatchNumber == MatchNumber;
  }
};

/// CheckAndImmMatcher - This checks to see if the current node is an 'and'
/// with something equivalent to the specified immediate.
class CheckAndImmMatcher : public Matcher {
  int64_t Value;

public:
  CheckAndImmMatcher(int64_t value) : Matcher(CheckAndImm), Value(value) {}

  int64_t getValue() const { return Value; }

  static bool classof(const Matcher *N) { return N->getKind() == CheckAndImm; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckAndImmMatcher>(M)->Value == Value;
  }
};

/// CheckOrImmMatcher - This checks to see if the current node is an 'and'
/// with something equivalent to the specified immediate.
class CheckOrImmMatcher : public Matcher {
  int64_t Value;

public:
  CheckOrImmMatcher(int64_t value) : Matcher(CheckOrImm), Value(value) {}

  int64_t getValue() const { return Value; }

  static bool classof(const Matcher *N) { return N->getKind() == CheckOrImm; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CheckOrImmMatcher>(M)->Value == Value;
  }
};

/// CheckImmAllOnesVMatcher - This checks if the current node is a build_vector
/// or splat_vector of all ones.
class CheckImmAllOnesVMatcher : public Matcher {
public:
  CheckImmAllOnesVMatcher() : Matcher(CheckImmAllOnesV) {}

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckImmAllOnesV;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return true; }
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// CheckImmAllZerosVMatcher - This checks if the current node is a
/// build_vector or splat_vector of all zeros.
class CheckImmAllZerosVMatcher : public Matcher {
public:
  CheckImmAllZerosVMatcher() : Matcher(CheckImmAllZerosV) {}

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckImmAllZerosV;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return true; }
  bool isContradictoryImpl(const Matcher *M) const override;
};

/// CheckFoldableChainNodeMatcher - This checks to see if the current node
/// (which defines a chain operand) is safe to fold into a larger pattern.
class CheckFoldableChainNodeMatcher : public Matcher {
public:
  CheckFoldableChainNodeMatcher() : Matcher(CheckFoldableChainNode) {}

  static bool classof(const Matcher *N) {
    return N->getKind() == CheckFoldableChainNode;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override { return true; }
};

/// EmitIntegerMatcher - This creates a new TargetConstant.
class EmitIntegerMatcher : public Matcher {
  // Optional string to give the value a symbolic name for readability.
  std::string Str;
  int64_t Val;
  ValueTypeByHwMode VT;

  unsigned ResultNo;

public:
  EmitIntegerMatcher(int64_t val, ValueTypeByHwMode vt, unsigned resultNo)
      : Matcher(EmitInteger), Val(val), VT(std::move(vt)), ResultNo(resultNo) {}
  EmitIntegerMatcher(const std::string &str, int64_t val, MVT vt,
                     unsigned resultNo)
      : Matcher(EmitInteger), Str(str), Val(val), VT(vt), ResultNo(resultNo) {}

  const std::string &getString() const { return Str; }
  int64_t getValue() const { return Val; }
  const ValueTypeByHwMode &getVT() const { return VT; }
  unsigned getResultNo() const { return ResultNo; }

  static bool classof(const Matcher *N) { return N->getKind() == EmitInteger; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<EmitIntegerMatcher>(M)->Val == Val &&
           cast<EmitIntegerMatcher>(M)->VT == VT &&
           cast<EmitIntegerMatcher>(M)->Str == Str;
  }
};

/// EmitRegisterMatcher - This creates a new TargetConstant.
class EmitRegisterMatcher : public Matcher {
  /// Reg - The def for the register that we're emitting.  If this is null, then
  /// this is a reference to zero_reg.
  const CodeGenRegister *Reg;
  ValueTypeByHwMode VT;

  unsigned ResultNo;

public:
  EmitRegisterMatcher(const CodeGenRegister *reg, ValueTypeByHwMode vt,
                      unsigned resultNo)
      : Matcher(EmitRegister), Reg(reg), VT(std::move(vt)), ResultNo(resultNo) {
  }

  const CodeGenRegister *getReg() const { return Reg; }
  const ValueTypeByHwMode &getVT() const { return VT; }
  unsigned getResultNo() const { return ResultNo; }

  static bool classof(const Matcher *N) { return N->getKind() == EmitRegister; }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<EmitRegisterMatcher>(M)->Reg == Reg &&
           cast<EmitRegisterMatcher>(M)->VT == VT;
  }
};

/// EmitConvertToTargetMatcher - Emit an operation that reads a specified
/// recorded node and converts it from being a ISD::Constant to
/// ISD::TargetConstant, likewise for ConstantFP.
class EmitConvertToTargetMatcher : public Matcher {
  // Recorded Node
  unsigned Slot;

  // Result
  unsigned ResultNo;

public:
  EmitConvertToTargetMatcher(unsigned slot, unsigned resultNo)
      : Matcher(EmitConvertToTarget), Slot(slot), ResultNo(resultNo) {}

  unsigned getSlot() const { return Slot; }
  unsigned getResultNo() const { return ResultNo; }

  static bool classof(const Matcher *N) {
    return N->getKind() == EmitConvertToTarget;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<EmitConvertToTargetMatcher>(M)->Slot == Slot;
  }
};

/// EmitMergeInputChainsMatcher - Emit a node that merges a list of input
/// chains together with a token factor.  The list of nodes are the nodes in the
/// matched pattern that have chain input/outputs.  This node adds all input
/// chains of these nodes if they are not themselves a node in the pattern.
class EmitMergeInputChainsMatcher : public Matcher {
  SmallVector<unsigned, 3> ChainNodes;

public:
  EmitMergeInputChainsMatcher(ArrayRef<unsigned> nodes)
      : Matcher(EmitMergeInputChains), ChainNodes(nodes) {}

  unsigned getNumNodes() const { return ChainNodes.size(); }

  unsigned getNode(unsigned i) const {
    assert(i < ChainNodes.size());
    return ChainNodes[i];
  }

  static bool classof(const Matcher *N) {
    return N->getKind() == EmitMergeInputChains;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<EmitMergeInputChainsMatcher>(M)->ChainNodes == ChainNodes;
  }
};

/// EmitCopyToRegMatcher - Emit a CopyToReg node from a value to a physreg,
/// pushing the chain and glue results.
///
class EmitCopyToRegMatcher : public Matcher {
  // Value to copy into the physreg.
  unsigned SrcSlot;
  // Register Destination
  const CodeGenRegister *DestPhysReg;

public:
  EmitCopyToRegMatcher(unsigned srcSlot, const CodeGenRegister *destPhysReg)
      : Matcher(EmitCopyToReg), SrcSlot(srcSlot), DestPhysReg(destPhysReg) {}

  unsigned getSrcSlot() const { return SrcSlot; }
  const CodeGenRegister *getDestPhysReg() const { return DestPhysReg; }

  static bool classof(const Matcher *N) {
    return N->getKind() == EmitCopyToReg;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<EmitCopyToRegMatcher>(M)->SrcSlot == SrcSlot &&
           cast<EmitCopyToRegMatcher>(M)->DestPhysReg == DestPhysReg;
  }
};

/// EmitNodeXFormMatcher - Emit an operation that runs an SDNodeXForm on a
/// recorded node and records the result.
class EmitNodeXFormMatcher : public Matcher {
  // Recorded Node
  unsigned Slot;
  // Transform
  const Record *NodeXForm;

  // Result
  unsigned ResultNo;

public:
  EmitNodeXFormMatcher(unsigned slot, const Record *nodeXForm,
                       unsigned resultNo)
      : Matcher(EmitNodeXForm), Slot(slot), NodeXForm(nodeXForm),
        ResultNo(resultNo) {}

  unsigned getSlot() const { return Slot; }
  const Record *getNodeXForm() const { return NodeXForm; }
  unsigned getResultNo() const { return ResultNo; }

  static bool classof(const Matcher *N) {
    return N->getKind() == EmitNodeXForm;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<EmitNodeXFormMatcher>(M)->Slot == Slot &&
           cast<EmitNodeXFormMatcher>(M)->NodeXForm == NodeXForm;
  }
};

/// EmitNodeMatcherCommon - Common class shared between EmitNode and
/// MorphNodeTo.
class EmitNodeMatcherCommon : public Matcher {
  const CodeGenInstruction &CGI;
  const SmallVector<ValueTypeByHwMode, 3> VTs;
  const SmallVector<unsigned, 6> Operands;
  bool HasChain, HasInGlue, HasOutGlue, HasMemRefs;

  /// NumFixedArityOperands - If this is a fixed arity node, this is set to -1.
  /// If this is a varidic node, this is set to the number of fixed arity
  /// operands in the root of the pattern.  The rest are appended to this node.
  int NumFixedArityOperands;

public:
  EmitNodeMatcherCommon(const CodeGenInstruction &cgi,
                        ArrayRef<ValueTypeByHwMode> vts,
                        ArrayRef<unsigned> operands, bool hasChain,
                        bool hasInGlue, bool hasOutGlue, bool hasmemrefs,
                        int numfixedarityoperands, bool isMorphNodeTo)
      : Matcher(isMorphNodeTo ? MorphNodeTo : EmitNode), CGI(cgi), VTs(vts),
        Operands(operands), HasChain(hasChain), HasInGlue(hasInGlue),
        HasOutGlue(hasOutGlue), HasMemRefs(hasmemrefs),
        NumFixedArityOperands(numfixedarityoperands) {}

  const CodeGenInstruction &getInstruction() const { return CGI; }

  unsigned getNumVTs() const { return VTs.size(); }
  const ValueTypeByHwMode &getVT(unsigned i) const {
    assert(i < VTs.size());
    return VTs[i];
  }

  unsigned getNumOperands() const { return Operands.size(); }
  unsigned getOperand(unsigned i) const {
    assert(i < Operands.size());
    return Operands[i];
  }

  ArrayRef<ValueTypeByHwMode> getVTList() const { return VTs; }
  ArrayRef<unsigned> getOperandList() const { return Operands; }

  bool hasChain() const { return HasChain; }
  bool hasInGlue() const { return HasInGlue; }
  bool hasOutGlue() const { return HasOutGlue; }
  bool hasMemRefs() const { return HasMemRefs; }
  int getNumFixedArityOperands() const { return NumFixedArityOperands; }

  static bool classof(const Matcher *N) {
    return N->getKind() == EmitNode || N->getKind() == MorphNodeTo;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override;
};

/// EmitNodeMatcher - This signals a successful match and generates a node.
class EmitNodeMatcher : public EmitNodeMatcherCommon {
  void anchor() override;
  unsigned FirstResultSlot;

public:
  EmitNodeMatcher(const CodeGenInstruction &cgi,
                  ArrayRef<ValueTypeByHwMode> vts, ArrayRef<unsigned> operands,
                  bool hasChain, bool hasInGlue, bool hasOutGlue,
                  bool hasmemrefs, int numfixedarityoperands,
                  unsigned firstresultslot)
      : EmitNodeMatcherCommon(cgi, vts, operands, hasChain, hasInGlue,
                              hasOutGlue, hasmemrefs, numfixedarityoperands,
                              false),
        FirstResultSlot(firstresultslot) {}

  unsigned getFirstResultSlot() const { return FirstResultSlot; }

  static bool classof(const Matcher *N) { return N->getKind() == EmitNode; }
};

class MorphNodeToMatcher : public EmitNodeMatcherCommon {
  void anchor() override;
  const PatternToMatch &Pattern;

public:
  MorphNodeToMatcher(const CodeGenInstruction &cgi,
                     ArrayRef<ValueTypeByHwMode> vts,
                     ArrayRef<unsigned> operands, bool hasChain, bool hasInGlue,
                     bool hasOutGlue, bool hasmemrefs,
                     int numfixedarityoperands, const PatternToMatch &pattern)
      : EmitNodeMatcherCommon(cgi, vts, operands, hasChain, hasInGlue,
                              hasOutGlue, hasmemrefs, numfixedarityoperands,
                              true),
        Pattern(pattern) {}

  const PatternToMatch &getPattern() const { return Pattern; }

  static bool classof(const Matcher *N) { return N->getKind() == MorphNodeTo; }
};

/// CompleteMatchMatcher - Complete a match by replacing the results of the
/// pattern with the newly generated nodes.  This also prints a comment
/// indicating the source and dest patterns.
class CompleteMatchMatcher : public Matcher {
  SmallVector<unsigned, 2> Results;
  const PatternToMatch &Pattern;

public:
  CompleteMatchMatcher(ArrayRef<unsigned> results,
                       const PatternToMatch &pattern)
      : Matcher(CompleteMatch), Results(results), Pattern(pattern) {}

  unsigned getNumResults() const { return Results.size(); }
  unsigned getResult(unsigned R) const { return Results[R]; }
  const PatternToMatch &getPattern() const { return Pattern; }

  static bool classof(const Matcher *N) {
    return N->getKind() == CompleteMatch;
  }

private:
  void printImpl(raw_ostream &OS, indent Indent) const override;
  bool isEqualImpl(const Matcher *M) const override {
    return cast<CompleteMatchMatcher>(M)->Results == Results &&
           &cast<CompleteMatchMatcher>(M)->Pattern == &Pattern;
  }
};

} // end namespace llvm

#endif // LLVM_UTILS_TABLEGEN_COMMON_DAGISELMATCHER_H
