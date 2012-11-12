//===- DAGISelMatcher.h - Representation of DAG pattern matcher -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef TBLGEN_DAGISELMATCHER_H
#define TBLGEN_DAGISELMATCHER_H

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace llvm {
  struct CodeGenRegister;
  class CodeGenDAGPatterns;
  class Matcher;
  class PatternToMatch;
  class raw_ostream;
  class ComplexPattern;
  class Record;
  class SDNodeInfo;
  class TreePredicateFn;
  class TreePattern;

Matcher *ConvertPatternToMatcher(const PatternToMatch &Pattern,unsigned Variant,
                                 const CodeGenDAGPatterns &CGP);
Matcher *OptimizeMatcher(Matcher *Matcher, const CodeGenDAGPatterns &CGP);
void EmitMatcherTable(const Matcher *Matcher, const CodeGenDAGPatterns &CGP,
                      raw_ostream &OS);


/// Matcher - Base class for all the DAG ISel Matcher representation
/// nodes.
class Matcher {
  // The next matcher node that is executed after this one.  Null if this is the
  // last stage of a match.
  OwningPtr<Matcher> Next;
  virtual void anchor();
public:
  enum KindTy {
    // Matcher state manipulation.
    Scope,                // Push a checking scope.
    RecordNode,           // Record the current node.
    RecordChild,          // Record a child of the current node.
    RecordMemRef,         // Record the memref in the current node.
    CaptureGlueInput,     // If the current node has an input glue, save it.
    MoveChild,            // Move current node to specified child.
    MoveParent,           // Move current node to parent.

    // Predicate checking.
    CheckSame,            // Fail if not same as prev match.
    CheckPatternPredicate,
    CheckPredicate,       // Fail if node predicate fails.
    CheckOpcode,          // Fail if not opcode.
    SwitchOpcode,         // Dispatch based on opcode.
    CheckType,            // Fail if not correct type.
    SwitchType,           // Dispatch based on type.
    CheckChildType,       // Fail if child has wrong type.
    CheckInteger,         // Fail if wrong val.
    CheckCondCode,        // Fail if not condcode.
    CheckValueType,
    CheckComplexPat,
    CheckAndImm,
    CheckOrImm,
    CheckFoldableChainNode,

    // Node creation/emisssion.
    EmitInteger,          // Create a TargetConstant
    EmitStringInteger,    // Create a TargetConstant from a string.
    EmitRegister,         // Create a register.
    EmitConvertToTarget,  // Convert a imm/fpimm to target imm/fpimm
    EmitMergeInputChains, // Merge together a chains for an input.
    EmitCopyToReg,        // Emit a copytoreg into a physreg.
    EmitNode,             // Create a DAG node
    EmitNodeXForm,        // Run a SDNodeXForm
    MarkGlueResults,      // Indicate which interior nodes have glue results.
    CompleteMatch,        // Finish a match and update the results.
    MorphNodeTo           // Build a node, finish a match and update results.
  };
  const KindTy Kind;

protected:
  Matcher(KindTy K) : Kind(K) {}
public:
  virtual ~Matcher() {}

  KindTy getKind() const { return Kind; }

  Matcher *getNext() { return Next.get(); }
  const Matcher *getNext() const { return Next.get(); }
  void setNext(Matcher *C) { Next.reset(C); }
  Matcher *takeNext() { return Next.take(); }

  OwningPtr<Matcher> &getNextPtr() { return Next; }

  bool isEqual(const Matcher *M) const {
    if (getKind() != M->getKind()) return false;
    return isEqualImpl(M);
  }

  unsigned getHash() const {
    // Clear the high bit so we don't conflict with tombstones etc.
    return ((getHashImpl() << 4) ^ getKind()) & (~0U>>1);
  }

  /// isSafeToReorderWithPatternPredicate - Return true if it is safe to sink a
  /// PatternPredicate node past this one.
  virtual bool isSafeToReorderWithPatternPredicate() const {
    return false;
  }

  /// isSimplePredicateNode - Return true if this is a simple predicate that
  /// operates on the node or its children without potential side effects or a
  /// change of the current node.
  bool isSimplePredicateNode() const {
    switch (getKind()) {
    default: return false;
    case CheckSame:
    case CheckPatternPredicate:
    case CheckPredicate:
    case CheckOpcode:
    case CheckType:
    case CheckChildType:
    case CheckInteger:
    case CheckCondCode:
    case CheckValueType:
    case CheckAndImm:
    case CheckOrImm:
    case CheckFoldableChainNode:
      return true;
    }
  }

  /// isSimplePredicateOrRecordNode - Return true if this is a record node or
  /// a simple predicate.
  bool isSimplePredicateOrRecordNode() const {
    return isSimplePredicateNode() ||
           getKind() == RecordNode || getKind() == RecordChild;
  }

  /// unlinkNode - Unlink the specified node from this chain.  If Other == this,
  /// we unlink the next pointer and return it.  Otherwise we unlink Other from
  /// the list and return this.
  Matcher *unlinkNode(Matcher *Other);

  /// canMoveBefore - Return true if this matcher is the same as Other, or if
  /// we can move this matcher past all of the nodes in-between Other and this
  /// node.  Other must be equal to or before this.
  bool canMoveBefore(const Matcher *Other) const;

  /// canMoveBefore - Return true if it is safe to move the current matcher
  /// across the specified one.
  bool canMoveBeforeNode(const Matcher *Other) const;

  /// isContradictory - Return true of these two matchers could never match on
  /// the same node.
  bool isContradictory(const Matcher *Other) const {
    // Since this predicate is reflexive, we canonicalize the ordering so that
    // we always match a node against nodes with kinds that are greater or equal
    // to them.  For example, we'll pass in a CheckType node as an argument to
    // the CheckOpcode method, not the other way around.
    if (getKind() < Other->getKind())
      return isContradictoryImpl(Other);
    return Other->isContradictoryImpl(this);
  }

  void print(raw_ostream &OS, unsigned indent = 0) const;
  void printOne(raw_ostream &OS) const;
  void dump() const;
protected:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const = 0;
  virtual bool isEqualImpl(const Matcher *M) const = 0;
  virtual unsigned getHashImpl() const = 0;
  virtual bool isContradictoryImpl(const Matcher *M) const { return false; }
};

/// ScopeMatcher - This attempts to match each of its children to find the first
/// one that successfully matches.  If one child fails, it tries the next child.
/// If none of the children match then this check fails.  It never has a 'next'.
class ScopeMatcher : public Matcher {
  SmallVector<Matcher*, 4> Children;
public:
  ScopeMatcher(Matcher *const *children, unsigned numchildren)
    : Matcher(Scope), Children(children, children+numchildren) {
  }
  virtual ~ScopeMatcher();

  unsigned getNumChildren() const { return Children.size(); }

  Matcher *getChild(unsigned i) { return Children[i]; }
  const Matcher *getChild(unsigned i) const { return Children[i]; }

  void resetChild(unsigned i, Matcher *N) {
    delete Children[i];
    Children[i] = N;
  }

  Matcher *takeChild(unsigned i) {
    Matcher *Res = Children[i];
    Children[i] = 0;
    return Res;
  }

  void setNumChildren(unsigned NC) {
    if (NC < Children.size()) {
      // delete any children we're about to lose pointers to.
      for (unsigned i = NC, e = Children.size(); i != e; ++i)
        delete Children[i];
    }
    Children.resize(NC);
  }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == Scope;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const { return false; }
  virtual unsigned getHashImpl() const { return 12312; }
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

  static inline bool classof(const Matcher *N) {
    return N->getKind() == RecordNode;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }
private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const { return true; }
  virtual unsigned getHashImpl() const { return 0; }
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

  static inline bool classof(const Matcher *N) {
    return N->getKind() == RecordChild;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<RecordChildMatcher>(M)->getChildNo() == getChildNo();
  }
  virtual unsigned getHashImpl() const { return getChildNo(); }
};

/// RecordMemRefMatcher - Save the current node's memref.
class RecordMemRefMatcher : public Matcher {
public:
  RecordMemRefMatcher() : Matcher(RecordMemRef) {}

  static inline bool classof(const Matcher *N) {
    return N->getKind() == RecordMemRef;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const { return true; }
  virtual unsigned getHashImpl() const { return 0; }
};


/// CaptureGlueInputMatcher - If the current record has a glue input, record
/// it so that it is used as an input to the generated code.
class CaptureGlueInputMatcher : public Matcher {
public:
  CaptureGlueInputMatcher() : Matcher(CaptureGlueInput) {}

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CaptureGlueInput;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const { return true; }
  virtual unsigned getHashImpl() const { return 0; }
};

/// MoveChildMatcher - This tells the interpreter to move into the
/// specified child node.
class MoveChildMatcher : public Matcher {
  unsigned ChildNo;
public:
  MoveChildMatcher(unsigned childNo) : Matcher(MoveChild), ChildNo(childNo) {}

  unsigned getChildNo() const { return ChildNo; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == MoveChild;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<MoveChildMatcher>(M)->getChildNo() == getChildNo();
  }
  virtual unsigned getHashImpl() const { return getChildNo(); }
};

/// MoveParentMatcher - This tells the interpreter to move to the parent
/// of the current node.
class MoveParentMatcher : public Matcher {
public:
  MoveParentMatcher() : Matcher(MoveParent) {}

  static inline bool classof(const Matcher *N) {
    return N->getKind() == MoveParent;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const { return true; }
  virtual unsigned getHashImpl() const { return 0; }
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

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckSame;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckSameMatcher>(M)->getMatchNumber() == getMatchNumber();
  }
  virtual unsigned getHashImpl() const { return getMatchNumber(); }
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

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckPatternPredicate;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckPatternPredicateMatcher>(M)->getPredicate() == Predicate;
  }
  virtual unsigned getHashImpl() const;
};

/// CheckPredicateMatcher - This checks the target-specific predicate to
/// see if the node is acceptable.
class CheckPredicateMatcher : public Matcher {
  TreePattern *Pred;
public:
  CheckPredicateMatcher(const TreePredicateFn &pred);

  TreePredicateFn getPredicate() const;

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckPredicate;
  }

  // TODO: Ok?
  //virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckPredicateMatcher>(M)->Pred == Pred;
  }
  virtual unsigned getHashImpl() const;
};


/// CheckOpcodeMatcher - This checks to see if the current node has the
/// specified opcode, if not it fails to match.
class CheckOpcodeMatcher : public Matcher {
  const SDNodeInfo &Opcode;
public:
  CheckOpcodeMatcher(const SDNodeInfo &opcode)
    : Matcher(CheckOpcode), Opcode(opcode) {}

  const SDNodeInfo &getOpcode() const { return Opcode; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckOpcode;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const;
  virtual unsigned getHashImpl() const;
  virtual bool isContradictoryImpl(const Matcher *M) const;
};

/// SwitchOpcodeMatcher - Switch based on the current node's opcode, dispatching
/// to one matcher per opcode.  If the opcode doesn't match any of the cases,
/// then the match fails.  This is semantically equivalent to a Scope node where
/// every child does a CheckOpcode, but is much faster.
class SwitchOpcodeMatcher : public Matcher {
  SmallVector<std::pair<const SDNodeInfo*, Matcher*>, 8> Cases;
public:
  SwitchOpcodeMatcher(const std::pair<const SDNodeInfo*, Matcher*> *cases,
                      unsigned numcases)
    : Matcher(SwitchOpcode), Cases(cases, cases+numcases) {}

  static inline bool classof(const Matcher *N) {
    return N->getKind() == SwitchOpcode;
  }

  unsigned getNumCases() const { return Cases.size(); }

  const SDNodeInfo &getCaseOpcode(unsigned i) const { return *Cases[i].first; }
  Matcher *getCaseMatcher(unsigned i) { return Cases[i].second; }
  const Matcher *getCaseMatcher(unsigned i) const { return Cases[i].second; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const { return false; }
  virtual unsigned getHashImpl() const { return 4123; }
};

/// CheckTypeMatcher - This checks to see if the current node has the
/// specified type at the specified result, if not it fails to match.
class CheckTypeMatcher : public Matcher {
  MVT::SimpleValueType Type;
  unsigned ResNo;
public:
  CheckTypeMatcher(MVT::SimpleValueType type, unsigned resno)
    : Matcher(CheckType), Type(type), ResNo(resno) {}

  MVT::SimpleValueType getType() const { return Type; }
  unsigned getResNo() const { return ResNo; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckType;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckTypeMatcher>(M)->Type == Type;
  }
  virtual unsigned getHashImpl() const { return Type; }
  virtual bool isContradictoryImpl(const Matcher *M) const;
};

/// SwitchTypeMatcher - Switch based on the current node's type, dispatching
/// to one matcher per case.  If the type doesn't match any of the cases,
/// then the match fails.  This is semantically equivalent to a Scope node where
/// every child does a CheckType, but is much faster.
class SwitchTypeMatcher : public Matcher {
  SmallVector<std::pair<MVT::SimpleValueType, Matcher*>, 8> Cases;
public:
  SwitchTypeMatcher(const std::pair<MVT::SimpleValueType, Matcher*> *cases,
                    unsigned numcases)
  : Matcher(SwitchType), Cases(cases, cases+numcases) {}

  static inline bool classof(const Matcher *N) {
    return N->getKind() == SwitchType;
  }

  unsigned getNumCases() const { return Cases.size(); }

  MVT::SimpleValueType getCaseType(unsigned i) const { return Cases[i].first; }
  Matcher *getCaseMatcher(unsigned i) { return Cases[i].second; }
  const Matcher *getCaseMatcher(unsigned i) const { return Cases[i].second; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const { return false; }
  virtual unsigned getHashImpl() const { return 4123; }
};


/// CheckChildTypeMatcher - This checks to see if a child node has the
/// specified type, if not it fails to match.
class CheckChildTypeMatcher : public Matcher {
  unsigned ChildNo;
  MVT::SimpleValueType Type;
public:
  CheckChildTypeMatcher(unsigned childno, MVT::SimpleValueType type)
    : Matcher(CheckChildType), ChildNo(childno), Type(type) {}

  unsigned getChildNo() const { return ChildNo; }
  MVT::SimpleValueType getType() const { return Type; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckChildType;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckChildTypeMatcher>(M)->ChildNo == ChildNo &&
           cast<CheckChildTypeMatcher>(M)->Type == Type;
  }
  virtual unsigned getHashImpl() const { return (Type << 3) | ChildNo; }
  virtual bool isContradictoryImpl(const Matcher *M) const;
};


/// CheckIntegerMatcher - This checks to see if the current node is a
/// ConstantSDNode with the specified integer value, if not it fails to match.
class CheckIntegerMatcher : public Matcher {
  int64_t Value;
public:
  CheckIntegerMatcher(int64_t value)
    : Matcher(CheckInteger), Value(value) {}

  int64_t getValue() const { return Value; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckInteger;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckIntegerMatcher>(M)->Value == Value;
  }
  virtual unsigned getHashImpl() const { return Value; }
  virtual bool isContradictoryImpl(const Matcher *M) const;
};

/// CheckCondCodeMatcher - This checks to see if the current node is a
/// CondCodeSDNode with the specified condition, if not it fails to match.
class CheckCondCodeMatcher : public Matcher {
  StringRef CondCodeName;
public:
  CheckCondCodeMatcher(StringRef condcodename)
    : Matcher(CheckCondCode), CondCodeName(condcodename) {}

  StringRef getCondCodeName() const { return CondCodeName; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckCondCode;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckCondCodeMatcher>(M)->CondCodeName == CondCodeName;
  }
  virtual unsigned getHashImpl() const;
};

/// CheckValueTypeMatcher - This checks to see if the current node is a
/// VTSDNode with the specified type, if not it fails to match.
class CheckValueTypeMatcher : public Matcher {
  StringRef TypeName;
public:
  CheckValueTypeMatcher(StringRef type_name)
    : Matcher(CheckValueType), TypeName(type_name) {}

  StringRef getTypeName() const { return TypeName; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckValueType;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckValueTypeMatcher>(M)->TypeName == TypeName;
  }
  virtual unsigned getHashImpl() const;
  bool isContradictoryImpl(const Matcher *M) const;
};



/// CheckComplexPatMatcher - This node runs the specified ComplexPattern on
/// the current node.
class CheckComplexPatMatcher : public Matcher {
  const ComplexPattern &Pattern;

  /// MatchNumber - This is the recorded nodes slot that contains the node we
  /// want to match against.
  unsigned MatchNumber;

  /// Name - The name of the node we're matching, for comment emission.
  std::string Name;

  /// FirstResult - This is the first slot in the RecordedNodes list that the
  /// result of the match populates.
  unsigned FirstResult;
public:
  CheckComplexPatMatcher(const ComplexPattern &pattern, unsigned matchnumber,
                         const std::string &name, unsigned firstresult)
    : Matcher(CheckComplexPat), Pattern(pattern), MatchNumber(matchnumber),
      Name(name), FirstResult(firstresult) {}

  const ComplexPattern &getPattern() const { return Pattern; }
  unsigned getMatchNumber() const { return MatchNumber; }

  const std::string getName() const { return Name; }
  unsigned getFirstResult() const { return FirstResult; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckComplexPat;
  }

  // Not safe to move a pattern predicate past a complex pattern.
  virtual bool isSafeToReorderWithPatternPredicate() const { return false; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return &cast<CheckComplexPatMatcher>(M)->Pattern == &Pattern &&
           cast<CheckComplexPatMatcher>(M)->MatchNumber == MatchNumber;
  }
  virtual unsigned getHashImpl() const {
    return (unsigned)(intptr_t)&Pattern ^ MatchNumber;
  }
};

/// CheckAndImmMatcher - This checks to see if the current node is an 'and'
/// with something equivalent to the specified immediate.
class CheckAndImmMatcher : public Matcher {
  int64_t Value;
public:
  CheckAndImmMatcher(int64_t value)
    : Matcher(CheckAndImm), Value(value) {}

  int64_t getValue() const { return Value; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckAndImm;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckAndImmMatcher>(M)->Value == Value;
  }
  virtual unsigned getHashImpl() const { return Value; }
};

/// CheckOrImmMatcher - This checks to see if the current node is an 'and'
/// with something equivalent to the specified immediate.
class CheckOrImmMatcher : public Matcher {
  int64_t Value;
public:
  CheckOrImmMatcher(int64_t value)
    : Matcher(CheckOrImm), Value(value) {}

  int64_t getValue() const { return Value; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckOrImm;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CheckOrImmMatcher>(M)->Value == Value;
  }
  virtual unsigned getHashImpl() const { return Value; }
};

/// CheckFoldableChainNodeMatcher - This checks to see if the current node
/// (which defines a chain operand) is safe to fold into a larger pattern.
class CheckFoldableChainNodeMatcher : public Matcher {
public:
  CheckFoldableChainNodeMatcher()
    : Matcher(CheckFoldableChainNode) {}

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CheckFoldableChainNode;
  }

  virtual bool isSafeToReorderWithPatternPredicate() const { return true; }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const { return true; }
  virtual unsigned getHashImpl() const { return 0; }
};

/// EmitIntegerMatcher - This creates a new TargetConstant.
class EmitIntegerMatcher : public Matcher {
  int64_t Val;
  MVT::SimpleValueType VT;
public:
  EmitIntegerMatcher(int64_t val, MVT::SimpleValueType vt)
    : Matcher(EmitInteger), Val(val), VT(vt) {}

  int64_t getValue() const { return Val; }
  MVT::SimpleValueType getVT() const { return VT; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == EmitInteger;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<EmitIntegerMatcher>(M)->Val == Val &&
           cast<EmitIntegerMatcher>(M)->VT == VT;
  }
  virtual unsigned getHashImpl() const { return (Val << 4) | VT; }
};

/// EmitStringIntegerMatcher - A target constant whose value is represented
/// by a string.
class EmitStringIntegerMatcher : public Matcher {
  std::string Val;
  MVT::SimpleValueType VT;
public:
  EmitStringIntegerMatcher(const std::string &val, MVT::SimpleValueType vt)
    : Matcher(EmitStringInteger), Val(val), VT(vt) {}

  const std::string &getValue() const { return Val; }
  MVT::SimpleValueType getVT() const { return VT; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == EmitStringInteger;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<EmitStringIntegerMatcher>(M)->Val == Val &&
           cast<EmitStringIntegerMatcher>(M)->VT == VT;
  }
  virtual unsigned getHashImpl() const;
};

/// EmitRegisterMatcher - This creates a new TargetConstant.
class EmitRegisterMatcher : public Matcher {
  /// Reg - The def for the register that we're emitting.  If this is null, then
  /// this is a reference to zero_reg.
  const CodeGenRegister *Reg;
  MVT::SimpleValueType VT;
public:
  EmitRegisterMatcher(const CodeGenRegister *reg, MVT::SimpleValueType vt)
    : Matcher(EmitRegister), Reg(reg), VT(vt) {}

  const CodeGenRegister *getReg() const { return Reg; }
  MVT::SimpleValueType getVT() const { return VT; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == EmitRegister;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<EmitRegisterMatcher>(M)->Reg == Reg &&
           cast<EmitRegisterMatcher>(M)->VT == VT;
  }
  virtual unsigned getHashImpl() const {
    return ((unsigned)(intptr_t)Reg) << 4 | VT;
  }
};

/// EmitConvertToTargetMatcher - Emit an operation that reads a specified
/// recorded node and converts it from being a ISD::Constant to
/// ISD::TargetConstant, likewise for ConstantFP.
class EmitConvertToTargetMatcher : public Matcher {
  unsigned Slot;
public:
  EmitConvertToTargetMatcher(unsigned slot)
    : Matcher(EmitConvertToTarget), Slot(slot) {}

  unsigned getSlot() const { return Slot; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == EmitConvertToTarget;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<EmitConvertToTargetMatcher>(M)->Slot == Slot;
  }
  virtual unsigned getHashImpl() const { return Slot; }
};

/// EmitMergeInputChainsMatcher - Emit a node that merges a list of input
/// chains together with a token factor.  The list of nodes are the nodes in the
/// matched pattern that have chain input/outputs.  This node adds all input
/// chains of these nodes if they are not themselves a node in the pattern.
class EmitMergeInputChainsMatcher : public Matcher {
  SmallVector<unsigned, 3> ChainNodes;
public:
  EmitMergeInputChainsMatcher(const unsigned *nodes, unsigned NumNodes)
    : Matcher(EmitMergeInputChains), ChainNodes(nodes, nodes+NumNodes) {}

  unsigned getNumNodes() const { return ChainNodes.size(); }

  unsigned getNode(unsigned i) const {
    assert(i < ChainNodes.size());
    return ChainNodes[i];
  }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == EmitMergeInputChains;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<EmitMergeInputChainsMatcher>(M)->ChainNodes == ChainNodes;
  }
  virtual unsigned getHashImpl() const;
};

/// EmitCopyToRegMatcher - Emit a CopyToReg node from a value to a physreg,
/// pushing the chain and glue results.
///
class EmitCopyToRegMatcher : public Matcher {
  unsigned SrcSlot; // Value to copy into the physreg.
  Record *DestPhysReg;
public:
  EmitCopyToRegMatcher(unsigned srcSlot, Record *destPhysReg)
    : Matcher(EmitCopyToReg), SrcSlot(srcSlot), DestPhysReg(destPhysReg) {}

  unsigned getSrcSlot() const { return SrcSlot; }
  Record *getDestPhysReg() const { return DestPhysReg; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == EmitCopyToReg;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<EmitCopyToRegMatcher>(M)->SrcSlot == SrcSlot &&
           cast<EmitCopyToRegMatcher>(M)->DestPhysReg == DestPhysReg;
  }
  virtual unsigned getHashImpl() const {
    return SrcSlot ^ ((unsigned)(intptr_t)DestPhysReg << 4);
  }
};



/// EmitNodeXFormMatcher - Emit an operation that runs an SDNodeXForm on a
/// recorded node and records the result.
class EmitNodeXFormMatcher : public Matcher {
  unsigned Slot;
  Record *NodeXForm;
public:
  EmitNodeXFormMatcher(unsigned slot, Record *nodeXForm)
    : Matcher(EmitNodeXForm), Slot(slot), NodeXForm(nodeXForm) {}

  unsigned getSlot() const { return Slot; }
  Record *getNodeXForm() const { return NodeXForm; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == EmitNodeXForm;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<EmitNodeXFormMatcher>(M)->Slot == Slot &&
           cast<EmitNodeXFormMatcher>(M)->NodeXForm == NodeXForm;
  }
  virtual unsigned getHashImpl() const {
    return Slot ^ ((unsigned)(intptr_t)NodeXForm << 4);
  }
};

/// EmitNodeMatcherCommon - Common class shared between EmitNode and
/// MorphNodeTo.
class EmitNodeMatcherCommon : public Matcher {
  std::string OpcodeName;
  const SmallVector<MVT::SimpleValueType, 3> VTs;
  const SmallVector<unsigned, 6> Operands;
  bool HasChain, HasInGlue, HasOutGlue, HasMemRefs;

  /// NumFixedArityOperands - If this is a fixed arity node, this is set to -1.
  /// If this is a varidic node, this is set to the number of fixed arity
  /// operands in the root of the pattern.  The rest are appended to this node.
  int NumFixedArityOperands;
public:
  EmitNodeMatcherCommon(const std::string &opcodeName,
                        const MVT::SimpleValueType *vts, unsigned numvts,
                        const unsigned *operands, unsigned numops,
                        bool hasChain, bool hasInGlue, bool hasOutGlue,
                        bool hasmemrefs,
                        int numfixedarityoperands, bool isMorphNodeTo)
    : Matcher(isMorphNodeTo ? MorphNodeTo : EmitNode), OpcodeName(opcodeName),
      VTs(vts, vts+numvts), Operands(operands, operands+numops),
      HasChain(hasChain), HasInGlue(hasInGlue), HasOutGlue(hasOutGlue),
      HasMemRefs(hasmemrefs), NumFixedArityOperands(numfixedarityoperands) {}

  const std::string &getOpcodeName() const { return OpcodeName; }

  unsigned getNumVTs() const { return VTs.size(); }
  MVT::SimpleValueType getVT(unsigned i) const {
    assert(i < VTs.size());
    return VTs[i];
  }

  unsigned getNumOperands() const { return Operands.size(); }
  unsigned getOperand(unsigned i) const {
    assert(i < Operands.size());
    return Operands[i];
  }

  const SmallVectorImpl<MVT::SimpleValueType> &getVTList() const { return VTs; }
  const SmallVectorImpl<unsigned> &getOperandList() const { return Operands; }


  bool hasChain() const { return HasChain; }
  bool hasInFlag() const { return HasInGlue; }
  bool hasOutFlag() const { return HasOutGlue; }
  bool hasMemRefs() const { return HasMemRefs; }
  int getNumFixedArityOperands() const { return NumFixedArityOperands; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == EmitNode || N->getKind() == MorphNodeTo;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const;
  virtual unsigned getHashImpl() const;
};

/// EmitNodeMatcher - This signals a successful match and generates a node.
class EmitNodeMatcher : public EmitNodeMatcherCommon {
  virtual void anchor();
  unsigned FirstResultSlot;
public:
  EmitNodeMatcher(const std::string &opcodeName,
                  const MVT::SimpleValueType *vts, unsigned numvts,
                  const unsigned *operands, unsigned numops,
                  bool hasChain, bool hasInFlag, bool hasOutFlag,
                  bool hasmemrefs,
                  int numfixedarityoperands, unsigned firstresultslot)
  : EmitNodeMatcherCommon(opcodeName, vts, numvts, operands, numops, hasChain,
                          hasInFlag, hasOutFlag, hasmemrefs,
                          numfixedarityoperands, false),
    FirstResultSlot(firstresultslot) {}

  unsigned getFirstResultSlot() const { return FirstResultSlot; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == EmitNode;
  }

};

class MorphNodeToMatcher : public EmitNodeMatcherCommon {
  virtual void anchor();
  const PatternToMatch &Pattern;
public:
  MorphNodeToMatcher(const std::string &opcodeName,
                     const MVT::SimpleValueType *vts, unsigned numvts,
                     const unsigned *operands, unsigned numops,
                     bool hasChain, bool hasInFlag, bool hasOutFlag,
                     bool hasmemrefs,
                     int numfixedarityoperands, const PatternToMatch &pattern)
    : EmitNodeMatcherCommon(opcodeName, vts, numvts, operands, numops, hasChain,
                            hasInFlag, hasOutFlag, hasmemrefs,
                            numfixedarityoperands, true),
      Pattern(pattern) {
  }

  const PatternToMatch &getPattern() const { return Pattern; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == MorphNodeTo;
  }
};

/// MarkGlueResultsMatcher - This node indicates which non-root nodes in the
/// pattern produce glue.  This allows CompleteMatchMatcher to update them
/// with the output glue of the resultant code.
class MarkGlueResultsMatcher : public Matcher {
  SmallVector<unsigned, 3> GlueResultNodes;
public:
  MarkGlueResultsMatcher(const unsigned *nodes, unsigned NumNodes)
    : Matcher(MarkGlueResults), GlueResultNodes(nodes, nodes+NumNodes) {}

  unsigned getNumNodes() const { return GlueResultNodes.size(); }

  unsigned getNode(unsigned i) const {
    assert(i < GlueResultNodes.size());
    return GlueResultNodes[i];
  }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == MarkGlueResults;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<MarkGlueResultsMatcher>(M)->GlueResultNodes == GlueResultNodes;
  }
  virtual unsigned getHashImpl() const;
};

/// CompleteMatchMatcher - Complete a match by replacing the results of the
/// pattern with the newly generated nodes.  This also prints a comment
/// indicating the source and dest patterns.
class CompleteMatchMatcher : public Matcher {
  SmallVector<unsigned, 2> Results;
  const PatternToMatch &Pattern;
public:
  CompleteMatchMatcher(const unsigned *results, unsigned numresults,
                       const PatternToMatch &pattern)
  : Matcher(CompleteMatch), Results(results, results+numresults),
    Pattern(pattern) {}

  unsigned getNumResults() const { return Results.size(); }
  unsigned getResult(unsigned R) const { return Results[R]; }
  const PatternToMatch &getPattern() const { return Pattern; }

  static inline bool classof(const Matcher *N) {
    return N->getKind() == CompleteMatch;
  }

private:
  virtual void printImpl(raw_ostream &OS, unsigned indent) const;
  virtual bool isEqualImpl(const Matcher *M) const {
    return cast<CompleteMatchMatcher>(M)->Results == Results &&
          &cast<CompleteMatchMatcher>(M)->Pattern == &Pattern;
  }
  virtual unsigned getHashImpl() const;
};

} // end namespace llvm

#endif
