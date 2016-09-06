//===- CodeGenDAGPatterns.h - Read DAG patterns from .td file ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CodeGenDAGPatterns class, which is used to read and
// represent the patterns present in a .td file for instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_CODEGENDAGPATTERNS_H
#define LLVM_UTILS_TABLEGEN_CODEGENDAGPATTERNS_H

#include "CodeGenIntrinsics.h"
#include "CodeGenTarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace llvm {
  class Record;
  class Init;
  class ListInit;
  class DagInit;
  class SDNodeInfo;
  class TreePattern;
  class TreePatternNode;
  class CodeGenDAGPatterns;
  class ComplexPattern;

/// EEVT::DAGISelGenValueType - These are some extended forms of
/// MVT::SimpleValueType that we use as lattice values during type inference.
/// The existing MVT iAny, fAny and vAny types suffice to represent
/// arbitrary integer, floating-point, and vector types, so only an unknown
/// value is needed.
namespace EEVT {
  /// TypeSet - This is either empty if it's completely unknown, or holds a set
  /// of types.  It is used during type inference because register classes can
  /// have multiple possible types and we don't know which one they get until
  /// type inference is complete.
  ///
  /// TypeSet can have three states:
  ///    Vector is empty: The type is completely unknown, it can be any valid
  ///       target type.
  ///    Vector has multiple constrained types: (e.g. v4i32 + v4f32) it is one
  ///       of those types only.
  ///    Vector has one concrete type: The type is completely known.
  ///
  class TypeSet {
    SmallVector<MVT::SimpleValueType, 4> TypeVec;
  public:
    TypeSet() {}
    TypeSet(MVT::SimpleValueType VT, TreePattern &TP);
    TypeSet(ArrayRef<MVT::SimpleValueType> VTList);

    bool isCompletelyUnknown() const { return TypeVec.empty(); }

    bool isConcrete() const {
      if (TypeVec.size() != 1) return false;
      unsigned char T = TypeVec[0]; (void)T;
      assert(T < MVT::LAST_VALUETYPE || T == MVT::iPTR || T == MVT::iPTRAny);
      return true;
    }

    MVT::SimpleValueType getConcrete() const {
      assert(isConcrete() && "Type isn't concrete yet");
      return (MVT::SimpleValueType)TypeVec[0];
    }

    bool isDynamicallyResolved() const {
      return getConcrete() == MVT::iPTR || getConcrete() == MVT::iPTRAny;
    }

    const SmallVectorImpl<MVT::SimpleValueType> &getTypeList() const {
      assert(!TypeVec.empty() && "Not a type list!");
      return TypeVec;
    }

    bool isVoid() const {
      return TypeVec.size() == 1 && TypeVec[0] == MVT::isVoid;
    }

    /// hasIntegerTypes - Return true if this TypeSet contains any integer value
    /// types.
    bool hasIntegerTypes() const;

    /// hasFloatingPointTypes - Return true if this TypeSet contains an fAny or
    /// a floating point value type.
    bool hasFloatingPointTypes() const;

    /// hasScalarTypes - Return true if this TypeSet contains a scalar value
    /// type.
    bool hasScalarTypes() const;

    /// hasVectorTypes - Return true if this TypeSet contains a vector value
    /// type.
    bool hasVectorTypes() const;

    /// getName() - Return this TypeSet as a string.
    std::string getName() const;

    /// MergeInTypeInfo - This merges in type information from the specified
    /// argument.  If 'this' changes, it returns true.  If the two types are
    /// contradictory (e.g. merge f32 into i32) then this flags an error.
    bool MergeInTypeInfo(const EEVT::TypeSet &InVT, TreePattern &TP);

    bool MergeInTypeInfo(MVT::SimpleValueType InVT, TreePattern &TP) {
      return MergeInTypeInfo(EEVT::TypeSet(InVT, TP), TP);
    }

    /// Force this type list to only contain integer types.
    bool EnforceInteger(TreePattern &TP);

    /// Force this type list to only contain floating point types.
    bool EnforceFloatingPoint(TreePattern &TP);

    /// EnforceScalar - Remove all vector types from this type list.
    bool EnforceScalar(TreePattern &TP);

    /// EnforceVector - Remove all non-vector types from this type list.
    bool EnforceVector(TreePattern &TP);

    /// EnforceSmallerThan - 'this' must be a smaller VT than Other.  Update
    /// this an other based on this information.
    bool EnforceSmallerThan(EEVT::TypeSet &Other, TreePattern &TP);

    /// EnforceVectorEltTypeIs - 'this' is now constrained to be a vector type
    /// whose element is VT.
    bool EnforceVectorEltTypeIs(EEVT::TypeSet &VT, TreePattern &TP);

    /// EnforceVectorEltTypeIs - 'this' is now constrained to be a vector type
    /// whose element is VT.
    bool EnforceVectorEltTypeIs(MVT::SimpleValueType VT, TreePattern &TP);

    /// EnforceVectorSubVectorTypeIs - 'this' is now constrained to
    /// be a vector type VT.
    bool EnforceVectorSubVectorTypeIs(EEVT::TypeSet &VT, TreePattern &TP);

    /// EnforceVectorSameNumElts - 'this' is now constrained to
    /// be a vector with same num elements as VT.
    bool EnforceVectorSameNumElts(EEVT::TypeSet &VT, TreePattern &TP);

    /// EnforceSameSize - 'this' is now constrained to be the same size as VT.
    bool EnforceSameSize(EEVT::TypeSet &VT, TreePattern &TP);

    bool operator!=(const TypeSet &RHS) const { return TypeVec != RHS.TypeVec; }
    bool operator==(const TypeSet &RHS) const { return TypeVec == RHS.TypeVec; }

  private:
    /// FillWithPossibleTypes - Set to all legal types and return true, only
    /// valid on completely unknown type sets.  If Pred is non-null, only MVTs
    /// that pass the predicate are added.
    bool FillWithPossibleTypes(TreePattern &TP,
                               bool (*Pred)(MVT::SimpleValueType) = nullptr,
                               const char *PredicateName = nullptr);
  };
}

/// Set type used to track multiply used variables in patterns
typedef std::set<std::string> MultipleUseVarSet;

/// SDTypeConstraint - This is a discriminated union of constraints,
/// corresponding to the SDTypeConstraint tablegen class in Target.td.
struct SDTypeConstraint {
  SDTypeConstraint(Record *R);

  unsigned OperandNo;   // The operand # this constraint applies to.
  enum {
    SDTCisVT, SDTCisPtrTy, SDTCisInt, SDTCisFP, SDTCisVec, SDTCisSameAs,
    SDTCisVTSmallerThanOp, SDTCisOpSmallerThanOp, SDTCisEltOfVec,
    SDTCisSubVecOfVec, SDTCVecEltisVT, SDTCisSameNumEltsAs, SDTCisSameSizeAs
  } ConstraintType;

  union {   // The discriminated union.
    struct {
      MVT::SimpleValueType VT;
    } SDTCisVT_Info;
    struct {
      unsigned OtherOperandNum;
    } SDTCisSameAs_Info;
    struct {
      unsigned OtherOperandNum;
    } SDTCisVTSmallerThanOp_Info;
    struct {
      unsigned BigOperandNum;
    } SDTCisOpSmallerThanOp_Info;
    struct {
      unsigned OtherOperandNum;
    } SDTCisEltOfVec_Info;
    struct {
      unsigned OtherOperandNum;
    } SDTCisSubVecOfVec_Info;
    struct {
      MVT::SimpleValueType VT;
    } SDTCVecEltisVT_Info;
    struct {
      unsigned OtherOperandNum;
    } SDTCisSameNumEltsAs_Info;
    struct {
      unsigned OtherOperandNum;
    } SDTCisSameSizeAs_Info;
  } x;

  /// ApplyTypeConstraint - Given a node in a pattern, apply this type
  /// constraint to the nodes operands.  This returns true if it makes a
  /// change, false otherwise.  If a type contradiction is found, an error
  /// is flagged.
  bool ApplyTypeConstraint(TreePatternNode *N, const SDNodeInfo &NodeInfo,
                           TreePattern &TP) const;
};

/// SDNodeInfo - One of these records is created for each SDNode instance in
/// the target .td file.  This represents the various dag nodes we will be
/// processing.
class SDNodeInfo {
  Record *Def;
  std::string EnumName;
  std::string SDClassName;
  unsigned Properties;
  unsigned NumResults;
  int NumOperands;
  std::vector<SDTypeConstraint> TypeConstraints;
public:
  SDNodeInfo(Record *R);  // Parse the specified record.

  unsigned getNumResults() const { return NumResults; }

  /// getNumOperands - This is the number of operands required or -1 if
  /// variadic.
  int getNumOperands() const { return NumOperands; }
  Record *getRecord() const { return Def; }
  const std::string &getEnumName() const { return EnumName; }
  const std::string &getSDClassName() const { return SDClassName; }

  const std::vector<SDTypeConstraint> &getTypeConstraints() const {
    return TypeConstraints;
  }

  /// getKnownType - If the type constraints on this node imply a fixed type
  /// (e.g. all stores return void, etc), then return it as an
  /// MVT::SimpleValueType.  Otherwise, return MVT::Other.
  MVT::SimpleValueType getKnownType(unsigned ResNo) const;

  /// hasProperty - Return true if this node has the specified property.
  ///
  bool hasProperty(enum SDNP Prop) const { return Properties & (1 << Prop); }

  /// ApplyTypeConstraints - Given a node in a pattern, apply the type
  /// constraints for this node to the operands of the node.  This returns
  /// true if it makes a change, false otherwise.  If a type contradiction is
  /// found, an error is flagged.
  bool ApplyTypeConstraints(TreePatternNode *N, TreePattern &TP) const {
    bool MadeChange = false;
    for (unsigned i = 0, e = TypeConstraints.size(); i != e; ++i)
      MadeChange |= TypeConstraints[i].ApplyTypeConstraint(N, *this, TP);
    return MadeChange;
  }
};
  
/// TreePredicateFn - This is an abstraction that represents the predicates on
/// a PatFrag node.  This is a simple one-word wrapper around a pointer to
/// provide nice accessors.
class TreePredicateFn {
  /// PatFragRec - This is the TreePattern for the PatFrag that we
  /// originally came from.
  TreePattern *PatFragRec;
public:
  /// TreePredicateFn constructor.  Here 'N' is a subclass of PatFrag.
  TreePredicateFn(TreePattern *N);

  
  TreePattern *getOrigPatFragRecord() const { return PatFragRec; }
  
  /// isAlwaysTrue - Return true if this is a noop predicate.
  bool isAlwaysTrue() const;
  
  bool isImmediatePattern() const { return !getImmCode().empty(); }
  
  /// getImmediatePredicateCode - Return the code that evaluates this pattern if
  /// this is an immediate predicate.  It is an error to call this on a
  /// non-immediate pattern.
  std::string getImmediatePredicateCode() const {
    std::string Result = getImmCode();
    assert(!Result.empty() && "Isn't an immediate pattern!");
    return Result;
  }
  
  
  bool operator==(const TreePredicateFn &RHS) const {
    return PatFragRec == RHS.PatFragRec;
  }

  bool operator!=(const TreePredicateFn &RHS) const { return !(*this == RHS); }

  /// Return the name to use in the generated code to reference this, this is
  /// "Predicate_foo" if from a pattern fragment "foo".
  std::string getFnName() const;
  
  /// getCodeToRunOnSDNode - Return the code for the function body that
  /// evaluates this predicate.  The argument is expected to be in "Node",
  /// not N.  This handles casting and conversion to a concrete node type as
  /// appropriate.
  std::string getCodeToRunOnSDNode() const;
  
private:
  std::string getPredCode() const;
  std::string getImmCode() const;
};
  

/// FIXME: TreePatternNode's can be shared in some cases (due to dag-shaped
/// patterns), and as such should be ref counted.  We currently just leak all
/// TreePatternNode objects!
class TreePatternNode {
  /// The type of each node result.  Before and during type inference, each
  /// result may be a set of possible types.  After (successful) type inference,
  /// each is a single concrete type.
  SmallVector<EEVT::TypeSet, 1> Types;

  /// Operator - The Record for the operator if this is an interior node (not
  /// a leaf).
  Record *Operator;

  /// Val - The init value (e.g. the "GPRC" record, or "7") for a leaf.
  ///
  Init *Val;

  /// Name - The name given to this node with the :$foo notation.
  ///
  std::string Name;

  /// PredicateFns - The predicate functions to execute on this node to check
  /// for a match.  If this list is empty, no predicate is involved.
  std::vector<TreePredicateFn> PredicateFns;

  /// TransformFn - The transformation function to execute on this node before
  /// it can be substituted into the resulting instruction on a pattern match.
  Record *TransformFn;

  std::vector<TreePatternNode*> Children;
public:
  TreePatternNode(Record *Op, const std::vector<TreePatternNode*> &Ch,
                  unsigned NumResults)
    : Operator(Op), Val(nullptr), TransformFn(nullptr), Children(Ch) {
    Types.resize(NumResults);
  }
  TreePatternNode(Init *val, unsigned NumResults)    // leaf ctor
    : Operator(nullptr), Val(val), TransformFn(nullptr) {
    Types.resize(NumResults);
  }
  ~TreePatternNode();

  bool hasName() const { return !Name.empty(); }
  const std::string &getName() const { return Name; }
  void setName(StringRef N) { Name.assign(N.begin(), N.end()); }

  bool isLeaf() const { return Val != nullptr; }

  // Type accessors.
  unsigned getNumTypes() const { return Types.size(); }
  MVT::SimpleValueType getType(unsigned ResNo) const {
    return Types[ResNo].getConcrete();
  }
  const SmallVectorImpl<EEVT::TypeSet> &getExtTypes() const { return Types; }
  const EEVT::TypeSet &getExtType(unsigned ResNo) const { return Types[ResNo]; }
  EEVT::TypeSet &getExtType(unsigned ResNo) { return Types[ResNo]; }
  void setType(unsigned ResNo, const EEVT::TypeSet &T) { Types[ResNo] = T; }

  bool hasTypeSet(unsigned ResNo) const {
    return Types[ResNo].isConcrete();
  }
  bool isTypeCompletelyUnknown(unsigned ResNo) const {
    return Types[ResNo].isCompletelyUnknown();
  }
  bool isTypeDynamicallyResolved(unsigned ResNo) const {
    return Types[ResNo].isDynamicallyResolved();
  }

  Init *getLeafValue() const { assert(isLeaf()); return Val; }
  Record *getOperator() const { assert(!isLeaf()); return Operator; }

  unsigned getNumChildren() const { return Children.size(); }
  TreePatternNode *getChild(unsigned N) const { return Children[N]; }
  void setChild(unsigned i, TreePatternNode *N) {
    Children[i] = N;
  }

  /// hasChild - Return true if N is any of our children.
  bool hasChild(const TreePatternNode *N) const {
    for (unsigned i = 0, e = Children.size(); i != e; ++i)
      if (Children[i] == N) return true;
    return false;
  }

  bool hasAnyPredicate() const { return !PredicateFns.empty(); }
  
  const std::vector<TreePredicateFn> &getPredicateFns() const {
    return PredicateFns;
  }
  void clearPredicateFns() { PredicateFns.clear(); }
  void setPredicateFns(const std::vector<TreePredicateFn> &Fns) {
    assert(PredicateFns.empty() && "Overwriting non-empty predicate list!");
    PredicateFns = Fns;
  }
  void addPredicateFn(const TreePredicateFn &Fn) {
    assert(!Fn.isAlwaysTrue() && "Empty predicate string!");
    if (!is_contained(PredicateFns, Fn))
      PredicateFns.push_back(Fn);
  }

  Record *getTransformFn() const { return TransformFn; }
  void setTransformFn(Record *Fn) { TransformFn = Fn; }

  /// getIntrinsicInfo - If this node corresponds to an intrinsic, return the
  /// CodeGenIntrinsic information for it, otherwise return a null pointer.
  const CodeGenIntrinsic *getIntrinsicInfo(const CodeGenDAGPatterns &CDP) const;

  /// getComplexPatternInfo - If this node corresponds to a ComplexPattern,
  /// return the ComplexPattern information, otherwise return null.
  const ComplexPattern *
  getComplexPatternInfo(const CodeGenDAGPatterns &CGP) const;

  /// Returns the number of MachineInstr operands that would be produced by this
  /// node if it mapped directly to an output Instruction's
  /// operand. ComplexPattern specifies this explicitly; MIOperandInfo gives it
  /// for Operands; otherwise 1.
  unsigned getNumMIResults(const CodeGenDAGPatterns &CGP) const;

  /// NodeHasProperty - Return true if this node has the specified property.
  bool NodeHasProperty(SDNP Property, const CodeGenDAGPatterns &CGP) const;

  /// TreeHasProperty - Return true if any node in this tree has the specified
  /// property.
  bool TreeHasProperty(SDNP Property, const CodeGenDAGPatterns &CGP) const;

  /// isCommutativeIntrinsic - Return true if the node is an intrinsic which is
  /// marked isCommutative.
  bool isCommutativeIntrinsic(const CodeGenDAGPatterns &CDP) const;

  void print(raw_ostream &OS) const;
  void dump() const;

public:   // Higher level manipulation routines.

  /// clone - Return a new copy of this tree.
  ///
  TreePatternNode *clone() const;

  /// RemoveAllTypes - Recursively strip all the types of this tree.
  void RemoveAllTypes();

  /// isIsomorphicTo - Return true if this node is recursively isomorphic to
  /// the specified node.  For this comparison, all of the state of the node
  /// is considered, except for the assigned name.  Nodes with differing names
  /// that are otherwise identical are considered isomorphic.
  bool isIsomorphicTo(const TreePatternNode *N,
                      const MultipleUseVarSet &DepVars) const;

  /// SubstituteFormalArguments - Replace the formal arguments in this tree
  /// with actual values specified by ArgMap.
  void SubstituteFormalArguments(std::map<std::string,
                                          TreePatternNode*> &ArgMap);

  /// InlinePatternFragments - If this pattern refers to any pattern
  /// fragments, inline them into place, giving us a pattern without any
  /// PatFrag references.
  TreePatternNode *InlinePatternFragments(TreePattern &TP);

  /// ApplyTypeConstraints - Apply all of the type constraints relevant to
  /// this node and its children in the tree.  This returns true if it makes a
  /// change, false otherwise.  If a type contradiction is found, flag an error.
  bool ApplyTypeConstraints(TreePattern &TP, bool NotRegisters);

  /// UpdateNodeType - Set the node type of N to VT if VT contains
  /// information.  If N already contains a conflicting type, then flag an
  /// error.  This returns true if any information was updated.
  ///
  bool UpdateNodeType(unsigned ResNo, const EEVT::TypeSet &InTy,
                      TreePattern &TP) {
    return Types[ResNo].MergeInTypeInfo(InTy, TP);
  }

  bool UpdateNodeType(unsigned ResNo, MVT::SimpleValueType InTy,
                      TreePattern &TP) {
    return Types[ResNo].MergeInTypeInfo(EEVT::TypeSet(InTy, TP), TP);
  }

  // Update node type with types inferred from an instruction operand or result
  // def from the ins/outs lists.
  // Return true if the type changed.
  bool UpdateNodeTypeFromInst(unsigned ResNo, Record *Operand, TreePattern &TP);

  /// ContainsUnresolvedType - Return true if this tree contains any
  /// unresolved types.
  bool ContainsUnresolvedType() const {
    for (unsigned i = 0, e = Types.size(); i != e; ++i)
      if (!Types[i].isConcrete()) return true;

    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      if (getChild(i)->ContainsUnresolvedType()) return true;
    return false;
  }

  /// canPatternMatch - If it is impossible for this pattern to match on this
  /// target, fill in Reason and return false.  Otherwise, return true.
  bool canPatternMatch(std::string &Reason, const CodeGenDAGPatterns &CDP);
};

inline raw_ostream &operator<<(raw_ostream &OS, const TreePatternNode &TPN) {
  TPN.print(OS);
  return OS;
}


/// TreePattern - Represent a pattern, used for instructions, pattern
/// fragments, etc.
///
class TreePattern {
  /// Trees - The list of pattern trees which corresponds to this pattern.
  /// Note that PatFrag's only have a single tree.
  ///
  std::vector<TreePatternNode*> Trees;

  /// NamedNodes - This is all of the nodes that have names in the trees in this
  /// pattern.
  StringMap<SmallVector<TreePatternNode*,1> > NamedNodes;

  /// TheRecord - The actual TableGen record corresponding to this pattern.
  ///
  Record *TheRecord;

  /// Args - This is a list of all of the arguments to this pattern (for
  /// PatFrag patterns), which are the 'node' markers in this pattern.
  std::vector<std::string> Args;

  /// CDP - the top-level object coordinating this madness.
  ///
  CodeGenDAGPatterns &CDP;

  /// isInputPattern - True if this is an input pattern, something to match.
  /// False if this is an output pattern, something to emit.
  bool isInputPattern;

  /// hasError - True if the currently processed nodes have unresolvable types
  /// or other non-fatal errors
  bool HasError;

  /// It's important that the usage of operands in ComplexPatterns is
  /// consistent: each named operand can be defined by at most one
  /// ComplexPattern. This records the ComplexPattern instance and the operand
  /// number for each operand encountered in a ComplexPattern to aid in that
  /// check.
  StringMap<std::pair<Record *, unsigned>> ComplexPatternOperands;
public:

  /// TreePattern constructor - Parse the specified DagInits into the
  /// current record.
  TreePattern(Record *TheRec, ListInit *RawPat, bool isInput,
              CodeGenDAGPatterns &ise);
  TreePattern(Record *TheRec, DagInit *Pat, bool isInput,
              CodeGenDAGPatterns &ise);
  TreePattern(Record *TheRec, TreePatternNode *Pat, bool isInput,
              CodeGenDAGPatterns &ise);

  /// getTrees - Return the tree patterns which corresponds to this pattern.
  ///
  const std::vector<TreePatternNode*> &getTrees() const { return Trees; }
  unsigned getNumTrees() const { return Trees.size(); }
  TreePatternNode *getTree(unsigned i) const { return Trees[i]; }
  TreePatternNode *getOnlyTree() const {
    assert(Trees.size() == 1 && "Doesn't have exactly one pattern!");
    return Trees[0];
  }

  const StringMap<SmallVector<TreePatternNode*,1> > &getNamedNodesMap() {
    if (NamedNodes.empty())
      ComputeNamedNodes();
    return NamedNodes;
  }

  /// getRecord - Return the actual TableGen record corresponding to this
  /// pattern.
  ///
  Record *getRecord() const { return TheRecord; }

  unsigned getNumArgs() const { return Args.size(); }
  const std::string &getArgName(unsigned i) const {
    assert(i < Args.size() && "Argument reference out of range!");
    return Args[i];
  }
  std::vector<std::string> &getArgList() { return Args; }

  CodeGenDAGPatterns &getDAGPatterns() const { return CDP; }

  /// InlinePatternFragments - If this pattern refers to any pattern
  /// fragments, inline them into place, giving us a pattern without any
  /// PatFrag references.
  void InlinePatternFragments() {
    for (unsigned i = 0, e = Trees.size(); i != e; ++i)
      Trees[i] = Trees[i]->InlinePatternFragments(*this);
  }

  /// InferAllTypes - Infer/propagate as many types throughout the expression
  /// patterns as possible.  Return true if all types are inferred, false
  /// otherwise.  Bail out if a type contradiction is found.
  bool InferAllTypes(const StringMap<SmallVector<TreePatternNode*,1> >
                          *NamedTypes=nullptr);

  /// error - If this is the first error in the current resolution step,
  /// print it and set the error flag.  Otherwise, continue silently.
  void error(const Twine &Msg);
  bool hasError() const {
    return HasError;
  }
  void resetError() {
    HasError = false;
  }

  void print(raw_ostream &OS) const;
  void dump() const;

private:
  TreePatternNode *ParseTreePattern(Init *DI, StringRef OpName);
  void ComputeNamedNodes();
  void ComputeNamedNodes(TreePatternNode *N);
};

/// DAGDefaultOperand - One of these is created for each OperandWithDefaultOps
/// that has a set ExecuteAlways / DefaultOps field.
struct DAGDefaultOperand {
  std::vector<TreePatternNode*> DefaultOps;
};

class DAGInstruction {
  TreePattern *Pattern;
  std::vector<Record*> Results;
  std::vector<Record*> Operands;
  std::vector<Record*> ImpResults;
  TreePatternNode *ResultPattern;
public:
  DAGInstruction(TreePattern *TP,
                 const std::vector<Record*> &results,
                 const std::vector<Record*> &operands,
                 const std::vector<Record*> &impresults)
    : Pattern(TP), Results(results), Operands(operands),
      ImpResults(impresults), ResultPattern(nullptr) {}

  TreePattern *getPattern() const { return Pattern; }
  unsigned getNumResults() const { return Results.size(); }
  unsigned getNumOperands() const { return Operands.size(); }
  unsigned getNumImpResults() const { return ImpResults.size(); }
  const std::vector<Record*>& getImpResults() const { return ImpResults; }

  void setResultPattern(TreePatternNode *R) { ResultPattern = R; }

  Record *getResult(unsigned RN) const {
    assert(RN < Results.size());
    return Results[RN];
  }

  Record *getOperand(unsigned ON) const {
    assert(ON < Operands.size());
    return Operands[ON];
  }

  Record *getImpResult(unsigned RN) const {
    assert(RN < ImpResults.size());
    return ImpResults[RN];
  }

  TreePatternNode *getResultPattern() const { return ResultPattern; }
};

/// PatternToMatch - Used by CodeGenDAGPatterns to keep tab of patterns
/// processed to produce isel.
class PatternToMatch {
public:
  PatternToMatch(Record *srcrecord, ListInit *preds,
                 TreePatternNode *src, TreePatternNode *dst,
                 const std::vector<Record*> &dstregs,
                 int complexity, unsigned uid)
    : SrcRecord(srcrecord), Predicates(preds), SrcPattern(src), DstPattern(dst),
      Dstregs(dstregs), AddedComplexity(complexity), ID(uid) {}

  Record          *SrcRecord;   // Originating Record for the pattern.
  ListInit        *Predicates;  // Top level predicate conditions to match.
  TreePatternNode *SrcPattern;  // Source pattern to match.
  TreePatternNode *DstPattern;  // Resulting pattern.
  std::vector<Record*> Dstregs; // Physical register defs being matched.
  int              AddedComplexity; // Add to matching pattern complexity.
  unsigned         ID;          // Unique ID for the record.

  Record          *getSrcRecord()  const { return SrcRecord; }
  ListInit        *getPredicates() const { return Predicates; }
  TreePatternNode *getSrcPattern() const { return SrcPattern; }
  TreePatternNode *getDstPattern() const { return DstPattern; }
  const std::vector<Record*> &getDstRegs() const { return Dstregs; }
  int         getAddedComplexity() const { return AddedComplexity; }

  std::string getPredicateCheck() const;

  /// Compute the complexity metric for the input pattern.  This roughly
  /// corresponds to the number of nodes that are covered.
  int getPatternComplexity(const CodeGenDAGPatterns &CGP) const;
};

class CodeGenDAGPatterns {
  RecordKeeper &Records;
  CodeGenTarget Target;
  CodeGenIntrinsicTable Intrinsics;
  CodeGenIntrinsicTable TgtIntrinsics;

  std::map<Record*, SDNodeInfo, LessRecordByID> SDNodes;
  std::map<Record*, std::pair<Record*, std::string>, LessRecordByID> SDNodeXForms;
  std::map<Record*, ComplexPattern, LessRecordByID> ComplexPatterns;
  std::map<Record *, std::unique_ptr<TreePattern>, LessRecordByID>
      PatternFragments;
  std::map<Record*, DAGDefaultOperand, LessRecordByID> DefaultOperands;
  std::map<Record*, DAGInstruction, LessRecordByID> Instructions;

  // Specific SDNode definitions:
  Record *intrinsic_void_sdnode;
  Record *intrinsic_w_chain_sdnode, *intrinsic_wo_chain_sdnode;

  /// PatternsToMatch - All of the things we are matching on the DAG.  The first
  /// value is the pattern to match, the second pattern is the result to
  /// emit.
  std::vector<PatternToMatch> PatternsToMatch;
public:
  CodeGenDAGPatterns(RecordKeeper &R);

  CodeGenTarget &getTargetInfo() { return Target; }
  const CodeGenTarget &getTargetInfo() const { return Target; }

  Record *getSDNodeNamed(const std::string &Name) const;

  const SDNodeInfo &getSDNodeInfo(Record *R) const {
    assert(SDNodes.count(R) && "Unknown node!");
    return SDNodes.find(R)->second;
  }

  // Node transformation lookups.
  typedef std::pair<Record*, std::string> NodeXForm;
  const NodeXForm &getSDNodeTransform(Record *R) const {
    assert(SDNodeXForms.count(R) && "Invalid transform!");
    return SDNodeXForms.find(R)->second;
  }

  typedef std::map<Record*, NodeXForm, LessRecordByID>::const_iterator
          nx_iterator;
  nx_iterator nx_begin() const { return SDNodeXForms.begin(); }
  nx_iterator nx_end() const { return SDNodeXForms.end(); }


  const ComplexPattern &getComplexPattern(Record *R) const {
    assert(ComplexPatterns.count(R) && "Unknown addressing mode!");
    return ComplexPatterns.find(R)->second;
  }

  const CodeGenIntrinsic &getIntrinsic(Record *R) const {
    for (unsigned i = 0, e = Intrinsics.size(); i != e; ++i)
      if (Intrinsics[i].TheDef == R) return Intrinsics[i];
    for (unsigned i = 0, e = TgtIntrinsics.size(); i != e; ++i)
      if (TgtIntrinsics[i].TheDef == R) return TgtIntrinsics[i];
    llvm_unreachable("Unknown intrinsic!");
  }

  const CodeGenIntrinsic &getIntrinsicInfo(unsigned IID) const {
    if (IID-1 < Intrinsics.size())
      return Intrinsics[IID-1];
    if (IID-Intrinsics.size()-1 < TgtIntrinsics.size())
      return TgtIntrinsics[IID-Intrinsics.size()-1];
    llvm_unreachable("Bad intrinsic ID!");
  }

  unsigned getIntrinsicID(Record *R) const {
    for (unsigned i = 0, e = Intrinsics.size(); i != e; ++i)
      if (Intrinsics[i].TheDef == R) return i;
    for (unsigned i = 0, e = TgtIntrinsics.size(); i != e; ++i)
      if (TgtIntrinsics[i].TheDef == R) return i + Intrinsics.size();
    llvm_unreachable("Unknown intrinsic!");
  }

  const DAGDefaultOperand &getDefaultOperand(Record *R) const {
    assert(DefaultOperands.count(R) &&"Isn't an analyzed default operand!");
    return DefaultOperands.find(R)->second;
  }

  // Pattern Fragment information.
  TreePattern *getPatternFragment(Record *R) const {
    assert(PatternFragments.count(R) && "Invalid pattern fragment request!");
    return PatternFragments.find(R)->second.get();
  }
  TreePattern *getPatternFragmentIfRead(Record *R) const {
    if (!PatternFragments.count(R))
      return nullptr;
    return PatternFragments.find(R)->second.get();
  }

  typedef std::map<Record *, std::unique_ptr<TreePattern>,
                   LessRecordByID>::const_iterator pf_iterator;
  pf_iterator pf_begin() const { return PatternFragments.begin(); }
  pf_iterator pf_end() const { return PatternFragments.end(); }

  // Patterns to match information.
  typedef std::vector<PatternToMatch>::const_iterator ptm_iterator;
  ptm_iterator ptm_begin() const { return PatternsToMatch.begin(); }
  ptm_iterator ptm_end() const { return PatternsToMatch.end(); }

  /// Parse the Pattern for an instruction, and insert the result in DAGInsts.
  typedef std::map<Record*, DAGInstruction, LessRecordByID> DAGInstMap;
  const DAGInstruction &parseInstructionPattern(
      CodeGenInstruction &CGI, ListInit *Pattern,
      DAGInstMap &DAGInsts);

  const DAGInstruction &getInstruction(Record *R) const {
    assert(Instructions.count(R) && "Unknown instruction!");
    return Instructions.find(R)->second;
  }

  Record *get_intrinsic_void_sdnode() const {
    return intrinsic_void_sdnode;
  }
  Record *get_intrinsic_w_chain_sdnode() const {
    return intrinsic_w_chain_sdnode;
  }
  Record *get_intrinsic_wo_chain_sdnode() const {
    return intrinsic_wo_chain_sdnode;
  }

  bool hasTargetIntrinsics() { return !TgtIntrinsics.empty(); }

private:
  void ParseNodeInfo();
  void ParseNodeTransforms();
  void ParseComplexPatterns();
  void ParsePatternFragments(bool OutFrags = false);
  void ParseDefaultOperands();
  void ParseInstructions();
  void ParsePatterns();
  void InferInstructionFlags();
  void GenerateVariants();
  void VerifyInstructionFlags();

  void AddPatternToMatch(TreePattern *Pattern, const PatternToMatch &PTM);
  void FindPatternInputsAndOutputs(TreePattern *I, TreePatternNode *Pat,
                                   std::map<std::string,
                                   TreePatternNode*> &InstInputs,
                                   std::map<std::string,
                                   TreePatternNode*> &InstResults,
                                   std::vector<Record*> &InstImpResults);
};
} // end namespace llvm

#endif
