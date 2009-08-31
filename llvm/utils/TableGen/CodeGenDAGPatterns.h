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

#ifndef CODEGEN_DAGPATTERNS_H
#define CODEGEN_DAGPATTERNS_H

#include <set>
#include <algorithm>
#include <vector>

#include "CodeGenTarget.h"
#include "CodeGenIntrinsics.h"

namespace llvm {
  class Record;
  struct Init;
  class ListInit;
  class DagInit;
  class SDNodeInfo;
  class TreePattern;
  class TreePatternNode;
  class CodeGenDAGPatterns;
  class ComplexPattern;

/// EEVT::DAGISelGenValueType - These are some extended forms of
/// MVT::SimpleValueType that we use as lattice values during type inference.
namespace EEVT {
  enum DAGISelGenValueType {
    isFP  = MVT::LAST_VALUETYPE,
    isInt,
    isVec,
    isUnknown
  };

  /// isExtIntegerInVTs - Return true if the specified extended value type
  /// vector contains isInt or an integer value type.
  bool isExtIntegerInVTs(const std::vector<unsigned char> &EVTs);

  /// isExtFloatingPointInVTs - Return true if the specified extended value
  /// type vector contains isFP or a FP value type.
  bool isExtFloatingPointInVTs(const std::vector<unsigned char> &EVTs);

  /// isExtVectorinVTs - Return true if the specified extended value type 
  /// vector contains isVec or a vector value type.
  bool isExtVectorInVTs(const std::vector<unsigned char> &EVTs);
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
    SDTCisVTSmallerThanOp, SDTCisOpSmallerThanOp, SDTCisEltOfVec
  } ConstraintType;
  
  union {   // The discriminated union.
    struct {
      unsigned char VT;
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
  } x;

  /// ApplyTypeConstraint - Given a node in a pattern, apply this type
  /// constraint to the nodes operands.  This returns true if it makes a
  /// change, false otherwise.  If a type contradiction is found, throw an
  /// exception.
  bool ApplyTypeConstraint(TreePatternNode *N, const SDNodeInfo &NodeInfo,
                           TreePattern &TP) const;
  
  /// getOperandNum - Return the node corresponding to operand #OpNo in tree
  /// N, which has NumResults results.
  TreePatternNode *getOperandNum(unsigned OpNo, TreePatternNode *N,
                                 unsigned NumResults) const;
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
  int getNumOperands() const { return NumOperands; }
  Record *getRecord() const { return Def; }
  const std::string &getEnumName() const { return EnumName; }
  const std::string &getSDClassName() const { return SDClassName; }
  
  const std::vector<SDTypeConstraint> &getTypeConstraints() const {
    return TypeConstraints;
  }
  
  /// hasProperty - Return true if this node has the specified property.
  ///
  bool hasProperty(enum SDNP Prop) const { return Properties & (1 << Prop); }

  /// ApplyTypeConstraints - Given a node in a pattern, apply the type
  /// constraints for this node to the operands of the node.  This returns
  /// true if it makes a change, false otherwise.  If a type contradiction is
  /// found, throw an exception.
  bool ApplyTypeConstraints(TreePatternNode *N, TreePattern &TP) const {
    bool MadeChange = false;
    for (unsigned i = 0, e = TypeConstraints.size(); i != e; ++i)
      MadeChange |= TypeConstraints[i].ApplyTypeConstraint(N, *this, TP);
    return MadeChange;
  }
};

/// FIXME: TreePatternNode's can be shared in some cases (due to dag-shaped
/// patterns), and as such should be ref counted.  We currently just leak all
/// TreePatternNode objects!
class TreePatternNode {
  /// The inferred type for this node, or EEVT::isUnknown if it hasn't
  /// been determined yet. This is a std::vector because during inference
  /// there may be multiple possible types.
  std::vector<unsigned char> Types;
  
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
  std::vector<std::string> PredicateFns;
  
  /// TransformFn - The transformation function to execute on this node before
  /// it can be substituted into the resulting instruction on a pattern match.
  Record *TransformFn;
  
  std::vector<TreePatternNode*> Children;
public:
  TreePatternNode(Record *Op, const std::vector<TreePatternNode*> &Ch) 
    : Types(), Operator(Op), Val(0), TransformFn(0),
    Children(Ch) { Types.push_back(EEVT::isUnknown); }
  TreePatternNode(Init *val)    // leaf ctor
    : Types(), Operator(0), Val(val), TransformFn(0) {
    Types.push_back(EEVT::isUnknown);
  }
  ~TreePatternNode();
  
  const std::string &getName() const { return Name; }
  void setName(const std::string &N) { Name = N; }
  
  bool isLeaf() const { return Val != 0; }
  bool hasTypeSet() const {
    return (Types[0] < MVT::LAST_VALUETYPE) || (Types[0] == MVT::iPTR) || 
          (Types[0] == MVT::iPTRAny);
  }
  bool isTypeCompletelyUnknown() const {
    return Types[0] == EEVT::isUnknown;
  }
  bool isTypeDynamicallyResolved() const {
    return (Types[0] == MVT::iPTR) || (Types[0] == MVT::iPTRAny);
  }
  MVT::SimpleValueType getTypeNum(unsigned Num) const {
    assert(hasTypeSet() && "Doesn't have a type yet!");
    assert(Types.size() > Num && "Type num out of range!");
    return (MVT::SimpleValueType)Types[Num];
  }
  unsigned char getExtTypeNum(unsigned Num) const { 
    assert(Types.size() > Num && "Extended type num out of range!");
    return Types[Num]; 
  }
  const std::vector<unsigned char> &getExtTypes() const { return Types; }
  void setTypes(const std::vector<unsigned char> &T) { Types = T; }
  void removeTypes() { Types = std::vector<unsigned char>(1, EEVT::isUnknown); }
  
  Init *getLeafValue() const { assert(isLeaf()); return Val; }
  Record *getOperator() const { assert(!isLeaf()); return Operator; }
  
  unsigned getNumChildren() const { return Children.size(); }
  TreePatternNode *getChild(unsigned N) const { return Children[N]; }
  void setChild(unsigned i, TreePatternNode *N) {
    Children[i] = N;
  }

  const std::vector<std::string> &getPredicateFns() const { return PredicateFns; }
  void clearPredicateFns() { PredicateFns.clear(); }
  void setPredicateFns(const std::vector<std::string> &Fns) {
    assert(PredicateFns.empty() && "Overwriting non-empty predicate list!");
    PredicateFns = Fns;
  }
  void addPredicateFn(const std::string &Fn) { 
    assert(!Fn.empty() && "Empty predicate string!");
    if (std::find(PredicateFns.begin(), PredicateFns.end(), Fn) ==
          PredicateFns.end())
      PredicateFns.push_back(Fn);
  }

  Record *getTransformFn() const { return TransformFn; }
  void setTransformFn(Record *Fn) { TransformFn = Fn; }
  
  /// getIntrinsicInfo - If this node corresponds to an intrinsic, return the
  /// CodeGenIntrinsic information for it, otherwise return a null pointer.
  const CodeGenIntrinsic *getIntrinsicInfo(const CodeGenDAGPatterns &CDP) const;

  /// isCommutativeIntrinsic - Return true if the node is an intrinsic which is
  /// marked isCommutative.
  bool isCommutativeIntrinsic(const CodeGenDAGPatterns &CDP) const;
  
  void print(raw_ostream &OS) const;
  void dump() const;
  
public:   // Higher level manipulation routines.

  /// clone - Return a new copy of this tree.
  ///
  TreePatternNode *clone() const;
  
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
  /// change, false otherwise.  If a type contradiction is found, throw an
  /// exception.
  bool ApplyTypeConstraints(TreePattern &TP, bool NotRegisters);
  
  /// UpdateNodeType - Set the node type of N to VT if VT contains
  /// information.  If N already contains a conflicting type, then throw an
  /// exception.  This returns true if any information was updated.
  ///
  bool UpdateNodeType(const std::vector<unsigned char> &ExtVTs,
                      TreePattern &TP);
  bool UpdateNodeType(unsigned char ExtVT, TreePattern &TP) {
    std::vector<unsigned char> ExtVTs(1, ExtVT);
    return UpdateNodeType(ExtVTs, TP);
  }
  
  /// ContainsUnresolvedType - Return true if this tree contains any
  /// unresolved types.
  bool ContainsUnresolvedType() const {
    if (!hasTypeSet() && !isTypeDynamicallyResolved()) return true;
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      if (getChild(i)->ContainsUnresolvedType()) return true;
    return false;
  }
  
  /// canPatternMatch - If it is impossible for this pattern to match on this
  /// target, fill in Reason and return false.  Otherwise, return true.
  bool canPatternMatch(std::string &Reason, const CodeGenDAGPatterns &CDP);
};


/// TreePattern - Represent a pattern, used for instructions, pattern
/// fragments, etc.
///
class TreePattern {
  /// Trees - The list of pattern trees which corresponds to this pattern.
  /// Note that PatFrag's only have a single tree.
  ///
  std::vector<TreePatternNode*> Trees;
  
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
  /// otherwise.  Throw an exception if a type contradiction is found.
  bool InferAllTypes();
  
  /// error - Throw an exception, prefixing it with information about this
  /// pattern.
  void error(const std::string &Msg) const;
  
  void print(raw_ostream &OS) const;
  void dump() const;
  
private:
  TreePatternNode *ParseTreePattern(DagInit *DI);
};

/// DAGDefaultOperand - One of these is created for each PredicateOperand
/// or OptionalDefOperand that has a set ExecuteAlways / DefaultOps field.
struct DAGDefaultOperand {
  std::vector<TreePatternNode*> DefaultOps;
};

class DAGInstruction {
  TreePattern *Pattern;
  std::vector<Record*> Results;
  std::vector<Record*> Operands;
  std::vector<Record*> ImpResults;
  std::vector<Record*> ImpOperands;
  TreePatternNode *ResultPattern;
public:
  DAGInstruction(TreePattern *TP,
                 const std::vector<Record*> &results,
                 const std::vector<Record*> &operands,
                 const std::vector<Record*> &impresults,
                 const std::vector<Record*> &impoperands)
    : Pattern(TP), Results(results), Operands(operands), 
      ImpResults(impresults), ImpOperands(impoperands),
      ResultPattern(0) {}

  const TreePattern *getPattern() const { return Pattern; }
  unsigned getNumResults() const { return Results.size(); }
  unsigned getNumOperands() const { return Operands.size(); }
  unsigned getNumImpResults() const { return ImpResults.size(); }
  unsigned getNumImpOperands() const { return ImpOperands.size(); }
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
  
  Record *getImpOperand(unsigned ON) const {
    assert(ON < ImpOperands.size());
    return ImpOperands[ON];
  }

  TreePatternNode *getResultPattern() const { return ResultPattern; }
};
  
/// PatternToMatch - Used by CodeGenDAGPatterns to keep tab of patterns
/// processed to produce isel.
struct PatternToMatch {
  PatternToMatch(ListInit *preds,
                 TreePatternNode *src, TreePatternNode *dst,
                 const std::vector<Record*> &dstregs,
                 unsigned complexity):
    Predicates(preds), SrcPattern(src), DstPattern(dst), Dstregs(dstregs),
    AddedComplexity(complexity) {};

  ListInit        *Predicates;  // Top level predicate conditions to match.
  TreePatternNode *SrcPattern;  // Source pattern to match.
  TreePatternNode *DstPattern;  // Resulting pattern.
  std::vector<Record*> Dstregs; // Physical register defs being matched.
  unsigned         AddedComplexity; // Add to matching pattern complexity.

  ListInit        *getPredicates() const { return Predicates; }
  TreePatternNode *getSrcPattern() const { return SrcPattern; }
  TreePatternNode *getDstPattern() const { return DstPattern; }
  const std::vector<Record*> &getDstRegs() const { return Dstregs; }
  unsigned         getAddedComplexity() const { return AddedComplexity; }

  std::string getPredicateCheck() const;
};

// Deterministic comparison of Record*.
struct RecordPtrCmp {
  bool operator()(const Record *LHS, const Record *RHS) const;
};
  
class CodeGenDAGPatterns {
  RecordKeeper &Records;
  CodeGenTarget Target;
  std::vector<CodeGenIntrinsic> Intrinsics;
  std::vector<CodeGenIntrinsic> TgtIntrinsics;
  
  std::map<Record*, SDNodeInfo, RecordPtrCmp> SDNodes;
  std::map<Record*, std::pair<Record*, std::string>, RecordPtrCmp> SDNodeXForms;
  std::map<Record*, ComplexPattern, RecordPtrCmp> ComplexPatterns;
  std::map<Record*, TreePattern*, RecordPtrCmp> PatternFragments;
  std::map<Record*, DAGDefaultOperand, RecordPtrCmp> DefaultOperands;
  std::map<Record*, DAGInstruction, RecordPtrCmp> Instructions;
  
  // Specific SDNode definitions:
  Record *intrinsic_void_sdnode;
  Record *intrinsic_w_chain_sdnode, *intrinsic_wo_chain_sdnode;
  
  /// PatternsToMatch - All of the things we are matching on the DAG.  The first
  /// value is the pattern to match, the second pattern is the result to
  /// emit.
  std::vector<PatternToMatch> PatternsToMatch;
public:
  CodeGenDAGPatterns(RecordKeeper &R); 
  ~CodeGenDAGPatterns();
  
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
  
  typedef std::map<Record*, NodeXForm, RecordPtrCmp>::const_iterator
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
    assert(0 && "Unknown intrinsic!");
    abort();
  }
  
  const CodeGenIntrinsic &getIntrinsicInfo(unsigned IID) const {
    if (IID-1 < Intrinsics.size())
      return Intrinsics[IID-1];
    if (IID-Intrinsics.size()-1 < TgtIntrinsics.size())
      return TgtIntrinsics[IID-Intrinsics.size()-1];
    assert(0 && "Bad intrinsic ID!");
    abort();
  }
  
  unsigned getIntrinsicID(Record *R) const {
    for (unsigned i = 0, e = Intrinsics.size(); i != e; ++i)
      if (Intrinsics[i].TheDef == R) return i;
    for (unsigned i = 0, e = TgtIntrinsics.size(); i != e; ++i)
      if (TgtIntrinsics[i].TheDef == R) return i + Intrinsics.size();
    assert(0 && "Unknown intrinsic!");
    abort();
  }
  
  const DAGDefaultOperand &getDefaultOperand(Record *R) {
    assert(DefaultOperands.count(R) &&"Isn't an analyzed default operand!");
    return DefaultOperands.find(R)->second;
  }
  
  // Pattern Fragment information.
  TreePattern *getPatternFragment(Record *R) const {
    assert(PatternFragments.count(R) && "Invalid pattern fragment request!");
    return PatternFragments.find(R)->second;
  }
  typedef std::map<Record*, TreePattern*, RecordPtrCmp>::const_iterator
          pf_iterator;
  pf_iterator pf_begin() const { return PatternFragments.begin(); }
  pf_iterator pf_end() const { return PatternFragments.end(); }

  // Patterns to match information.
  typedef std::vector<PatternToMatch>::const_iterator ptm_iterator;
  ptm_iterator ptm_begin() const { return PatternsToMatch.begin(); }
  ptm_iterator ptm_end() const { return PatternsToMatch.end(); }
  
  
  
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
  
private:
  void ParseNodeInfo();
  void ParseNodeTransforms();
  void ParseComplexPatterns();
  void ParsePatternFragments();
  void ParseDefaultOperands();
  void ParseInstructions();
  void ParsePatterns();
  void InferInstructionFlags();
  void GenerateVariants();
  
  void FindPatternInputsAndOutputs(TreePattern *I, TreePatternNode *Pat,
                                   std::map<std::string,
                                   TreePatternNode*> &InstInputs,
                                   std::map<std::string,
                                   TreePatternNode*> &InstResults,
                                   std::vector<Record*> &InstImpInputs,
                                   std::vector<Record*> &InstImpResults);
};
} // end namespace llvm

#endif
