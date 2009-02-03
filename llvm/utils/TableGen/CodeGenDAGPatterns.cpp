//===- CodeGenDAGPatterns.cpp - Read DAG patterns from .td file -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CodeGenDAGPatterns class, which is used to read and
// represent the patterns present in a .td file for instructions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenDAGPatterns.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Streams.h"
#include <set>
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Helpers for working with extended types.

/// FilterVTs - Filter a list of VT's according to a predicate.
///
template<typename T>
static std::vector<MVT::SimpleValueType>
FilterVTs(const std::vector<MVT::SimpleValueType> &InVTs, T Filter) {
  std::vector<MVT::SimpleValueType> Result;
  for (unsigned i = 0, e = InVTs.size(); i != e; ++i)
    if (Filter(InVTs[i]))
      Result.push_back(InVTs[i]);
  return Result;
}

template<typename T>
static std::vector<unsigned char> 
FilterEVTs(const std::vector<unsigned char> &InVTs, T Filter) {
  std::vector<unsigned char> Result;
  for (unsigned i = 0, e = InVTs.size(); i != e; ++i)
    if (Filter((MVT::SimpleValueType)InVTs[i]))
      Result.push_back(InVTs[i]);
  return Result;
}

static std::vector<unsigned char>
ConvertVTs(const std::vector<MVT::SimpleValueType> &InVTs) {
  std::vector<unsigned char> Result;
  for (unsigned i = 0, e = InVTs.size(); i != e; ++i)
    Result.push_back(InVTs[i]);
  return Result;
}

static inline bool isInteger(MVT::SimpleValueType VT) {
  return MVT(VT).isInteger();
}

static inline bool isFloatingPoint(MVT::SimpleValueType VT) {
  return MVT(VT).isFloatingPoint();
}

static inline bool isVector(MVT::SimpleValueType VT) {
  return MVT(VT).isVector();
}

static bool LHSIsSubsetOfRHS(const std::vector<unsigned char> &LHS,
                             const std::vector<unsigned char> &RHS) {
  if (LHS.size() > RHS.size()) return false;
  for (unsigned i = 0, e = LHS.size(); i != e; ++i)
    if (std::find(RHS.begin(), RHS.end(), LHS[i]) == RHS.end())
      return false;
  return true;
}

/// isExtIntegerVT - Return true if the specified extended value type vector
/// contains isInt or an integer value type.
namespace llvm {
namespace EMVT {
bool isExtIntegerInVTs(const std::vector<unsigned char> &EVTs) {
  assert(!EVTs.empty() && "Cannot check for integer in empty ExtVT list!");
  return EVTs[0] == isInt || !(FilterEVTs(EVTs, isInteger).empty());
}

/// isExtFloatingPointVT - Return true if the specified extended value type 
/// vector contains isFP or a FP value type.
bool isExtFloatingPointInVTs(const std::vector<unsigned char> &EVTs) {
  assert(!EVTs.empty() && "Cannot check for integer in empty ExtVT list!");
  return EVTs[0] == isFP || !(FilterEVTs(EVTs, isFloatingPoint).empty());
}
} // end namespace EMVT.
} // end namespace llvm.


/// Dependent variable map for CodeGenDAGPattern variant generation
typedef std::map<std::string, int> DepVarMap;

/// Const iterator shorthand for DepVarMap
typedef DepVarMap::const_iterator DepVarMap_citer;

namespace {
void FindDepVarsOf(TreePatternNode *N, DepVarMap &DepMap) {
  if (N->isLeaf()) {
    if (dynamic_cast<DefInit*>(N->getLeafValue()) != NULL) {
      DepMap[N->getName()]++;
    }
  } else {
    for (size_t i = 0, e = N->getNumChildren(); i != e; ++i)
      FindDepVarsOf(N->getChild(i), DepMap);
  }
}

//! Find dependent variables within child patterns
/*!
 */
void FindDepVars(TreePatternNode *N, MultipleUseVarSet &DepVars) {
  DepVarMap depcounts;
  FindDepVarsOf(N, depcounts);
  for (DepVarMap_citer i = depcounts.begin(); i != depcounts.end(); ++i) {
    if (i->second > 1) {            // std::pair<std::string, int>
      DepVars.insert(i->first);
    }
  }
}

//! Dump the dependent variable set:
void DumpDepVars(MultipleUseVarSet &DepVars) {
  if (DepVars.empty()) {
    DOUT << "<empty set>";
  } else {
    DOUT << "[ ";
    for (MultipleUseVarSet::const_iterator i = DepVars.begin(), e = DepVars.end();
         i != e; ++i) {
      DOUT << (*i) << " ";
    }
    DOUT << "]";
  }
}
}

//===----------------------------------------------------------------------===//
// PatternToMatch implementation
//

/// getPredicateCheck - Return a single string containing all of this
/// pattern's predicates concatenated with "&&" operators.
///
std::string PatternToMatch::getPredicateCheck() const {
  std::string PredicateCheck;
  for (unsigned i = 0, e = Predicates->getSize(); i != e; ++i) {
    if (DefInit *Pred = dynamic_cast<DefInit*>(Predicates->getElement(i))) {
      Record *Def = Pred->getDef();
      if (!Def->isSubClassOf("Predicate")) {
#ifndef NDEBUG
        Def->dump();
#endif
        assert(0 && "Unknown predicate type!");
      }
      if (!PredicateCheck.empty())
        PredicateCheck += " && ";
      PredicateCheck += "(" + Def->getValueAsString("CondString") + ")";
    }
  }

  return PredicateCheck;
}

//===----------------------------------------------------------------------===//
// SDTypeConstraint implementation
//

SDTypeConstraint::SDTypeConstraint(Record *R) {
  OperandNo = R->getValueAsInt("OperandNum");
  
  if (R->isSubClassOf("SDTCisVT")) {
    ConstraintType = SDTCisVT;
    x.SDTCisVT_Info.VT = getValueType(R->getValueAsDef("VT"));
  } else if (R->isSubClassOf("SDTCisPtrTy")) {
    ConstraintType = SDTCisPtrTy;
  } else if (R->isSubClassOf("SDTCisInt")) {
    ConstraintType = SDTCisInt;
  } else if (R->isSubClassOf("SDTCisFP")) {
    ConstraintType = SDTCisFP;
  } else if (R->isSubClassOf("SDTCisSameAs")) {
    ConstraintType = SDTCisSameAs;
    x.SDTCisSameAs_Info.OtherOperandNum = R->getValueAsInt("OtherOperandNum");
  } else if (R->isSubClassOf("SDTCisVTSmallerThanOp")) {
    ConstraintType = SDTCisVTSmallerThanOp;
    x.SDTCisVTSmallerThanOp_Info.OtherOperandNum = 
      R->getValueAsInt("OtherOperandNum");
  } else if (R->isSubClassOf("SDTCisOpSmallerThanOp")) {
    ConstraintType = SDTCisOpSmallerThanOp;
    x.SDTCisOpSmallerThanOp_Info.BigOperandNum = 
      R->getValueAsInt("BigOperandNum");
  } else if (R->isSubClassOf("SDTCisIntVectorOfSameSize")) {
    ConstraintType = SDTCisIntVectorOfSameSize;
    x.SDTCisIntVectorOfSameSize_Info.OtherOperandNum =
      R->getValueAsInt("OtherOpNum");
  } else if (R->isSubClassOf("SDTCisEltOfVec")) {
    ConstraintType = SDTCisEltOfVec;
    x.SDTCisEltOfVec_Info.OtherOperandNum =
      R->getValueAsInt("OtherOpNum");
  } else {
    cerr << "Unrecognized SDTypeConstraint '" << R->getName() << "'!\n";
    exit(1);
  }
}

/// getOperandNum - Return the node corresponding to operand #OpNo in tree
/// N, which has NumResults results.
TreePatternNode *SDTypeConstraint::getOperandNum(unsigned OpNo,
                                                 TreePatternNode *N,
                                                 unsigned NumResults) const {
  assert(NumResults <= 1 &&
         "We only work with nodes with zero or one result so far!");
  
  if (OpNo >= (NumResults + N->getNumChildren())) {
    cerr << "Invalid operand number " << OpNo << " ";
    N->dump();
    cerr << '\n';
    exit(1);
  }

  if (OpNo < NumResults)
    return N;  // FIXME: need value #
  else
    return N->getChild(OpNo-NumResults);
}

/// ApplyTypeConstraint - Given a node in a pattern, apply this type
/// constraint to the nodes operands.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, throw an
/// exception.
bool SDTypeConstraint::ApplyTypeConstraint(TreePatternNode *N,
                                           const SDNodeInfo &NodeInfo,
                                           TreePattern &TP) const {
  unsigned NumResults = NodeInfo.getNumResults();
  assert(NumResults <= 1 &&
         "We only work with nodes with zero or one result so far!");
  
  // Check that the number of operands is sane.  Negative operands -> varargs.
  if (NodeInfo.getNumOperands() >= 0) {
    if (N->getNumChildren() != (unsigned)NodeInfo.getNumOperands())
      TP.error(N->getOperator()->getName() + " node requires exactly " +
               itostr(NodeInfo.getNumOperands()) + " operands!");
  }

  const CodeGenTarget &CGT = TP.getDAGPatterns().getTargetInfo();
  
  TreePatternNode *NodeToApply = getOperandNum(OperandNo, N, NumResults);
  
  switch (ConstraintType) {
  default: assert(0 && "Unknown constraint type!");
  case SDTCisVT:
    // Operand must be a particular type.
    return NodeToApply->UpdateNodeType(x.SDTCisVT_Info.VT, TP);
  case SDTCisPtrTy: {
    // Operand must be same as target pointer type.
    return NodeToApply->UpdateNodeType(MVT::iPTR, TP);
  }
  case SDTCisInt: {
    // If there is only one integer type supported, this must be it.
    std::vector<MVT::SimpleValueType> IntVTs =
      FilterVTs(CGT.getLegalValueTypes(), isInteger);

    // If we found exactly one supported integer type, apply it.
    if (IntVTs.size() == 1)
      return NodeToApply->UpdateNodeType(IntVTs[0], TP);
    return NodeToApply->UpdateNodeType(EMVT::isInt, TP);
  }
  case SDTCisFP: {
    // If there is only one FP type supported, this must be it.
    std::vector<MVT::SimpleValueType> FPVTs =
      FilterVTs(CGT.getLegalValueTypes(), isFloatingPoint);
        
    // If we found exactly one supported FP type, apply it.
    if (FPVTs.size() == 1)
      return NodeToApply->UpdateNodeType(FPVTs[0], TP);
    return NodeToApply->UpdateNodeType(EMVT::isFP, TP);
  }
  case SDTCisSameAs: {
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisSameAs_Info.OtherOperandNum, N, NumResults);
    return NodeToApply->UpdateNodeType(OtherNode->getExtTypes(), TP) |
           OtherNode->UpdateNodeType(NodeToApply->getExtTypes(), TP);
  }
  case SDTCisVTSmallerThanOp: {
    // The NodeToApply must be a leaf node that is a VT.  OtherOperandNum must
    // have an integer type that is smaller than the VT.
    if (!NodeToApply->isLeaf() ||
        !dynamic_cast<DefInit*>(NodeToApply->getLeafValue()) ||
        !static_cast<DefInit*>(NodeToApply->getLeafValue())->getDef()
               ->isSubClassOf("ValueType"))
      TP.error(N->getOperator()->getName() + " expects a VT operand!");
    MVT::SimpleValueType VT =
     getValueType(static_cast<DefInit*>(NodeToApply->getLeafValue())->getDef());
    if (!isInteger(VT))
      TP.error(N->getOperator()->getName() + " VT operand must be integer!");
    
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisVTSmallerThanOp_Info.OtherOperandNum, N,NumResults);
    
    // It must be integer.
    bool MadeChange = false;
    MadeChange |= OtherNode->UpdateNodeType(EMVT::isInt, TP);
    
    // This code only handles nodes that have one type set.  Assert here so
    // that we can change this if we ever need to deal with multiple value
    // types at this point.
    assert(OtherNode->getExtTypes().size() == 1 && "Node has too many types!");
    if (OtherNode->hasTypeSet() && OtherNode->getTypeNum(0) <= VT)
      OtherNode->UpdateNodeType(MVT::Other, TP);  // Throw an error.
    return false;
  }
  case SDTCisOpSmallerThanOp: {
    TreePatternNode *BigOperand =
      getOperandNum(x.SDTCisOpSmallerThanOp_Info.BigOperandNum, N, NumResults);

    // Both operands must be integer or FP, but we don't care which.
    bool MadeChange = false;
    
    // This code does not currently handle nodes which have multiple types,
    // where some types are integer, and some are fp.  Assert that this is not
    // the case.
    assert(!(EMVT::isExtIntegerInVTs(NodeToApply->getExtTypes()) &&
             EMVT::isExtFloatingPointInVTs(NodeToApply->getExtTypes())) &&
           !(EMVT::isExtIntegerInVTs(BigOperand->getExtTypes()) &&
             EMVT::isExtFloatingPointInVTs(BigOperand->getExtTypes())) &&
           "SDTCisOpSmallerThanOp does not handle mixed int/fp types!");
    if (EMVT::isExtIntegerInVTs(NodeToApply->getExtTypes()))
      MadeChange |= BigOperand->UpdateNodeType(EMVT::isInt, TP);
    else if (EMVT::isExtFloatingPointInVTs(NodeToApply->getExtTypes()))
      MadeChange |= BigOperand->UpdateNodeType(EMVT::isFP, TP);
    if (EMVT::isExtIntegerInVTs(BigOperand->getExtTypes()))
      MadeChange |= NodeToApply->UpdateNodeType(EMVT::isInt, TP);
    else if (EMVT::isExtFloatingPointInVTs(BigOperand->getExtTypes()))
      MadeChange |= NodeToApply->UpdateNodeType(EMVT::isFP, TP);

    std::vector<MVT::SimpleValueType> VTs = CGT.getLegalValueTypes();

    if (EMVT::isExtIntegerInVTs(NodeToApply->getExtTypes())) {
      VTs = FilterVTs(VTs, isInteger);
    } else if (EMVT::isExtFloatingPointInVTs(NodeToApply->getExtTypes())) {
      VTs = FilterVTs(VTs, isFloatingPoint);
    } else {
      VTs.clear();
    }

    switch (VTs.size()) {
    default:         // Too many VT's to pick from.
    case 0: break;   // No info yet.
    case 1: 
      // Only one VT of this flavor.  Cannot ever satisify the constraints.
      return NodeToApply->UpdateNodeType(MVT::Other, TP);  // throw
    case 2:
      // If we have exactly two possible types, the little operand must be the
      // small one, the big operand should be the big one.  Common with 
      // float/double for example.
      assert(VTs[0] < VTs[1] && "Should be sorted!");
      MadeChange |= NodeToApply->UpdateNodeType(VTs[0], TP);
      MadeChange |= BigOperand->UpdateNodeType(VTs[1], TP);
      break;
    }    
    return MadeChange;
  }
  case SDTCisIntVectorOfSameSize: {
    TreePatternNode *OtherOperand =
      getOperandNum(x.SDTCisIntVectorOfSameSize_Info.OtherOperandNum,
                    N, NumResults);
    if (OtherOperand->hasTypeSet()) {
      if (!isVector(OtherOperand->getTypeNum(0)))
        TP.error(N->getOperator()->getName() + " VT operand must be a vector!");
      MVT IVT = OtherOperand->getTypeNum(0);
      unsigned NumElements = IVT.getVectorNumElements();
      IVT = MVT::getIntVectorWithNumElements(NumElements);
      return NodeToApply->UpdateNodeType(IVT.getSimpleVT(), TP);
    }
    return false;
  }
  case SDTCisEltOfVec: {
    TreePatternNode *OtherOperand =
      getOperandNum(x.SDTCisIntVectorOfSameSize_Info.OtherOperandNum,
                    N, NumResults);
    if (OtherOperand->hasTypeSet()) {
      if (!isVector(OtherOperand->getTypeNum(0)))
        TP.error(N->getOperator()->getName() + " VT operand must be a vector!");
      MVT IVT = OtherOperand->getTypeNum(0);
      IVT = IVT.getVectorElementType();
      return NodeToApply->UpdateNodeType(IVT.getSimpleVT(), TP);
    }
    return false;
  }
  }  
  return false;
}

//===----------------------------------------------------------------------===//
// SDNodeInfo implementation
//
SDNodeInfo::SDNodeInfo(Record *R) : Def(R) {
  EnumName    = R->getValueAsString("Opcode");
  SDClassName = R->getValueAsString("SDClass");
  Record *TypeProfile = R->getValueAsDef("TypeProfile");
  NumResults = TypeProfile->getValueAsInt("NumResults");
  NumOperands = TypeProfile->getValueAsInt("NumOperands");
  
  // Parse the properties.
  Properties = 0;
  std::vector<Record*> PropList = R->getValueAsListOfDefs("Properties");
  for (unsigned i = 0, e = PropList.size(); i != e; ++i) {
    if (PropList[i]->getName() == "SDNPCommutative") {
      Properties |= 1 << SDNPCommutative;
    } else if (PropList[i]->getName() == "SDNPAssociative") {
      Properties |= 1 << SDNPAssociative;
    } else if (PropList[i]->getName() == "SDNPHasChain") {
      Properties |= 1 << SDNPHasChain;
    } else if (PropList[i]->getName() == "SDNPOutFlag") {
      Properties |= 1 << SDNPOutFlag;
    } else if (PropList[i]->getName() == "SDNPInFlag") {
      Properties |= 1 << SDNPInFlag;
    } else if (PropList[i]->getName() == "SDNPOptInFlag") {
      Properties |= 1 << SDNPOptInFlag;
    } else if (PropList[i]->getName() == "SDNPMayStore") {
      Properties |= 1 << SDNPMayStore;
    } else if (PropList[i]->getName() == "SDNPMayLoad") {
      Properties |= 1 << SDNPMayLoad;
    } else if (PropList[i]->getName() == "SDNPSideEffect") {
      Properties |= 1 << SDNPSideEffect;
    } else if (PropList[i]->getName() == "SDNPMemOperand") {
      Properties |= 1 << SDNPMemOperand;
    } else {
      cerr << "Unknown SD Node property '" << PropList[i]->getName()
           << "' on node '" << R->getName() << "'!\n";
      exit(1);
    }
  }
  
  
  // Parse the type constraints.
  std::vector<Record*> ConstraintList =
    TypeProfile->getValueAsListOfDefs("Constraints");
  TypeConstraints.assign(ConstraintList.begin(), ConstraintList.end());
}

//===----------------------------------------------------------------------===//
// TreePatternNode implementation
//

TreePatternNode::~TreePatternNode() {
#if 0 // FIXME: implement refcounted tree nodes!
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    delete getChild(i);
#endif
}

/// UpdateNodeType - Set the node type of N to VT if VT contains
/// information.  If N already contains a conflicting type, then throw an
/// exception.  This returns true if any information was updated.
///
bool TreePatternNode::UpdateNodeType(const std::vector<unsigned char> &ExtVTs,
                                     TreePattern &TP) {
  assert(!ExtVTs.empty() && "Cannot update node type with empty type vector!");
  
  if (ExtVTs[0] == EMVT::isUnknown || LHSIsSubsetOfRHS(getExtTypes(), ExtVTs))
    return false;
  if (isTypeCompletelyUnknown() || LHSIsSubsetOfRHS(ExtVTs, getExtTypes())) {
    setTypes(ExtVTs);
    return true;
  }

  if (getExtTypeNum(0) == MVT::iPTR || getExtTypeNum(0) == MVT::iPTRAny) {
    if (ExtVTs[0] == MVT::iPTR || ExtVTs[0] == MVT::iPTRAny ||
        ExtVTs[0] == EMVT::isInt)
      return false;
    if (EMVT::isExtIntegerInVTs(ExtVTs)) {
      std::vector<unsigned char> FVTs = FilterEVTs(ExtVTs, isInteger);
      if (FVTs.size()) {
        setTypes(ExtVTs);
        return true;
      }
    }
  }

  if ((ExtVTs[0] == EMVT::isInt || ExtVTs[0] == MVT::iAny) &&
      EMVT::isExtIntegerInVTs(getExtTypes())) {
    assert(hasTypeSet() && "should be handled above!");
    std::vector<unsigned char> FVTs = FilterEVTs(getExtTypes(), isInteger);
    if (getExtTypes() == FVTs)
      return false;
    setTypes(FVTs);
    return true;
  }
  if ((ExtVTs[0] == MVT::iPTR || ExtVTs[0] == MVT::iPTRAny) &&
      EMVT::isExtIntegerInVTs(getExtTypes())) {
    //assert(hasTypeSet() && "should be handled above!");
    std::vector<unsigned char> FVTs = FilterEVTs(getExtTypes(), isInteger);
    if (getExtTypes() == FVTs)
      return false;
    if (FVTs.size()) {
      setTypes(FVTs);
      return true;
    }
  }      
  if ((ExtVTs[0] == EMVT::isFP || ExtVTs[0] == MVT::fAny) &&
      EMVT::isExtFloatingPointInVTs(getExtTypes())) {
    assert(hasTypeSet() && "should be handled above!");
    std::vector<unsigned char> FVTs =
      FilterEVTs(getExtTypes(), isFloatingPoint);
    if (getExtTypes() == FVTs)
      return false;
    setTypes(FVTs);
    return true;
  }
      
  // If we know this is an int or fp type, and we are told it is a specific one,
  // take the advice.
  //
  // Similarly, we should probably set the type here to the intersection of
  // {isInt|isFP} and ExtVTs
  if (((getExtTypeNum(0) == EMVT::isInt || getExtTypeNum(0) == MVT::iAny) &&
       EMVT::isExtIntegerInVTs(ExtVTs)) ||
      ((getExtTypeNum(0) == EMVT::isFP || getExtTypeNum(0) == MVT::fAny) &&
       EMVT::isExtFloatingPointInVTs(ExtVTs))) {
    setTypes(ExtVTs);
    return true;
  }
  if (getExtTypeNum(0) == EMVT::isInt &&
      (ExtVTs[0] == MVT::iPTR || ExtVTs[0] == MVT::iPTRAny)) {
    setTypes(ExtVTs);
    return true;
  }

  if (isLeaf()) {
    dump();
    cerr << " ";
    TP.error("Type inference contradiction found in node!");
  } else {
    TP.error("Type inference contradiction found in node " + 
             getOperator()->getName() + "!");
  }
  return true; // unreachable
}


void TreePatternNode::print(std::ostream &OS) const {
  if (isLeaf()) {
    OS << *getLeafValue();
  } else {
    OS << "(" << getOperator()->getName();
  }
  
  // FIXME: At some point we should handle printing all the value types for 
  // nodes that are multiply typed.
  switch (getExtTypeNum(0)) {
  case MVT::Other: OS << ":Other"; break;
  case EMVT::isInt: OS << ":isInt"; break;
  case EMVT::isFP : OS << ":isFP"; break;
  case EMVT::isUnknown: ; /*OS << ":?";*/ break;
  case MVT::iPTR:  OS << ":iPTR"; break;
  case MVT::iPTRAny:  OS << ":iPTRAny"; break;
  default: {
    std::string VTName = llvm::getName(getTypeNum(0));
    // Strip off MVT:: prefix if present.
    if (VTName.substr(0,5) == "MVT::")
      VTName = VTName.substr(5);
    OS << ":" << VTName;
    break;
  }
  }

  if (!isLeaf()) {
    if (getNumChildren() != 0) {
      OS << " ";
      getChild(0)->print(OS);
      for (unsigned i = 1, e = getNumChildren(); i != e; ++i) {
        OS << ", ";
        getChild(i)->print(OS);
      }
    }
    OS << ")";
  }
  
  for (unsigned i = 0, e = PredicateFns.size(); i != e; ++i)
    OS << "<<P:" << PredicateFns[i] << ">>";
  if (TransformFn)
    OS << "<<X:" << TransformFn->getName() << ">>";
  if (!getName().empty())
    OS << ":$" << getName();

}
void TreePatternNode::dump() const {
  print(*cerr.stream());
}

/// isIsomorphicTo - Return true if this node is recursively
/// isomorphic to the specified node.  For this comparison, the node's
/// entire state is considered. The assigned name is ignored, since
/// nodes with differing names are considered isomorphic. However, if
/// the assigned name is present in the dependent variable set, then
/// the assigned name is considered significant and the node is
/// isomorphic if the names match.
bool TreePatternNode::isIsomorphicTo(const TreePatternNode *N,
                                     const MultipleUseVarSet &DepVars) const {
  if (N == this) return true;
  if (N->isLeaf() != isLeaf() || getExtTypes() != N->getExtTypes() ||
      getPredicateFns() != N->getPredicateFns() ||
      getTransformFn() != N->getTransformFn())
    return false;

  if (isLeaf()) {
    if (DefInit *DI = dynamic_cast<DefInit*>(getLeafValue())) {
      if (DefInit *NDI = dynamic_cast<DefInit*>(N->getLeafValue())) {
        return ((DI->getDef() == NDI->getDef())
                && (DepVars.find(getName()) == DepVars.end()
                    || getName() == N->getName()));
      }
    }
    return getLeafValue() == N->getLeafValue();
  }
  
  if (N->getOperator() != getOperator() ||
      N->getNumChildren() != getNumChildren()) return false;
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->isIsomorphicTo(N->getChild(i), DepVars))
      return false;
  return true;
}

/// clone - Make a copy of this tree and all of its children.
///
TreePatternNode *TreePatternNode::clone() const {
  TreePatternNode *New;
  if (isLeaf()) {
    New = new TreePatternNode(getLeafValue());
  } else {
    std::vector<TreePatternNode*> CChildren;
    CChildren.reserve(Children.size());
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      CChildren.push_back(getChild(i)->clone());
    New = new TreePatternNode(getOperator(), CChildren);
  }
  New->setName(getName());
  New->setTypes(getExtTypes());
  New->setPredicateFns(getPredicateFns());
  New->setTransformFn(getTransformFn());
  return New;
}

/// SubstituteFormalArguments - Replace the formal arguments in this tree
/// with actual values specified by ArgMap.
void TreePatternNode::
SubstituteFormalArguments(std::map<std::string, TreePatternNode*> &ArgMap) {
  if (isLeaf()) return;
  
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = getChild(i);
    if (Child->isLeaf()) {
      Init *Val = Child->getLeafValue();
      if (dynamic_cast<DefInit*>(Val) &&
          static_cast<DefInit*>(Val)->getDef()->getName() == "node") {
        // We found a use of a formal argument, replace it with its value.
        TreePatternNode *NewChild = ArgMap[Child->getName()];
        assert(NewChild && "Couldn't find formal argument!");
        assert((Child->getPredicateFns().empty() ||
                NewChild->getPredicateFns() == Child->getPredicateFns()) &&
               "Non-empty child predicate clobbered!");
        setChild(i, NewChild);
      }
    } else {
      getChild(i)->SubstituteFormalArguments(ArgMap);
    }
  }
}


/// InlinePatternFragments - If this pattern refers to any pattern
/// fragments, inline them into place, giving us a pattern without any
/// PatFrag references.
TreePatternNode *TreePatternNode::InlinePatternFragments(TreePattern &TP) {
  if (isLeaf()) return this;  // nothing to do.
  Record *Op = getOperator();
  
  if (!Op->isSubClassOf("PatFrag")) {
    // Just recursively inline children nodes.
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
      TreePatternNode *Child = getChild(i);
      TreePatternNode *NewChild = Child->InlinePatternFragments(TP);

      assert((Child->getPredicateFns().empty() ||
              NewChild->getPredicateFns() == Child->getPredicateFns()) &&
             "Non-empty child predicate clobbered!");

      setChild(i, NewChild);
    }
    return this;
  }

  // Otherwise, we found a reference to a fragment.  First, look up its
  // TreePattern record.
  TreePattern *Frag = TP.getDAGPatterns().getPatternFragment(Op);
  
  // Verify that we are passing the right number of operands.
  if (Frag->getNumArgs() != Children.size())
    TP.error("'" + Op->getName() + "' fragment requires " +
             utostr(Frag->getNumArgs()) + " operands!");

  TreePatternNode *FragTree = Frag->getOnlyTree()->clone();

  std::string Code = Op->getValueAsCode("Predicate");
  if (!Code.empty())
    FragTree->addPredicateFn("Predicate_"+Op->getName());

  // Resolve formal arguments to their actual value.
  if (Frag->getNumArgs()) {
    // Compute the map of formal to actual arguments.
    std::map<std::string, TreePatternNode*> ArgMap;
    for (unsigned i = 0, e = Frag->getNumArgs(); i != e; ++i)
      ArgMap[Frag->getArgName(i)] = getChild(i)->InlinePatternFragments(TP);
  
    FragTree->SubstituteFormalArguments(ArgMap);
  }
  
  FragTree->setName(getName());
  FragTree->UpdateNodeType(getExtTypes(), TP);

  // Transfer in the old predicates.
  for (unsigned i = 0, e = getPredicateFns().size(); i != e; ++i)
    FragTree->addPredicateFn(getPredicateFns()[i]);

  // Get a new copy of this fragment to stitch into here.
  //delete this;    // FIXME: implement refcounting!
  
  // The fragment we inlined could have recursive inlining that is needed.  See
  // if there are any pattern fragments in it and inline them as needed.
  return FragTree->InlinePatternFragments(TP);
}

/// getImplicitType - Check to see if the specified record has an implicit
/// type which should be applied to it.  This infer the type of register
/// references from the register file information, for example.
///
static std::vector<unsigned char> getImplicitType(Record *R, bool NotRegisters,
                                      TreePattern &TP) {
  // Some common return values
  std::vector<unsigned char> Unknown(1, EMVT::isUnknown);
  std::vector<unsigned char> Other(1, MVT::Other);

  // Check to see if this is a register or a register class...
  if (R->isSubClassOf("RegisterClass")) {
    if (NotRegisters) 
      return Unknown;
    const CodeGenRegisterClass &RC = 
      TP.getDAGPatterns().getTargetInfo().getRegisterClass(R);
    return ConvertVTs(RC.getValueTypes());
  } else if (R->isSubClassOf("PatFrag")) {
    // Pattern fragment types will be resolved when they are inlined.
    return Unknown;
  } else if (R->isSubClassOf("Register")) {
    if (NotRegisters) 
      return Unknown;
    const CodeGenTarget &T = TP.getDAGPatterns().getTargetInfo();
    return T.getRegisterVTs(R);
  } else if (R->isSubClassOf("ValueType") || R->isSubClassOf("CondCode")) {
    // Using a VTSDNode or CondCodeSDNode.
    return Other;
  } else if (R->isSubClassOf("ComplexPattern")) {
    if (NotRegisters) 
      return Unknown;
    std::vector<unsigned char>
    ComplexPat(1, TP.getDAGPatterns().getComplexPattern(R).getValueType());
    return ComplexPat;
  } else if (R->getName() == "ptr_rc") {
    Other[0] = MVT::iPTR;
    return Other;
  } else if (R->getName() == "node" || R->getName() == "srcvalue" ||
             R->getName() == "zero_reg") {
    // Placeholder.
    return Unknown;
  }
  
  TP.error("Unknown node flavor used in pattern: " + R->getName());
  return Other;
}


/// getIntrinsicInfo - If this node corresponds to an intrinsic, return the
/// CodeGenIntrinsic information for it, otherwise return a null pointer.
const CodeGenIntrinsic *TreePatternNode::
getIntrinsicInfo(const CodeGenDAGPatterns &CDP) const {
  if (getOperator() != CDP.get_intrinsic_void_sdnode() &&
      getOperator() != CDP.get_intrinsic_w_chain_sdnode() &&
      getOperator() != CDP.get_intrinsic_wo_chain_sdnode())
    return 0;
    
  unsigned IID = 
    dynamic_cast<IntInit*>(getChild(0)->getLeafValue())->getValue();
  return &CDP.getIntrinsicInfo(IID);
}

/// isCommutativeIntrinsic - Return true if the node corresponds to a
/// commutative intrinsic.
bool
TreePatternNode::isCommutativeIntrinsic(const CodeGenDAGPatterns &CDP) const {
  if (const CodeGenIntrinsic *Int = getIntrinsicInfo(CDP))
    return Int->isCommutative;
  return false;
}


/// ApplyTypeConstraints - Apply all of the type constraints relevant to
/// this node and its children in the tree.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, throw an
/// exception.
bool TreePatternNode::ApplyTypeConstraints(TreePattern &TP, bool NotRegisters) {
  CodeGenDAGPatterns &CDP = TP.getDAGPatterns();
  if (isLeaf()) {
    if (DefInit *DI = dynamic_cast<DefInit*>(getLeafValue())) {
      // If it's a regclass or something else known, include the type.
      return UpdateNodeType(getImplicitType(DI->getDef(), NotRegisters, TP),TP);
    } else if (IntInit *II = dynamic_cast<IntInit*>(getLeafValue())) {
      // Int inits are always integers. :)
      bool MadeChange = UpdateNodeType(EMVT::isInt, TP);
      
      if (hasTypeSet()) {
        // At some point, it may make sense for this tree pattern to have
        // multiple types.  Assert here that it does not, so we revisit this
        // code when appropriate.
        assert(getExtTypes().size() >= 1 && "TreePattern doesn't have a type!");
        MVT::SimpleValueType VT = getTypeNum(0);
        for (unsigned i = 1, e = getExtTypes().size(); i != e; ++i)
          assert(getTypeNum(i) == VT && "TreePattern has too many types!");
        
        VT = getTypeNum(0);
        if (VT != MVT::iPTR && VT != MVT::iPTRAny) {
          unsigned Size = MVT(VT).getSizeInBits();
          // Make sure that the value is representable for this type.
          if (Size < 32) {
            int Val = (II->getValue() << (32-Size)) >> (32-Size);
            if (Val != II->getValue()) {
              // If sign-extended doesn't fit, does it fit as unsigned?
              unsigned ValueMask;
              unsigned UnsignedVal;
              ValueMask = unsigned(~uint32_t(0UL) >> (32-Size));
              UnsignedVal = unsigned(II->getValue());

              if ((ValueMask & UnsignedVal) != UnsignedVal) {
                TP.error("Integer value '" + itostr(II->getValue())+
                         "' is out of range for type '" + 
                         getEnumName(getTypeNum(0)) + "'!");
              }
            }
         }
       }
      }
      
      return MadeChange;
    }
    return false;
  }
  
  // special handling for set, which isn't really an SDNode.
  if (getOperator()->getName() == "set") {
    assert (getNumChildren() >= 2 && "Missing RHS of a set?");
    unsigned NC = getNumChildren();
    bool MadeChange = false;
    for (unsigned i = 0; i < NC-1; ++i) {
      MadeChange = getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
      MadeChange |= getChild(NC-1)->ApplyTypeConstraints(TP, NotRegisters);
    
      // Types of operands must match.
      MadeChange |= getChild(i)->UpdateNodeType(getChild(NC-1)->getExtTypes(),
                                                TP);
      MadeChange |= getChild(NC-1)->UpdateNodeType(getChild(i)->getExtTypes(),
                                                   TP);
      MadeChange |= UpdateNodeType(MVT::isVoid, TP);
    }
    return MadeChange;
  } else if (getOperator()->getName() == "implicit" ||
             getOperator()->getName() == "parallel") {
    bool MadeChange = false;
    for (unsigned i = 0; i < getNumChildren(); ++i)
      MadeChange = getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    MadeChange |= UpdateNodeType(MVT::isVoid, TP);
    return MadeChange;
  } else if (const CodeGenIntrinsic *Int = getIntrinsicInfo(CDP)) {
    bool MadeChange = false;

    // Apply the result type to the node.
    unsigned NumRetVTs = Int->IS.RetVTs.size();
    unsigned NumParamVTs = Int->IS.ParamVTs.size();

    for (unsigned i = 0, e = NumRetVTs; i != e; ++i)
      MadeChange |= UpdateNodeType(Int->IS.RetVTs[i], TP);

    if (getNumChildren() != NumParamVTs + NumRetVTs)
      TP.error("Intrinsic '" + Int->Name + "' expects " +
               utostr(NumParamVTs + NumRetVTs - 1) + " operands, not " +
               utostr(getNumChildren() - 1) + " operands!");

    // Apply type info to the intrinsic ID.
    MadeChange |= getChild(0)->UpdateNodeType(MVT::iPTR, TP);
    
    for (unsigned i = NumRetVTs, e = getNumChildren(); i != e; ++i) {
      MVT::SimpleValueType OpVT = Int->IS.ParamVTs[i - NumRetVTs];
      MadeChange |= getChild(i)->UpdateNodeType(OpVT, TP);
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    }
    return MadeChange;
  } else if (getOperator()->isSubClassOf("SDNode")) {
    const SDNodeInfo &NI = CDP.getSDNodeInfo(getOperator());
    
    bool MadeChange = NI.ApplyTypeConstraints(this, TP);
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    // Branch, etc. do not produce results and top-level forms in instr pattern
    // must have void types.
    if (NI.getNumResults() == 0)
      MadeChange |= UpdateNodeType(MVT::isVoid, TP);
    
    // If this is a vector_shuffle operation, apply types to the build_vector
    // operation.  The types of the integers don't matter, but this ensures they
    // won't get checked.
    if (getOperator()->getName() == "vector_shuffle" &&
        getChild(2)->getOperator()->getName() == "build_vector") {
      TreePatternNode *BV = getChild(2);
      const std::vector<MVT::SimpleValueType> &LegalVTs
        = CDP.getTargetInfo().getLegalValueTypes();
      MVT::SimpleValueType LegalIntVT = MVT::Other;
      for (unsigned i = 0, e = LegalVTs.size(); i != e; ++i)
        if (isInteger(LegalVTs[i]) && !isVector(LegalVTs[i])) {
          LegalIntVT = LegalVTs[i];
          break;
        }
      assert(LegalIntVT != MVT::Other && "No legal integer VT?");
            
      for (unsigned i = 0, e = BV->getNumChildren(); i != e; ++i)
        MadeChange |= BV->getChild(i)->UpdateNodeType(LegalIntVT, TP);
    }
    return MadeChange;  
  } else if (getOperator()->isSubClassOf("Instruction")) {
    const DAGInstruction &Inst = CDP.getInstruction(getOperator());
    bool MadeChange = false;
    unsigned NumResults = Inst.getNumResults();
    
    assert(NumResults <= 1 &&
           "Only supports zero or one result instrs!");

    CodeGenInstruction &InstInfo =
      CDP.getTargetInfo().getInstruction(getOperator()->getName());
    // Apply the result type to the node
    if (NumResults == 0 || InstInfo.NumDefs == 0) {
      MadeChange = UpdateNodeType(MVT::isVoid, TP);
    } else {
      Record *ResultNode = Inst.getResult(0);
      
      if (ResultNode->getName() == "ptr_rc") {
        std::vector<unsigned char> VT;
        VT.push_back(MVT::iPTR);
        MadeChange = UpdateNodeType(VT, TP);
      } else if (ResultNode->getName() == "unknown") {
        std::vector<unsigned char> VT;
        VT.push_back(EMVT::isUnknown);
        MadeChange = UpdateNodeType(VT, TP);
      } else {
        assert(ResultNode->isSubClassOf("RegisterClass") &&
               "Operands should be register classes!");

        const CodeGenRegisterClass &RC = 
          CDP.getTargetInfo().getRegisterClass(ResultNode);
        MadeChange = UpdateNodeType(ConvertVTs(RC.getValueTypes()), TP);
      }
    }

    unsigned ChildNo = 0;
    for (unsigned i = 0, e = Inst.getNumOperands(); i != e; ++i) {
      Record *OperandNode = Inst.getOperand(i);
      
      // If the instruction expects a predicate or optional def operand, we
      // codegen this by setting the operand to it's default value if it has a
      // non-empty DefaultOps field.
      if ((OperandNode->isSubClassOf("PredicateOperand") ||
           OperandNode->isSubClassOf("OptionalDefOperand")) &&
          !CDP.getDefaultOperand(OperandNode).DefaultOps.empty())
        continue;
       
      // Verify that we didn't run out of provided operands.
      if (ChildNo >= getNumChildren())
        TP.error("Instruction '" + getOperator()->getName() +
                 "' expects more operands than were provided.");
      
      MVT::SimpleValueType VT;
      TreePatternNode *Child = getChild(ChildNo++);
      if (OperandNode->isSubClassOf("RegisterClass")) {
        const CodeGenRegisterClass &RC = 
          CDP.getTargetInfo().getRegisterClass(OperandNode);
        MadeChange |= Child->UpdateNodeType(ConvertVTs(RC.getValueTypes()), TP);
      } else if (OperandNode->isSubClassOf("Operand")) {
        VT = getValueType(OperandNode->getValueAsDef("Type"));
        MadeChange |= Child->UpdateNodeType(VT, TP);
      } else if (OperandNode->getName() == "ptr_rc") {
        MadeChange |= Child->UpdateNodeType(MVT::iPTR, TP);
      } else if (OperandNode->getName() == "unknown") {
        MadeChange |= Child->UpdateNodeType(EMVT::isUnknown, TP);
      } else {
        assert(0 && "Unknown operand type!");
        abort();
      }
      MadeChange |= Child->ApplyTypeConstraints(TP, NotRegisters);
    }

    if (ChildNo != getNumChildren())
      TP.error("Instruction '" + getOperator()->getName() +
               "' was provided too many operands!");
    
    return MadeChange;
  } else {
    assert(getOperator()->isSubClassOf("SDNodeXForm") && "Unknown node type!");
    
    // Node transforms always take one operand.
    if (getNumChildren() != 1)
      TP.error("Node transform '" + getOperator()->getName() +
               "' requires one operand!");

    // If either the output or input of the xform does not have exact
    // type info. We assume they must be the same. Otherwise, it is perfectly
    // legal to transform from one type to a completely different type.
    if (!hasTypeSet() || !getChild(0)->hasTypeSet()) {
      bool MadeChange = UpdateNodeType(getChild(0)->getExtTypes(), TP);
      MadeChange |= getChild(0)->UpdateNodeType(getExtTypes(), TP);
      return MadeChange;
    }
    return false;
  }
}

/// OnlyOnRHSOfCommutative - Return true if this value is only allowed on the
/// RHS of a commutative operation, not the on LHS.
static bool OnlyOnRHSOfCommutative(TreePatternNode *N) {
  if (!N->isLeaf() && N->getOperator()->getName() == "imm")
    return true;
  if (N->isLeaf() && dynamic_cast<IntInit*>(N->getLeafValue()))
    return true;
  return false;
}


/// canPatternMatch - If it is impossible for this pattern to match on this
/// target, fill in Reason and return false.  Otherwise, return true.  This is
/// used as a santity check for .td files (to prevent people from writing stuff
/// that can never possibly work), and to prevent the pattern permuter from
/// generating stuff that is useless.
bool TreePatternNode::canPatternMatch(std::string &Reason, 
                                      const CodeGenDAGPatterns &CDP) {
  if (isLeaf()) return true;

  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->canPatternMatch(Reason, CDP))
      return false;

  // If this is an intrinsic, handle cases that would make it not match.  For
  // example, if an operand is required to be an immediate.
  if (getOperator()->isSubClassOf("Intrinsic")) {
    // TODO:
    return true;
  }
  
  // If this node is a commutative operator, check that the LHS isn't an
  // immediate.
  const SDNodeInfo &NodeInfo = CDP.getSDNodeInfo(getOperator());
  bool isCommIntrinsic = isCommutativeIntrinsic(CDP);
  if (NodeInfo.hasProperty(SDNPCommutative) || isCommIntrinsic) {
    // Scan all of the operands of the node and make sure that only the last one
    // is a constant node, unless the RHS also is.
    if (!OnlyOnRHSOfCommutative(getChild(getNumChildren()-1))) {
      bool Skip = isCommIntrinsic ? 1 : 0; // First operand is intrinsic id.
      for (unsigned i = Skip, e = getNumChildren()-1; i != e; ++i)
        if (OnlyOnRHSOfCommutative(getChild(i))) {
          Reason="Immediate value must be on the RHS of commutative operators!";
          return false;
        }
    }
  }
  
  return true;
}

//===----------------------------------------------------------------------===//
// TreePattern implementation
//

TreePattern::TreePattern(Record *TheRec, ListInit *RawPat, bool isInput,
                         CodeGenDAGPatterns &cdp) : TheRecord(TheRec), CDP(cdp){
   isInputPattern = isInput;
   for (unsigned i = 0, e = RawPat->getSize(); i != e; ++i)
     Trees.push_back(ParseTreePattern((DagInit*)RawPat->getElement(i)));
}

TreePattern::TreePattern(Record *TheRec, DagInit *Pat, bool isInput,
                         CodeGenDAGPatterns &cdp) : TheRecord(TheRec), CDP(cdp){
  isInputPattern = isInput;
  Trees.push_back(ParseTreePattern(Pat));
}

TreePattern::TreePattern(Record *TheRec, TreePatternNode *Pat, bool isInput,
                         CodeGenDAGPatterns &cdp) : TheRecord(TheRec), CDP(cdp){
  isInputPattern = isInput;
  Trees.push_back(Pat);
}



void TreePattern::error(const std::string &Msg) const {
  dump();
  throw "In " + TheRecord->getName() + ": " + Msg;
}

TreePatternNode *TreePattern::ParseTreePattern(DagInit *Dag) {
  DefInit *OpDef = dynamic_cast<DefInit*>(Dag->getOperator());
  if (!OpDef) error("Pattern has unexpected operator type!");
  Record *Operator = OpDef->getDef();
  
  if (Operator->isSubClassOf("ValueType")) {
    // If the operator is a ValueType, then this must be "type cast" of a leaf
    // node.
    if (Dag->getNumArgs() != 1)
      error("Type cast only takes one operand!");
    
    Init *Arg = Dag->getArg(0);
    TreePatternNode *New;
    if (DefInit *DI = dynamic_cast<DefInit*>(Arg)) {
      Record *R = DI->getDef();
      if (R->isSubClassOf("SDNode") || R->isSubClassOf("PatFrag")) {
        Dag->setArg(0, new DagInit(DI,
                                std::vector<std::pair<Init*, std::string> >()));
        return ParseTreePattern(Dag);
      }
      New = new TreePatternNode(DI);
    } else if (DagInit *DI = dynamic_cast<DagInit*>(Arg)) {
      New = ParseTreePattern(DI);
    } else if (IntInit *II = dynamic_cast<IntInit*>(Arg)) {
      New = new TreePatternNode(II);
      if (!Dag->getArgName(0).empty())
        error("Constant int argument should not have a name!");
    } else if (BitsInit *BI = dynamic_cast<BitsInit*>(Arg)) {
      // Turn this into an IntInit.
      Init *II = BI->convertInitializerTo(new IntRecTy());
      if (II == 0 || !dynamic_cast<IntInit*>(II))
        error("Bits value must be constants!");
      
      New = new TreePatternNode(dynamic_cast<IntInit*>(II));
      if (!Dag->getArgName(0).empty())
        error("Constant int argument should not have a name!");
    } else {
      Arg->dump();
      error("Unknown leaf value for tree pattern!");
      return 0;
    }
    
    // Apply the type cast.
    New->UpdateNodeType(getValueType(Operator), *this);
    New->setName(Dag->getArgName(0));
    return New;
  }
  
  // Verify that this is something that makes sense for an operator.
  if (!Operator->isSubClassOf("PatFrag") && !Operator->isSubClassOf("SDNode") &&
      !Operator->isSubClassOf("Instruction") && 
      !Operator->isSubClassOf("SDNodeXForm") &&
      !Operator->isSubClassOf("Intrinsic") &&
      Operator->getName() != "set" &&
      Operator->getName() != "implicit" &&
      Operator->getName() != "parallel")
    error("Unrecognized node '" + Operator->getName() + "'!");
  
  //  Check to see if this is something that is illegal in an input pattern.
  if (isInputPattern && (Operator->isSubClassOf("Instruction") ||
                         Operator->isSubClassOf("SDNodeXForm")))
    error("Cannot use '" + Operator->getName() + "' in an input pattern!");
  
  std::vector<TreePatternNode*> Children;
  
  for (unsigned i = 0, e = Dag->getNumArgs(); i != e; ++i) {
    Init *Arg = Dag->getArg(i);
    if (DagInit *DI = dynamic_cast<DagInit*>(Arg)) {
      Children.push_back(ParseTreePattern(DI));
      if (Children.back()->getName().empty())
        Children.back()->setName(Dag->getArgName(i));
    } else if (DefInit *DefI = dynamic_cast<DefInit*>(Arg)) {
      Record *R = DefI->getDef();
      // Direct reference to a leaf DagNode or PatFrag?  Turn it into a
      // TreePatternNode if its own.
      if (R->isSubClassOf("SDNode") || R->isSubClassOf("PatFrag")) {
        Dag->setArg(i, new DagInit(DefI,
                              std::vector<std::pair<Init*, std::string> >()));
        --i;  // Revisit this node...
      } else {
        TreePatternNode *Node = new TreePatternNode(DefI);
        Node->setName(Dag->getArgName(i));
        Children.push_back(Node);
        
        // Input argument?
        if (R->getName() == "node") {
          if (Dag->getArgName(i).empty())
            error("'node' argument requires a name to match with operand list");
          Args.push_back(Dag->getArgName(i));
        }
      }
    } else if (IntInit *II = dynamic_cast<IntInit*>(Arg)) {
      TreePatternNode *Node = new TreePatternNode(II);
      if (!Dag->getArgName(i).empty())
        error("Constant int argument should not have a name!");
      Children.push_back(Node);
    } else if (BitsInit *BI = dynamic_cast<BitsInit*>(Arg)) {
      // Turn this into an IntInit.
      Init *II = BI->convertInitializerTo(new IntRecTy());
      if (II == 0 || !dynamic_cast<IntInit*>(II))
        error("Bits value must be constants!");
      
      TreePatternNode *Node = new TreePatternNode(dynamic_cast<IntInit*>(II));
      if (!Dag->getArgName(i).empty())
        error("Constant int argument should not have a name!");
      Children.push_back(Node);
    } else {
      cerr << '"';
      Arg->dump();
      cerr << "\": ";
      error("Unknown leaf value for tree pattern!");
    }
  }
  
  // If the operator is an intrinsic, then this is just syntactic sugar for for
  // (intrinsic_* <number>, ..children..).  Pick the right intrinsic node, and 
  // convert the intrinsic name to a number.
  if (Operator->isSubClassOf("Intrinsic")) {
    const CodeGenIntrinsic &Int = getDAGPatterns().getIntrinsic(Operator);
    unsigned IID = getDAGPatterns().getIntrinsicID(Operator)+1;

    // If this intrinsic returns void, it must have side-effects and thus a
    // chain.
    if (Int.IS.RetVTs[0] == MVT::isVoid) {
      Operator = getDAGPatterns().get_intrinsic_void_sdnode();
    } else if (Int.ModRef != CodeGenIntrinsic::NoMem) {
      // Has side-effects, requires chain.
      Operator = getDAGPatterns().get_intrinsic_w_chain_sdnode();
    } else {
      // Otherwise, no chain.
      Operator = getDAGPatterns().get_intrinsic_wo_chain_sdnode();
    }
    
    TreePatternNode *IIDNode = new TreePatternNode(new IntInit(IID));
    Children.insert(Children.begin(), IIDNode);
  }
  
  return new TreePatternNode(Operator, Children);
}

/// InferAllTypes - Infer/propagate as many types throughout the expression
/// patterns as possible.  Return true if all types are infered, false
/// otherwise.  Throw an exception if a type contradiction is found.
bool TreePattern::InferAllTypes() {
  bool MadeChange = true;
  while (MadeChange) {
    MadeChange = false;
    for (unsigned i = 0, e = Trees.size(); i != e; ++i)
      MadeChange |= Trees[i]->ApplyTypeConstraints(*this, false);
  }
  
  bool HasUnresolvedTypes = false;
  for (unsigned i = 0, e = Trees.size(); i != e; ++i)
    HasUnresolvedTypes |= Trees[i]->ContainsUnresolvedType();
  return !HasUnresolvedTypes;
}

void TreePattern::print(std::ostream &OS) const {
  OS << getRecord()->getName();
  if (!Args.empty()) {
    OS << "(" << Args[0];
    for (unsigned i = 1, e = Args.size(); i != e; ++i)
      OS << ", " << Args[i];
    OS << ")";
  }
  OS << ": ";
  
  if (Trees.size() > 1)
    OS << "[\n";
  for (unsigned i = 0, e = Trees.size(); i != e; ++i) {
    OS << "\t";
    Trees[i]->print(OS);
    OS << "\n";
  }

  if (Trees.size() > 1)
    OS << "]\n";
}

void TreePattern::dump() const { print(*cerr.stream()); }

//===----------------------------------------------------------------------===//
// CodeGenDAGPatterns implementation
//

// FIXME: REMOVE OSTREAM ARGUMENT
CodeGenDAGPatterns::CodeGenDAGPatterns(RecordKeeper &R) : Records(R) {
  Intrinsics = LoadIntrinsics(Records);
  ParseNodeInfo();
  ParseNodeTransforms();
  ParseComplexPatterns();
  ParsePatternFragments();
  ParseDefaultOperands();
  ParseInstructions();
  ParsePatterns();
  
  // Generate variants.  For example, commutative patterns can match
  // multiple ways.  Add them to PatternsToMatch as well.
  GenerateVariants();

  // Infer instruction flags.  For example, we can detect loads,
  // stores, and side effects in many cases by examining an
  // instruction's pattern.
  InferInstructionFlags();
}

CodeGenDAGPatterns::~CodeGenDAGPatterns() {
  for (std::map<Record*, TreePattern*>::iterator I = PatternFragments.begin(),
       E = PatternFragments.end(); I != E; ++I)
    delete I->second;
}


Record *CodeGenDAGPatterns::getSDNodeNamed(const std::string &Name) const {
  Record *N = Records.getDef(Name);
  if (!N || !N->isSubClassOf("SDNode")) {
    cerr << "Error getting SDNode '" << Name << "'!\n";
    exit(1);
  }
  return N;
}

// Parse all of the SDNode definitions for the target, populating SDNodes.
void CodeGenDAGPatterns::ParseNodeInfo() {
  std::vector<Record*> Nodes = Records.getAllDerivedDefinitions("SDNode");
  while (!Nodes.empty()) {
    SDNodes.insert(std::make_pair(Nodes.back(), Nodes.back()));
    Nodes.pop_back();
  }

  // Get the buildin intrinsic nodes.
  intrinsic_void_sdnode     = getSDNodeNamed("intrinsic_void");
  intrinsic_w_chain_sdnode  = getSDNodeNamed("intrinsic_w_chain");
  intrinsic_wo_chain_sdnode = getSDNodeNamed("intrinsic_wo_chain");
}

/// ParseNodeTransforms - Parse all SDNodeXForm instances into the SDNodeXForms
/// map, and emit them to the file as functions.
void CodeGenDAGPatterns::ParseNodeTransforms() {
  std::vector<Record*> Xforms = Records.getAllDerivedDefinitions("SDNodeXForm");
  while (!Xforms.empty()) {
    Record *XFormNode = Xforms.back();
    Record *SDNode = XFormNode->getValueAsDef("Opcode");
    std::string Code = XFormNode->getValueAsCode("XFormFunction");
    SDNodeXForms.insert(std::make_pair(XFormNode, NodeXForm(SDNode, Code)));

    Xforms.pop_back();
  }
}

void CodeGenDAGPatterns::ParseComplexPatterns() {
  std::vector<Record*> AMs = Records.getAllDerivedDefinitions("ComplexPattern");
  while (!AMs.empty()) {
    ComplexPatterns.insert(std::make_pair(AMs.back(), AMs.back()));
    AMs.pop_back();
  }
}


/// ParsePatternFragments - Parse all of the PatFrag definitions in the .td
/// file, building up the PatternFragments map.  After we've collected them all,
/// inline fragments together as necessary, so that there are no references left
/// inside a pattern fragment to a pattern fragment.
///
void CodeGenDAGPatterns::ParsePatternFragments() {
  std::vector<Record*> Fragments = Records.getAllDerivedDefinitions("PatFrag");
  
  // First step, parse all of the fragments.
  for (unsigned i = 0, e = Fragments.size(); i != e; ++i) {
    DagInit *Tree = Fragments[i]->getValueAsDag("Fragment");
    TreePattern *P = new TreePattern(Fragments[i], Tree, true, *this);
    PatternFragments[Fragments[i]] = P;
    
    // Validate the argument list, converting it to set, to discard duplicates.
    std::vector<std::string> &Args = P->getArgList();
    std::set<std::string> OperandsSet(Args.begin(), Args.end());
    
    if (OperandsSet.count(""))
      P->error("Cannot have unnamed 'node' values in pattern fragment!");
    
    // Parse the operands list.
    DagInit *OpsList = Fragments[i]->getValueAsDag("Operands");
    DefInit *OpsOp = dynamic_cast<DefInit*>(OpsList->getOperator());
    // Special cases: ops == outs == ins. Different names are used to
    // improve readibility.
    if (!OpsOp ||
        (OpsOp->getDef()->getName() != "ops" &&
         OpsOp->getDef()->getName() != "outs" &&
         OpsOp->getDef()->getName() != "ins"))
      P->error("Operands list should start with '(ops ... '!");
    
    // Copy over the arguments.       
    Args.clear();
    for (unsigned j = 0, e = OpsList->getNumArgs(); j != e; ++j) {
      if (!dynamic_cast<DefInit*>(OpsList->getArg(j)) ||
          static_cast<DefInit*>(OpsList->getArg(j))->
          getDef()->getName() != "node")
        P->error("Operands list should all be 'node' values.");
      if (OpsList->getArgName(j).empty())
        P->error("Operands list should have names for each operand!");
      if (!OperandsSet.count(OpsList->getArgName(j)))
        P->error("'" + OpsList->getArgName(j) +
                 "' does not occur in pattern or was multiply specified!");
      OperandsSet.erase(OpsList->getArgName(j));
      Args.push_back(OpsList->getArgName(j));
    }
    
    if (!OperandsSet.empty())
      P->error("Operands list does not contain an entry for operand '" +
               *OperandsSet.begin() + "'!");

    // If there is a code init for this fragment, keep track of the fact that
    // this fragment uses it.
    std::string Code = Fragments[i]->getValueAsCode("Predicate");
    if (!Code.empty())
      P->getOnlyTree()->addPredicateFn("Predicate_"+Fragments[i]->getName());
    
    // If there is a node transformation corresponding to this, keep track of
    // it.
    Record *Transform = Fragments[i]->getValueAsDef("OperandTransform");
    if (!getSDNodeTransform(Transform).second.empty())    // not noop xform?
      P->getOnlyTree()->setTransformFn(Transform);
  }
  
  // Now that we've parsed all of the tree fragments, do a closure on them so
  // that there are not references to PatFrags left inside of them.
  for (unsigned i = 0, e = Fragments.size(); i != e; ++i) {
    TreePattern *ThePat = PatternFragments[Fragments[i]];
    ThePat->InlinePatternFragments();
        
    // Infer as many types as possible.  Don't worry about it if we don't infer
    // all of them, some may depend on the inputs of the pattern.
    try {
      ThePat->InferAllTypes();
    } catch (...) {
      // If this pattern fragment is not supported by this target (no types can
      // satisfy its constraints), just ignore it.  If the bogus pattern is
      // actually used by instructions, the type consistency error will be
      // reported there.
    }
    
    // If debugging, print out the pattern fragment result.
    DEBUG(ThePat->dump());
  }
}

void CodeGenDAGPatterns::ParseDefaultOperands() {
  std::vector<Record*> DefaultOps[2];
  DefaultOps[0] = Records.getAllDerivedDefinitions("PredicateOperand");
  DefaultOps[1] = Records.getAllDerivedDefinitions("OptionalDefOperand");

  // Find some SDNode.
  assert(!SDNodes.empty() && "No SDNodes parsed?");
  Init *SomeSDNode = new DefInit(SDNodes.begin()->first);
  
  for (unsigned iter = 0; iter != 2; ++iter) {
    for (unsigned i = 0, e = DefaultOps[iter].size(); i != e; ++i) {
      DagInit *DefaultInfo = DefaultOps[iter][i]->getValueAsDag("DefaultOps");
    
      // Clone the DefaultInfo dag node, changing the operator from 'ops' to
      // SomeSDnode so that we can parse this.
      std::vector<std::pair<Init*, std::string> > Ops;
      for (unsigned op = 0, e = DefaultInfo->getNumArgs(); op != e; ++op)
        Ops.push_back(std::make_pair(DefaultInfo->getArg(op),
                                     DefaultInfo->getArgName(op)));
      DagInit *DI = new DagInit(SomeSDNode, Ops);
    
      // Create a TreePattern to parse this.
      TreePattern P(DefaultOps[iter][i], DI, false, *this);
      assert(P.getNumTrees() == 1 && "This ctor can only produce one tree!");

      // Copy the operands over into a DAGDefaultOperand.
      DAGDefaultOperand DefaultOpInfo;
    
      TreePatternNode *T = P.getTree(0);
      for (unsigned op = 0, e = T->getNumChildren(); op != e; ++op) {
        TreePatternNode *TPN = T->getChild(op);
        while (TPN->ApplyTypeConstraints(P, false))
          /* Resolve all types */;
      
        if (TPN->ContainsUnresolvedType()) {
          if (iter == 0)
            throw "Value #" + utostr(i) + " of PredicateOperand '" +
              DefaultOps[iter][i]->getName() + "' doesn't have a concrete type!";
          else
            throw "Value #" + utostr(i) + " of OptionalDefOperand '" +
              DefaultOps[iter][i]->getName() + "' doesn't have a concrete type!";
        }
        DefaultOpInfo.DefaultOps.push_back(TPN);
      }

      // Insert it into the DefaultOperands map so we can find it later.
      DefaultOperands[DefaultOps[iter][i]] = DefaultOpInfo;
    }
  }
}

/// HandleUse - Given "Pat" a leaf in the pattern, check to see if it is an
/// instruction input.  Return true if this is a real use.
static bool HandleUse(TreePattern *I, TreePatternNode *Pat,
                      std::map<std::string, TreePatternNode*> &InstInputs,
                      std::vector<Record*> &InstImpInputs) {
  // No name -> not interesting.
  if (Pat->getName().empty()) {
    if (Pat->isLeaf()) {
      DefInit *DI = dynamic_cast<DefInit*>(Pat->getLeafValue());
      if (DI && DI->getDef()->isSubClassOf("RegisterClass"))
        I->error("Input " + DI->getDef()->getName() + " must be named!");
      else if (DI && DI->getDef()->isSubClassOf("Register")) 
        InstImpInputs.push_back(DI->getDef());
        ;
    }
    return false;
  }

  Record *Rec;
  if (Pat->isLeaf()) {
    DefInit *DI = dynamic_cast<DefInit*>(Pat->getLeafValue());
    if (!DI) I->error("Input $" + Pat->getName() + " must be an identifier!");
    Rec = DI->getDef();
  } else {
    assert(Pat->getNumChildren() == 0 && "can't be a use with children!");
    Rec = Pat->getOperator();
  }

  // SRCVALUE nodes are ignored.
  if (Rec->getName() == "srcvalue")
    return false;

  TreePatternNode *&Slot = InstInputs[Pat->getName()];
  if (!Slot) {
    Slot = Pat;
  } else {
    Record *SlotRec;
    if (Slot->isLeaf()) {
      SlotRec = dynamic_cast<DefInit*>(Slot->getLeafValue())->getDef();
    } else {
      assert(Slot->getNumChildren() == 0 && "can't be a use with children!");
      SlotRec = Slot->getOperator();
    }
    
    // Ensure that the inputs agree if we've already seen this input.
    if (Rec != SlotRec)
      I->error("All $" + Pat->getName() + " inputs must agree with each other");
    if (Slot->getExtTypes() != Pat->getExtTypes())
      I->error("All $" + Pat->getName() + " inputs must agree with each other");
  }
  return true;
}

/// FindPatternInputsAndOutputs - Scan the specified TreePatternNode (which is
/// part of "I", the instruction), computing the set of inputs and outputs of
/// the pattern.  Report errors if we see anything naughty.
void CodeGenDAGPatterns::
FindPatternInputsAndOutputs(TreePattern *I, TreePatternNode *Pat,
                            std::map<std::string, TreePatternNode*> &InstInputs,
                            std::map<std::string, TreePatternNode*>&InstResults,
                            std::vector<Record*> &InstImpInputs,
                            std::vector<Record*> &InstImpResults) {
  if (Pat->isLeaf()) {
    bool isUse = HandleUse(I, Pat, InstInputs, InstImpInputs);
    if (!isUse && Pat->getTransformFn())
      I->error("Cannot specify a transform function for a non-input value!");
    return;
  } else if (Pat->getOperator()->getName() == "implicit") {
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i) {
      TreePatternNode *Dest = Pat->getChild(i);
      if (!Dest->isLeaf())
        I->error("implicitly defined value should be a register!");
    
      DefInit *Val = dynamic_cast<DefInit*>(Dest->getLeafValue());
      if (!Val || !Val->getDef()->isSubClassOf("Register"))
        I->error("implicitly defined value should be a register!");
      InstImpResults.push_back(Val->getDef());
    }
    return;
  } else if (Pat->getOperator()->getName() != "set") {
    // If this is not a set, verify that the children nodes are not void typed,
    // and recurse.
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i) {
      if (Pat->getChild(i)->getExtTypeNum(0) == MVT::isVoid)
        I->error("Cannot have void nodes inside of patterns!");
      FindPatternInputsAndOutputs(I, Pat->getChild(i), InstInputs, InstResults,
                                  InstImpInputs, InstImpResults);
    }
    
    // If this is a non-leaf node with no children, treat it basically as if
    // it were a leaf.  This handles nodes like (imm).
    bool isUse = false;
    if (Pat->getNumChildren() == 0)
      isUse = HandleUse(I, Pat, InstInputs, InstImpInputs);
    
    if (!isUse && Pat->getTransformFn())
      I->error("Cannot specify a transform function for a non-input value!");
    return;
  } 
  
  // Otherwise, this is a set, validate and collect instruction results.
  if (Pat->getNumChildren() == 0)
    I->error("set requires operands!");
  
  if (Pat->getTransformFn())
    I->error("Cannot specify a transform function on a set node!");
  
  // Check the set destinations.
  unsigned NumDests = Pat->getNumChildren()-1;
  for (unsigned i = 0; i != NumDests; ++i) {
    TreePatternNode *Dest = Pat->getChild(i);
    if (!Dest->isLeaf())
      I->error("set destination should be a register!");
    
    DefInit *Val = dynamic_cast<DefInit*>(Dest->getLeafValue());
    if (!Val)
      I->error("set destination should be a register!");

    if (Val->getDef()->isSubClassOf("RegisterClass") ||
        Val->getDef()->getName() == "ptr_rc") {
      if (Dest->getName().empty())
        I->error("set destination must have a name!");
      if (InstResults.count(Dest->getName()))
        I->error("cannot set '" + Dest->getName() +"' multiple times");
      InstResults[Dest->getName()] = Dest;
    } else if (Val->getDef()->isSubClassOf("Register")) {
      InstImpResults.push_back(Val->getDef());
    } else {
      I->error("set destination should be a register!");
    }
  }
    
  // Verify and collect info from the computation.
  FindPatternInputsAndOutputs(I, Pat->getChild(NumDests),
                              InstInputs, InstResults,
                              InstImpInputs, InstImpResults);
}

//===----------------------------------------------------------------------===//
// Instruction Analysis
//===----------------------------------------------------------------------===//

class InstAnalyzer {
  const CodeGenDAGPatterns &CDP;
  bool &mayStore;
  bool &mayLoad;
  bool &HasSideEffects;
public:
  InstAnalyzer(const CodeGenDAGPatterns &cdp,
               bool &maystore, bool &mayload, bool &hse)
    : CDP(cdp), mayStore(maystore), mayLoad(mayload), HasSideEffects(hse){
  }

  /// Analyze - Analyze the specified instruction, returning true if the
  /// instruction had a pattern.
  bool Analyze(Record *InstRecord) {
    const TreePattern *Pattern = CDP.getInstruction(InstRecord).getPattern();
    if (Pattern == 0) {
      HasSideEffects = 1;
      return false;  // No pattern.
    }

    // FIXME: Assume only the first tree is the pattern. The others are clobber
    // nodes.
    AnalyzeNode(Pattern->getTree(0));
    return true;
  }

private:
  void AnalyzeNode(const TreePatternNode *N) {
    if (N->isLeaf()) {
      if (DefInit *DI = dynamic_cast<DefInit*>(N->getLeafValue())) {
        Record *LeafRec = DI->getDef();
        // Handle ComplexPattern leaves.
        if (LeafRec->isSubClassOf("ComplexPattern")) {
          const ComplexPattern &CP = CDP.getComplexPattern(LeafRec);
          if (CP.hasProperty(SDNPMayStore)) mayStore = true;
          if (CP.hasProperty(SDNPMayLoad)) mayLoad = true;
          if (CP.hasProperty(SDNPSideEffect)) HasSideEffects = true;
        }
      }
      return;
    }

    // Analyze children.
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
      AnalyzeNode(N->getChild(i));

    // Ignore set nodes, which are not SDNodes.
    if (N->getOperator()->getName() == "set")
      return;

    // Get information about the SDNode for the operator.
    const SDNodeInfo &OpInfo = CDP.getSDNodeInfo(N->getOperator());

    // Notice properties of the node.
    if (OpInfo.hasProperty(SDNPMayStore)) mayStore = true;
    if (OpInfo.hasProperty(SDNPMayLoad)) mayLoad = true;
    if (OpInfo.hasProperty(SDNPSideEffect)) HasSideEffects = true;

    if (const CodeGenIntrinsic *IntInfo = N->getIntrinsicInfo(CDP)) {
      // If this is an intrinsic, analyze it.
      if (IntInfo->ModRef >= CodeGenIntrinsic::ReadArgMem)
        mayLoad = true;// These may load memory.

      if (IntInfo->ModRef >= CodeGenIntrinsic::WriteArgMem)
        mayStore = true;// Intrinsics that can write to memory are 'mayStore'.

      if (IntInfo->ModRef >= CodeGenIntrinsic::WriteMem)
        // WriteMem intrinsics can have other strange effects.
        HasSideEffects = true;
    }
  }

};

static void InferFromPattern(const CodeGenInstruction &Inst,
                             bool &MayStore, bool &MayLoad,
                             bool &HasSideEffects,
                             const CodeGenDAGPatterns &CDP) {
  MayStore = MayLoad = HasSideEffects = false;

  bool HadPattern =
    InstAnalyzer(CDP, MayStore, MayLoad, HasSideEffects).Analyze(Inst.TheDef);

  // InstAnalyzer only correctly analyzes mayStore/mayLoad so far.
  if (Inst.mayStore) {  // If the .td file explicitly sets mayStore, use it.
    // If we decided that this is a store from the pattern, then the .td file
    // entry is redundant.
    if (MayStore)
      fprintf(stderr,
              "Warning: mayStore flag explicitly set on instruction '%s'"
              " but flag already inferred from pattern.\n",
              Inst.TheDef->getName().c_str());
    MayStore = true;
  }

  if (Inst.mayLoad) {  // If the .td file explicitly sets mayLoad, use it.
    // If we decided that this is a load from the pattern, then the .td file
    // entry is redundant.
    if (MayLoad)
      fprintf(stderr,
              "Warning: mayLoad flag explicitly set on instruction '%s'"
              " but flag already inferred from pattern.\n",
              Inst.TheDef->getName().c_str());
    MayLoad = true;
  }

  if (Inst.neverHasSideEffects) {
    if (HadPattern)
      fprintf(stderr, "Warning: neverHasSideEffects set on instruction '%s' "
              "which already has a pattern\n", Inst.TheDef->getName().c_str());
    HasSideEffects = false;
  }

  if (Inst.hasSideEffects) {
    if (HasSideEffects)
      fprintf(stderr, "Warning: hasSideEffects set on instruction '%s' "
              "which already inferred this.\n", Inst.TheDef->getName().c_str());
    HasSideEffects = true;
  }
}

/// ParseInstructions - Parse all of the instructions, inlining and resolving
/// any fragments involved.  This populates the Instructions list with fully
/// resolved instructions.
void CodeGenDAGPatterns::ParseInstructions() {
  std::vector<Record*> Instrs = Records.getAllDerivedDefinitions("Instruction");
  
  for (unsigned i = 0, e = Instrs.size(); i != e; ++i) {
    ListInit *LI = 0;
    
    if (dynamic_cast<ListInit*>(Instrs[i]->getValueInit("Pattern")))
      LI = Instrs[i]->getValueAsListInit("Pattern");
    
    // If there is no pattern, only collect minimal information about the
    // instruction for its operand list.  We have to assume that there is one
    // result, as we have no detailed info.
    if (!LI || LI->getSize() == 0) {
      std::vector<Record*> Results;
      std::vector<Record*> Operands;
      
      CodeGenInstruction &InstInfo =Target.getInstruction(Instrs[i]->getName());

      if (InstInfo.OperandList.size() != 0) {
        if (InstInfo.NumDefs == 0) {
          // These produce no results
          for (unsigned j = 0, e = InstInfo.OperandList.size(); j < e; ++j)
            Operands.push_back(InstInfo.OperandList[j].Rec);
        } else {
          // Assume the first operand is the result.
          Results.push_back(InstInfo.OperandList[0].Rec);
      
          // The rest are inputs.
          for (unsigned j = 1, e = InstInfo.OperandList.size(); j < e; ++j)
            Operands.push_back(InstInfo.OperandList[j].Rec);
        }
      }
      
      // Create and insert the instruction.
      std::vector<Record*> ImpResults;
      std::vector<Record*> ImpOperands;
      Instructions.insert(std::make_pair(Instrs[i], 
                          DAGInstruction(0, Results, Operands, ImpResults,
                                         ImpOperands)));
      continue;  // no pattern.
    }
    
    // Parse the instruction.
    TreePattern *I = new TreePattern(Instrs[i], LI, true, *this);
    // Inline pattern fragments into it.
    I->InlinePatternFragments();
    
    // Infer as many types as possible.  If we cannot infer all of them, we can
    // never do anything with this instruction pattern: report it to the user.
    if (!I->InferAllTypes())
      I->error("Could not infer all types in pattern!");
    
    // InstInputs - Keep track of all of the inputs of the instruction, along 
    // with the record they are declared as.
    std::map<std::string, TreePatternNode*> InstInputs;
    
    // InstResults - Keep track of all the virtual registers that are 'set'
    // in the instruction, including what reg class they are.
    std::map<std::string, TreePatternNode*> InstResults;

    std::vector<Record*> InstImpInputs;
    std::vector<Record*> InstImpResults;
    
    // Verify that the top-level forms in the instruction are of void type, and
    // fill in the InstResults map.
    for (unsigned j = 0, e = I->getNumTrees(); j != e; ++j) {
      TreePatternNode *Pat = I->getTree(j);
      if (Pat->getExtTypeNum(0) != MVT::isVoid)
        I->error("Top-level forms in instruction pattern should have"
                 " void types");

      // Find inputs and outputs, and verify the structure of the uses/defs.
      FindPatternInputsAndOutputs(I, Pat, InstInputs, InstResults,
                                  InstImpInputs, InstImpResults);
    }

    // Now that we have inputs and outputs of the pattern, inspect the operands
    // list for the instruction.  This determines the order that operands are
    // added to the machine instruction the node corresponds to.
    unsigned NumResults = InstResults.size();

    // Parse the operands list from the (ops) list, validating it.
    assert(I->getArgList().empty() && "Args list should still be empty here!");
    CodeGenInstruction &CGI = Target.getInstruction(Instrs[i]->getName());

    // Check that all of the results occur first in the list.
    std::vector<Record*> Results;
    TreePatternNode *Res0Node = NULL;
    for (unsigned i = 0; i != NumResults; ++i) {
      if (i == CGI.OperandList.size())
        I->error("'" + InstResults.begin()->first +
                 "' set but does not appear in operand list!");
      const std::string &OpName = CGI.OperandList[i].Name;
      
      // Check that it exists in InstResults.
      TreePatternNode *RNode = InstResults[OpName];
      if (RNode == 0)
        I->error("Operand $" + OpName + " does not exist in operand list!");
        
      if (i == 0)
        Res0Node = RNode;
      Record *R = dynamic_cast<DefInit*>(RNode->getLeafValue())->getDef();
      if (R == 0)
        I->error("Operand $" + OpName + " should be a set destination: all "
                 "outputs must occur before inputs in operand list!");
      
      if (CGI.OperandList[i].Rec != R)
        I->error("Operand $" + OpName + " class mismatch!");
      
      // Remember the return type.
      Results.push_back(CGI.OperandList[i].Rec);
      
      // Okay, this one checks out.
      InstResults.erase(OpName);
    }

    // Loop over the inputs next.  Make a copy of InstInputs so we can destroy
    // the copy while we're checking the inputs.
    std::map<std::string, TreePatternNode*> InstInputsCheck(InstInputs);

    std::vector<TreePatternNode*> ResultNodeOperands;
    std::vector<Record*> Operands;
    for (unsigned i = NumResults, e = CGI.OperandList.size(); i != e; ++i) {
      CodeGenInstruction::OperandInfo &Op = CGI.OperandList[i];
      const std::string &OpName = Op.Name;
      if (OpName.empty())
        I->error("Operand #" + utostr(i) + " in operands list has no name!");

      if (!InstInputsCheck.count(OpName)) {
        // If this is an predicate operand or optional def operand with an
        // DefaultOps set filled in, we can ignore this.  When we codegen it,
        // we will do so as always executed.
        if (Op.Rec->isSubClassOf("PredicateOperand") ||
            Op.Rec->isSubClassOf("OptionalDefOperand")) {
          // Does it have a non-empty DefaultOps field?  If so, ignore this
          // operand.
          if (!getDefaultOperand(Op.Rec).DefaultOps.empty())
            continue;
        }
        I->error("Operand $" + OpName +
                 " does not appear in the instruction pattern");
      }
      TreePatternNode *InVal = InstInputsCheck[OpName];
      InstInputsCheck.erase(OpName);   // It occurred, remove from map.
      
      if (InVal->isLeaf() &&
          dynamic_cast<DefInit*>(InVal->getLeafValue())) {
        Record *InRec = static_cast<DefInit*>(InVal->getLeafValue())->getDef();
        if (Op.Rec != InRec && !InRec->isSubClassOf("ComplexPattern"))
          I->error("Operand $" + OpName + "'s register class disagrees"
                   " between the operand and pattern");
      }
      Operands.push_back(Op.Rec);
      
      // Construct the result for the dest-pattern operand list.
      TreePatternNode *OpNode = InVal->clone();
      
      // No predicate is useful on the result.
      OpNode->clearPredicateFns();
      
      // Promote the xform function to be an explicit node if set.
      if (Record *Xform = OpNode->getTransformFn()) {
        OpNode->setTransformFn(0);
        std::vector<TreePatternNode*> Children;
        Children.push_back(OpNode);
        OpNode = new TreePatternNode(Xform, Children);
      }
      
      ResultNodeOperands.push_back(OpNode);
    }
    
    if (!InstInputsCheck.empty())
      I->error("Input operand $" + InstInputsCheck.begin()->first +
               " occurs in pattern but not in operands list!");

    TreePatternNode *ResultPattern =
      new TreePatternNode(I->getRecord(), ResultNodeOperands);
    // Copy fully inferred output node type to instruction result pattern.
    if (NumResults > 0)
      ResultPattern->setTypes(Res0Node->getExtTypes());

    // Create and insert the instruction.
    // FIXME: InstImpResults and InstImpInputs should not be part of
    // DAGInstruction.
    DAGInstruction TheInst(I, Results, Operands, InstImpResults, InstImpInputs);
    Instructions.insert(std::make_pair(I->getRecord(), TheInst));

    // Use a temporary tree pattern to infer all types and make sure that the
    // constructed result is correct.  This depends on the instruction already
    // being inserted into the Instructions map.
    TreePattern Temp(I->getRecord(), ResultPattern, false, *this);
    Temp.InferAllTypes();

    DAGInstruction &TheInsertedInst = Instructions.find(I->getRecord())->second;
    TheInsertedInst.setResultPattern(Temp.getOnlyTree());
    
    DEBUG(I->dump());
  }
   
  // If we can, convert the instructions to be patterns that are matched!
  for (std::map<Record*, DAGInstruction>::iterator II = Instructions.begin(),
       E = Instructions.end(); II != E; ++II) {
    DAGInstruction &TheInst = II->second;
    const TreePattern *I = TheInst.getPattern();
    if (I == 0) continue;  // No pattern.

    // FIXME: Assume only the first tree is the pattern. The others are clobber
    // nodes.
    TreePatternNode *Pattern = I->getTree(0);
    TreePatternNode *SrcPattern;
    if (Pattern->getOperator()->getName() == "set") {
      SrcPattern = Pattern->getChild(Pattern->getNumChildren()-1)->clone();
    } else{
      // Not a set (store or something?)
      SrcPattern = Pattern;
    }
    
    std::string Reason;
    if (!SrcPattern->canPatternMatch(Reason, *this))
      I->error("Instruction can never match: " + Reason);
    
    Record *Instr = II->first;
    TreePatternNode *DstPattern = TheInst.getResultPattern();
    PatternsToMatch.
      push_back(PatternToMatch(Instr->getValueAsListInit("Predicates"),
                               SrcPattern, DstPattern, TheInst.getImpResults(),
                               Instr->getValueAsInt("AddedComplexity")));
  }
}


void CodeGenDAGPatterns::InferInstructionFlags() {
  std::map<std::string, CodeGenInstruction> &InstrDescs =
    Target.getInstructions();
  for (std::map<std::string, CodeGenInstruction>::iterator
         II = InstrDescs.begin(), E = InstrDescs.end(); II != E; ++II) {
    CodeGenInstruction &InstInfo = II->second;
    // Determine properties of the instruction from its pattern.
    bool MayStore, MayLoad, HasSideEffects;
    InferFromPattern(InstInfo, MayStore, MayLoad, HasSideEffects, *this);
    InstInfo.mayStore = MayStore;
    InstInfo.mayLoad = MayLoad;
    InstInfo.hasSideEffects = HasSideEffects;
  }
}

void CodeGenDAGPatterns::ParsePatterns() {
  std::vector<Record*> Patterns = Records.getAllDerivedDefinitions("Pattern");

  for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
    DagInit *Tree = Patterns[i]->getValueAsDag("PatternToMatch");
    DefInit *OpDef = dynamic_cast<DefInit*>(Tree->getOperator());
    Record *Operator = OpDef->getDef();
    TreePattern *Pattern;
    if (Operator->getName() != "parallel")
      Pattern = new TreePattern(Patterns[i], Tree, true, *this);
    else {
      std::vector<Init*> Values;
      for (unsigned j = 0, ee = Tree->getNumArgs(); j != ee; ++j)
        Values.push_back(Tree->getArg(j));
      ListInit *LI = new ListInit(Values);
      Pattern = new TreePattern(Patterns[i], LI, true, *this);
    }

    // Inline pattern fragments into it.
    Pattern->InlinePatternFragments();
    
    ListInit *LI = Patterns[i]->getValueAsListInit("ResultInstrs");
    if (LI->getSize() == 0) continue;  // no pattern.
    
    // Parse the instruction.
    TreePattern *Result = new TreePattern(Patterns[i], LI, false, *this);
    
    // Inline pattern fragments into it.
    Result->InlinePatternFragments();

    if (Result->getNumTrees() != 1)
      Result->error("Cannot handle instructions producing instructions "
                    "with temporaries yet!");
    
    bool IterateInference;
    bool InferredAllPatternTypes, InferredAllResultTypes;
    do {
      // Infer as many types as possible.  If we cannot infer all of them, we
      // can never do anything with this pattern: report it to the user.
      InferredAllPatternTypes = Pattern->InferAllTypes();
      
      // Infer as many types as possible.  If we cannot infer all of them, we
      // can never do anything with this pattern: report it to the user.
      InferredAllResultTypes = Result->InferAllTypes();

      // Apply the type of the result to the source pattern.  This helps us
      // resolve cases where the input type is known to be a pointer type (which
      // is considered resolved), but the result knows it needs to be 32- or
      // 64-bits.  Infer the other way for good measure.
      IterateInference = Pattern->getTree(0)->
        UpdateNodeType(Result->getTree(0)->getExtTypes(), *Result);
      IterateInference |= Result->getTree(0)->
        UpdateNodeType(Pattern->getTree(0)->getExtTypes(), *Result);
    } while (IterateInference);

    // Verify that we inferred enough types that we can do something with the
    // pattern and result.  If these fire the user has to add type casts.
    if (!InferredAllPatternTypes)
      Pattern->error("Could not infer all types in pattern!");
    if (!InferredAllResultTypes)
      Result->error("Could not infer all types in pattern result!");
    
    // Validate that the input pattern is correct.
    std::map<std::string, TreePatternNode*> InstInputs;
    std::map<std::string, TreePatternNode*> InstResults;
    std::vector<Record*> InstImpInputs;
    std::vector<Record*> InstImpResults;
    for (unsigned j = 0, ee = Pattern->getNumTrees(); j != ee; ++j)
      FindPatternInputsAndOutputs(Pattern, Pattern->getTree(j),
                                  InstInputs, InstResults,
                                  InstImpInputs, InstImpResults);

    // Promote the xform function to be an explicit node if set.
    TreePatternNode *DstPattern = Result->getOnlyTree();
    std::vector<TreePatternNode*> ResultNodeOperands;
    for (unsigned ii = 0, ee = DstPattern->getNumChildren(); ii != ee; ++ii) {
      TreePatternNode *OpNode = DstPattern->getChild(ii);
      if (Record *Xform = OpNode->getTransformFn()) {
        OpNode->setTransformFn(0);
        std::vector<TreePatternNode*> Children;
        Children.push_back(OpNode);
        OpNode = new TreePatternNode(Xform, Children);
      }
      ResultNodeOperands.push_back(OpNode);
    }
    DstPattern = Result->getOnlyTree();
    if (!DstPattern->isLeaf())
      DstPattern = new TreePatternNode(DstPattern->getOperator(),
                                       ResultNodeOperands);
    DstPattern->setTypes(Result->getOnlyTree()->getExtTypes());
    TreePattern Temp(Result->getRecord(), DstPattern, false, *this);
    Temp.InferAllTypes();

    std::string Reason;
    if (!Pattern->getTree(0)->canPatternMatch(Reason, *this))
      Pattern->error("Pattern can never match: " + Reason);
    
    PatternsToMatch.
      push_back(PatternToMatch(Patterns[i]->getValueAsListInit("Predicates"),
                               Pattern->getTree(0),
                               Temp.getOnlyTree(), InstImpResults,
                               Patterns[i]->getValueAsInt("AddedComplexity")));
  }
}

/// CombineChildVariants - Given a bunch of permutations of each child of the
/// 'operator' node, put them together in all possible ways.
static void CombineChildVariants(TreePatternNode *Orig, 
               const std::vector<std::vector<TreePatternNode*> > &ChildVariants,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 CodeGenDAGPatterns &CDP,
                                 const MultipleUseVarSet &DepVars) {
  // Make sure that each operand has at least one variant to choose from.
  for (unsigned i = 0, e = ChildVariants.size(); i != e; ++i)
    if (ChildVariants[i].empty())
      return;
        
  // The end result is an all-pairs construction of the resultant pattern.
  std::vector<unsigned> Idxs;
  Idxs.resize(ChildVariants.size());
  bool NotDone;
  do {
#ifndef NDEBUG
    if (DebugFlag && !Idxs.empty()) {
      cerr << Orig->getOperator()->getName() << ": Idxs = [ ";
        for (unsigned i = 0; i < Idxs.size(); ++i) {
          cerr << Idxs[i] << " ";
      }
      cerr << "]\n";
    }
#endif
    // Create the variant and add it to the output list.
    std::vector<TreePatternNode*> NewChildren;
    for (unsigned i = 0, e = ChildVariants.size(); i != e; ++i)
      NewChildren.push_back(ChildVariants[i][Idxs[i]]);
    TreePatternNode *R = new TreePatternNode(Orig->getOperator(), NewChildren);
    
    // Copy over properties.
    R->setName(Orig->getName());
    R->setPredicateFns(Orig->getPredicateFns());
    R->setTransformFn(Orig->getTransformFn());
    R->setTypes(Orig->getExtTypes());
    
    // If this pattern cannot match, do not include it as a variant.
    std::string ErrString;
    if (!R->canPatternMatch(ErrString, CDP)) {
      delete R;
    } else {
      bool AlreadyExists = false;
      
      // Scan to see if this pattern has already been emitted.  We can get
      // duplication due to things like commuting:
      //   (and GPRC:$a, GPRC:$b) -> (and GPRC:$b, GPRC:$a)
      // which are the same pattern.  Ignore the dups.
      for (unsigned i = 0, e = OutVariants.size(); i != e; ++i)
        if (R->isIsomorphicTo(OutVariants[i], DepVars)) {
          AlreadyExists = true;
          break;
        }
      
      if (AlreadyExists)
        delete R;
      else
        OutVariants.push_back(R);
    }
    
    // Increment indices to the next permutation by incrementing the
    // indicies from last index backward, e.g., generate the sequence
    // [0, 0], [0, 1], [1, 0], [1, 1].
    int IdxsIdx;
    for (IdxsIdx = Idxs.size() - 1; IdxsIdx >= 0; --IdxsIdx) {
      if (++Idxs[IdxsIdx] == ChildVariants[IdxsIdx].size())
        Idxs[IdxsIdx] = 0;
      else
        break;
    }
    NotDone = (IdxsIdx >= 0);
  } while (NotDone);
}

/// CombineChildVariants - A helper function for binary operators.
///
static void CombineChildVariants(TreePatternNode *Orig, 
                                 const std::vector<TreePatternNode*> &LHS,
                                 const std::vector<TreePatternNode*> &RHS,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 CodeGenDAGPatterns &CDP,
                                 const MultipleUseVarSet &DepVars) {
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.push_back(LHS);
  ChildVariants.push_back(RHS);
  CombineChildVariants(Orig, ChildVariants, OutVariants, CDP, DepVars);
}  


static void GatherChildrenOfAssociativeOpcode(TreePatternNode *N,
                                     std::vector<TreePatternNode *> &Children) {
  assert(N->getNumChildren()==2 &&"Associative but doesn't have 2 children!");
  Record *Operator = N->getOperator();
  
  // Only permit raw nodes.
  if (!N->getName().empty() || !N->getPredicateFns().empty() ||
      N->getTransformFn()) {
    Children.push_back(N);
    return;
  }

  if (N->getChild(0)->isLeaf() || N->getChild(0)->getOperator() != Operator)
    Children.push_back(N->getChild(0));
  else
    GatherChildrenOfAssociativeOpcode(N->getChild(0), Children);

  if (N->getChild(1)->isLeaf() || N->getChild(1)->getOperator() != Operator)
    Children.push_back(N->getChild(1));
  else
    GatherChildrenOfAssociativeOpcode(N->getChild(1), Children);
}

/// GenerateVariantsOf - Given a pattern N, generate all permutations we can of
/// the (potentially recursive) pattern by using algebraic laws.
///
static void GenerateVariantsOf(TreePatternNode *N,
                               std::vector<TreePatternNode*> &OutVariants,
                               CodeGenDAGPatterns &CDP,
                               const MultipleUseVarSet &DepVars) {
  // We cannot permute leaves.
  if (N->isLeaf()) {
    OutVariants.push_back(N);
    return;
  }

  // Look up interesting info about the node.
  const SDNodeInfo &NodeInfo = CDP.getSDNodeInfo(N->getOperator());

  // If this node is associative, reassociate.
  if (NodeInfo.hasProperty(SDNPAssociative)) {
    // Reassociate by pulling together all of the linked operators 
    std::vector<TreePatternNode*> MaximalChildren;
    GatherChildrenOfAssociativeOpcode(N, MaximalChildren);

    // Only handle child sizes of 3.  Otherwise we'll end up trying too many
    // permutations.
    if (MaximalChildren.size() == 3) {
      // Find the variants of all of our maximal children.
      std::vector<TreePatternNode*> AVariants, BVariants, CVariants;
      GenerateVariantsOf(MaximalChildren[0], AVariants, CDP, DepVars);
      GenerateVariantsOf(MaximalChildren[1], BVariants, CDP, DepVars);
      GenerateVariantsOf(MaximalChildren[2], CVariants, CDP, DepVars);
      
      // There are only two ways we can permute the tree:
      //   (A op B) op C    and    A op (B op C)
      // Within these forms, we can also permute A/B/C.
      
      // Generate legal pair permutations of A/B/C.
      std::vector<TreePatternNode*> ABVariants;
      std::vector<TreePatternNode*> BAVariants;
      std::vector<TreePatternNode*> ACVariants;
      std::vector<TreePatternNode*> CAVariants;
      std::vector<TreePatternNode*> BCVariants;
      std::vector<TreePatternNode*> CBVariants;
      CombineChildVariants(N, AVariants, BVariants, ABVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, AVariants, BAVariants, CDP, DepVars);
      CombineChildVariants(N, AVariants, CVariants, ACVariants, CDP, DepVars);
      CombineChildVariants(N, CVariants, AVariants, CAVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, CVariants, BCVariants, CDP, DepVars);
      CombineChildVariants(N, CVariants, BVariants, CBVariants, CDP, DepVars);

      // Combine those into the result: (x op x) op x
      CombineChildVariants(N, ABVariants, CVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BAVariants, CVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, ACVariants, BVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, CAVariants, BVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BCVariants, AVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, CBVariants, AVariants, OutVariants, CDP, DepVars);

      // Combine those into the result: x op (x op x)
      CombineChildVariants(N, CVariants, ABVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, CVariants, BAVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, ACVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, CAVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, AVariants, BCVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, AVariants, CBVariants, OutVariants, CDP, DepVars);
      return;
    }
  }
  
  // Compute permutations of all children.
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.resize(N->getNumChildren());
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
    GenerateVariantsOf(N->getChild(i), ChildVariants[i], CDP, DepVars);

  // Build all permutations based on how the children were formed.
  CombineChildVariants(N, ChildVariants, OutVariants, CDP, DepVars);

  // If this node is commutative, consider the commuted order.
  bool isCommIntrinsic = N->isCommutativeIntrinsic(CDP);
  if (NodeInfo.hasProperty(SDNPCommutative) || isCommIntrinsic) {
    assert((N->getNumChildren()==2 || isCommIntrinsic) &&
           "Commutative but doesn't have 2 children!");
    // Don't count children which are actually register references.
    unsigned NC = 0;
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
      TreePatternNode *Child = N->getChild(i);
      if (Child->isLeaf())
        if (DefInit *DI = dynamic_cast<DefInit*>(Child->getLeafValue())) {
          Record *RR = DI->getDef();
          if (RR->isSubClassOf("Register"))
            continue;
        }
      NC++;
    }
    // Consider the commuted order.
    if (isCommIntrinsic) {
      // Commutative intrinsic. First operand is the intrinsic id, 2nd and 3rd
      // operands are the commutative operands, and there might be more operands
      // after those.
      assert(NC >= 3 &&
             "Commutative intrinsic should have at least 3 childrean!");
      std::vector<std::vector<TreePatternNode*> > Variants;
      Variants.push_back(ChildVariants[0]); // Intrinsic id.
      Variants.push_back(ChildVariants[2]);
      Variants.push_back(ChildVariants[1]);
      for (unsigned i = 3; i != NC; ++i)
        Variants.push_back(ChildVariants[i]);
      CombineChildVariants(N, Variants, OutVariants, CDP, DepVars);
    } else if (NC == 2)
      CombineChildVariants(N, ChildVariants[1], ChildVariants[0],
                           OutVariants, CDP, DepVars);
  }
}


// GenerateVariants - Generate variants.  For example, commutative patterns can
// match multiple ways.  Add them to PatternsToMatch as well.
void CodeGenDAGPatterns::GenerateVariants() {
  DOUT << "Generating instruction variants.\n";
  
  // Loop over all of the patterns we've collected, checking to see if we can
  // generate variants of the instruction, through the exploitation of
  // identities.  This permits the target to provide agressive matching without
  // the .td file having to contain tons of variants of instructions.
  //
  // Note that this loop adds new patterns to the PatternsToMatch list, but we
  // intentionally do not reconsider these.  Any variants of added patterns have
  // already been added.
  //
  for (unsigned i = 0, e = PatternsToMatch.size(); i != e; ++i) {
    MultipleUseVarSet             DepVars;
    std::vector<TreePatternNode*> Variants;
    FindDepVars(PatternsToMatch[i].getSrcPattern(), DepVars);
    DOUT << "Dependent/multiply used variables: ";
    DEBUG(DumpDepVars(DepVars));
    DOUT << "\n";
    GenerateVariantsOf(PatternsToMatch[i].getSrcPattern(), Variants, *this, DepVars);

    assert(!Variants.empty() && "Must create at least original variant!");
    Variants.erase(Variants.begin());  // Remove the original pattern.

    if (Variants.empty())  // No variants for this pattern.
      continue;

    DOUT << "FOUND VARIANTS OF: ";
    DEBUG(PatternsToMatch[i].getSrcPattern()->dump());
    DOUT << "\n";

    for (unsigned v = 0, e = Variants.size(); v != e; ++v) {
      TreePatternNode *Variant = Variants[v];

      DOUT << "  VAR#" << v <<  ": ";
      DEBUG(Variant->dump());
      DOUT << "\n";
      
      // Scan to see if an instruction or explicit pattern already matches this.
      bool AlreadyExists = false;
      for (unsigned p = 0, e = PatternsToMatch.size(); p != e; ++p) {
        // Check to see if this variant already exists.
        if (Variant->isIsomorphicTo(PatternsToMatch[p].getSrcPattern(), DepVars)) {
          DOUT << "  *** ALREADY EXISTS, ignoring variant.\n";
          AlreadyExists = true;
          break;
        }
      }
      // If we already have it, ignore the variant.
      if (AlreadyExists) continue;

      // Otherwise, add it to the list of patterns we have.
      PatternsToMatch.
        push_back(PatternToMatch(PatternsToMatch[i].getPredicates(),
                                 Variant, PatternsToMatch[i].getDstPattern(),
                                 PatternsToMatch[i].getDstRegs(),
                                 PatternsToMatch[i].getAddedComplexity()));
    }

    DOUT << "\n";
  }
}

