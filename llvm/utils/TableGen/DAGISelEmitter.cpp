//===- DAGISelEmitter.cpp - Generate an instruction selector --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a DAG instruction selector.
//
//===----------------------------------------------------------------------===//

#include "DAGISelEmitter.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Helpers for working with extended types.

/// FilterVTs - Filter a list of VT's according to a predicate.
///
template<typename T>
static std::vector<MVT::ValueType> 
FilterVTs(const std::vector<MVT::ValueType> &InVTs, T Filter) {
  std::vector<MVT::ValueType> Result;
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
    if (Filter((MVT::ValueType)InVTs[i]))
      Result.push_back(InVTs[i]);
  return Result;
}

static std::vector<unsigned char>
ConvertVTs(const std::vector<MVT::ValueType> &InVTs) {
  std::vector<unsigned char> Result;
  for (unsigned i = 0, e = InVTs.size(); i != e; ++i)
      Result.push_back(InVTs[i]);
  return Result;
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
static bool isExtIntegerInVTs(const std::vector<unsigned char> &EVTs) {
  assert(!EVTs.empty() && "Cannot check for integer in empty ExtVT list!");
  return EVTs[0] == MVT::isInt || !(FilterEVTs(EVTs, MVT::isInteger).empty());
}

/// isExtFloatingPointVT - Return true if the specified extended value type 
/// vector contains isFP or a FP value type.
static bool isExtFloatingPointInVTs(const std::vector<unsigned char> &EVTs) {
  assert(!EVTs.empty() && "Cannot check for integer in empty ExtVT list!");
  return EVTs[0] == MVT::isFP ||
         !(FilterEVTs(EVTs, MVT::isFloatingPoint).empty());
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
  } else {
    std::cerr << "Unrecognized SDTypeConstraint '" << R->getName() << "'!\n";
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
    std::cerr << "Invalid operand number " << OpNo << " ";
    N->dump();
    std::cerr << '\n';
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

  const CodeGenTarget &CGT = TP.getDAGISelEmitter().getTargetInfo();
  
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
    std::vector<MVT::ValueType> IntVTs =
      FilterVTs(CGT.getLegalValueTypes(), MVT::isInteger);

    // If we found exactly one supported integer type, apply it.
    if (IntVTs.size() == 1)
      return NodeToApply->UpdateNodeType(IntVTs[0], TP);
    return NodeToApply->UpdateNodeType(MVT::isInt, TP);
  }
  case SDTCisFP: {
    // If there is only one FP type supported, this must be it.
    std::vector<MVT::ValueType> FPVTs =
      FilterVTs(CGT.getLegalValueTypes(), MVT::isFloatingPoint);
        
    // If we found exactly one supported FP type, apply it.
    if (FPVTs.size() == 1)
      return NodeToApply->UpdateNodeType(FPVTs[0], TP);
    return NodeToApply->UpdateNodeType(MVT::isFP, TP);
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
    MVT::ValueType VT =
     getValueType(static_cast<DefInit*>(NodeToApply->getLeafValue())->getDef());
    if (!MVT::isInteger(VT))
      TP.error(N->getOperator()->getName() + " VT operand must be integer!");
    
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisVTSmallerThanOp_Info.OtherOperandNum, N,NumResults);
    
    // It must be integer.
    bool MadeChange = false;
    MadeChange |= OtherNode->UpdateNodeType(MVT::isInt, TP);
    
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
    assert(!(isExtIntegerInVTs(NodeToApply->getExtTypes()) &&
             isExtFloatingPointInVTs(NodeToApply->getExtTypes())) &&
           !(isExtIntegerInVTs(BigOperand->getExtTypes()) &&
             isExtFloatingPointInVTs(BigOperand->getExtTypes())) &&
           "SDTCisOpSmallerThanOp does not handle mixed int/fp types!");
    if (isExtIntegerInVTs(NodeToApply->getExtTypes()))
      MadeChange |= BigOperand->UpdateNodeType(MVT::isInt, TP);
    else if (isExtFloatingPointInVTs(NodeToApply->getExtTypes()))
      MadeChange |= BigOperand->UpdateNodeType(MVT::isFP, TP);
    if (isExtIntegerInVTs(BigOperand->getExtTypes()))
      MadeChange |= NodeToApply->UpdateNodeType(MVT::isInt, TP);
    else if (isExtFloatingPointInVTs(BigOperand->getExtTypes()))
      MadeChange |= NodeToApply->UpdateNodeType(MVT::isFP, TP);

    std::vector<MVT::ValueType> VTs = CGT.getLegalValueTypes();
    
    if (isExtIntegerInVTs(NodeToApply->getExtTypes())) {
      VTs = FilterVTs(VTs, MVT::isInteger);
    } else if (isExtFloatingPointInVTs(NodeToApply->getExtTypes())) {
      VTs = FilterVTs(VTs, MVT::isFloatingPoint);
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
      if (!MVT::isVector(OtherOperand->getTypeNum(0)))
        TP.error(N->getOperator()->getName() + " VT operand must be a vector!");
      MVT::ValueType IVT = OtherOperand->getTypeNum(0);
      IVT = MVT::getIntVectorWithNumElements(MVT::getVectorNumElements(IVT));
      return NodeToApply->UpdateNodeType(IVT, TP);
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
    } else {
      std::cerr << "Unknown SD Node property '" << PropList[i]->getName()
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
  
  if (ExtVTs[0] == MVT::isUnknown || LHSIsSubsetOfRHS(getExtTypes(), ExtVTs)) 
    return false;
  if (isTypeCompletelyUnknown() || LHSIsSubsetOfRHS(ExtVTs, getExtTypes())) {
    setTypes(ExtVTs);
    return true;
  }

  if (getExtTypeNum(0) == MVT::iPTR) {
    if (ExtVTs[0] == MVT::iPTR || ExtVTs[0] == MVT::isInt)
      return false;
    if (isExtIntegerInVTs(ExtVTs)) {
      std::vector<unsigned char> FVTs = FilterEVTs(ExtVTs, MVT::isInteger);
      if (FVTs.size()) {
        setTypes(ExtVTs);
        return true;
      }
    }
  }
  
  if (ExtVTs[0] == MVT::isInt && isExtIntegerInVTs(getExtTypes())) {
    assert(hasTypeSet() && "should be handled above!");
    std::vector<unsigned char> FVTs = FilterEVTs(getExtTypes(), MVT::isInteger);
    if (getExtTypes() == FVTs)
      return false;
    setTypes(FVTs);
    return true;
  }
  if (ExtVTs[0] == MVT::iPTR && isExtIntegerInVTs(getExtTypes())) {
    //assert(hasTypeSet() && "should be handled above!");
    std::vector<unsigned char> FVTs = FilterEVTs(getExtTypes(), MVT::isInteger);
    if (getExtTypes() == FVTs)
      return false;
    if (FVTs.size()) {
      setTypes(FVTs);
      return true;
    }
  }      
  if (ExtVTs[0] == MVT::isFP  && isExtFloatingPointInVTs(getExtTypes())) {
    assert(hasTypeSet() && "should be handled above!");
    std::vector<unsigned char> FVTs =
      FilterEVTs(getExtTypes(), MVT::isFloatingPoint);
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
  if ((getExtTypeNum(0) == MVT::isInt && isExtIntegerInVTs(ExtVTs)) ||
      (getExtTypeNum(0) == MVT::isFP  && isExtFloatingPointInVTs(ExtVTs))) {
    setTypes(ExtVTs);
    return true;
  }
  if (getExtTypeNum(0) == MVT::isInt && ExtVTs[0] == MVT::iPTR) {
    setTypes(ExtVTs);
    return true;
  }

  if (isLeaf()) {
    dump();
    std::cerr << " ";
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
  case MVT::isInt: OS << ":isInt"; break;
  case MVT::isFP : OS << ":isFP"; break;
  case MVT::isUnknown: ; /*OS << ":?";*/ break;
  case MVT::iPTR:  OS << ":iPTR"; break;
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
  
  if (!PredicateFn.empty())
    OS << "<<P:" << PredicateFn << ">>";
  if (TransformFn)
    OS << "<<X:" << TransformFn->getName() << ">>";
  if (!getName().empty())
    OS << ":$" << getName();

}
void TreePatternNode::dump() const {
  print(std::cerr);
}

/// isIsomorphicTo - Return true if this node is recursively isomorphic to
/// the specified node.  For this comparison, all of the state of the node
/// is considered, except for the assigned name.  Nodes with differing names
/// that are otherwise identical are considered isomorphic.
bool TreePatternNode::isIsomorphicTo(const TreePatternNode *N) const {
  if (N == this) return true;
  if (N->isLeaf() != isLeaf() || getExtTypes() != N->getExtTypes() ||
      getPredicateFn() != N->getPredicateFn() ||
      getTransformFn() != N->getTransformFn())
    return false;

  if (isLeaf()) {
    if (DefInit *DI = dynamic_cast<DefInit*>(getLeafValue()))
      if (DefInit *NDI = dynamic_cast<DefInit*>(N->getLeafValue()))
        return DI->getDef() == NDI->getDef();
    return getLeafValue() == N->getLeafValue();
  }
  
  if (N->getOperator() != getOperator() ||
      N->getNumChildren() != getNumChildren()) return false;
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->isIsomorphicTo(N->getChild(i)))
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
  New->setPredicateFn(getPredicateFn());
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
        Child = ArgMap[Child->getName()];
        assert(Child && "Couldn't find formal argument!");
        setChild(i, Child);
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
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      setChild(i, getChild(i)->InlinePatternFragments(TP));
    return this;
  }

  // Otherwise, we found a reference to a fragment.  First, look up its
  // TreePattern record.
  TreePattern *Frag = TP.getDAGISelEmitter().getPatternFragment(Op);
  
  // Verify that we are passing the right number of operands.
  if (Frag->getNumArgs() != Children.size())
    TP.error("'" + Op->getName() + "' fragment requires " +
             utostr(Frag->getNumArgs()) + " operands!");

  TreePatternNode *FragTree = Frag->getOnlyTree()->clone();

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
  
  // Get a new copy of this fragment to stitch into here.
  //delete this;    // FIXME: implement refcounting!
  return FragTree;
}

/// getImplicitType - Check to see if the specified record has an implicit
/// type which should be applied to it.  This infer the type of register
/// references from the register file information, for example.
///
static std::vector<unsigned char> getImplicitType(Record *R, bool NotRegisters,
                                      TreePattern &TP) {
  // Some common return values
  std::vector<unsigned char> Unknown(1, MVT::isUnknown);
  std::vector<unsigned char> Other(1, MVT::Other);

  // Check to see if this is a register or a register class...
  if (R->isSubClassOf("RegisterClass")) {
    if (NotRegisters) 
      return Unknown;
    const CodeGenRegisterClass &RC = 
      TP.getDAGISelEmitter().getTargetInfo().getRegisterClass(R);
    return ConvertVTs(RC.getValueTypes());
  } else if (R->isSubClassOf("PatFrag")) {
    // Pattern fragment types will be resolved when they are inlined.
    return Unknown;
  } else if (R->isSubClassOf("Register")) {
    if (NotRegisters) 
      return Unknown;
    const CodeGenTarget &T = TP.getDAGISelEmitter().getTargetInfo();
    return T.getRegisterVTs(R);
  } else if (R->isSubClassOf("ValueType") || R->isSubClassOf("CondCode")) {
    // Using a VTSDNode or CondCodeSDNode.
    return Other;
  } else if (R->isSubClassOf("ComplexPattern")) {
    if (NotRegisters) 
      return Unknown;
    std::vector<unsigned char>
    ComplexPat(1, TP.getDAGISelEmitter().getComplexPattern(R).getValueType());
    return ComplexPat;
  } else if (R->getName() == "node" || R->getName() == "srcvalue") {
    // Placeholder.
    return Unknown;
  }
  
  TP.error("Unknown node flavor used in pattern: " + R->getName());
  return Other;
}

/// ApplyTypeConstraints - Apply all of the type constraints relevent to
/// this node and its children in the tree.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, throw an
/// exception.
bool TreePatternNode::ApplyTypeConstraints(TreePattern &TP, bool NotRegisters) {
  DAGISelEmitter &ISE = TP.getDAGISelEmitter();
  if (isLeaf()) {
    if (DefInit *DI = dynamic_cast<DefInit*>(getLeafValue())) {
      // If it's a regclass or something else known, include the type.
      return UpdateNodeType(getImplicitType(DI->getDef(), NotRegisters, TP),TP);
    } else if (IntInit *II = dynamic_cast<IntInit*>(getLeafValue())) {
      // Int inits are always integers. :)
      bool MadeChange = UpdateNodeType(MVT::isInt, TP);
      
      if (hasTypeSet()) {
        // At some point, it may make sense for this tree pattern to have
        // multiple types.  Assert here that it does not, so we revisit this
        // code when appropriate.
        assert(getExtTypes().size() >= 1 && "TreePattern doesn't have a type!");
        MVT::ValueType VT = getTypeNum(0);
        for (unsigned i = 1, e = getExtTypes().size(); i != e; ++i)
          assert(getTypeNum(i) == VT && "TreePattern has too many types!");
        
        VT = getTypeNum(0);
        if (VT != MVT::iPTR) {
          unsigned Size = MVT::getSizeInBits(VT);
          // Make sure that the value is representable for this type.
          if (Size < 32) {
            int Val = (II->getValue() << (32-Size)) >> (32-Size);
            if (Val != II->getValue())
              TP.error("Sign-extended integer value '" + itostr(II->getValue())+
                       "' is out of range for type '" + 
                       getEnumName(getTypeNum(0)) + "'!");
          }
        }
      }
      
      return MadeChange;
    }
    return false;
  }
  
  // special handling for set, which isn't really an SDNode.
  if (getOperator()->getName() == "set") {
    assert (getNumChildren() == 2 && "Only handle 2 operand set's for now!");
    bool MadeChange = getChild(0)->ApplyTypeConstraints(TP, NotRegisters);
    MadeChange |= getChild(1)->ApplyTypeConstraints(TP, NotRegisters);
    
    // Types of operands must match.
    MadeChange |= getChild(0)->UpdateNodeType(getChild(1)->getExtTypes(), TP);
    MadeChange |= getChild(1)->UpdateNodeType(getChild(0)->getExtTypes(), TP);
    MadeChange |= UpdateNodeType(MVT::isVoid, TP);
    return MadeChange;
  } else if (getOperator() == ISE.get_intrinsic_void_sdnode() ||
             getOperator() == ISE.get_intrinsic_w_chain_sdnode() ||
             getOperator() == ISE.get_intrinsic_wo_chain_sdnode()) {
    unsigned IID = 
    dynamic_cast<IntInit*>(getChild(0)->getLeafValue())->getValue();
    const CodeGenIntrinsic &Int = ISE.getIntrinsicInfo(IID);
    bool MadeChange = false;
    
    // Apply the result type to the node.
    MadeChange = UpdateNodeType(Int.ArgVTs[0], TP);
    
    if (getNumChildren() != Int.ArgVTs.size())
      TP.error("Intrinsic '" + Int.Name + "' expects " +
               utostr(Int.ArgVTs.size()-1) + " operands, not " +
               utostr(getNumChildren()-1) + " operands!");

    // Apply type info to the intrinsic ID.
    MadeChange |= getChild(0)->UpdateNodeType(MVT::iPTR, TP);
    
    for (unsigned i = 1, e = getNumChildren(); i != e; ++i) {
      MVT::ValueType OpVT = Int.ArgVTs[i];
      MadeChange |= getChild(i)->UpdateNodeType(OpVT, TP);
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    }
    return MadeChange;
  } else if (getOperator()->isSubClassOf("SDNode")) {
    const SDNodeInfo &NI = ISE.getSDNodeInfo(getOperator());
    
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
      const std::vector<MVT::ValueType> &LegalVTs
        = ISE.getTargetInfo().getLegalValueTypes();
      MVT::ValueType LegalIntVT = MVT::Other;
      for (unsigned i = 0, e = LegalVTs.size(); i != e; ++i)
        if (MVT::isInteger(LegalVTs[i]) && !MVT::isVector(LegalVTs[i])) {
          LegalIntVT = LegalVTs[i];
          break;
        }
      assert(LegalIntVT != MVT::Other && "No legal integer VT?");
            
      for (unsigned i = 0, e = BV->getNumChildren(); i != e; ++i)
        MadeChange |= BV->getChild(i)->UpdateNodeType(LegalIntVT, TP);
    }
    return MadeChange;  
  } else if (getOperator()->isSubClassOf("Instruction")) {
    const DAGInstruction &Inst = ISE.getInstruction(getOperator());
    bool MadeChange = false;
    unsigned NumResults = Inst.getNumResults();
    
    assert(NumResults <= 1 &&
           "Only supports zero or one result instrs!");

    CodeGenInstruction &InstInfo =
      ISE.getTargetInfo().getInstruction(getOperator()->getName());
    // Apply the result type to the node
    if (NumResults == 0 || InstInfo.noResults) { // FIXME: temporary hack...
      MadeChange = UpdateNodeType(MVT::isVoid, TP);
    } else {
      Record *ResultNode = Inst.getResult(0);
      assert(ResultNode->isSubClassOf("RegisterClass") &&
             "Operands should be register classes!");

      const CodeGenRegisterClass &RC = 
        ISE.getTargetInfo().getRegisterClass(ResultNode);
      MadeChange = UpdateNodeType(ConvertVTs(RC.getValueTypes()), TP);
    }

    if (getNumChildren() != Inst.getNumOperands())
      TP.error("Instruction '" + getOperator()->getName() + " expects " +
               utostr(Inst.getNumOperands()) + " operands, not " +
               utostr(getNumChildren()) + " operands!");
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
      Record *OperandNode = Inst.getOperand(i);
      MVT::ValueType VT;
      if (OperandNode->isSubClassOf("RegisterClass")) {
        const CodeGenRegisterClass &RC = 
          ISE.getTargetInfo().getRegisterClass(OperandNode);
        //VT = RC.getValueTypeNum(0);
        MadeChange |=getChild(i)->UpdateNodeType(ConvertVTs(RC.getValueTypes()),
                                                 TP);
      } else if (OperandNode->isSubClassOf("Operand")) {
        VT = getValueType(OperandNode->getValueAsDef("Type"));
        MadeChange |= getChild(i)->UpdateNodeType(VT, TP);
      } else {
        assert(0 && "Unknown operand type!");
        abort();
      }
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    }
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

/// canPatternMatch - If it is impossible for this pattern to match on this
/// target, fill in Reason and return false.  Otherwise, return true.  This is
/// used as a santity check for .td files (to prevent people from writing stuff
/// that can never possibly work), and to prevent the pattern permuter from
/// generating stuff that is useless.
bool TreePatternNode::canPatternMatch(std::string &Reason, DAGISelEmitter &ISE){
  if (isLeaf()) return true;

  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->canPatternMatch(Reason, ISE))
      return false;

  // If this is an intrinsic, handle cases that would make it not match.  For
  // example, if an operand is required to be an immediate.
  if (getOperator()->isSubClassOf("Intrinsic")) {
    // TODO:
    return true;
  }
  
  // If this node is a commutative operator, check that the LHS isn't an
  // immediate.
  const SDNodeInfo &NodeInfo = ISE.getSDNodeInfo(getOperator());
  if (NodeInfo.hasProperty(SDNodeInfo::SDNPCommutative)) {
    // Scan all of the operands of the node and make sure that only the last one
    // is a constant node.
    for (unsigned i = 0, e = getNumChildren()-1; i != e; ++i)
      if (!getChild(i)->isLeaf() && 
          getChild(i)->getOperator()->getName() == "imm") {
        Reason = "Immediate value must be on the RHS of commutative operators!";
        return false;
      }
  }
  
  return true;
}

//===----------------------------------------------------------------------===//
// TreePattern implementation
//

TreePattern::TreePattern(Record *TheRec, ListInit *RawPat, bool isInput,
                         DAGISelEmitter &ise) : TheRecord(TheRec), ISE(ise) {
   isInputPattern = isInput;
   for (unsigned i = 0, e = RawPat->getSize(); i != e; ++i)
     Trees.push_back(ParseTreePattern((DagInit*)RawPat->getElement(i)));
}

TreePattern::TreePattern(Record *TheRec, DagInit *Pat, bool isInput,
                         DAGISelEmitter &ise) : TheRecord(TheRec), ISE(ise) {
  isInputPattern = isInput;
  Trees.push_back(ParseTreePattern(Pat));
}

TreePattern::TreePattern(Record *TheRec, TreePatternNode *Pat, bool isInput,
                         DAGISelEmitter &ise) : TheRecord(TheRec), ISE(ise) {
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
      Operator->getName() != "set")
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
      std::cerr << '"';
      Arg->dump();
      std::cerr << "\": ";
      error("Unknown leaf value for tree pattern!");
    }
  }
  
  // If the operator is an intrinsic, then this is just syntactic sugar for for
  // (intrinsic_* <number>, ..children..).  Pick the right intrinsic node, and 
  // convert the intrinsic name to a number.
  if (Operator->isSubClassOf("Intrinsic")) {
    const CodeGenIntrinsic &Int = getDAGISelEmitter().getIntrinsic(Operator);
    unsigned IID = getDAGISelEmitter().getIntrinsicID(Operator)+1;

    // If this intrinsic returns void, it must have side-effects and thus a
    // chain.
    if (Int.ArgVTs[0] == MVT::isVoid) {
      Operator = getDAGISelEmitter().get_intrinsic_void_sdnode();
    } else if (Int.ModRef != CodeGenIntrinsic::NoMem) {
      // Has side-effects, requires chain.
      Operator = getDAGISelEmitter().get_intrinsic_w_chain_sdnode();
    } else {
      // Otherwise, no chain.
      Operator = getDAGISelEmitter().get_intrinsic_wo_chain_sdnode();
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

void TreePattern::dump() const { print(std::cerr); }



//===----------------------------------------------------------------------===//
// DAGISelEmitter implementation
//

// Parse all of the SDNode definitions for the target, populating SDNodes.
void DAGISelEmitter::ParseNodeInfo() {
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
void DAGISelEmitter::ParseNodeTransforms(std::ostream &OS) {
  OS << "\n// Node transformations.\n";
  std::vector<Record*> Xforms = Records.getAllDerivedDefinitions("SDNodeXForm");
  while (!Xforms.empty()) {
    Record *XFormNode = Xforms.back();
    Record *SDNode = XFormNode->getValueAsDef("Opcode");
    std::string Code = XFormNode->getValueAsCode("XFormFunction");
    SDNodeXForms.insert(std::make_pair(XFormNode,
                                       std::make_pair(SDNode, Code)));

    if (!Code.empty()) {
      std::string ClassName = getSDNodeInfo(SDNode).getSDClassName();
      const char *C2 = ClassName == "SDNode" ? "N" : "inN";

      OS << "inline SDOperand Transform_" << XFormNode->getName()
         << "(SDNode *" << C2 << ") {\n";
      if (ClassName != "SDNode")
        OS << "  " << ClassName << " *N = cast<" << ClassName << ">(inN);\n";
      OS << Code << "\n}\n";
    }

    Xforms.pop_back();
  }
}

void DAGISelEmitter::ParseComplexPatterns() {
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
/// This also emits all of the predicate functions to the output file.
///
void DAGISelEmitter::ParsePatternFragments(std::ostream &OS) {
  std::vector<Record*> Fragments = Records.getAllDerivedDefinitions("PatFrag");
  
  // First step, parse all of the fragments and emit predicate functions.
  OS << "\n// Predicate functions.\n";
  for (unsigned i = 0, e = Fragments.size(); i != e; ++i) {
    DagInit *Tree = Fragments[i]->getValueAsDag("Fragment");
    TreePattern *P = new TreePattern(Fragments[i], Tree, true, *this);
    PatternFragments[Fragments[i]] = P;
    
    // Validate the argument list, converting it to map, to discard duplicates.
    std::vector<std::string> &Args = P->getArgList();
    std::set<std::string> OperandsMap(Args.begin(), Args.end());
    
    if (OperandsMap.count(""))
      P->error("Cannot have unnamed 'node' values in pattern fragment!");
    
    // Parse the operands list.
    DagInit *OpsList = Fragments[i]->getValueAsDag("Operands");
    DefInit *OpsOp = dynamic_cast<DefInit*>(OpsList->getOperator());
    if (!OpsOp || OpsOp->getDef()->getName() != "ops")
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
      if (!OperandsMap.count(OpsList->getArgName(j)))
        P->error("'" + OpsList->getArgName(j) +
                 "' does not occur in pattern or was multiply specified!");
      OperandsMap.erase(OpsList->getArgName(j));
      Args.push_back(OpsList->getArgName(j));
    }
    
    if (!OperandsMap.empty())
      P->error("Operands list does not contain an entry for operand '" +
               *OperandsMap.begin() + "'!");

    // If there is a code init for this fragment, emit the predicate code and
    // keep track of the fact that this fragment uses it.
    std::string Code = Fragments[i]->getValueAsCode("Predicate");
    if (!Code.empty()) {
      assert(!P->getOnlyTree()->isLeaf() && "Can't be a leaf!");
      std::string ClassName =
        getSDNodeInfo(P->getOnlyTree()->getOperator()).getSDClassName();
      const char *C2 = ClassName == "SDNode" ? "N" : "inN";
      
      OS << "inline bool Predicate_" << Fragments[i]->getName()
         << "(SDNode *" << C2 << ") {\n";
      if (ClassName != "SDNode")
        OS << "  " << ClassName << " *N = cast<" << ClassName << ">(inN);\n";
      OS << Code << "\n}\n";
      P->getOnlyTree()->setPredicateFn("Predicate_"+Fragments[i]->getName());
    }
    
    // If there is a node transformation corresponding to this, keep track of
    // it.
    Record *Transform = Fragments[i]->getValueAsDef("OperandTransform");
    if (!getSDNodeTransform(Transform).second.empty())    // not noop xform?
      P->getOnlyTree()->setTransformFn(Transform);
  }
  
  OS << "\n\n";

  // Now that we've parsed all of the tree fragments, do a closure on them so
  // that there are not references to PatFrags left inside of them.
  for (std::map<Record*, TreePattern*>::iterator I = PatternFragments.begin(),
       E = PatternFragments.end(); I != E; ++I) {
    TreePattern *ThePat = I->second;
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
void DAGISelEmitter::
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
  else if (Pat->getNumChildren() & 1)
    I->error("set requires an even number of operands");
  
  if (Pat->getTransformFn())
    I->error("Cannot specify a transform function on a set node!");
  
  // Check the set destinations.
  unsigned NumValues = Pat->getNumChildren()/2;
  for (unsigned i = 0; i != NumValues; ++i) {
    TreePatternNode *Dest = Pat->getChild(i);
    if (!Dest->isLeaf())
      I->error("set destination should be a register!");
    
    DefInit *Val = dynamic_cast<DefInit*>(Dest->getLeafValue());
    if (!Val)
      I->error("set destination should be a register!");

    if (Val->getDef()->isSubClassOf("RegisterClass")) {
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
    
    // Verify and collect info from the computation.
    FindPatternInputsAndOutputs(I, Pat->getChild(i+NumValues),
                                InstInputs, InstResults,
                                InstImpInputs, InstImpResults);
  }
}

/// ParseInstructions - Parse all of the instructions, inlining and resolving
/// any fragments involved.  This populates the Instructions list with fully
/// resolved instructions.
void DAGISelEmitter::ParseInstructions() {
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
        // FIXME: temporary hack...
        if (InstInfo.noResults) {
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
    std::vector<std::string> &Args = I->getArgList();
    assert(Args.empty() && "Args list should still be empty here!");
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
      const std::string &OpName = CGI.OperandList[i].Name;
      if (OpName.empty())
        I->error("Operand #" + utostr(i) + " in operands list has no name!");

      if (!InstInputsCheck.count(OpName))
        I->error("Operand $" + OpName +
                 " does not appear in the instruction pattern");
      TreePatternNode *InVal = InstInputsCheck[OpName];
      InstInputsCheck.erase(OpName);   // It occurred, remove from map.
      
      if (InVal->isLeaf() &&
          dynamic_cast<DefInit*>(InVal->getLeafValue())) {
        Record *InRec = static_cast<DefInit*>(InVal->getLeafValue())->getDef();
        if (CGI.OperandList[i].Rec != InRec &&
            !InRec->isSubClassOf("ComplexPattern"))
          I->error("Operand $" + OpName + "'s register class disagrees"
                   " between the operand and pattern");
      }
      Operands.push_back(CGI.OperandList[i].Rec);
      
      // Construct the result for the dest-pattern operand list.
      TreePatternNode *OpNode = InVal->clone();
      
      // No predicate is useful on the result.
      OpNode->setPredicateFn("");
      
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
    TreePattern *I = TheInst.getPattern();
    if (I == 0) continue;  // No pattern.

    if (I->getNumTrees() != 1) {
      std::cerr << "CANNOT HANDLE: " << I->getRecord()->getName() << " yet!";
      continue;
    }
    TreePatternNode *Pattern = I->getTree(0);
    TreePatternNode *SrcPattern;
    if (Pattern->getOperator()->getName() == "set") {
      if (Pattern->getNumChildren() != 2)
        continue;  // Not a set of a single value (not handled so far)

      SrcPattern = Pattern->getChild(1)->clone();    
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
                               SrcPattern, DstPattern,
                               Instr->getValueAsInt("AddedComplexity")));
  }
}

void DAGISelEmitter::ParsePatterns() {
  std::vector<Record*> Patterns = Records.getAllDerivedDefinitions("Pattern");

  for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
    DagInit *Tree = Patterns[i]->getValueAsDag("PatternToMatch");
    TreePattern *Pattern = new TreePattern(Patterns[i], Tree, true, *this);

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
      
      // Infer as many types as possible.  If we cannot infer all of them, we can
      // never do anything with this pattern: report it to the user.
      InferredAllResultTypes = Result->InferAllTypes();

      // Apply the type of the result to the source pattern.  This helps us
      // resolve cases where the input type is known to be a pointer type (which
      // is considered resolved), but the result knows it needs to be 32- or
      // 64-bits.  Infer the other way for good measure.
      IterateInference = Pattern->getOnlyTree()->
        UpdateNodeType(Result->getOnlyTree()->getExtTypes(), *Result);
      IterateInference |= Result->getOnlyTree()->
        UpdateNodeType(Pattern->getOnlyTree()->getExtTypes(), *Result);
    } while (IterateInference);

    // Verify that we inferred enough types that we can do something with the
    // pattern and result.  If these fire the user has to add type casts.
    if (!InferredAllPatternTypes)
      Pattern->error("Could not infer all types in pattern!");
    if (!InferredAllResultTypes)
      Result->error("Could not infer all types in pattern result!");
    
    // Validate that the input pattern is correct.
    {
      std::map<std::string, TreePatternNode*> InstInputs;
      std::map<std::string, TreePatternNode*> InstResults;
      std::vector<Record*> InstImpInputs;
      std::vector<Record*> InstImpResults;
      FindPatternInputsAndOutputs(Pattern, Pattern->getOnlyTree(),
                                  InstInputs, InstResults,
                                  InstImpInputs, InstImpResults);
    }

    // Promote the xform function to be an explicit node if set.
    std::vector<TreePatternNode*> ResultNodeOperands;
    TreePatternNode *DstPattern = Result->getOnlyTree();
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
    if (!Pattern->getOnlyTree()->canPatternMatch(Reason, *this))
      Pattern->error("Pattern can never match: " + Reason);
    
    PatternsToMatch.
      push_back(PatternToMatch(Patterns[i]->getValueAsListInit("Predicates"),
                               Pattern->getOnlyTree(),
                               Temp.getOnlyTree(),
                               Patterns[i]->getValueAsInt("AddedComplexity")));
  }
}

/// CombineChildVariants - Given a bunch of permutations of each child of the
/// 'operator' node, put them together in all possible ways.
static void CombineChildVariants(TreePatternNode *Orig, 
               const std::vector<std::vector<TreePatternNode*> > &ChildVariants,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 DAGISelEmitter &ISE) {
  // Make sure that each operand has at least one variant to choose from.
  for (unsigned i = 0, e = ChildVariants.size(); i != e; ++i)
    if (ChildVariants[i].empty())
      return;
        
  // The end result is an all-pairs construction of the resultant pattern.
  std::vector<unsigned> Idxs;
  Idxs.resize(ChildVariants.size());
  bool NotDone = true;
  while (NotDone) {
    // Create the variant and add it to the output list.
    std::vector<TreePatternNode*> NewChildren;
    for (unsigned i = 0, e = ChildVariants.size(); i != e; ++i)
      NewChildren.push_back(ChildVariants[i][Idxs[i]]);
    TreePatternNode *R = new TreePatternNode(Orig->getOperator(), NewChildren);
    
    // Copy over properties.
    R->setName(Orig->getName());
    R->setPredicateFn(Orig->getPredicateFn());
    R->setTransformFn(Orig->getTransformFn());
    R->setTypes(Orig->getExtTypes());
    
    // If this pattern cannot every match, do not include it as a variant.
    std::string ErrString;
    if (!R->canPatternMatch(ErrString, ISE)) {
      delete R;
    } else {
      bool AlreadyExists = false;
      
      // Scan to see if this pattern has already been emitted.  We can get
      // duplication due to things like commuting:
      //   (and GPRC:$a, GPRC:$b) -> (and GPRC:$b, GPRC:$a)
      // which are the same pattern.  Ignore the dups.
      for (unsigned i = 0, e = OutVariants.size(); i != e; ++i)
        if (R->isIsomorphicTo(OutVariants[i])) {
          AlreadyExists = true;
          break;
        }
      
      if (AlreadyExists)
        delete R;
      else
        OutVariants.push_back(R);
    }
    
    // Increment indices to the next permutation.
    NotDone = false;
    // Look for something we can increment without causing a wrap-around.
    for (unsigned IdxsIdx = 0; IdxsIdx != Idxs.size(); ++IdxsIdx) {
      if (++Idxs[IdxsIdx] < ChildVariants[IdxsIdx].size()) {
        NotDone = true;   // Found something to increment.
        break;
      }
      Idxs[IdxsIdx] = 0;
    }
  }
}

/// CombineChildVariants - A helper function for binary operators.
///
static void CombineChildVariants(TreePatternNode *Orig, 
                                 const std::vector<TreePatternNode*> &LHS,
                                 const std::vector<TreePatternNode*> &RHS,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 DAGISelEmitter &ISE) {
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.push_back(LHS);
  ChildVariants.push_back(RHS);
  CombineChildVariants(Orig, ChildVariants, OutVariants, ISE);
}  


static void GatherChildrenOfAssociativeOpcode(TreePatternNode *N,
                                     std::vector<TreePatternNode *> &Children) {
  assert(N->getNumChildren()==2 &&"Associative but doesn't have 2 children!");
  Record *Operator = N->getOperator();
  
  // Only permit raw nodes.
  if (!N->getName().empty() || !N->getPredicateFn().empty() ||
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
                               DAGISelEmitter &ISE) {
  // We cannot permute leaves.
  if (N->isLeaf()) {
    OutVariants.push_back(N);
    return;
  }

  // Look up interesting info about the node.
  const SDNodeInfo &NodeInfo = ISE.getSDNodeInfo(N->getOperator());

  // If this node is associative, reassociate.
  if (NodeInfo.hasProperty(SDNodeInfo::SDNPAssociative)) {
    // Reassociate by pulling together all of the linked operators 
    std::vector<TreePatternNode*> MaximalChildren;
    GatherChildrenOfAssociativeOpcode(N, MaximalChildren);

    // Only handle child sizes of 3.  Otherwise we'll end up trying too many
    // permutations.
    if (MaximalChildren.size() == 3) {
      // Find the variants of all of our maximal children.
      std::vector<TreePatternNode*> AVariants, BVariants, CVariants;
      GenerateVariantsOf(MaximalChildren[0], AVariants, ISE);
      GenerateVariantsOf(MaximalChildren[1], BVariants, ISE);
      GenerateVariantsOf(MaximalChildren[2], CVariants, ISE);
      
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
      CombineChildVariants(N, AVariants, BVariants, ABVariants, ISE);
      CombineChildVariants(N, BVariants, AVariants, BAVariants, ISE);
      CombineChildVariants(N, AVariants, CVariants, ACVariants, ISE);
      CombineChildVariants(N, CVariants, AVariants, CAVariants, ISE);
      CombineChildVariants(N, BVariants, CVariants, BCVariants, ISE);
      CombineChildVariants(N, CVariants, BVariants, CBVariants, ISE);

      // Combine those into the result: (x op x) op x
      CombineChildVariants(N, ABVariants, CVariants, OutVariants, ISE);
      CombineChildVariants(N, BAVariants, CVariants, OutVariants, ISE);
      CombineChildVariants(N, ACVariants, BVariants, OutVariants, ISE);
      CombineChildVariants(N, CAVariants, BVariants, OutVariants, ISE);
      CombineChildVariants(N, BCVariants, AVariants, OutVariants, ISE);
      CombineChildVariants(N, CBVariants, AVariants, OutVariants, ISE);

      // Combine those into the result: x op (x op x)
      CombineChildVariants(N, CVariants, ABVariants, OutVariants, ISE);
      CombineChildVariants(N, CVariants, BAVariants, OutVariants, ISE);
      CombineChildVariants(N, BVariants, ACVariants, OutVariants, ISE);
      CombineChildVariants(N, BVariants, CAVariants, OutVariants, ISE);
      CombineChildVariants(N, AVariants, BCVariants, OutVariants, ISE);
      CombineChildVariants(N, AVariants, CBVariants, OutVariants, ISE);
      return;
    }
  }
  
  // Compute permutations of all children.
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.resize(N->getNumChildren());
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
    GenerateVariantsOf(N->getChild(i), ChildVariants[i], ISE);

  // Build all permutations based on how the children were formed.
  CombineChildVariants(N, ChildVariants, OutVariants, ISE);

  // If this node is commutative, consider the commuted order.
  if (NodeInfo.hasProperty(SDNodeInfo::SDNPCommutative)) {
    assert(N->getNumChildren()==2 &&"Commutative but doesn't have 2 children!");
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
    if (NC == 2)
      CombineChildVariants(N, ChildVariants[1], ChildVariants[0],
                           OutVariants, ISE);
  }
}


// GenerateVariants - Generate variants.  For example, commutative patterns can
// match multiple ways.  Add them to PatternsToMatch as well.
void DAGISelEmitter::GenerateVariants() {
  
  DEBUG(std::cerr << "Generating instruction variants.\n");
  
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
    std::vector<TreePatternNode*> Variants;
    GenerateVariantsOf(PatternsToMatch[i].getSrcPattern(), Variants, *this);

    assert(!Variants.empty() && "Must create at least original variant!");
    Variants.erase(Variants.begin());  // Remove the original pattern.

    if (Variants.empty())  // No variants for this pattern.
      continue;

    DEBUG(std::cerr << "FOUND VARIANTS OF: ";
          PatternsToMatch[i].getSrcPattern()->dump();
          std::cerr << "\n");

    for (unsigned v = 0, e = Variants.size(); v != e; ++v) {
      TreePatternNode *Variant = Variants[v];

      DEBUG(std::cerr << "  VAR#" << v <<  ": ";
            Variant->dump();
            std::cerr << "\n");
      
      // Scan to see if an instruction or explicit pattern already matches this.
      bool AlreadyExists = false;
      for (unsigned p = 0, e = PatternsToMatch.size(); p != e; ++p) {
        // Check to see if this variant already exists.
        if (Variant->isIsomorphicTo(PatternsToMatch[p].getSrcPattern())) {
          DEBUG(std::cerr << "  *** ALREADY EXISTS, ignoring variant.\n");
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
                                 PatternsToMatch[i].getAddedComplexity()));
    }

    DEBUG(std::cerr << "\n");
  }
}


// NodeIsComplexPattern - return true if N is a leaf node and a subclass of
// ComplexPattern.
static bool NodeIsComplexPattern(TreePatternNode *N)
{
  return (N->isLeaf() &&
          dynamic_cast<DefInit*>(N->getLeafValue()) &&
          static_cast<DefInit*>(N->getLeafValue())->getDef()->
          isSubClassOf("ComplexPattern"));
}

// NodeGetComplexPattern - return the pointer to the ComplexPattern if N
// is a leaf node and a subclass of ComplexPattern, else it returns NULL.
static const ComplexPattern *NodeGetComplexPattern(TreePatternNode *N,
                                                   DAGISelEmitter &ISE)
{
  if (N->isLeaf() &&
      dynamic_cast<DefInit*>(N->getLeafValue()) &&
      static_cast<DefInit*>(N->getLeafValue())->getDef()->
      isSubClassOf("ComplexPattern")) {
    return &ISE.getComplexPattern(static_cast<DefInit*>(N->getLeafValue())
                                  ->getDef());
  }
  return NULL;
}

/// getPatternSize - Return the 'size' of this pattern.  We want to match large
/// patterns before small ones.  This is used to determine the size of a
/// pattern.
static unsigned getPatternSize(TreePatternNode *P, DAGISelEmitter &ISE) {
  assert((isExtIntegerInVTs(P->getExtTypes()) || 
          isExtFloatingPointInVTs(P->getExtTypes()) ||
          P->getExtTypeNum(0) == MVT::isVoid ||
          P->getExtTypeNum(0) == MVT::Flag ||
          P->getExtTypeNum(0) == MVT::iPTR) && 
         "Not a valid pattern node to size!");
  unsigned Size = 2;  // The node itself.
  // If the root node is a ConstantSDNode, increases its size.
  // e.g. (set R32:$dst, 0).
  if (P->isLeaf() && dynamic_cast<IntInit*>(P->getLeafValue()))
    Size++;

  // FIXME: This is a hack to statically increase the priority of patterns
  // which maps a sub-dag to a complex pattern. e.g. favors LEA over ADD.
  // Later we can allow complexity / cost for each pattern to be (optionally)
  // specified. To get best possible pattern match we'll need to dynamically
  // calculate the complexity of all patterns a dag can potentially map to.
  const ComplexPattern *AM = NodeGetComplexPattern(P, ISE);
  if (AM)
    Size += AM->getNumOperands() * 2;

  // If this node has some predicate function that must match, it adds to the
  // complexity of this node.
  if (!P->getPredicateFn().empty())
    ++Size;
  
  // Count children in the count if they are also nodes.
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = P->getChild(i);
    if (!Child->isLeaf() && Child->getExtTypeNum(0) != MVT::Other)
      Size += getPatternSize(Child, ISE);
    else if (Child->isLeaf()) {
      if (dynamic_cast<IntInit*>(Child->getLeafValue())) 
        Size += 3;  // Matches a ConstantSDNode (+2) and a specific value (+1).
      else if (NodeIsComplexPattern(Child))
        Size += getPatternSize(Child, ISE);
      else if (!Child->getPredicateFn().empty())
        ++Size;
    }
  }
  
  return Size;
}

/// getResultPatternCost - Compute the number of instructions for this pattern.
/// This is a temporary hack.  We should really include the instruction
/// latencies in this calculation.
static unsigned getResultPatternCost(TreePatternNode *P, DAGISelEmitter &ISE) {
  if (P->isLeaf()) return 0;
  
  unsigned Cost = 0;
  Record *Op = P->getOperator();
  if (Op->isSubClassOf("Instruction")) {
    Cost++;
    CodeGenInstruction &II = ISE.getTargetInfo().getInstruction(Op->getName());
    if (II.usesCustomDAGSchedInserter)
      Cost += 10;
  }
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i)
    Cost += getResultPatternCost(P->getChild(i), ISE);
  return Cost;
}

/// getResultPatternCodeSize - Compute the code size of instructions for this
/// pattern.
static unsigned getResultPatternSize(TreePatternNode *P, DAGISelEmitter &ISE) {
  if (P->isLeaf()) return 0;

  unsigned Cost = 0;
  Record *Op = P->getOperator();
  if (Op->isSubClassOf("Instruction")) {
    Cost += Op->getValueAsInt("CodeSize");
  }
  for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i)
    Cost += getResultPatternSize(P->getChild(i), ISE);
  return Cost;
}

// PatternSortingPredicate - return true if we prefer to match LHS before RHS.
// In particular, we want to match maximal patterns first and lowest cost within
// a particular complexity first.
struct PatternSortingPredicate {
  PatternSortingPredicate(DAGISelEmitter &ise) : ISE(ise) {};
  DAGISelEmitter &ISE;

  bool operator()(PatternToMatch *LHS,
                  PatternToMatch *RHS) {
    unsigned LHSSize = getPatternSize(LHS->getSrcPattern(), ISE);
    unsigned RHSSize = getPatternSize(RHS->getSrcPattern(), ISE);
    LHSSize += LHS->getAddedComplexity();
    RHSSize += RHS->getAddedComplexity();
    if (LHSSize > RHSSize) return true;   // LHS -> bigger -> less cost
    if (LHSSize < RHSSize) return false;
    
    // If the patterns have equal complexity, compare generated instruction cost
    unsigned LHSCost = getResultPatternCost(LHS->getDstPattern(), ISE);
    unsigned RHSCost = getResultPatternCost(RHS->getDstPattern(), ISE);
    if (LHSCost < RHSCost) return true;
    if (LHSCost > RHSCost) return false;

    return getResultPatternSize(LHS->getDstPattern(), ISE) <
      getResultPatternSize(RHS->getDstPattern(), ISE);
  }
};

/// getRegisterValueType - Look up and return the first ValueType of specified 
/// RegisterClass record
static MVT::ValueType getRegisterValueType(Record *R, const CodeGenTarget &T) {
  if (const CodeGenRegisterClass *RC = T.getRegisterClassForRegister(R))
    return RC->getValueTypeNum(0);
  return MVT::Other;
}


/// RemoveAllTypes - A quick recursive walk over a pattern which removes all
/// type information from it.
static void RemoveAllTypes(TreePatternNode *N) {
  N->removeTypes();
  if (!N->isLeaf())
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
      RemoveAllTypes(N->getChild(i));
}

Record *DAGISelEmitter::getSDNodeNamed(const std::string &Name) const {
  Record *N = Records.getDef(Name);
  if (!N || !N->isSubClassOf("SDNode")) {
    std::cerr << "Error getting SDNode '" << Name << "'!\n";
    exit(1);
  }
  return N;
}

/// NodeHasProperty - return true if TreePatternNode has the specified
/// property.
static bool NodeHasProperty(TreePatternNode *N, SDNodeInfo::SDNP Property,
                            DAGISelEmitter &ISE)
{
  if (N->isLeaf()) return false;
  Record *Operator = N->getOperator();
  if (!Operator->isSubClassOf("SDNode")) return false;

  const SDNodeInfo &NodeInfo = ISE.getSDNodeInfo(Operator);
  return NodeInfo.hasProperty(Property);
}

static bool PatternHasProperty(TreePatternNode *N, SDNodeInfo::SDNP Property,
                               DAGISelEmitter &ISE)
{
  if (NodeHasProperty(N, Property, ISE))
    return true;

  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = N->getChild(i);
    if (PatternHasProperty(Child, Property, ISE))
      return true;
  }

  return false;
}

class PatternCodeEmitter {
private:
  DAGISelEmitter &ISE;

  // Predicates.
  ListInit *Predicates;
  // Pattern cost.
  unsigned Cost;
  // Instruction selector pattern.
  TreePatternNode *Pattern;
  // Matched instruction.
  TreePatternNode *Instruction;
  
  // Node to name mapping
  std::map<std::string, std::string> VariableMap;
  // Node to operator mapping
  std::map<std::string, Record*> OperatorMap;
  // Names of all the folded nodes which produce chains.
  std::vector<std::pair<std::string, unsigned> > FoldedChains;
  std::set<std::string> Duplicates;

  /// GeneratedCode - This is the buffer that we emit code to.  The first bool
  /// indicates whether this is an exit predicate (something that should be
  /// tested, and if true, the match fails) [when true] or normal code to emit
  /// [when false].
  std::vector<std::pair<bool, std::string> > &GeneratedCode;
  /// GeneratedDecl - This is the set of all SDOperand declarations needed for
  /// the set of patterns for each top-level opcode.
  std::set<std::pair<unsigned, std::string> > &GeneratedDecl;
  /// TargetOpcodes - The target specific opcodes used by the resulting
  /// instructions.
  std::vector<std::string> &TargetOpcodes;
  std::vector<std::string> &TargetVTs;

  std::string ChainName;
  unsigned TmpNo;
  unsigned OpcNo;
  unsigned VTNo;
  
  void emitCheck(const std::string &S) {
    if (!S.empty())
      GeneratedCode.push_back(std::make_pair(true, S));
  }
  void emitCode(const std::string &S) {
    if (!S.empty())
      GeneratedCode.push_back(std::make_pair(false, S));
  }
  void emitDecl(const std::string &S, unsigned T=0) {
    assert(!S.empty() && "Invalid declaration");
    GeneratedDecl.insert(std::make_pair(T, S));
  }
  void emitOpcode(const std::string &Opc) {
    TargetOpcodes.push_back(Opc);
    OpcNo++;
  }
  void emitVT(const std::string &VT) {
    TargetVTs.push_back(VT);
    VTNo++;
  }
public:
  PatternCodeEmitter(DAGISelEmitter &ise, ListInit *preds,
                     TreePatternNode *pattern, TreePatternNode *instr,
                     std::vector<std::pair<bool, std::string> > &gc,
                     std::set<std::pair<unsigned, std::string> > &gd,
                     std::vector<std::string> &to,
                     std::vector<std::string> &tv)
  : ISE(ise), Predicates(preds), Pattern(pattern), Instruction(instr),
    GeneratedCode(gc), GeneratedDecl(gd), TargetOpcodes(to), TargetVTs(tv),
    TmpNo(0), OpcNo(0), VTNo(0) {}

  /// EmitMatchCode - Emit a matcher for N, going to the label for PatternNo
  /// if the match fails. At this point, we already know that the opcode for N
  /// matches, and the SDNode for the result has the RootName specified name.
  void EmitMatchCode(TreePatternNode *N, TreePatternNode *P,
                     const std::string &RootName, const std::string &ParentName,
                     const std::string &ChainSuffix, bool &FoundChain) {
    bool isRoot = (P == NULL);
    // Emit instruction predicates. Each predicate is just a string for now.
    if (isRoot) {
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
            PredicateCheck += " || ";
          PredicateCheck += "(" + Def->getValueAsString("CondString") + ")";
        }
      }
      
      emitCheck(PredicateCheck);
    }

    if (N->isLeaf()) {
      if (IntInit *II = dynamic_cast<IntInit*>(N->getLeafValue())) {
        emitCheck("cast<ConstantSDNode>(" + RootName +
                  ")->getSignExtended() == " + itostr(II->getValue()));
        return;
      } else if (!NodeIsComplexPattern(N)) {
        assert(0 && "Cannot match this as a leaf value!");
        abort();
      }
    }
  
    // If this node has a name associated with it, capture it in VariableMap. If
    // we already saw this in the pattern, emit code to verify dagness.
    if (!N->getName().empty()) {
      std::string &VarMapEntry = VariableMap[N->getName()];
      if (VarMapEntry.empty()) {
        VarMapEntry = RootName;
      } else {
        // If we get here, this is a second reference to a specific name.  Since
        // we already have checked that the first reference is valid, we don't
        // have to recursively match it, just check that it's the same as the
        // previously named thing.
        emitCheck(VarMapEntry + " == " + RootName);
        return;
      }

      if (!N->isLeaf())
        OperatorMap[N->getName()] = N->getOperator();
    }


    // Emit code to load the child nodes and match their contents recursively.
    unsigned OpNo = 0;
    bool NodeHasChain = NodeHasProperty   (N, SDNodeInfo::SDNPHasChain, ISE);
    bool HasChain     = PatternHasProperty(N, SDNodeInfo::SDNPHasChain, ISE);
    bool HasOutFlag   = PatternHasProperty(N, SDNodeInfo::SDNPOutFlag,  ISE);
    bool EmittedUseCheck = false;
    if (HasChain) {
      if (NodeHasChain)
        OpNo = 1;
      if (!isRoot) {
        const SDNodeInfo &CInfo = ISE.getSDNodeInfo(N->getOperator());
        // Multiple uses of actual result?
        emitCheck(RootName + ".hasOneUse()");
        EmittedUseCheck = true;
        if (NodeHasChain) {
          // FIXME: Don't fold if 1) the parent node writes a flag, 2) the node
          // has a chain use.
          // This a workaround for this problem:
          //
          //          [ch, r : ld]
          //             ^ ^
          //             | |
          //      [XX]--/   \- [flag : cmp]
          //       ^             ^
          //       |             |
          //       \---[br flag]-
          //
          // cmp + br should be considered as a single node as they are flagged
          // together. So, if the ld is folded into the cmp, the XX node in the
          // graph is now both an operand and a use of the ld/cmp/br node.
          if (NodeHasProperty(P, SDNodeInfo::SDNPOutFlag, ISE))
            emitCheck(ParentName + ".Val->isOnlyUse(" +  RootName + ".Val)");

          // If the immediate use can somehow reach this node through another
          // path, then can't fold it either or it will create a cycle.
          // e.g. In the following diagram, XX can reach ld through YY. If
          // ld is folded into XX, then YY is both a predecessor and a successor
          // of XX.
          //
          //         [ld]
          //         ^  ^
          //         |  |
          //        /   \---
          //      /        [YY]
          //      |         ^
          //     [XX]-------|
          const SDNodeInfo &PInfo = ISE.getSDNodeInfo(P->getOperator());
          if (PInfo.getNumOperands() > 1 ||
              PInfo.hasProperty(SDNodeInfo::SDNPHasChain) ||
              PInfo.hasProperty(SDNodeInfo::SDNPInFlag) ||
              PInfo.hasProperty(SDNodeInfo::SDNPOptInFlag))
            emitCheck("CanBeFoldedBy(" + RootName + ".Val, " + ParentName +
                      ".Val)");
        }
      }

      if (NodeHasChain) {
        if (FoundChain)
          emitCheck("Chain.Val == " + RootName + ".Val");
        else
          FoundChain = true;
        ChainName = "Chain" + ChainSuffix;
        emitDecl(ChainName);
        emitCode(ChainName + " = " + RootName +
                 ".getOperand(0);");
      }
    }

    // Don't fold any node which reads or writes a flag and has multiple uses.
    // FIXME: We really need to separate the concepts of flag and "glue". Those
    // real flag results, e.g. X86CMP output, can have multiple uses.
    // FIXME: If the optional incoming flag does not exist. Then it is ok to
    // fold it.
    if (!isRoot &&
        (PatternHasProperty(N, SDNodeInfo::SDNPInFlag, ISE) ||
         PatternHasProperty(N, SDNodeInfo::SDNPOptInFlag, ISE) ||
         PatternHasProperty(N, SDNodeInfo::SDNPOutFlag, ISE))) {
      const SDNodeInfo &CInfo = ISE.getSDNodeInfo(N->getOperator());
      if (!EmittedUseCheck) {
        // Multiple uses of actual result?
        emitCheck(RootName + ".hasOneUse()");
      }
    }

    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i, ++OpNo) {
      emitDecl(RootName + utostr(OpNo));
      emitCode(RootName + utostr(OpNo) + " = " +
               RootName + ".getOperand(" +utostr(OpNo) + ");");
      TreePatternNode *Child = N->getChild(i);
    
      if (!Child->isLeaf()) {
        // If it's not a leaf, recursively match.
        const SDNodeInfo &CInfo = ISE.getSDNodeInfo(Child->getOperator());
        emitCheck(RootName + utostr(OpNo) + ".getOpcode() == " +
                  CInfo.getEnumName());
        EmitMatchCode(Child, N, RootName + utostr(OpNo), RootName,
                      ChainSuffix + utostr(OpNo), FoundChain);
        if (NodeHasProperty(Child, SDNodeInfo::SDNPHasChain, ISE))
          FoldedChains.push_back(std::make_pair(RootName + utostr(OpNo),
                                                CInfo.getNumResults()));
      } else {
        // If this child has a name associated with it, capture it in VarMap. If
        // we already saw this in the pattern, emit code to verify dagness.
        if (!Child->getName().empty()) {
          std::string &VarMapEntry = VariableMap[Child->getName()];
          if (VarMapEntry.empty()) {
            VarMapEntry = RootName + utostr(OpNo);
          } else {
            // If we get here, this is a second reference to a specific name.
            // Since we already have checked that the first reference is valid,
            // we don't have to recursively match it, just check that it's the
            // same as the previously named thing.
            emitCheck(VarMapEntry + " == " + RootName + utostr(OpNo));
            Duplicates.insert(RootName + utostr(OpNo));
            continue;
          }
        }
      
        // Handle leaves of various types.
        if (DefInit *DI = dynamic_cast<DefInit*>(Child->getLeafValue())) {
          Record *LeafRec = DI->getDef();
          if (LeafRec->isSubClassOf("RegisterClass")) {
            // Handle register references.  Nothing to do here.
          } else if (LeafRec->isSubClassOf("Register")) {
            // Handle register references.
          } else if (LeafRec->isSubClassOf("ComplexPattern")) {
            // Handle complex pattern. Nothing to do here.
          } else if (LeafRec->getName() == "srcvalue") {
            // Place holder for SRCVALUE nodes. Nothing to do here.
          } else if (LeafRec->isSubClassOf("ValueType")) {
            // Make sure this is the specified value type.
            emitCheck("cast<VTSDNode>(" + RootName + utostr(OpNo) +
                      ")->getVT() == MVT::" + LeafRec->getName());
          } else if (LeafRec->isSubClassOf("CondCode")) {
            // Make sure this is the specified cond code.
            emitCheck("cast<CondCodeSDNode>(" + RootName + utostr(OpNo) +
                      ")->get() == ISD::" + LeafRec->getName());
          } else {
#ifndef NDEBUG
            Child->dump();
            std::cerr << " ";
#endif
            assert(0 && "Unknown leaf type!");
          }
        } else if (IntInit *II =
                       dynamic_cast<IntInit*>(Child->getLeafValue())) {
          emitCheck("isa<ConstantSDNode>(" + RootName + utostr(OpNo) + ")");
          unsigned CTmp = TmpNo++;
          emitCode("int64_t CN"+utostr(CTmp)+" = cast<ConstantSDNode>("+
                   RootName + utostr(OpNo) + ")->getSignExtended();");

          emitCheck("CN" + utostr(CTmp) + " == " +itostr(II->getValue()));
        } else {
#ifndef NDEBUG
          Child->dump();
#endif
          assert(0 && "Unknown leaf type!");
        }
      }
    }

    // If there is a node predicate for this, emit the call.
    if (!N->getPredicateFn().empty())
      emitCheck(N->getPredicateFn() + "(" + RootName + ".Val)");
  }

  /// EmitResultCode - Emit the action for a pattern.  Now that it has matched
  /// we actually have to build a DAG!
  std::pair<unsigned, unsigned>
  EmitResultCode(TreePatternNode *N, bool LikeLeaf = false,
                 bool isRoot = false) {
    // This is something selected from the pattern we matched.
    if (!N->getName().empty()) {
      std::string &Val = VariableMap[N->getName()];
      assert(!Val.empty() &&
             "Variable referenced but not defined and not caught earlier!");
      if (Val[0] == 'T' && Val[1] == 'm' && Val[2] == 'p') {
        // Already selected this operand, just return the tmpval.
        return std::make_pair(1, atoi(Val.c_str()+3));
      }

      const ComplexPattern *CP;
      unsigned ResNo = TmpNo++;
      unsigned NumRes = 1;
      if (!N->isLeaf() && N->getOperator()->getName() == "imm") {
        assert(N->getExtTypes().size() == 1 && "Multiple types not handled!");
        std::string CastType;
        switch (N->getTypeNum(0)) {
        default: assert(0 && "Unknown type for constant node!");
        case MVT::i1:  CastType = "bool"; break;
        case MVT::i8:  CastType = "unsigned char"; break;
        case MVT::i16: CastType = "unsigned short"; break;
        case MVT::i32: CastType = "unsigned"; break;
        case MVT::i64: CastType = "uint64_t"; break;
        }
        emitDecl("Tmp" + utostr(ResNo));
        emitCode("Tmp" + utostr(ResNo) + 
                 " = CurDAG->getTargetConstant(((" + CastType +
                 ") cast<ConstantSDNode>(" + Val + ")->getValue()), " +
                 getEnumName(N->getTypeNum(0)) + ");");
      } else if (!N->isLeaf() && N->getOperator()->getName() == "texternalsym"){
        Record *Op = OperatorMap[N->getName()];
        // Transform ExternalSymbol to TargetExternalSymbol
        if (Op && Op->getName() == "externalsym") {
          emitDecl("Tmp" + utostr(ResNo));
          emitCode("Tmp" + utostr(ResNo) + " = CurDAG->getTarget"
                   "ExternalSymbol(cast<ExternalSymbolSDNode>(" +
                   Val + ")->getSymbol(), " +
                   getEnumName(N->getTypeNum(0)) + ");");
        } else {
          emitDecl("Tmp" + utostr(ResNo));
          emitCode("Tmp" + utostr(ResNo) + " = " + Val + ";");
        }
      } else if (!N->isLeaf() && N->getOperator()->getName() == "tglobaladdr") {
        Record *Op = OperatorMap[N->getName()];
        // Transform GlobalAddress to TargetGlobalAddress
        if (Op && Op->getName() == "globaladdr") {
          emitDecl("Tmp" + utostr(ResNo));
          emitCode("Tmp" + utostr(ResNo) + " = CurDAG->getTarget"
                   "GlobalAddress(cast<GlobalAddressSDNode>(" + Val +
                   ")->getGlobal(), " + getEnumName(N->getTypeNum(0)) +
                   ");");
        } else {
          emitDecl("Tmp" + utostr(ResNo));
          emitCode("Tmp" + utostr(ResNo) + " = " + Val + ";");
        }
      } else if (!N->isLeaf() && N->getOperator()->getName() == "texternalsym"){
        emitDecl("Tmp" + utostr(ResNo));
        emitCode("Tmp" + utostr(ResNo) + " = " + Val + ";");
      } else if (!N->isLeaf() && N->getOperator()->getName() == "tconstpool") {
        emitDecl("Tmp" + utostr(ResNo));
        emitCode("Tmp" + utostr(ResNo) + " = " + Val + ";");
      } else if (N->isLeaf() && (CP = NodeGetComplexPattern(N, ISE))) {
        std::string Fn = CP->getSelectFunc();
        NumRes = CP->getNumOperands();
        for (unsigned i = 0; i < NumRes; ++i)
          emitDecl("CPTmp" + utostr(i+ResNo));

        std::string Code = Fn + "(" + Val;
        for (unsigned i = 0; i < NumRes; i++)
          Code += ", CPTmp" + utostr(i + ResNo);
        emitCheck(Code + ")");

        for (unsigned i = 0; i < NumRes; ++i) {
          emitDecl("Tmp" + utostr(i+ResNo));
          emitCode("AddToQueue(Tmp" + utostr(i+ResNo) + ", CPTmp" +
                   utostr(i+ResNo) + ");");
        }

        TmpNo = ResNo + NumRes;
      } else {
        emitDecl("Tmp" + utostr(ResNo));
        // This node, probably wrapped in a SDNodeXForms, behaves like a leaf
        // node even if it isn't one. Don't select it.
        if (LikeLeaf)
          emitCode("Tmp" + utostr(ResNo) + " = " + Val + ";");
        else {
          emitCode("AddToQueue(Tmp" + utostr(ResNo) + ", " + Val + ");");
          if (isRoot && N->isLeaf()) {
            emitCode("ReplaceUses(N, Tmp" + utostr(ResNo) + ");");
            emitCode("Result = Tmp" + utostr(ResNo) + ";");
            emitCode("return;");
          }
        }
      }
      // Add Tmp<ResNo> to VariableMap, so that we don't multiply select this
      // value if used multiple times by this pattern result.
      Val = "Tmp"+utostr(ResNo);
      return std::make_pair(NumRes, ResNo);
    }
    if (N->isLeaf()) {
      // If this is an explicit register reference, handle it.
      if (DefInit *DI = dynamic_cast<DefInit*>(N->getLeafValue())) {
        unsigned ResNo = TmpNo++;
        if (DI->getDef()->isSubClassOf("Register")) {
          emitDecl("Tmp" + utostr(ResNo));
          emitCode("Tmp" + utostr(ResNo) + " = CurDAG->getRegister(" +
                   ISE.getQualifiedName(DI->getDef()) + ", " +
                   getEnumName(N->getTypeNum(0)) + ");");
          return std::make_pair(1, ResNo);
        }
      } else if (IntInit *II = dynamic_cast<IntInit*>(N->getLeafValue())) {
        unsigned ResNo = TmpNo++;
        assert(N->getExtTypes().size() == 1 && "Multiple types not handled!");
        emitDecl("Tmp" + utostr(ResNo));
        emitCode("Tmp" + utostr(ResNo) + 
                 " = CurDAG->getTargetConstant(" + itostr(II->getValue()) +
                 ", " + getEnumName(N->getTypeNum(0)) + ");");
        return std::make_pair(1, ResNo);
      }
    
#ifndef NDEBUG
      N->dump();
#endif
      assert(0 && "Unknown leaf type!");
      return std::make_pair(1, ~0U);
    }

    Record *Op = N->getOperator();
    if (Op->isSubClassOf("Instruction")) {
      const CodeGenTarget &CGT = ISE.getTargetInfo();
      CodeGenInstruction &II = CGT.getInstruction(Op->getName());
      const DAGInstruction &Inst = ISE.getInstruction(Op);
      TreePattern *InstPat = Inst.getPattern();
      TreePatternNode *InstPatNode =
        isRoot ? (InstPat ? InstPat->getOnlyTree() : Pattern)
               : (InstPat ? InstPat->getOnlyTree() : NULL);
      if (InstPatNode && InstPatNode->getOperator()->getName() == "set") {
        InstPatNode = InstPatNode->getChild(1);
      }
      bool HasVarOps     = isRoot && II.hasVariableNumberOfOperands;
      bool HasImpInputs  = isRoot && Inst.getNumImpOperands() > 0;
      bool HasImpResults = isRoot && Inst.getNumImpResults() > 0;
      bool NodeHasOptInFlag = isRoot &&
        PatternHasProperty(Pattern, SDNodeInfo::SDNPOptInFlag, ISE);
      bool NodeHasInFlag  = isRoot &&
        PatternHasProperty(Pattern, SDNodeInfo::SDNPInFlag, ISE);
      bool NodeHasOutFlag = HasImpResults || (isRoot &&
        PatternHasProperty(Pattern, SDNodeInfo::SDNPOutFlag, ISE));
      bool NodeHasChain = InstPatNode &&
        PatternHasProperty(InstPatNode, SDNodeInfo::SDNPHasChain, ISE);
      bool InputHasChain = isRoot &&
        NodeHasProperty(Pattern, SDNodeInfo::SDNPHasChain, ISE);

      if (NodeHasInFlag || NodeHasOutFlag || NodeHasOptInFlag || HasImpInputs)
        emitDecl("InFlag");
      if (NodeHasOptInFlag) {
        emitDecl("HasInFlag", 2);
        emitCode("HasInFlag = "
           "(N.getOperand(N.getNumOperands()-1).getValueType() == MVT::Flag);");
      }
      if (HasVarOps)
        emitCode("std::vector<SDOperand> Ops;");

      // How many results is this pattern expected to produce?
      unsigned PatResults = 0;
      for (unsigned i = 0, e = Pattern->getExtTypes().size(); i != e; i++) {
        MVT::ValueType VT = Pattern->getTypeNum(i);
        if (VT != MVT::isVoid && VT != MVT::Flag)
          PatResults++;
      }

      // Determine operand emission order. Complex pattern first.
      std::vector<std::pair<unsigned, TreePatternNode*> > EmitOrder;
      std::vector<std::pair<unsigned, TreePatternNode*> >::iterator OI;
      for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
        TreePatternNode *Child = N->getChild(i);
        if (i == 0) {
          EmitOrder.push_back(std::make_pair(i, Child));
          OI = EmitOrder.begin();
        } else if (NodeIsComplexPattern(Child)) {
          OI = EmitOrder.insert(OI, std::make_pair(i, Child));
        } else {
          EmitOrder.push_back(std::make_pair(i, Child));
        }
      }

      // Emit all of the operands.
      std::vector<std::pair<unsigned, unsigned> > NumTemps(EmitOrder.size());
      for (unsigned i = 0, e = EmitOrder.size(); i != e; ++i) {
        unsigned OpOrder       = EmitOrder[i].first;
        TreePatternNode *Child = EmitOrder[i].second;
        std::pair<unsigned, unsigned> NumTemp =  EmitResultCode(Child);
        NumTemps[OpOrder] = NumTemp;
      }

      // List all the operands in the right order.
      std::vector<unsigned> Ops;
      for (unsigned i = 0, e = NumTemps.size(); i != e; i++) {
        for (unsigned j = 0; j < NumTemps[i].first; j++)
          Ops.push_back(NumTemps[i].second + j);
      }

      // Emit all the chain and CopyToReg stuff.
      bool ChainEmitted = NodeHasChain;
      if (NodeHasChain)
        emitCode("AddToQueue(" + ChainName + ", " + ChainName + ");");
      if (NodeHasInFlag || HasImpInputs)
        EmitInFlagSelectCode(Pattern, "N", ChainEmitted, true);
      if (NodeHasOptInFlag) {
        emitCode("if (HasInFlag)");
        emitCode("  AddToQueue(InFlag, N.getOperand(N.getNumOperands()-1));");
      }

      unsigned NumResults = Inst.getNumResults();    
      unsigned ResNo = TmpNo++;
      if (!isRoot || InputHasChain || NodeHasChain || NodeHasOutFlag ||
          NodeHasOptInFlag) {
        std::string Code;
        std::string Code2;
        std::string NodeName;
        if (!isRoot) {
          NodeName = "Tmp" + utostr(ResNo);
          emitDecl(NodeName);
          Code2 = NodeName + " = SDOperand(";
        } else {
          NodeName = "ResNode";
          emitDecl(NodeName, true);
          Code2 = NodeName + " = ";
        }
        Code = "CurDAG->getTargetNode(Opc" + utostr(OpcNo);
        emitOpcode(II.Namespace + "::" + II.TheDef->getName());

        // Output order: results, chain, flags
        // Result types.
        if (NumResults > 0 && N->getTypeNum(0) != MVT::isVoid) {
          Code += ", VT" + utostr(VTNo);
          emitVT(getEnumName(N->getTypeNum(0)));
        }
        if (NodeHasChain)
          Code += ", MVT::Other";
        if (NodeHasOutFlag)
          Code += ", MVT::Flag";

        // Inputs.
        for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
          if (HasVarOps)
            emitCode("Ops.push_back(Tmp" + utostr(Ops[i]) + ");");
          else
            Code += ", Tmp" + utostr(Ops[i]);
        }

        if (HasVarOps) {
          if (NodeHasInFlag || HasImpInputs)
            emitCode("for (unsigned i = 2, e = N.getNumOperands()-1; "
                     "i != e; ++i) {");
          else if (NodeHasOptInFlag) 
            emitCode("for (unsigned i = 2, e = N.getNumOperands()-"
                     "(HasInFlag?1:0); i != e; ++i) {");
          else
            emitCode("for (unsigned i = 2, e = N.getNumOperands(); "
                     "i != e; ++i) {");
          emitCode("  SDOperand VarOp(0, 0);");
          emitCode("  AddToQueue(VarOp, N.getOperand(i));");
          emitCode("  Ops.push_back(VarOp);");
          emitCode("}");
        }

        if (NodeHasChain) {
          if (HasVarOps)
            emitCode("Ops.push_back(" + ChainName + ");");
          else
            Code += ", " + ChainName;
        }
        if (NodeHasInFlag || HasImpInputs) {
          if (HasVarOps)
            emitCode("Ops.push_back(InFlag);");
          else
            Code += ", InFlag";
        } else if (NodeHasOptInFlag && HasVarOps) {
          emitCode("if (HasInFlag)");
          emitCode("  Ops.push_back(InFlag);");
        }

        if (HasVarOps)
          Code += ", &Ops[0], Ops.size()";
        else if (NodeHasOptInFlag)
          Code = "HasInFlag ? " + Code + ", InFlag) : " + Code;

        if (!isRoot)
          Code += "), 0";
        emitCode(Code2 + Code + ");");

        if (NodeHasChain)
          // Remember which op produces the chain.
          if (!isRoot)
            emitCode(ChainName + " = SDOperand(" + NodeName +
                     ".Val, " + utostr(PatResults) + ");");
          else
            emitCode(ChainName + " = SDOperand(" + NodeName +
                     ", " + utostr(PatResults) + ");");

        if (!isRoot)
          return std::make_pair(1, ResNo);

        for (unsigned i = 0; i < NumResults; i++)
          emitCode("ReplaceUses(SDOperand(N.Val, " +
                   utostr(i) + "), SDOperand(ResNode, " + utostr(i) + "));");

        if (NodeHasOutFlag)
          emitCode("InFlag = SDOperand(ResNode, " + 
                   utostr(NumResults + (unsigned)NodeHasChain) + ");");

        if (HasImpResults && EmitCopyFromRegs(N, ChainEmitted)) {
          emitCode("ReplaceUses(SDOperand(N.Val, 0), SDOperand(ResNode, 0));");
          NumResults = 1;
        }

        if (InputHasChain)
          emitCode("ReplaceUses(SDOperand(N.Val, " + 
                   utostr(PatResults) + "), SDOperand(" + ChainName + ".Val, " +
                   ChainName + ".ResNo" + "));");

        if (FoldedChains.size() > 0) {
          std::string Code;
          for (unsigned j = 0, e = FoldedChains.size(); j < e; j++)
            emitCode("ReplaceUses(SDOperand(" +
                     FoldedChains[j].first + ".Val, " + 
                     utostr(FoldedChains[j].second) + "), SDOperand(ResNode, " +
                     utostr(NumResults) + "));");
        }

        if (NodeHasOutFlag)
          emitCode("ReplaceUses(SDOperand(N.Val, " +
                   utostr(PatResults + (unsigned)InputHasChain) +"), InFlag);");

        // User does not expect the instruction would produce a chain!
        bool AddedChain = NodeHasChain && !InputHasChain;
        if (AddedChain && NodeHasOutFlag) {
          if (PatResults == 0) {
            emitCode("Result = SDOperand(ResNode, N.ResNo+1);");
          } else {
            emitCode("if (N.ResNo < " + utostr(PatResults) + ")");
            emitCode("  Result = SDOperand(ResNode, N.ResNo);");
            emitCode("else");
            emitCode("  Result = SDOperand(ResNode, N.ResNo+1);");
          }
        } else if (InputHasChain && !NodeHasChain) {
          // One of the inner node produces a chain.
          emitCode("if (N.ResNo < " + utostr(PatResults) + ")");
          emitCode("  Result = SDOperand(ResNode, N.ResNo);");
          if (NodeHasOutFlag) {
            emitCode("else if (N.ResNo > " + utostr(PatResults) + ")");
            emitCode("  Result = SDOperand(ResNode, N.ResNo-1);");
          }
          emitCode("else");
          emitCode("  Result = SDOperand(" + ChainName + ".Val, " +
                   ChainName + ".ResNo);");
        } else {
          emitCode("Result = SDOperand(ResNode, N.ResNo);");
        }
      } else {
        // If this instruction is the root, and if there is only one use of it,
        // use SelectNodeTo instead of getTargetNode to avoid an allocation.
        emitCode("if (N.Val->hasOneUse()) {");
        std::string Code = "  Result = CurDAG->SelectNodeTo(N.Val, Opc" +
          utostr(OpcNo);
        if (N->getTypeNum(0) != MVT::isVoid)
          Code += ", VT" + utostr(VTNo);
        if (NodeHasOutFlag)
          Code += ", MVT::Flag";
        for (unsigned i = 0, e = Ops.size(); i != e; ++i)
          Code += ", Tmp" + utostr(Ops[i]);
        if (NodeHasInFlag || HasImpInputs)
          Code += ", InFlag";
        emitCode(Code + ");");
        emitCode("} else {");
        emitDecl("ResNode", 1);
        Code = "  ResNode = CurDAG->getTargetNode(Opc" + utostr(OpcNo);
        emitOpcode(II.Namespace + "::" + II.TheDef->getName());
        if (N->getTypeNum(0) != MVT::isVoid) {
          Code += ", VT" + utostr(VTNo);
          emitVT(getEnumName(N->getTypeNum(0)));
        }
        if (NodeHasOutFlag)
          Code += ", MVT::Flag";
        for (unsigned i = 0, e = Ops.size(); i != e; ++i)
          Code += ", Tmp" + utostr(Ops[i]);
        if (NodeHasInFlag || HasImpInputs)
          Code += ", InFlag";
        emitCode(Code + ");");
        emitCode("  ReplaceUses(N, SDOperand(ResNode, 0));");
        emitCode("  Result = SDOperand(ResNode, 0);");
        emitCode("}");
      }

      if (isRoot)
        emitCode("return;");
      return std::make_pair(1, ResNo);
    } else if (Op->isSubClassOf("SDNodeXForm")) {
      assert(N->getNumChildren() == 1 && "node xform should have one child!");
      // PatLeaf node - the operand may or may not be a leaf node. But it should
      // behave like one.
      unsigned OpVal = EmitResultCode(N->getChild(0), true).second;
      unsigned ResNo = TmpNo++;
      emitDecl("Tmp" + utostr(ResNo));
      emitCode("Tmp" + utostr(ResNo) + " = Transform_" + Op->getName()
               + "(Tmp" + utostr(OpVal) + ".Val);");
      if (isRoot) {
        emitCode("ReplaceUses(N, Tmp" + utostr(ResNo) + ");");
        emitCode("Result = Tmp" + utostr(ResNo) + ";");
        emitCode("return;");
      }
      return std::make_pair(1, ResNo);
    } else {
      N->dump();
      std::cerr << "\n";
      throw std::string("Unknown node in result pattern!");
    }
  }

  /// InsertOneTypeCheck - Insert a type-check for an unresolved type in 'Pat'
  /// and add it to the tree. 'Pat' and 'Other' are isomorphic trees except that 
  /// 'Pat' may be missing types.  If we find an unresolved type to add a check
  /// for, this returns true otherwise false if Pat has all types.
  bool InsertOneTypeCheck(TreePatternNode *Pat, TreePatternNode *Other,
                          const std::string &Prefix, bool isRoot = false) {
    // Did we find one?
    if (Pat->getExtTypes() != Other->getExtTypes()) {
      // Move a type over from 'other' to 'pat'.
      Pat->setTypes(Other->getExtTypes());
      // The top level node type is checked outside of the select function.
      if (!isRoot)
        emitCheck(Prefix + ".Val->getValueType(0) == " +
                  getName(Pat->getTypeNum(0)));
      return true;
    }
  
    unsigned OpNo =
      (unsigned) NodeHasProperty(Pat, SDNodeInfo::SDNPHasChain, ISE);
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i, ++OpNo)
      if (InsertOneTypeCheck(Pat->getChild(i), Other->getChild(i),
                             Prefix + utostr(OpNo)))
        return true;
    return false;
  }

private:
  /// EmitInFlagSelectCode - Emit the flag operands for the DAG that is
  /// being built.
  void EmitInFlagSelectCode(TreePatternNode *N, const std::string &RootName,
                            bool &ChainEmitted, bool isRoot = false) {
    const CodeGenTarget &T = ISE.getTargetInfo();
    unsigned OpNo =
      (unsigned) NodeHasProperty(N, SDNodeInfo::SDNPHasChain, ISE);
    bool HasInFlag = NodeHasProperty(N, SDNodeInfo::SDNPInFlag, ISE);
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i, ++OpNo) {
      TreePatternNode *Child = N->getChild(i);
      if (!Child->isLeaf()) {
        EmitInFlagSelectCode(Child, RootName + utostr(OpNo), ChainEmitted);
      } else {
        if (DefInit *DI = dynamic_cast<DefInit*>(Child->getLeafValue())) {
          if (!Child->getName().empty()) {
            std::string Name = RootName + utostr(OpNo);
            if (Duplicates.find(Name) != Duplicates.end())
              // A duplicate! Do not emit a copy for this node.
              continue;
          }

          Record *RR = DI->getDef();
          if (RR->isSubClassOf("Register")) {
            MVT::ValueType RVT = getRegisterValueType(RR, T);
            if (RVT == MVT::Flag) {
              emitCode("AddToQueue(InFlag, " + RootName + utostr(OpNo) + ");");
            } else {
              if (!ChainEmitted) {
                emitDecl("Chain");
                emitCode("Chain = CurDAG->getEntryNode();");
                ChainName = "Chain";
                ChainEmitted = true;
              }
              emitCode("AddToQueue(" + RootName + utostr(OpNo) + ", " +
                       RootName + utostr(OpNo) + ");");
              emitCode("ResNode = CurDAG->getCopyToReg(" + ChainName +
                       ", CurDAG->getRegister(" + ISE.getQualifiedName(RR) +
                       ", " + getEnumName(RVT) + "), " +
                       RootName + utostr(OpNo) + ", InFlag).Val;");
              emitCode(ChainName + " = SDOperand(ResNode, 0);");
              emitCode("InFlag = SDOperand(ResNode, 1);");
            }
          }
        }
      }
    }

    if (HasInFlag)
      emitCode("AddToQueue(InFlag, " + RootName +
               ".getOperand(" + utostr(OpNo) + "));");
  }

  /// EmitCopyFromRegs - Emit code to copy result to physical registers
  /// as specified by the instruction. It returns true if any copy is
  /// emitted.
  bool EmitCopyFromRegs(TreePatternNode *N, bool &ChainEmitted) {
    bool RetVal = false;
    Record *Op = N->getOperator();
    if (Op->isSubClassOf("Instruction")) {
      const DAGInstruction &Inst = ISE.getInstruction(Op);
      const CodeGenTarget &CGT = ISE.getTargetInfo();
      unsigned NumImpResults  = Inst.getNumImpResults();
      for (unsigned i = 0; i < NumImpResults; i++) {
        Record *RR = Inst.getImpResult(i);
        if (RR->isSubClassOf("Register")) {
          MVT::ValueType RVT = getRegisterValueType(RR, CGT);
          if (RVT != MVT::Flag) {
            if (!ChainEmitted) {
              emitDecl("Chain");
              emitCode("Chain = CurDAG->getEntryNode();");
              ChainEmitted = true;
              ChainName = "Chain";
            }
            emitCode("ResNode = CurDAG->getCopyFromReg(" + ChainName +
                     ", " + ISE.getQualifiedName(RR) + ", " + getEnumName(RVT) +
                     ", InFlag).Val;");
            emitCode(ChainName + " = SDOperand(ResNode, 1);");
            emitCode("InFlag = SDOperand(ResNode, 2);");
            RetVal = true;
          }
        }
      }
    }
    return RetVal;
  }
};

/// EmitCodeForPattern - Given a pattern to match, emit code to the specified
/// stream to match the pattern, and generate the code for the match if it
/// succeeds.  Returns true if the pattern is not guaranteed to match.
void DAGISelEmitter::GenerateCodeForPattern(PatternToMatch &Pattern,
                      std::vector<std::pair<bool, std::string> > &GeneratedCode,
                         std::set<std::pair<unsigned, std::string> > &GeneratedDecl,
                                        std::vector<std::string> &TargetOpcodes,
                                            std::vector<std::string> &TargetVTs) {
  PatternCodeEmitter Emitter(*this, Pattern.getPredicates(),
                             Pattern.getSrcPattern(), Pattern.getDstPattern(),
                             GeneratedCode, GeneratedDecl,
                             TargetOpcodes, TargetVTs);

  // Emit the matcher, capturing named arguments in VariableMap.
  bool FoundChain = false;
  Emitter.EmitMatchCode(Pattern.getSrcPattern(), NULL, "N", "", "", FoundChain);

  // TP - Get *SOME* tree pattern, we don't care which.
  TreePattern &TP = *PatternFragments.begin()->second;
  
  // At this point, we know that we structurally match the pattern, but the
  // types of the nodes may not match.  Figure out the fewest number of type 
  // comparisons we need to emit.  For example, if there is only one integer
  // type supported by a target, there should be no type comparisons at all for
  // integer patterns!
  //
  // To figure out the fewest number of type checks needed, clone the pattern,
  // remove the types, then perform type inference on the pattern as a whole.
  // If there are unresolved types, emit an explicit check for those types,
  // apply the type to the tree, then rerun type inference.  Iterate until all
  // types are resolved.
  //
  TreePatternNode *Pat = Pattern.getSrcPattern()->clone();
  RemoveAllTypes(Pat);
  
  do {
    // Resolve/propagate as many types as possible.
    try {
      bool MadeChange = true;
      while (MadeChange)
        MadeChange = Pat->ApplyTypeConstraints(TP,
                                               true/*Ignore reg constraints*/);
    } catch (...) {
      assert(0 && "Error: could not find consistent types for something we"
             " already decided was ok!");
      abort();
    }

    // Insert a check for an unresolved type and add it to the tree.  If we find
    // an unresolved type to add a check for, this returns true and we iterate,
    // otherwise we are done.
  } while (Emitter.InsertOneTypeCheck(Pat, Pattern.getSrcPattern(), "N", true));

  Emitter.EmitResultCode(Pattern.getDstPattern(), false, true /*the root*/);
  delete Pat;
}

/// EraseCodeLine - Erase one code line from all of the patterns.  If removing
/// a line causes any of them to be empty, remove them and return true when
/// done.
static bool EraseCodeLine(std::vector<std::pair<PatternToMatch*, 
                          std::vector<std::pair<bool, std::string> > > >
                          &Patterns) {
  bool ErasedPatterns = false;
  for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
    Patterns[i].second.pop_back();
    if (Patterns[i].second.empty()) {
      Patterns.erase(Patterns.begin()+i);
      --i; --e;
      ErasedPatterns = true;
    }
  }
  return ErasedPatterns;
}

/// EmitPatterns - Emit code for at least one pattern, but try to group common
/// code together between the patterns.
void DAGISelEmitter::EmitPatterns(std::vector<std::pair<PatternToMatch*, 
                                  std::vector<std::pair<bool, std::string> > > >
                                  &Patterns, unsigned Indent,
                                  std::ostream &OS) {
  typedef std::pair<bool, std::string> CodeLine;
  typedef std::vector<CodeLine> CodeList;
  typedef std::vector<std::pair<PatternToMatch*, CodeList> > PatternList;
  
  if (Patterns.empty()) return;
  
  // Figure out how many patterns share the next code line.  Explicitly copy
  // FirstCodeLine so that we don't invalidate a reference when changing
  // Patterns.
  const CodeLine FirstCodeLine = Patterns.back().second.back();
  unsigned LastMatch = Patterns.size()-1;
  while (LastMatch != 0 && Patterns[LastMatch-1].second.back() == FirstCodeLine)
    --LastMatch;
  
  // If not all patterns share this line, split the list into two pieces.  The
  // first chunk will use this line, the second chunk won't.
  if (LastMatch != 0) {
    PatternList Shared(Patterns.begin()+LastMatch, Patterns.end());
    PatternList Other(Patterns.begin(), Patterns.begin()+LastMatch);
    
    // FIXME: Emit braces?
    if (Shared.size() == 1) {
      PatternToMatch &Pattern = *Shared.back().first;
      OS << "\n" << std::string(Indent, ' ') << "// Pattern: ";
      Pattern.getSrcPattern()->print(OS);
      OS << "\n" << std::string(Indent, ' ') << "// Emits: ";
      Pattern.getDstPattern()->print(OS);
      OS << "\n";
      unsigned AddedComplexity = Pattern.getAddedComplexity();
      OS << std::string(Indent, ' ') << "// Pattern complexity = "
         << getPatternSize(Pattern.getSrcPattern(), *this) + AddedComplexity
         << "  cost = "
         << getResultPatternCost(Pattern.getDstPattern(), *this)
         << "  size = "
         << getResultPatternSize(Pattern.getDstPattern(), *this) << "\n";
    }
    if (!FirstCodeLine.first) {
      OS << std::string(Indent, ' ') << "{\n";
      Indent += 2;
    }
    EmitPatterns(Shared, Indent, OS);
    if (!FirstCodeLine.first) {
      Indent -= 2;
      OS << std::string(Indent, ' ') << "}\n";
    }
    
    if (Other.size() == 1) {
      PatternToMatch &Pattern = *Other.back().first;
      OS << "\n" << std::string(Indent, ' ') << "// Pattern: ";
      Pattern.getSrcPattern()->print(OS);
      OS << "\n" << std::string(Indent, ' ') << "// Emits: ";
      Pattern.getDstPattern()->print(OS);
      OS << "\n";
      unsigned AddedComplexity = Pattern.getAddedComplexity();
      OS << std::string(Indent, ' ') << "// Pattern complexity = "
         << getPatternSize(Pattern.getSrcPattern(), *this) + AddedComplexity
         << "  cost = "
         << getResultPatternCost(Pattern.getDstPattern(), *this)
         << "  size = "
         << getResultPatternSize(Pattern.getDstPattern(), *this) << "\n";
    }
    EmitPatterns(Other, Indent, OS);
    return;
  }
  
  // Remove this code from all of the patterns that share it.
  bool ErasedPatterns = EraseCodeLine(Patterns);
  
  bool isPredicate = FirstCodeLine.first;
  
  // Otherwise, every pattern in the list has this line.  Emit it.
  if (!isPredicate) {
    // Normal code.
    OS << std::string(Indent, ' ') << FirstCodeLine.second << "\n";
  } else {
    OS << std::string(Indent, ' ') << "if (" << FirstCodeLine.second;
    
    // If the next code line is another predicate, and if all of the pattern
    // in this group share the same next line, emit it inline now.  Do this
    // until we run out of common predicates.
    while (!ErasedPatterns && Patterns.back().second.back().first) {
      // Check that all of fhe patterns in Patterns end with the same predicate.
      bool AllEndWithSamePredicate = true;
      for (unsigned i = 0, e = Patterns.size(); i != e; ++i)
        if (Patterns[i].second.back() != Patterns.back().second.back()) {
          AllEndWithSamePredicate = false;
          break;
        }
      // If all of the predicates aren't the same, we can't share them.
      if (!AllEndWithSamePredicate) break;
      
      // Otherwise we can.  Emit it shared now.
      OS << " &&\n" << std::string(Indent+4, ' ')
         << Patterns.back().second.back().second;
      ErasedPatterns = EraseCodeLine(Patterns);
    }
    
    OS << ") {\n";
    Indent += 2;
  }
  
  EmitPatterns(Patterns, Indent, OS);
  
  if (isPredicate)
    OS << std::string(Indent-2, ' ') << "}\n";
}



namespace {
  /// CompareByRecordName - An ordering predicate that implements less-than by
  /// comparing the names records.
  struct CompareByRecordName {
    bool operator()(const Record *LHS, const Record *RHS) const {
      // Sort by name first.
      if (LHS->getName() < RHS->getName()) return true;
      // If both names are equal, sort by pointer.
      return LHS->getName() == RHS->getName() && LHS < RHS;
    }
  };
}

void DAGISelEmitter::EmitInstructionSelector(std::ostream &OS) {
  std::string InstNS = Target.inst_begin()->second.Namespace;
  if (!InstNS.empty()) InstNS += "::";
  
  // Group the patterns by their top-level opcodes.
  std::map<Record*, std::vector<PatternToMatch*>,
    CompareByRecordName> PatternsByOpcode;
  // All unique target node emission functions.
  std::map<std::string, unsigned> EmitFunctions;
  for (unsigned i = 0, e = PatternsToMatch.size(); i != e; ++i) {
    TreePatternNode *Node = PatternsToMatch[i].getSrcPattern();
    if (!Node->isLeaf()) {
      PatternsByOpcode[Node->getOperator()].push_back(&PatternsToMatch[i]);
    } else {
      const ComplexPattern *CP;
      if (IntInit *II = 
          dynamic_cast<IntInit*>(Node->getLeafValue())) {
        PatternsByOpcode[getSDNodeNamed("imm")].push_back(&PatternsToMatch[i]);
      } else if ((CP = NodeGetComplexPattern(Node, *this))) {
        std::vector<Record*> OpNodes = CP->getRootNodes();
        for (unsigned j = 0, e = OpNodes.size(); j != e; j++) {
          PatternsByOpcode[OpNodes[j]]
            .insert(PatternsByOpcode[OpNodes[j]].begin(), &PatternsToMatch[i]);
        }
      } else {
        std::cerr << "Unrecognized opcode '";
        Node->dump();
        std::cerr << "' on tree pattern '";
        std::cerr << 
           PatternsToMatch[i].getDstPattern()->getOperator()->getName();
        std::cerr << "'!\n";
        exit(1);
      }
    }
  }

  // For each opcode, there might be multiple select functions, one per
  // ValueType of the node (or its first operand if it doesn't produce a
  // non-chain result.
  std::map<std::string, std::vector<std::string> > OpcodeVTMap;

  // Emit one Select_* method for each top-level opcode.  We do this instead of
  // emitting one giant switch statement to support compilers where this will
  // result in the recursive functions taking less stack space.
  for (std::map<Record*, std::vector<PatternToMatch*>,
       CompareByRecordName>::iterator PBOI = PatternsByOpcode.begin(),
       E = PatternsByOpcode.end(); PBOI != E; ++PBOI) {
    const std::string &OpName = PBOI->first->getName();
    const SDNodeInfo &OpcodeInfo = getSDNodeInfo(PBOI->first);
    std::vector<PatternToMatch*> &PatternsOfOp = PBOI->second;
    assert(!PatternsOfOp.empty() && "No patterns but map has entry?");

    // We want to emit all of the matching code now.  However, we want to emit
    // the matches in order of minimal cost.  Sort the patterns so the least
    // cost one is at the start.
    std::stable_sort(PatternsOfOp.begin(), PatternsOfOp.end(),
                     PatternSortingPredicate(*this));

    // Split them into groups by type.
    std::map<MVT::ValueType, std::vector<PatternToMatch*> > PatternsByType;
    for (unsigned i = 0, e = PatternsOfOp.size(); i != e; ++i) {
      PatternToMatch *Pat = PatternsOfOp[i];
      TreePatternNode *SrcPat = Pat->getSrcPattern();
      if (OpcodeInfo.getNumResults() == 0 && SrcPat->getNumChildren() > 0)
        SrcPat = SrcPat->getChild(0);
      MVT::ValueType VT = SrcPat->getTypeNum(0);
      std::map<MVT::ValueType, std::vector<PatternToMatch*> >::iterator TI = 
        PatternsByType.find(VT);
      if (TI != PatternsByType.end())
        TI->second.push_back(Pat);
      else {
        std::vector<PatternToMatch*> PVec;
        PVec.push_back(Pat);
        PatternsByType.insert(std::make_pair(VT, PVec));
      }
    }

    for (std::map<MVT::ValueType, std::vector<PatternToMatch*> >::iterator
           II = PatternsByType.begin(), EE = PatternsByType.end(); II != EE;
         ++II) {
      MVT::ValueType OpVT = II->first;
      std::vector<PatternToMatch*> &Patterns = II->second;
      typedef std::vector<std::pair<bool, std::string> > CodeList;
      typedef std::vector<std::pair<bool, std::string> >::iterator CodeListI;
    
      std::vector<std::pair<PatternToMatch*, CodeList> > CodeForPatterns;
      std::vector<std::vector<std::string> > PatternOpcodes;
      std::vector<std::vector<std::string> > PatternVTs;
      std::vector<std::set<std::pair<unsigned, std::string> > > PatternDecls;
      std::set<std::pair<unsigned, std::string> > AllGenDecls;
      for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
        CodeList GeneratedCode;
        std::set<std::pair<unsigned, std::string> > GeneratedDecl;
        std::vector<std::string> TargetOpcodes;
        std::vector<std::string> TargetVTs;
        GenerateCodeForPattern(*Patterns[i], GeneratedCode, GeneratedDecl,
                               TargetOpcodes, TargetVTs);
        for (std::set<std::pair<unsigned, std::string> >::iterator
               si = GeneratedDecl.begin(), se = GeneratedDecl.end(); si!=se; ++si)
          AllGenDecls.insert(*si);
        CodeForPatterns.push_back(std::make_pair(Patterns[i], GeneratedCode));
        PatternDecls.push_back(GeneratedDecl);
        PatternOpcodes.push_back(TargetOpcodes);
        PatternVTs.push_back(TargetVTs);
      }
    
      // Scan the code to see if all of the patterns are reachable and if it is
      // possible that the last one might not match.
      bool mightNotMatch = true;
      for (unsigned i = 0, e = CodeForPatterns.size(); i != e; ++i) {
        CodeList &GeneratedCode = CodeForPatterns[i].second;
        mightNotMatch = false;

        for (unsigned j = 0, e = GeneratedCode.size(); j != e; ++j) {
          if (GeneratedCode[j].first) { // predicate.
            mightNotMatch = true;
            break;
          }
        }
      
        // If this pattern definitely matches, and if it isn't the last one, the
        // patterns after it CANNOT ever match.  Error out.
        if (mightNotMatch == false && i != CodeForPatterns.size()-1) {
          std::cerr << "Pattern '";
          CodeForPatterns[i+1].first->getSrcPattern()->print(std::cerr);
          std::cerr << "' is impossible to select!\n";
          exit(1);
        }
      }

      // Factor target node emission code (emitted by EmitResultCode) into
      // separate functions. Uniquing and share them among all instruction
      // selection routines.
      for (unsigned i = 0, e = CodeForPatterns.size(); i != e; ++i) {
        CodeList &GeneratedCode = CodeForPatterns[i].second;
        std::vector<std::string> &TargetOpcodes = PatternOpcodes[i];
        std::vector<std::string> &TargetVTs = PatternVTs[i];
        std::set<std::pair<unsigned, std::string> > Decls = PatternDecls[i];
        int CodeSize = (int)GeneratedCode.size();
        int LastPred = -1;
        for (int j = CodeSize-1; j >= 0; --j) {
          if (GeneratedCode[j].first) {
            LastPred = j;
            break;
          }
        }

        std::string CalleeDecls;
        std::string CalleeCode = "(SDOperand &Result, const SDOperand &N";
        std::string CallerCode = "(Result, N";
        for (unsigned j = 0, e = TargetOpcodes.size(); j != e; ++j) {
          CalleeCode += ", unsigned Opc" + utostr(j);
          CallerCode += ", " + TargetOpcodes[j];
        }
        for (unsigned j = 0, e = TargetVTs.size(); j != e; ++j) {
          CalleeCode += ", MVT::ValueType VT" + utostr(j);
          CallerCode += ", " + TargetVTs[j];
        }
        for (std::set<std::pair<unsigned, std::string> >::iterator
               I = Decls.begin(), E = Decls.end(); I != E; ++I) {
          std::string Name = I->second;
          if (I->first == 0) {
            if (Name == "InFlag" ||
                (Name.size() > 3 &&
                 Name[0] == 'T' && Name[1] == 'm' && Name[2] == 'p')) {
              CalleeDecls += "  SDOperand " + Name + "(0, 0);\n";
              continue;
            }
            CalleeCode += ", SDOperand &" + Name;
            CallerCode += ", " + Name;
          } else if (I->first == 1) {
            if (Name == "ResNode") {
              CalleeDecls += "  SDNode *" + Name + " = NULL;\n";
              continue;
            }
            CalleeCode += ", SDNode *" + Name;
            CallerCode += ", " + Name;
          } else {
            CalleeCode += ", bool " + Name;
            CallerCode += ", " + Name;
          }
        }
        CallerCode += ");";
        CalleeCode += ") ";
        // Prevent emission routines from being inlined to reduce selection
        // routines stack frame sizes.
        CalleeCode += "NOINLINE ";
        CalleeCode += "{\n" + CalleeDecls;
        for (int j = LastPred+1; j < CodeSize; ++j)
          CalleeCode += "  " + GeneratedCode[j].second + '\n';
        for (int j = LastPred+1; j < CodeSize; ++j)
          GeneratedCode.pop_back();
        CalleeCode += "}\n";

        // Uniquing the emission routines.
        unsigned EmitFuncNum;
        std::map<std::string, unsigned>::iterator EFI =
          EmitFunctions.find(CalleeCode);
        if (EFI != EmitFunctions.end()) {
          EmitFuncNum = EFI->second;
        } else {
          EmitFuncNum = EmitFunctions.size();
          EmitFunctions.insert(std::make_pair(CalleeCode, EmitFuncNum));
          OS << "void " << "Emit_" << utostr(EmitFuncNum) << CalleeCode;
        }

        // Replace the emission code within selection routines with calls to the
        // emission functions.
        CallerCode = "Emit_" + utostr(EmitFuncNum) + CallerCode;
        GeneratedCode.push_back(std::make_pair(false, CallerCode));
        GeneratedCode.push_back(std::make_pair(false, "return;"));
      }

      // Print function.
      std::string OpVTStr = (OpVT != MVT::isVoid && OpVT != MVT::iPTR)
        ? getEnumName(OpVT).substr(5) : "" ;
      std::map<std::string, std::vector<std::string> >::iterator OpVTI =
        OpcodeVTMap.find(OpName);
      if (OpVTI == OpcodeVTMap.end()) {
        std::vector<std::string> VTSet;
        VTSet.push_back(OpVTStr);
        OpcodeVTMap.insert(std::make_pair(OpName, VTSet));
      } else
        OpVTI->second.push_back(OpVTStr);

      OS << "void Select_" << OpName << (OpVTStr != "" ? "_" : "")
         << OpVTStr << "(SDOperand &Result, const SDOperand &N) {\n";    

      // Print all declarations.
      for (std::set<std::pair<unsigned, std::string> >::iterator
             I = AllGenDecls.begin(), E = AllGenDecls.end(); I != E; ++I)
        if (I->first == 0)
          OS << "  SDOperand " << I->second << "(0, 0);\n";
        else if (I->first == 1)
          OS << "  SDNode *" << I->second << " = NULL;\n";
        else
          OS << "  bool " << I->second << " = false;\n";

      // Loop through and reverse all of the CodeList vectors, as we will be
      // accessing them from their logical front, but accessing the end of a
      // vector is more efficient.
      for (unsigned i = 0, e = CodeForPatterns.size(); i != e; ++i) {
        CodeList &GeneratedCode = CodeForPatterns[i].second;
        std::reverse(GeneratedCode.begin(), GeneratedCode.end());
      }
    
      // Next, reverse the list of patterns itself for the same reason.
      std::reverse(CodeForPatterns.begin(), CodeForPatterns.end());
    
      // Emit all of the patterns now, grouped together to share code.
      EmitPatterns(CodeForPatterns, 2, OS);
    
      // If the last pattern has predicates (which could fail) emit code to catch
      // the case where nothing handles a pattern.
      if (mightNotMatch) {
        OS << "  std::cerr << \"Cannot yet select: \";\n";
        if (OpcodeInfo.getEnumName() != "ISD::INTRINSIC_W_CHAIN" &&
            OpcodeInfo.getEnumName() != "ISD::INTRINSIC_WO_CHAIN" &&
            OpcodeInfo.getEnumName() != "ISD::INTRINSIC_VOID") {
          OS << "  N.Val->dump(CurDAG);\n";
        } else {
          OS << "  unsigned iid = cast<ConstantSDNode>(N.getOperand("
            "N.getOperand(0).getValueType() == MVT::Other))->getValue();\n"
             << "  std::cerr << \"intrinsic %\"<< "
            "Intrinsic::getName((Intrinsic::ID)iid);\n";
        }
        OS << "  std::cerr << '\\n';\n"
           << "  abort();\n";
      }
      OS << "}\n\n";
    }
  }
  
  // Emit boilerplate.
  OS << "void Select_INLINEASM(SDOperand& Result, SDOperand N) {\n"
     << "  std::vector<SDOperand> Ops(N.Val->op_begin(), N.Val->op_end());\n"
     << "  AddToQueue(Ops[0], N.getOperand(0)); // Select the chain.\n\n"
     << "  // Select the flag operand.\n"
     << "  if (Ops.back().getValueType() == MVT::Flag)\n"
     << "    AddToQueue(Ops.back(), Ops.back());\n"
     << "  SelectInlineAsmMemoryOperands(Ops, *CurDAG);\n"
     << "  std::vector<MVT::ValueType> VTs;\n"
     << "  VTs.push_back(MVT::Other);\n"
     << "  VTs.push_back(MVT::Flag);\n"
     << "  SDOperand New = CurDAG->getNode(ISD::INLINEASM, VTs, &Ops[0], "
                 "Ops.size());\n"
     << "  ReplaceUses(SDOperand(N.Val, 0), New);\n"
     << "  ReplaceUses(SDOperand(N.Val, 1), SDOperand(New.Val, 1));\n"
     << "  Result = New.getValue(N.ResNo);\n"
     << "  return;\n"
     << "}\n\n";
  
  OS << "// The main instruction selector code.\n"
     << "void SelectCode(SDOperand &Result, SDOperand N) {\n"
     << "  if (N.getOpcode() >= ISD::BUILTIN_OP_END &&\n"
     << "      N.getOpcode() < (ISD::BUILTIN_OP_END+" << InstNS
     << "INSTRUCTION_LIST_END)) {\n"
     << "    Result = N;\n"
     << "    return;   // Already selected.\n"
     << "  }\n\n"
     << "  switch (N.getOpcode()) {\n"
     << "  default: break;\n"
     << "  case ISD::EntryToken:       // These leaves remain the same.\n"
     << "  case ISD::BasicBlock:\n"
     << "  case ISD::Register:\n"
     << "  case ISD::HANDLENODE:\n"
     << "  case ISD::TargetConstant:\n"
     << "  case ISD::TargetConstantPool:\n"
     << "  case ISD::TargetFrameIndex:\n"
     << "  case ISD::TargetJumpTable:\n"
     << "  case ISD::TargetGlobalAddress: {\n"
     << "    Result = N;\n"
     << "    return;\n"
     << "  }\n"
     << "  case ISD::AssertSext:\n"
     << "  case ISD::AssertZext: {\n"
     << "    AddToQueue(Result, N.getOperand(0));\n"
     << "    ReplaceUses(N, Result);\n"
     << "    return;\n"
     << "  }\n"
     << "  case ISD::TokenFactor:\n"
     << "  case ISD::CopyFromReg:\n"
     << "  case ISD::CopyToReg: {\n"
     << "    for (unsigned i = 0, e = N.getNumOperands(); i != e; ++i) {\n"
     << "      SDOperand Dummy;\n"
     << "      AddToQueue(Dummy, N.getOperand(i));\n"
     << "    }\n"
     << "    Result = N;\n"
     << "    return;\n"
     << "  }\n"
     << "  case ISD::INLINEASM:  Select_INLINEASM(Result, N); return;\n";

    
  // Loop over all of the case statements, emiting a call to each method we
  // emitted above.
  for (std::map<Record*, std::vector<PatternToMatch*>,
                CompareByRecordName>::iterator PBOI = PatternsByOpcode.begin(),
       E = PatternsByOpcode.end(); PBOI != E; ++PBOI) {
    const SDNodeInfo &OpcodeInfo = getSDNodeInfo(PBOI->first);
    const std::string &OpName = PBOI->first->getName();
    // Potentially multiple versions of select for this opcode. One for each
    // ValueType of the node (or its first true operand if it doesn't produce a
    // result.
    std::map<std::string, std::vector<std::string> >::iterator OpVTI =
      OpcodeVTMap.find(OpName);
    std::vector<std::string> &OpVTs = OpVTI->second;
    OS << "  case " << OpcodeInfo.getEnumName() << ": {\n";
    if (OpVTs.size() == 1) {
      std::string &VTStr = OpVTs[0];
      OS << "    Select_" << OpName
         << (VTStr != "" ? "_" : "") << VTStr << "(Result, N);\n";
    } else {
      if (OpcodeInfo.getNumResults())
        OS << "    MVT::ValueType NVT = N.Val->getValueType(0);\n";
      else if (OpcodeInfo.hasProperty(SDNodeInfo::SDNPHasChain))
        OS << "    MVT::ValueType NVT = (N.getNumOperands() > 1) ?"
           << " N.getOperand(1).Val->getValueType(0) : MVT::isVoid;\n";
      else
        OS << "    MVT::ValueType NVT = (N.getNumOperands() > 0) ?"
           << " N.getOperand(0).Val->getValueType(0) : MVT::isVoid;\n";
      int ElseCase = -1;
      bool First = true;
      for (unsigned i = 0, e = OpVTs.size(); i < e; ++i) {
        std::string &VTStr = OpVTs[i];
        if (VTStr == "") {
          ElseCase = i;
          continue;
        }
        OS << (First ? "    if" : "    else if")
           << " (NVT == MVT::" << VTStr << ")\n"
           << "      Select_" << OpName
           << "_" << VTStr << "(Result, N);\n";
        First = false;
      }
      if (ElseCase != -1)
        OS << "    else\n" << "      Select_" << OpName << "(Result, N);\n";
      else
        OS << "    else\n" << "      break;\n";
    }
    OS << "    return;\n";
    OS << "  }\n";
  }

  OS << "  } // end of big switch.\n\n"
     << "  std::cerr << \"Cannot yet select: \";\n"
     << "  if (N.getOpcode() != ISD::INTRINSIC_W_CHAIN &&\n"
     << "      N.getOpcode() != ISD::INTRINSIC_WO_CHAIN &&\n"
     << "      N.getOpcode() != ISD::INTRINSIC_VOID) {\n"
     << "    N.Val->dump(CurDAG);\n"
     << "  } else {\n"
     << "    unsigned iid = cast<ConstantSDNode>(N.getOperand("
               "N.getOperand(0).getValueType() == MVT::Other))->getValue();\n"
     << "    std::cerr << \"intrinsic %\"<< "
                        "Intrinsic::getName((Intrinsic::ID)iid);\n"
     << "  }\n"
     << "  std::cerr << '\\n';\n"
     << "  abort();\n"
     << "}\n";
}

void DAGISelEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("DAG Instruction Selector for the " + Target.getName() +
                       " target", OS);
  
  OS << "// *** NOTE: This file is #included into the middle of the target\n"
     << "// *** instruction selector class.  These functions are really "
     << "methods.\n\n";
  
  OS << "#if defined(__GNUC__) && \\\n";
  OS << "    ((__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4)))\n";
  OS << "#define NOINLINE __attribute__((noinline))\n";
  OS << "#else\n";
  OS << "#define NOINLINE\n";
  OS << "#endif\n\n";

  OS << "// Instruction selector priority queue:\n"
     << "std::vector<SDNode*> ISelQueue;\n";
  OS << "/// Keep track of nodes which have already been added to queue.\n"
     << "unsigned char *ISelQueued;\n";
  OS << "/// Keep track of nodes which have already been selected.\n"
     << "unsigned char *ISelSelected;\n";
  OS << "/// Dummy parameter to ReplaceAllUsesOfValueWith().\n"
     << "std::vector<SDNode*> ISelKilled;\n\n";

  OS << "/// Sorting functions for the selection queue.\n"
     << "struct isel_sort : public std::binary_function"
     << "<SDNode*, SDNode*, bool> {\n"
     << "  bool operator()(const SDNode* left, const SDNode* right) "
     << "const {\n"
     << "    return (left->getNodeId() > right->getNodeId());\n"
     << "  }\n"
     << "};\n\n";

  OS << "inline void setQueued(int Id) {\n";
  OS << "  ISelQueued[Id / 8] |= 1 << (Id % 8);\n";
  OS << "}\n";
  OS << "inline bool isQueued(int Id) {\n";
  OS << "  return ISelQueued[Id / 8] & (1 << (Id % 8));\n";
  OS << "}\n";
  OS << "inline void setSelected(int Id) {\n";
  OS << "  ISelSelected[Id / 8] |= 1 << (Id % 8);\n";
  OS << "}\n";
  OS << "inline bool isSelected(int Id) {\n";
  OS << "  return ISelSelected[Id / 8] & (1 << (Id % 8));\n";
  OS << "}\n\n";

  OS << "inline void AddToQueue(SDOperand &Result, SDOperand N) {\n";
  OS << "  Result = N;\n";
  OS << "  int Id = N.Val->getNodeId();\n";
  OS << "  if (Id != -1 && !isQueued(Id)) {\n";
  OS << "    ISelQueue.push_back(N.Val);\n";
 OS << "    std::push_heap(ISelQueue.begin(), ISelQueue.end(), isel_sort());\n";
  OS << "    setQueued(Id);\n";
  OS << "  }\n";
  OS << "}\n\n";

  OS << "inline void RemoveKilled() {\n";
OS << "  unsigned NumKilled = ISelKilled.size();\n";
  OS << "  if (NumKilled) {\n";
  OS << "    for (unsigned i = 0; i != NumKilled; ++i) {\n";
  OS << "      SDNode *Temp = ISelKilled[i];\n";
  OS << "      std::remove(ISelQueue.begin(), ISelQueue.end(), Temp);\n";
  OS << "    };\n";
 OS << "    std::make_heap(ISelQueue.begin(), ISelQueue.end(), isel_sort());\n";
  OS << "    ISelKilled.clear();\n";
  OS << "  }\n";
  OS << "}\n\n";

  OS << "inline void ReplaceUses(SDOperand F, SDOperand T) {\n";
  OS << "  CurDAG->ReplaceAllUsesOfValueWith(F, T, ISelKilled);\n";
  OS << "  setSelected(F.Val->getNodeId());\n";
  OS << "  RemoveKilled();\n";
  OS << "}\n\n";

  OS << "// SelectRoot - Top level entry to DAG isel.\n";
  OS << "SDOperand SelectRoot(SDOperand Root) {\n";
  OS << "  SelectRootInit();\n";
  OS << "  unsigned NumBytes = (DAGSize + 7) / 8;\n";
  OS << "  ISelQueued   = new unsigned char[NumBytes];\n";
  OS << "  ISelSelected = new unsigned char[NumBytes];\n";
  OS << "  memset(ISelQueued,   0, NumBytes);\n";
  OS << "  memset(ISelSelected, 0, NumBytes);\n";
  OS << "\n";
  OS << "  SDOperand ResNode;\n";
  OS << "  Select(ResNode, Root);\n";
  OS << "  while (!ISelQueue.empty()) {\n";
  OS << "    SDOperand Tmp;\n";
  OS << "    SDNode *Node = ISelQueue.front();\n";
  OS << "    std::pop_heap(ISelQueue.begin(), ISelQueue.end(), isel_sort());\n";
  OS << "    ISelQueue.pop_back();\n";
  OS << "    if (!isSelected(Node->getNodeId()))\n";
  OS << "      Select(Tmp, SDOperand(Node, 0));\n";
  OS << "  }\n";
  OS << "\n";
  OS << "  delete[] ISelQueued;\n";
  OS << "  ISelQueued = NULL;\n";
  OS << "  delete[] ISelSelected;\n";
  OS << "  ISelSelected = NULL;\n";
  OS << "  return ResNode;\n";
  OS << "}\n";
  
  Intrinsics = LoadIntrinsics(Records);
  ParseNodeInfo();
  ParseNodeTransforms(OS);
  ParseComplexPatterns();
  ParsePatternFragments(OS);
  ParseInstructions();
  ParsePatterns();
  
  // Generate variants.  For example, commutative patterns can match
  // multiple ways.  Add them to PatternsToMatch as well.
  GenerateVariants();

  
  DEBUG(std::cerr << "\n\nALL PATTERNS TO MATCH:\n\n";
        for (unsigned i = 0, e = PatternsToMatch.size(); i != e; ++i) {
          std::cerr << "PATTERN: ";  PatternsToMatch[i].getSrcPattern()->dump();
          std::cerr << "\nRESULT:  ";PatternsToMatch[i].getDstPattern()->dump();
          std::cerr << "\n";
        });
  
  // At this point, we have full information about the 'Patterns' we need to
  // parse, both implicitly from instructions as well as from explicit pattern
  // definitions.  Emit the resultant instruction selector.
  EmitInstructionSelector(OS);  
  
  for (std::map<Record*, TreePattern*>::iterator I = PatternFragments.begin(),
       E = PatternFragments.end(); I != E; ++I)
    delete I->second;
  PatternFragments.clear();

  Instructions.clear();
}
