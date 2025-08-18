//===- DAGISelMatcher.cpp - Representation of DAG pattern matcher ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAGISelMatcher.h"
#include "CodeGenDAGPatterns.h"
#include "CodeGenInstruction.h"
#include "CodeGenRegisters.h"
#include "CodeGenTarget.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
using namespace llvm;

void Matcher::anchor() {}

void Matcher::dump() const { print(errs()); }

void Matcher::print(raw_ostream &OS, indent Indent) const {
  printImpl(OS, Indent);
  if (Next)
    return Next->print(OS, Indent);
}

void Matcher::printOne(raw_ostream &OS) const { printImpl(OS, indent(0)); }

/// unlinkNode - Unlink the specified node from this chain.  If Other == this,
/// we unlink the next pointer and return it.  Otherwise we unlink Other from
/// the list and return this.
Matcher *Matcher::unlinkNode(Matcher *Other) {
  if (this == Other)
    return takeNext();

  // Scan until we find the predecessor of Other.
  Matcher *Cur = this;
  for (; Cur && Cur->getNext() != Other; Cur = Cur->getNext())
    /*empty*/;

  if (!Cur)
    return nullptr;
  Cur->takeNext();
  Cur->setNext(Other->takeNext());
  return this;
}

/// canMoveBefore - Return true if this matcher is the same as Other, or if
/// we can move this matcher past all of the nodes in-between Other and this
/// node.  Other must be equal to or before this.
bool Matcher::canMoveBefore(const Matcher *Other) const {
  for (;; Other = Other->getNext()) {
    assert(Other && "Other didn't come before 'this'?");
    if (this == Other)
      return true;

    // We have to be able to move this node across the Other node.
    if (!canMoveBeforeNode(Other))
      return false;
  }
}

/// canMoveBeforeNode - Return true if it is safe to move the current matcher
/// across the specified one.
bool Matcher::canMoveBeforeNode(const Matcher *Other) const {
  // We can move simple predicates before record nodes.
  if (isSimplePredicateNode())
    return Other->isSimplePredicateOrRecordNode();

  // We can move record nodes across simple predicates.
  if (isSimplePredicateOrRecordNode())
    return isSimplePredicateNode();

  // We can't move record nodes across each other etc.
  return false;
}

ScopeMatcher::~ScopeMatcher() {
  for (Matcher *C : Children)
    delete C;
}

SwitchOpcodeMatcher::~SwitchOpcodeMatcher() {
  for (auto &C : Cases)
    delete C.second;
}

SwitchTypeMatcher::~SwitchTypeMatcher() {
  for (auto &C : Cases)
    delete C.second;
}

CheckPredicateMatcher::CheckPredicateMatcher(const TreePredicateFn &pred,
                                             ArrayRef<unsigned> Ops)
    : Matcher(CheckPredicate), Pred(pred.getOrigPatFragRecord()),
      Operands(Ops) {}

TreePredicateFn CheckPredicateMatcher::getPredicate() const {
  return TreePredicateFn(Pred);
}

unsigned CheckPredicateMatcher::getNumOperands() const {
  return Operands.size();
}

unsigned CheckPredicateMatcher::getOperandNo(unsigned i) const {
  assert(i < Operands.size());
  return Operands[i];
}

// printImpl methods.

void ScopeMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "Scope\n";
  for (const Matcher *C : Children) {
    if (!C)
      OS << Indent + 1 << "NULL POINTER\n";
    else
      C->print(OS, Indent + 2);
  }
}

void RecordMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "Record\n";
}

void RecordChildMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "RecordChild: " << ChildNo << '\n';
}

void RecordMemRefMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "RecordMemRef\n";
}

void CaptureGlueInputMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CaptureGlueInput\n";
}

void MoveChildMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "MoveChild " << ChildNo << '\n';
}

void MoveSiblingMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "MoveSibling " << SiblingNo << '\n';
}

void MoveParentMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "MoveParent\n";
}

void CheckSameMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckSame " << MatchNumber << '\n';
}

void CheckChildSameMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckChild" << ChildNo << "Same\n";
}

void CheckPatternPredicateMatcher::printImpl(raw_ostream &OS,
                                             indent Indent) const {
  OS << Indent << "CheckPatternPredicate " << Predicate << '\n';
}

void CheckPredicateMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckPredicate " << getPredicate().getFnName() << '\n';
}

void CheckOpcodeMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckOpcode " << Opcode.getEnumName() << '\n';
}

void SwitchOpcodeMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "SwitchOpcode: {\n";
  for (const auto &C : Cases) {
    OS << Indent << "case " << C.first->getEnumName() << ":\n";
    C.second->print(OS, Indent + 2);
  }
  OS << Indent << "}\n";
}

void CheckTypeMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckType " << getEnumName(Type) << ", ResNo=" << ResNo
     << '\n';
}

void SwitchTypeMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "SwitchType: {\n";
  for (const auto &C : Cases) {
    OS << Indent << "case " << getEnumName(C.first) << ":\n";
    C.second->print(OS, Indent + 2);
  }
  OS << Indent << "}\n";
}

void CheckChildTypeMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckChildType " << ChildNo << " " << getEnumName(Type)
     << '\n';
}

void CheckIntegerMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckInteger " << Value << '\n';
}

void CheckChildIntegerMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckChildInteger " << ChildNo << " " << Value << '\n';
}

void CheckCondCodeMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckCondCode ISD::" << CondCodeName << '\n';
}

void CheckChild2CondCodeMatcher::printImpl(raw_ostream &OS,
                                           indent Indent) const {
  OS << Indent << "CheckChild2CondCode ISD::" << CondCodeName << '\n';
}

void CheckValueTypeMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckValueType " << getEnumName(VT) << '\n';
}

void CheckComplexPatMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckComplexPat " << Pattern.getSelectFunc() << '\n';
}

void CheckAndImmMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckAndImm " << Value << '\n';
}

void CheckOrImmMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckOrImm " << Value << '\n';
}

void CheckFoldableChainNodeMatcher::printImpl(raw_ostream &OS,
                                              indent Indent) const {
  OS << Indent << "CheckFoldableChainNode\n";
}

void CheckImmAllOnesVMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckAllOnesV\n";
}

void CheckImmAllZerosVMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CheckAllZerosV\n";
}

void EmitIntegerMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "EmitInteger " << Val << " VT=" << getEnumName(VT) << '\n';
}

void EmitStringIntegerMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "EmitStringInteger " << Val << " VT=" << getEnumName(VT)
     << '\n';
}

void EmitRegisterMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "EmitRegister ";
  if (Reg)
    OS << Reg->getName();
  else
    OS << "zero_reg";
  OS << " VT=" << getEnumName(VT) << '\n';
}

void EmitConvertToTargetMatcher::printImpl(raw_ostream &OS,
                                           indent Indent) const {
  OS << Indent << "EmitConvertToTarget " << Slot << '\n';
}

void EmitMergeInputChainsMatcher::printImpl(raw_ostream &OS,
                                            indent Indent) const {
  OS << Indent << "EmitMergeInputChains <todo: args>\n";
}

void EmitCopyToRegMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "EmitCopyToReg <todo: args>\n";
}

void EmitNodeXFormMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "EmitNodeXForm " << NodeXForm->getName() << " Slot=" << Slot
     << '\n';
}

void EmitNodeMatcherCommon::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent;
  OS << (isa<MorphNodeToMatcher>(this) ? "MorphNodeTo: " : "EmitNode: ")
     << CGI.Namespace << "::" << CGI.TheDef->getName() << ": <todo flags> ";

  for (MVT::SimpleValueType VT : VTs)
    OS << ' ' << getEnumName(VT);
  OS << '(';
  for (unsigned Operand : Operands)
    OS << Operand << ' ';
  OS << ")\n";
}

void CompleteMatchMatcher::printImpl(raw_ostream &OS, indent Indent) const {
  OS << Indent << "CompleteMatch <todo args>\n";
  OS << Indent << "Src = " << Pattern.getSrcPattern() << "\n";
  OS << Indent << "Dst = " << Pattern.getDstPattern() << "\n";
}

bool CheckOpcodeMatcher::isEqualImpl(const Matcher *M) const {
  // Note: pointer equality isn't enough here, we have to check the enum names
  // to ensure that the nodes are for the same opcode.
  return cast<CheckOpcodeMatcher>(M)->Opcode.getEnumName() ==
         Opcode.getEnumName();
}

bool EmitNodeMatcherCommon::isEqualImpl(const Matcher *m) const {
  const EmitNodeMatcherCommon *M = cast<EmitNodeMatcherCommon>(m);
  return &M->CGI == &CGI && M->VTs == VTs && M->Operands == Operands &&
         M->HasChain == HasChain && M->HasInGlue == HasInGlue &&
         M->HasOutGlue == HasOutGlue && M->HasMemRefs == HasMemRefs &&
         M->NumFixedArityOperands == NumFixedArityOperands;
}

void EmitNodeMatcher::anchor() {}

void MorphNodeToMatcher::anchor() {}

// isContradictoryImpl Implementations.

static bool TypesAreContradictory(MVT::SimpleValueType T1,
                                  MVT::SimpleValueType T2) {
  // If the two types are the same, then they are the same, so they don't
  // contradict.
  if (T1 == T2)
    return false;

  // If either type is about iPtr, then they don't conflict unless the other
  // one is not a scalar integer type.
  if (T1 == MVT::iPTR)
    return !MVT(T2).isInteger() || MVT(T2).isVector();

  if (T2 == MVT::iPTR)
    return !MVT(T1).isInteger() || MVT(T1).isVector();

  // Otherwise, they are two different non-iPTR types, they conflict.
  return true;
}

bool CheckOpcodeMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckOpcodeMatcher *COM = dyn_cast<CheckOpcodeMatcher>(M)) {
    // One node can't have two different opcodes!
    // Note: pointer equality isn't enough here, we have to check the enum names
    // to ensure that the nodes are for the same opcode.
    return COM->getOpcode().getEnumName() != getOpcode().getEnumName();
  }

  // If the node has a known type, and if the type we're checking for is
  // different, then we know they contradict.  For example, a check for
  // ISD::STORE will never be true at the same time a check for Type i32 is.
  if (const CheckTypeMatcher *CT = dyn_cast<CheckTypeMatcher>(M)) {
    // If checking for a result the opcode doesn't have, it can't match.
    if (CT->getResNo() >= getOpcode().getNumResults())
      return true;

    MVT::SimpleValueType NodeType = getOpcode().getKnownType(CT->getResNo());
    if (NodeType != MVT::Other)
      return TypesAreContradictory(NodeType, CT->getType());
  }

  return false;
}

bool CheckTypeMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckTypeMatcher *CT = dyn_cast<CheckTypeMatcher>(M))
    return TypesAreContradictory(getType(), CT->getType());
  return false;
}

bool CheckChildTypeMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckChildTypeMatcher *CC = dyn_cast<CheckChildTypeMatcher>(M)) {
    // If the two checks are about different nodes, we don't know if they
    // conflict!
    if (CC->getChildNo() != getChildNo())
      return false;

    return TypesAreContradictory(getType(), CC->getType());
  }
  return false;
}

bool CheckIntegerMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckIntegerMatcher *CIM = dyn_cast<CheckIntegerMatcher>(M))
    return CIM->getValue() != getValue();
  return false;
}

bool CheckChildIntegerMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckChildIntegerMatcher *CCIM =
          dyn_cast<CheckChildIntegerMatcher>(M)) {
    // If the two checks are about different nodes, we don't know if they
    // conflict!
    if (CCIM->getChildNo() != getChildNo())
      return false;

    return CCIM->getValue() != getValue();
  }
  return false;
}

bool CheckValueTypeMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckValueTypeMatcher *CVT = dyn_cast<CheckValueTypeMatcher>(M))
    return CVT->getVT() != getVT();
  return false;
}

bool CheckImmAllOnesVMatcher::isContradictoryImpl(const Matcher *M) const {
  // AllZeros is contradictory.
  return isa<CheckImmAllZerosVMatcher>(M);
}

bool CheckImmAllZerosVMatcher::isContradictoryImpl(const Matcher *M) const {
  // AllOnes is contradictory.
  return isa<CheckImmAllOnesVMatcher>(M);
}

bool CheckCondCodeMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const auto *CCCM = dyn_cast<CheckCondCodeMatcher>(M))
    return CCCM->getCondCodeName() != getCondCodeName();
  return false;
}

bool CheckChild2CondCodeMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const auto *CCCCM = dyn_cast<CheckChild2CondCodeMatcher>(M))
    return CCCCM->getCondCodeName() != getCondCodeName();
  return false;
}
