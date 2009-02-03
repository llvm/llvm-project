//===- CodeGenInstruction.cpp - CodeGen Instruction Class Wrapper ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CodeGenInstruction class.
//
//===----------------------------------------------------------------------===//

#include "CodeGenInstruction.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include <set>
using namespace llvm;

static void ParseConstraint(const std::string &CStr, CodeGenInstruction *I) {
  // FIXME: Only supports TIED_TO for now.
  std::string::size_type pos = CStr.find_first_of('=');
  assert(pos != std::string::npos && "Unrecognized constraint");
  std::string::size_type start = CStr.find_first_not_of(" \t");
  std::string Name = CStr.substr(start, pos);
  
  // TIED_TO: $src1 = $dst
  std::string::size_type wpos = Name.find_first_of(" \t");
  if (wpos == std::string::npos)
    throw "Illegal format for tied-to constraint: '" + CStr + "'";
  std::string DestOpName = Name.substr(0, wpos);
  std::pair<unsigned,unsigned> DestOp = I->ParseOperandName(DestOpName, false);
  
  Name = CStr.substr(pos+1);
  wpos = Name.find_first_not_of(" \t");
  if (wpos == std::string::npos)
    throw "Illegal format for tied-to constraint: '" + CStr + "'";
  
  std::pair<unsigned,unsigned> SrcOp =
  I->ParseOperandName(Name.substr(wpos), false);
  if (SrcOp > DestOp)
    throw "Illegal tied-to operand constraint '" + CStr + "'";
  
  
  unsigned FlatOpNo = I->getFlattenedOperandNumber(SrcOp);
  // Build the string for the operand.
  std::string OpConstraint =
  "((" + utostr(FlatOpNo) + " << 16) | (1 << TOI::TIED_TO))";
  
  
  if (!I->OperandList[DestOp.first].Constraints[DestOp.second].empty())
    throw "Operand '" + DestOpName + "' cannot have multiple constraints!";
  I->OperandList[DestOp.first].Constraints[DestOp.second] = OpConstraint;
}

static void ParseConstraints(const std::string &CStr, CodeGenInstruction *I) {
  // Make sure the constraints list for each operand is large enough to hold
  // constraint info, even if none is present.
  for (unsigned i = 0, e = I->OperandList.size(); i != e; ++i) 
    I->OperandList[i].Constraints.resize(I->OperandList[i].MINumOperands);
  
  if (CStr.empty()) return;
  
  const std::string delims(",");
  std::string::size_type bidx, eidx;
  
  bidx = CStr.find_first_not_of(delims);
  while (bidx != std::string::npos) {
    eidx = CStr.find_first_of(delims, bidx);
    if (eidx == std::string::npos)
      eidx = CStr.length();
    
    ParseConstraint(CStr.substr(bidx, eidx), I);
    bidx = CStr.find_first_not_of(delims, eidx);
  }
}

CodeGenInstruction::CodeGenInstruction(Record *R, const std::string &AsmStr)
  : TheDef(R), AsmString(AsmStr) {
  Namespace = R->getValueAsString("Namespace");

  isReturn     = R->getValueAsBit("isReturn");
  isBranch     = R->getValueAsBit("isBranch");
  isIndirectBranch = R->getValueAsBit("isIndirectBranch");
  isBarrier    = R->getValueAsBit("isBarrier");
  isCall       = R->getValueAsBit("isCall");
  canFoldAsLoad = R->getValueAsBit("canFoldAsLoad");
  mayLoad      = R->getValueAsBit("mayLoad");
  mayStore     = R->getValueAsBit("mayStore");
  bool isTwoAddress = R->getValueAsBit("isTwoAddress");
  isPredicable = R->getValueAsBit("isPredicable");
  isConvertibleToThreeAddress = R->getValueAsBit("isConvertibleToThreeAddress");
  isCommutable = R->getValueAsBit("isCommutable");
  isTerminator = R->getValueAsBit("isTerminator");
  isReMaterializable = R->getValueAsBit("isReMaterializable");
  hasDelaySlot = R->getValueAsBit("hasDelaySlot");
  usesCustomDAGSchedInserter = R->getValueAsBit("usesCustomDAGSchedInserter");
  hasCtrlDep   = R->getValueAsBit("hasCtrlDep");
  isNotDuplicable = R->getValueAsBit("isNotDuplicable");
  hasSideEffects = R->getValueAsBit("hasSideEffects");
  mayHaveSideEffects = R->getValueAsBit("mayHaveSideEffects");
  neverHasSideEffects = R->getValueAsBit("neverHasSideEffects");
  isAsCheapAsAMove = R->getValueAsBit("isAsCheapAsAMove");
  hasOptionalDef = false;
  isVariadic = false;

  if (mayHaveSideEffects + neverHasSideEffects + hasSideEffects > 1)
    throw R->getName() + ": multiple conflicting side-effect flags set!";

  DagInit *DI;
  try {
    DI = R->getValueAsDag("OutOperandList");
  } catch (...) {
    // Error getting operand list, just ignore it (sparcv9).
    AsmString.clear();
    OperandList.clear();
    return;
  }
  NumDefs = DI->getNumArgs();

  DagInit *IDI;
  try {
    IDI = R->getValueAsDag("InOperandList");
  } catch (...) {
    // Error getting operand list, just ignore it (sparcv9).
    AsmString.clear();
    OperandList.clear();
    return;
  }
  DI = (DagInit*)(new BinOpInit(BinOpInit::CONCAT, DI, IDI))->Fold();

  unsigned MIOperandNo = 0;
  std::set<std::string> OperandNames;
  for (unsigned i = 0, e = DI->getNumArgs(); i != e; ++i) {
    DefInit *Arg = dynamic_cast<DefInit*>(DI->getArg(i));
    if (!Arg)
      throw "Illegal operand for the '" + R->getName() + "' instruction!";

    Record *Rec = Arg->getDef();
    std::string PrintMethod = "printOperand";
    unsigned NumOps = 1;
    DagInit *MIOpInfo = 0;
    if (Rec->isSubClassOf("Operand")) {
      PrintMethod = Rec->getValueAsString("PrintMethod");
      MIOpInfo = Rec->getValueAsDag("MIOperandInfo");
      
      // Verify that MIOpInfo has an 'ops' root value.
      if (!dynamic_cast<DefInit*>(MIOpInfo->getOperator()) ||
          dynamic_cast<DefInit*>(MIOpInfo->getOperator())
               ->getDef()->getName() != "ops")
        throw "Bad value for MIOperandInfo in operand '" + Rec->getName() +
              "'\n";

      // If we have MIOpInfo, then we have #operands equal to number of entries
      // in MIOperandInfo.
      if (unsigned NumArgs = MIOpInfo->getNumArgs())
        NumOps = NumArgs;

      if (Rec->isSubClassOf("PredicateOperand"))
        isPredicable = true;
      else if (Rec->isSubClassOf("OptionalDefOperand"))
        hasOptionalDef = true;
    } else if (Rec->getName() == "variable_ops") {
      isVariadic = true;
      continue;
    } else if (!Rec->isSubClassOf("RegisterClass") && 
               Rec->getName() != "ptr_rc" && Rec->getName() != "unknown")
      throw "Unknown operand class '" + Rec->getName() +
            "' in '" + R->getName() + "' instruction!";

    // Check that the operand has a name and that it's unique.
    if (DI->getArgName(i).empty())
      throw "In instruction '" + R->getName() + "', operand #" + utostr(i) +
        " has no name!";
    if (!OperandNames.insert(DI->getArgName(i)).second)
      throw "In instruction '" + R->getName() + "', operand #" + utostr(i) +
        " has the same name as a previous operand!";
    
    OperandList.push_back(OperandInfo(Rec, DI->getArgName(i), PrintMethod, 
                                      MIOperandNo, NumOps, MIOpInfo));
    MIOperandNo += NumOps;
  }

  // Parse Constraints.
  ParseConstraints(R->getValueAsString("Constraints"), this);
  
  // For backward compatibility: isTwoAddress means operand 1 is tied to
  // operand 0.
  if (isTwoAddress) {
    if (!OperandList[1].Constraints[0].empty())
      throw R->getName() + ": cannot use isTwoAddress property: instruction "
            "already has constraint set!";
    OperandList[1].Constraints[0] = "((0 << 16) | (1 << TOI::TIED_TO))";
  }
  
  // Any operands with unset constraints get 0 as their constraint.
  for (unsigned op = 0, e = OperandList.size(); op != e; ++op)
    for (unsigned j = 0, e = OperandList[op].MINumOperands; j != e; ++j)
      if (OperandList[op].Constraints[j].empty())
        OperandList[op].Constraints[j] = "0";
  
  // Parse the DisableEncoding field.
  std::string DisableEncoding = R->getValueAsString("DisableEncoding");
  while (1) {
    std::string OpName = getToken(DisableEncoding, " ,\t");
    if (OpName.empty()) break;

    // Figure out which operand this is.
    std::pair<unsigned,unsigned> Op = ParseOperandName(OpName, false);

    // Mark the operand as not-to-be encoded.
    if (Op.second >= OperandList[Op.first].DoNotEncode.size())
      OperandList[Op.first].DoNotEncode.resize(Op.second+1);
    OperandList[Op.first].DoNotEncode[Op.second] = true;
  }
}

/// getOperandNamed - Return the index of the operand with the specified
/// non-empty name.  If the instruction does not have an operand with the
/// specified name, throw an exception.
///
unsigned CodeGenInstruction::getOperandNamed(const std::string &Name) const {
  assert(!Name.empty() && "Cannot search for operand with no name!");
  for (unsigned i = 0, e = OperandList.size(); i != e; ++i)
    if (OperandList[i].Name == Name) return i;
  throw "Instruction '" + TheDef->getName() +
        "' does not have an operand named '$" + Name + "'!";
}

std::pair<unsigned,unsigned> 
CodeGenInstruction::ParseOperandName(const std::string &Op,
                                     bool AllowWholeOp) {
  if (Op.empty() || Op[0] != '$')
    throw TheDef->getName() + ": Illegal operand name: '" + Op + "'";
  
  std::string OpName = Op.substr(1);
  std::string SubOpName;
  
  // Check to see if this is $foo.bar.
  std::string::size_type DotIdx = OpName.find_first_of(".");
  if (DotIdx != std::string::npos) {
    SubOpName = OpName.substr(DotIdx+1);
    if (SubOpName.empty())
      throw TheDef->getName() + ": illegal empty suboperand name in '" +Op +"'";
    OpName = OpName.substr(0, DotIdx);
  }
  
  unsigned OpIdx = getOperandNamed(OpName);

  if (SubOpName.empty()) {  // If no suboperand name was specified:
    // If one was needed, throw.
    if (OperandList[OpIdx].MINumOperands > 1 && !AllowWholeOp &&
        SubOpName.empty())
      throw TheDef->getName() + ": Illegal to refer to"
            " whole operand part of complex operand '" + Op + "'";
  
    // Otherwise, return the operand.
    return std::make_pair(OpIdx, 0U);
  }
  
  // Find the suboperand number involved.
  DagInit *MIOpInfo = OperandList[OpIdx].MIOperandInfo;
  if (MIOpInfo == 0)
    throw TheDef->getName() + ": unknown suboperand name in '" + Op + "'";
  
  // Find the operand with the right name.
  for (unsigned i = 0, e = MIOpInfo->getNumArgs(); i != e; ++i)
    if (MIOpInfo->getArgName(i) == SubOpName)
      return std::make_pair(OpIdx, i);

  // Otherwise, didn't find it!
  throw TheDef->getName() + ": unknown suboperand name in '" + Op + "'";
}
