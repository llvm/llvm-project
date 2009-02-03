//===- CodeGenInstruction.h - Instruction Class Wrapper ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a wrapper class for the 'Instruction' TableGen class.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_INSTRUCTION_H
#define CODEGEN_INSTRUCTION_H

#include "llvm/CodeGen/ValueTypes.h"
#include <string>
#include <vector>
#include <utility>

namespace llvm {
  class Record;
  class DagInit;

  class CodeGenInstruction {
  public:
    Record *TheDef;            // The actual record defining this instruction.
    std::string Namespace;     // The namespace the instruction is in.

    /// AsmString - The format string used to emit a .s file for the
    /// instruction.
    std::string AsmString;
    
    /// OperandInfo - The information we keep track of for each operand in the
    /// operand list for a tablegen instruction.
    struct OperandInfo {
      /// Rec - The definition this operand is declared as.
      ///
      Record *Rec;

      /// Name - If this operand was assigned a symbolic name, this is it,
      /// otherwise, it's empty.
      std::string Name;

      /// PrinterMethodName - The method used to print operands of this type in
      /// the asmprinter.
      std::string PrinterMethodName;

      /// MIOperandNo - Currently (this is meant to be phased out), some logical
      /// operands correspond to multiple MachineInstr operands.  In the X86
      /// target for example, one address operand is represented as 4
      /// MachineOperands.  Because of this, the operand number in the
      /// OperandList may not match the MachineInstr operand num.  Until it
      /// does, this contains the MI operand index of this operand.
      unsigned MIOperandNo;
      unsigned MINumOperands;   // The number of operands.

      /// DoNotEncode - Bools are set to true in this vector for each operand in
      /// the DisableEncoding list.  These should not be emitted by the code
      /// emitter.
      std::vector<bool> DoNotEncode;
      
      /// MIOperandInfo - Default MI operand type. Note an operand may be made
      /// up of multiple MI operands.
      DagInit *MIOperandInfo;
      
      /// Constraint info for this operand.  This operand can have pieces, so we
      /// track constraint info for each.
      std::vector<std::string> Constraints;

      OperandInfo(Record *R, const std::string &N, const std::string &PMN, 
                  unsigned MION, unsigned MINO, DagInit *MIOI)
        : Rec(R), Name(N), PrinterMethodName(PMN), MIOperandNo(MION),
          MINumOperands(MINO), MIOperandInfo(MIOI) {}
    };

    /// NumDefs - Number of def operands declared.
    ///
    unsigned NumDefs;

    /// OperandList - The list of declared operands, along with their declared
    /// type (which is a record).
    std::vector<OperandInfo> OperandList;

    // Various boolean values we track for the instruction.
    bool isReturn;
    bool isBranch;
    bool isIndirectBranch;
    bool isBarrier;
    bool isCall;
    bool canFoldAsLoad;
    bool mayLoad, mayStore;
    bool isPredicable;
    bool isConvertibleToThreeAddress;
    bool isCommutable;
    bool isTerminator;
    bool isReMaterializable;
    bool hasDelaySlot;
    bool usesCustomDAGSchedInserter;
    bool isVariadic;
    bool hasCtrlDep;
    bool isNotDuplicable;
    bool hasOptionalDef;
    bool hasSideEffects;
    bool mayHaveSideEffects;
    bool neverHasSideEffects;
    bool isAsCheapAsAMove;
    
    /// ParseOperandName - Parse an operand name like "$foo" or "$foo.bar",
    /// where $foo is a whole operand and $foo.bar refers to a suboperand.
    /// This throws an exception if the name is invalid.  If AllowWholeOp is
    /// true, references to operands with suboperands are allowed, otherwise
    /// not.
    std::pair<unsigned,unsigned> ParseOperandName(const std::string &Op,
                                                  bool AllowWholeOp = true);
    
    /// getFlattenedOperandNumber - Flatten a operand/suboperand pair into a
    /// flat machineinstr operand #.
    unsigned getFlattenedOperandNumber(std::pair<unsigned,unsigned> Op) const {
      return OperandList[Op.first].MIOperandNo + Op.second;
    }
    
    /// getSubOperandNumber - Unflatten a operand number into an
    /// operand/suboperand pair.
    std::pair<unsigned,unsigned> getSubOperandNumber(unsigned Op) const {
      for (unsigned i = 0; ; ++i) {
        assert(i < OperandList.size() && "Invalid flat operand #");
        if (OperandList[i].MIOperandNo+OperandList[i].MINumOperands > Op)
          return std::make_pair(i, Op-OperandList[i].MIOperandNo);
      }
    }
    
    
    /// isFlatOperandNotEmitted - Return true if the specified flat operand #
    /// should not be emitted with the code emitter.
    bool isFlatOperandNotEmitted(unsigned FlatOpNo) const {
      std::pair<unsigned,unsigned> Op = getSubOperandNumber(FlatOpNo);
      if (OperandList[Op.first].DoNotEncode.size() > Op.second)
        return OperandList[Op.first].DoNotEncode[Op.second];
      return false;
    }

    CodeGenInstruction(Record *R, const std::string &AsmStr);

    /// getOperandNamed - Return the index of the operand with the specified
    /// non-empty name.  If the instruction does not have an operand with the
    /// specified name, throw an exception.
    unsigned getOperandNamed(const std::string &Name) const;
  };
}

#endif
