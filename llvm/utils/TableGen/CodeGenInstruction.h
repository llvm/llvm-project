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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include <string>
#include <vector>
#include <utility>

namespace llvm {
  class Record;
  class DagInit;
  class CodeGenTarget;
  class StringRef;

  class CGIOperandList {
  public:
    class ConstraintInfo {
      enum { None, EarlyClobber, Tied } Kind;
      unsigned OtherTiedOperand;
    public:
      ConstraintInfo() : Kind(None) {}

      static ConstraintInfo getEarlyClobber() {
        ConstraintInfo I;
        I.Kind = EarlyClobber;
        I.OtherTiedOperand = 0;
        return I;
      }

      static ConstraintInfo getTied(unsigned Op) {
        ConstraintInfo I;
        I.Kind = Tied;
        I.OtherTiedOperand = Op;
        return I;
      }

      bool isNone() const { return Kind == None; }
      bool isEarlyClobber() const { return Kind == EarlyClobber; }
      bool isTied() const { return Kind == Tied; }

      unsigned getTiedOperand() const {
        assert(isTied());
        return OtherTiedOperand;
      }
    };

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

      /// EncoderMethodName - The method used to get the machine operand value
      /// for binary encoding. "getMachineOpValue" by default.
      std::string EncoderMethodName;

      /// OperandType - A value from MCOI::OperandType representing the type of
      /// the operand.
      std::string OperandType;

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
      std::vector<ConstraintInfo> Constraints;

      OperandInfo(Record *R, const std::string &N, const std::string &PMN,
                  const std::string &EMN, const std::string &OT, unsigned MION,
                  unsigned MINO, DagInit *MIOI)
      : Rec(R), Name(N), PrinterMethodName(PMN), EncoderMethodName(EMN),
        OperandType(OT), MIOperandNo(MION), MINumOperands(MINO),
        MIOperandInfo(MIOI) {}


      /// getTiedOperand - If this operand is tied to another one, return the
      /// other operand number.  Otherwise, return -1.
      int getTiedRegister() const {
        for (unsigned j = 0, e = Constraints.size(); j != e; ++j) {
          const CGIOperandList::ConstraintInfo &CI = Constraints[j];
          if (CI.isTied()) return CI.getTiedOperand();
        }
        return -1;
      }
    };

    CGIOperandList(Record *D);

    Record *TheDef;            // The actual record containing this OperandList.

    /// NumDefs - Number of def operands declared, this is the number of
    /// elements in the instruction's (outs) list.
    ///
    unsigned NumDefs;

    /// OperandList - The list of declared operands, along with their declared
    /// type (which is a record).
    std::vector<OperandInfo> OperandList;

    // Information gleaned from the operand list.
    bool isPredicable;
    bool hasOptionalDef;
    bool isVariadic;

    // Provide transparent accessors to the operand list.
    bool empty() const { return OperandList.empty(); }
    unsigned size() const { return OperandList.size(); }
    const OperandInfo &operator[](unsigned i) const { return OperandList[i]; }
    OperandInfo &operator[](unsigned i) { return OperandList[i]; }
    OperandInfo &back() { return OperandList.back(); }
    const OperandInfo &back() const { return OperandList.back(); }


    /// getOperandNamed - Return the index of the operand with the specified
    /// non-empty name.  If the instruction does not have an operand with the
    /// specified name, abort.
    unsigned getOperandNamed(StringRef Name) const;

    /// hasOperandNamed - Query whether the instruction has an operand of the
    /// given name. If so, return true and set OpIdx to the index of the
    /// operand. Otherwise, return false.
    bool hasOperandNamed(StringRef Name, unsigned &OpIdx) const;

    /// ParseOperandName - Parse an operand name like "$foo" or "$foo.bar",
    /// where $foo is a whole operand and $foo.bar refers to a suboperand.
    /// This aborts if the name is invalid.  If AllowWholeOp is true, references
    /// to operands with suboperands are allowed, otherwise not.
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

    void ProcessDisableEncoding(std::string Value);
  };


  class CodeGenInstruction {
  public:
    Record *TheDef;            // The actual record defining this instruction.
    std::string Namespace;     // The namespace the instruction is in.

    /// AsmString - The format string used to emit a .s file for the
    /// instruction.
    std::string AsmString;

    /// Operands - This is information about the (ins) and (outs) list specified
    /// to the instruction.
    CGIOperandList Operands;

    /// ImplicitDefs/ImplicitUses - These are lists of registers that are
    /// implicitly defined and used by the instruction.
    std::vector<Record*> ImplicitDefs, ImplicitUses;

    // Various boolean values we track for the instruction.
    bool isReturn;
    bool isBranch;
    bool isIndirectBranch;
    bool isCompare;
    bool isMoveImm;
    bool isBitcast;
    bool isSelect;
    bool isBarrier;
    bool isCall;
    bool canFoldAsLoad;
    bool mayLoad;
    bool mayLoad_Unset;
    bool mayStore;
    bool mayStore_Unset;
    bool isPredicable;
    bool isConvertibleToThreeAddress;
    bool isCommutable;
    bool isTerminator;
    bool isReMaterializable;
    bool hasDelaySlot;
    bool usesCustomInserter;
    bool hasPostISelHook;
    bool hasCtrlDep;
    bool isNotDuplicable;
    bool hasSideEffects;
    bool hasSideEffects_Unset;
    bool neverHasSideEffects;
    bool isAsCheapAsAMove;
    bool hasExtraSrcRegAllocReq;
    bool hasExtraDefRegAllocReq;
    bool isCodeGenOnly;
    bool isPseudo;

    /// Are there any undefined flags?
    bool hasUndefFlags() const {
      return mayLoad_Unset || mayStore_Unset || hasSideEffects_Unset;
    }

    // The record used to infer instruction flags, or NULL if no flag values
    // have been inferred.
    Record *InferredFrom;

    CodeGenInstruction(Record *R);

    /// HasOneImplicitDefWithKnownVT - If the instruction has at least one
    /// implicit def and it has a known VT, return the VT, otherwise return
    /// MVT::Other.
    MVT::SimpleValueType
      HasOneImplicitDefWithKnownVT(const CodeGenTarget &TargetInfo) const;


    /// FlattenAsmStringVariants - Flatten the specified AsmString to only
    /// include text from the specified variant, returning the new string.
    static std::string FlattenAsmStringVariants(StringRef AsmString,
                                                unsigned Variant);
  };


  /// CodeGenInstAlias - This represents an InstAlias definition.
  class CodeGenInstAlias {
  public:
    Record *TheDef;            // The actual record defining this InstAlias.

    /// AsmString - The format string used to emit a .s file for the
    /// instruction.
    std::string AsmString;

    /// Result - The result instruction.
    DagInit *Result;

    /// ResultInst - The instruction generated by the alias (decoded from
    /// Result).
    CodeGenInstruction *ResultInst;


    struct ResultOperand {
    private:
      std::string Name;
      Record *R;

      int64_t Imm;
    public:
      enum {
        K_Record,
        K_Imm,
        K_Reg
      } Kind;

      ResultOperand(std::string N, Record *r) : Name(N), R(r), Kind(K_Record) {}
      ResultOperand(int64_t I) : Imm(I), Kind(K_Imm) {}
      ResultOperand(Record *r) : R(r), Kind(K_Reg) {}

      bool isRecord() const { return Kind == K_Record; }
      bool isImm() const { return Kind == K_Imm; }
      bool isReg() const { return Kind == K_Reg; }

      StringRef getName() const { assert(isRecord()); return Name; }
      Record *getRecord() const { assert(isRecord()); return R; }
      int64_t getImm() const { assert(isImm()); return Imm; }
      Record *getRegister() const { assert(isReg()); return R; }
    };

    /// ResultOperands - The decoded operands for the result instruction.
    std::vector<ResultOperand> ResultOperands;

    /// ResultInstOperandIndex - For each operand, this vector holds a pair of
    /// indices to identify the corresponding operand in the result
    /// instruction.  The first index specifies the operand and the second
    /// index specifies the suboperand.  If there are no suboperands or if all
    /// of them are matched by the operand, the second value should be -1.
    std::vector<std::pair<unsigned, int> > ResultInstOperandIndex;

    CodeGenInstAlias(Record *R, CodeGenTarget &T);

    bool tryAliasOpMatch(DagInit *Result, unsigned AliasOpNo,
                         Record *InstOpRec, bool hasSubOps, ArrayRef<SMLoc> Loc,
                         CodeGenTarget &T, ResultOperand &ResOp);
  };
}

#endif
