//===------------------- StackMaps.h - StackMaps ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_STACKMAPS
#define LLVM_STACKMAPS

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include <map>
#include <vector>

namespace llvm {

class AsmPrinter;
class MCExpr;

/// \brief MI-level patchpoint operands.
///
/// MI patchpoint operations take the form:
/// [<def>], <id>, <numBytes>, <target>, <numArgs>, <cc>, ...
///
/// IR patchpoint intrinsics do not have the <cc> operand because calling
/// convention is part of the subclass data.
///
/// SD patchpoint nodes do not have a def operand because it is part of the
/// SDValue.
///
/// Patchpoints following the anyregcc convention are handled specially. For
/// these, the stack map also records the location of the return value and
/// arguments.
class PatchPointOpers {
public:
  /// Enumerate the meta operands.
  enum { IDPos, NBytesPos, TargetPos, NArgPos, CCPos, MetaEnd };
private:
  const MachineInstr *MI;
  bool HasDef;
  bool IsAnyReg;
public:
  explicit PatchPointOpers(const MachineInstr *MI);

  bool isAnyReg() const { return IsAnyReg; }
  bool hasDef() const { return HasDef; }

  unsigned getMetaIdx(unsigned Pos = 0) const {
    assert(Pos < MetaEnd && "Meta operand index out of range.");
    return (HasDef ? 1 : 0) + Pos;
  }

  const MachineOperand &getMetaOper(unsigned Pos) {
    return MI->getOperand(getMetaIdx(Pos));
  }

  unsigned getArgIdx() const { return getMetaIdx() + MetaEnd; }

  /// Get the operand index of the variable list of non-argument operands.
  /// These hold the "live state".
  unsigned getVarIdx() const {
    return getMetaIdx() + MetaEnd
      + MI->getOperand(getMetaIdx(NArgPos)).getImm();
  }

  /// Get the index at which stack map locations will be recorded.
  /// Arguments are not recorded unless the anyregcc convention is used.
  unsigned getStackMapStartIdx() const {
    if (IsAnyReg)
      return getArgIdx();
    return getVarIdx();
  }

  /// \brief Get the next scratch register operand index.
  unsigned getNextScratchIdx(unsigned StartIdx = 0) const;
};

class StackMaps {
public:
  struct Location {
    enum LocationType { Unprocessed, Register, Direct, Indirect, Constant,
                        ConstantIndex };
    LocationType LocType;
    unsigned Size;
    unsigned Reg;
    int64_t Offset;
    Location() : LocType(Unprocessed), Size(0), Reg(0), Offset(0) {}
    Location(LocationType LocType, unsigned Size, unsigned Reg, int64_t Offset)
      : LocType(LocType), Size(Size), Reg(Reg), Offset(Offset) {}
  };

  // Typedef a function pointer for functions that parse sequences of operands
  // and return a Location, plus a new "next" operand iterator.
  typedef std::pair<Location, MachineInstr::const_mop_iterator>
    (*OperandParser)(MachineInstr::const_mop_iterator,
                     MachineInstr::const_mop_iterator, const TargetMachine&);

  // OpTypes are used to encode information about the following logical
  // operand (which may consist of several MachineOperands) for the
  // OpParser.
  typedef enum { DirectMemRefOp, IndirectMemRefOp, ConstantOp } OpType;

  StackMaps(AsmPrinter &AP, OperandParser OpParser)
    : AP(AP), OpParser(OpParser) {}

  /// \brief Generate a stackmap record for a stackmap instruction.
  ///
  /// MI must be a raw STACKMAP, not a PATCHPOINT.
  void recordStackMap(const MachineInstr &MI);

  /// \brief Generate a stackmap record for a patchpoint instruction.
  void recordPatchPoint(const MachineInstr &MI);

  /// If there is any stack map data, create a stack map section and serialize
  /// the map info into it. This clears the stack map data structures
  /// afterwards.
  void serializeToStackMapSection();

private:
  typedef SmallVector<Location, 8> LocationVec;

  struct CallsiteInfo {
    const MCExpr *CSOffsetExpr;
    unsigned ID;
    LocationVec Locations;
    CallsiteInfo() : CSOffsetExpr(0), ID(0) {}
    CallsiteInfo(const MCExpr *CSOffsetExpr, unsigned ID,
                 LocationVec Locations)
      : CSOffsetExpr(CSOffsetExpr), ID(ID), Locations(Locations) {}
  };

  typedef std::vector<CallsiteInfo> CallsiteInfoList;

  struct ConstantPool {
  private:
    typedef std::map<int64_t, size_t> ConstantsMap;
    std::vector<int64_t> ConstantsList;
    ConstantsMap ConstantIndexes;

  public:
    size_t getNumConstants() const { return ConstantsList.size(); }
    int64_t getConstant(size_t Idx) const { return ConstantsList[Idx]; }
    size_t getConstantIndex(int64_t ConstVal) {
      size_t NextIdx = ConstantsList.size();
      ConstantsMap::const_iterator I =
        ConstantIndexes.insert(ConstantIndexes.end(),
                               std::make_pair(ConstVal, NextIdx));
      if (I->second == NextIdx)
        ConstantsList.push_back(ConstVal);
      return I->second;
    }
  };

  AsmPrinter &AP;
  OperandParser OpParser;
  CallsiteInfoList CSInfos;
  ConstantPool ConstPool;

  /// This should be called by the MC lowering code _immediately_ before
  /// lowering the MI to an MCInst. It records where the operands for the
  /// instruction are stored, and outputs a label to record the offset of
  /// the call from the start of the text section. In special cases (e.g. AnyReg
  /// calling convention) the return register is also recorded if requested.
  void recordStackMapOpers(const MachineInstr &MI, uint32_t ID,
                           MachineInstr::const_mop_iterator MOI,
                           MachineInstr::const_mop_iterator MOE,
                           bool recordResult = false);
};

}

#endif // LLVM_STACKMAPS
