//===- InstrInfoEmitter.cpp - Generate a Instruction Set Desc. --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#include "Basic/SequenceToOffsetTable.h"
#include "Common/CodeGenDAGPatterns.h"
#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenSchedule.h"
#include "Common/CodeGenTarget.h"
#include "Common/PredicateExpander.h"
#include "Common/SubtargetFeatureInfo.h"
#include "Common/Types.h"
#include "TableGenBackends.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TGTimer.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

static cl::OptionCategory InstrInfoEmitterCat("Options for -gen-instr-info");
static cl::opt<bool> ExpandMIOperandInfo(
    "instr-info-expand-mi-operand-info",
    cl::desc("Expand operand's MIOperandInfo DAG into suboperands"),
    cl::cat(InstrInfoEmitterCat), cl::init(true));

namespace {

class InstrInfoEmitter {
  const RecordKeeper &Records;
  const CodeGenDAGPatterns CDP;
  const CodeGenSchedModels &SchedModels;

public:
  InstrInfoEmitter(const RecordKeeper &R)
      : Records(R), CDP(R), SchedModels(CDP.getTargetInfo().getSchedModels()) {}

  // run - Output the instruction set description.
  void run(raw_ostream &OS);

private:
  void emitEnums(raw_ostream &OS,
                 ArrayRef<const CodeGenInstruction *> NumberedInstructions);

  typedef std::vector<std::string> OperandInfoTy;
  typedef std::vector<OperandInfoTy> OperandInfoListTy;
  typedef std::map<OperandInfoTy, unsigned> OperandInfoMapTy;

  /// Generate member functions in the target-specific GenInstrInfo class.
  ///
  /// This method is used to custom expand TIIPredicate definitions.
  /// See file llvm/Target/TargetInstPredicates.td for a description of what is
  /// a TIIPredicate and how to use it.
  void emitTIIHelperMethods(raw_ostream &OS, StringRef TargetName,
                            bool ExpandDefinition = true);

  /// Expand TIIPredicate definitions to functions that accept a const MCInst
  /// reference.
  void emitMCIIHelperMethods(raw_ostream &OS, StringRef TargetName);

  /// Write verifyInstructionPredicates methods.
  void emitFeatureVerifier(raw_ostream &OS, const CodeGenTarget &Target);
  void emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                  const Record *InstrInfo,
                  std::map<std::vector<const Record *>, unsigned> &EL,
                  const OperandInfoMapTy &OperandInfo, raw_ostream &OS);
  void emitOperandTypeMappings(
      raw_ostream &OS, const CodeGenTarget &Target,
      ArrayRef<const CodeGenInstruction *> NumberedInstructions);
  void emitOperandNameMappings(
      raw_ostream &OS, const CodeGenTarget &Target,
      ArrayRef<const CodeGenInstruction *> TargetInstructions);
  void emitLogicalOperandSizeMappings(
      raw_ostream &OS, StringRef Namespace,
      ArrayRef<const CodeGenInstruction *> TargetInstructions);

  // Operand information.
  unsigned CollectOperandInfo(OperandInfoListTy &OperandInfoList,
                              OperandInfoMapTy &OperandInfoMap);
  void EmitOperandInfo(raw_ostream &OS, OperandInfoListTy &OperandInfoList);
  OperandInfoTy GetOperandInfo(const CodeGenInstruction &Inst);
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Operand Info Emission.
//===----------------------------------------------------------------------===//

InstrInfoEmitter::OperandInfoTy
InstrInfoEmitter::GetOperandInfo(const CodeGenInstruction &Inst) {
  OperandInfoTy Result;

  for (auto &Op : Inst.Operands) {
    // Handle aggregate operands and normal operands the same way by expanding
    // either case into a list of operands for this op.
    std::vector<CGIOperandList::OperandInfo> OperandList;

    // This might be a multiple operand thing. Targets like X86 have registers
    // in their multi-operand operands. It may also be an anonymous operand,
    // which has a single operand, but no declared class for the operand.
    const DagInit *MIOI = Op.MIOperandInfo;

    if (!MIOI || MIOI->getNumArgs() == 0) {
      // Single, anonymous, operand.
      OperandList.push_back(Op);
    } else {
      for (unsigned j = 0, e = Op.MINumOperands; j != e; ++j) {
        OperandList.push_back(Op);

        auto *OpR = cast<DefInit>(MIOI->getArg(j))->getDef();
        OperandList.back().Rec = OpR;
      }
    }

    for (const auto &[OpInfo, Constraint] :
         zip_equal(OperandList, Op.Constraints)) {
      const Record *OpR = OpInfo.Rec;
      std::string Res;

      if (OpR->isSubClassOf("RegisterOperand"))
        OpR = OpR->getValueAsDef("RegClass");
      if (OpR->isSubClassOf("RegisterClass"))
        Res += getQualifiedName(OpR) + "RegClassID, ";
      else if (OpR->isSubClassOf("PointerLikeRegClass"))
        Res += utostr(OpR->getValueAsInt("RegClassKind")) + ", ";
      else
        // -1 means the operand does not have a fixed register class.
        Res += "-1, ";

      // Fill in applicable flags.
      Res += "0";

      // Ptr value whose register class is resolved via callback.
      if (OpR->isSubClassOf("PointerLikeRegClass"))
        Res += "|(1<<MCOI::LookupPtrRegClass)";

      // Predicate operands.  Check to see if the original unexpanded operand
      // was of type PredicateOp.
      if (Op.Rec->isSubClassOf("PredicateOp"))
        Res += "|(1<<MCOI::Predicate)";

      // Optional def operands.  Check to see if the original unexpanded operand
      // was of type OptionalDefOperand.
      if (Op.Rec->isSubClassOf("OptionalDefOperand"))
        Res += "|(1<<MCOI::OptionalDef)";

      // Branch target operands.  Check to see if the original unexpanded
      // operand was of type BranchTargetOperand.
      if (Op.Rec->isSubClassOf("BranchTargetOperand"))
        Res += "|(1<<MCOI::BranchTarget)";

      // Fill in operand type.
      Res += ", ";
      assert(!Op.OperandType.empty() && "Invalid operand type.");
      Res += Op.OperandType;

      // Fill in constraint info.
      Res += ", ";

      if (Constraint.isNone()) {
        Res += "0";
      } else if (Constraint.isEarlyClobber()) {
        Res += "MCOI_EARLY_CLOBBER";
      } else {
        assert(Constraint.isTied());
        Res += "MCOI_TIED_TO(" + utostr(Constraint.getTiedOperand()) + ")";
      }

      Result.push_back(Res);
    }
  }

  return Result;
}

unsigned
InstrInfoEmitter::CollectOperandInfo(OperandInfoListTy &OperandInfoList,
                                     OperandInfoMapTy &OperandInfoMap) {
  const CodeGenTarget &Target = CDP.getTargetInfo();
  unsigned Offset = 0;
  for (const CodeGenInstruction *Inst : Target.getInstructions()) {
    OperandInfoTy OperandInfo = GetOperandInfo(*Inst);
    if (OperandInfoMap.try_emplace(OperandInfo, Offset).second) {
      OperandInfoList.push_back(OperandInfo);
      Offset += OperandInfo.size();
    }
  }
  return Offset;
}

void InstrInfoEmitter::EmitOperandInfo(raw_ostream &OS,
                                       OperandInfoListTy &OperandInfoList) {
  unsigned Offset = 0;
  for (auto &OperandInfo : OperandInfoList) {
    OS << "    /* " << Offset << " */";
    for (auto &Info : OperandInfo)
      OS << " { " << Info << " },";
    OS << '\n';
    Offset += OperandInfo.size();
  }
}

static void emitGetInstructionIndexForOpLookup(
    raw_ostream &OS, const MapVector<SmallVector<int>, unsigned> &OperandMap,
    ArrayRef<unsigned> InstructionIndex) {
  StringRef Type = OperandMap.size() <= UINT8_MAX + 1 ? "uint8_t" : "uint16_t";
  OS << "LLVM_READONLY static " << Type
     << " getInstructionIndexForOpLookup(uint16_t Opcode) {\n"
        "  static constexpr "
     << Type << " InstructionIndex[] = {";
  for (auto [TableIndex, Entry] : enumerate(InstructionIndex))
    OS << (TableIndex % 16 == 0 ? "\n    " : " ") << Entry << ',';
  OS << "\n  };\n"
        "  return InstructionIndex[Opcode];\n"
        "}\n";
}

static void
emitGetNamedOperandIdx(raw_ostream &OS,
                       const MapVector<SmallVector<int>, unsigned> &OperandMap,
                       unsigned MaxOperandNo, unsigned NumOperandNames) {
  OS << "LLVM_READONLY int16_t getNamedOperandIdx(uint16_t Opcode, OpName "
        "Name) {\n";
  OS << "  assert(Name != OpName::NUM_OPERAND_NAMES);\n";
  if (!NumOperandNames) {
    // There are no operands, so no need to emit anything
    OS << "  return -1;\n}\n";
    return;
  }
  assert(MaxOperandNo <= INT16_MAX &&
         "Too many operands for the operand name -> index table");
  StringRef Type = MaxOperandNo <= INT8_MAX ? "int8_t" : "int16_t";
  OS << "  static constexpr " << Type << " OperandMap[][" << NumOperandNames
     << "] = {\n";
  for (const auto &[OpList, _] : OperandMap) {
    // Emit a row of the OperandMap table.
    OS << "    {";
    for (unsigned ID = 0; ID < NumOperandNames; ++ID)
      OS << (ID < OpList.size() ? OpList[ID] : -1) << ", ";
    OS << "},\n";
  }
  OS << "  };\n";

  OS << "  unsigned InstrIdx = getInstructionIndexForOpLookup(Opcode);\n"
        "  return OperandMap[InstrIdx][(unsigned)Name];\n"
        "}\n";
}

static void
emitGetOperandIdxName(raw_ostream &OS,
                      MapVector<StringRef, unsigned> OperandNameToID,
                      const MapVector<SmallVector<int>, unsigned> &OperandMap,
                      unsigned MaxNumOperands, unsigned NumOperandNames) {
  OS << "LLVM_READONLY OpName getOperandIdxName(uint16_t Opcode, int16_t Idx) "
        "{\n";
  OS << "  assert(Idx >= 0 && Idx < " << MaxNumOperands << ");\n";
  if (!MaxNumOperands) {
    // There are no operands, so no need to emit anything
    OS << "  return -1;\n}\n";
    return;
  }
  OS << "  static constexpr OpName OperandMap[][" << MaxNumOperands
     << "] = {\n";
  for (const auto &[OpList, _] : OperandMap) {
    SmallVector<unsigned> IDs(MaxNumOperands, NumOperandNames);
    for (const auto &[ID, Idx] : enumerate(OpList)) {
      if (Idx >= 0)
        IDs[Idx] = ID;
    }
    // Emit a row of the OperandMap table. Map operand indices to enum values.
    OS << "    {";
    for (unsigned ID : IDs) {
      if (ID == NumOperandNames)
        OS << "OpName::NUM_OPERAND_NAMES, ";
      else
        OS << "OpName::" << OperandNameToID.getArrayRef()[ID].first << ", ";
    }
    OS << "},\n";
  }
  OS << "  };\n";

  OS << "  unsigned InstrIdx = getInstructionIndexForOpLookup(Opcode);\n"
        "  return OperandMap[InstrIdx][(unsigned)Idx];\n"
        "}\n";
}

/// Generate a table and function for looking up the indices of operands by
/// name.
///
/// This code generates:
/// - An enum in the llvm::TargetNamespace::OpName namespace, with one entry
///   for each operand name.
/// - A 2-dimensional table for mapping OpName enum values to operand indices.
/// - A function called getNamedOperandIdx(uint16_t Opcode, OpName Name)
///   for looking up the operand index for an instruction, given a value from
///   OpName enum
/// - A 2-dimensional table for mapping operand indices to OpName enum values.
/// - A function called getOperandIdxName(uint16_t Opcode, int16_t Idx)
///   for looking up the OpName enum for an instruction, given the operand
///   index. This is the inverse of getNamedOperandIdx().
///
/// Fixed/Predefined instructions do not have UseNamedOperandTable enabled, so
/// we can just skip them. Hence accept just the TargetInstructions.
void InstrInfoEmitter::emitOperandNameMappings(
    raw_ostream &OS, const CodeGenTarget &Target,
    ArrayRef<const CodeGenInstruction *> TargetInstructions) {
  StringRef Namespace = Target.getInstNamespace();

  // Map of operand names to their ID.
  MapVector<StringRef, unsigned> OperandNameToID;

  /// A key in this map is a vector mapping OpName ID values to instruction
  /// operand indices or -1 (but without any trailing -1 values which will be
  /// added later). The corresponding value in this map is the index of that row
  /// in the emitted OperandMap table. This map helps to unique entries among
  /// instructions that have identical OpName -> Operand index mapping.
  MapVector<SmallVector<int>, unsigned> OperandMap;

  // Max operand index seen.
  unsigned MaxOperandNo = 0;

  // Fixed/Predefined instructions do not have UseNamedOperandTable enabled, so
  // add a dummy map entry for them.
  OperandMap.try_emplace({}, 0);
  unsigned FirstTargetVal = TargetInstructions.front()->EnumVal;
  SmallVector<unsigned> InstructionIndex(FirstTargetVal, 0);
  for (const CodeGenInstruction *Inst : TargetInstructions) {
    if (!Inst->TheDef->getValueAsBit("UseNamedOperandTable")) {
      InstructionIndex.push_back(0);
      continue;
    }
    SmallVector<int> OpList;
    for (const auto &Info : Inst->Operands) {
      unsigned ID =
          OperandNameToID.try_emplace(Info.Name, OperandNameToID.size())
              .first->second;
      OpList.resize(std::max((unsigned)OpList.size(), ID + 1), -1);
      OpList[ID] = Info.MIOperandNo;
      MaxOperandNo = std::max(MaxOperandNo, Info.MIOperandNo);
    }
    auto [It, Inserted] =
        OperandMap.try_emplace(std::move(OpList), OperandMap.size());
    InstructionIndex.push_back(It->second);
  }

  const size_t NumOperandNames = OperandNameToID.size();
  const unsigned MaxNumOperands = MaxOperandNo + 1;

  OS << "#ifdef GET_INSTRINFO_OPERAND_ENUM\n";
  OS << "#undef GET_INSTRINFO_OPERAND_ENUM\n";
  OS << "namespace llvm::" << Namespace << " {\n";

  assert(NumOperandNames <= UINT16_MAX &&
         "Too many operands for the operand index -> name table");
  StringRef EnumType = getMinimalTypeForRange(NumOperandNames);
  OS << "enum class OpName : " << EnumType << " {\n";
  for (const auto &[Op, I] : OperandNameToID)
    OS << "  " << Op << " = " << I << ",\n";
  OS << "  NUM_OPERAND_NAMES = " << NumOperandNames << ",\n";
  OS << "}; // enum class OpName\n\n";

  OS << "LLVM_READONLY int16_t getNamedOperandIdx(uint16_t Opcode, OpName "
        "Name);\n";
  OS << "LLVM_READONLY OpName getOperandIdxName(uint16_t Opcode, int16_t "
        "Idx);\n";
  OS << "} // end namespace llvm::" << Namespace << '\n';
  OS << "#endif //GET_INSTRINFO_OPERAND_ENUM\n\n";

  OS << "#ifdef GET_INSTRINFO_NAMED_OPS\n";
  OS << "#undef GET_INSTRINFO_NAMED_OPS\n";
  OS << "namespace llvm::" << Namespace << " {\n";

  emitGetInstructionIndexForOpLookup(OS, OperandMap, InstructionIndex);

  emitGetNamedOperandIdx(OS, OperandMap, MaxOperandNo, NumOperandNames);
  emitGetOperandIdxName(OS, OperandNameToID, OperandMap, MaxNumOperands,
                        NumOperandNames);

  OS << "} // end namespace llvm::" << Namespace << '\n';
  OS << "#endif //GET_INSTRINFO_NAMED_OPS\n\n";
}

/// Generate an enum for all the operand types for this target, under the
/// llvm::TargetNamespace::OpTypes namespace.
/// Operand types are all definitions derived of the Operand Target.td class.
///
void InstrInfoEmitter::emitOperandTypeMappings(
    raw_ostream &OS, const CodeGenTarget &Target,
    ArrayRef<const CodeGenInstruction *> NumberedInstructions) {
  StringRef Namespace = Target.getInstNamespace();

  // These generated functions are used only by the X86 target
  // (in bolt/lib/Target/X86/X86MCPlusBuilder.cpp). So emit them only
  // for X86.
  if (Namespace != "X86")
    return;

  ArrayRef<const Record *> Operands =
      Records.getAllDerivedDefinitions("Operand");
  ArrayRef<const Record *> RegisterOperands =
      Records.getAllDerivedDefinitions("RegisterOperand");
  ArrayRef<const Record *> RegisterClasses =
      Records.getAllDerivedDefinitions("RegisterClass");

  OS << "#ifdef GET_INSTRINFO_OPERAND_TYPES_ENUM\n";
  OS << "#undef GET_INSTRINFO_OPERAND_TYPES_ENUM\n";
  OS << "namespace llvm::" << Namespace << "::OpTypes {\n";
  OS << "enum OperandType {\n";

  unsigned EnumVal = 0;
  for (ArrayRef<const Record *> RecordsToAdd :
       {Operands, RegisterOperands, RegisterClasses}) {
    for (const Record *Op : RecordsToAdd) {
      if (!Op->isAnonymous())
        OS << "  " << Op->getName() << " = " << EnumVal << ",\n";
      ++EnumVal;
    }
  }

  OS << "  OPERAND_TYPE_LIST_END"
     << "\n};\n";
  OS << "} // end namespace llvm::" << Namespace << "::OpTypes\n";
  OS << "#endif // GET_INSTRINFO_OPERAND_TYPES_ENUM\n\n";

  OS << "#ifdef GET_INSTRINFO_OPERAND_TYPE\n";
  OS << "#undef GET_INSTRINFO_OPERAND_TYPE\n";
  OS << "namespace llvm::" << Namespace << " {\n";
  OS << "LLVM_READONLY\n";
  OS << "static int getOperandType(uint16_t Opcode, uint16_t OpIdx) {\n";
  auto getInstrName = [&](int I) -> StringRef {
    return NumberedInstructions[I]->TheDef->getName();
  };
  // TODO: Factor out duplicate operand lists to compress the tables.
  std::vector<size_t> OperandOffsets;
  std::vector<const Record *> OperandRecords;
  size_t CurrentOffset = 0;
  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    OperandOffsets.push_back(CurrentOffset);
    for (const auto &Op : Inst->Operands) {
      const DagInit *MIOI = Op.MIOperandInfo;
      if (!ExpandMIOperandInfo || !MIOI || MIOI->getNumArgs() == 0) {
        // Single, anonymous, operand.
        OperandRecords.push_back(Op.Rec);
        ++CurrentOffset;
      } else {
        for (const Init *Arg : MIOI->getArgs()) {
          OperandRecords.push_back(cast<DefInit>(Arg)->getDef());
          ++CurrentOffset;
        }
      }
    }
  }

  // Emit the table of offsets (indexes) into the operand type table.
  // Size the unsigned integer offset to save space.
  assert(OperandRecords.size() <= UINT32_MAX &&
         "Too many operands for offset table");
  OS << "  static constexpr " << getMinimalTypeForRange(OperandRecords.size());
  OS << " Offsets[] = {\n";
  for (const auto &[Idx, Offset] : enumerate(OperandOffsets))
    OS << "    " << Offset << ", // " << getInstrName(Idx) << '\n';
  OS << "  };\n";

  // Add an entry for the end so that we don't need to special case it below.
  OperandOffsets.push_back(OperandRecords.size());

  // Emit the actual operand types in a flat table.
  // Size the signed integer operand type to save space.
  assert(EnumVal <= INT16_MAX &&
         "Too many operand types for operand types table");
  OS << "\n  using namespace OpTypes;\n";
  OS << "  static";
  OS << (EnumVal <= INT8_MAX ? " constexpr int8_t" : " constexpr int16_t");
  OS << " OpcodeOperandTypes[] = {";
  size_t CurOffset = 0;
  for (auto [Idx, OpR] : enumerate(OperandRecords)) {
    // We print each Opcode's operands in its own row.
    if (Idx == OperandOffsets[CurOffset]) {
      OS << "\n    /* " << getInstrName(CurOffset) << " */\n    ";
      while (OperandOffsets[++CurOffset] == Idx)
        OS << "/* " << getInstrName(CurOffset) << " */\n    ";
    }
    if ((OpR->isSubClassOf("Operand") || OpR->isSubClassOf("RegisterOperand") ||
         OpR->isSubClassOf("RegisterClass")) &&
        !OpR->isAnonymous())
      OS << OpR->getName();
    else
      OS << -1;
    OS << ", ";
  }
  OS << "\n  };\n";

  OS << "  return OpcodeOperandTypes[Offsets[Opcode] + OpIdx];\n";
  OS << "}\n";
  OS << "} // end namespace llvm::" << Namespace << '\n';
  OS << "#endif // GET_INSTRINFO_OPERAND_TYPE\n\n";

  OS << "#ifdef GET_INSTRINFO_MEM_OPERAND_SIZE\n";
  OS << "#undef GET_INSTRINFO_MEM_OPERAND_SIZE\n";
  OS << "namespace llvm::" << Namespace << " {\n";
  OS << "LLVM_READONLY\n";
  OS << "static int getMemOperandSize(int OpType) {\n";
  OS << "  switch (OpType) {\n";
  std::map<int, SmallVector<StringRef, 0>> SizeToOperandName;
  for (const Record *Op : Operands) {
    if (!Op->isSubClassOf("X86MemOperand"))
      continue;
    if (int Size = Op->getValueAsInt("Size"))
      SizeToOperandName[Size].push_back(Op->getName());
  }
  OS << "  default: return 0;\n";
  for (const auto &[Size, OperandNames] : SizeToOperandName) {
    for (const StringRef &OperandName : OperandNames)
      OS << "  case OpTypes::" << OperandName << ":\n";
    OS << "    return " << Size << ";\n\n";
  }
  OS << "  }\n}\n";
  OS << "} // end namespace llvm::" << Namespace << '\n';
  OS << "#endif // GET_INSTRINFO_MEM_OPERAND_SIZE\n\n";
}

// Fixed/Predefined instructions do not have UseLogicalOperandMappings
// enabled, so we can just skip them. Hence accept TargetInstructions.
void InstrInfoEmitter::emitLogicalOperandSizeMappings(
    raw_ostream &OS, StringRef Namespace,
    ArrayRef<const CodeGenInstruction *> TargetInstructions) {
  std::map<std::vector<unsigned>, unsigned> LogicalOpSizeMap;
  std::map<unsigned, std::vector<std::string>> InstMap;

  size_t LogicalOpListSize = 0U;
  std::vector<unsigned> LogicalOpList;

  for (const auto *Inst : TargetInstructions) {
    if (!Inst->TheDef->getValueAsBit("UseLogicalOperandMappings"))
      continue;

    LogicalOpList.clear();
    llvm::transform(Inst->Operands, std::back_inserter(LogicalOpList),
                    [](const CGIOperandList::OperandInfo &Op) -> unsigned {
                      auto *MIOI = Op.MIOperandInfo;
                      if (!MIOI || MIOI->getNumArgs() == 0)
                        return 1;
                      return MIOI->getNumArgs();
                    });
    LogicalOpListSize = std::max(LogicalOpList.size(), LogicalOpListSize);

    auto I =
        LogicalOpSizeMap.try_emplace(LogicalOpList, LogicalOpSizeMap.size())
            .first;
    InstMap[I->second].push_back(
        (Namespace + "::" + Inst->TheDef->getName()).str());
  }

  OS << "#ifdef GET_INSTRINFO_LOGICAL_OPERAND_SIZE_MAP\n";
  OS << "#undef GET_INSTRINFO_LOGICAL_OPERAND_SIZE_MAP\n";
  OS << "namespace llvm::" << Namespace << " {\n";
  OS << "LLVM_READONLY static unsigned\n";
  OS << "getLogicalOperandSize(uint16_t Opcode, uint16_t LogicalOpIdx) {\n";
  if (!InstMap.empty()) {
    std::vector<const std::vector<unsigned> *> LogicalOpSizeList(
        LogicalOpSizeMap.size());
    for (auto &P : LogicalOpSizeMap) {
      LogicalOpSizeList[P.second] = &P.first;
    }
    OS << "  static const unsigned SizeMap[][" << LogicalOpListSize
       << "] = {\n";
    for (auto &R : LogicalOpSizeList) {
      const auto &Row = *R;
      OS << "   {";
      int i;
      for (i = 0; i < static_cast<int>(Row.size()); ++i) {
        OS << Row[i] << ", ";
      }
      for (; i < static_cast<int>(LogicalOpListSize); ++i) {
        OS << "0, ";
      }
      OS << "}, \n";
    }
    OS << "  };\n";

    OS << "  switch (Opcode) {\n";
    OS << "  default: return LogicalOpIdx;\n";
    for (auto &P : InstMap) {
      auto OpMapIdx = P.first;
      const auto &Insts = P.second;
      for (const auto &Inst : Insts) {
        OS << "  case " << Inst << ":\n";
      }
      OS << "    return SizeMap[" << OpMapIdx << "][LogicalOpIdx];\n";
    }
    OS << "  }\n";
  } else {
    OS << "  return LogicalOpIdx;\n";
  }
  OS << "}\n";

  OS << "LLVM_READONLY static inline unsigned\n";
  OS << "getLogicalOperandIdx(uint16_t Opcode, uint16_t LogicalOpIdx) {\n";
  OS << "  auto S = 0U;\n";
  OS << "  for (auto i = 0U; i < LogicalOpIdx; ++i)\n";
  OS << "    S += getLogicalOperandSize(Opcode, i);\n";
  OS << "  return S;\n";
  OS << "}\n";

  OS << "} // end namespace llvm::" << Namespace << '\n';
  OS << "#endif // GET_INSTRINFO_LOGICAL_OPERAND_SIZE_MAP\n\n";
}

void InstrInfoEmitter::emitMCIIHelperMethods(raw_ostream &OS,
                                             StringRef TargetName) {
  ArrayRef<const Record *> TIIPredicates =
      Records.getAllDerivedDefinitions("TIIPredicate");

  OS << "#ifdef GET_INSTRINFO_MC_HELPER_DECLS\n";
  OS << "#undef GET_INSTRINFO_MC_HELPER_DECLS\n\n";

  OS << "namespace llvm {\n";
  OS << "class MCInst;\n";
  OS << "class FeatureBitset;\n\n";

  OS << "namespace " << TargetName << "_MC {\n\n";

  for (const Record *Rec : TIIPredicates) {
    OS << "bool " << Rec->getValueAsString("FunctionName")
       << "(const MCInst &MI);\n";
  }

  OS << "void verifyInstructionPredicates(unsigned Opcode, const FeatureBitset "
        "&Features);\n";

  OS << "\n} // end namespace " << TargetName << "_MC\n";
  OS << "} // end namespace llvm\n\n";

  OS << "#endif // GET_INSTRINFO_MC_HELPER_DECLS\n\n";

  OS << "#ifdef GET_INSTRINFO_MC_HELPERS\n";
  OS << "#undef GET_INSTRINFO_MC_HELPERS\n\n";

  OS << "namespace llvm::" << TargetName << "_MC {\n";

  PredicateExpander PE(TargetName);
  PE.setExpandForMC(true);

  for (const Record *Rec : TIIPredicates) {
    OS << "bool " << Rec->getValueAsString("FunctionName");
    OS << "(const MCInst &MI) {\n";

    OS << PE.getIndent();
    PE.expandStatement(OS, Rec->getValueAsDef("Body"));
    OS << "\n}\n\n";
  }

  OS << "} // end namespace llvm::" << TargetName << "_MC\n";

  OS << "#endif // GET_GENISTRINFO_MC_HELPERS\n\n";
}

static std::string
getNameForFeatureBitset(ArrayRef<const Record *> FeatureBitset) {
  std::string Name = "CEFBS";
  for (const Record *Feature : FeatureBitset)
    Name += ("_" + Feature->getName()).str();
  return Name;
}

void InstrInfoEmitter::emitFeatureVerifier(raw_ostream &OS,
                                           const CodeGenTarget &Target) {
  const auto &All = SubtargetFeatureInfo::getAll(Records);
  SubtargetFeatureInfoMap SubtargetFeatures;
  SubtargetFeatures.insert(All.begin(), All.end());

  OS << "#if (defined(ENABLE_INSTR_PREDICATE_VERIFIER) && !defined(NDEBUG)) "
     << "||\\\n"
     << "    defined(GET_AVAILABLE_OPCODE_CHECKER)\n"
     << "#define GET_COMPUTE_FEATURES\n"
     << "#endif\n";
  OS << "#ifdef GET_COMPUTE_FEATURES\n"
     << "#undef GET_COMPUTE_FEATURES\n"
     << "namespace llvm::" << Target.getName() << "_MC {\n";

  // Emit the subtarget feature enumeration.
  SubtargetFeatureInfo::emitSubtargetFeatureBitEnumeration(SubtargetFeatures,
                                                           OS);
  // Emit the available features compute function.
  OS << "inline ";
  SubtargetFeatureInfo::emitComputeAssemblerAvailableFeatures(
      Target.getName(), "", "computeAvailableFeatures", SubtargetFeatures, OS);

  std::vector<std::vector<const Record *>> FeatureBitsets;
  for (const CodeGenInstruction *Inst : Target.getInstructions()) {
    FeatureBitsets.emplace_back();
    for (const Record *Predicate :
         Inst->TheDef->getValueAsListOfDefs("Predicates")) {
      const auto &I = SubtargetFeatures.find(Predicate);
      if (I != SubtargetFeatures.end())
        FeatureBitsets.back().push_back(I->second.TheDef);
    }
  }

  llvm::sort(FeatureBitsets, [&](ArrayRef<const Record *> A,
                                 ArrayRef<const Record *> B) {
    if (A.size() < B.size())
      return true;
    if (A.size() > B.size())
      return false;
    for (auto Pair : zip(A, B)) {
      if (std::get<0>(Pair)->getName() < std::get<1>(Pair)->getName())
        return true;
      if (std::get<0>(Pair)->getName() > std::get<1>(Pair)->getName())
        return false;
    }
    return false;
  });
  FeatureBitsets.erase(llvm::unique(FeatureBitsets), FeatureBitsets.end());
  OS << "inline FeatureBitset computeRequiredFeatures(unsigned Opcode) {\n"
     << "  enum : " << getMinimalTypeForRange(FeatureBitsets.size()) << " {\n"
     << "    CEFBS_None,\n";
  for (const auto &FeatureBitset : FeatureBitsets) {
    if (FeatureBitset.empty())
      continue;
    OS << "    " << getNameForFeatureBitset(FeatureBitset) << ",\n";
  }
  OS << "  };\n\n"
     << "  static constexpr FeatureBitset FeatureBitsets[] = {\n"
     << "    {}, // CEFBS_None\n";
  for (const auto &FeatureBitset : FeatureBitsets) {
    if (FeatureBitset.empty())
      continue;
    OS << "    {";
    for (const auto &Feature : FeatureBitset) {
      const auto &I = SubtargetFeatures.find(Feature);
      assert(I != SubtargetFeatures.end() && "Didn't import predicate?");
      OS << I->second.getEnumBitName() << ", ";
    }
    OS << "},\n";
  }
  OS << "  };\n"
     << "  static constexpr " << getMinimalTypeForRange(FeatureBitsets.size())
     << " RequiredFeaturesRefs[] = {\n";
  unsigned InstIdx = 0;
  for (const CodeGenInstruction *Inst : Target.getInstructions()) {
    OS << "    CEFBS";
    unsigned NumPredicates = 0;
    for (const Record *Predicate :
         Inst->TheDef->getValueAsListOfDefs("Predicates")) {
      const auto &I = SubtargetFeatures.find(Predicate);
      if (I != SubtargetFeatures.end()) {
        OS << '_' << I->second.TheDef->getName();
        NumPredicates++;
      }
    }
    if (!NumPredicates)
      OS << "_None";
    OS << ", // " << Inst->TheDef->getName() << " = " << InstIdx << '\n';
    InstIdx++;
  }
  OS << "  };\n\n"
     << "  assert(Opcode < " << InstIdx << ");\n"
     << "  return FeatureBitsets[RequiredFeaturesRefs[Opcode]];\n"
     << "}\n\n";

  OS << "} // end namespace llvm::" << Target.getName() << "_MC\n"
     << "#endif // GET_COMPUTE_FEATURES\n\n";

  OS << "#ifdef GET_AVAILABLE_OPCODE_CHECKER\n"
     << "#undef GET_AVAILABLE_OPCODE_CHECKER\n"
     << "namespace llvm::" << Target.getName() << "_MC {\n";
  OS << "bool isOpcodeAvailable("
     << "unsigned Opcode, const FeatureBitset &Features) {\n"
     << "  FeatureBitset AvailableFeatures = "
     << "computeAvailableFeatures(Features);\n"
     << "  FeatureBitset RequiredFeatures = "
     << "computeRequiredFeatures(Opcode);\n"
     << "  FeatureBitset MissingFeatures =\n"
     << "      (AvailableFeatures & RequiredFeatures) ^\n"
     << "      RequiredFeatures;\n"
     << "  return !MissingFeatures.any();\n"
     << "}\n";
  OS << "} // end namespace llvm::" << Target.getName() << "_MC\n"
     << "#endif // GET_AVAILABLE_OPCODE_CHECKER\n\n";

  OS << "#ifdef ENABLE_INSTR_PREDICATE_VERIFIER\n"
     << "#undef ENABLE_INSTR_PREDICATE_VERIFIER\n"
     << "#include <sstream>\n\n";

  OS << "namespace llvm::" << Target.getName() << "_MC {\n";

  // Emit the name table for error messages.
  OS << "#ifndef NDEBUG\n";
  SubtargetFeatureInfo::emitNameTable(SubtargetFeatures, OS);
  OS << "#endif // NDEBUG\n\n";

  // Emit the predicate verifier.
  OS << "void verifyInstructionPredicates(\n"
     << "    unsigned Opcode, const FeatureBitset &Features) {\n"
     << "#ifndef NDEBUG\n";
  OS << "  FeatureBitset AvailableFeatures = "
        "computeAvailableFeatures(Features);\n";
  OS << "  FeatureBitset RequiredFeatures = "
     << "computeRequiredFeatures(Opcode);\n";
  OS << "  FeatureBitset MissingFeatures =\n"
     << "      (AvailableFeatures & RequiredFeatures) ^\n"
     << "      RequiredFeatures;\n"
     << "  if (MissingFeatures.any()) {\n"
     << "    std::ostringstream Msg;\n"
     << "    Msg << \"Attempting to emit \" << &" << Target.getName()
     << "InstrNameData[" << Target.getName() << "InstrNameIndices[Opcode]]\n"
     << "        << \" instruction but the \";\n"
     << "    for (unsigned i = 0, e = MissingFeatures.size(); i != e; ++i)\n"
     << "      if (MissingFeatures.test(i))\n"
     << "        Msg << SubtargetFeatureNames[i] << \" \";\n"
     << "    Msg << \"predicate(s) are not met\";\n"
     << "    report_fatal_error(Msg.str().c_str());\n"
     << "  }\n"
     << "#endif // NDEBUG\n";
  OS << "}\n";
  OS << "} // end namespace llvm::" << Target.getName() << "_MC\n";
  OS << "#endif // ENABLE_INSTR_PREDICATE_VERIFIER\n\n";
}

void InstrInfoEmitter::emitTIIHelperMethods(raw_ostream &OS,
                                            StringRef TargetName,
                                            bool ExpandDefinition) {
  ArrayRef<const Record *> TIIPredicates =
      Records.getAllDerivedDefinitions("TIIPredicate");
  if (TIIPredicates.empty())
    return;

  PredicateExpander PE(TargetName);
  PE.setExpandForMC(false);

  for (const Record *Rec : TIIPredicates) {
    OS << (ExpandDefinition ? "" : "static ") << "bool ";
    if (ExpandDefinition)
      OS << TargetName << "InstrInfo::";
    OS << Rec->getValueAsString("FunctionName");
    OS << "(const MachineInstr &MI)";
    if (!ExpandDefinition) {
      OS << ";\n";
      continue;
    }

    OS << " {\n";
    OS << PE.getIndent();
    PE.expandStatement(OS, Rec->getValueAsDef("Body"));
    OS << "\n}\n\n";
  }
}

//===----------------------------------------------------------------------===//
// Main Output.
//===----------------------------------------------------------------------===//

// run - Emit the main instruction description records for the target...
void InstrInfoEmitter::run(raw_ostream &OS) {
  TGTimer &Timer = Records.getTimer();
  Timer.startTimer("Analyze DAG patterns");

  emitSourceFileHeader("Target Instruction Enum Values and Descriptors", OS);

  const CodeGenTarget &Target = CDP.getTargetInfo();
  ArrayRef<const CodeGenInstruction *> NumberedInstructions =
      Target.getInstructions();

  emitEnums(OS, NumberedInstructions);

  StringRef TargetName = Target.getName();
  const Record *InstrInfo = Target.getInstructionSet();

  // Collect all of the operand info records.
  Timer.startTimer("Collect operand info");
  OperandInfoListTy OperandInfoList;
  OperandInfoMapTy OperandInfoMap;
  unsigned OperandInfoSize =
      CollectOperandInfo(OperandInfoList, OperandInfoMap);

  // Collect all of the instruction's implicit uses and defs.
  // Also collect which features are enabled by instructions to control
  // emission of various mappings.

  bool HasUseLogicalOperandMappings = false;
  bool HasUseNamedOperandTable = false;

  Timer.startTimer("Collect uses/defs");
  std::map<std::vector<const Record *>, unsigned> EmittedLists;
  std::vector<std::vector<const Record *>> ImplicitLists;
  unsigned ImplicitListSize = 0;
  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    HasUseLogicalOperandMappings |=
        Inst->TheDef->getValueAsBit("UseLogicalOperandMappings");
    HasUseNamedOperandTable |=
        Inst->TheDef->getValueAsBit("UseNamedOperandTable");

    std::vector<const Record *> ImplicitOps = Inst->ImplicitUses;
    llvm::append_range(ImplicitOps, Inst->ImplicitDefs);
    if (EmittedLists.try_emplace(ImplicitOps, ImplicitListSize).second) {
      ImplicitLists.push_back(ImplicitOps);
      ImplicitListSize += ImplicitOps.size();
    }
  }

  OS << "#if defined(GET_INSTRINFO_MC_DESC) || "
        "defined(GET_INSTRINFO_CTOR_DTOR)\n";
  OS << "namespace llvm {\n\n";

  OS << "struct " << TargetName << "InstrTable {\n";
  OS << "  MCInstrDesc Insts[" << NumberedInstructions.size() << "];\n";
  OS << "  static_assert(alignof(MCInstrDesc) >= alignof(MCOperandInfo), "
        "\"Unwanted padding between Insts and OperandInfo\");\n";
  OS << "  MCOperandInfo OperandInfo[" << OperandInfoSize << "];\n";
  OS << "  static_assert(alignof(MCOperandInfo) >= alignof(MCPhysReg), "
        "\"Unwanted padding between OperandInfo and ImplicitOps\");\n";
  OS << "  MCPhysReg ImplicitOps[" << std::max(ImplicitListSize, 1U) << "];\n";
  OS << "};\n\n";

  OS << "} // end namespace llvm\n";
  OS << "#endif // defined(GET_INSTRINFO_MC_DESC) || "
        "defined(GET_INSTRINFO_CTOR_DTOR)\n\n";

  OS << "#ifdef GET_INSTRINFO_MC_DESC\n";
  OS << "#undef GET_INSTRINFO_MC_DESC\n";
  OS << "namespace llvm {\n\n";

  // Emit all of the MCInstrDesc records in reverse ENUM ordering.
  Timer.startTimer("Emit InstrDesc records");
  OS << "static_assert(sizeof(MCOperandInfo) % sizeof(MCPhysReg) == 0);\n";
  OS << "static constexpr unsigned " << TargetName << "ImpOpBase = sizeof "
     << TargetName << "InstrTable::OperandInfo / (sizeof(MCPhysReg));\n\n";

  OS << "extern const " << TargetName << "InstrTable " << TargetName
     << "Descs = {\n  {\n";
  SequenceToOffsetTable<StringRef> InstrNames;
  unsigned Num = NumberedInstructions.size();
  for (const CodeGenInstruction *Inst : reverse(NumberedInstructions)) {
    // Keep a list of the instruction names.
    InstrNames.add(Inst->TheDef->getName());
    // Emit the record into the table.
    emitRecord(*Inst, --Num, InstrInfo, EmittedLists, OperandInfoMap, OS);
  }

  OS << "  }, {\n";

  // Emit all of the operand info records.
  Timer.startTimer("Emit operand info");
  EmitOperandInfo(OS, OperandInfoList);

  OS << "  }, {\n";

  // Emit all of the instruction's implicit uses and defs.
  Timer.startTimer("Emit uses/defs");
  for (auto &List : ImplicitLists) {
    OS << "    /* " << EmittedLists[List] << " */";
    for (auto &Reg : List)
      OS << ' ' << getQualifiedName(Reg) << ',';
    OS << '\n';
  }

  OS << "  }\n};\n\n";

  // Emit the array of instruction names.
  Timer.startTimer("Emit instruction names");
  InstrNames.layout();
  InstrNames.emitStringLiteralDef(OS, Twine("extern const char ") + TargetName +
                                          "InstrNameData[]");

  OS << "extern const unsigned " << TargetName << "InstrNameIndices[] = {";
  Num = 0;
  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    // Newline every eight entries.
    if (Num % 8 == 0)
      OS << "\n    ";
    OS << InstrNames.get(Inst->TheDef->getName()) << "U, ";
    ++Num;
  }
  OS << "\n};\n\n";

  bool HasDeprecationFeatures =
      llvm::any_of(NumberedInstructions, [](const CodeGenInstruction *Inst) {
        return !Inst->HasComplexDeprecationPredicate &&
               !Inst->DeprecatedReason.empty();
      });
  if (HasDeprecationFeatures) {
    OS << "extern const uint8_t " << TargetName
       << "InstrDeprecationFeatures[] = {";
    Num = 0;
    for (const CodeGenInstruction *Inst : NumberedInstructions) {
      if (Num % 8 == 0)
        OS << "\n    ";
      if (!Inst->HasComplexDeprecationPredicate &&
          !Inst->DeprecatedReason.empty())
        OS << Target.getInstNamespace() << "::" << Inst->DeprecatedReason
           << ", ";
      else
        OS << "uint8_t(-1), ";
      ++Num;
    }
    OS << "\n};\n\n";
  }

  bool HasComplexDeprecationInfos =
      llvm::any_of(NumberedInstructions, [](const CodeGenInstruction *Inst) {
        return Inst->HasComplexDeprecationPredicate;
      });
  if (HasComplexDeprecationInfos) {
    OS << "extern const MCInstrInfo::ComplexDeprecationPredicate " << TargetName
       << "InstrComplexDeprecationInfos[] = {";
    Num = 0;
    for (const CodeGenInstruction *Inst : NumberedInstructions) {
      if (Num % 8 == 0)
        OS << "\n    ";
      if (Inst->HasComplexDeprecationPredicate)
        // Emit a function pointer to the complex predicate method.
        OS << "&get" << Inst->DeprecatedReason << "DeprecationInfo, ";
      else
        OS << "nullptr, ";
      ++Num;
    }
    OS << "\n};\n\n";
  }

  // MCInstrInfo initialization routine.
  Timer.startTimer("Emit initialization routine");
  OS << "static inline void Init" << TargetName
     << "MCInstrInfo(MCInstrInfo *II) {\n";
  OS << "  II->InitMCInstrInfo(" << TargetName << "Descs.Insts, " << TargetName
     << "InstrNameIndices, " << TargetName << "InstrNameData, ";
  if (HasDeprecationFeatures)
    OS << TargetName << "InstrDeprecationFeatures, ";
  else
    OS << "nullptr, ";
  if (HasComplexDeprecationInfos)
    OS << TargetName << "InstrComplexDeprecationInfos, ";
  else
    OS << "nullptr, ";
  OS << NumberedInstructions.size() << ");\n}\n\n";

  OS << "} // end namespace llvm\n";

  OS << "#endif // GET_INSTRINFO_MC_DESC\n\n";

  // Create a TargetInstrInfo subclass to hide the MC layer initialization.
  OS << "#ifdef GET_INSTRINFO_HEADER\n";
  OS << "#undef GET_INSTRINFO_HEADER\n";

  Twine ClassName = TargetName + "GenInstrInfo";
  OS << "namespace llvm {\n";
  OS << "struct " << ClassName << " : public TargetInstrInfo {\n"
     << "  explicit " << ClassName
     << "(unsigned CFSetupOpcode = ~0u, unsigned CFDestroyOpcode = ~0u, "
        "unsigned CatchRetOpcode = ~0u, unsigned ReturnOpcode = ~0u);\n"
     << "  ~" << ClassName << "() override = default;\n";

  OS << "\n};\n} // end namespace llvm\n";

  OS << "#endif // GET_INSTRINFO_HEADER\n\n";

  OS << "#ifdef GET_INSTRINFO_HELPER_DECLS\n";
  OS << "#undef GET_INSTRINFO_HELPER_DECLS\n\n";
  emitTIIHelperMethods(OS, TargetName, /* ExpandDefinition = */ false);
  OS << '\n';
  OS << "#endif // GET_INSTRINFO_HELPER_DECLS\n\n";

  OS << "#ifdef GET_INSTRINFO_HELPERS\n";
  OS << "#undef GET_INSTRINFO_HELPERS\n\n";
  emitTIIHelperMethods(OS, TargetName, /* ExpandDefinition = */ true);
  OS << "#endif // GET_INSTRINFO_HELPERS\n\n";

  OS << "#ifdef GET_INSTRINFO_CTOR_DTOR\n";
  OS << "#undef GET_INSTRINFO_CTOR_DTOR\n";

  OS << "namespace llvm {\n";
  OS << "extern const " << TargetName << "InstrTable " << TargetName
     << "Descs;\n";
  OS << "extern const unsigned " << TargetName << "InstrNameIndices[];\n";
  OS << "extern const char " << TargetName << "InstrNameData[];\n";
  if (HasDeprecationFeatures)
    OS << "extern const uint8_t " << TargetName
       << "InstrDeprecationFeatures[];\n";
  if (HasComplexDeprecationInfos)
    OS << "extern const MCInstrInfo::ComplexDeprecationPredicate " << TargetName
       << "InstrComplexDeprecationInfos[];\n";
  OS << ClassName << "::" << ClassName
     << "(unsigned CFSetupOpcode, unsigned CFDestroyOpcode, unsigned "
        "CatchRetOpcode, unsigned ReturnOpcode)\n"
     << "  : TargetInstrInfo(CFSetupOpcode, CFDestroyOpcode, CatchRetOpcode, "
        "ReturnOpcode) {\n"
     << "  InitMCInstrInfo(" << TargetName << "Descs.Insts, " << TargetName
     << "InstrNameIndices, " << TargetName << "InstrNameData, ";
  if (HasDeprecationFeatures)
    OS << TargetName << "InstrDeprecationFeatures, ";
  else
    OS << "nullptr, ";
  if (HasComplexDeprecationInfos)
    OS << TargetName << "InstrComplexDeprecationInfos, ";
  else
    OS << "nullptr, ";
  OS << NumberedInstructions.size() << ");\n}\n";
  OS << "} // end namespace llvm\n";

  OS << "#endif // GET_INSTRINFO_CTOR_DTOR\n\n";

  ArrayRef<const CodeGenInstruction *> TargetInstructions =
      Target.getTargetInstructions();

  if (HasUseNamedOperandTable) {
    Timer.startTimer("Emit operand name mappings");
    emitOperandNameMappings(OS, Target, TargetInstructions);
  }

  Timer.startTimer("Emit operand type mappings");
  emitOperandTypeMappings(OS, Target, NumberedInstructions);

  if (HasUseLogicalOperandMappings) {
    Timer.startTimer("Emit logical operand size mappings");
    emitLogicalOperandSizeMappings(OS, TargetName, TargetInstructions);
  }

  Timer.startTimer("Emit helper methods");
  emitMCIIHelperMethods(OS, TargetName);

  Timer.startTimer("Emit verifier methods");
  emitFeatureVerifier(OS, Target);

  Timer.startTimer("Emit map table");
  EmitMapTable(Records, OS);
}

void InstrInfoEmitter::emitRecord(
    const CodeGenInstruction &Inst, unsigned Num, const Record *InstrInfo,
    std::map<std::vector<const Record *>, unsigned> &EmittedLists,
    const OperandInfoMapTy &OperandInfoMap, raw_ostream &OS) {
  int MinOperands = 0;
  if (!Inst.Operands.empty())
    // Each logical operand can be multiple MI operands.
    MinOperands =
        Inst.Operands.back().MIOperandNo + Inst.Operands.back().MINumOperands;
  // Even the logical output operand may be multiple MI operands.
  int DefOperands = 0;
  if (Inst.Operands.NumDefs) {
    auto &Opnd = Inst.Operands[Inst.Operands.NumDefs - 1];
    DefOperands = Opnd.MIOperandNo + Opnd.MINumOperands;
  }

  OS << "    { ";
  OS << Num << ",\t" << MinOperands << ",\t" << DefOperands << ",\t"
     << Inst.TheDef->getValueAsInt("Size") << ",\t"
     << SchedModels.getSchedClassIdx(Inst) << ",\t";

  const CodeGenTarget &Target = CDP.getTargetInfo();

  // Emit the implicit use/def list...
  OS << Inst.ImplicitUses.size() << ",\t" << Inst.ImplicitDefs.size() << ",\t";
  std::vector<const Record *> ImplicitOps = Inst.ImplicitUses;
  llvm::append_range(ImplicitOps, Inst.ImplicitDefs);

  // Emit the operand info offset.
  OperandInfoTy OperandInfo = GetOperandInfo(Inst);
  OS << OperandInfoMap.find(OperandInfo)->second << ",\t";

  // Emit implicit operand base.
  OS << Target.getName() << "ImpOpBase + " << EmittedLists[ImplicitOps]
     << ",\t0";

  // Emit all of the target independent flags...
  if (Inst.isPreISelOpcode)
    OS << "|(1ULL<<MCID::PreISelOpcode)";
  if (Inst.isPseudo)
    OS << "|(1ULL<<MCID::Pseudo)";
  if (Inst.isMeta)
    OS << "|(1ULL<<MCID::Meta)";
  if (Inst.isReturn)
    OS << "|(1ULL<<MCID::Return)";
  if (Inst.isEHScopeReturn)
    OS << "|(1ULL<<MCID::EHScopeReturn)";
  if (Inst.isBranch)
    OS << "|(1ULL<<MCID::Branch)";
  if (Inst.isIndirectBranch)
    OS << "|(1ULL<<MCID::IndirectBranch)";
  if (Inst.isCompare)
    OS << "|(1ULL<<MCID::Compare)";
  if (Inst.isMoveImm)
    OS << "|(1ULL<<MCID::MoveImm)";
  if (Inst.isMoveReg)
    OS << "|(1ULL<<MCID::MoveReg)";
  if (Inst.isBitcast)
    OS << "|(1ULL<<MCID::Bitcast)";
  if (Inst.isAdd)
    OS << "|(1ULL<<MCID::Add)";
  if (Inst.isTrap)
    OS << "|(1ULL<<MCID::Trap)";
  if (Inst.isSelect)
    OS << "|(1ULL<<MCID::Select)";
  if (Inst.isBarrier)
    OS << "|(1ULL<<MCID::Barrier)";
  if (Inst.hasDelaySlot)
    OS << "|(1ULL<<MCID::DelaySlot)";
  if (Inst.isCall)
    OS << "|(1ULL<<MCID::Call)";
  if (Inst.canFoldAsLoad)
    OS << "|(1ULL<<MCID::FoldableAsLoad)";
  if (Inst.mayLoad)
    OS << "|(1ULL<<MCID::MayLoad)";
  if (Inst.mayStore)
    OS << "|(1ULL<<MCID::MayStore)";
  if (Inst.mayRaiseFPException)
    OS << "|(1ULL<<MCID::MayRaiseFPException)";
  if (Inst.isPredicable)
    OS << "|(1ULL<<MCID::Predicable)";
  if (Inst.isConvertibleToThreeAddress)
    OS << "|(1ULL<<MCID::ConvertibleTo3Addr)";
  if (Inst.isCommutable)
    OS << "|(1ULL<<MCID::Commutable)";
  if (Inst.isTerminator)
    OS << "|(1ULL<<MCID::Terminator)";
  if (Inst.isReMaterializable)
    OS << "|(1ULL<<MCID::Rematerializable)";
  if (Inst.isNotDuplicable)
    OS << "|(1ULL<<MCID::NotDuplicable)";
  if (Inst.Operands.hasOptionalDef)
    OS << "|(1ULL<<MCID::HasOptionalDef)";
  if (Inst.usesCustomInserter)
    OS << "|(1ULL<<MCID::UsesCustomInserter)";
  if (Inst.hasPostISelHook)
    OS << "|(1ULL<<MCID::HasPostISelHook)";
  if (Inst.Operands.isVariadic)
    OS << "|(1ULL<<MCID::Variadic)";
  if (Inst.hasSideEffects)
    OS << "|(1ULL<<MCID::UnmodeledSideEffects)";
  if (Inst.isAsCheapAsAMove)
    OS << "|(1ULL<<MCID::CheapAsAMove)";
  if (!Target.getAllowRegisterRenaming() || Inst.hasExtraSrcRegAllocReq)
    OS << "|(1ULL<<MCID::ExtraSrcRegAllocReq)";
  if (!Target.getAllowRegisterRenaming() || Inst.hasExtraDefRegAllocReq)
    OS << "|(1ULL<<MCID::ExtraDefRegAllocReq)";
  if (Inst.isRegSequence)
    OS << "|(1ULL<<MCID::RegSequence)";
  if (Inst.isExtractSubreg)
    OS << "|(1ULL<<MCID::ExtractSubreg)";
  if (Inst.isInsertSubreg)
    OS << "|(1ULL<<MCID::InsertSubreg)";
  if (Inst.isConvergent)
    OS << "|(1ULL<<MCID::Convergent)";
  if (Inst.variadicOpsAreDefs)
    OS << "|(1ULL<<MCID::VariadicOpsAreDefs)";
  if (Inst.isAuthenticated)
    OS << "|(1ULL<<MCID::Authenticated)";

  // Emit all of the target-specific flags...
  const BitsInit *TSF = Inst.TheDef->getValueAsBitsInit("TSFlags");
  if (!TSF)
    PrintFatalError(Inst.TheDef->getLoc(), "no TSFlags?");
  uint64_t Value = 0;
  for (unsigned i = 0, e = TSF->getNumBits(); i != e; ++i) {
    if (const auto *Bit = dyn_cast<BitInit>(TSF->getBit(i)))
      Value |= uint64_t(Bit->getValue()) << i;
    else
      PrintFatalError(Inst.TheDef->getLoc(),
                      "Invalid TSFlags bit in " + Inst.TheDef->getName());
  }
  OS << ", 0x";
  OS.write_hex(Value);
  OS << "ULL";

  OS << " },  // Inst #" << Num << " = " << Inst.TheDef->getName() << '\n';
}

// emitEnums - Print out enum values for all of the instructions.
void InstrInfoEmitter::emitEnums(
    raw_ostream &OS,
    ArrayRef<const CodeGenInstruction *> NumberedInstructions) {
  OS << "#ifdef GET_INSTRINFO_ENUM\n";
  OS << "#undef GET_INSTRINFO_ENUM\n";

  const CodeGenTarget &Target = CDP.getTargetInfo();
  StringRef Namespace = Target.getInstNamespace();

  if (Namespace.empty())
    PrintFatalError("No instructions defined!");

  OS << "namespace llvm::" << Namespace << " {\n";

  OS << "  enum {\n";
  for (const CodeGenInstruction *Inst : NumberedInstructions)
    OS << "    " << Inst->TheDef->getName()
       << "\t= " << Target.getInstrIntValue(Inst->TheDef) << ",\n";
  OS << "    INSTRUCTION_LIST_END = " << NumberedInstructions.size() << '\n';
  OS << "  };\n\n";
  OS << "} // end namespace llvm::" << Namespace << '\n';
  OS << "#endif // GET_INSTRINFO_ENUM\n\n";

  OS << "#ifdef GET_INSTRINFO_SCHED_ENUM\n";
  OS << "#undef GET_INSTRINFO_SCHED_ENUM\n";
  OS << "namespace llvm::" << Namespace << "::Sched {\n\n";
  OS << "  enum {\n";
  auto ExplictClasses = SchedModels.explicitSchedClasses();
  for (const auto &[Idx, Class] : enumerate(ExplictClasses))
    OS << "    " << Class.Name << "\t= " << Idx << ",\n";
  OS << "    SCHED_LIST_END = " << ExplictClasses.size() << '\n';
  OS << "  };\n";
  OS << "} // end namespace llvm::" << Namespace << "::Sched\n";

  OS << "#endif // GET_INSTRINFO_SCHED_ENUM\n\n";
}

static TableGen::Emitter::OptClass<InstrInfoEmitter>
    X("gen-instr-info", "Generate instruction descriptions");
