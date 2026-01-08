//===-------- CompressInstEmitter.cpp - Generator for Compression ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// CompressInstEmitter implements a tablegen-driven CompressPat based
// Instruction Compression mechanism.
//
//===----------------------------------------------------------------------===//
//
// CompressInstEmitter implements a tablegen-driven CompressPat Instruction
// Compression mechanism for generating compressed instructions from the
// expanded instruction form.

// This tablegen backend processes CompressPat declarations in a
// td file and generates all the required checks to validate the pattern
// declarations; validate the input and output operands to generate the correct
// compressed instructions. The checks include validating different types of
// operands; register operands, immediate operands, fixed register and fixed
// immediate inputs.
//
// Example:
// /// Defines a Pat match between compressed and uncompressed instruction.
// /// The relationship and helper function generation are handled by
// /// CompressInstEmitter backend.
// class CompressPat<dag input, dag output, list<Predicate> predicates = []> {
//   /// Uncompressed instruction description.
//   dag Input = input;
//   /// Compressed instruction description.
//   dag Output = output;
//   /// Predicates that must be true for this to match.
//   list<Predicate> Predicates = predicates;
//   /// Duplicate match when tied operand is just different.
//   bit isCompressOnly = false;
// }
//
// let Predicates = [HasStdExtC] in {
// def : CompressPat<(ADD GPRNoX0:$rs1, GPRNoX0:$rs1, GPRNoX0:$rs2),
//                   (C_ADD GPRNoX0:$rs1, GPRNoX0:$rs2)>;
// }
//
// The <TargetName>GenCompressInstEmitter.inc is an auto-generated header
// file which exports two functions for compressing/uncompressing MCInst
// instructions, plus some helper functions:
//
// bool compressInst(MCInst &OutInst, const MCInst &MI,
//                   const MCSubtargetInfo &STI);
//
// bool uncompressInst(MCInst &OutInst, const MCInst &MI,
//                     const MCSubtargetInfo &STI);
//
// In addition, it exports a function for checking whether
// an instruction is compressable:
//
// bool isCompressibleInst(const MachineInstr& MI,
//                         const <TargetName>Subtarget &STI);
//
// The clients that include this auto-generated header file and
// invoke these functions can compress an instruction before emitting
// it in the target-specific ASM or ELF streamer or can uncompress
// an instruction before printing it when the expanded instruction
// format aliases is favored.

//===----------------------------------------------------------------------===//

#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenRegisters.h"
#include "Common/CodeGenTarget.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <limits>
#include <set>
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "compress-inst-emitter"

namespace {
class CompressInstEmitter {
  struct OpData {
    enum MapKind { Operand, Imm, Reg } Kind;
    // Info for an operand.
    struct OpndInfo {
      // Record from the Dag.
      const Record *DagRec;
      // Operand number mapped to.
      unsigned Idx;
      // Tied operand index within the instruction.
      int TiedOpIdx;
    };
    union {
      OpndInfo OpInfo;
      // Integer immediate value.
      int64_t ImmVal;
      // Physical register.
      const Record *RegRec;
    };
  };
  struct ArgData {
    unsigned DAGOpNo;
    unsigned MIOpNo;
  };
  struct CompressPat {
    // The source instruction definition.
    CodeGenInstruction Source;
    // The destination instruction to transform to.
    CodeGenInstruction Dest;
    // Required target features to enable pattern.
    std::vector<const Record *> PatReqFeatures;
    // Maps operands in the Source Instruction to
    // the corresponding Dest instruction operand.
    IndexedMap<OpData> SourceOperandMap;
    // Maps operands in the Dest Instruction
    // to the corresponding Source instruction operand.
    IndexedMap<OpData> DestOperandMap;

    bool IsCompressOnly;
    CompressPat(const CodeGenInstruction &S, const CodeGenInstruction &D,
                std::vector<const Record *> RF,
                const IndexedMap<OpData> &SourceMap,
                const IndexedMap<OpData> &DestMap, bool IsCompressOnly)
        : Source(S), Dest(D), PatReqFeatures(std::move(RF)),
          SourceOperandMap(SourceMap), DestOperandMap(DestMap),
          IsCompressOnly(IsCompressOnly) {}
  };
  enum EmitterType { Compress, Uncompress, CheckCompress };
  const RecordKeeper &Records;
  const CodeGenTarget Target;
  std::vector<CompressPat> CompressPatterns;
  void addDagOperandMapping(const Record *Rec, const DagInit *Dag,
                            const CodeGenInstruction &Inst,
                            IndexedMap<OpData> &OperandMap,
                            StringMap<ArgData> &Operands, bool IsSourceInst);
  void evaluateCompressPat(const Record *Compress);
  void emitCompressInstEmitter(raw_ostream &OS, EmitterType EType);
  bool validateTypes(const Record *DagOpType, const Record *InstOpType,
                     bool IsSourceInst);
  bool validateRegister(const Record *Reg, const Record *RegClass);
  void checkDagOperandMapping(const Record *Rec,
                              const StringMap<ArgData> &DestOperands,
                              const DagInit *SourceDag, const DagInit *DestDag);

  void createInstOperandMapping(const Record *Rec, const DagInit *SourceDag,
                                const DagInit *DestDag,
                                IndexedMap<OpData> &SourceOperandMap,
                                IndexedMap<OpData> &DestOperandMap,
                                StringMap<ArgData> &SourceOperands,
                                const CodeGenInstruction &DestInst);

public:
  CompressInstEmitter(const RecordKeeper &R) : Records(R), Target(R) {}

  void run(raw_ostream &OS);
};
} // End anonymous namespace.

bool CompressInstEmitter::validateRegister(const Record *Reg,
                                           const Record *RegClass) {
  assert(Reg->isSubClassOf("Register") && "Reg record should be a Register");
  assert(RegClass->isSubClassOf("RegisterClass") &&
         "RegClass record should be a RegisterClass");
  const CodeGenRegisterClass &RC = Target.getRegisterClass(RegClass);
  const CodeGenRegister *R = Target.getRegBank().getReg(Reg);
  assert(R != nullptr && "Register not defined!!");
  return RC.contains(R);
}

bool CompressInstEmitter::validateTypes(const Record *DagOpType,
                                        const Record *InstOpType,
                                        bool IsSourceInst) {
  if (DagOpType == InstOpType)
    return true;

  if (DagOpType->isSubClassOf("RegisterClass") &&
      InstOpType->isSubClassOf("RegisterClass")) {
    const CodeGenRegisterClass &RC = Target.getRegisterClass(InstOpType);
    const CodeGenRegisterClass &SubRC = Target.getRegisterClass(DagOpType);
    return RC.hasSubClass(&SubRC);
  }

  // At this point either or both types are not registers, reject the pattern.
  if (DagOpType->isSubClassOf("RegisterClass") ||
      InstOpType->isSubClassOf("RegisterClass"))
    return false;

  // Let further validation happen when compress()/uncompress() functions are
  // invoked.
  LLVM_DEBUG(dbgs() << (IsSourceInst ? "Input" : "Output")
                    << " Dag Operand Type: '" << DagOpType->getName()
                    << "' and "
                    << "Instruction Operand Type: '" << InstOpType->getName()
                    << "' can't be checked at pattern validation time!\n");
  return true;
}

static bool validateArgsTypes(const Init *Arg1, const Init *Arg2) {
  return cast<DefInit>(Arg1)->getDef() == cast<DefInit>(Arg2)->getDef();
}

/// The patterns in the Dag contain different types of operands:
/// Register operands, e.g.: GPRC:$rs1; Fixed registers, e.g: X1; Immediate
/// operands, e.g.: simm6:$imm; Fixed immediate operands, e.g.: 0. This function
/// maps Dag operands to its corresponding instruction operands. For register
/// operands and fixed registers it expects the Dag operand type to be contained
/// in the instantiated instruction operand type. For immediate operands and
/// immediates no validation checks are enforced at pattern validation time.
void CompressInstEmitter::addDagOperandMapping(const Record *Rec,
                                               const DagInit *Dag,
                                               const CodeGenInstruction &Inst,
                                               IndexedMap<OpData> &OperandMap,
                                               StringMap<ArgData> &Operands,
                                               bool IsSourceInst) {
  unsigned NumMIOperands = 0;
  if (!Inst.Operands.empty())
    NumMIOperands =
        Inst.Operands.back().MIOperandNo + Inst.Operands.back().MINumOperands;
  OperandMap.grow(NumMIOperands);

  // Tied operands are not represented in the DAG so we count them separately.
  unsigned DAGOpNo = 0;
  unsigned OpNo = 0;
  for (const auto &Opnd : Inst.Operands) {
    int TiedOpIdx = Opnd.getTiedRegister();
    if (-1 != TiedOpIdx) {
      assert((unsigned)TiedOpIdx < OpNo);
      // Set the entry in OperandMap for the tied operand we're skipping.
      OperandMap[OpNo] = OperandMap[TiedOpIdx];
      ++OpNo;

      // Source instructions can have at most 1 tied operand.
      if (IsSourceInst && (OpNo - DAGOpNo > 1))
        PrintFatalError(Rec->getLoc(),
                        "Input operands for Inst '" + Inst.getName() +
                            "' and input Dag operand count mismatch");

      continue;
    }
    for (unsigned SubOp = 0; SubOp != Opnd.MINumOperands;
         ++SubOp, ++OpNo, ++DAGOpNo) {
      const Record *OpndRec = Opnd.Rec;
      if (Opnd.MINumOperands > 1)
        OpndRec = cast<DefInit>(Opnd.MIOperandInfo->getArg(SubOp))->getDef();

      if (DAGOpNo >= Dag->getNumArgs())
        PrintFatalError(Rec->getLoc(), "Inst '" + Inst.getName() +
                                           "' and Dag operand count mismatch");

      if (const auto *DI = dyn_cast<DefInit>(Dag->getArg(DAGOpNo))) {
        if (DI->getDef()->isSubClassOf("Register")) {
          // Check if the fixed register belongs to the Register class.
          if (!validateRegister(DI->getDef(), OpndRec))
            PrintFatalError(Rec->getLoc(),
                            "Error in Dag '" + Dag->getAsString() +
                                "'Register: '" + DI->getDef()->getName() +
                                "' is not in register class '" +
                                OpndRec->getName() + "'");
          OperandMap[OpNo].Kind = OpData::Reg;
          OperandMap[OpNo].RegRec = DI->getDef();
          continue;
        }
        // Validate that Dag operand type matches the type defined in the
        // corresponding instruction. Operands in the input and output Dag
        // patterns are allowed to be a subclass of the type specified in the
        // corresponding instruction operand instead of being an exact match.
        if (!validateTypes(DI->getDef(), OpndRec, IsSourceInst))
          PrintFatalError(Rec->getLoc(),
                          "Error in Dag '" + Dag->getAsString() +
                              "'. Operand '" + Dag->getArgNameStr(DAGOpNo) +
                              "' has type '" + DI->getDef()->getName() +
                              "' which does not match the type '" +
                              OpndRec->getName() +
                              "' in the corresponding instruction operand!");

        OperandMap[OpNo].Kind = OpData::Operand;
        OperandMap[OpNo].OpInfo.DagRec = DI->getDef();
        OperandMap[OpNo].OpInfo.TiedOpIdx = -1;

        // Create a mapping between the operand name in the Dag (e.g. $rs1) and
        // its index in the list of Dag operands and check that operands with
        // the same name have the same type. For example in 'C_ADD $rs1, $rs2'
        // we generate the mapping $rs1 --> 0, $rs2 ---> 1. If the operand
        // appears twice in the same Dag (tied in the compressed instruction),
        // we note the previous index in the TiedOpIdx field.
        StringRef ArgName = Dag->getArgNameStr(DAGOpNo);
        if (ArgName.empty())
          continue;

        if (IsSourceInst) {
          auto It = Operands.find(ArgName);
          if (It != Operands.end()) {
            OperandMap[OpNo].OpInfo.TiedOpIdx = It->getValue().MIOpNo;
            if (OperandMap[It->getValue().MIOpNo].OpInfo.DagRec != DI->getDef())
              PrintFatalError(Rec->getLoc(),
                              "Input Operand '" + ArgName +
                                  "' has a mismatched tied operand!");
          }
        }

        Operands[ArgName] = {DAGOpNo, OpNo};
      } else if (const auto *II = dyn_cast<IntInit>(Dag->getArg(DAGOpNo))) {
        // Validate that corresponding instruction operand expects an immediate.
        if (!OpndRec->isSubClassOf("Operand"))
          PrintFatalError(Rec->getLoc(), "Error in Dag '" + Dag->getAsString() +
                                             "' Found immediate: '" +
                                             II->getAsString() +
                                             "' but corresponding instruction "
                                             "operand expected a register!");
        // No pattern validation check possible for values of fixed immediate.
        OperandMap[OpNo].Kind = OpData::Imm;
        OperandMap[OpNo].ImmVal = II->getValue();
        LLVM_DEBUG(
            dbgs() << "  Found immediate '" << II->getValue() << "' at "
                   << (IsSourceInst ? "input " : "output ")
                   << "Dag. No validation time check possible for values of "
                      "fixed immediate.\n");
      } else {
        llvm_unreachable("Unhandled CompressPat argument type!");
      }
    }
  }

  // We shouldn't have extra Dag operands.
  if (DAGOpNo != Dag->getNumArgs())
    PrintFatalError(Rec->getLoc(), "Inst '" + Inst.getName() +
                                       "' and Dag operand count mismatch");
}

// Check that all names in the source DAG appear in the destionation DAG.
void CompressInstEmitter::checkDagOperandMapping(
    const Record *Rec, const StringMap<ArgData> &DestOperands,
    const DagInit *SourceDag, const DagInit *DestDag) {

  for (unsigned I = 0; I < SourceDag->getNumArgs(); ++I) {
    // Skip fixed immediates and registers, they were handled in
    // addDagOperandMapping.
    StringRef ArgName = SourceDag->getArgNameStr(I);
    if (ArgName.empty())
      continue;

    auto It = DestOperands.find(ArgName);
    if (It == DestOperands.end())
      PrintFatalError(Rec->getLoc(), "Operand " + ArgName +
                                         " defined in Input Dag but not used in"
                                         " Output Dag!");
    // Input Dag operand types must match output Dag operand type.
    if (!validateArgsTypes(DestDag->getArg(It->getValue().DAGOpNo),
                           SourceDag->getArg(I)))
      PrintFatalError(Rec->getLoc(), "Type mismatch between Input and "
                                     "Output Dag operand '" +
                                         ArgName + "'!");
  }
}

/// Map operand names in the Dag to their index in both corresponding input and
/// output instructions. Validate that operands defined in the input are
/// used in the output pattern while populating the maps.
void CompressInstEmitter::createInstOperandMapping(
    const Record *Rec, const DagInit *SourceDag, const DagInit *DestDag,
    IndexedMap<OpData> &SourceOperandMap, IndexedMap<OpData> &DestOperandMap,
    StringMap<ArgData> &SourceOperands, const CodeGenInstruction &DestInst) {
  // TiedCount keeps track of the number of operands skipped in Inst
  // operands list to get to the corresponding Dag operand.
  unsigned TiedCount = 0;
  LLVM_DEBUG(dbgs() << "  Operand mapping:\n  Source   Dest\n");
  unsigned OpNo = 0;
  for (const auto &Operand : DestInst.Operands) {
    int TiedInstOpIdx = Operand.getTiedRegister();
    if (TiedInstOpIdx != -1) {
      ++TiedCount;
      assert((unsigned)TiedInstOpIdx < OpNo);
      DestOperandMap[OpNo] = DestOperandMap[TiedInstOpIdx];
      if (DestOperandMap[OpNo].Kind == OpData::Operand)
        // No need to fill the SourceOperandMap here since it was mapped to
        // destination operand 'TiedInstOpIdx' in a previous iteration.
        LLVM_DEBUG(dbgs() << "    " << DestOperandMap[OpNo].OpInfo.Idx
                          << " ====> " << OpNo
                          << "  Dest operand tied with operand '"
                          << TiedInstOpIdx << "'\n");
      ++OpNo;
      continue;
    }

    for (unsigned SubOp = 0; SubOp != Operand.MINumOperands; ++SubOp, ++OpNo) {
      // Skip fixed immediates and registers, they were handled in
      // addDagOperandMapping.
      if (DestOperandMap[OpNo].Kind != OpData::Operand)
        continue;

      unsigned DagArgIdx = OpNo - TiedCount;
      StringRef ArgName = DestDag->getArgNameStr(DagArgIdx);
      auto SourceOp = SourceOperands.find(ArgName);
      if (SourceOp == SourceOperands.end())
        PrintFatalError(Rec->getLoc(),
                        "Output Dag operand '" + ArgName +
                            "' has no matching input Dag operand.");

      assert(ArgName ==
                 SourceDag->getArgNameStr(SourceOp->getValue().DAGOpNo) &&
             "Incorrect operand mapping detected!\n");

      unsigned SourceOpNo = SourceOp->getValue().MIOpNo;
      DestOperandMap[OpNo].OpInfo.Idx = SourceOpNo;
      SourceOperandMap[SourceOpNo].OpInfo.Idx = OpNo;
      LLVM_DEBUG(dbgs() << "    " << SourceOpNo << " ====> " << OpNo << "\n");
    }
  }
}

/// Validates the CompressPattern and create operand mapping.
/// These are the checks to validate a CompressPat pattern declarations.
/// Error out with message under these conditions:
/// - Dag Input opcode is an expanded instruction and Dag Output opcode is a
///   compressed instruction.
/// - Operands in Dag Input must be all used in Dag Output.
///   Register Operand type in Dag Input Type must be contained in the
///   corresponding Source Instruction type.
/// - Register Operand type in Dag Input must be the same as in Dag Ouput.
/// - Register Operand type in Dag Output must be the same as the
///   corresponding Destination Inst type.
/// - Immediate Operand type in Dag Input must be the same as in Dag Ouput.
/// - Immediate Operand type in Dag Ouput must be the same as the corresponding
///   Destination Instruction type.
/// - Fixed register must be contained in the corresponding Source Instruction
///   type.
/// - Fixed register must be contained in the corresponding Destination
///   Instruction type.
/// Warning message printed under these conditions:
/// - Fixed immediate in Dag Input or Dag Ouput cannot be checked at this time
///   and generate warning.
/// - Immediate operand type in Dag Input differs from the corresponding Source
///   Instruction type and generate a warning.
void CompressInstEmitter::evaluateCompressPat(const Record *Rec) {
  // Validate input Dag operands.
  const DagInit *SourceDag = Rec->getValueAsDag("Input");
  assert(SourceDag && "Missing 'Input' in compress pattern!");
  LLVM_DEBUG(dbgs() << "Input: " << *SourceDag << "\n");

  // Checking we are transforming from compressed to uncompressed instructions.
  const Record *SourceOperator = SourceDag->getOperatorAsDef(Rec->getLoc());
  CodeGenInstruction SourceInst(SourceOperator);

  // Validate output Dag operands.
  const DagInit *DestDag = Rec->getValueAsDag("Output");
  assert(DestDag && "Missing 'Output' in compress pattern!");
  LLVM_DEBUG(dbgs() << "Output: " << *DestDag << "\n");

  const Record *DestOperator = DestDag->getOperatorAsDef(Rec->getLoc());
  CodeGenInstruction DestInst(DestOperator);

  if (SourceOperator->getValueAsInt("Size") <=
      DestOperator->getValueAsInt("Size"))
    PrintFatalError(
        Rec->getLoc(),
        "Compressed instruction '" + DestOperator->getName() +
            "'is not strictly smaller than the uncompressed instruction '" +
            SourceOperator->getName() + "' !");

  // Fill the mapping from the source to destination instructions.

  IndexedMap<OpData> SourceOperandMap;
  // Map from arg name to DAG operand number and MI operand number.
  StringMap<ArgData> SourceOperands;
  // Create a mapping between source Dag operands and source Inst operands.
  addDagOperandMapping(Rec, SourceDag, SourceInst, SourceOperandMap,
                       SourceOperands, /*IsSourceInst*/ true);

  IndexedMap<OpData> DestOperandMap;
  // Map from arg name to DAG operand number and MI operand number.
  StringMap<ArgData> DestOperands;
  // Create a mapping between destination Dag operands and destination Inst
  // operands.
  addDagOperandMapping(Rec, DestDag, DestInst, DestOperandMap, DestOperands,
                       /*IsSourceInst*/ false);

  checkDagOperandMapping(Rec, DestOperands, SourceDag, DestDag);
  // Create operand mapping between the source and destination instructions.
  createInstOperandMapping(Rec, SourceDag, DestDag, SourceOperandMap,
                           DestOperandMap, SourceOperands, DestInst);

  // Get the target features for the CompressPat.
  std::vector<const Record *> PatReqFeatures;
  std::vector<const Record *> RF = Rec->getValueAsListOfDefs("Predicates");
  copy_if(RF, std::back_inserter(PatReqFeatures), [](const Record *R) {
    return R->getValueAsBit("AssemblerMatcherPredicate");
  });

  CompressPatterns.emplace_back(SourceInst, DestInst, std::move(PatReqFeatures),
                                SourceOperandMap, DestOperandMap,
                                Rec->getValueAsBit("isCompressOnly"));
}

static void
getReqFeatures(std::set<std::pair<bool, StringRef>> &FeaturesSet,
               std::set<std::set<std::pair<bool, StringRef>>> &AnyOfFeatureSets,
               ArrayRef<const Record *> ReqFeatures) {
  for (const Record *R : ReqFeatures) {
    const DagInit *D = R->getValueAsDag("AssemblerCondDag");
    std::string CombineType = D->getOperator()->getAsString();
    if (CombineType != "any_of" && CombineType != "all_of")
      PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
    if (D->getNumArgs() == 0)
      PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
    bool IsOr = CombineType == "any_of";
    std::set<std::pair<bool, StringRef>> AnyOfSet;

    for (auto *Arg : D->getArgs()) {
      bool IsNot = false;
      if (auto *NotArg = dyn_cast<DagInit>(Arg)) {
        if (NotArg->getOperator()->getAsString() != "not" ||
            NotArg->getNumArgs() != 1)
          PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
        Arg = NotArg->getArg(0);
        IsNot = true;
      }
      if (!isa<DefInit>(Arg) ||
          !cast<DefInit>(Arg)->getDef()->isSubClassOf("SubtargetFeature"))
        PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
      if (IsOr)
        AnyOfSet.emplace(IsNot, cast<DefInit>(Arg)->getDef()->getName());
      else
        FeaturesSet.emplace(IsNot, cast<DefInit>(Arg)->getDef()->getName());
    }

    if (IsOr)
      AnyOfFeatureSets.insert(std::move(AnyOfSet));
  }
}

static unsigned getPredicates(DenseMap<const Record *, unsigned> &PredicateMap,
                              std::vector<const Record *> &Predicates,
                              const Record *Rec, StringRef Name) {
  unsigned &Entry = PredicateMap[Rec];
  if (Entry)
    return Entry;

  if (!Rec->isValueUnset(Name)) {
    Predicates.push_back(Rec);
    Entry = Predicates.size();
    return Entry;
  }

  PrintFatalError(Rec->getLoc(), "No " + Name +
                                     " predicate on this operand at all: '" +
                                     Rec->getName() + "'");
  return 0;
}

static void printPredicates(ArrayRef<const Record *> Predicates, StringRef Name,
                            raw_ostream &OS) {
  for (unsigned I = 0; I < Predicates.size(); ++I) {
    StringRef Pred = Predicates[I]->getValueAsString(Name);
    Pred = Pred.trim();
    OS.indent(2) << "case " << I + 1 << ": {\n";
    OS.indent(4) << "// " << Predicates[I]->getName() << "\n";
    OS.indent(4) << Pred << "\n";
    OS.indent(2) << "}\n";
  }
}

static void mergeCondAndCode(raw_ostream &CombinedStream, StringRef CondStr,
                             StringRef CodeStr) {
  CombinedStream.indent(4) << "if (" << CondStr << ") {\n";
  CombinedStream << CodeStr;
  CombinedStream.indent(4) << "  return true;\n";
  CombinedStream.indent(4) << "} // if\n";
}

void CompressInstEmitter::emitCompressInstEmitter(raw_ostream &OS,
                                                  EmitterType EType) {
  const Record *AsmWriter = Target.getAsmWriter();
  if (!AsmWriter->getValueAsInt("PassSubtarget"))
    PrintFatalError(AsmWriter->getLoc(),
                    "'PassSubtarget' is false. SubTargetInfo object is needed "
                    "for target features.");

  StringRef TargetName = Target.getName();

  // Sort entries in CompressPatterns to handle instructions that can have more
  // than one candidate for compression\uncompression, e.g ADD can be
  // transformed to a C_ADD or a C_MV. When emitting 'uncompress()' function the
  // source and destination are flipped and the sort key needs to change
  // accordingly.
  llvm::stable_sort(CompressPatterns, [EType](const CompressPat &LHS,
                                              const CompressPat &RHS) {
    if (EType == EmitterType::Compress || EType == EmitterType::CheckCompress)
      return LHS.Source.getName() < RHS.Source.getName();
    return LHS.Dest.getName() < RHS.Dest.getName();
  });

  // A list of MCOperandPredicates for all operands in use, and the reverse map.
  std::vector<const Record *> MCOpPredicates;
  DenseMap<const Record *, unsigned> MCOpPredicateMap;
  // A list of ImmLeaf Predicates for all operands in use, and the reverse map.
  std::vector<const Record *> ImmLeafPredicates;
  DenseMap<const Record *, unsigned> ImmLeafPredicateMap;

  std::string F;
  std::string FH;
  raw_string_ostream Func(F);
  raw_string_ostream FuncH(FH);

  if (EType == EmitterType::Compress)
    OS << "\n#ifdef GEN_COMPRESS_INSTR\n"
       << "#undef GEN_COMPRESS_INSTR\n\n";
  else if (EType == EmitterType::Uncompress)
    OS << "\n#ifdef GEN_UNCOMPRESS_INSTR\n"
       << "#undef GEN_UNCOMPRESS_INSTR\n\n";
  else if (EType == EmitterType::CheckCompress)
    OS << "\n#ifdef GEN_CHECK_COMPRESS_INSTR\n"
       << "#undef GEN_CHECK_COMPRESS_INSTR\n\n";

  if (EType == EmitterType::Compress) {
    FuncH << "static bool compressInst(MCInst &OutInst,\n";
    FuncH.indent(25) << "const MCInst &MI,\n";
    FuncH.indent(25) << "const MCSubtargetInfo &STI) {\n";
  } else if (EType == EmitterType::Uncompress) {
    FuncH << "static bool uncompressInst(MCInst &OutInst,\n";
    FuncH.indent(27) << "const MCInst &MI,\n";
    FuncH.indent(27) << "const MCSubtargetInfo &STI) {\n";
  } else if (EType == EmitterType::CheckCompress) {
    FuncH << "static bool isCompressibleInst(const MachineInstr &MI,\n";
    FuncH.indent(31) << "const " << TargetName << "Subtarget &STI) {\n";
  }

  if (CompressPatterns.empty()) {
    OS << FH;
    OS.indent(2) << "return false;\n}\n";
    if (EType == EmitterType::Compress)
      OS << "\n#endif //GEN_COMPRESS_INSTR\n";
    else if (EType == EmitterType::Uncompress)
      OS << "\n#endif //GEN_UNCOMPRESS_INSTR\n\n";
    else if (EType == EmitterType::CheckCompress)
      OS << "\n#endif //GEN_CHECK_COMPRESS_INSTR\n\n";
    return;
  }

  std::string CaseString;
  raw_string_ostream CaseStream(CaseString);
  StringRef PrevOp;
  StringRef CurOp;
  CaseStream << "  switch (MI.getOpcode()) {\n";
  CaseStream << "  default: return false;\n";

  bool CompressOrCheck =
      EType == EmitterType::Compress || EType == EmitterType::CheckCompress;
  bool CompressOrUncompress =
      EType == EmitterType::Compress || EType == EmitterType::Uncompress;
  std::string ValidatorName =
      CompressOrUncompress
          ? (TargetName + "ValidateMCOperandFor" +
             (EType == EmitterType::Compress ? "Compress" : "Uncompress"))
                .str()
          : "";

  for (const auto &CompressPat : CompressPatterns) {
    if (EType == EmitterType::Uncompress && CompressPat.IsCompressOnly)
      continue;

    std::string CondString;
    std::string CodeString;
    raw_string_ostream CondStream(CondString);
    raw_string_ostream CodeStream(CodeString);
    const CodeGenInstruction &Source =
        CompressOrCheck ? CompressPat.Source : CompressPat.Dest;
    const CodeGenInstruction &Dest =
        CompressOrCheck ? CompressPat.Dest : CompressPat.Source;
    const IndexedMap<OpData> &SourceOperandMap =
        CompressOrCheck ? CompressPat.SourceOperandMap
                        : CompressPat.DestOperandMap;
    const IndexedMap<OpData> &DestOperandMap =
        CompressOrCheck ? CompressPat.DestOperandMap
                        : CompressPat.SourceOperandMap;

    CurOp = Source.getName();
    // Check current and previous opcode to decide to continue or end a case.
    if (CurOp != PrevOp) {
      if (!PrevOp.empty()) {
        CaseStream.indent(4) << "break;\n";
        CaseStream.indent(2) << "} // case " + PrevOp + "\n";
      }
      CaseStream.indent(2) << "case " + TargetName + "::" + CurOp + ": {\n";
    }

    std::set<std::pair<bool, StringRef>> FeaturesSet;
    std::set<std::set<std::pair<bool, StringRef>>> AnyOfFeatureSets;
    // Add CompressPat required features.
    getReqFeatures(FeaturesSet, AnyOfFeatureSets, CompressPat.PatReqFeatures);

    // Add Dest instruction required features.
    std::vector<const Record *> ReqFeatures;
    std::vector<const Record *> RF =
        Dest.TheDef->getValueAsListOfDefs("Predicates");
    copy_if(RF, std::back_inserter(ReqFeatures), [](const Record *R) {
      return R->getValueAsBit("AssemblerMatcherPredicate");
    });
    getReqFeatures(FeaturesSet, AnyOfFeatureSets, ReqFeatures);

    ListSeparator CondSep(" &&\n        ");

    // Emit checks for all required features.
    for (auto &Op : FeaturesSet) {
      StringRef Not = Op.first ? "!" : "";
      CondStream << CondSep << Not << "STI.getFeatureBits()[" << TargetName
                 << "::" << Op.second << "]";
    }

    // Emit checks for all required feature groups.
    for (auto &Set : AnyOfFeatureSets) {
      CondStream << CondSep << "(";
      for (auto &Op : Set) {
        bool IsLast = &Op == &*Set.rbegin();
        StringRef Not = Op.first ? "!" : "";
        CondStream << Not << "STI.getFeatureBits()[" << TargetName
                   << "::" << Op.second << "]";
        if (!IsLast)
          CondStream << " || ";
      }
      CondStream << ")";
    }

    // Start Source Inst operands validation.
    unsigned OpNo = 0;
    for (const auto &SourceOperand : Source.Operands) {
      for (unsigned SubOp = 0; SubOp != SourceOperand.MINumOperands; ++SubOp) {
        // Check for fixed immediates\registers in the source instruction.
        switch (SourceOperandMap[OpNo].Kind) {
        case OpData::Operand:
          if (SourceOperandMap[OpNo].OpInfo.TiedOpIdx != -1) {
            if (Source.Operands[OpNo].Rec->isSubClassOf("RegisterClass"))
              CondStream << CondSep << "MI.getOperand(" << OpNo
                         << ").isReg() && MI.getOperand("
                         << SourceOperandMap[OpNo].OpInfo.TiedOpIdx
                         << ").isReg()" << CondSep << "(MI.getOperand(" << OpNo
                         << ").getReg() == MI.getOperand("
                         << SourceOperandMap[OpNo].OpInfo.TiedOpIdx
                         << ").getReg())";
            else
              PrintFatalError("Unexpected tied operand types!");
          }

          // We don't need to do anything for source instruction operand checks.
          break;
        case OpData::Imm:
          CondStream << CondSep << "MI.getOperand(" << OpNo << ").isImm()"
                     << CondSep << "(MI.getOperand(" << OpNo
                     << ").getImm() == " << SourceOperandMap[OpNo].ImmVal
                     << ")";
          break;
        case OpData::Reg: {
          const Record *Reg = SourceOperandMap[OpNo].RegRec;
          CondStream << CondSep << "MI.getOperand(" << OpNo << ").isReg()"
                     << CondSep << "(MI.getOperand(" << OpNo
                     << ").getReg() == " << TargetName << "::" << Reg->getName()
                     << ")";
          break;
        }
        }
        ++OpNo;
      }
    }
    CodeStream.indent(6) << "// " << Dest.AsmString << "\n";
    if (CompressOrUncompress)
      CodeStream.indent(6) << "OutInst.setOpcode(" << TargetName
                           << "::" << Dest.getName() << ");\n";
    OpNo = 0;
    for (const auto &DestOperand : Dest.Operands) {
      CodeStream.indent(6) << "// Operand: " << DestOperand.Name << "\n";

      for (unsigned SubOp = 0; SubOp != DestOperand.MINumOperands; ++SubOp) {
        const Record *DestRec = DestOperand.Rec;

        if (DestOperand.MINumOperands > 1)
          DestRec =
              cast<DefInit>(DestOperand.MIOperandInfo->getArg(SubOp))->getDef();

        switch (DestOperandMap[OpNo].Kind) {
        case OpData::Operand: {
          unsigned OpIdx = DestOperandMap[OpNo].OpInfo.Idx;
          const Record *DagRec = DestOperandMap[OpNo].OpInfo.DagRec;
          // Check that the operand in the Source instruction fits
          // the type for the Dest instruction.
          if (DagRec->isSubClassOf("RegisterClass") ||
              DagRec->isSubClassOf("RegisterOperand")) {
            auto *ClassRec = DagRec->isSubClassOf("RegisterClass")
                                 ? DagRec
                                 : DagRec->getValueAsDef("RegClass");
            // This is a register operand. Check the register class.
            // Don't check register class if this is a tied operand, it was done
            // for the operand it's tied to.
            if (DestOperand.getTiedRegister() == -1) {
              CondStream << CondSep << "MI.getOperand(" << OpIdx << ").isReg()";
              if (EType == EmitterType::CheckCompress)
                CondStream << " && MI.getOperand(" << OpIdx
                           << ").getReg().isPhysical()";
              CondStream << CondSep << TargetName << "MCRegisterClasses["
                         << TargetName << "::" << ClassRec->getName()
                         << "RegClassID].contains(MI.getOperand(" << OpIdx
                         << ").getReg())";
            }

            if (CompressOrUncompress)
              CodeStream.indent(6)
                  << "OutInst.addOperand(MI.getOperand(" << OpIdx << "));\n";
          } else {
            // Handling immediate operands.
            if (CompressOrUncompress) {
              unsigned Entry = getPredicates(MCOpPredicateMap, MCOpPredicates,
                                             DagRec, "MCOperandPredicate");
              CondStream << CondSep << ValidatorName << "("
                         << "MI.getOperand(" << OpIdx << "), STI, " << Entry
                         << " /* " << DagRec->getName() << " */)";
              // Also check DestRec if different than DagRec.
              if (DagRec != DestRec) {
                Entry = getPredicates(MCOpPredicateMap, MCOpPredicates, DestRec,
                                      "MCOperandPredicate");
                CondStream << CondSep << ValidatorName << "("
                           << "MI.getOperand(" << OpIdx << "), STI, " << Entry
                           << " /* " << DestRec->getName() << " */)";
              }
            } else {
              unsigned Entry =
                  getPredicates(ImmLeafPredicateMap, ImmLeafPredicates, DagRec,
                                "ImmediateCode");
              CondStream << CondSep << "MI.getOperand(" << OpIdx << ").isImm()";
              CondStream << CondSep << TargetName << "ValidateMachineOperand("
                         << "MI.getOperand(" << OpIdx << "), &STI, " << Entry
                         << " /* " << DagRec->getName() << " */)";
              if (DagRec != DestRec) {
                Entry = getPredicates(ImmLeafPredicateMap, ImmLeafPredicates,
                                      DestRec, "ImmediateCode");
                CondStream << CondSep << "MI.getOperand(" << OpIdx
                           << ").isImm()";
                CondStream << CondSep << TargetName << "ValidateMachineOperand("
                           << "MI.getOperand(" << OpIdx << "), &STI, " << Entry
                           << " /* " << DestRec->getName() << " */)";
              }
            }
            if (CompressOrUncompress)
              CodeStream.indent(6)
                  << "OutInst.addOperand(MI.getOperand(" << OpIdx << "));\n";
          }
          break;
        }
        case OpData::Imm: {
          if (CompressOrUncompress) {
            unsigned Entry = getPredicates(MCOpPredicateMap, MCOpPredicates,
                                           DestRec, "MCOperandPredicate");
            CondStream << CondSep << ValidatorName << "("
                       << "MCOperand::createImm(" << DestOperandMap[OpNo].ImmVal
                       << "), STI, " << Entry << " /* " << DestRec->getName()
                       << " */)";
          } else {
            unsigned Entry =
                getPredicates(ImmLeafPredicateMap, ImmLeafPredicates, DestRec,
                              "ImmediateCode");
            CondStream << CondSep << TargetName
                       << "ValidateMachineOperand(MachineOperand::CreateImm("
                       << DestOperandMap[OpNo].ImmVal << "), &STI, " << Entry
                       << " /* " << DestRec->getName() << " */)";
          }
          if (CompressOrUncompress)
            CodeStream.indent(6) << "OutInst.addOperand(MCOperand::createImm("
                                 << DestOperandMap[OpNo].ImmVal << "));\n";
        } break;
        case OpData::Reg: {
          if (CompressOrUncompress) {
            // Fixed register has been validated at pattern validation time.
            const Record *Reg = DestOperandMap[OpNo].RegRec;
            CodeStream.indent(6)
                << "OutInst.addOperand(MCOperand::createReg(" << TargetName
                << "::" << Reg->getName() << "));\n";
          }
        } break;
        }
        ++OpNo;
      }
    }
    if (CompressOrUncompress)
      CodeStream.indent(6) << "OutInst.setLoc(MI.getLoc());\n";
    mergeCondAndCode(CaseStream, CondString, CodeString);
    PrevOp = CurOp;
  }
  Func << CaseString;
  Func.indent(4) << "break;\n";
  // Close brace for the last case.
  Func.indent(2) << "} // case " << CurOp << "\n";
  Func.indent(2) << "} // switch\n";
  Func.indent(2) << "return false;\n}\n";

  if (!MCOpPredicates.empty()) {
    auto IndentLength = ValidatorName.size() + 13;
    OS << "static bool " << ValidatorName << "(const MCOperand &MCOp,\n";
    OS.indent(IndentLength) << "const MCSubtargetInfo &STI,\n";
    OS.indent(IndentLength) << "unsigned PredicateIndex) {\n";
    OS << "  switch (PredicateIndex) {\n"
       << "  default:\n"
       << "    llvm_unreachable(\"Unknown MCOperandPredicate kind\");\n"
       << "    break;\n";

    printPredicates(MCOpPredicates, "MCOperandPredicate", OS);

    OS << "  }\n"
       << "}\n\n";
  }

  if (!ImmLeafPredicates.empty()) {
    auto IndentLength = TargetName.size() + 35;
    OS << "static bool " << TargetName
       << "ValidateMachineOperand(const MachineOperand &MO,\n";
    OS.indent(IndentLength)
        << "const " << TargetName << "Subtarget *Subtarget,\n";
    OS.indent(IndentLength)
        << "unsigned PredicateIndex) {\n"
        << "  int64_t Imm = MO.getImm();\n"
        << "  switch (PredicateIndex) {\n"
        << "  default:\n"
        << "    llvm_unreachable(\"Unknown ImmLeaf Predicate kind\");\n"
        << "    break;\n";

    printPredicates(ImmLeafPredicates, "ImmediateCode", OS);

    OS << "  }\n"
       << "}\n\n";
  }

  OS << FH;
  OS << F;

  if (EType == EmitterType::Compress)
    OS << "\n#endif //GEN_COMPRESS_INSTR\n";
  else if (EType == EmitterType::Uncompress)
    OS << "\n#endif //GEN_UNCOMPRESS_INSTR\n\n";
  else if (EType == EmitterType::CheckCompress)
    OS << "\n#endif //GEN_CHECK_COMPRESS_INSTR\n\n";
}

void CompressInstEmitter::run(raw_ostream &OS) {
  // Process the CompressPat definitions, validating them as we do so.
  for (const Record *Pat : Records.getAllDerivedDefinitions("CompressPat"))
    evaluateCompressPat(Pat);

  // Emit file header.
  emitSourceFileHeader("Compress instruction Source Fragment", OS, Records);
  // Generate compressInst() function.
  emitCompressInstEmitter(OS, EmitterType::Compress);
  // Generate uncompressInst() function.
  emitCompressInstEmitter(OS, EmitterType::Uncompress);
  // Generate isCompressibleInst() function.
  emitCompressInstEmitter(OS, EmitterType::CheckCompress);
}

static TableGen::Emitter::OptClass<CompressInstEmitter>
    X("gen-compress-inst-emitter", "Generate compressed instructions.");
