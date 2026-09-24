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
// GEN_COMPRESS_MACHINE_INSTR and GEN_UNCOMPRESS_MACHINE_INSTR provide
// post-register-allocation overloads:
//
// bool compressInst(MachineInstr &MI, const <TargetName>Subtarget &STI);
// bool uncompressInst(MachineInstr &MI, const <TargetName>Subtarget &STI);
//
// These overloads rewrite MachineInstrs in place. Targets decide whether
// bundles and delay-slot instructions may change.
//
// In addition, it exports a function for checking whether
// an instruction is compressible:
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
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <limits>
#include <optional>
#include <set>
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "compress-inst-emitter"

namespace {
class CompressInstEmitter {
  struct OpData {
    enum MapKind { Operand, Imm, Reg } Kind;
    StringRef Name;
    // Corresponding operand with the same def/use role, for MachineInstrs.
    std::optional<unsigned> MachineIdx;
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
  enum EmitterType {
    Compress,
    Uncompress,
    CompressMachine,
    UncompressMachine,
    CheckCompress
  };
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
  bool validateRegister(const Record *Reg, const Record *RegClass,
                        ArrayRef<SMLoc> Loc);
  void checkDagOperandMapping(const Record *Rec,
                              const StringMap<ArgData> &DestOperands,
                              const DagInit *SourceDag, const DagInit *DestDag);

  void createInstOperandMapping(const Record *Rec, const DagInit *SourceDag,
                                const DagInit *DestDag,
                                IndexedMap<OpData> &SourceOperandMap,
                                IndexedMap<OpData> &DestOperandMap,
                                StringMap<ArgData> &SourceOperands,
                                const CodeGenInstruction &SourceInst,
                                const CodeGenInstruction &DestInst);

public:
  CompressInstEmitter(const RecordKeeper &R) : Records(R), Target(R) {}

  void run(raw_ostream &OS);
};
} // End anonymous namespace.

bool CompressInstEmitter::validateRegister(const Record *Reg,
                                           const Record *RegClass,
                                           ArrayRef<SMLoc> Loc) {
  assert((Reg->isSubClassOf("Register") ||
          Reg->isSubClassOf("RegisterByHwMode")) &&
         "Reg record should be a Register");
  RegClass = Target.getAsRegClassLike(RegClass);
  assert(RegClass && RegClass->isSubClassOf("RegisterClassLike") &&
         "RegClass record should be RegisterClassLike");
  return Target.getRegBank().regClassContainsReg(RegClass, Reg, Loc);
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
        if (DI->getDef()->isSubClassOf("Register") ||
            DI->getDef()->isSubClassOf("RegisterByHwMode")) {
          // Check if the fixed register belongs to the Register class.
          if (!validateRegister(DI->getDef(), OpndRec, Rec->getLoc()))
            PrintFatalError(Rec->getLoc(),
                            "Error in Dag '" + Dag->getAsString() +
                                "': Register '" + DI->getDef()->getName() +
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
        OperandMap[OpNo].Name = ArgName;
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
    StringMap<ArgData> &SourceOperands, const CodeGenInstruction &SourceInst,
    const CodeGenInstruction &DestInst) {
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

  // Retain separate def and use mappings for both rewrite directions.
  unsigned NumSourceOps = SourceInst.Operands.empty()
                              ? 0
                              : SourceInst.Operands.back().MIOperandNo +
                                    SourceInst.Operands.back().MINumOperands;
  for (unsigned SrcNo = 0; SrcNo != NumSourceOps; ++SrcNo) {
    OpData &Src = SourceOperandMap[SrcNo];
    if (Src.Kind == OpData::Imm)
      continue;
    bool IsReg =
        Src.Kind == OpData::Reg || Target.getAsRegClassLike(Src.OpInfo.DagRec);
    for (unsigned DstNo = 0; DstNo != OpNo; ++DstNo) {
      OpData &Dst = DestOperandMap[DstNo];
      if (Src.Kind != Dst.Kind ||
          (IsReg && (SrcNo < SourceInst.Operands.NumDefs) !=
                        (DstNo < DestInst.Operands.NumDefs)))
        continue;
      if (Src.Kind == OpData::Reg ? Src.RegRec != Dst.RegRec
                                  : Src.Name != Dst.Name)
        continue;
      if (!Src.MachineIdx)
        Src.MachineIdx = DstNo;
      if (!Dst.MachineIdx)
        Dst.MachineIdx = SrcNo;
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
                           DestOperandMap, SourceOperands, SourceInst,
                           DestInst);

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
  bool IsUncompress = EType == EmitterType::Uncompress ||
                      EType == EmitterType::UncompressMachine;
  bool IsMachineRewrite = EType == EmitterType::CompressMachine ||
                          EType == EmitterType::UncompressMachine;

  // Sort entries in CompressPatterns to handle instructions that can have more
  // than one candidate for compression\uncompression, e.g ADD can be
  // transformed to a C_ADD or a C_MV. When emitting 'uncompress()' function the
  // source and destination are flipped and the sort key needs to change
  // accordingly.
  llvm::stable_sort(CompressPatterns, [IsUncompress](const CompressPat &LHS,
                                                     const CompressPat &RHS) {
    if (!IsUncompress)
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

  auto GetEmitterGuard = [EType]() -> StringRef {
    switch (EType) {
    case EmitterType::Compress:
      return "GEN_COMPRESS_INSTR";
    case EmitterType::Uncompress:
      return "GEN_UNCOMPRESS_INSTR";
    case EmitterType::CompressMachine:
      return "GEN_COMPRESS_MACHINE_INSTR";
    case EmitterType::UncompressMachine:
      return "GEN_UNCOMPRESS_MACHINE_INSTR";
    case EmitterType::CheckCompress:
      return "GEN_CHECK_COMPRESS_INSTR";
    }
    llvm_unreachable("Invalid emitter type");
  };

  IfDefEmitter IfDef(OS, GetEmitterGuard());

  if (IsMachineRewrite) {
    std::string Guard = (TargetName + "_MACHINE_INSTR_REWRITE").str();
    OS << "#ifndef " << Guard << "\n#define " << Guard << "\n";
    OS << R"cpp(
// Map identifies the source operand for each copied destination operand.
static inline void rewriteMachineInstr(
    MachineInstr &MI, const MCInstrDesc &Desc,
    SmallVector<MachineOperand, 8> Operands, SmallVector<std::optional<unsigned>, 8> Map) {
  SmallVector<bool, 8> Used(MI.getNumOperands(), false);
  for (unsigned I = 0, E = Operands.size(); I != E; ++I) {
    // Reuse implicit registers made explicit by expansion, retaining their flags.
    if (!Map[I] && Operands[I].isReg())
      for (unsigned J = MI.getNumExplicitOperands(), End = MI.getNumOperands();
           J != End; ++J) {
        const MachineOperand &MO = MI.getOperand(J);
        if (!Used[J] && MO.isReg() && MO.isImplicit() &&
            MO.getReg() == Operands[I].getReg() &&
            MO.isDef() == Operands[I].isDef()) {
          Operands[I] = MO;
          Operands[I].setImplicit(false);
          Map[I] = J;
          break;
        }
      }
    if (Map[I])
      Used[*Map[I]] = true;
  }

  // Trailing operands are not described by the pattern.
  unsigned NumFixedOperands = MI.getDesc().getNumOperands();
  for (unsigned I = NumFixedOperands, E = MI.getNumOperands(); I != E; ++I) {
    if (Used[I])
      continue;
    Operands.push_back(MI.getOperand(I));
    Map.push_back(I);
  }

  // Registers omitted by the encoding remain implicit dependencies.
  for (unsigned I = 0; I != NumFixedOperands; ++I) {
    if (Used[I] || !MI.getOperand(I).isReg())
      continue;
    MachineOperand MO = MI.getOperand(I);
    MO.setImplicit();
    MO.setIsRenamable(false);
    Operands.push_back(MO);
    Map.push_back(I);
  }

  auto AddImplicit = [&](MCPhysReg Reg, bool IsDef) {
    for (const MachineOperand &MO : Operands)
      if (MO.isReg() && MO.isImplicit() && MO.getReg() == Reg &&
          MO.isDef() == IsDef)
        return;
    Operands.push_back(MachineOperand::CreateReg(Reg, IsDef, true));
    Map.push_back(std::nullopt);
  };
  for (MCPhysReg Reg : Desc.implicit_defs())
    AddImplicit(Reg, true);
  for (MCPhysReg Reg : Desc.implicit_uses())
    AddImplicit(Reg, false);

  // Keep kill flags on the last use after reordering operands.
  for (unsigned I = 0, E = Operands.size(); I != E; ++I) {
    MachineOperand &MO = Operands[I];
    if (!MO.isReg() || !MO.isUse() || !MO.isKill())
      continue;
    for (unsigned J = I + 1; J != E; ++J) {
      MachineOperand &Later = Operands[J];
      if (Later.isReg() && Later.isUse() && Later.getReg() == MO.getReg()) {
        MO.setIsKill(false);
        Later.setIsKill(true);
      }
    }
  }

  unsigned OldDebugNum = MI.peekDebugInstrNum();
  if (OldDebugNum) {
    MI.dropDebugNumber();
    unsigned NewDebugNum = MI.getDebugInstrNum();
    for (unsigned I = 0, E = Operands.size(); I != E; ++I)
      if (Map[I] && Operands[I].isReg() && Operands[I].isDef())
        MI.getMF()->makeDebugValueSubstitution(
            {OldDebugNum, *Map[I]}, {NewDebugNum, I});
    if (MI.mayStore())
      MI.getMF()->makeDebugValueSubstitution(
          {OldDebugNum, MachineFunction::DebugOperandMemNumber},
          {NewDebugNum, MachineFunction::DebugOperandMemNumber});
  }

  // Remove uses before defs to clear ties; addOperand applies the new constraints.
  while (MI.getNumOperands())
    MI.removeOperand(MI.getNumOperands() - 1);
  MI.setDesc(Desc);
  for (const MachineOperand &MO : Operands)
    MI.addOperand(MO);
}

)cpp";
    OS << "#endif // " << Guard << "\n\n";
  }

  if (EType == EmitterType::Compress) {
    FuncH << "static bool compressInst(MCInst &OutInst,\n";
    FuncH.indent(25) << "const MCInst &MI,\n";
    FuncH.indent(25) << "const MCSubtargetInfo &STI) {\n";
  } else if (EType == EmitterType::Uncompress) {
    FuncH << "static bool uncompressInst(MCInst &OutInst,\n";
    FuncH.indent(27) << "const MCInst &MI,\n";
    FuncH.indent(27) << "const MCSubtargetInfo &STI) {\n";
  } else if (IsMachineRewrite) {
    FuncH << "static bool "
          << (IsUncompress ? "uncompressInst" : "compressInst")
          << "(MachineInstr &MI, const " << TargetName << "Subtarget &STI) {\n";
  } else {
    FuncH << "static bool isCompressibleInst(const MachineInstr &MI,\n";
    FuncH.indent(31) << "const " << TargetName << "Subtarget &STI) {\n";
  }
  // HwModeId is used if we have any RegClassByHwMode patterns
  if (!Target.getAllRegClassByHwMode().empty())
    FuncH.indent(2) << "[[maybe_unused]] unsigned HwModeId = "
                    << "STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo);\n";

  if (CompressPatterns.empty()) {
    OS << FH;
    OS.indent(2) << "return false;\n}\n";
    return;
  }

  std::string CaseString;
  raw_string_ostream CaseStream(CaseString);
  StringRef PrevOp;
  StringRef CurOp;
  CaseStream << "  switch (MI.getOpcode()) {\n";
  CaseStream << "  default: return false;\n";

  bool EmitOperands = EType != EmitterType::CheckCompress;
  StringRef AddOperand =
      IsMachineRewrite ? "Operands.push_back(" : "OutInst.addOperand(";
  StringRef OperandType =
      IsMachineRewrite ? "MachineOperand::Create" : "MCOperand::create";
  std::string MachineValidatorName =
      (TargetName + "ValidateMachineOperand" +
       (IsMachineRewrite ? (IsUncompress ? "ForUncompress" : "ForCompress")
                         : ""))
          .str();
  bool IsMCInst =
      EType == EmitterType::Compress || EType == EmitterType::Uncompress;
  std::string ValidatorName =
      IsMCInst ? (TargetName + "ValidateMCOperandFor" +
                  (EType == EmitterType::Compress ? "Compress" : "Uncompress"))
                     .str()
               : "";

  for (const auto &CompressPat : CompressPatterns) {
    if (IsUncompress && CompressPat.IsCompressOnly)
      continue;

    std::string CondString;
    std::string CodeString;
    raw_string_ostream CondStream(CondString);
    raw_string_ostream CodeStream(CodeString);
    const CodeGenInstruction &Source =
        !IsUncompress ? CompressPat.Source : CompressPat.Dest;
    const CodeGenInstruction &Dest =
        !IsUncompress ? CompressPat.Dest : CompressPat.Source;
    const IndexedMap<OpData> &SourceOperandMap =
        !IsUncompress ? CompressPat.SourceOperandMap
                      : CompressPat.DestOperandMap;
    const IndexedMap<OpData> &DestOperandMap =
        !IsUncompress ? CompressPat.DestOperandMap
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
            if (Target.getAsRegClassLike(SourceOperand.Rec))
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
                     << ").getReg() == ";
          if (Reg->isSubClassOf("RegisterByHwMode")) {
            RegisterByHwMode(Reg, Target.getRegBank())
                .emitResolverCall(CondStream, "HwModeId");
          } else {
            CondStream << TargetName << "::" << Reg->getName();
          }
          CondStream << ")";
          break;
        }
        }
        ++OpNo;
      }
    }
    CodeStream.indent(6) << "// " << Dest.AsmString << "\n";
    if (IsMCInst)
      CodeStream.indent(6) << "OutInst.setOpcode(" << TargetName
                           << "::" << Dest.getName() << ");\n";
    SmallVector<std::optional<unsigned>, 8> MachineMap;
    if (IsMachineRewrite)
      CodeStream.indent(6) << "SmallVector<MachineOperand, 8> Operands;\n";
    OpNo = 0;
    for (const auto &DestOperand : Dest.Operands) {
      CodeStream.indent(6) << "// Operand: " << DestOperand.Name << "\n";

      for (unsigned SubOp = 0; SubOp != DestOperand.MINumOperands; ++SubOp) {
        const Record *DestRec = DestOperand.Rec;

        if (DestOperand.MINumOperands > 1)
          DestRec =
              cast<DefInit>(DestOperand.MIOperandInfo->getArg(SubOp))->getDef();

        std::string OperandExpr;
        raw_string_ostream OperandStream(OperandExpr);
        switch (DestOperandMap[OpNo].Kind) {
        case OpData::Operand: {
          unsigned OpIdx = DestOperandMap[OpNo].OpInfo.Idx;
          const Record *DagRec = DestOperandMap[OpNo].OpInfo.DagRec;
          // Check that the operand in the Source instruction fits
          // the type for the Dest instruction.
          if (auto *ClassRec = Target.getAsRegClassLike(DagRec)) {
            // This is a register operand. Check the register class.
            // Don't check register class if this is a tied operand, it was done
            // for the operand it's tied to.
            if (DestOperand.getTiedRegister() == -1) {
              CondStream << CondSep << "MI.getOperand(" << OpIdx << ").isReg()";
              if (!IsMCInst)
                CondStream << " && MI.getOperand(" << OpIdx
                           << ").getReg().isPhysical()";
              CondStream << CondSep << "get" << TargetName
                         << "MCRegisterClass(";
              if (ClassRec->isSubClassOf("RegClassByHwMode")) {
                CondStream << TargetName << "RegClassByHwModeTables[HwModeId]["
                           << TargetName << "::" << ClassRec->getName() << "]";
              } else {
                CondStream << TargetName << "::" << ClassRec->getName()
                           << "RegClassID";
              }
              CondStream << ").contains(MI.getOperand(" << OpIdx
                         << ").getReg())";
            }

          } else {
            // Handling immediate operands.
            if (IsMCInst) {
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
              CondStream << CondSep << MachineValidatorName << "("
                         << "MI.getOperand(" << OpIdx << "), &STI, " << Entry
                         << " /* " << DagRec->getName() << " */)";
              if (DagRec != DestRec) {
                Entry = getPredicates(ImmLeafPredicateMap, ImmLeafPredicates,
                                      DestRec, "ImmediateCode");
                CondStream << CondSep << "MI.getOperand(" << OpIdx
                           << ").isImm()";
                CondStream << CondSep << MachineValidatorName << "("
                           << "MI.getOperand(" << OpIdx << "), &STI, " << Entry
                           << " /* " << DestRec->getName() << " */)";
              }
            }
          }
          OperandStream << "MI.getOperand(" << OpIdx << ")";
          break;
        }
        case OpData::Imm: {
          if (IsMCInst) {
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
            CondStream << CondSep << MachineValidatorName
                       << "(MachineOperand::CreateImm("
                       << DestOperandMap[OpNo].ImmVal << "), &STI, " << Entry
                       << " /* " << DestRec->getName() << " */)";
          }
          OperandStream << OperandType << "Imm(" << DestOperandMap[OpNo].ImmVal
                        << ")";
        } break;
        case OpData::Reg: {
          // Fixed register has been validated at pattern validation time.
          const Record *Reg = DestOperandMap[OpNo].RegRec;
          OperandStream << OperandType << "Reg(";
          if (Reg->isSubClassOf("RegisterByHwMode"))
            RegisterByHwMode(Reg, Target.getRegBank())
                .emitResolverCall(OperandStream, "HwModeId");
          else
            OperandStream << TargetName << "::" << Reg->getName();
          if (IsMachineRewrite)
            OperandStream << ", "
                          << (OpNo < Dest.Operands.NumDefs ? "true" : "false");
          OperandStream << ")";
          break;
        }
        }
        if (IsMachineRewrite) {
          std::optional<unsigned> Index = DestOperandMap[OpNo].MachineIdx;
          MachineMap.push_back(Index);
          if (Index)
            OperandExpr = "MI.getOperand(" + utostr(*Index) + ")";
          else if (DestOperandMap[OpNo].Kind == OpData::Operand)
            // A variable operand must have a source with the same def/use role.
            CodeStream.indent(6) << "return false;\n";
        }
        if (EmitOperands)
          CodeStream.indent(6) << AddOperand << OperandExpr << ");\n";
        ++OpNo;
      }
    }
    if (IsMCInst)
      CodeStream.indent(6) << "OutInst.setLoc(MI.getLoc());\n";
    if (IsMachineRewrite) {
      CodeStream.indent(6) << "rewriteMachineInstr(MI, STI.getInstrInfo()->get("
                           << TargetName << "::" << Dest.getName()
                           << "), std::move(Operands), {";
      ListSeparator Sep;
      for (std::optional<unsigned> Index : MachineMap)
        CodeStream << Sep << (Index ? utostr(*Index) : "std::nullopt");
      CodeStream << "});\n";
    }
    mergeCondAndCode(CaseStream, CondString, CodeString);
    PrevOp = CurOp;
  }
  Func << CaseString;
  if (!PrevOp.empty()) {
    Func.indent(4) << "break;\n";
    Func.indent(2) << "} // case " << PrevOp << "\n";
  }
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
    auto IndentLength = MachineValidatorName.size() + 13;
    OS << "static bool " << MachineValidatorName
       << "(const MachineOperand &MO,\n";
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
  emitCompressInstEmitter(OS, EmitterType::CompressMachine);
  emitCompressInstEmitter(OS, EmitterType::UncompressMachine);
  // Generate isCompressibleInst() function.
  emitCompressInstEmitter(OS, EmitterType::CheckCompress);
}

static TableGen::Emitter::OptClass<CompressInstEmitter>
    X("gen-compress-inst-emitter", "Generate compressed instructions.");
