//===- CodeGenTarget.cpp - CodeGen Target Class Wrapper -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class wraps target description classes used by the various code
// generation TableGen backends.  This makes it easier to access the data and
// provides a single place that needs to check it for validity.  All of these
// classes abort on error conditions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "CodeGenInstruction.h"
#include "CodeGenIntrinsics.h"
#include "CodeGenSchedule.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ModRef.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <iterator>
#include <tuple>
using namespace llvm;

cl::OptionCategory AsmParserCat("Options for -gen-asm-parser");
cl::OptionCategory AsmWriterCat("Options for -gen-asm-writer");

static cl::opt<unsigned>
    AsmParserNum("asmparsernum", cl::init(0),
                 cl::desc("Make -gen-asm-parser emit assembly parser #N"),
                 cl::cat(AsmParserCat));

static cl::opt<unsigned>
    AsmWriterNum("asmwriternum", cl::init(0),
                 cl::desc("Make -gen-asm-writer emit assembly writer #N"),
                 cl::cat(AsmWriterCat));

/// getValueType - Return the MVT::SimpleValueType that the specified TableGen
/// record corresponds to.
MVT::SimpleValueType llvm::getValueType(Record *Rec) {
  return (MVT::SimpleValueType)Rec->getValueAsInt("Value");
}

StringRef llvm::getName(MVT::SimpleValueType T) {
  switch (T) {
  case MVT::Other:   return "UNKNOWN";
  case MVT::iPTR:    return "TLI.getPointerTy()";
  case MVT::iPTRAny: return "TLI.getPointerTy()";
  default: return getEnumName(T);
  }
}

StringRef llvm::getEnumName(MVT::SimpleValueType T) {
  // clang-format off
  switch (T) {
  case MVT::Other:    return "MVT::Other";
  case MVT::i1:       return "MVT::i1";
  case MVT::i2:       return "MVT::i2";
  case MVT::i4:       return "MVT::i4";
  case MVT::i8:       return "MVT::i8";
  case MVT::i16:      return "MVT::i16";
  case MVT::i32:      return "MVT::i32";
  case MVT::i64:      return "MVT::i64";
  case MVT::i128:     return "MVT::i128";
  case MVT::Any:      return "MVT::Any";
  case MVT::iAny:     return "MVT::iAny";
  case MVT::fAny:     return "MVT::fAny";
  case MVT::vAny:     return "MVT::vAny";
  case MVT::f16:      return "MVT::f16";
  case MVT::bf16:     return "MVT::bf16";
  case MVT::f32:      return "MVT::f32";
  case MVT::f64:      return "MVT::f64";
  case MVT::f80:      return "MVT::f80";
  case MVT::f128:     return "MVT::f128";
  case MVT::ppcf128:  return "MVT::ppcf128";
  case MVT::x86mmx:   return "MVT::x86mmx";
  case MVT::x86amx:   return "MVT::x86amx";
  case MVT::i64x8:    return "MVT::i64x8";
  case MVT::Glue:     return "MVT::Glue";
  case MVT::isVoid:   return "MVT::isVoid";
  case MVT::v1i1:     return "MVT::v1i1";
  case MVT::v2i1:     return "MVT::v2i1";
  case MVT::v4i1:     return "MVT::v4i1";
  case MVT::v8i1:     return "MVT::v8i1";
  case MVT::v16i1:    return "MVT::v16i1";
  case MVT::v32i1:    return "MVT::v32i1";
  case MVT::v64i1:    return "MVT::v64i1";
  case MVT::v128i1:   return "MVT::v128i1";
  case MVT::v256i1:   return "MVT::v256i1";
  case MVT::v512i1:   return "MVT::v512i1";
  case MVT::v1024i1:  return "MVT::v1024i1";
  case MVT::v2048i1:  return "MVT::v2048i1";
  case MVT::v128i2:   return "MVT::v128i2";
  case MVT::v256i2:   return "MVT::v256i2";
  case MVT::v64i4:    return "MVT::v64i4";
  case MVT::v128i4:   return "MVT::v128i4";
  case MVT::v1i8:     return "MVT::v1i8";
  case MVT::v2i8:     return "MVT::v2i8";
  case MVT::v4i8:     return "MVT::v4i8";
  case MVT::v8i8:     return "MVT::v8i8";
  case MVT::v16i8:    return "MVT::v16i8";
  case MVT::v32i8:    return "MVT::v32i8";
  case MVT::v64i8:    return "MVT::v64i8";
  case MVT::v128i8:   return "MVT::v128i8";
  case MVT::v256i8:   return "MVT::v256i8";
  case MVT::v512i8:   return "MVT::v512i8";
  case MVT::v1024i8:  return "MVT::v1024i8";
  case MVT::v1i16:    return "MVT::v1i16";
  case MVT::v2i16:    return "MVT::v2i16";
  case MVT::v3i16:    return "MVT::v3i16";
  case MVT::v4i16:    return "MVT::v4i16";
  case MVT::v8i16:    return "MVT::v8i16";
  case MVT::v16i16:   return "MVT::v16i16";
  case MVT::v32i16:   return "MVT::v32i16";
  case MVT::v64i16:   return "MVT::v64i16";
  case MVT::v128i16:  return "MVT::v128i16";
  case MVT::v256i16:  return "MVT::v256i16";
  case MVT::v512i16:  return "MVT::v512i16";
  case MVT::v1i32:    return "MVT::v1i32";
  case MVT::v2i32:    return "MVT::v2i32";
  case MVT::v3i32:    return "MVT::v3i32";
  case MVT::v4i32:    return "MVT::v4i32";
  case MVT::v5i32:    return "MVT::v5i32";
  case MVT::v6i32:    return "MVT::v6i32";
  case MVT::v7i32:    return "MVT::v7i32";
  case MVT::v8i32:    return "MVT::v8i32";
  case MVT::v9i32:    return "MVT::v9i32";
  case MVT::v10i32:   return "MVT::v10i32";
  case MVT::v11i32:   return "MVT::v11i32";
  case MVT::v12i32:   return "MVT::v12i32";
  case MVT::v16i32:   return "MVT::v16i32";
  case MVT::v32i32:   return "MVT::v32i32";
  case MVT::v64i32:   return "MVT::v64i32";
  case MVT::v128i32:  return "MVT::v128i32";
  case MVT::v256i32:  return "MVT::v256i32";
  case MVT::v512i32:  return "MVT::v512i32";
  case MVT::v1024i32: return "MVT::v1024i32";
  case MVT::v2048i32: return "MVT::v2048i32";
  case MVT::v1i64:    return "MVT::v1i64";
  case MVT::v2i64:    return "MVT::v2i64";
  case MVT::v3i64:    return "MVT::v3i64";
  case MVT::v4i64:    return "MVT::v4i64";
  case MVT::v8i64:    return "MVT::v8i64";
  case MVT::v16i64:   return "MVT::v16i64";
  case MVT::v32i64:   return "MVT::v32i64";
  case MVT::v64i64:   return "MVT::v64i64";
  case MVT::v128i64:  return "MVT::v128i64";
  case MVT::v256i64:  return "MVT::v256i64";
  case MVT::v1i128:   return "MVT::v1i128";
  case MVT::v1f16:    return "MVT::v1f16";
  case MVT::v2f16:    return "MVT::v2f16";
  case MVT::v3f16:    return "MVT::v3f16";
  case MVT::v4f16:    return "MVT::v4f16";
  case MVT::v8f16:    return "MVT::v8f16";
  case MVT::v16f16:   return "MVT::v16f16";
  case MVT::v32f16:   return "MVT::v32f16";
  case MVT::v64f16:   return "MVT::v64f16";
  case MVT::v128f16:  return "MVT::v128f16";
  case MVT::v256f16:  return "MVT::v256f16";
  case MVT::v512f16:  return "MVT::v512f16";
  case MVT::v2bf16:   return "MVT::v2bf16";
  case MVT::v3bf16:   return "MVT::v3bf16";
  case MVT::v4bf16:   return "MVT::v4bf16";
  case MVT::v8bf16:   return "MVT::v8bf16";
  case MVT::v16bf16:  return "MVT::v16bf16";
  case MVT::v32bf16:  return "MVT::v32bf16";
  case MVT::v64bf16:  return "MVT::v64bf16";
  case MVT::v128bf16: return "MVT::v128bf16";
  case MVT::v1f32:    return "MVT::v1f32";
  case MVT::v2f32:    return "MVT::v2f32";
  case MVT::v3f32:    return "MVT::v3f32";
  case MVT::v4f32:    return "MVT::v4f32";
  case MVT::v5f32:    return "MVT::v5f32";
  case MVT::v6f32:    return "MVT::v6f32";
  case MVT::v7f32:    return "MVT::v7f32";
  case MVT::v8f32:    return "MVT::v8f32";
  case MVT::v9f32:    return "MVT::v9f32";
  case MVT::v10f32:   return "MVT::v10f32";
  case MVT::v11f32:   return "MVT::v11f32";
  case MVT::v12f32:   return "MVT::v12f32";
  case MVT::v16f32:   return "MVT::v16f32";
  case MVT::v32f32:   return "MVT::v32f32";
  case MVT::v64f32:   return "MVT::v64f32";
  case MVT::v128f32:  return "MVT::v128f32";
  case MVT::v256f32:  return "MVT::v256f32";
  case MVT::v512f32:  return "MVT::v512f32";
  case MVT::v1024f32: return "MVT::v1024f32";
  case MVT::v2048f32: return "MVT::v2048f32";
  case MVT::v1f64:    return "MVT::v1f64";
  case MVT::v2f64:    return "MVT::v2f64";
  case MVT::v3f64:    return "MVT::v3f64";
  case MVT::v4f64:    return "MVT::v4f64";
  case MVT::v8f64:    return "MVT::v8f64";
  case MVT::v16f64:   return "MVT::v16f64";
  case MVT::v32f64:   return "MVT::v32f64";
  case MVT::v64f64:   return "MVT::v64f64";
  case MVT::v128f64:  return "MVT::v128f64";
  case MVT::v256f64:  return "MVT::v256f64";
  case MVT::nxv1i1:   return "MVT::nxv1i1";
  case MVT::nxv2i1:   return "MVT::nxv2i1";
  case MVT::nxv4i1:   return "MVT::nxv4i1";
  case MVT::nxv8i1:   return "MVT::nxv8i1";
  case MVT::nxv16i1:  return "MVT::nxv16i1";
  case MVT::nxv32i1:  return "MVT::nxv32i1";
  case MVT::nxv64i1:  return "MVT::nxv64i1";
  case MVT::nxv1i8:   return "MVT::nxv1i8";
  case MVT::nxv2i8:   return "MVT::nxv2i8";
  case MVT::nxv4i8:   return "MVT::nxv4i8";
  case MVT::nxv8i8:   return "MVT::nxv8i8";
  case MVT::nxv16i8:  return "MVT::nxv16i8";
  case MVT::nxv32i8:  return "MVT::nxv32i8";
  case MVT::nxv64i8:  return "MVT::nxv64i8";
  case MVT::nxv1i16:  return "MVT::nxv1i16";
  case MVT::nxv2i16:  return "MVT::nxv2i16";
  case MVT::nxv4i16:  return "MVT::nxv4i16";
  case MVT::nxv8i16:  return "MVT::nxv8i16";
  case MVT::nxv16i16: return "MVT::nxv16i16";
  case MVT::nxv32i16: return "MVT::nxv32i16";
  case MVT::nxv1i32:  return "MVT::nxv1i32";
  case MVT::nxv2i32:  return "MVT::nxv2i32";
  case MVT::nxv4i32:  return "MVT::nxv4i32";
  case MVT::nxv8i32:  return "MVT::nxv8i32";
  case MVT::nxv16i32: return "MVT::nxv16i32";
  case MVT::nxv32i32: return "MVT::nxv32i32";
  case MVT::nxv1i64:  return "MVT::nxv1i64";
  case MVT::nxv2i64:  return "MVT::nxv2i64";
  case MVT::nxv4i64:  return "MVT::nxv4i64";
  case MVT::nxv8i64:  return "MVT::nxv8i64";
  case MVT::nxv16i64: return "MVT::nxv16i64";
  case MVT::nxv32i64: return "MVT::nxv32i64";
  case MVT::nxv1f16:  return "MVT::nxv1f16";
  case MVT::nxv2f16:  return "MVT::nxv2f16";
  case MVT::nxv4f16:  return "MVT::nxv4f16";
  case MVT::nxv8f16:  return "MVT::nxv8f16";
  case MVT::nxv16f16: return "MVT::nxv16f16";
  case MVT::nxv32f16: return "MVT::nxv32f16";
  case MVT::nxv1bf16:  return "MVT::nxv1bf16";
  case MVT::nxv2bf16:  return "MVT::nxv2bf16";
  case MVT::nxv4bf16:  return "MVT::nxv4bf16";
  case MVT::nxv8bf16:  return "MVT::nxv8bf16";
  case MVT::nxv16bf16: return "MVT::nxv16bf16";
  case MVT::nxv32bf16: return "MVT::nxv32bf16";
  case MVT::nxv1f32:   return "MVT::nxv1f32";
  case MVT::nxv2f32:   return "MVT::nxv2f32";
  case MVT::nxv4f32:   return "MVT::nxv4f32";
  case MVT::nxv8f32:   return "MVT::nxv8f32";
  case MVT::nxv16f32:  return "MVT::nxv16f32";
  case MVT::nxv1f64:   return "MVT::nxv1f64";
  case MVT::nxv2f64:   return "MVT::nxv2f64";
  case MVT::nxv4f64:   return "MVT::nxv4f64";
  case MVT::nxv8f64:   return "MVT::nxv8f64";
  case MVT::token:     return "MVT::token";
  case MVT::Metadata:  return "MVT::Metadata";
  case MVT::iPTR:      return "MVT::iPTR";
  case MVT::iPTRAny:   return "MVT::iPTRAny";
  case MVT::Untyped:   return "MVT::Untyped";
  case MVT::funcref:   return "MVT::funcref";
  case MVT::externref: return "MVT::externref";
  default: llvm_unreachable("ILLEGAL VALUE TYPE!");
  }
  // clang-format on
}

/// getQualifiedName - Return the name of the specified record, with a
/// namespace qualifier if the record contains one.
///
std::string llvm::getQualifiedName(const Record *R) {
  std::string Namespace;
  if (R->getValue("Namespace"))
    Namespace = std::string(R->getValueAsString("Namespace"));
  if (Namespace.empty())
    return std::string(R->getName());
  return Namespace + "::" + R->getName().str();
}


/// getTarget - Return the current instance of the Target class.
///
CodeGenTarget::CodeGenTarget(RecordKeeper &records)
  : Records(records), CGH(records) {
  std::vector<Record*> Targets = Records.getAllDerivedDefinitions("Target");
  if (Targets.size() == 0)
    PrintFatalError("No 'Target' subclasses defined!");
  if (Targets.size() != 1)
    PrintFatalError("Multiple subclasses of Target defined!");
  TargetRec = Targets[0];
}

CodeGenTarget::~CodeGenTarget() {
}

StringRef CodeGenTarget::getName() const { return TargetRec->getName(); }

/// getInstNamespace - Find and return the target machine's instruction
/// namespace. The namespace is cached because it is requested multiple times.
StringRef CodeGenTarget::getInstNamespace() const {
  if (InstNamespace.empty()) {
    for (const CodeGenInstruction *Inst : getInstructionsByEnumValue()) {
      // We are not interested in the "TargetOpcode" namespace.
      if (Inst->Namespace != "TargetOpcode") {
        InstNamespace = Inst->Namespace;
        break;
      }
    }
  }

  return InstNamespace;
}

StringRef CodeGenTarget::getRegNamespace() const {
  auto &RegClasses = RegBank->getRegClasses();
  return RegClasses.size() > 0 ? RegClasses.front().Namespace : "";
}

Record *CodeGenTarget::getInstructionSet() const {
  return TargetRec->getValueAsDef("InstructionSet");
}

bool CodeGenTarget::getAllowRegisterRenaming() const {
  return TargetRec->getValueAsInt("AllowRegisterRenaming");
}

/// getAsmParser - Return the AssemblyParser definition for this target.
///
Record *CodeGenTarget::getAsmParser() const {
  std::vector<Record*> LI = TargetRec->getValueAsListOfDefs("AssemblyParsers");
  if (AsmParserNum >= LI.size())
    PrintFatalError("Target does not have an AsmParser #" +
                    Twine(AsmParserNum) + "!");
  return LI[AsmParserNum];
}

/// getAsmParserVariant - Return the AssemblyParserVariant definition for
/// this target.
///
Record *CodeGenTarget::getAsmParserVariant(unsigned i) const {
  std::vector<Record*> LI =
    TargetRec->getValueAsListOfDefs("AssemblyParserVariants");
  if (i >= LI.size())
    PrintFatalError("Target does not have an AsmParserVariant #" + Twine(i) +
                    "!");
  return LI[i];
}

/// getAsmParserVariantCount - Return the AssemblyParserVariant definition
/// available for this target.
///
unsigned CodeGenTarget::getAsmParserVariantCount() const {
  std::vector<Record*> LI =
    TargetRec->getValueAsListOfDefs("AssemblyParserVariants");
  return LI.size();
}

/// getAsmWriter - Return the AssemblyWriter definition for this target.
///
Record *CodeGenTarget::getAsmWriter() const {
  std::vector<Record*> LI = TargetRec->getValueAsListOfDefs("AssemblyWriters");
  if (AsmWriterNum >= LI.size())
    PrintFatalError("Target does not have an AsmWriter #" +
                    Twine(AsmWriterNum) + "!");
  return LI[AsmWriterNum];
}

CodeGenRegBank &CodeGenTarget::getRegBank() const {
  if (!RegBank)
    RegBank = std::make_unique<CodeGenRegBank>(Records, getHwModes());
  return *RegBank;
}

std::optional<CodeGenRegisterClass *> CodeGenTarget::getSuperRegForSubReg(
    const ValueTypeByHwMode &ValueTy, CodeGenRegBank &RegBank,
    const CodeGenSubRegIndex *SubIdx, bool MustBeAllocatable) const {
  std::vector<CodeGenRegisterClass *> Candidates;
  auto &RegClasses = RegBank.getRegClasses();

  // Try to find a register class which supports ValueTy, and also contains
  // SubIdx.
  for (CodeGenRegisterClass &RC : RegClasses) {
    // Is there a subclass of this class which contains this subregister index?
    CodeGenRegisterClass *SubClassWithSubReg = RC.getSubClassWithSubReg(SubIdx);
    if (!SubClassWithSubReg)
      continue;

    // We have a class. Check if it supports this value type.
    if (!llvm::is_contained(SubClassWithSubReg->VTs, ValueTy))
      continue;

    // If necessary, check that it is allocatable.
    if (MustBeAllocatable && !SubClassWithSubReg->Allocatable)
      continue;

    // We have a register class which supports both the value type and
    // subregister index. Remember it.
    Candidates.push_back(SubClassWithSubReg);
  }

  // If we didn't find anything, we're done.
  if (Candidates.empty())
    return std::nullopt;

  // Find and return the largest of our candidate classes.
  llvm::stable_sort(Candidates, [&](const CodeGenRegisterClass *A,
                                    const CodeGenRegisterClass *B) {
    if (A->getMembers().size() > B->getMembers().size())
      return true;

    if (A->getMembers().size() < B->getMembers().size())
      return false;

    // Order by name as a tie-breaker.
    return StringRef(A->getName()) < B->getName();
  });

  return Candidates[0];
}

void CodeGenTarget::ReadRegAltNameIndices() const {
  RegAltNameIndices = Records.getAllDerivedDefinitions("RegAltNameIndex");
  llvm::sort(RegAltNameIndices, LessRecord());
}

/// getRegisterByName - If there is a register with the specific AsmName,
/// return it.
const CodeGenRegister *CodeGenTarget::getRegisterByName(StringRef Name) const {
  return getRegBank().getRegistersByName().lookup(Name);
}

std::vector<ValueTypeByHwMode> CodeGenTarget::getRegisterVTs(Record *R)
      const {
  const CodeGenRegister *Reg = getRegBank().getReg(R);
  std::vector<ValueTypeByHwMode> Result;
  for (const auto &RC : getRegBank().getRegClasses()) {
    if (RC.contains(Reg)) {
      ArrayRef<ValueTypeByHwMode> InVTs = RC.getValueTypes();
      llvm::append_range(Result, InVTs);
    }
  }

  // Remove duplicates.
  llvm::sort(Result);
  Result.erase(std::unique(Result.begin(), Result.end()), Result.end());
  return Result;
}


void CodeGenTarget::ReadLegalValueTypes() const {
  for (const auto &RC : getRegBank().getRegClasses())
    llvm::append_range(LegalValueTypes, RC.VTs);

  // Remove duplicates.
  llvm::sort(LegalValueTypes);
  LegalValueTypes.erase(std::unique(LegalValueTypes.begin(),
                                    LegalValueTypes.end()),
                        LegalValueTypes.end());
}

CodeGenSchedModels &CodeGenTarget::getSchedModels() const {
  if (!SchedModels)
    SchedModels = std::make_unique<CodeGenSchedModels>(Records, *this);
  return *SchedModels;
}

void CodeGenTarget::ReadInstructions() const {
  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");
  if (Insts.size() <= 2)
    PrintFatalError("No 'Instruction' subclasses defined!");

  // Parse the instructions defined in the .td file.
  for (unsigned i = 0, e = Insts.size(); i != e; ++i)
    Instructions[Insts[i]] = std::make_unique<CodeGenInstruction>(Insts[i]);
}

static const CodeGenInstruction *
GetInstByName(const char *Name,
              const DenseMap<const Record*,
                             std::unique_ptr<CodeGenInstruction>> &Insts,
              RecordKeeper &Records) {
  const Record *Rec = Records.getDef(Name);

  const auto I = Insts.find(Rec);
  if (!Rec || I == Insts.end())
    PrintFatalError(Twine("Could not find '") + Name + "' instruction!");
  return I->second.get();
}

static const char *FixedInstrs[] = {
#define HANDLE_TARGET_OPCODE(OPC) #OPC,
#include "llvm/Support/TargetOpcodes.def"
    nullptr};

unsigned CodeGenTarget::getNumFixedInstructions() {
  return std::size(FixedInstrs) - 1;
}

/// Return all of the instructions defined by the target, ordered by
/// their enum value.
void CodeGenTarget::ComputeInstrsByEnum() const {
  const auto &Insts = getInstructions();
  for (const char *const *p = FixedInstrs; *p; ++p) {
    const CodeGenInstruction *Instr = GetInstByName(*p, Insts, Records);
    assert(Instr && "Missing target independent instruction");
    assert(Instr->Namespace == "TargetOpcode" && "Bad namespace");
    InstrsByEnum.push_back(Instr);
  }
  unsigned EndOfPredefines = InstrsByEnum.size();
  assert(EndOfPredefines == getNumFixedInstructions() &&
         "Missing generic opcode");

  for (const auto &I : Insts) {
    const CodeGenInstruction *CGI = I.second.get();
    if (CGI->Namespace != "TargetOpcode") {
      InstrsByEnum.push_back(CGI);
      if (CGI->TheDef->getValueAsBit("isPseudo"))
        ++NumPseudoInstructions;
    }
  }

  assert(InstrsByEnum.size() == Insts.size() && "Missing predefined instr");

  // All of the instructions are now in random order based on the map iteration.
  llvm::sort(
      InstrsByEnum.begin() + EndOfPredefines, InstrsByEnum.end(),
      [](const CodeGenInstruction *Rec1, const CodeGenInstruction *Rec2) {
        const auto &D1 = *Rec1->TheDef;
        const auto &D2 = *Rec2->TheDef;
        return std::make_tuple(!D1.getValueAsBit("isPseudo"), D1.getName()) <
               std::make_tuple(!D2.getValueAsBit("isPseudo"), D2.getName());
      });
}


/// isLittleEndianEncoding - Return whether this target encodes its instruction
/// in little-endian format, i.e. bits laid out in the order [0..n]
///
bool CodeGenTarget::isLittleEndianEncoding() const {
  return getInstructionSet()->getValueAsBit("isLittleEndianEncoding");
}

/// reverseBitsForLittleEndianEncoding - For little-endian instruction bit
/// encodings, reverse the bit order of all instructions.
void CodeGenTarget::reverseBitsForLittleEndianEncoding() {
  if (!isLittleEndianEncoding())
    return;

  std::vector<Record *> Insts =
      Records.getAllDerivedDefinitions("InstructionEncoding");
  for (Record *R : Insts) {
    if (R->getValueAsString("Namespace") == "TargetOpcode" ||
        R->getValueAsBit("isPseudo"))
      continue;

    BitsInit *BI = R->getValueAsBitsInit("Inst");

    unsigned numBits = BI->getNumBits();

    SmallVector<Init *, 16> NewBits(numBits);

    for (unsigned bit = 0, end = numBits / 2; bit != end; ++bit) {
      unsigned bitSwapIdx = numBits - bit - 1;
      Init *OrigBit = BI->getBit(bit);
      Init *BitSwap = BI->getBit(bitSwapIdx);
      NewBits[bit]        = BitSwap;
      NewBits[bitSwapIdx] = OrigBit;
    }
    if (numBits % 2) {
      unsigned middle = (numBits + 1) / 2;
      NewBits[middle] = BI->getBit(middle);
    }

    BitsInit *NewBI = BitsInit::get(Records, NewBits);

    // Update the bits in reversed order so that emitInstrOpBits will get the
    // correct endianness.
    R->getValue("Inst")->setValue(NewBI);
  }
}

/// guessInstructionProperties - Return true if it's OK to guess instruction
/// properties instead of raising an error.
///
/// This is configurable as a temporary migration aid. It will eventually be
/// permanently false.
bool CodeGenTarget::guessInstructionProperties() const {
  return getInstructionSet()->getValueAsBit("guessInstructionProperties");
}

//===----------------------------------------------------------------------===//
// ComplexPattern implementation
//
ComplexPattern::ComplexPattern(Record *R) {
  Ty          = R->getValueAsDef("Ty");
  NumOperands = R->getValueAsInt("NumOperands");
  SelectFunc = std::string(R->getValueAsString("SelectFunc"));
  RootNodes   = R->getValueAsListOfDefs("RootNodes");

  // FIXME: This is a hack to statically increase the priority of patterns which
  // maps a sub-dag to a complex pattern. e.g. favors LEA over ADD. To get best
  // possible pattern match we'll need to dynamically calculate the complexity
  // of all patterns a dag can potentially map to.
  int64_t RawComplexity = R->getValueAsInt("Complexity");
  if (RawComplexity == -1)
    Complexity = NumOperands * 3;
  else
    Complexity = RawComplexity;

  // FIXME: Why is this different from parseSDPatternOperatorProperties?
  // Parse the properties.
  Properties = 0;
  std::vector<Record*> PropList = R->getValueAsListOfDefs("Properties");
  for (unsigned i = 0, e = PropList.size(); i != e; ++i)
    if (PropList[i]->getName() == "SDNPHasChain") {
      Properties |= 1 << SDNPHasChain;
    } else if (PropList[i]->getName() == "SDNPOptInGlue") {
      Properties |= 1 << SDNPOptInGlue;
    } else if (PropList[i]->getName() == "SDNPMayStore") {
      Properties |= 1 << SDNPMayStore;
    } else if (PropList[i]->getName() == "SDNPMayLoad") {
      Properties |= 1 << SDNPMayLoad;
    } else if (PropList[i]->getName() == "SDNPSideEffect") {
      Properties |= 1 << SDNPSideEffect;
    } else if (PropList[i]->getName() == "SDNPMemOperand") {
      Properties |= 1 << SDNPMemOperand;
    } else if (PropList[i]->getName() == "SDNPVariadic") {
      Properties |= 1 << SDNPVariadic;
    } else if (PropList[i]->getName() == "SDNPWantRoot") {
      Properties |= 1 << SDNPWantRoot;
    } else if (PropList[i]->getName() == "SDNPWantParent") {
      Properties |= 1 << SDNPWantParent;
    } else {
      PrintFatalError(R->getLoc(), "Unsupported SD Node property '" +
                                       PropList[i]->getName() +
                                       "' on ComplexPattern '" + R->getName() +
                                       "'!");
    }
}

//===----------------------------------------------------------------------===//
// CodeGenIntrinsic Implementation
//===----------------------------------------------------------------------===//

CodeGenIntrinsicTable::CodeGenIntrinsicTable(const RecordKeeper &RC) {
  std::vector<Record *> IntrProperties =
      RC.getAllDerivedDefinitions("IntrinsicProperty");

  std::vector<Record *> DefaultProperties;
  for (Record *Rec : IntrProperties)
    if (Rec->getValueAsBit("IsDefault"))
      DefaultProperties.push_back(Rec);

  std::vector<Record *> Defs = RC.getAllDerivedDefinitions("Intrinsic");
  Intrinsics.reserve(Defs.size());

  for (unsigned I = 0, e = Defs.size(); I != e; ++I)
    Intrinsics.push_back(CodeGenIntrinsic(Defs[I], DefaultProperties));

  llvm::sort(Intrinsics,
             [](const CodeGenIntrinsic &LHS, const CodeGenIntrinsic &RHS) {
               return std::tie(LHS.TargetPrefix, LHS.Name) <
                      std::tie(RHS.TargetPrefix, RHS.Name);
             });
  Targets.push_back({"", 0, 0});
  for (size_t I = 0, E = Intrinsics.size(); I < E; ++I)
    if (Intrinsics[I].TargetPrefix != Targets.back().Name) {
      Targets.back().Count = I - Targets.back().Offset;
      Targets.push_back({Intrinsics[I].TargetPrefix, I, 0});
    }
  Targets.back().Count = Intrinsics.size() - Targets.back().Offset;
}

CodeGenIntrinsic::CodeGenIntrinsic(Record *R,
                                   std::vector<Record *> DefaultProperties) {
  TheDef = R;
  std::string DefName = std::string(R->getName());
  ArrayRef<SMLoc> DefLoc = R->getLoc();
  Properties = 0;
  isOverloaded = false;
  isCommutative = false;
  canThrow = false;
  isNoReturn = false;
  isNoCallback = false;
  isNoSync = false;
  isNoFree = false;
  isWillReturn = false;
  isCold = false;
  isNoDuplicate = false;
  isNoMerge = false;
  isConvergent = false;
  isSpeculatable = false;
  hasSideEffects = false;

  if (DefName.size() <= 4 || DefName.substr(0, 4) != "int_")
    PrintFatalError(DefLoc,
                    "Intrinsic '" + DefName + "' does not start with 'int_'!");

  EnumName = DefName.substr(4);

  if (R->getValue(
          "ClangBuiltinName")) // Ignore a missing ClangBuiltinName field.
    ClangBuiltinName = std::string(R->getValueAsString("ClangBuiltinName"));
  if (R->getValue("MSBuiltinName")) // Ignore a missing MSBuiltinName field.
    MSBuiltinName = std::string(R->getValueAsString("MSBuiltinName"));

  TargetPrefix = std::string(R->getValueAsString("TargetPrefix"));
  Name = std::string(R->getValueAsString("LLVMName"));

  if (Name == "") {
    // If an explicit name isn't specified, derive one from the DefName.
    Name = "llvm.";

    for (unsigned i = 0, e = EnumName.size(); i != e; ++i)
      Name += (EnumName[i] == '_') ? '.' : EnumName[i];
  } else {
    // Verify it starts with "llvm.".
    if (Name.size() <= 5 || Name.substr(0, 5) != "llvm.")
      PrintFatalError(DefLoc, "Intrinsic '" + DefName +
                                  "'s name does not start with 'llvm.'!");
  }

  // If TargetPrefix is specified, make sure that Name starts with
  // "llvm.<targetprefix>.".
  if (!TargetPrefix.empty()) {
    if (Name.size() < 6 + TargetPrefix.size() ||
        Name.substr(5, 1 + TargetPrefix.size()) != (TargetPrefix + "."))
      PrintFatalError(DefLoc, "Intrinsic '" + DefName +
                                  "' does not start with 'llvm." +
                                  TargetPrefix + ".'!");
  }

  ListInit *RetTypes = R->getValueAsListInit("RetTypes");
  ListInit *ParamTypes = R->getValueAsListInit("ParamTypes");

  // First collate a list of overloaded types.
  std::vector<MVT::SimpleValueType> OverloadedVTs;
  for (ListInit *TypeList : {RetTypes, ParamTypes}) {
    for (unsigned i = 0, e = TypeList->size(); i != e; ++i) {
      Record *TyEl = TypeList->getElementAsRecord(i);
      assert(TyEl->isSubClassOf("LLVMType") && "Expected a type!");

      if (TyEl->isSubClassOf("LLVMMatchType"))
        continue;

      MVT::SimpleValueType VT = getValueType(TyEl->getValueAsDef("VT"));
      if (MVT(VT).isOverloaded()) {
        OverloadedVTs.push_back(VT);
        isOverloaded = true;
      }
    }
  }

  // Parse the list of return types.
  ListInit *TypeList = RetTypes;
  for (unsigned i = 0, e = TypeList->size(); i != e; ++i) {
    Record *TyEl = TypeList->getElementAsRecord(i);
    assert(TyEl->isSubClassOf("LLVMType") && "Expected a type!");
    MVT::SimpleValueType VT;
    if (TyEl->isSubClassOf("LLVMMatchType")) {
      unsigned MatchTy = TyEl->getValueAsInt("Number");
      assert(MatchTy < OverloadedVTs.size() && "Invalid matching number!");
      VT = OverloadedVTs[MatchTy];
      // It only makes sense to use the extended and truncated vector element
      // variants with iAny types; otherwise, if the intrinsic is not
      // overloaded, all the types can be specified directly.
      assert(((!TyEl->isSubClassOf("LLVMExtendedType") &&
               !TyEl->isSubClassOf("LLVMTruncatedType")) ||
              VT == MVT::iAny || VT == MVT::vAny) &&
             "Expected iAny or vAny type");
    } else {
      VT = getValueType(TyEl->getValueAsDef("VT"));
    }

    // Reject invalid types.
    if (VT == MVT::isVoid)
      PrintFatalError(DefLoc, "Intrinsic '" + DefName +
                                  " has void in result type list!");

    IS.RetVTs.push_back(VT);
    IS.RetTypeDefs.push_back(TyEl);
  }

  // Parse the list of parameter types.
  TypeList = ParamTypes;
  for (unsigned i = 0, e = TypeList->size(); i != e; ++i) {
    Record *TyEl = TypeList->getElementAsRecord(i);
    assert(TyEl->isSubClassOf("LLVMType") && "Expected a type!");
    MVT::SimpleValueType VT;
    if (TyEl->isSubClassOf("LLVMMatchType")) {
      unsigned MatchTy = TyEl->getValueAsInt("Number");
      if (MatchTy >= OverloadedVTs.size()) {
        PrintError(R->getLoc(), "Parameter #" + Twine(i) +
                                    " has out of bounds matching "
                                    "number " +
                                    Twine(MatchTy));
        PrintFatalError(DefLoc,
                        Twine("ParamTypes is ") + TypeList->getAsString());
      }
      VT = OverloadedVTs[MatchTy];
      // It only makes sense to use the extended and truncated vector element
      // variants with iAny types; otherwise, if the intrinsic is not
      // overloaded, all the types can be specified directly.
      assert(((!TyEl->isSubClassOf("LLVMExtendedType") &&
               !TyEl->isSubClassOf("LLVMTruncatedType")) ||
              VT == MVT::iAny || VT == MVT::vAny) &&
             "Expected iAny or vAny type");
    } else
      VT = getValueType(TyEl->getValueAsDef("VT"));

    // Reject invalid types.
    if (VT == MVT::isVoid && i != e - 1 /*void at end means varargs*/)
      PrintFatalError(DefLoc, "Intrinsic '" + DefName +
                                  " has void in result type list!");

    IS.ParamVTs.push_back(VT);
    IS.ParamTypeDefs.push_back(TyEl);
  }

  // Parse the intrinsic properties.
  ListInit *PropList = R->getValueAsListInit("IntrProperties");
  for (unsigned i = 0, e = PropList->size(); i != e; ++i) {
    Record *Property = PropList->getElementAsRecord(i);
    assert(Property->isSubClassOf("IntrinsicProperty") &&
           "Expected a property!");

    setProperty(Property);
  }

  // Set default properties to true.
  setDefaultProperties(R, DefaultProperties);

  // Also record the SDPatternOperator Properties.
  Properties = parseSDPatternOperatorProperties(R);

  // Sort the argument attributes for later benefit.
  for (auto &Attrs : ArgumentAttributes)
    llvm::sort(Attrs);
}

void CodeGenIntrinsic::setDefaultProperties(
    Record *R, std::vector<Record *> DefaultProperties) {
  // opt-out of using default attributes.
  if (R->getValueAsBit("DisableDefaultAttributes"))
    return;

  for (Record *Rec : DefaultProperties)
    setProperty(Rec);
}

void CodeGenIntrinsic::setProperty(Record *R) {
  if (R->getName() == "IntrNoMem")
    ME = MemoryEffects::none();
  else if (R->getName() == "IntrReadMem") {
    if (ME.onlyWritesMemory())
      PrintFatalError(TheDef->getLoc(),
                      Twine("IntrReadMem cannot be used after IntrNoMem or "
                            "IntrWriteMem. Default is ReadWrite"));
    ME &= MemoryEffects::readOnly();
  } else if (R->getName() == "IntrWriteMem") {
    if (ME.onlyReadsMemory())
      PrintFatalError(TheDef->getLoc(),
                      Twine("IntrWriteMem cannot be used after IntrNoMem or "
                            "IntrReadMem. Default is ReadWrite"));
    ME &= MemoryEffects::writeOnly();
  } else if (R->getName() == "IntrArgMemOnly")
    ME &= MemoryEffects::argMemOnly();
  else if (R->getName() == "IntrInaccessibleMemOnly")
    ME &= MemoryEffects::inaccessibleMemOnly();
  else if (R->getName() == "IntrInaccessibleMemOrArgMemOnly")
    ME &= MemoryEffects::inaccessibleOrArgMemOnly();
  else if (R->getName() == "Commutative")
    isCommutative = true;
  else if (R->getName() == "Throws")
    canThrow = true;
  else if (R->getName() == "IntrNoDuplicate")
    isNoDuplicate = true;
  else if (R->getName() == "IntrNoMerge")
    isNoMerge = true;
  else if (R->getName() == "IntrConvergent")
    isConvergent = true;
  else if (R->getName() == "IntrNoReturn")
    isNoReturn = true;
  else if (R->getName() == "IntrNoCallback")
    isNoCallback = true;
  else if (R->getName() == "IntrNoSync")
    isNoSync = true;
  else if (R->getName() == "IntrNoFree")
    isNoFree = true;
  else if (R->getName() == "IntrWillReturn")
    isWillReturn = !isNoReturn;
  else if (R->getName() == "IntrCold")
    isCold = true;
  else if (R->getName() == "IntrSpeculatable")
    isSpeculatable = true;
  else if (R->getName() == "IntrHasSideEffects")
    hasSideEffects = true;
  else if (R->isSubClassOf("NoCapture")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, NoCapture);
  } else if (R->isSubClassOf("NoAlias")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, NoAlias);
  } else if (R->isSubClassOf("NoUndef")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, NoUndef);
  } else if (R->isSubClassOf("NonNull")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, NonNull);
  } else if (R->isSubClassOf("Returned")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, Returned);
  } else if (R->isSubClassOf("ReadOnly")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, ReadOnly);
  } else if (R->isSubClassOf("WriteOnly")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, WriteOnly);
  } else if (R->isSubClassOf("ReadNone")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, ReadNone);
  } else if (R->isSubClassOf("ImmArg")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    addArgAttribute(ArgNo, ImmArg);
  } else if (R->isSubClassOf("Align")) {
    unsigned ArgNo = R->getValueAsInt("ArgNo");
    uint64_t Align = R->getValueAsInt("Align");
    addArgAttribute(ArgNo, Alignment, Align);
  } else
    llvm_unreachable("Unknown property!");
}

bool CodeGenIntrinsic::isParamAPointer(unsigned ParamIdx) const {
  if (ParamIdx >= IS.ParamVTs.size())
    return false;
  MVT ParamType = MVT(IS.ParamVTs[ParamIdx]);
  return ParamType == MVT::iPTR || ParamType == MVT::iPTRAny;
}

bool CodeGenIntrinsic::isParamImmArg(unsigned ParamIdx) const {
  // Convert argument index to attribute index starting from `FirstArgIndex`.
  ++ParamIdx;
  if (ParamIdx >= ArgumentAttributes.size())
    return false;
  ArgAttribute Val{ImmArg, 0};
  return std::binary_search(ArgumentAttributes[ParamIdx].begin(),
                            ArgumentAttributes[ParamIdx].end(), Val);
}

void CodeGenIntrinsic::addArgAttribute(unsigned Idx, ArgAttrKind AK,
                                       uint64_t V) {
  if (Idx >= ArgumentAttributes.size())
    ArgumentAttributes.resize(Idx + 1);
  ArgumentAttributes[Idx].emplace_back(AK, V);
}
