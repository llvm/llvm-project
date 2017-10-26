//===- FileAnalysis.cpp -----------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FileAnalysis.h"
#include "GraphBuilder.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

using Instr = llvm::cfi_verify::FileAnalysis::Instr;

namespace llvm {
namespace cfi_verify {

Expected<FileAnalysis> FileAnalysis::Create(StringRef Filename) {
  // Open the filename provided.
  Expected<object::OwningBinary<object::Binary>> BinaryOrErr =
      object::createBinary(Filename);
  if (!BinaryOrErr)
    return BinaryOrErr.takeError();

  // Construct the object and allow it to take ownership of the binary.
  object::OwningBinary<object::Binary> Binary = std::move(BinaryOrErr.get());
  FileAnalysis Analysis(std::move(Binary));

  Analysis.Object = dyn_cast<object::ObjectFile>(Analysis.Binary.getBinary());
  if (!Analysis.Object)
    return make_error<UnsupportedDisassembly>("Failed to cast object");

  Analysis.ObjectTriple = Analysis.Object->makeTriple();
  Analysis.Features = Analysis.Object->getFeatures();

  // Init the rest of the object.
  if (auto InitResponse = Analysis.initialiseDisassemblyMembers())
    return std::move(InitResponse);

  if (auto SectionParseResponse = Analysis.parseCodeSections())
    return std::move(SectionParseResponse);

  return std::move(Analysis);
}

FileAnalysis::FileAnalysis(object::OwningBinary<object::Binary> Binary)
    : Binary(std::move(Binary)) {}

FileAnalysis::FileAnalysis(const Triple &ObjectTriple,
                           const SubtargetFeatures &Features)
    : ObjectTriple(ObjectTriple), Features(Features) {}

bool FileAnalysis::isIndirectInstructionCFIProtected(uint64_t Address) const {
  const Instr *InstrMetaPtr = getInstruction(Address);
  if (!InstrMetaPtr)
    return false;

  const auto &InstrDesc = MII->get(InstrMetaPtr->Instruction.getOpcode());

  if (!InstrDesc.mayAffectControlFlow(InstrMetaPtr->Instruction, *RegisterInfo))
    return false;

  if (!usesRegisterOperand(*InstrMetaPtr))
    return false;

  auto Flows = GraphBuilder::buildFlowGraph(*this, Address);

  if (!Flows.OrphanedNodes.empty())
    return false;

  for (const auto &BranchNode : Flows.ConditionalBranchNodes) {
    if (!BranchNode.CFIProtection)
      return false;
  }

  return true;
}

const Instr *
FileAnalysis::getPrevInstructionSequential(const Instr &InstrMeta) const {
  std::map<uint64_t, Instr>::const_iterator KV =
      Instructions.find(InstrMeta.VMAddress);
  if (KV == Instructions.end() || KV == Instructions.begin())
    return nullptr;

  if (!(--KV)->second.Valid)
    return nullptr;

  return &KV->second;
}

const Instr *
FileAnalysis::getNextInstructionSequential(const Instr &InstrMeta) const {
  std::map<uint64_t, Instr>::const_iterator KV =
      Instructions.find(InstrMeta.VMAddress);
  if (KV == Instructions.end() || ++KV == Instructions.end())
    return nullptr;

  if (!KV->second.Valid)
    return nullptr;

  return &KV->second;
}

bool FileAnalysis::usesRegisterOperand(const Instr &InstrMeta) const {
  for (const auto &Operand : InstrMeta.Instruction) {
    if (Operand.isReg())
      return true;
  }
  return false;
}

const Instr *FileAnalysis::getInstruction(uint64_t Address) const {
  const auto &InstrKV = Instructions.find(Address);
  if (InstrKV == Instructions.end())
    return nullptr;

  return &InstrKV->second;
}

const Instr &FileAnalysis::getInstructionOrDie(uint64_t Address) const {
  const auto &InstrKV = Instructions.find(Address);
  assert(InstrKV != Instructions.end() && "Address doesn't exist.");
  return InstrKV->second;
}

bool FileAnalysis::isCFITrap(const Instr &InstrMeta) const {
  return MII->getName(InstrMeta.Instruction.getOpcode()) == "TRAP";
}

bool FileAnalysis::canFallThrough(const Instr &InstrMeta) const {
  if (!InstrMeta.Valid)
    return false;

  if (isCFITrap(InstrMeta))
    return false;

  const auto &InstrDesc = MII->get(InstrMeta.Instruction.getOpcode());
  if (InstrDesc.mayAffectControlFlow(InstrMeta.Instruction, *RegisterInfo))
    return InstrDesc.isConditionalBranch();

  return true;
}

const Instr *
FileAnalysis::getDefiniteNextInstruction(const Instr &InstrMeta) const {
  if (!InstrMeta.Valid)
    return nullptr;

  if (isCFITrap(InstrMeta))
    return nullptr;

  const auto &InstrDesc = MII->get(InstrMeta.Instruction.getOpcode());
  const Instr *NextMetaPtr;
  if (InstrDesc.mayAffectControlFlow(InstrMeta.Instruction, *RegisterInfo)) {
    if (InstrDesc.isConditionalBranch())
      return nullptr;

    uint64_t Target;
    if (!MIA->evaluateBranch(InstrMeta.Instruction, InstrMeta.VMAddress,
                             InstrMeta.InstructionSize, Target))
      return nullptr;

    NextMetaPtr = getInstruction(Target);
  } else {
    NextMetaPtr =
        getInstruction(InstrMeta.VMAddress + InstrMeta.InstructionSize);
  }

  if (!NextMetaPtr || !NextMetaPtr->Valid)
    return nullptr;

  return NextMetaPtr;
}

std::set<const Instr *>
FileAnalysis::getDirectControlFlowXRefs(const Instr &InstrMeta) const {
  std::set<const Instr *> CFCrossReferences;
  const Instr *PrevInstruction = getPrevInstructionSequential(InstrMeta);

  if (PrevInstruction && canFallThrough(*PrevInstruction))
    CFCrossReferences.insert(PrevInstruction);

  const auto &TargetRefsKV = StaticBranchTargetings.find(InstrMeta.VMAddress);
  if (TargetRefsKV == StaticBranchTargetings.end())
    return CFCrossReferences;

  for (uint64_t SourceInstrAddress : TargetRefsKV->second) {
    const auto &SourceInstrKV = Instructions.find(SourceInstrAddress);
    if (SourceInstrKV == Instructions.end()) {
      errs() << "Failed to find source instruction at address "
             << format_hex(SourceInstrAddress, 2)
             << " for the cross-reference to instruction at address "
             << format_hex(InstrMeta.VMAddress, 2) << ".\n";
      continue;
    }

    CFCrossReferences.insert(&SourceInstrKV->second);
  }

  return CFCrossReferences;
}

const std::set<uint64_t> &FileAnalysis::getIndirectInstructions() const {
  return IndirectInstructions;
}

const MCRegisterInfo *FileAnalysis::getRegisterInfo() const {
  return RegisterInfo.get();
}

const MCInstrInfo *FileAnalysis::getMCInstrInfo() const { return MII.get(); }

const MCInstrAnalysis *FileAnalysis::getMCInstrAnalysis() const {
  return MIA.get();
}

Error FileAnalysis::initialiseDisassemblyMembers() {
  std::string TripleName = ObjectTriple.getTriple();
  ArchName = "";
  MCPU = "";
  std::string ErrorString;

  ObjectTarget =
      TargetRegistry::lookupTarget(ArchName, ObjectTriple, ErrorString);
  if (!ObjectTarget)
    return make_error<UnsupportedDisassembly>(
        (Twine("Couldn't find target \"") + ObjectTriple.getTriple() +
         "\", failed with error: " + ErrorString)
            .str());

  RegisterInfo.reset(ObjectTarget->createMCRegInfo(TripleName));
  if (!RegisterInfo)
    return make_error<UnsupportedDisassembly>(
        "Failed to initialise RegisterInfo.");

  AsmInfo.reset(ObjectTarget->createMCAsmInfo(*RegisterInfo, TripleName));
  if (!AsmInfo)
    return make_error<UnsupportedDisassembly>("Failed to initialise AsmInfo.");

  SubtargetInfo.reset(ObjectTarget->createMCSubtargetInfo(
      TripleName, MCPU, Features.getString()));
  if (!SubtargetInfo)
    return make_error<UnsupportedDisassembly>(
        "Failed to initialise SubtargetInfo.");

  MII.reset(ObjectTarget->createMCInstrInfo());
  if (!MII)
    return make_error<UnsupportedDisassembly>("Failed to initialise MII.");

  Context.reset(new MCContext(AsmInfo.get(), RegisterInfo.get(), &MOFI));

  Disassembler.reset(
      ObjectTarget->createMCDisassembler(*SubtargetInfo, *Context));

  if (!Disassembler)
    return make_error<UnsupportedDisassembly>(
        "No disassembler available for target");

  MIA.reset(ObjectTarget->createMCInstrAnalysis(MII.get()));

  Printer.reset(ObjectTarget->createMCInstPrinter(
      ObjectTriple, AsmInfo->getAssemblerDialect(), *AsmInfo, *MII,
      *RegisterInfo));

  return Error::success();
}

Error FileAnalysis::parseCodeSections() {
  for (const object::SectionRef &Section : Object->sections()) {
    // Ensure only executable sections get analysed.
    if (!(object::ELFSectionRef(Section).getFlags() & ELF::SHF_EXECINSTR))
      continue;

    StringRef SectionContents;
    if (Section.getContents(SectionContents))
      return make_error<StringError>("Failed to retrieve section contents",
                                     inconvertibleErrorCode());

    ArrayRef<uint8_t> SectionBytes((const uint8_t *)SectionContents.data(),
                                   Section.getSize());
    parseSectionContents(SectionBytes, Section.getAddress());
  }
  return Error::success();
}

void FileAnalysis::parseSectionContents(ArrayRef<uint8_t> SectionBytes,
                                        uint64_t SectionAddress) {
  MCInst Instruction;
  Instr InstrMeta;
  uint64_t InstructionSize;

  for (uint64_t Byte = 0; Byte < SectionBytes.size();) {
    bool ValidInstruction =
        Disassembler->getInstruction(Instruction, InstructionSize,
                                     SectionBytes.drop_front(Byte), 0, nulls(),
                                     outs()) == MCDisassembler::Success;

    Byte += InstructionSize;

    uint64_t VMAddress = SectionAddress + Byte - InstructionSize;
    InstrMeta.Instruction = Instruction;
    InstrMeta.VMAddress = VMAddress;
    InstrMeta.InstructionSize = InstructionSize;
    InstrMeta.Valid = ValidInstruction;
    addInstruction(InstrMeta);

    if (!ValidInstruction)
      continue;

    // Skip additional parsing for instructions that do not affect the control
    // flow.
    const auto &InstrDesc = MII->get(Instruction.getOpcode());
    if (!InstrDesc.mayAffectControlFlow(Instruction, *RegisterInfo))
      continue;

    uint64_t Target;
    if (MIA->evaluateBranch(Instruction, VMAddress, InstructionSize, Target)) {
      // If the target can be evaluated, it's not indirect.
      StaticBranchTargetings[Target].push_back(VMAddress);
      continue;
    }

    if (!usesRegisterOperand(InstrMeta))
      continue;

    IndirectInstructions.insert(VMAddress);
  }
}

void FileAnalysis::addInstruction(const Instr &Instruction) {
  const auto &KV =
      Instructions.insert(std::make_pair(Instruction.VMAddress, Instruction));
  if (!KV.second) {
    errs() << "Failed to add instruction at address "
           << format_hex(Instruction.VMAddress, 2)
           << ": Instruction at this address already exists.\n";
    exit(EXIT_FAILURE);
  }
}

UnsupportedDisassembly::UnsupportedDisassembly(StringRef Text) : Text(Text) {}

char UnsupportedDisassembly::ID;
void UnsupportedDisassembly::log(raw_ostream &OS) const {
  OS << "Could not initialise disassembler: " << Text;
}

std::error_code UnsupportedDisassembly::convertToErrorCode() const {
  return std::error_code();
}

} // namespace cfi_verify
} // namespace llvm
