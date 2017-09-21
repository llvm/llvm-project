//===-- llvm-cfi-verify.cpp - CFI Verification tool for LLVM --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool verifies Control Flow Integrity (CFI) instrumentation by static
// binary anaylsis. See the design document in /docs/CFIVerify.rst for more
// information.
//
// This tool is currently incomplete. It currently only does disassembly for
// object files, and searches through the code for indirect control flow
// instructions, printing them once found.
//
//===----------------------------------------------------------------------===//

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
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdlib>

using namespace llvm;
using namespace llvm::object;

cl::opt<bool> ArgDumpSymbols("sym", cl::desc("Dump the symbol table."));
cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::Required);

static void printSymbols(const ObjectFile *Object) {
  for (const SymbolRef &Symbol : Object->symbols()) {
    outs() << "Symbol [" << format_hex_no_prefix(Symbol.getValue(), 2)
           << "] = ";

    auto SymbolName = Symbol.getName();
    if (SymbolName)
      outs() << *SymbolName;
    else
      outs() << "UNKNOWN";

    if (Symbol.getFlags() & SymbolRef::SF_Hidden)
      outs() << " .hidden";

    outs() << " (Section = ";

    auto SymbolSection = Symbol.getSection();
    if (SymbolSection) {
      StringRef SymbolSectionName;
      if ((*SymbolSection)->getName(SymbolSectionName))
        outs() << "UNKNOWN)";
      else
        outs() << SymbolSectionName << ")";
    } else {
      outs() << "N/A)";
    }

    outs() << "\n";
  }
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllDisassemblers();

  Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(InputFilename);
  if (!BinaryOrErr) {
    errs() << "Failed to open file.\n";
    return EXIT_FAILURE;
  }

  Binary &Binary = *BinaryOrErr.get().getBinary();
  ObjectFile *Object = dyn_cast<ObjectFile>(&Binary);
  if (!Object) {
    errs() << "Disassembling of non-objects not currently supported.\n";
    return EXIT_FAILURE;
  }

  Triple TheTriple = Object->makeTriple();
  std::string TripleName = TheTriple.getTriple();
  std::string ArchName = "";
  std::string ErrorString;

  const Target *TheTarget =
      TargetRegistry::lookupTarget(ArchName, TheTriple, ErrorString);

  if (!TheTarget) {
    errs() << "Couldn't find target \"" << TheTriple.getTriple()
           << "\", failed with error: " << ErrorString << ".\n";
    return EXIT_FAILURE;
  }

  SubtargetFeatures Features = Object->getFeatures();

  std::unique_ptr<const MCRegisterInfo> RegisterInfo(
      TheTarget->createMCRegInfo(TripleName));
  if (!RegisterInfo) {
    errs() << "Failed to initialise RegisterInfo.\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*RegisterInfo, TripleName));
  if (!AsmInfo) {
    errs() << "Failed to initialise AsmInfo.\n";
    return EXIT_FAILURE;
  }

  std::string MCPU = "";
  std::unique_ptr<MCSubtargetInfo> SubtargetInfo(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, Features.getString()));
  if (!SubtargetInfo) {
    errs() << "Failed to initialise SubtargetInfo.\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII) {
    errs() << "Failed to initialise MII.\n";
    return EXIT_FAILURE;
  }

  MCObjectFileInfo MOFI;
  MCContext Context(AsmInfo.get(), RegisterInfo.get(), &MOFI);

  std::unique_ptr<const MCDisassembler> Disassembler(
      TheTarget->createMCDisassembler(*SubtargetInfo, Context));

  if (!Disassembler) {
    errs() << "No disassembler available for target.";
    return EXIT_FAILURE;
  }

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));

  std::unique_ptr<MCInstPrinter> Printer(
      TheTarget->createMCInstPrinter(TheTriple, AsmInfo->getAssemblerDialect(),
                                     *AsmInfo, *MII, *RegisterInfo));

  if (ArgDumpSymbols)
    printSymbols(Object);

  for (const SectionRef &Section : Object->sections()) {
    outs() << "Section [" << format_hex_no_prefix(Section.getAddress(), 2)
           << "] = ";
    StringRef SectionName;

    if (Section.getName(SectionName))
      outs() << "UNKNOWN.\n";
    else
      outs() << SectionName << "\n";

    StringRef SectionContents;
    if (Section.getContents(SectionContents)) {
      errs() << "Failed to retrieve section contents.\n";
      return EXIT_FAILURE;
    }

    MCInst Instruction;
    uint64_t InstructionSize;

    ArrayRef<uint8_t> SectionBytes((const uint8_t *)SectionContents.data(),
                                   Section.getSize());

    for (uint64_t Byte = 0; Byte < Section.getSize();) {
      bool BadInstruction = false;

      // Disassemble the instruction.
      if (Disassembler->getInstruction(
              Instruction, InstructionSize, SectionBytes.drop_front(Byte), 0,
              nulls(), outs()) != MCDisassembler::Success) {
        BadInstruction = true;
      }

      Byte += InstructionSize;

      if (BadInstruction)
        continue;

      // Skip instructions that do not affect the control flow.
      const auto &InstrDesc = MII->get(Instruction.getOpcode());
      if (!InstrDesc.mayAffectControlFlow(Instruction, *RegisterInfo))
        continue;

      // Skip instructions that do not operate on register operands.
      bool UsesRegisterOperand = false;
      for (const auto &Operand : Instruction) {
        if (Operand.isReg())
          UsesRegisterOperand = true;
      }

      if (!UsesRegisterOperand)
        continue;

      // Print the instruction address.
      outs() << "    "
             << format_hex(Section.getAddress() + Byte - InstructionSize, 2)
             << ": ";

      // Print the instruction bytes.
      for (uint64_t i = 0; i < InstructionSize; ++i) {
        outs() << format_hex_no_prefix(SectionBytes[Byte - InstructionSize + i],
                                       2)
               << " ";
      }

      // Print the instruction.
      outs() << " | " << MII->getName(Instruction.getOpcode()) << " ";
      Instruction.dump_pretty(outs(), Printer.get());

      outs() << "\n";
    }
  }

  return EXIT_SUCCESS;
}
