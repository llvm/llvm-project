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

#include "lib/FileAnalysis.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SpecialCaseList.h"

#include <cstdlib>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::cfi_verify;

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::Required);
cl::opt<std::string> BlacklistFilename(cl::Positional,
                                       cl::desc("[blacklist file]"),
                                       cl::init("-"));

ExitOnError ExitOnErr;

void printIndirectCFInstructions(FileAnalysis &Analysis,
                                 const SpecialCaseList *SpecialCaseList) {
  uint64_t ExpectedProtected = 0;
  uint64_t UnexpectedProtected = 0;
  uint64_t ExpectedUnprotected = 0;
  uint64_t UnexpectedUnprotected = 0;

  symbolize::LLVMSymbolizer &Symbolizer = Analysis.getSymbolizer();

  for (uint64_t Address : Analysis.getIndirectInstructions()) {
    const auto &InstrMeta = Analysis.getInstructionOrDie(Address);

    bool CFIProtected = Analysis.isIndirectInstructionCFIProtected(Address);

    if (CFIProtected)
      outs() << "P ";
    else
      outs() << "U ";

    outs() << format_hex(Address, 2) << " | "
           << Analysis.getMCInstrInfo()->getName(
                  InstrMeta.Instruction.getOpcode())
           << " \n";

    if (IgnoreDWARFFlag) {
      if (CFIProtected)
        ExpectedProtected++;
      else
        UnexpectedUnprotected++;
      continue;
    }

    auto InliningInfo = Symbolizer.symbolizeInlinedCode(InputFilename, Address);
    if (!InliningInfo || InliningInfo->getNumberOfFrames() == 0) {
      errs() << "Failed to symbolise " << format_hex(Address, 2)
             << " with line tables from " << InputFilename << "\n";
      exit(EXIT_FAILURE);
    }

    const auto &LineInfo =
        InliningInfo->getFrame(InliningInfo->getNumberOfFrames() - 1);

    // Print the inlining symbolisation of this instruction.
    for (uint32_t i = 0; i < InliningInfo->getNumberOfFrames(); ++i) {
      const auto &Line = InliningInfo->getFrame(i);
      outs() << "  " << format_hex(Address, 2) << " = " << Line.FileName << ":"
             << Line.Line << ":" << Line.Column << " (" << Line.FunctionName
             << ")\n";
    }

    if (!SpecialCaseList) {
      if (CFIProtected)
        ExpectedProtected++;
      else
        UnexpectedUnprotected++;
      continue;
    }

    bool MatchesBlacklistRule = false;
    if (SpecialCaseList->inSection("cfi-icall", "src", LineInfo.FileName) ||
        SpecialCaseList->inSection("cfi-vcall", "src", LineInfo.FileName)) {
      outs() << "BLACKLIST MATCH, 'src'\n";
      MatchesBlacklistRule = true;
    }

    if (SpecialCaseList->inSection("cfi-icall", "fun", LineInfo.FunctionName) ||
        SpecialCaseList->inSection("cfi-vcall", "fun", LineInfo.FunctionName)) {
      outs() << "BLACKLIST MATCH, 'fun'\n";
      MatchesBlacklistRule = true;
    }

    if (MatchesBlacklistRule) {
      if (CFIProtected) {
        UnexpectedProtected++;
        outs() << "====> Unexpected Protected\n";
      } else {
        ExpectedUnprotected++;
        outs() << "====> Expected Unprotected\n";
      }
    } else {
      if (CFIProtected) {
        ExpectedProtected++;
        outs() << "====> Expected Protected\n";
      } else {
        UnexpectedUnprotected++;
        outs() << "====> Unexpected Unprotected\n";
      }
    }
  }

  uint64_t IndirectCFInstructions = ExpectedProtected + UnexpectedProtected +
                                    ExpectedUnprotected + UnexpectedUnprotected;

  if (IndirectCFInstructions == 0)
    outs() << "No indirect CF instructions found.\n";

  outs() << formatv("Expected Protected: {0} ({1:P})\n"
                    "Unexpected Protected: {2} ({3:P})\n"
                    "Expected Unprotected: {4} ({5:P})\n"
                    "Unexpected Unprotected (BAD): {6} ({7:P})\n",
                    ExpectedProtected,
                    ((double)ExpectedProtected) / IndirectCFInstructions,
                    UnexpectedProtected,
                    ((double)UnexpectedProtected) / IndirectCFInstructions,
                    ExpectedUnprotected,
                    ((double)ExpectedUnprotected) / IndirectCFInstructions,
                    UnexpectedUnprotected,
                    ((double)UnexpectedUnprotected) / IndirectCFInstructions);
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "Identifies whether Control Flow Integrity protects all indirect control "
      "flow instructions in the provided object file, DSO or binary.\nNote: "
      "Anything statically linked into the provided file *must* be compiled "
      "with '-g'. This can be relaxed through the '--ignore-dwarf' flag.");

  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllDisassemblers();

  std::unique_ptr<SpecialCaseList> SpecialCaseList;
  if (BlacklistFilename != "-") {
    std::string Error;
    SpecialCaseList = SpecialCaseList::create({BlacklistFilename}, Error);
    if (!SpecialCaseList) {
      errs() << "Failed to get blacklist: " << Error << "\n";
      exit(EXIT_FAILURE);
    }
  }

  FileAnalysis Analysis = ExitOnErr(FileAnalysis::Create(InputFilename));
  printIndirectCFInstructions(Analysis, SpecialCaseList.get());

  return EXIT_SUCCESS;
}
