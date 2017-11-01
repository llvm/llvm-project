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

#include <cstdlib>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::cfi_verify;

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::Required);

ExitOnError ExitOnErr;

void printIndirectCFInstructions(FileAnalysis &Analysis) {
  uint64_t ProtectedCount = 0;
  uint64_t UnprotectedCount = 0;

  for (uint64_t Address : Analysis.getIndirectInstructions()) {
    const auto &InstrMeta = Analysis.getInstructionOrDie(Address);

    if (Analysis.isIndirectInstructionCFIProtected(Address)) {
      outs() << "P ";
      ProtectedCount++;
    } else {
      outs() << "U ";
      UnprotectedCount++;
    }

    outs() << format_hex(Address, 2) << " | "
           << Analysis.getMCInstrInfo()->getName(
                  InstrMeta.Instruction.getOpcode())
           << " ";
    outs() << "\n";

    if (Analysis.hasLineTableInfo()) {
      for (const auto &LineKV : Analysis.getLineInfoForAddressRange(Address)) {
        outs() << "  " << format_hex(LineKV.first, 2) << " = "
               << LineKV.second.FileName << ":" << LineKV.second.Line << ":"
               << LineKV.second.Column << " (" << LineKV.second.FunctionName
               << ")\n";
      }
    }
  }

  if (ProtectedCount || UnprotectedCount)
    outs() << formatv(
        "Unprotected: {0} ({1:P}), Protected: {2} ({3:P})\n", UnprotectedCount,
        (((double)UnprotectedCount) / (UnprotectedCount + ProtectedCount)),
        ProtectedCount,
        (((double)ProtectedCount) / (UnprotectedCount + ProtectedCount)));
  else
    outs() << "No indirect CF instructions found.\n";
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

  FileAnalysis Analysis = ExitOnErr(FileAnalysis::Create(InputFilename));
  printIndirectCFInstructions(Analysis);

  return EXIT_SUCCESS;
}
