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

#include <cstdlib>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::cfi_verify;

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::Required);

ExitOnError ExitOnErr;

void printIndirectCFInstructions(const FileAnalysis &Verifier) {
  for (uint64_t Address : Verifier.getIndirectInstructions()) {
    const auto &InstrMeta = Verifier.getInstructionOrDie(Address);
    outs() << format_hex(Address, 2) << " |"
           << Verifier.getMCInstrInfo()->getName(
                  InstrMeta.Instruction.getOpcode())
           << " ";
    InstrMeta.Instruction.print(outs());
    outs() << "\n";
    outs() << "  Protected? "
           << Verifier.isIndirectInstructionCFIProtected(Address) << "\n";
  }
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllDisassemblers();

  FileAnalysis Verifier = ExitOnErr(FileAnalysis::Create(InputFilename));
  printIndirectCFInstructions(Verifier);

  return EXIT_SUCCESS;
}
