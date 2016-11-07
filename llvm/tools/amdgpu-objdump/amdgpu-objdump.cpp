//===-- amdgpu-objdump.cpp - Object file dumping for HSA code object ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program dumps content of HSA code object file.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Disassembler/CodeObjectDisassembler.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace object;

static cl::opt<std::string>
InputFilename("<object file>",
              cl::desc("HSA code object filename"),
              cl::Positional,
              cl::ValueRequired);

LLVM_ATTRIBUTE_NORETURN static void report_error(StringRef ToolName,
                                                 StringRef File,
                                                 llvm::Error E) {
  assert(E);
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS, "");
  OS.flush();
  errs() << ToolName << ": '" << File << "': " << Buf;
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN static void report_error(StringRef ToolName,
                                                 StringRef File,
                                                 std::error_code EC) {
  assert(EC);
  errs() << ToolName << ": '" << File << "': " << EC.message() << ".\n";
  exit(1);
}

int main(int argc, char *argv[]) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize assembly printer/parser.
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();

  cl::ParseCommandLineOptions(argc, argv, "HSA code object dumper\n");

  StringRef ToolName = argv[0];
  StringRef TripleName = Triple::normalize("amdgcn-unknown-amdhsa");

  auto BinaryOr = createBinary(InputFilename);
  if (!BinaryOr)
    report_error(ToolName, InputFilename, BinaryOr.takeError());
  const auto *Binary = BinaryOr->getBinary();

  // setup context
  const auto &TheTarget = getTheGCNTarget();

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget.createMCRegInfo(TripleName));
  if (!MRI)
    report_fatal_error("error: no register info");

  std::unique_ptr<MCAsmInfo> AsmInfo(TheTarget.createMCAsmInfo(*MRI, TripleName));
  if (!AsmInfo)
    report_fatal_error("error: no assembly info");

  std::unique_ptr<MCInstrInfo> MII(TheTarget.createMCInstrInfo());
  if (!MII)
    report_fatal_error("error: no instruction info");
  
  MCObjectFileInfo MOFI;
  MCContext Ctx(AsmInfo.get(), MRI.get(), &MOFI);
  MOFI.InitMCObjectFileInfo(Triple(TripleName), false, CodeModel::Default, Ctx);

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  MCInstPrinter *IP(TheTarget.createMCInstPrinter(Triple(TripleName),
                                                  AsmPrinterVariant,
                                                  *AsmInfo, *MII, *MRI));
  if (!IP)
    report_fatal_error("error: no instruction printer");

  auto FOut = make_unique<formatted_raw_ostream>(outs());
  std::unique_ptr<MCStreamer> MCS(
    TheTarget.createAsmStreamer(Ctx, std::move(FOut), true, false, IP,
                                nullptr, nullptr, false));

  CodeObjectDisassembler CODisasm(&Ctx, TripleName, IP, MCS->getTargetStreamer());

  auto EC = CODisasm.Disassemble(Binary->getMemoryBufferRef(), errs());
  if (EC)
    report_error(ToolName, InputFilename, EC);

  return EXIT_SUCCESS;
}