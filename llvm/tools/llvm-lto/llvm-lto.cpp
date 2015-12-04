//===-- llvm-lto: a simple command-line program to link modules with LTO --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program takes in a list of bitcode files, links them, performs link-time
// optimization, and outputs an object file.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Object/FunctionIndexObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <list>

using namespace llvm;

static cl::opt<char>
OptLevel("O",
         cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                  "(default = '-O2')"),
         cl::Prefix,
         cl::ZeroOrMore,
         cl::init('2'));

static cl::opt<bool> DisableVerify(
    "disable-verify", cl::init(false),
    cl::desc("Do not run the verifier during the optimization pipeline"));

static cl::opt<bool>
DisableInline("disable-inlining", cl::init(false),
  cl::desc("Do not run the inliner pass"));

static cl::opt<bool>
DisableGVNLoadPRE("disable-gvn-loadpre", cl::init(false),
  cl::desc("Do not run the GVN load PRE pass"));

static cl::opt<bool>
DisableLTOVectorization("disable-lto-vectorization", cl::init(false),
  cl::desc("Do not run loop or slp vectorization during LTO"));

static cl::opt<bool>
UseDiagnosticHandler("use-diagnostic-handler", cl::init(false),
  cl::desc("Use a diagnostic handler to test the handler interface"));

static cl::opt<bool>
    ThinLTO("thinlto", cl::init(false),
            cl::desc("Only write combined global index for ThinLTO backends"));

static cl::opt<bool>
SaveModuleFile("save-merged-module", cl::init(false),
               cl::desc("Write merged LTO module to file before CodeGen"));

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
  cl::desc("<input bitcode files>"));

static cl::opt<std::string>
OutputFilename("o", cl::init(""),
  cl::desc("Override output filename"),
  cl::value_desc("filename"));

static cl::list<std::string>
ExportedSymbols("exported-symbol",
  cl::desc("Symbol to export from the resulting object file"),
  cl::ZeroOrMore);

static cl::list<std::string>
DSOSymbols("dso-symbol",
  cl::desc("Symbol to put in the symtab in the resulting dso"),
  cl::ZeroOrMore);

static cl::opt<bool> ListSymbolsOnly(
    "list-symbols-only", cl::init(false),
    cl::desc("Instead of running LTO, list the symbols in each IR file"));

static cl::opt<bool> SetMergedModule(
    "set-merged-module", cl::init(false),
    cl::desc("Use the first input module as the merged module"));

static cl::opt<unsigned> Parallelism("j", cl::Prefix, cl::init(1),
                                     cl::desc("Number of backend threads"));

namespace {
struct ModuleInfo {
  std::vector<bool> CanBeHidden;
};
}

static void handleDiagnostics(lto_codegen_diagnostic_severity_t Severity,
                              const char *Msg, void *) {
  errs() << "llvm-lto: ";
  switch (Severity) {
  case LTO_DS_NOTE:
    errs() << "note: ";
    break;
  case LTO_DS_REMARK:
    errs() << "remark: ";
    break;
  case LTO_DS_ERROR:
    errs() << "error: ";
    break;
  case LTO_DS_WARNING:
    errs() << "warning: ";
    break;
  }
  errs() << Msg << "\n";
}

static void diagnosticHandler(const DiagnosticInfo &DI) {
  raw_ostream &OS = errs();
  OS << "llvm-lto: ";
  switch (DI.getSeverity()) {
  case DS_Error:
    OS << "error: ";
    break;
  case DS_Warning:
    OS << "warning: ";
    break;
  case DS_Remark:
    OS << "remark: ";
    break;
  case DS_Note:
    OS << "note: ";
    break;
  }

  DiagnosticPrinterRawOStream DP(OS);
  DI.print(DP);
  OS << '\n';

  if (DI.getSeverity() == DS_Error)
    exit(1);
}

static void error(const Twine &Msg) {
  errs() << "llvm-lto: " << Msg << '\n';
  exit(1);
}

static void error(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

template <typename T>
static void error(const ErrorOr<T> &V, const Twine &Prefix) {
  error(V.getError(), Prefix);
}

static std::unique_ptr<LTOModule>
getLocalLTOModule(StringRef Path, std::unique_ptr<MemoryBuffer> &Buffer,
                  const TargetOptions &Options) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(Path);
  error(BufferOrErr, "error loading file '" + Path + "'");
  Buffer = std::move(BufferOrErr.get());
  std::string Error;
  std::unique_ptr<LTOModule> Ret(LTOModule::createInLocalContext(
      Buffer->getBufferStart(), Buffer->getBufferSize(), Options, Error, Path));
  if (!Error.empty())
    error("error loading file '" + Path + "' " + Error);
  return Ret;
}

/// \brief List symbols in each IR file.
///
/// The main point here is to provide lit-testable coverage for the LTOModule
/// functionality that's exposed by the C API to list symbols.  Moreover, this
/// provides testing coverage for modules that have been created in their own
/// contexts.
static void listSymbols(const TargetOptions &Options) {
  for (auto &Filename : InputFilenames) {
    std::unique_ptr<MemoryBuffer> Buffer;
    std::unique_ptr<LTOModule> Module =
        getLocalLTOModule(Filename, Buffer, Options);

    // List the symbols.
    outs() << Filename << ":\n";
    for (int I = 0, E = Module->getSymbolCount(); I != E; ++I)
      outs() << Module->getSymbolName(I) << "\n";
  }
}

/// Create a combined index file from the input IR files and write it.
///
/// This is meant to enable testing of ThinLTO combined index generation,
/// currently available via the gold plugin via -thinlto.
static void createCombinedFunctionIndex() {
  FunctionInfoIndex CombinedIndex;
  uint64_t NextModuleId = 0;
  for (auto &Filename : InputFilenames) {
    ErrorOr<std::unique_ptr<FunctionInfoIndex>> IndexOrErr =
        llvm::getFunctionIndexForFile(Filename, diagnosticHandler);
    error(IndexOrErr, "error loading file '" + Filename + "'");
    std::unique_ptr<FunctionInfoIndex> Index = std::move(IndexOrErr.get());
    // Skip files without a function summary.
    if (!Index)
      continue;
    CombinedIndex.mergeFrom(std::move(Index), ++NextModuleId);
  }
  std::error_code EC;
  assert(!OutputFilename.empty());
  raw_fd_ostream OS(OutputFilename + ".thinlto.bc", EC,
                    sys::fs::OpenFlags::F_None);
  error(EC, "error opening the file '" + OutputFilename + ".thinlto.bc'");
  WriteFunctionSummaryToFile(CombinedIndex, OS);
  OS.close();
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm LTO linker\n");

  if (OptLevel < '0' || OptLevel > '3')
    error("optimization level must be between 0 and 3");

  // Initialize the configured targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // set up the TargetOptions for the machine
  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();

  if (ListSymbolsOnly) {
    listSymbols(Options);
    return 0;
  }

  if (ThinLTO) {
    createCombinedFunctionIndex();
    return 0;
  }

  unsigned BaseArg = 0;

  LTOCodeGenerator CodeGen(getGlobalContext());

  if (UseDiagnosticHandler)
    CodeGen.setDiagnosticHandler(handleDiagnostics, nullptr);

  CodeGen.setCodePICModel(RelocModel);

  CodeGen.setDebugInfo(LTO_DEBUG_MODEL_DWARF);
  CodeGen.setTargetOptions(Options);

  llvm::StringSet<llvm::MallocAllocator> DSOSymbolsSet;
  for (unsigned i = 0; i < DSOSymbols.size(); ++i)
    DSOSymbolsSet.insert(DSOSymbols[i]);

  std::vector<std::string> KeptDSOSyms;

  for (unsigned i = BaseArg; i < InputFilenames.size(); ++i) {
    std::string error;
    std::unique_ptr<LTOModule> Module(
        LTOModule::createFromFile(InputFilenames[i].c_str(), Options, error));
    if (!error.empty()) {
      errs() << argv[0] << ": error loading file '" << InputFilenames[i]
             << "': " << error << "\n";
      return 1;
    }

    unsigned NumSyms = Module->getSymbolCount();
    for (unsigned I = 0; I < NumSyms; ++I) {
      StringRef Name = Module->getSymbolName(I);
      if (!DSOSymbolsSet.count(Name))
        continue;
      lto_symbol_attributes Attrs = Module->getSymbolAttributes(I);
      unsigned Scope = Attrs & LTO_SYMBOL_SCOPE_MASK;
      if (Scope != LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN)
        KeptDSOSyms.push_back(Name);
    }

    // We use the first input module as the destination module when
    // SetMergedModule is true.
    if (SetMergedModule && i == BaseArg) {
      // Transfer ownership to the code generator.
      CodeGen.setModule(std::move(Module));
    } else if (!CodeGen.addModule(Module.get())) {
      // Print a message here so that we know addModule() did not abort.
      errs() << argv[0] << ": error adding file '" << InputFilenames[i] << "'\n";
      return 1;
    }
  }

  // Add all the exported symbols to the table of symbols to preserve.
  for (unsigned i = 0; i < ExportedSymbols.size(); ++i)
    CodeGen.addMustPreserveSymbol(ExportedSymbols[i].c_str());

  // Add all the dso symbols to the table of symbols to expose.
  for (unsigned i = 0; i < KeptDSOSyms.size(); ++i)
    CodeGen.addMustPreserveSymbol(KeptDSOSyms[i].c_str());

  // Set cpu and attrs strings for the default target/subtarget.
  CodeGen.setCpu(MCPU.c_str());

  CodeGen.setOptLevel(OptLevel - '0');

  std::string attrs;
  for (unsigned i = 0; i < MAttrs.size(); ++i) {
    if (i > 0)
      attrs.append(",");
    attrs.append(MAttrs[i]);
  }

  if (!attrs.empty())
    CodeGen.setAttr(attrs.c_str());

  if (FileType.getNumOccurrences())
    CodeGen.setFileType(FileType);

  if (!OutputFilename.empty()) {
    if (!CodeGen.optimize(DisableVerify, DisableInline, DisableGVNLoadPRE,
                          DisableLTOVectorization)) {
      // Diagnostic messages should have been printed by the handler.
      errs() << argv[0] << ": error optimizing the code\n";
      return 1;
    }

    if (SaveModuleFile) {
      std::string ModuleFilename = OutputFilename;
      ModuleFilename += ".merged.bc";
      std::string ErrMsg;

      if (!CodeGen.writeMergedModules(ModuleFilename.c_str())) {
        errs() << argv[0] << ": writing merged module failed.\n";
        return 1;
      }
    }

    std::list<tool_output_file> OSs;
    std::vector<raw_pwrite_stream *> OSPtrs;
    for (unsigned I = 0; I != Parallelism; ++I) {
      std::string PartFilename = OutputFilename;
      if (Parallelism != 1)
        PartFilename += "." + utostr(I);
      std::error_code EC;
      OSs.emplace_back(PartFilename, EC, sys::fs::F_None);
      if (EC) {
        errs() << argv[0] << ": error opening the file '" << PartFilename
               << "': " << EC.message() << "\n";
        return 1;
      }
      OSPtrs.push_back(&OSs.back().os());
    }

    if (!CodeGen.compileOptimized(OSPtrs)) {
      // Diagnostic messages should have been printed by the handler.
      errs() << argv[0] << ": error compiling the code\n";
      return 1;
    }

    for (tool_output_file &OS : OSs)
      OS.keep();
  } else {
    if (Parallelism != 1) {
      errs() << argv[0] << ": -j must be specified together with -o\n";
      return 1;
    }

    if (SaveModuleFile) {
      errs() << argv[0] << ": -save-merged-module must be specified with -o\n";
      return 1;
    }

    const char *OutputName = nullptr;
    if (!CodeGen.compile_to_file(&OutputName, DisableVerify, DisableInline,
                                 DisableGVNLoadPRE, DisableLTOVectorization)) {
      // Diagnostic messages should have been printed by the handler.
      errs() << argv[0] << ": error compiling the code\n";
      return 1;
    }

    outs() << "Wrote native object file '" << OutputName << "'\n";
  }

  return 0;
}
