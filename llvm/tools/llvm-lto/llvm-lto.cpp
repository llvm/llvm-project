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
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<char>
OptLevel("O",
         cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                  "(default = '-O2')"),
         cl::Prefix,
         cl::ZeroOrMore,
         cl::init('2'));

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

namespace {
struct ModuleInfo {
  std::vector<bool> CanBeHidden;
};
}

static void handleDiagnostics(lto_codegen_diagnostic_severity_t Severity,
                              const char *Msg, void *) {
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

static std::unique_ptr<LTOModule>
getLocalLTOModule(StringRef Path, std::unique_ptr<MemoryBuffer> &Buffer,
                  const TargetOptions &Options, std::string &Error) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(Path);
  if (std::error_code EC = BufferOrErr.getError()) {
    Error = EC.message();
    return nullptr;
  }
  Buffer = std::move(BufferOrErr.get());
  return std::unique_ptr<LTOModule>(LTOModule::createInLocalContext(
      Buffer->getBufferStart(), Buffer->getBufferSize(), Options, Error, Path));
}

/// \brief List symbols in each IR file.
///
/// The main point here is to provide lit-testable coverage for the LTOModule
/// functionality that's exposed by the C API to list symbols.  Moreover, this
/// provides testing coverage for modules that have been created in their own
/// contexts.
static int listSymbols(StringRef Command, const TargetOptions &Options) {
  for (auto &Filename : InputFilenames) {
    std::string Error;
    std::unique_ptr<MemoryBuffer> Buffer;
    std::unique_ptr<LTOModule> Module =
        getLocalLTOModule(Filename, Buffer, Options, Error);
    if (!Module) {
      errs() << Command << ": error loading file '" << Filename
             << "': " << Error << "\n";
      return 1;
    }

    // List the symbols.
    outs() << Filename << ":\n";
    for (int I = 0, E = Module->getSymbolCount(); I != E; ++I)
      outs() << Module->getSymbolName(I) << "\n";
  }
  return 0;
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm LTO linker\n");

  if (OptLevel < '0' || OptLevel > '3') {
    errs() << argv[0] << ": optimization level must be between 0 and 3\n";
    return 1;
  }

  // Initialize the configured targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // set up the TargetOptions for the machine
  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();

  if (ListSymbolsOnly)
    return listSymbols(argv[0], Options);

  unsigned BaseArg = 0;

  LTOCodeGenerator CodeGen;

  if (UseDiagnosticHandler)
    CodeGen.setDiagnosticHandler(handleDiagnostics, nullptr);

  switch (RelocModel) {
  case Reloc::Static:
    CodeGen.setCodePICModel(LTO_CODEGEN_PIC_MODEL_STATIC);
    break;
  case Reloc::PIC_:
    CodeGen.setCodePICModel(LTO_CODEGEN_PIC_MODEL_DYNAMIC);
    break;
  case Reloc::DynamicNoPIC:
    CodeGen.setCodePICModel(LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC);
    break;
  default:
    CodeGen.setCodePICModel(LTO_CODEGEN_PIC_MODEL_DEFAULT);
  }

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

    LTOModule *LTOMod = Module.get();

    // We use the first input module as the destination module when
    // SetMergedModule is true.
    if (SetMergedModule && i == BaseArg) {
      // Transfer ownership to the code generator.
      CodeGen.setModule(Module.release());
    } else if (!CodeGen.addModule(Module.get())) {
      // Print a message here so that we know addModule() did not abort.
      errs() << argv[0] << ": error adding file '" << InputFilenames[i] << "'\n";
      return 1;
    }

    unsigned NumSyms = LTOMod->getSymbolCount();
    for (unsigned I = 0; I < NumSyms; ++I) {
      StringRef Name = LTOMod->getSymbolName(I);
      if (!DSOSymbolsSet.count(Name))
        continue;
      lto_symbol_attributes Attrs = LTOMod->getSymbolAttributes(I);
      unsigned Scope = Attrs & LTO_SYMBOL_SCOPE_MASK;
      if (Scope != LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN)
        KeptDSOSyms.push_back(Name);
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

  if (!OutputFilename.empty()) {
    std::string ErrorInfo;
    std::unique_ptr<MemoryBuffer> Code = CodeGen.compile(
        DisableInline, DisableGVNLoadPRE, DisableLTOVectorization, ErrorInfo);
    if (!Code) {
      errs() << argv[0]
             << ": error compiling the code: " << ErrorInfo << "\n";
      return 1;
    }

    std::error_code EC;
    raw_fd_ostream FileStream(OutputFilename, EC, sys::fs::F_None);
    if (EC) {
      errs() << argv[0] << ": error opening the file '" << OutputFilename
             << "': " << EC.message() << "\n";
      return 1;
    }

    FileStream.write(Code->getBufferStart(), Code->getBufferSize());
  } else {
    std::string ErrorInfo;
    const char *OutputName = nullptr;
    if (!CodeGen.compile_to_file(&OutputName, DisableInline,
                                 DisableGVNLoadPRE, DisableLTOVectorization,
                                 ErrorInfo)) {
      errs() << argv[0]
             << ": error compiling the code: " << ErrorInfo
             << "\n";
      return 1;
    }

    outs() << "Wrote native object file '" << OutputName << "'\n";
  }

  return 0;
}
