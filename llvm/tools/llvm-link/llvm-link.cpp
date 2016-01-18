//===- llvm-link.cpp - Low-level LLVM linker ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-link a.bc b.bc c.bc -o x.bc
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker/Linker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/FunctionInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Object/FunctionIndexObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>
using namespace llvm;

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
               cl::desc("<input bitcode files>"));

static cl::list<std::string> OverridingInputs(
    "override", cl::ZeroOrMore, cl::value_desc("filename"),
    cl::desc(
        "input bitcode file which can override previously defined symbol(s)"));

// Option to simulate function importing for testing. This enables using
// llvm-link to simulate ThinLTO backend processes.
static cl::list<std::string> Imports(
    "import", cl::ZeroOrMore, cl::value_desc("function:filename"),
    cl::desc("Pair of function name and filename, where function should be "
             "imported from bitcode in filename"));

// Option to support testing of function importing. The function index
// must be specified in the case were we request imports via the -import
// option, as well as when compiling any module with functions that may be
// exported (imported by a different llvm-link -import invocation), to ensure
// consistent promotion and renaming of locals.
static cl::opt<std::string> FunctionIndex("functionindex",
                                          cl::desc("Function index filename"),
                                          cl::init(""),
                                          cl::value_desc("filename"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"), cl::init("-"),
               cl::value_desc("filename"));

static cl::opt<bool>
Internalize("internalize", cl::desc("Internalize linked symbols"));

static cl::opt<bool>
OnlyNeeded("only-needed", cl::desc("Link only needed symbols"));

static cl::opt<bool>
Force("f", cl::desc("Enable binary output on terminals"));

static cl::opt<bool>
OutputAssembly("S",
         cl::desc("Write output as LLVM assembly"), cl::Hidden);

static cl::opt<bool>
Verbose("v", cl::desc("Print information about actions taken"));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print assembly as linked"), cl::Hidden);

static cl::opt<bool>
SuppressWarnings("suppress-warnings", cl::desc("Suppress all linking warnings"),
                 cl::init(false));

static cl::opt<bool> PreserveBitcodeUseListOrder(
    "preserve-bc-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM bitcode."),
    cl::init(true), cl::Hidden);

static cl::opt<bool> PreserveAssemblyUseListOrder(
    "preserve-ll-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM assembly."),
    cl::init(false), cl::Hidden);

// Read the specified bitcode file in and return it. This routine searches the
// link path for the specified file to try to find it...
//
static std::unique_ptr<Module>
loadFile(const char *argv0, const std::string &FN, LLVMContext &Context) {
  SMDiagnostic Err;
  if (Verbose) errs() << "Loading '" << FN << "'\n";
  std::unique_ptr<Module> Result = getLazyIRFileModule(FN, Err, Context);
  if (!Result)
    Err.print(argv0, errs());

  Result->materializeMetadata();
  UpgradeDebugInfo(*Result);

  return Result;
}

static void diagnosticHandler(const DiagnosticInfo &DI) {
  unsigned Severity = DI.getSeverity();
  switch (Severity) {
  case DS_Error:
    errs() << "ERROR: ";
    break;
  case DS_Warning:
    if (SuppressWarnings)
      return;
    errs() << "WARNING: ";
    break;
  case DS_Remark:
  case DS_Note:
    llvm_unreachable("Only expecting warnings and errors");
  }

  DiagnosticPrinterRawOStream DP(errs());
  DI.print(DP);
  errs() << '\n';
}

/// Load a function index if requested by the -functionindex option.
static ErrorOr<std::unique_ptr<FunctionInfoIndex>>
loadIndex(LLVMContext &Context, const Module *ExportingModule = nullptr) {
  assert(!FunctionIndex.empty());
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(FunctionIndex);
  std::error_code EC = FileOrErr.getError();
  if (EC)
    return EC;
  MemoryBufferRef BufferRef = (FileOrErr.get())->getMemBufferRef();
  ErrorOr<std::unique_ptr<object::FunctionIndexObjectFile>> ObjOrErr =
      object::FunctionIndexObjectFile::create(BufferRef, Context,
                                              ExportingModule);
  EC = ObjOrErr.getError();
  if (EC)
    return EC;

  object::FunctionIndexObjectFile &Obj = **ObjOrErr;
  return Obj.takeIndex();
}

/// Import any functions requested via the -import option.
static bool importFunctions(const char *argv0, LLVMContext &Context,
                            Linker &L) {
  for (const auto &Import : Imports) {
    // Identify the requested function and its bitcode source file.
    size_t Idx = Import.find(':');
    if (Idx == std::string::npos) {
      errs() << "Import parameter bad format: " << Import << "\n";
      return false;
    }
    std::string FunctionName = Import.substr(0, Idx);
    std::string FileName = Import.substr(Idx + 1, std::string::npos);

    // Load the specified source module.
    std::unique_ptr<Module> M = loadFile(argv0, FileName, Context);
    if (!M.get()) {
      errs() << argv0 << ": error loading file '" << FileName << "'\n";
      return false;
    }

    if (verifyModule(*M, &errs())) {
      errs() << argv0 << ": " << FileName
             << ": error: input module is broken!\n";
      return false;
    }

    Function *F = M->getFunction(FunctionName);
    if (!F) {
      errs() << "Ignoring import request for non-existent function "
             << FunctionName << " from " << FileName << "\n";
      continue;
    }
    // We cannot import weak_any functions without possibly affecting the
    // order they are seen and selected by the linker, changing program
    // semantics.
    if (F->hasWeakAnyLinkage()) {
      errs() << "Ignoring import request for weak-any function " << FunctionName
             << " from " << FileName << "\n";
      continue;
    }

    if (Verbose)
      errs() << "Importing " << FunctionName << " from " << FileName << "\n";

    std::unique_ptr<FunctionInfoIndex> Index;
    if (!FunctionIndex.empty()) {
      ErrorOr<std::unique_ptr<FunctionInfoIndex>> IndexOrErr =
          loadIndex(Context);
      std::error_code EC = IndexOrErr.getError();
      if (EC) {
        errs() << EC.message() << '\n';
        return false;
      }
      Index = std::move(IndexOrErr.get());
    }

    // Link in the specified function.
    if (L.linkInModule(M.get(), Linker::Flags::None, Index.get(), F))
      return false;
  }
  return true;
}

static bool linkFiles(const char *argv0, LLVMContext &Context, Linker &L,
                      const cl::list<std::string> &Files,
                      unsigned Flags) {
  // Filter out flags that don't apply to the first file we load.
  unsigned ApplicableFlags = Flags & Linker::Flags::OverrideFromSrc;
  for (const auto &File : Files) {
    std::unique_ptr<Module> M = loadFile(argv0, File, Context);
    if (!M.get()) {
      errs() << argv0 << ": error loading file '" << File << "'\n";
      return false;
    }

    if (verifyModule(*M, &errs())) {
      errs() << argv0 << ": " << File << ": error: input module is broken!\n";
      return false;
    }

    // If a function index is supplied, load it so linkInModule can treat
    // local functions/variables as exported and promote if necessary.
    std::unique_ptr<FunctionInfoIndex> Index;
    if (!FunctionIndex.empty()) {
      ErrorOr<std::unique_ptr<FunctionInfoIndex>> IndexOrErr =
          loadIndex(Context, &*M);
      std::error_code EC = IndexOrErr.getError();
      if (EC) {
        errs() << EC.message() << '\n';
        return false;
      }
      Index = std::move(IndexOrErr.get());
    }

    if (Verbose)
      errs() << "Linking in '" << File << "'\n";

    if (L.linkInModule(M.get(), ApplicableFlags, Index.get()))
      return false;
    // All linker flags apply to linking of subsequent files.
    ApplicableFlags = Flags;
  }

  return true;
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm linker\n");

  auto Composite = make_unique<Module>("llvm-link", Context);
  Linker L(Composite.get(), diagnosticHandler);

  unsigned Flags = Linker::Flags::None;
  if (Internalize)
    Flags |= Linker::Flags::InternalizeLinkedSymbols;
  if (OnlyNeeded)
    Flags |= Linker::Flags::LinkOnlyNeeded;

  // First add all the regular input files
  if (!linkFiles(argv[0], Context, L, InputFilenames, Flags))
    return 1;

  // Next the -override ones.
  if (!linkFiles(argv[0], Context, L, OverridingInputs,
                 Flags | Linker::Flags::OverrideFromSrc))
    return 1;

  // Import any functions requested via -import
  if (!importFunctions(argv[0], Context, L))
    return 1;

  if (DumpAsm) errs() << "Here's the assembly:\n" << *Composite;

  std::error_code EC;
  tool_output_file Out(OutputFilename, EC, sys::fs::F_None);
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }

  if (verifyModule(*Composite, &errs())) {
    errs() << argv[0] << ": error: linked module is broken!\n";
    return 1;
  }

  if (Verbose) errs() << "Writing bitcode...\n";
  if (OutputAssembly) {
    Composite->print(Out.os(), nullptr, PreserveAssemblyUseListOrder);
  } else if (Force || !CheckBitcodeOutputToConsole(Out.os(), true))
    WriteBitcodeToFile(Composite.get(), Out.os(), PreserveBitcodeUseListOrder);

  // Declare success.
  Out.keep();

  return 0;
}
