//===- FunctionImport.cpp - ThinLTO Summary-based Function Import ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Function import based on summaries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/FunctionImport.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/FunctionIndexObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
using namespace llvm;

#define DEBUG_TYPE "function-import"

// Load lazily a module from \p FileName in \p Context.
static std::unique_ptr<Module> loadFile(const std::string &FileName,
                                        LLVMContext &Context) {
  SMDiagnostic Err;
  DEBUG(dbgs() << "Loading '" << FileName << "'\n");
  std::unique_ptr<Module> Result = getLazyIRFileModule(FileName, Err, Context);
  if (!Result) {
    Err.print("function-import", errs());
    return nullptr;
  }

  Result->materializeMetadata();
  UpgradeDebugInfo(*Result);

  return Result;
}

// Get a Module for \p FileName from the cache, or load it lazily.
Module &FunctionImporter::getOrLoadModule(StringRef FileName) {
  auto &Module = ModuleMap[FileName];
  if (!Module)
    Module = loadFile(FileName, Context);
  return *Module;
}

// Automatically import functions in Module \p M based on the summaries index.
//
// The current implementation imports every called functions that exists in the
// summaries index.
bool FunctionImporter::importFunctions(Module &M) {
  assert(&Context == &M.getContext());

  bool Changed = false;

  /// First step is collecting the called functions and the one defined in this
  /// module.
  StringSet<> CalledFunctions;
  for (auto &F : M) {
    if (F.isDeclaration() || F.hasFnAttribute(Attribute::OptimizeNone))
      continue;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (isa<CallInst>(I)) {
          DEBUG(dbgs() << "Found a call: '" << I << "'\n");
          auto CalledFunction = cast<CallInst>(I).getCalledFunction();
          if (CalledFunction && CalledFunction->hasName() &&
              CalledFunction->isDeclaration())
            CalledFunctions.insert(CalledFunction->getName());
        }
      }
    }
  }

  /// Second step: for every call to an external function, try to import it.

  // Linker that will be used for importing function
  Linker L(&M, DiagnosticHandler);

  /// Insert initial called function set in a worklist, so that we can add
  /// transively called functions when importing.
  SmallVector<StringRef, 64> Worklist;
  for (auto &CalledFunction : CalledFunctions)
    Worklist.push_back(CalledFunction.first());

  while (!Worklist.empty()) {
    auto CalledFunctionName = Worklist.pop_back_val();
    DEBUG(dbgs() << "Process import for " << CalledFunctionName << "\n");

    // Try to get a summary for this function call.
    auto InfoList = Index.findFunctionInfoList(CalledFunctionName);
    if (InfoList == Index.end()) {
      DEBUG(dbgs() << "No summary for " << CalledFunctionName
                   << " Ignoring.\n");
      continue;
    }
    assert(!InfoList->second.empty() && "No summary, error at import?");

    // Comdat can have multiple entries, FIXME: what do we do with them?
    auto &Info = InfoList->second[0];
    assert(Info && "Nullptr in list, error importing summaries?\n");

    auto *Summary = Info->functionSummary();
    if (!Summary) {
      // FIXME: in case we are lazyloading summaries, we can do it now.
      dbgs() << "Missing summary for  " << CalledFunctionName
             << ", error at import?\n";
      llvm_unreachable("Missing summary");
    }

    //
    // No profitability notion right now, just import all the time...
    //

    // Get the module path from the summary.
    auto FileName = Summary->modulePath();
    DEBUG(dbgs() << "Importing " << CalledFunctionName << " from " << FileName
                 << "\n");

    // Get the module for the import (potentially from the cache).
    auto &Module = getOrLoadModule(FileName);

    // The function that we will import!
    GlobalValue *SGV = Module.getNamedValue(CalledFunctionName);
    Function *F = dyn_cast<Function>(SGV);
    if (!F && isa<GlobalAlias>(SGV)) {
      auto *SGA = dyn_cast<GlobalAlias>(SGV);
      F = dyn_cast<Function>(SGA->getBaseObject());
    }
    if (!F) {
      errs() << "Can't load function '" << CalledFunctionName << "' in Module '"
             << FileName << "', error in the summary?\n";
      llvm_unreachable("Can't load function in Module");
    }

    // We cannot import weak_any functions without possibly affecting the
    // order they are seen and selected by the linker, changing program
    // semantics.
    if (F->hasWeakAnyLinkage()) {
      DEBUG(dbgs() << "Ignoring import request for weak-any function "
                   << CalledFunctionName << " from " << FileName << "\n");
      continue;
    }

    // Link in the specified function.
    if (L.linkInModule(&Module, Linker::Flags::None, &Index, F))
      report_fatal_error("Function Import: link error");

    // TODO: Process the newly imported function and add callees to the
    // worklist.

    Changed = true;
  }
  return Changed;
}

/// Summary file to use for function importing when using -function-import from
/// the command line.
static cl::opt<std::string>
    SummaryFile("summary-file",
                cl::desc("The summary file to use for function importing."));

static void diagnosticHandler(const DiagnosticInfo &DI) {
  raw_ostream &OS = errs();
  DiagnosticPrinterRawOStream DP(OS);
  DI.print(DP);
  OS << '\n';
}

/// Parse the function index out of an IR file and return the function
/// index object if found, or nullptr if not.
static std::unique_ptr<FunctionInfoIndex>
getFunctionIndexForFile(StringRef Path, std::string &Error,
                        DiagnosticHandlerFunction DiagnosticHandler) {
  std::unique_ptr<MemoryBuffer> Buffer;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(Path);
  if (std::error_code EC = BufferOrErr.getError()) {
    Error = EC.message();
    return nullptr;
  }
  Buffer = std::move(BufferOrErr.get());
  ErrorOr<std::unique_ptr<object::FunctionIndexObjectFile>> ObjOrErr =
      object::FunctionIndexObjectFile::create(Buffer->getMemBufferRef(),
                                              DiagnosticHandler);
  if (std::error_code EC = ObjOrErr.getError()) {
    Error = EC.message();
    return nullptr;
  }
  return (*ObjOrErr)->takeIndex();
}

/// Pass that performs cross-module function import provided a summary file.
class FunctionImportPass : public ModulePass {

public:
  /// Pass identification, replacement for typeid
  static char ID;

  explicit FunctionImportPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    if (SummaryFile.empty()) {
      report_fatal_error("error: -function-import requires -summary-file\n");
    }
    std::string Error;
    std::unique_ptr<FunctionInfoIndex> Index =
        getFunctionIndexForFile(SummaryFile, Error, diagnosticHandler);
    if (!Index) {
      errs() << "Error loading file '" << SummaryFile << "': " << Error << "\n";
      return false;
    }

    // Perform the import now.
    FunctionImporter Importer(M.getContext(), *Index, diagnosticHandler);
    return Importer.importFunctions(M);

    return false;
  }
};

char FunctionImportPass::ID = 0;
INITIALIZE_PASS_BEGIN(FunctionImportPass, "function-import",
                      "Summary Based Function Import", false, false)
INITIALIZE_PASS_END(FunctionImportPass, "function-import",
                    "Summary Based Function Import", false, false)

namespace llvm {
Pass *createFunctionImportPass() { return new FunctionImportPass(); }
}
