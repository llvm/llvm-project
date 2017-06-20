//===- OrcLazyJIT.cpp - Basic Orc-based JIT for lazy execution ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcLazyJIT.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <system_error>

using namespace llvm;

namespace {

enum class DumpKind {
  NoDump,
  DumpFuncsToStdOut,
  DumpModsToStdOut,
  DumpModsToDisk
};

} // end anonymous namespace

static cl::opt<DumpKind> OrcDumpKind(
    "orc-lazy-debug", cl::desc("Debug dumping for the orc-lazy JIT."),
    cl::init(DumpKind::NoDump),
    cl::values(clEnumValN(DumpKind::NoDump, "no-dump", "Don't dump anything."),
               clEnumValN(DumpKind::DumpFuncsToStdOut, "funcs-to-stdout",
                          "Dump function names to stdout."),
               clEnumValN(DumpKind::DumpModsToStdOut, "mods-to-stdout",
                          "Dump modules to stdout."),
               clEnumValN(DumpKind::DumpModsToDisk, "mods-to-disk",
                          "Dump modules to the current "
                          "working directory. (WARNING: "
                          "will overwrite existing files).")),
    cl::Hidden);

static cl::opt<bool> OrcInlineStubs("orc-lazy-inline-stubs",
                                    cl::desc("Try to inline stubs"),
                                    cl::init(true), cl::Hidden);

OrcLazyJIT::TransformFtor OrcLazyJIT::createDebugDumper() {
  switch (OrcDumpKind) {
  case DumpKind::NoDump:
    return [](std::unique_ptr<Module> M) { return M; };

  case DumpKind::DumpFuncsToStdOut:
    return [](std::unique_ptr<Module> M) {
      printf("[ ");

      for (const auto &F : *M) {
        if (F.isDeclaration())
          continue;

        if (F.hasName()) {
          std::string Name(F.getName());
          printf("%s ", Name.c_str());
        } else
          printf("<anon> ");
      }

      printf("]\n");
      return M;
    };

  case DumpKind::DumpModsToStdOut:
    return [](std::unique_ptr<Module> M) {
             outs() << "----- Module Start -----\n" << *M
                    << "----- Module End -----\n";

             return M;
           };

  case DumpKind::DumpModsToDisk:
    return [](std::unique_ptr<Module> M) {
             std::error_code EC;
             raw_fd_ostream Out(M->getModuleIdentifier() + ".ll", EC,
                                sys::fs::F_Text);
             if (EC) {
               errs() << "Couldn't open " << M->getModuleIdentifier()
                      << " for dumping.\nError:" << EC.message() << "\n";
               exit(1);
             }
             Out << *M;
             return M;
           };
  }
  llvm_unreachable("Unknown DumpKind");
}

// Defined in lli.cpp.
CodeGenOpt::Level getOptLevel();

template <typename PtrTy>
static PtrTy fromTargetAddress(JITTargetAddress Addr) {
  return reinterpret_cast<PtrTy>(static_cast<uintptr_t>(Addr));
}

int llvm::runOrcLazyJIT(std::vector<std::unique_ptr<Module>> Ms,
                        const std::vector<std::string> &Args) {
  // Add the program's symbols into the JIT's search space.
  if (sys::DynamicLibrary::LoadLibraryPermanently(nullptr)) {
    errs() << "Error loading program symbols.\n";
    return 1;
  }

  // Grab a target machine and try to build a factory function for the
  // target-specific Orc callback manager.
  EngineBuilder EB;
  EB.setOptLevel(getOptLevel());
  auto TM = std::unique_ptr<TargetMachine>(EB.selectTarget());
  Triple T(TM->getTargetTriple());
  auto CompileCallbackMgr = orc::createLocalCompileCallbackManager(T, 0);

  // If we couldn't build the factory function then there must not be a callback
  // manager for this target. Bail out.
  if (!CompileCallbackMgr) {
    errs() << "No callback manager available for target '"
           << TM->getTargetTriple().str() << "'.\n";
    return 1;
  }

  auto IndirectStubsMgrBuilder = orc::createLocalIndirectStubsManagerBuilder(T);

  // If we couldn't build a stubs-manager-builder for this target then bail out.
  if (!IndirectStubsMgrBuilder) {
    errs() << "No indirect stubs manager available for target '"
           << TM->getTargetTriple().str() << "'.\n";
    return 1;
  }

  // Everything looks good. Build the JIT.
  OrcLazyJIT J(std::move(TM), std::move(CompileCallbackMgr),
               std::move(IndirectStubsMgrBuilder),
               OrcInlineStubs);

  // Add the module, look up main and run it.
  J.addModuleSet(std::move(Ms));
  auto MainSym = J.findSymbol("main");

  if (!MainSym) {
    errs() << "Could not find main function.\n";
    return 1;
  }

  using MainFnPtr = int (*)(int, const char*[]);
  std::vector<const char *> ArgV;
  for (auto &Arg : Args)
    ArgV.push_back(Arg.c_str());
  auto Main = fromTargetAddress<MainFnPtr>(MainSym.getAddress());
  return Main(ArgV.size(), (const char**)ArgV.data());
}
