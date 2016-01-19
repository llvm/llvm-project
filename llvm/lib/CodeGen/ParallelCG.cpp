//===-- ParallelCG.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines functions that can be used for parallel code generation.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ParallelCG.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/thread.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/SplitModule.h"

using namespace llvm;

static void codegen(Module *M, llvm::raw_pwrite_stream &OS,
                    const Target *TheTarget, StringRef CPU, StringRef Features,
                    const TargetOptions &Options, Reloc::Model RM,
                    CodeModel::Model CM, CodeGenOpt::Level OL,
                    TargetMachine::CodeGenFileType FileType) {
  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      M->getTargetTriple(), CPU, Features, Options, RM, CM, OL));

  legacy::PassManager CodeGenPasses;
  if (TM->addPassesToEmitFile(CodeGenPasses, OS, FileType))
    report_fatal_error("Failed to setup codegen");
  CodeGenPasses.run(*M);
}

std::unique_ptr<Module>
llvm::splitCodeGen(std::unique_ptr<Module> M,
                   ArrayRef<llvm::raw_pwrite_stream *> OSs, StringRef CPU,
                   StringRef Features, const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM, CodeGenOpt::Level OL,
                   TargetMachine::CodeGenFileType FileType,
                   bool PreserveLocals) {
  StringRef TripleStr = M->getTargetTriple();
  std::string ErrMsg;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleStr, ErrMsg);
  if (!TheTarget)
    report_fatal_error(Twine("Target not found: ") + ErrMsg);

  if (OSs.size() == 1) {
    codegen(M.get(), *OSs[0], TheTarget, CPU, Features, Options, RM, CM,
            OL, FileType);
    return M;
  }

  std::vector<thread> Threads;
  SplitModule(std::move(M), OSs.size(), [&](std::unique_ptr<Module> MPart) {
    // We want to clone the module in a new context to multi-thread the codegen.
    // We do it by serializing partition modules to bitcode (while still on the
    // main thread, in order to avoid data races) and spinning up new threads
    // which deserialize the partitions into separate contexts.
    // FIXME: Provide a more direct way to do this in LLVM.
    SmallVector<char, 0> BC;
    raw_svector_ostream BCOS(BC);
    WriteBitcodeToFile(MPart.get(), BCOS);

    llvm::raw_pwrite_stream *ThreadOS = OSs[Threads.size()];
    Threads.emplace_back(
        [TheTarget, CPU, Features, Options, RM, CM, OL, FileType,
         ThreadOS](const SmallVector<char, 0> &BC) {
          LLVMContext Ctx;
          ErrorOr<std::unique_ptr<Module>> MOrErr =
              parseBitcodeFile(MemoryBufferRef(StringRef(BC.data(), BC.size()),
                                               "<split-module>"),
                               Ctx);
          if (!MOrErr)
            report_fatal_error("Failed to read bitcode");
          std::unique_ptr<Module> MPartInCtx = std::move(MOrErr.get());

          codegen(MPartInCtx.get(), *ThreadOS, TheTarget, CPU, Features,
                  Options, RM, CM, OL, FileType);
        },
        // Pass BC using std::move to ensure that it get moved rather than
        // copied into the thread's context.
        std::move(BC));
  }, PreserveLocals);

  for (thread &T : Threads)
    T.join();

  return {};
}
