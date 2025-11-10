//===----- InlineBitcodeLibrary.cpp - Link bitcode to the module ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass links and inlines Ripple functions from other modules.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InlineBitcodeLibrary.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Use.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Ripple/Ripple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "inline-bitcode-library"

namespace {
class DiagnosticInfoInlineBitcodeLib : public DiagnosticInfoGeneric {
  // Used to store the message when the input twine is not a single StringRef
  SmallVector<char, 0> ErrStr;
  // The error message
  StringRef ErrStrRef;
  // Keep a Twine to ErrStrRef for DiagnosticInfoGeneric
  Twine ErrTwine;

public:
  DiagnosticInfoInlineBitcodeLib(DiagnosticSeverity Severity, const Twine &Msg)
      : DiagnosticInfoGeneric(nullptr, ErrTwine, Severity),
        ErrStrRef(Msg.toStringRef(ErrStr)), ErrTwine(ErrStrRef) {}
};
} // namespace

namespace llvm {
namespace RippleCL {
llvm::cl::opt<bool> RippleDisableLinking(
    "ripple-disable-link", llvm::cl::init(false),
    llvm::cl::desc("Disable linking and inlining the library function in "
                   "linkbitcode pass"));
} // namespace RippleCL
} // namespace llvm

namespace {
std::unique_ptr<Module> getBitcodeModule(LLVMContext &Context,
                                         StringRef PathToBitcode) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrError =
      MemoryBuffer::getFile(PathToBitcode);
  if (!BufferOrError) {
    Context.diagnose(DiagnosticInfoInlineBitcodeLib(
        DiagnosticSeverity::DS_Error,
        Twine("reading bitcode file: ") + PathToBitcode +
            Twine(BufferOrError.getError().message())));
    return nullptr;
  }
  // Parse the bitcode file
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BufferOrError.get());
  Expected<std::unique_ptr<Module>> ModuleOrErr =
      parseBitcodeFile(Buffer->getMemBufferRef(), Context);
  if (!ModuleOrErr) {
    Context.diagnose(DiagnosticInfoInlineBitcodeLib(
        DiagnosticSeverity::DS_Error,
        Twine("parsing bitcode file: ") + toString(ModuleOrErr.takeError())));
    return nullptr;
  }
  return std::move(*ModuleOrErr);
}
} // namespace

PreservedAnalyses InlineBitcodeLibraryPass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  LLVM_DEBUG(dbgs() << "Applying inline-bitcode-library pass to '"
                    << M.getName() << "'\n");
  LLVM_DEBUG(dbgs() << "\n Module Before bit code is linked:\n\n";
             M.print(dbgs(), nullptr); dbgs() << "\n");

  // early exit
  if (RippleCL::RippleDisableLinking)
    return PreservedAnalyses::all();

  bool DoLink = false;
  // Examine the names of functions in the module, and enable linking only if a
  // ripple library function is declared and used in the module. By convention,
  // a ripple library function's name starts with the "ripple_" prefix.
  constexpr StringRef RipplePrefix = "ripple_";
  for (auto &F : M) {
    if (F.getName().starts_with(RipplePrefix) && F.isDeclaration() &&
        !F.use_empty()) {
      LLVM_DEBUG(
          dbgs()
          << "\nLinking enabled because this ripple library function is found: "
          << F.getName() << "\n");
      DoLink = true;
      break;
    }
  }
  if (!DoLink || RippleCL::RippleLibs.empty())
    return PreservedAnalyses::all();

  bool ModifiedModule = false;
  for (const auto &BitcodePath : RippleCL::RippleLibs) {
    if (BitcodePath.empty())
      continue;
    std::unique_ptr<Module> BCModule =
        getBitcodeModule(M.getContext(), BitcodePath);
    if (!BCModule)
      continue;
    // Collect the names of the ripple library functions in the bitcode
    DenseSet<StringRef> FunctionsInBitcode;
    for (auto &F : *BCModule) {
      if (!F.isDeclaration() && F.getName().starts_with(RipplePrefix)) {
        if (!isInlineViable(F).isSuccess())
          M.getContext().diagnose(DiagnosticInfoInlineBitcodeLib(
              DiagnosticSeverity::DS_Warning,
              "external ripple function '" + F.getName() +
                  "' is not viable for inlining"));
        else
          FunctionsInBitcode.insert(F.getName());
      }
    }
    // Link the bitcode
    Linker Linker(M);
    if (Linker.linkInModule(std::move(BCModule))) {
      M.getContext().diagnose(DiagnosticInfoInlineBitcodeLib(
          DiagnosticSeverity::DS_Error, "could not link the module"));
      continue;
    }
    ModifiedModule = true;
    // Inline the functions from the bitcode
    DenseSet<CallInst *> CallsToInline;
    for (const auto &FuncName : FunctionsInBitcode) {
      if (Function *F = M.getFunction(FuncName)) {
        if (!F->isDeclaration()) {
          CallsToInline.clear();
          for (auto &U : F->uses())
            if (CallInst *CI = dyn_cast<CallInst>(U.getUser()))
              CallsToInline.insert(CI);
          for (CallInst *CI : CallsToInline) {
            InlineFunctionInfo IFI;
            InlineResult IR = InlineFunction(*CI, IFI);
            if (!IR.isSuccess())
              M.getContext().diagnose(DiagnosticInfoInlineBitcodeLib(
                  DiagnosticSeverity::DS_Error,
                  Twine("failed to inline function: ") + FuncName +
                      "; Reason: " + IR.getFailureReason()));
          }
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "\n Module after bit code is linked and inlined:\n\n";
             M.print(dbgs(), nullptr); dbgs() << "\n");

  return ModifiedModule ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
