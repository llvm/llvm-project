//===------------- llubi.cpp - LLVM UB-aware Interpreter --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility provides an UB-aware interpreter for programs in LLVM bitcode.
// It is not built on top of the existing ExecutionEngine interface, but instead
// implements its own value representation, state tracking and interpreter loop.
//
//===----------------------------------------------------------------------===//

#include "lib/Context.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> InputFile(cl::desc("<input bitcode>"),
                                      cl::Positional, cl::init("-"));

static cl::list<std::string> InputArgv(cl::ConsumeAfter,
                                       cl::desc("<program arguments>..."));

static cl::opt<std::string>
    EntryFunc("entry-function",
              cl::desc("Specify the entry function (default = 'main') "
                       "of the executable"),
              cl::value_desc("function"), cl::init("main"));

static cl::opt<std::string>
    FakeArgv0("fake-argv0",
              cl::desc("Override the 'argv[0]' value passed into the executing"
                       " program"),
              cl::value_desc("executable"));

static cl::opt<bool>
    Verbose("verbose", cl::desc("Print results for each instruction executed."),
            cl::init(false));

cl::OptionCategory InterpreterCategory("Interpreter Options");

static cl::opt<unsigned> MaxMem(
    "max-mem",
    cl::desc("Max amount of memory (in bytes) that can be allocated by the"
             " program, including stack, heap, and global variables."
             " Set to 0 to disable the limit."),
    cl::value_desc("N"), cl::init(0), cl::cat(InterpreterCategory));

static cl::opt<unsigned>
    MaxSteps("max-steps",
             cl::desc("Max number of instructions executed."
                      " Set to 0 to disable the limit."),
             cl::value_desc("N"), cl::init(0), cl::cat(InterpreterCategory));

static cl::opt<unsigned> MaxStackDepth(
    "max-stack-depth",
    cl::desc("Max stack depth (default = 256). Set to 0 to disable the limit."),
    cl::value_desc("N"), cl::init(256), cl::cat(InterpreterCategory));

static cl::opt<unsigned>
    VScale("vscale", cl::desc("The value of llvm.vscale (default = 4)"),
           cl::value_desc("N"), cl::init(4), cl::cat(InterpreterCategory));

static cl::opt<unsigned>
    Seed("seed",
         cl::desc("Random seed for non-deterministic behavior (default = 0)"),
         cl::value_desc("N"), cl::init(0), cl::cat(InterpreterCategory));

cl::opt<ubi::UndefValueBehavior> UndefBehavior(
    "", cl::desc("Choose undef value behavior:"),
    cl::values(clEnumVal(ubi::UndefValueBehavior::NonDeterministic,
                         "Each load of an uninitialized byte yields a freshly "
                         "random value."),
               clEnumVal(ubi::UndefValueBehavior::Zero,
                         "All uses of an uninitialized byte yield zero.")));

class VerboseEventHandler : public ubi::EventHandler {
public:
  bool onInstructionExecuted(Instruction &I,
                             const ubi::AnyValue &Result) override {
    if (Result.isNone()) {
      errs() << I << '\n';
    } else {
      errs() << I << " => " << Result << '\n';
    }

    return true;
  }

  void onImmediateUB(StringRef Msg) override {
    errs() << "Immediate UB detected: " << Msg << '\n';
  }

  void onError(StringRef Msg) override { errs() << "Error: " << Msg << '\n'; }

  bool onBBJump(Instruction &I, BasicBlock &To) override {
    errs() << I << " jump to ";
    To.printAsOperand(errs(), /*PrintType=*/false);
    errs() << '\n';
    return true;
  }

  bool onFunctionEntry(Function &F, ArrayRef<ubi::AnyValue> Args,
                       CallBase *CallSite) override {
    errs() << "Entering function: " << F.getName() << '\n';
    size_t ArgSize = F.arg_size();
    for (auto &&[Idx, Arg] : enumerate(Args)) {
      if (Idx >= ArgSize)
        errs() << "  vaarg[" << (Idx - ArgSize) << "] = " << Arg << '\n';
      else
        errs() << "  " << *F.getArg(Idx) << " = " << Arg << '\n';
    }
    return true;
  }

  bool onFunctionExit(Function &F, const ubi::AnyValue &RetVal) override {
    errs() << "Exiting function: " << F.getName() << '\n';
    return true;
  }

  void onUnrecognizedInstruction(Instruction &I) override {
    errs() << "Unrecognized instruction: " << I << '\n';
  }
};

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "llvm ub-aware interpreter\n");

  if (EntryFunc.empty()) {
    WithColor::error() << "--entry-function name cannot be empty\n";
    return 1;
  }

  LLVMContext Context;

  // Load the bitcode...
  SMDiagnostic Err;
  std::unique_ptr<Module> Owner = parseIRFile(InputFile, Err, Context);
  Module *Mod = Owner.get();
  if (!Mod) {
    Err.print(argv[0], errs());
    return 1;
  }

  // If the user specifically requested an argv[0] to pass into the program,
  // do it now.
  if (!FakeArgv0.empty()) {
    InputFile = static_cast<std::string>(FakeArgv0);
  } else {
    // Otherwise, if there is a .bc suffix on the executable strip it off, it
    // might confuse the program.
    if (StringRef(InputFile).ends_with(".bc"))
      InputFile.erase(InputFile.length() - 3);
  }

  // Add the module's name to the start of the vector of arguments to main().
  InputArgv.insert(InputArgv.begin(), InputFile);

  // Initialize the execution context and set parameters.
  ubi::Context Ctx(*Mod);
  Ctx.setMemoryLimit(MaxMem);
  Ctx.setVScale(VScale);
  Ctx.setMaxSteps(MaxSteps);
  Ctx.setMaxStackDepth(MaxStackDepth);
  Ctx.setUndefValueBehavior(UndefBehavior);
  Ctx.reseed(Seed);

  if (!Ctx.initGlobalValues()) {
    WithColor::error() << "Failed to initialize global values (e.g., the "
                          "memory limit may be too low).\n";
    return 1;
  }

  // Call the main function from M as if its signature were:
  //   int main (int argc, char **argv)
  // using the contents of Args to determine argc & argv
  Function *EntryFn = Mod->getFunction(EntryFunc);
  if (!EntryFn) {
    WithColor::error() << '\'' << EntryFunc
                       << "\' function not found in module.\n";
    return 1;
  }
  TargetLibraryInfo TLI(Ctx.getTLIImpl());
  Type *IntTy = IntegerType::get(Ctx.getContext(), TLI.getIntSize());
  Type *PtrTy = PointerType::getUnqual(Ctx.getContext());
  auto *MainFuncTy = FunctionType::get(IntTy, {IntTy, PtrTy}, false);
  SmallVector<ubi::AnyValue> Args;
  if (EntryFn->getFunctionType() == MainFuncTy) {
    Args.push_back(
        Ctx.getConstantValue(ConstantInt::get(IntTy, InputArgv.size())));

    uint32_t PtrSize = Ctx.getDataLayout().getPointerSize();
    uint64_t PtrsSize = PtrSize * (InputArgv.size() + 1);
    auto ArgvPtrsMem = Ctx.allocate(PtrsSize, 8, "argv",
                                    /*AS=*/0, ubi::MemInitKind::Zeroed);
    if (!ArgvPtrsMem) {
      WithColor::error() << "Failed to allocate memory for argv pointers.\n";
      return 1;
    }
    for (const auto &[Idx, Arg] : enumerate(InputArgv)) {
      uint64_t Size = Arg.length() + 1;
      auto ArgvStrMem = Ctx.allocate(Size, 8, "argv_str",
                                     /*AS=*/0, ubi::MemInitKind::Zeroed);
      if (!ArgvStrMem) {
        WithColor::error() << "Failed to allocate memory for argv strings.\n";
        return 1;
      }
      ubi::Pointer ArgPtr = Ctx.deriveFromMemoryObject(ArgvStrMem);
      Ctx.storeRawBytes(*ArgvStrMem, 0, Arg.c_str(), Arg.length());
      Ctx.store(*ArgvPtrsMem, Idx * PtrSize, ArgPtr, PtrTy);
    }
    Args.push_back(Ctx.deriveFromMemoryObject(ArgvPtrsMem));
  } else if (!EntryFn->arg_empty()) {
    // If the signature does not match (e.g., llvm-reduce change the signature
    // of main), it will pass null values for all arguments.
    WithColor::warning()
        << "The signature of function '" << EntryFunc
        << "' does not match 'int main(int, char**)', passing null values for "
           "all arguments.\n";
    Args.reserve(EntryFn->arg_size());
    for (Argument &Arg : EntryFn->args())
      Args.push_back(ubi::AnyValue::getNullValue(Ctx, Arg.getType()));
  }

  ubi::EventHandler NoopHandler;
  VerboseEventHandler VerboseHandler;
  ubi::AnyValue RetVal;
  if (!Ctx.runFunction(*EntryFn, Args, RetVal,
                       Verbose ? VerboseHandler : NoopHandler)) {
    WithColor::error() << "Execution of function '" << EntryFunc
                       << "' failed.\n";
    return 1;
  }

  // If the function returns an integer, return that as the exit code.
  if (EntryFn->getReturnType()->isIntegerTy()) {
    assert(!RetVal.isNone() && "Expected a return value from entry function");
    if (RetVal.isPoison()) {
      WithColor::error() << "Execution of function '" << EntryFunc
                         << "' resulted in poison return value.\n";
      return 1;
    }
    APInt Result = RetVal.asInteger();
    return (int)Result.extractBitsAsZExtValue(
        std::min(Result.getBitWidth(), 8U), 0);
  }
  return 0;
}
