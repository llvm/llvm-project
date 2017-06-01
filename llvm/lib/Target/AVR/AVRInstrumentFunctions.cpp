//===-- AVRInstrumentFunctions.cpp - Insert instrumentation for testing ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass takes a function and inserts calls to hook functions which are
// told the name, arguments, and results of function calls.
//
// The hooks can do anything with the information given. It is possible to
// send the data through a serial connection in order to runs tests on
// bare metal.
//
//===----------------------------------------------------------------------===//

#include "AVR.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

using namespace llvm;

#define AVR_INSTRUMENT_FUNCTIONS_NAME "AVR function instrumentation pass"

namespace {

// External symbols that we emit calls to.
namespace symbols {

#define SYMBOL_PREFIX "avr_instrumentation"

  const StringRef PREFIX = SYMBOL_PREFIX;

  // void (i16 argCount);
  const StringRef BEGIN_FUNCTION_SIGNATURE = SYMBOL_PREFIX "_begin_signature";
  // void(i16 argCount);
  const StringRef END_FUNCTION_SIGNATURE = SYMBOL_PREFIX "_end_signature";

#undef SYMBOL_PREFIX
}

class AVRInstrumentFunctions : public FunctionPass {
public:
  static char ID;

  AVRInstrumentFunctions() : FunctionPass(ID) {
    initializeAVRInstrumentFunctionsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return AVR_INSTRUMENT_FUNCTIONS_NAME; }
};

char AVRInstrumentFunctions::ID = 0;

/// Creates a pointer to a string.
static Value *CreateStringPtr(BasicBlock &BB, StringRef Str) {
  LLVMContext &Ctx = BB.getContext();
  IntegerType *I8 = Type::getInt8Ty(Ctx);

  Constant *ConstantStr = ConstantDataArray::getString(Ctx, Str);
  GlobalVariable *GlobalStr = new GlobalVariable(*BB.getParent()->getParent(),
                                                 ConstantStr->getType(),
                                                 true, /* is a constant */
                                                 GlobalValue::PrivateLinkage,
                                                 ConstantStr);
  return GetElementPtrInst::CreateInBounds(GlobalStr,
    {ConstantInt::get(I8, 0), ConstantInt::get(I8, 0)}, "", &BB);
}

static std::string GetTypeName(Type &Ty) {
  if (auto *IntTy = dyn_cast<IntegerType>(&Ty)) {
    return std::string("i") + std::to_string(IntTy->getBitWidth());
  }

  if (Ty.isFloatingPointTy()) {
    return std::string("f") + std::to_string(Ty.getPrimitiveSizeInBits());
  }

  llvm_unreachable("unknown return type");
}

/// Builds a call to one of the signature begin/end hooks.
static void BuildSignatureCall(StringRef SymName, BasicBlock &BB, Function &F) {
  LLVMContext &Ctx = F.getContext();
  IntegerType *I16 = Type::getInt16Ty(Ctx);

  FunctionType *FnType = FunctionType::get(Type::getVoidTy(Ctx),
    {Type::getInt8PtrTy(Ctx), I16}, false);

  Constant *Fn = F.getParent()->getOrInsertFunction(SymName, FnType);
  Value *FunctionName = CreateStringPtr(BB, F.getName());

  Value *Args[] = {FunctionName,
                   ConstantInt::get(I16, F.getArgumentList().size())};
  CallInst::Create(Fn, Args, "", &BB);
}

/// Builds instructions to call into an external function to
/// notify about a function signature beginning.
static void BuildBeginSignature(BasicBlock &BB, Function &F) {
  return BuildSignatureCall(symbols::BEGIN_FUNCTION_SIGNATURE, BB, F);
}

/// Builds instructions to call into an external function to
/// notify about a function signature ending.
static void BuildEndSignature(BasicBlock &BB, Function &F) {
  return BuildSignatureCall(symbols::END_FUNCTION_SIGNATURE, BB, F);
}

/// Get the name of the external symbol that we need to call
/// to notify about this argument.
static std::string GetArgumentSymbolName(Argument &Arg) {
  return (symbols::PREFIX + "_argument_" + GetTypeName(*Arg.getType())).str();
}

/// Builds a call to one of the argument hooks.
static void BuildArgument(BasicBlock &BB, Argument &Arg) {
  Function &F = *Arg.getParent();
  LLVMContext &Ctx = F.getContext();

  Type *I8 = Type::getInt8Ty(Ctx);

  FunctionType *FnType = FunctionType::get(Type::getVoidTy(Ctx),
    {Type::getInt8PtrTy(Ctx), I8, Arg.getType()}, false);

  Constant *Fn = F.getParent()->getOrInsertFunction(
    GetArgumentSymbolName(Arg), FnType);
  Value *ArgName = CreateStringPtr(BB, Arg.getName());

  Value *Args[] = {ArgName, ConstantInt::get(I8, Arg.getArgNo()), &Arg};
  CallInst::Create(Fn, Args, "", &BB);
}

/// Builds a call to all of the function signature hooks.
static void BuildSignature(BasicBlock &BB, Function &F) {
  BuildBeginSignature(BB, F);
  for (Argument &Arg : F.args()) { BuildArgument(BB, Arg); }
  BuildEndSignature(BB, F);
}

/// Builds the instrumentation entry block.
static void BuildEntryBlock(Function &F) {
  BasicBlock &EntryBlock = F.getEntryBlock();

  // Create a new basic block at the start of the existing entry block.
  BasicBlock *BB = BasicBlock::Create(F.getContext(),
                                      "instrumentation_entry",
                                      &F, &EntryBlock);

  BuildSignature(*BB, F);

  // Jump to the actual entry block.
  BranchInst::Create(&EntryBlock, BB);
}

static std::string GetReturnSymbolName(Value &Val) {
  return (symbols::PREFIX + "_result_" + GetTypeName(*Val.getType())).str();
}

static void BuildExitHook(Instruction &I) {
  Function &F = *I.getParent()->getParent();
  LLVMContext &Ctx = F.getContext();

  if (auto *Ret = dyn_cast<ReturnInst>(&I)) {
    Value *RetVal = Ret->getReturnValue();
    assert(RetVal && "should only be instrumenting functions with return values");

    FunctionType *FnType = FunctionType::get(Type::getVoidTy(Ctx),
      {RetVal->getType()}, false);

    Constant *Fn = F.getParent()->getOrInsertFunction(
      GetReturnSymbolName(*RetVal), FnType);

    // Call the result hook just before the return.
    CallInst::Create(Fn, {RetVal}, "", &I);
  }
}

/// Runs return hooks before all returns in a function.
static void BuildExitHooks(Function &F) {
  for (BasicBlock &BB : F) {
    auto BBI = BB.begin(), E = BB.end();
    while (BBI != E) {
      auto NBBI = std::next(BBI);

      BuildExitHook(*BBI);

      // Modified |= expandMI(BB, MBBI);
      BBI = NBBI;
    }
  }
}

static bool ShouldInstrument(Function &F) {
  // No point reporting results if there are none.
  return !F.getReturnType()->isVoidTy();
}

bool AVRInstrumentFunctions::runOnFunction(Function &F) {
  if (ShouldInstrument(F)) {
    BuildEntryBlock(F);
    BuildExitHooks(F);
  }

  return true;
}

} // end of anonymous namespace

INITIALIZE_PASS(AVRInstrumentFunctions, "avr-instrument-functions",
                AVR_INSTRUMENT_FUNCTIONS_NAME, false, false)

namespace llvm {

FunctionPass *createAVRInstrumentFunctionsPass() { return new AVRInstrumentFunctions(); }

} // end of namespace llvm
