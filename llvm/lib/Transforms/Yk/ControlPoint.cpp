//===- ControlPoint.cpp - Synthesise the yk control point -----------------===//
//
// This pass finds the user's call to the dummy control point and replaces it
// with a call to a new control point that implements the necessary logic to
// drive the yk JIT.
//
// The pass converts an interpreter loop that looks like this:
//
// ```
// pc = 0;
// while (...) {
//     yk_control_point(); // <- dummy control point
//     bc = program[pc];
//     switch (bc) {
//         // bytecode handlers here.
//     }
// }
// ```
//
// Into one that looks like this (note that this transformation happens at the
// IR level):
//
// ```
// // The YkCtrlPointStruct contains one member for each live LLVM variable
// // just before the call to the control point.
// struct YkCtrlPointStruct {
//     size_t pc;
// }
//
// pc = 0;
// while (...) {
//     struct YkCtrlPointStruct cp_in = { pc };
//     // Now we call the patched control point.
//     YkCtrlPointStruct cp_out = yk_new_control_point(cp_in);
//     pc = cp_out.pc;
//     bc = program[pc];
//     switch (bc) {
//         // bytecode handlers here.
//     }
// }
// ```
//
// The call to the dummy control point must be the first thing that appears in
// an interpreter dispatch loop.
//
// YKFIXME: The control point cannot yet be used in an interpreter using
// threaded dispatch.
//
// YKFIXME: The tracing logic is currently over-simplified. The following items
// need to be fixed:
//
//  - The address of `YkLocation` instances are used for identity, but they are
//    intended to be freely moved by the user.
//
//  - Tracing starts when we encounter a location for which we have no machine
//    code. A hot counter should be used instead.
//
//  - There can be only one compiled trace for now. There should be a code
//    cache mapping from JIT locations to their machine code.
//
//  - The interpreter is assumed to be single threaded. We should implement a
//    synchronisation function in Rust code that synchronises many threads which
//    are calling the control point concurrently. This function should return a
//    value that indicates if we should start/stop tracing, or jump to machine
//    code etc.
//
//  - Guards are currently assumed to abort the program.
//    https://github.com/ykjit/yk/issues/443
//
//  - The block that performs the call to JITted code branches back to itself
//    to achieve rudimentary trace stitching. The looping should really be
//    implemented in the JITted code itself so that it isn't necessary to
//    repeatedly enter and exit the JITted code.
//    https://github.com/ykjit/yk/issues/442
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Yk/ControlPoint.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>

#define DEBUG_TYPE "yk-control-point"
#define JIT_STATE_PREFIX "jit-state: "

using namespace llvm;

/// Find the call to the dummy control point that we want to patch.
/// Returns either a pointer the call instruction, or `nullptr` if the call
/// could not be found.
/// YKFIXME: For now assumes there's only one control point.
CallInst *findControlPointCall(Module &M) {
  // Find the declaration of `yk_control_point()`.
  Function *CtrlPoint = M.getFunction(YK_DUMMY_CONTROL_POINT);
  if (CtrlPoint == nullptr)
    return nullptr;

  // Find the call site of `yk_control_point()`.
  Value::user_iterator U = CtrlPoint->user_begin();
  if (U == CtrlPoint->user_end())
    return nullptr;

  return cast<CallInst>(*U);
}

/// Creates a call for printing debug information inside the control point.
void createJITStatePrint(IRBuilder<> &Builder, Module *Mod, std::string Str) {
  if (std::getenv("YKD_PRINT_JITSTATE") == nullptr)
    return;
  LLVMContext &Context = Mod->getContext();
  FunctionCallee Puts = Mod->getOrInsertFunction(
      "__yk_debug_print",
      FunctionType::get(Type::getVoidTy(Context),
                        PointerType::get(Type::getInt8Ty(Context), 0), true));
  Value *PutsString =
      Builder.CreateGlobalStringPtr(StringRef(JIT_STATE_PREFIX + Str + "\n"));
  Builder.CreateCall(Puts, PutsString);
}

/// Generates the new control point, which includes all logic to start/stop
/// tracing and to compile/execute traces.
void createControlPoint(Module &Mod, Function *F, std::vector<Value *> LiveVars,
                        StructType *YkCtrlPointStruct, Type *YkLocTy) {
  auto &Context = Mod.getContext();

  // Create control point blocks and setup the IRBuilder.
  BasicBlock *CtrlPointEntry = BasicBlock::Create(Context, "cpentry", F);
  BasicBlock *BBTracing = BasicBlock::Create(Context, "bbtracing", F);
  BasicBlock *BBNotTracing = BasicBlock::Create(Context, "bbnottracing", F);
  BasicBlock *BBHasTrace = BasicBlock::Create(Context, "bbhastrace", F);
  BasicBlock *BBExecuteTrace = BasicBlock::Create(Context, "bbhastrace", F);
  BasicBlock *BBHasNoTrace = BasicBlock::Create(Context, "bbhasnotrace", F);
  BasicBlock *BBReturn = BasicBlock::Create(Context, "bbreturn", F);
  BasicBlock *BBStopTracing = BasicBlock::Create(Context, "bbstoptracing", F);
  IRBuilder<> Builder(CtrlPointEntry);

  // Some frequently used constants.
  ConstantInt *Int0 = ConstantInt::get(Context, APInt(8, 0));
  Constant *PtNull = Constant::getNullValue(Type::getInt8PtrTy(Context));

  // Add definitions for __yktrace functions.
  Function *FuncStartTracing = llvm::Function::Create(
      FunctionType::get(Type::getVoidTy(Context), {Type::getInt64Ty(Context)},
                        false),
      GlobalValue::ExternalLinkage, "__yktrace_start_tracing", Mod);

  Function *FuncStopTracing = llvm::Function::Create(
      FunctionType::get(Type::getInt8PtrTy(Context), {}, false),
      GlobalValue::ExternalLinkage, "__yktrace_stop_tracing", Mod);

  Function *FuncCompileTrace = llvm::Function::Create(
      FunctionType::get(Type::getInt8PtrTy(Context),
                        {Type::getInt8PtrTy(Context)}, false),
      GlobalValue::ExternalLinkage, "__yktrace_irtrace_compile", Mod);

  // Generate global variables to hold the state of the JIT.
  GlobalVariable *GVTracing = new GlobalVariable(
      Mod, Type::getInt8Ty(Context), false, GlobalVariable::InternalLinkage,
      Int0, "tracing", (GlobalVariable *)nullptr);

  GlobalVariable *GVCompiledTrace = new GlobalVariable(
      Mod, Type::getInt8PtrTy(Context), false, GlobalVariable::InternalLinkage,
      PtNull, "compiled_trace", (GlobalVariable *)nullptr);

  GlobalVariable *GVStartLoc = new GlobalVariable(
      Mod, YkLocTy, false, GlobalVariable::InternalLinkage,
      Constant::getNullValue(YkLocTy), "start_loc", (GlobalVariable *)nullptr);

  // Create control point entry block. Checks if we are currently tracing.
  Value *GVTracingVal = Builder.CreateLoad(Type::getInt8Ty(Context), GVTracing);
  Value *IsTracing =
      Builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, GVTracingVal, Int0);
  Builder.CreateCondBr(IsTracing, BBNotTracing, BBTracing);

  // Create block for "not tracing" case. Checks if we already compiled a trace.
  Builder.SetInsertPoint(BBNotTracing);
  Value *GVCompiledTraceVal =
      Builder.CreateLoad(Type::getInt8PtrTy(Context), GVCompiledTrace);
  Value *HasTrace = Builder.CreateICmp(CmpInst::Predicate::ICMP_EQ,
                                       GVCompiledTraceVal, PtNull);
  Builder.CreateCondBr(HasTrace, BBHasNoTrace, BBHasTrace);

  // Create block that starts tracing.
  Builder.SetInsertPoint(BBHasNoTrace);
  createJITStatePrint(Builder, &Mod, "start-tracing");
  Builder.CreateCall(FuncStartTracing->getFunctionType(), FuncStartTracing,
                     {ConstantInt::get(Context, APInt(64, 1))});
  Builder.CreateStore(ConstantInt::get(Context, APInt(8, 1)), GVTracing);
  Builder.CreateStore(F->getArg(0), GVStartLoc);
  Builder.CreateBr(BBReturn);

  // Create block that checks if we've reached the same location again so we
  // can execute a compiled trace.
  Builder.SetInsertPoint(BBHasTrace);
  Value *ValStartLoc = Builder.CreateLoad(YkLocTy, GVStartLoc);
  Value *ExecTraceCond = Builder.CreateICmp(CmpInst::Predicate::ICMP_EQ,
                                            ValStartLoc, F->getArg(0));
  Builder.CreateCondBr(ExecTraceCond, BBExecuteTrace, BBReturn);

  // Create block that executes a compiled trace.
  Builder.SetInsertPoint(BBExecuteTrace);
  std::vector<Type *> TypeParams;
  for (Value *LV : LiveVars) {
    TypeParams.push_back(LV->getType());
  }
  FunctionType *FType =
      FunctionType::get(YkCtrlPointStruct, {YkCtrlPointStruct}, false);
  Value *CastTrace =
      Builder.CreateBitCast(GVCompiledTraceVal, FType->getPointerTo());
  createJITStatePrint(Builder, &Mod, "enter-jit-code");
  CallInst *CTResult = Builder.CreateCall(FType, CastTrace, F->getArg(1));
  createJITStatePrint(Builder, &Mod, "exit-jit-code");
  CTResult->setTailCall(true);
  Builder.CreateBr(BBExecuteTrace);

  // Create block that decides when to stop tracing.
  Builder.SetInsertPoint(BBTracing);
  Value *ValStartLoc2 = Builder.CreateLoad(YkLocTy, GVStartLoc);
  Value *StopTracingCond = Builder.CreateICmp(CmpInst::Predicate::ICMP_EQ,
                                              ValStartLoc2, F->getArg(0));
  Builder.CreateCondBr(StopTracingCond, BBStopTracing, BBReturn);

  // Create block that stops tracing, compiles a trace, and stores it in a
  // global variable.
  Builder.SetInsertPoint(BBStopTracing);
  Value *TR =
      Builder.CreateCall(FuncStopTracing->getFunctionType(), FuncStopTracing);
  Value *CT = Builder.CreateCall(FuncCompileTrace->getFunctionType(),
                                 FuncCompileTrace, {TR});
  Builder.CreateStore(CT, GVCompiledTrace);
  Builder.CreateStore(ConstantInt::get(Context, APInt(8, 0)), GVTracing);
  createJITStatePrint(Builder, &Mod, "stop-tracing");
  Builder.CreateBr(BBReturn);

  // Create return block. Returns the unchanged YkCtrlPointStruct if no
  // compiled trace was executed, otherwise return a new YkCtrlPointStruct
  // which contains the changed interpreter state.
  Builder.SetInsertPoint(BBReturn);
  Value *YkCtrlPointVars = F->getArg(1);
  PHINode *Phi = Builder.CreatePHI(YkCtrlPointStruct, 3);
  Phi->addIncoming(YkCtrlPointVars, BBHasTrace);
  Phi->addIncoming(YkCtrlPointVars, BBTracing);
  Phi->addIncoming(YkCtrlPointVars, BBHasNoTrace);
  Phi->addIncoming(YkCtrlPointVars, BBStopTracing);
  Builder.CreateRet(Phi);
}

/// Extract all live variables that need to be passed into the control point.
std::vector<Value *> getLiveVars(DominatorTree &DT, CallInst *OldCtrlPoint) {
  std::vector<Value *> Vec;
  Function *Func = OldCtrlPoint->getFunction();
  for (auto &BB : *Func) {
    if (!DT.dominates(cast<Instruction>(OldCtrlPoint), &BB)) {
      for (auto &I : BB) {
        if ((!I.getType()->isVoidTy()) &&
            (DT.dominates(&I, cast<Instruction>(OldCtrlPoint)))) {
          Vec.push_back(&I);
        }
      }
    }
  }
  return Vec;
}

namespace llvm {
void initializeYkControlPointPass(PassRegistry &);
}

namespace {
class YkControlPoint : public ModulePass {
public:
  static char ID;
  YkControlPoint() : ModulePass(ID) {
    initializeYkControlPointPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Context = M.getContext();

    // Locate the "dummy" control point provided by the user.
    CallInst *OldCtrlPointCall = findControlPointCall(M);
    if (OldCtrlPointCall == nullptr) {
      Context.emitError(
          "ykllvm couldn't find the call to `yk_control_point()`");
      return false;
    }

    // Replace old control point call.
    IRBuilder<> Builder(OldCtrlPointCall);

    // Get function containing the control point.
    Function *Caller = OldCtrlPointCall->getFunction();

    // Find all live variables just before the call to the control point.
    DominatorTree DT(*Caller);
    std::vector<Value *> LiveVals = getLiveVars(DT, OldCtrlPointCall);
    if (LiveVals.size() == 0) {
      Context.emitError(
          "The interpreter loop has no live variables!\n"
          "ykllvm doesn't support this scenario, as such an interpreter would "
          "make little sense.");
      return false;
    }

    // Generate the YkCtrlPointVars struct. This struct is used to package up a
    // copy of all LLVM variables that are live just before the call to the
    // control point. These are passed in to the patched control point so that
    // they can be used as inputs and outputs to JITted trace code. The control
    // point returns a new YkCtrlPointVars whose members may have been mutated
    // by JITted trace code (if a trace was executed).
    std::vector<Type *> TypeParams;
    for (Value *V : LiveVals) {
      TypeParams.push_back(V->getType());
    }
    StructType *CtrlPointReturnTy =
        StructType::create(TypeParams, "YkCtrlPointVars");

    // Create the new control point.
    Type *YkLocTy = OldCtrlPointCall->getArgOperand(0)->getType();
    FunctionType *FType = FunctionType::get(
        CtrlPointReturnTy, {YkLocTy, CtrlPointReturnTy}, false);
    Function *NF = Function::Create(FType, GlobalVariable::ExternalLinkage,
                                    YK_NEW_CONTROL_POINT, M);

    // Instantiate the YkCtrlPointStruct to pass in to the control point.
    Value *InputStruct = cast<Value>(Constant::getNullValue(CtrlPointReturnTy));
    unsigned LvIdx = 0;
    for (Value *LV : LiveVals) {
      InputStruct = Builder.CreateInsertValue(InputStruct, LV, LvIdx);
      assert(LvIdx != UINT_MAX);
      LvIdx++;
    }

    // Insert call to the new control point.
    CallInst *CtrlPointRet = Builder.CreateCall(
        NF, {OldCtrlPointCall->getArgOperand(0), InputStruct});

    // Once the control point returns we need to extract the (potentially
    // mutated) values from the returned YkCtrlPointStruct and reassign them to
    // their corresponding live variables. In LLVM IR we can do this by simply
    // replacing all future references with the new values.
    LvIdx = 0;
    for (Value *LV : LiveVals) {
      Value *New = Builder.CreateExtractValue(cast<Value>(CtrlPointRet), LvIdx);
      LV->replaceUsesWithIf(
          New, [&](Use &U) { return DT.dominates(CtrlPointRet, U); });
      assert(LvIdx != UINT_MAX);
      LvIdx++;
    }

    // Replace the call to the dummy control point.
    OldCtrlPointCall->eraseFromParent();

    // Generate new control point logic.
    createControlPoint(M, NF, LiveVals, CtrlPointReturnTy, YkLocTy);
    return true;
  }
};
} // namespace

char YkControlPoint::ID = 0;
INITIALIZE_PASS(YkControlPoint, DEBUG_TYPE, "yk control point", false, false)

ModulePass *llvm::createYkControlPointPass() { return new YkControlPoint(); }
