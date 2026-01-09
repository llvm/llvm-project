//===-- WasmEHPrepare - Prepare excepton handling for WebAssembly --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation is designed for use by code generators which use
// WebAssembly exception handling scheme. This currently supports C++
// exceptions.
//
// WebAssembly exception handling uses Windows exception IR for the middle level
// representation. This pass does the following transformation for every
// catchpad block:
// (In C-style pseudocode)
//
// - Before:
//   catchpad ...
//   exn = wasm.get.exception();
//   selector = wasm.get.selector();
//   ...
//
// - After:
//   catchpad ...
//   exn = wasm.catch(WebAssembly::CPP_EXCEPTION);
//   // Only add below in case it's not a single catch (...)
//   wasm.landingpad.index(index);
//   __wasm_lpad_context.lpad_index = index;
//   __wasm_lpad_context.lsda = wasm.lsda();
//   __wasm_lpad_context.selector = 0;
//   personality_fn(1, _UA_SEARCH_PHASE, exn->exception_class, exn,
//     (struct _Unwind_Context *)&__wasm_lpad_context);
//   selector = __wasm_lpad_context.selector;
//   ...
//
//
// * Background: Direct personality function call
// In WebAssembly EH, the VM is responsible for unwinding the stack once an
// exception is thrown. After the stack is unwound, the control flow is
// transfered to WebAssembly 'catch' instruction.
//
// In Itanium EH, if the personality function decides there is no matching catch
// clause in a call frame and no cleanup action to perform, the unwinder doesn't
// stop there and continues unwinding. But in Wasm EH, the unwinder stops at
// every call frame with a catch intruction, after which the personality
// function is called from the compiler-generated user code here.
//
// In libunwind, we have this struct that serves as a communication channel
// between the compiler-generated user code and the personality function in
// libcxxabi.
//
// struct _Unwind_LandingPadContext {
//   uintptr_t lpad_index;
//   uintptr_t lsda;
//   uintptr_t selector;
// };
// struct _Unwind_LandingPadContext __wasm_lpad_context = ...;
//
// We pass a landing pad index, and the address of LSDA for the current function
// to the personality function, and we retrieve the selector after it returns.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/WasmEHPrepare.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/WasmEHFuncInfo.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-eh-prepare"

namespace {
class WasmEHPrepareImpl {
  friend class WasmEHPrepare;

  Type *LPadContextTy = nullptr; // type of 'struct _Unwind_LandingPadContext'
  Type *UnwindExceptionTy = nullptr;       // type of 'struct _Unwind_Exception'
  GlobalVariable *LPadContextGV = nullptr; // __wasm_lpad_context

  // Field addresses of struct _Unwind_LandingPadContext
  Value *LPadIndexField = nullptr; // lpad_index field
  Value *LSDAField = nullptr;      // lsda field
  Value *SelectorField = nullptr;  // selector

  Function *ThrowF = nullptr;       // wasm.throw() intrinsic
  Function *LPadIndexF = nullptr;   // wasm.landingpad.index() intrinsic
  Function *LSDAF = nullptr;        // wasm.lsda() intrinsic
  Function *GetExnF = nullptr;      // wasm.get.exception() intrinsic
  Function *CatchF = nullptr;       // wasm.catch() intrinsic
  Function *GetSelectorF = nullptr; // wasm.get.ehselector() intrinsic
  FunctionCallee PersonalityF = nullptr;

  bool prepareThrows(Function &F);
  bool prepareEHPads(Function &F);
  void prepareEHPad(BasicBlock *BB, bool NeedPersonality, unsigned Index = 0);

public:
  WasmEHPrepareImpl() = default;
  WasmEHPrepareImpl(Type *LPadContextTy_, Type *UnwindExceptionTy_)
      : LPadContextTy(LPadContextTy_), UnwindExceptionTy(UnwindExceptionTy_) {}
  bool runOnFunction(Function &F);
};

class WasmEHPrepare : public FunctionPass {
  WasmEHPrepareImpl P;

public:
  static char ID; // Pass identification, replacement for typeid

  WasmEHPrepare() : FunctionPass(ID) {}
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override { return P.runOnFunction(F); }

  StringRef getPassName() const override {
    return "WebAssembly Exception handling preparation";
  }
};

} // end anonymous namespace

PreservedAnalyses WasmEHPreparePass::run(Function &F,
                                         FunctionAnalysisManager &) {
  auto &Context = F.getContext();
  auto *I32Ty = Type::getInt32Ty(Context);
  auto *I64Ty = Type::getInt64Ty(Context);
  auto *PtrTy = PointerType::get(Context, 0);
  auto *LPadContextTy =
      StructType::get(I32Ty /*lpad_index*/, PtrTy /*lsda*/, I32Ty /*selector*/);
  auto *UnwindExceptionTy =
      StructType::get(I64Ty /* exception_class*/, PtrTy /* exception_cleanup */,
                      I32Ty /* private_1 */, I32Ty /* private_2 */);
  WasmEHPrepareImpl P(LPadContextTy, UnwindExceptionTy);
  bool Changed = P.runOnFunction(F);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses ::all();
}

char WasmEHPrepare::ID = 0;
INITIALIZE_PASS_BEGIN(WasmEHPrepare, DEBUG_TYPE,
                      "Prepare WebAssembly exceptions", false, false)
INITIALIZE_PASS_END(WasmEHPrepare, DEBUG_TYPE, "Prepare WebAssembly exceptions",
                    false, false)

FunctionPass *llvm::createWasmEHPass() { return new WasmEHPrepare(); }

bool WasmEHPrepare::doInitialization(Module &M) {
  IRBuilder<> IRB(M.getContext());
  auto *I32Ty = IRB.getInt32Ty();
  auto *I64Ty = IRB.getInt64Ty();
  auto *PtrTy = IRB.getPtrTy();
  P.LPadContextTy = StructType::get(I32Ty, // lpad_index
                                    PtrTy, // lsda
                                    I32Ty  // selector
  );
  P.UnwindExceptionTy = StructType::get(I64Ty, /* exception_class*/
                                        PtrTy  /* exception_cleanup */
  );
  return false;
}

// Erase the specified BBs if the BB does not have any remaining predecessors,
// and also all its dead children.
template <typename Container>
static void eraseDeadBBsAndChildren(const Container &BBs) {
  SmallVector<BasicBlock *, 8> WL(BBs.begin(), BBs.end());
  while (!WL.empty()) {
    auto *BB = WL.pop_back_val();
    if (!pred_empty(BB))
      continue;
    WL.append(succ_begin(BB), succ_end(BB));
    DeleteDeadBlock(BB);
  }
}

bool WasmEHPrepareImpl::runOnFunction(Function &F) {
  bool Changed = false;
  Changed |= prepareThrows(F);
  Changed |= prepareEHPads(F);
  return Changed;
}

bool WasmEHPrepareImpl::prepareThrows(Function &F) {
  Module &M = *F.getParent();
  IRBuilder<> IRB(F.getContext());
  bool Changed = false;

  // wasm.throw() intinsic, which will be lowered to wasm 'throw' instruction.
  ThrowF = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::wasm_throw);
  // Insert an unreachable instruction after a call to @llvm.wasm.throw and
  // delete all following instructions within the BB, and delete all the dead
  // children of the BB as well.
  for (User *U : ThrowF->users()) {
    auto *ThrowI = dyn_cast<CallInst>(U);
    if (!ThrowI || ThrowI->getFunction() != &F)
      continue;
    Changed = true;
    auto *BB = ThrowI->getParent();
    SmallVector<BasicBlock *, 4> Succs(successors(BB));
    BB->erase(std::next(BasicBlock::iterator(ThrowI)), BB->end());
    IRB.SetInsertPoint(BB);
    IRB.CreateUnreachable();
    eraseDeadBBsAndChildren(Succs);
  }

  return Changed;
}

bool WasmEHPrepareImpl::prepareEHPads(Function &F) {
  Module &M = *F.getParent();
  IRBuilder<> IRB(F.getContext());

  SmallVector<BasicBlock *, 16> CatchPads;
  SmallVector<BasicBlock *, 16> CleanupPads;
  for (BasicBlock &BB : F) {
    if (!BB.isEHPad())
      continue;
    BasicBlock::iterator Pad = BB.getFirstNonPHIIt();
    if (isa<CatchPadInst>(Pad))
      CatchPads.push_back(&BB);
    else if (isa<CleanupPadInst>(Pad))
      CleanupPads.push_back(&BB);
  }
  if (CatchPads.empty() && CleanupPads.empty())
    return false;

  if (!F.hasPersonalityFn())
    return false;

  auto Personality = classifyEHPersonality(F.getPersonalityFn());

  if (!isScopedEHPersonality(Personality)) {
    report_fatal_error("Function '" + F.getName() +
                       "' does not have a correct Wasm personality function "
                       "'__gxx_personality_wasm0'");
  }
  assert(F.hasPersonalityFn() && "Personality function not found");

  // __wasm_lpad_context global variable.
  // This variable should be thread local. If the target does not support TLS,
  // we depend on CoalesceFeaturesAndStripAtomics to downgrade it to
  // non-thread-local ones, in which case we don't allow this object to be
  // linked with other objects using shared memory.
  LPadContextGV = M.getOrInsertGlobal("__wasm_lpad_context", LPadContextTy);
  LPadContextGV->setThreadLocalMode(GlobalValue::GeneralDynamicTLSModel);

  LPadIndexField = LPadContextGV;
  LSDAField = IRB.CreateConstInBoundsGEP2_32(LPadContextTy, LPadContextGV, 0, 1,
                                             "lsda_gep");
  SelectorField = IRB.CreateConstInBoundsGEP2_32(LPadContextTy, LPadContextGV,
                                                 0, 2, "selector_gep");

  // wasm.landingpad.index() intrinsic, which is to specify landingpad index
  LPadIndexF =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::wasm_landingpad_index);
  // wasm.lsda() intrinsic. Returns the address of LSDA table for the current
  // function.
  LSDAF = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::wasm_lsda);
  // wasm.get.exception() and wasm.get.ehselector() intrinsics. Calls to these
  // are generated in clang.
  GetExnF =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::wasm_get_exception);
  GetSelectorF =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::wasm_get_ehselector);

  // wasm.catch() will be lowered down to wasm 'catch' instruction in
  // instruction selection.
  CatchF = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::wasm_catch);

  auto *I32Ty = IRB.getInt32Ty();
  auto *I64Ty = IRB.getInt64Ty();
  auto *PtrTy = IRB.getPtrTy();
  auto *PersPrototype =
      FunctionType::get(I32Ty, {I32Ty, I32Ty, I64Ty, PtrTy, PtrTy}, false);
  PersonalityF =
      M.getOrInsertFunction(getEHPersonalityName(Personality), PersPrototype);

  unsigned Index = 0;
  for (auto *BB : CatchPads) {
    auto *CPI = cast<CatchPadInst>(BB->getFirstNonPHIIt());
    // In case of a single catch (...), we don't need to emit a personalify
    // function call
    if (CPI->arg_size() == 1 &&
        cast<Constant>(CPI->getArgOperand(0))->isNullValue())
      prepareEHPad(BB, false);
    else
      prepareEHPad(BB, true, Index++);
  }

  // Cleanup pads don't need a personality function call.
  for (auto *BB : CleanupPads)
    prepareEHPad(BB, false);

  return true;
}

// Prepare an EH pad for Wasm EH handling. If NeedPersonality is false, Index is
// ignored.
void WasmEHPrepareImpl::prepareEHPad(BasicBlock *BB, bool NeedPersonality,
                                     unsigned Index) {
  assert(BB->isEHPad() && "BB is not an EHPad!");
  IRBuilder<> IRB(BB->getContext());
  IRB.SetInsertPoint(BB, BB->getFirstInsertionPt());

  auto *FPI = cast<FuncletPadInst>(BB->getFirstNonPHIIt());
  Instruction *GetExnCI = nullptr, *GetSelectorCI = nullptr;
  for (auto &U : FPI->uses()) {
    if (auto *CI = dyn_cast<CallInst>(U.getUser())) {
      if (CI->getCalledOperand() == GetExnF)
        GetExnCI = CI;
      if (CI->getCalledOperand() == GetSelectorF)
        GetSelectorCI = CI;
    }
  }

  // Cleanup pads do not have any of wasm.get.exception() or
  // wasm.get.ehselector() calls. We need to do nothing.
  if (!GetExnCI) {
    assert(!GetSelectorCI &&
           "wasm.get.ehselector() cannot exist w/o wasm.get.exception()");
    return;
  }

  // Replace wasm.get.exception intrinsic with wasm.catch intrinsic, which will
  // be lowered to wasm 'catch' instruction. We do this mainly because
  // instruction selection cannot handle wasm.get.exception intrinsic's token
  // argument.
  Instruction *CatchCI =
      IRB.CreateCall(CatchF, {IRB.getInt32(WebAssembly::CPP_EXCEPTION)}, "exn");
  GetExnCI->replaceAllUsesWith(CatchCI);
  GetExnCI->eraseFromParent();

  // In case it is a catchpad with single catch (...) or a cleanuppad, we don't
  // need to call personality function because we don't need a selector.
  if (!NeedPersonality) {
    if (GetSelectorCI) {
      assert(GetSelectorCI->use_empty() &&
             "wasm.get.ehselector() still has uses!");
      GetSelectorCI->eraseFromParent();
    }
    return;
  }
  IRB.SetInsertPoint(CatchCI->getNextNode());

  // This is to create a map of <landingpad EH label, landingpad index> in
  // SelectionDAGISel, which is to be used in EHStreamer to emit LSDA tables.
  // Pseudocode: wasm.landingpad.index(Index);
  IRB.CreateCall(LPadIndexF, {FPI, IRB.getInt32(Index)});

  // Pseudocode: __wasm_lpad_context.lpad_index = index;
  IRB.CreateStore(IRB.getInt32(Index), LPadIndexField);

  auto *CPI = cast<CatchPadInst>(FPI);
  // TODO Sometimes storing the LSDA address every time is not necessary, in
  // case it is already set in a dominating EH pad and there is no function call
  // between from that EH pad to here. Consider optimizing those cases.
  // Pseudocode: __wasm_lpad_context.lsda = wasm.lsda();
  IRB.CreateStore(IRB.CreateCall(LSDAF), LSDAField);

  // Pseudocode: __wasm_lpad_context.selector = 0;
  IRB.CreateStore(IRB.getInt32(0), SelectorField);

  // Pseudocode: &exn->exception_class
  auto *ExceptionClassPtr = IRB.CreateConstInBoundsGEP2_32(
      UnwindExceptionTy, CatchCI, 0, 0, "exception_class_gep");
  auto *ExceptionClass = IRB.CreateLoad(IRB.getInt64Ty(), ExceptionClassPtr);
  auto *UASearchPhase = IRB.getInt32(1);

  // Pseudocode:
  // personality_fn(1, _UA_SEARCH_PHASE, exn->exception_class, exn,
  //                (struct _Unwind_Context *)&__wasm_lpad_context);
  CallInst *PersCI = IRB.CreateCall(
      PersonalityF,
      {IRB.getInt32(1), UASearchPhase, ExceptionClass, CatchCI, LPadContextGV},
      OperandBundleDef("funclet", CPI));
  PersCI->setDoesNotThrow();

  // Pseudocode: int selector = __wasm_lpad_context.selector;
  Instruction *Selector =
      IRB.CreateLoad(IRB.getInt32Ty(), SelectorField, "selector");

  // Replace the return value from wasm.get.ehselector() with the selector value
  // loaded from __wasm_lpad_context.selector.
  assert(GetSelectorCI && "wasm.get.ehselector() call does not exist");
  GetSelectorCI->replaceAllUsesWith(Selector);
  GetSelectorCI->eraseFromParent();
}

void llvm::calculateWasmEHInfo(const Function *F, WasmEHFuncInfo &EHInfo) {
  // If an exception is not caught by a catchpad (i.e., it is a foreign
  // exception), it will unwind to its parent catchswitch's unwind destination.
  // We don't record an unwind destination for cleanuppads because every
  // exception should be caught by it.
  for (const auto &BB : *F) {
    if (!BB.isEHPad())
      continue;
    const Instruction *Pad = &*BB.getFirstNonPHIIt();

    if (const auto *CatchPad = dyn_cast<CatchPadInst>(Pad)) {
      const auto *UnwindBB = CatchPad->getCatchSwitch()->getUnwindDest();
      if (!UnwindBB)
        continue;
      const Instruction *UnwindPad = &*UnwindBB->getFirstNonPHIIt();
      if (const auto *CatchSwitch = dyn_cast<CatchSwitchInst>(UnwindPad))
        // Currently there should be only one handler per a catchswitch.
        EHInfo.setUnwindDest(&BB, *CatchSwitch->handlers().begin());
      else // cleanuppad
        EHInfo.setUnwindDest(&BB, UnwindBB);
    }
  }
}
