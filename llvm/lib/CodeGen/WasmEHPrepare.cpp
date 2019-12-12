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
//   exn = wasm.extract.exception();
//   // Only add below in case it's not a single catch (...)
//   wasm.landingpad.index(index);
//   __wasm_lpad_context.lpad_index = index;
//   __wasm_lpad_context.lsda = wasm.lsda();
//   _Unwind_CallPersonality(exn);
//   selector = __wasm.landingpad_context.selector;
//   ...
//
//
// * Background: Direct personality function call
// In WebAssembly EH, the VM is responsible for unwinding the stack once an
// exception is thrown. After the stack is unwound, the control flow is
// transfered to WebAssembly 'catch' instruction.
//
// Unwinding the stack is not done by libunwind but the VM, so the personality
// function in libcxxabi cannot be called from libunwind during the unwinding
// process. So after a catch instruction, we insert a call to a wrapper function
// in libunwind that in turn calls the real personality function.
//
// In Itanium EH, if the personality function decides there is no matching catch
// clause in a call frame and no cleanup action to perform, the unwinder doesn't
// stop there and continues unwinding. But in Wasm EH, the unwinder stops at
// every call frame with a catch intruction, after which the personality
// function is called from the compiler-generated user code here.
//
// In libunwind, we have this struct that serves as a communincation channel
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
// And this wrapper in libunwind calls the personality function.
//
// _Unwind_Reason_Code _Unwind_CallPersonality(void *exception_ptr) {
//   struct _Unwind_Exception *exception_obj =
//       (struct _Unwind_Exception *)exception_ptr;
//   _Unwind_Reason_Code ret = __gxx_personality_v0(
//       1, _UA_CLEANUP_PHASE, exception_obj->exception_class, exception_obj,
//       (struct _Unwind_Context *)__wasm_lpad_context);
//   return ret;
// }
//
// We pass a landing pad index, and the address of LSDA for the current function
// to the wrapper function _Unwind_CallPersonality in libunwind, and we retrieve
// the selector after it returns.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/WasmEHFuncInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "wasmehprepare"

namespace {
class WasmEHPrepare : public FunctionPass {
  Type *LPadContextTy = nullptr; // type of 'struct _Unwind_LandingPadContext'
  GlobalVariable *LPadContextGV = nullptr; // __wasm_lpad_context

  // Field addresses of struct _Unwind_LandingPadContext
  Value *LPadIndexField = nullptr; // lpad_index field
  Value *LSDAField = nullptr;      // lsda field
  Value *SelectorField = nullptr;  // selector

  Function *ThrowF = nullptr;       // wasm.throw() intrinsic
  Function *LPadIndexF = nullptr;   // wasm.landingpad.index() intrinsic
  Function *LSDAF = nullptr;        // wasm.lsda() intrinsic
  Function *GetExnF = nullptr;      // wasm.get.exception() intrinsic
  Function *ExtractExnF = nullptr;  // wasm.extract.exception() intrinsic
  Function *GetSelectorF = nullptr; // wasm.get.ehselector() intrinsic
  FunctionCallee CallPersonalityF =
      nullptr; // _Unwind_CallPersonality() wrapper

  bool prepareEHPads(Function &F);
  bool prepareThrows(Function &F);

  void prepareEHPad(BasicBlock *BB, bool NeedLSDA, unsigned Index = 0);
  void prepareTerminateCleanupPad(BasicBlock *BB);

public:
  static char ID; // Pass identification, replacement for typeid

  WasmEHPrepare() : FunctionPass(ID) {}

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override {
    return "WebAssembly Exception handling preparation";
  }
};
} // end anonymous namespace

char WasmEHPrepare::ID = 0;
INITIALIZE_PASS(WasmEHPrepare, DEBUG_TYPE, "Prepare WebAssembly exceptions",
                false, false)

FunctionPass *llvm::createWasmEHPass() { return new WasmEHPrepare(); }

bool WasmEHPrepare::doInitialization(Module &M) {
  IRBuilder<> IRB(M.getContext());
  LPadContextTy = StructType::get(IRB.getInt32Ty(),   // lpad_index
                                  IRB.getInt8PtrTy(), // lsda
                                  IRB.getInt32Ty()    // selector
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
    if (pred_begin(BB) != pred_end(BB))
      continue;
    WL.append(succ_begin(BB), succ_end(BB));
    DeleteDeadBlock(BB);
  }
}

bool WasmEHPrepare::runOnFunction(Function &F) {
  bool Changed = false;
  Changed |= prepareThrows(F);
  Changed |= prepareEHPads(F);
  return Changed;
}

bool WasmEHPrepare::prepareThrows(Function &F) {
  Module &M = *F.getParent();
  IRBuilder<> IRB(F.getContext());
  bool Changed = false;

  // wasm.throw() intinsic, which will be lowered to wasm 'throw' instruction.
  ThrowF = Intrinsic::getDeclaration(&M, Intrinsic::wasm_throw);
  // Insert an unreachable instruction after a call to @llvm.wasm.throw and
  // delete all following instructions within the BB, and delete all the dead
  // children of the BB as well.
  for (User *U : ThrowF->users()) {
    // A call to @llvm.wasm.throw() is only generated from __cxa_throw()
    // builtin call within libcxxabi, and cannot be an InvokeInst.
    auto *ThrowI = cast<CallInst>(U);
    if (ThrowI->getFunction() != &F)
      continue;
    Changed = true;
    auto *BB = ThrowI->getParent();
    SmallVector<BasicBlock *, 4> Succs(succ_begin(BB), succ_end(BB));
    auto &InstList = BB->getInstList();
    InstList.erase(std::next(BasicBlock::iterator(ThrowI)), InstList.end());
    IRB.SetInsertPoint(BB);
    IRB.CreateUnreachable();
    eraseDeadBBsAndChildren(Succs);
  }

  return Changed;
}

bool WasmEHPrepare::prepareEHPads(Function &F) {
  Module &M = *F.getParent();
  IRBuilder<> IRB(F.getContext());

  SmallVector<BasicBlock *, 16> CatchPads;
  SmallVector<BasicBlock *, 16> CleanupPads;
  for (BasicBlock &BB : F) {
    if (!BB.isEHPad())
      continue;
    auto *Pad = BB.getFirstNonPHI();
    if (isa<CatchPadInst>(Pad))
      CatchPads.push_back(&BB);
    else if (isa<CleanupPadInst>(Pad))
      CleanupPads.push_back(&BB);
  }

  if (CatchPads.empty() && CleanupPads.empty())
    return false;
  assert(F.hasPersonalityFn() && "Personality function not found");

  // __wasm_lpad_context global variable
  LPadContextGV = cast<GlobalVariable>(
      M.getOrInsertGlobal("__wasm_lpad_context", LPadContextTy));
  LPadIndexField = IRB.CreateConstGEP2_32(LPadContextTy, LPadContextGV, 0, 0,
                                          "lpad_index_gep");
  LSDAField =
      IRB.CreateConstGEP2_32(LPadContextTy, LPadContextGV, 0, 1, "lsda_gep");
  SelectorField = IRB.CreateConstGEP2_32(LPadContextTy, LPadContextGV, 0, 2,
                                         "selector_gep");

  // wasm.landingpad.index() intrinsic, which is to specify landingpad index
  LPadIndexF = Intrinsic::getDeclaration(&M, Intrinsic::wasm_landingpad_index);
  // wasm.lsda() intrinsic. Returns the address of LSDA table for the current
  // function.
  LSDAF = Intrinsic::getDeclaration(&M, Intrinsic::wasm_lsda);
  // wasm.get.exception() and wasm.get.ehselector() intrinsics. Calls to these
  // are generated in clang.
  GetExnF = Intrinsic::getDeclaration(&M, Intrinsic::wasm_get_exception);
  GetSelectorF = Intrinsic::getDeclaration(&M, Intrinsic::wasm_get_ehselector);

  // wasm.extract.exception() is the same as wasm.get.exception() but it does
  // not take a token argument. This will be lowered down to EXTRACT_EXCEPTION
  // pseudo instruction in instruction selection, which will be expanded using
  // 'br_on_exn' instruction later.
  ExtractExnF =
      Intrinsic::getDeclaration(&M, Intrinsic::wasm_extract_exception);

  // _Unwind_CallPersonality() wrapper function, which calls the personality
  CallPersonalityF = M.getOrInsertFunction(
      "_Unwind_CallPersonality", IRB.getInt32Ty(), IRB.getInt8PtrTy());
  if (Function *F = dyn_cast<Function>(CallPersonalityF.getCallee()))
    F->setDoesNotThrow();

  unsigned Index = 0;
  for (auto *BB : CatchPads) {
    auto *CPI = cast<CatchPadInst>(BB->getFirstNonPHI());
    // In case of a single catch (...), we don't need to emit LSDA
    if (CPI->getNumArgOperands() == 1 &&
        cast<Constant>(CPI->getArgOperand(0))->isNullValue())
      prepareEHPad(BB, false);
    else
      prepareEHPad(BB, true, Index++);
  }

  // Cleanup pads don't need LSDA.
  for (auto *BB : CleanupPads)
    prepareEHPad(BB, false);

  return true;
}

// Prepare an EH pad for Wasm EH handling. If NeedLSDA is false, Index is
// ignored.
void WasmEHPrepare::prepareEHPad(BasicBlock *BB, bool NeedLSDA,
                                 unsigned Index) {
  assert(BB->isEHPad() && "BB is not an EHPad!");
  IRBuilder<> IRB(BB->getContext());
  IRB.SetInsertPoint(&*BB->getFirstInsertionPt());

  auto *FPI = cast<FuncletPadInst>(BB->getFirstNonPHI());
  Instruction *GetExnCI = nullptr, *GetSelectorCI = nullptr;
  for (auto &U : FPI->uses()) {
    if (auto *CI = dyn_cast<CallInst>(U.getUser())) {
      if (CI->getCalledValue() == GetExnF)
        GetExnCI = CI;
      if (CI->getCalledValue() == GetSelectorF)
        GetSelectorCI = CI;
    }
  }

  // Cleanup pads w/o __clang_call_terminate call do not have any of
  // wasm.get.exception() or wasm.get.ehselector() calls. We need to do nothing.
  if (!GetExnCI) {
    assert(!GetSelectorCI &&
           "wasm.get.ehselector() cannot exist w/o wasm.get.exception()");
    return;
  }

  Instruction *ExtractExnCI = IRB.CreateCall(ExtractExnF, {}, "exn");
  GetExnCI->replaceAllUsesWith(ExtractExnCI);
  GetExnCI->eraseFromParent();

  // In case it is a catchpad with single catch (...) or a cleanuppad, we don't
  // need to call personality function because we don't need a selector.
  if (!NeedLSDA) {
    if (GetSelectorCI) {
      assert(GetSelectorCI->use_empty() &&
             "wasm.get.ehselector() still has uses!");
      GetSelectorCI->eraseFromParent();
    }
    return;
  }
  IRB.SetInsertPoint(ExtractExnCI->getNextNode());

  // This is to create a map of <landingpad EH label, landingpad index> in
  // SelectionDAGISel, which is to be used in EHStreamer to emit LSDA tables.
  // Pseudocode: wasm.landingpad.index(Index);
  IRB.CreateCall(LPadIndexF, {FPI, IRB.getInt32(Index)});

  // Pseudocode: __wasm_lpad_context.lpad_index = index;
  IRB.CreateStore(IRB.getInt32(Index), LPadIndexField);

  // Store LSDA address only if this catchpad belongs to a top-level
  // catchswitch. If there is another catchpad that dominates this pad, we don't
  // need to store LSDA address again, because they are the same throughout the
  // function and have been already stored before.
  // TODO Can we not store LSDA address in user function but make libcxxabi
  // compute it?
  auto *CPI = cast<CatchPadInst>(FPI);
  if (isa<ConstantTokenNone>(CPI->getCatchSwitch()->getParentPad()))
    // Pseudocode: __wasm_lpad_context.lsda = wasm.lsda();
    IRB.CreateStore(IRB.CreateCall(LSDAF), LSDAField);

  // Pseudocode: _Unwind_CallPersonality(exn);
  CallInst *PersCI = IRB.CreateCall(CallPersonalityF, ExtractExnCI,
                                    OperandBundleDef("funclet", CPI));
  PersCI->setDoesNotThrow();

  // Pseudocode: int selector = __wasm.landingpad_context.selector;
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
    const Instruction *Pad = BB.getFirstNonPHI();

    if (const auto *CatchPad = dyn_cast<CatchPadInst>(Pad)) {
      const auto *UnwindBB = CatchPad->getCatchSwitch()->getUnwindDest();
      if (!UnwindBB)
        continue;
      const Instruction *UnwindPad = UnwindBB->getFirstNonPHI();
      if (const auto *CatchSwitch = dyn_cast<CatchSwitchInst>(UnwindPad))
        // Currently there should be only one handler per a catchswitch.
        EHInfo.setEHPadUnwindDest(&BB, *CatchSwitch->handlers().begin());
      else // cleanuppad
        EHInfo.setEHPadUnwindDest(&BB, UnwindBB);
    }
  }
}
