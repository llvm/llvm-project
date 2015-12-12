//===-- X86WinEHState - Insert EH state updates for win32 exceptions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// All functions using an MSVC EH personality use an explicitly updated state
// number stored in an exception registration stack object. The registration
// object is linked into a thread-local chain of registrations stored at fs:00.
// This pass adds the registration object and EH state updates.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "winehstate"

namespace llvm { void initializeWinEHStatePassPass(PassRegistry &); }

namespace {
class WinEHStatePass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid.

  WinEHStatePass() : FunctionPass(ID) {
    initializeWinEHStatePassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &Fn) override;

  bool doInitialization(Module &M) override;

  bool doFinalization(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  const char *getPassName() const override {
    return "Windows 32-bit x86 EH state insertion";
  }

private:
  void emitExceptionRegistrationRecord(Function *F);

  void linkExceptionRegistration(IRBuilder<> &Builder, Function *Handler);
  void unlinkExceptionRegistration(IRBuilder<> &Builder);
  void addStateStores(Function &F, WinEHFuncInfo &FuncInfo);
  void insertStateNumberStore(Value *ParentRegNode, Instruction *IP, int State);

  Value *emitEHLSDA(IRBuilder<> &Builder, Function *F);

  Function *generateLSDAInEAXThunk(Function *ParentFunc);

  // Module-level type getters.
  Type *getEHLinkRegistrationType();
  Type *getSEHRegistrationType();
  Type *getCXXEHRegistrationType();

  // Per-module data.
  Module *TheModule = nullptr;
  StructType *EHLinkRegistrationTy = nullptr;
  StructType *CXXEHRegistrationTy = nullptr;
  StructType *SEHRegistrationTy = nullptr;
  Function *FrameRecover = nullptr;
  Function *FrameAddress = nullptr;
  Function *FrameEscape = nullptr;
  Function *RestoreFrame = nullptr;

  // Per-function state
  EHPersonality Personality = EHPersonality::Unknown;
  Function *PersonalityFn = nullptr;

  /// The stack allocation containing all EH data, including the link in the
  /// fs:00 chain and the current state.
  AllocaInst *RegNode = nullptr;

  /// Struct type of RegNode. Used for GEPing.
  Type *RegNodeTy = nullptr;

  /// The index of the state field of RegNode.
  int StateFieldIndex = ~0U;

  /// The linked list node subobject inside of RegNode.
  Value *Link = nullptr;
};
}

FunctionPass *llvm::createX86WinEHStatePass() { return new WinEHStatePass(); }

char WinEHStatePass::ID = 0;

INITIALIZE_PASS(WinEHStatePass, "x86-winehstate",
                "Insert stores for EH state numbers", false, false)

bool WinEHStatePass::doInitialization(Module &M) {
  TheModule = &M;
  FrameEscape = Intrinsic::getDeclaration(TheModule, Intrinsic::localescape);
  FrameRecover = Intrinsic::getDeclaration(TheModule, Intrinsic::localrecover);
  FrameAddress = Intrinsic::getDeclaration(TheModule, Intrinsic::frameaddress);
  RestoreFrame =
      Intrinsic::getDeclaration(TheModule, Intrinsic::x86_seh_restoreframe);
  return false;
}

bool WinEHStatePass::doFinalization(Module &M) {
  assert(TheModule == &M);
  TheModule = nullptr;
  EHLinkRegistrationTy = nullptr;
  CXXEHRegistrationTy = nullptr;
  SEHRegistrationTy = nullptr;
  FrameEscape = nullptr;
  FrameRecover = nullptr;
  FrameAddress = nullptr;
  return false;
}

void WinEHStatePass::getAnalysisUsage(AnalysisUsage &AU) const {
  // This pass should only insert a stack allocation, memory accesses, and
  // localrecovers.
  AU.setPreservesCFG();
}

bool WinEHStatePass::runOnFunction(Function &F) {
  // Check the personality. Do nothing if this personality doesn't use funclets.
  if (!F.hasPersonalityFn())
    return false;
  PersonalityFn =
      dyn_cast<Function>(F.getPersonalityFn()->stripPointerCasts());
  if (!PersonalityFn)
    return false;
  Personality = classifyEHPersonality(PersonalityFn);
  if (!isFuncletEHPersonality(Personality))
    return false;

  // Skip this function if there are no EH pads and we aren't using IR-level
  // outlining.
  bool HasPads = false;
  for (BasicBlock &BB : F) {
    if (BB.isEHPad()) {
      HasPads = true;
      break;
    }
  }
  if (!HasPads)
    return false;

  // Disable frame pointer elimination in this function.
  // FIXME: Do the nested handlers need to keep the parent ebp in ebp, or can we
  // use an arbitrary register?
  F.addFnAttr("no-frame-pointer-elim", "true");

  emitExceptionRegistrationRecord(&F);

  // The state numbers calculated here in IR must agree with what we calculate
  // later on for the MachineFunction. In particular, if an IR pass deletes an
  // unreachable EH pad after this point before machine CFG construction, we
  // will be in trouble. If this assumption is ever broken, we should turn the
  // numbers into an immutable analysis pass.
  WinEHFuncInfo FuncInfo;
  addStateStores(F, FuncInfo);

  // Reset per-function state.
  PersonalityFn = nullptr;
  Personality = EHPersonality::Unknown;
  return true;
}

/// Get the common EH registration subobject:
///   typedef _EXCEPTION_DISPOSITION (*PEXCEPTION_ROUTINE)(
///       _EXCEPTION_RECORD *, void *, _CONTEXT *, void *);
///   struct EHRegistrationNode {
///     EHRegistrationNode *Next;
///     PEXCEPTION_ROUTINE Handler;
///   };
Type *WinEHStatePass::getEHLinkRegistrationType() {
  if (EHLinkRegistrationTy)
    return EHLinkRegistrationTy;
  LLVMContext &Context = TheModule->getContext();
  EHLinkRegistrationTy = StructType::create(Context, "EHRegistrationNode");
  Type *FieldTys[] = {
      EHLinkRegistrationTy->getPointerTo(0), // EHRegistrationNode *Next
      Type::getInt8PtrTy(Context) // EXCEPTION_DISPOSITION (*Handler)(...)
  };
  EHLinkRegistrationTy->setBody(FieldTys, false);
  return EHLinkRegistrationTy;
}

/// The __CxxFrameHandler3 registration node:
///   struct CXXExceptionRegistration {
///     void *SavedESP;
///     EHRegistrationNode SubRecord;
///     int32_t TryLevel;
///   };
Type *WinEHStatePass::getCXXEHRegistrationType() {
  if (CXXEHRegistrationTy)
    return CXXEHRegistrationTy;
  LLVMContext &Context = TheModule->getContext();
  Type *FieldTys[] = {
      Type::getInt8PtrTy(Context), // void *SavedESP
      getEHLinkRegistrationType(), // EHRegistrationNode SubRecord
      Type::getInt32Ty(Context)    // int32_t TryLevel
  };
  CXXEHRegistrationTy =
      StructType::create(FieldTys, "CXXExceptionRegistration");
  return CXXEHRegistrationTy;
}

/// The _except_handler3/4 registration node:
///   struct EH4ExceptionRegistration {
///     void *SavedESP;
///     _EXCEPTION_POINTERS *ExceptionPointers;
///     EHRegistrationNode SubRecord;
///     int32_t EncodedScopeTable;
///     int32_t TryLevel;
///   };
Type *WinEHStatePass::getSEHRegistrationType() {
  if (SEHRegistrationTy)
    return SEHRegistrationTy;
  LLVMContext &Context = TheModule->getContext();
  Type *FieldTys[] = {
      Type::getInt8PtrTy(Context), // void *SavedESP
      Type::getInt8PtrTy(Context), // void *ExceptionPointers
      getEHLinkRegistrationType(), // EHRegistrationNode SubRecord
      Type::getInt32Ty(Context),   // int32_t EncodedScopeTable
      Type::getInt32Ty(Context)    // int32_t TryLevel
  };
  SEHRegistrationTy = StructType::create(FieldTys, "SEHExceptionRegistration");
  return SEHRegistrationTy;
}

// Emit an exception registration record. These are stack allocations with the
// common subobject of two pointers: the previous registration record (the old
// fs:00) and the personality function for the current frame. The data before
// and after that is personality function specific.
void WinEHStatePass::emitExceptionRegistrationRecord(Function *F) {
  assert(Personality == EHPersonality::MSVC_CXX ||
         Personality == EHPersonality::MSVC_X86SEH);

  StringRef PersonalityName = PersonalityFn->getName();
  IRBuilder<> Builder(&F->getEntryBlock(), F->getEntryBlock().begin());
  Type *Int8PtrType = Builder.getInt8PtrTy();
  if (Personality == EHPersonality::MSVC_CXX) {
    RegNodeTy = getCXXEHRegistrationType();
    RegNode = Builder.CreateAlloca(RegNodeTy);
    // SavedESP = llvm.stacksave()
    Value *SP = Builder.CreateCall(
        Intrinsic::getDeclaration(TheModule, Intrinsic::stacksave), {});
    Builder.CreateStore(SP, Builder.CreateStructGEP(RegNodeTy, RegNode, 0));
    // TryLevel = -1
    StateFieldIndex = 2;
    insertStateNumberStore(RegNode, &*Builder.GetInsertPoint(), -1);
    // Handler = __ehhandler$F
    Function *Trampoline = generateLSDAInEAXThunk(F);
    Link = Builder.CreateStructGEP(RegNodeTy, RegNode, 1);
    linkExceptionRegistration(Builder, Trampoline);
  } else if (Personality == EHPersonality::MSVC_X86SEH) {
    // If _except_handler4 is in use, some additional guard checks and prologue
    // stuff is required.
    bool UseStackGuard = (PersonalityName == "_except_handler4");
    RegNodeTy = getSEHRegistrationType();
    RegNode = Builder.CreateAlloca(RegNodeTy);
    // SavedESP = llvm.stacksave()
    Value *SP = Builder.CreateCall(
        Intrinsic::getDeclaration(TheModule, Intrinsic::stacksave), {});
    Builder.CreateStore(SP, Builder.CreateStructGEP(RegNodeTy, RegNode, 0));
    // TryLevel = -2 / -1
    StateFieldIndex = 4;
    insertStateNumberStore(RegNode, &*Builder.GetInsertPoint(),
                           UseStackGuard ? -2 : -1);
    // ScopeTable = llvm.x86.seh.lsda(F)
    Value *FI8 = Builder.CreateBitCast(F, Int8PtrType);
    Value *LSDA = Builder.CreateCall(
        Intrinsic::getDeclaration(TheModule, Intrinsic::x86_seh_lsda), FI8);
    Type *Int32Ty = Type::getInt32Ty(TheModule->getContext());
    LSDA = Builder.CreatePtrToInt(LSDA, Int32Ty);
    // If using _except_handler4, xor the address of the table with
    // __security_cookie.
    if (UseStackGuard) {
      Value *Cookie =
          TheModule->getOrInsertGlobal("__security_cookie", Int32Ty);
      Value *Val = Builder.CreateLoad(Int32Ty, Cookie);
      LSDA = Builder.CreateXor(LSDA, Val);
    }
    Builder.CreateStore(LSDA, Builder.CreateStructGEP(RegNodeTy, RegNode, 3));
    Link = Builder.CreateStructGEP(RegNodeTy, RegNode, 2);
    linkExceptionRegistration(Builder, PersonalityFn);
  } else {
    llvm_unreachable("unexpected personality function");
  }

  // Insert an unlink before all returns.
  for (BasicBlock &BB : *F) {
    TerminatorInst *T = BB.getTerminator();
    if (!isa<ReturnInst>(T))
      continue;
    Builder.SetInsertPoint(T);
    unlinkExceptionRegistration(Builder);
  }
}

Value *WinEHStatePass::emitEHLSDA(IRBuilder<> &Builder, Function *F) {
  Value *FI8 = Builder.CreateBitCast(F, Type::getInt8PtrTy(F->getContext()));
  return Builder.CreateCall(
      Intrinsic::getDeclaration(TheModule, Intrinsic::x86_seh_lsda), FI8);
}

/// Generate a thunk that puts the LSDA of ParentFunc in EAX and then calls
/// PersonalityFn, forwarding the parameters passed to PEXCEPTION_ROUTINE:
///   typedef _EXCEPTION_DISPOSITION (*PEXCEPTION_ROUTINE)(
///       _EXCEPTION_RECORD *, void *, _CONTEXT *, void *);
/// We essentially want this code:
///   movl $lsda, %eax
///   jmpl ___CxxFrameHandler3
Function *WinEHStatePass::generateLSDAInEAXThunk(Function *ParentFunc) {
  LLVMContext &Context = ParentFunc->getContext();
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int8PtrType = Type::getInt8PtrTy(Context);
  Type *ArgTys[5] = {Int8PtrType, Int8PtrType, Int8PtrType, Int8PtrType,
                     Int8PtrType};
  FunctionType *TrampolineTy =
      FunctionType::get(Int32Ty, makeArrayRef(&ArgTys[0], 4),
                        /*isVarArg=*/false);
  FunctionType *TargetFuncTy =
      FunctionType::get(Int32Ty, makeArrayRef(&ArgTys[0], 5),
                        /*isVarArg=*/false);
  Function *Trampoline =
      Function::Create(TrampolineTy, GlobalValue::InternalLinkage,
                       Twine("__ehhandler$") + GlobalValue::getRealLinkageName(
                                                   ParentFunc->getName()),
                       TheModule);
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", Trampoline);
  IRBuilder<> Builder(EntryBB);
  Value *LSDA = emitEHLSDA(Builder, ParentFunc);
  Value *CastPersonality =
      Builder.CreateBitCast(PersonalityFn, TargetFuncTy->getPointerTo());
  auto AI = Trampoline->arg_begin();
  Value *Args[5] = {LSDA, &*AI++, &*AI++, &*AI++, &*AI++};
  CallInst *Call = Builder.CreateCall(CastPersonality, Args);
  // Can't use musttail due to prototype mismatch, but we can use tail.
  Call->setTailCall(true);
  // Set inreg so we pass it in EAX.
  Call->addAttribute(1, Attribute::InReg);
  Builder.CreateRet(Call);
  return Trampoline;
}

void WinEHStatePass::linkExceptionRegistration(IRBuilder<> &Builder,
                                               Function *Handler) {
  // Emit the .safeseh directive for this function.
  Handler->addFnAttr("safeseh");

  Type *LinkTy = getEHLinkRegistrationType();
  // Handler = Handler
  Value *HandlerI8 = Builder.CreateBitCast(Handler, Builder.getInt8PtrTy());
  Builder.CreateStore(HandlerI8, Builder.CreateStructGEP(LinkTy, Link, 1));
  // Next = [fs:00]
  Constant *FSZero =
      Constant::getNullValue(LinkTy->getPointerTo()->getPointerTo(257));
  Value *Next = Builder.CreateLoad(FSZero);
  Builder.CreateStore(Next, Builder.CreateStructGEP(LinkTy, Link, 0));
  // [fs:00] = Link
  Builder.CreateStore(Link, FSZero);
}

void WinEHStatePass::unlinkExceptionRegistration(IRBuilder<> &Builder) {
  // Clone Link into the current BB for better address mode folding.
  if (auto *GEP = dyn_cast<GetElementPtrInst>(Link)) {
    GEP = cast<GetElementPtrInst>(GEP->clone());
    Builder.Insert(GEP);
    Link = GEP;
  }
  Type *LinkTy = getEHLinkRegistrationType();
  // [fs:00] = Link->Next
  Value *Next =
      Builder.CreateLoad(Builder.CreateStructGEP(LinkTy, Link, 0));
  Constant *FSZero =
      Constant::getNullValue(LinkTy->getPointerTo()->getPointerTo(257));
  Builder.CreateStore(Next, FSZero);
}

void WinEHStatePass::addStateStores(Function &F, WinEHFuncInfo &FuncInfo) {
  // Mark the registration node. The backend needs to know which alloca it is so
  // that it can recover the original frame pointer.
  IRBuilder<> Builder(RegNode->getParent(), std::next(RegNode->getIterator()));
  Value *RegNodeI8 = Builder.CreateBitCast(RegNode, Builder.getInt8PtrTy());
  Builder.CreateCall(
      Intrinsic::getDeclaration(TheModule, Intrinsic::x86_seh_ehregnode),
      {RegNodeI8});

  // Calculate state numbers.
  if (isAsynchronousEHPersonality(Personality))
    calculateSEHStateNumbers(&F, FuncInfo);
  else
    calculateWinCXXEHStateNumbers(&F, FuncInfo);

  // Iterate all the instructions and emit state number stores.
  DenseMap<BasicBlock *, ColorVector> BlockColors = colorEHFunclets(F);
  for (BasicBlock &BB : F) {
    // Figure out what state we should assign calls in this block.
    int BaseState = -1;
    auto &BBColors = BlockColors[&BB];

    assert(BBColors.size() == 1 &&
           "multi-color BB not removed by preparation");
    BasicBlock *FuncletEntryBB = BBColors.front();
    if (auto *FuncletPad =
            dyn_cast<FuncletPadInst>(FuncletEntryBB->getFirstNonPHI())) {
      auto BaseStateI = FuncInfo.FuncletBaseStateMap.find(FuncletPad);
      if (BaseStateI != FuncInfo.FuncletBaseStateMap.end())
        BaseState = BaseStateI->second;
    }

    for (Instruction &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        // Possibly throwing call instructions have no actions to take after
        // an unwind. Ensure they are in the -1 state.
        if (CI->doesNotThrow())
          continue;
        insertStateNumberStore(RegNode, CI, BaseState);
      } else if (auto *II = dyn_cast<InvokeInst>(&I)) {
        // Look up the state number of the landingpad this unwinds to.
        assert(FuncInfo.InvokeStateMap.count(II) && "invoke has no state!");
        int State = FuncInfo.InvokeStateMap[II];
        insertStateNumberStore(RegNode, II, State);
      }
    }
  }
}

void WinEHStatePass::insertStateNumberStore(Value *ParentRegNode,
                                            Instruction *IP, int State) {
  IRBuilder<> Builder(IP);
  Value *StateField =
      Builder.CreateStructGEP(RegNodeTy, ParentRegNode, StateFieldIndex);
  Builder.CreateStore(Builder.getInt32(State), StateField);
}
