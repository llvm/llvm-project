//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// On arm64e, Clang emits ConstantPtrAuth expressions in global initializers
// to represent signed pointers. These are normally resolved by the dynamic
// linker, but LLDB's JIT does not run the linker, so they must be resolved
// manually. This pass replaces each ConstantPtrAuth in a global initializer
// with the unsigned pointer and emits a constructor function that signs the
// pointer at runtime using the ptrauth intrinsics.
//
// Example: given "static int (*fp)(int, int) = &mul;", Clang emits:
//
//   @fp = internal global ptr ptrauth (ptr @mul, i32 0)
//
// This pass transforms it into:
//
//   @fp = internal global ptr @mul
//   @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
//       [{ i32, ptr, ptr } { i32 0, ptr @ptrauth.sign, ptr null }]
//
//   define internal void @ptrauth.sign() {
//     %1 = load ptr, ptr @fp, align 8
//     %2 = ptrtoint ptr %1 to i64
//     %3 = call i64 @llvm.ptrauth.sign(i64 %2, i32 0, i64 0)
//     %4 = inttoptr i64 %3 to ptr
//     store ptr %4, ptr @fp, align 8
//     ret void
//   }
//
//===----------------------------------------------------------------------===//

#include "InjectPointerSigningFixups.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include "llvm/IR/GlobalPtrAuthInfo.h"

using namespace llvm;

namespace {
struct ExprStep {
  ConstantExpr *CE;
  unsigned OperandIdx;
};

/// Used to keep track of ConstantPtrAuth expressions in global initializers.
struct GlobalInitPtrAuthFixup {
  GlobalVariable *GV;
  ConstantPtrAuth *CPA;
  /// ConstantAggregate types are walekd via GEP indices.
  SmallVector<unsigned> GEPPath;
  /// ConstantExpr types are traversed via ExprStep (ConstantExpr + Operand
  /// index).
  SmallVector<ExprStep> ExprPath;
  GlobalInitPtrAuthFixup(GlobalVariable *GV, ConstantPtrAuth *CPA,
                         const SmallVectorImpl<unsigned> &GEPPath,
                         const SmallVectorImpl<ExprStep> &ExprPath)
      : GV(GV), CPA(CPA), GEPPath(GEPPath.begin(), GEPPath.end()),
        ExprPath(ExprPath.begin(), ExprPath.end()) {}
};

/// Used to keep track of extern_weak ConstantPtrAuth expressions inlined in
/// instructions.
struct WeakInlinePtrAuthFixup {
  Instruction *Inst;
  unsigned OperandIdx;
  ConstantPtrAuth *CPA;
};
} // namespace

/// Recursively walk a constant looking for ConstantPtrAuth expressions in
/// global initializers.
static void
findGlobalInitPtrAuth(Constant *C, GlobalVariable &GV,
                      SmallVectorImpl<unsigned> &GEPPath,
                      SmallVectorImpl<ExprStep> &ExprPath,
                      SmallVectorImpl<GlobalInitPtrAuthFixup> &Fixups) {
  if (auto *CPA = dyn_cast<ConstantPtrAuth>(C)) {
    Fixups.emplace_back(&GV, CPA, GEPPath, ExprPath);
    return;
  }
  if (isa<ConstantAggregate>(C)) {
    for (unsigned I = 0, E = C->getNumOperands(); I != E; ++I) {
      if (auto *COp = dyn_cast<Constant>(C->getOperand(I))) {
        GEPPath.push_back(I);
        findGlobalInitPtrAuth(COp, GV, GEPPath, ExprPath, Fixups);
        GEPPath.pop_back();
      }
    }
    return;
  }

  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    for (unsigned I = 0, E = C->getNumOperands(); I != E; ++I) {
      if (auto *COp = dyn_cast<Constant>(C->getOperand(I))) {
        ExprPath.push_back({CE, I});
        findGlobalInitPtrAuth(COp, GV, GEPPath, ExprPath, Fixups);
        ExprPath.pop_back();
      }
    }
  }
}

namespace {
struct PerModuleUtils {
  PerModuleUtils(Module &M) {
    Int32Ty = Type::getInt32Ty(M.getContext());
    IntPtrTy = Type::getInt64Ty(M.getContext());
    BlendIntrinsic =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ptrauth_blend);
    SignIntrinsic =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ptrauth_sign);

    FixupFunction = Function::Create(
        FunctionType::get(Type::getVoidTy(M.getContext()), false),
        GlobalValue::InternalLinkage, "lldb.arm64.sign_pointers", &M);
    FixupFunction->insert(FixupFunction->end(),
                          BasicBlock::Create(M.getContext()));
    B = std::make_unique<IRBuilder<>>(&FixupFunction->back());
  }

  Type *Int32Ty = nullptr;
  Type *IntPtrTy = nullptr;
  Function *BlendIntrinsic = nullptr;
  Function *SignIntrinsic = nullptr;
  Function *FixupFunction = nullptr;
  std::unique_ptr<IRBuilder<>> B;
};

struct PerGlobalUtils {
  PerGlobalUtils(GlobalPtrAuthInfo PtrAuthInfo)
      : PtrAuthInfo(std::move(PtrAuthInfo)) {
    GlobalVariable *AuthGlobal =
        const_cast<GlobalVariable *>(this->PtrAuthInfo.getGV());
    Pointee = const_cast<Constant *>(
        this->PtrAuthInfo.getPointer()->stripPointerCasts());
    PtrForInsts = new GlobalVariable(
        *AuthGlobal->getParent(), Pointee->getType(), false,
        GlobalValue::PrivateLinkage,
        ConstantExpr::getPointerCast(AuthGlobal, Pointee->getType()));
  }

  Instruction *getPtrLoadFor(Function &F, Type &T) {
    auto Key = std::make_pair(&F, &T);
    auto LI = LoadsForCaller.find(Key);
    if (LI == LoadsForCaller.end()) {
      BasicBlock &Entry = F.getEntryBlock();
      if (&T == PtrForInsts->getType()) {
        for (auto &I : Entry)
          if (!isa<AllocaInst>(I)) {
            auto *Load =
                new LoadInst(Pointee->getType(), PtrForInsts, Twine(), &I);
            LI = LoadsForCaller.insert(std::make_pair(Key, Load)).first;
            break;
          }
      } else {
        auto *OriginalLoad = getPtrLoadFor(F, *PtrForInsts->getType());
        auto *Cast = new BitCastInst(OriginalLoad, &T, "",
                                     &*std::next(OriginalLoad->getIterator()));
        LI = LoadsForCaller.insert(std::make_pair(Key, Cast)).first;
      }
    }

    assert(LI != LoadsForCaller.end() && "No load available/added");
    return LI->second;
  }

  GlobalPtrAuthInfo PtrAuthInfo;
  Constant *Pointee = nullptr;
  GlobalVariable *PtrForInsts = nullptr;
  std::map<std::pair<Function*, Type*>, Instruction*> LoadsForCaller;
};

Error makeStringError(const char *Msg, Value *V) {
  std::string ErrString;
  raw_string_ostream ErrStringStream(ErrString);
  ErrStringStream << Msg << " '" << *V << "'";
  return make_error<StringError>(ErrStringStream.str(),
                                 inconvertibleErrorCode());
}

Value *createStructGEP(PerModuleUtils &MUtils, GlobalVariable &V,
                       std::vector<uint64_t> Idxs) {
  Idxs.push_back(0);
  std::vector<Value *> GEPArgs;
  GEPArgs.reserve(Idxs.size());
  while (!Idxs.empty()) {
    GEPArgs.push_back(ConstantInt::get(MUtils.Int32Ty, Idxs.back()));
    Idxs.pop_back();
  }
  return MUtils.B->CreateGEP(V.getValueType(), &V, GEPArgs);
}

void processGlobalVariable(PerModuleUtils &MUtils, PerGlobalUtils &GUtils,
                           GlobalVariable &V, std::vector<uint64_t> Idxs) {
  auto &B = *MUtils.B;
  Value *Discriminator =
      const_cast<ConstantInt *>(GUtils.PtrAuthInfo.getDiscriminator());
  Value *PtrLoc =
      Idxs.empty() ? &V : createStructGEP(MUtils, V, std::move(Idxs));
  Type *PtrType =
      Idxs.empty() ? V.getType()
                   : GetElementPtrInst::getIndexedType(V.getValueType(), Idxs);

  if (GUtils.PtrAuthInfo.hasAddressDiversity())
    Discriminator = B.CreateCall(
        MUtils.BlendIntrinsic,
        {B.CreatePointerCast(PtrLoc, MUtils.IntPtrTy), Discriminator});

  Value *RawPtr = B.CreateLoad(PtrType, PtrLoc);
  Value *SignedPtr = B.CreateCall(
      MUtils.SignIntrinsic,
      {B.CreatePointerCast(RawPtr, MUtils.IntPtrTy),
       const_cast<ConstantInt *>(GUtils.PtrAuthInfo.getKey()), Discriminator});

  B.CreateStore(B.CreateBitOrPointerCast(SignedPtr, PtrType), PtrLoc);
}

void processInstruction(PerModuleUtils &MUtils, PerGlobalUtils &GUtils, Use &U,
                        std::vector<uint64_t> Idxs) {
  assert(Idxs.empty() &&
         "Accessing aggregate in instruction. Need a GEPExpr for this.");
  Instruction *V = cast<Instruction>(U.getUser());
  Function &F = *V->getParent()->getParent();
  Type *UseType = U.get()->getType();

  U.set(GUtils.getPtrLoadFor(F, *UseType));
}

Error processPtrAuthUsers(PerModuleUtils &MUtils, PerGlobalUtils &GUtils,
                          Use &U, std::vector<uint64_t> Idxs = {}) {
  Value *V = U.getUser();
  assert(V != nullptr);

  // Recurse through any casts.
  if (isa<ConstantExpr>(V) && cast<ConstantExpr>(V)->isCast()) {
    for (auto &U2 : V->uses())
      if (auto Err = processPtrAuthUsers(MUtils, GUtils, U2, Idxs))
        return Err;
  } else if (isa<ConstantAggregate>(V)) {
    Idxs.push_back(U.getOperandNo());
    for (auto &U2 : V->uses())
      if (auto Err = processPtrAuthUsers(MUtils, GUtils, U2, Idxs))
        return Err;
  } else if (isa<GlobalVariable>(V)) {
    processGlobalVariable(MUtils, GUtils, cast<GlobalVariable>(*V), Idxs);
  } else if (isa<Instruction>(V)) {
    processInstruction(MUtils, GUtils, U, Idxs);
  } else if (isa<ConstantExpr>(V)) {
    auto *VExpr = cast<ConstantExpr>(V);
    if (isa<GEPOperator>(VExpr)) {
      // We only support constant GEPs introduced when folding pointer casts.
      Type *VExprType = VExpr->getType();

      // Check that the types line up for a pointer cast.
      if (VExprType !=
          PointerType::get(GUtils.PtrAuthInfo.getPointer()->getType(), 0))
        return makeStringError("Type mismatch while rewriting ptrauth use", V);

      // Check that all indexes are constant zero.
      for (auto &Op :
           make_range(std::next(VExpr->op_begin()), VExpr->op_end())) {
        if (!isa<ConstantInt>(Op))
          return makeStringError(
              "Cannot rewrite ptrauth use with non-constant indexes", Op);
        if (!cast<ConstantInt>(Op)->isZero())
          return makeStringError(
              "Cannot rewrite ptrauth use with non-zero indexes", Op);
      }

      for (auto &U2 : VExpr->uses()) {
        if (auto Err = processPtrAuthUsers(MUtils, GUtils, U2, Idxs))
          return Err;
      }
    }
  } else
    return makeStringError("Unable to rewrite ptrauth use", V);

  return Error::success();
}

static Error
HandleLLVMPtrauthSection(llvm::Module &M,
                         lldb_private::ExecutionPolicy execution_policy) {
  std::vector<GlobalVariable *> PtrAuthVarsToDelete;
  PerModuleUtils MUtils(M);

  for (auto &G : M.globals()) {
    if (G.getSection() != "llvm.ptrauth")
      continue;

    PtrAuthVarsToDelete.push_back(&G);

    // If this ptrauth global is unused, skip it. The fixup pass could end up
    // introducing a real use of it otherwise.
    G.removeDeadConstantUsers();
    if (G.getNumUses() == 0)
      continue;

    auto PtrAuthInfo = GlobalPtrAuthInfo::tryAnalyze(&G);
    if (!PtrAuthInfo)
      return PtrAuthInfo.takeError();

    PerGlobalUtils GUtils(std::move(*PtrAuthInfo));
    for (auto &U : G.uses())
      if (auto Err = processPtrAuthUsers(MUtils, GUtils, U))
        return Err;

    // Replace all uses of the ptrauth global with the uses of the non-auth
    // global.
    G.replaceAllUsesWith(
        ConstantExpr::getPointerCast(GUtils.Pointee, G.getType()));
  }

  for (auto *G : PtrAuthVarsToDelete) {
    assert(G && G->user_empty() &&
           "All references to G should have been dropped");
    G->eraseFromParent();
  }

  // If we never wrote any fixup code, erase the fixup function and bail.
  if (MUtils.FixupFunction->getEntryBlock().empty()) {
    MUtils.FixupFunction->eraseFromParent();
    return Error::success();
  }

  // Close off the function.
  MUtils.B->CreateRetVoid();

  // Update the global ctor list to call the pointer fixup function first.
  auto *UInt8PtrTy =
      PointerType::getUnqual(llvm::Type::getInt8Ty(M.getContext()));
  StructType *CtorType = StructType::get(
      M.getContext(),
      {MUtils.Int32Ty, MUtils.FixupFunction->getType(), UInt8PtrTy});
  Constant *PtrFixupCtor = ConstantStruct::get(
      CtorType, {ConstantInt::get(MUtils.Int32Ty, 0), MUtils.FixupFunction,
                 Constant::getNullValue(UInt8PtrTy)});

  const char *LLVMGlobalCtorsName = "llvm.global_ctors";
  GlobalVariable *OldCtorList = M.getNamedGlobal(LLVMGlobalCtorsName);
  std::vector<Constant *> CtorListArgs;
  CtorListArgs.push_back(PtrFixupCtor);

  if (OldCtorList) {
    // If the old ctor list has any uses then bail out. Don't know how to
    // rewrite them.
    if (OldCtorList->getNumUses() != 0)
      return makeStringError("Global ctors variable has users, so can not be "
                             "rewritten to include pointer fixups: ",
                             OldCtorList);

    for (auto &Op : OldCtorList->getInitializer()->operands())
      CtorListArgs.push_back(cast<Constant>(Op.get()));
  }

  ArrayType *CtorListType = ArrayType::get(CtorType, CtorListArgs.size());
  Constant *CtorListInit = ConstantArray::get(CtorListType, CtorListArgs);

  GlobalVariable *NewCtorList = new GlobalVariable(
      M, CtorListType, false, GlobalValue::AppendingLinkage, CtorListInit);

  if (OldCtorList) {
    NewCtorList->takeName(OldCtorList);
    OldCtorList->eraseFromParent();
  } else
    NewCtorList->setName(LLVMGlobalCtorsName);

  return Error::success();
}
} // namespace

namespace lldb_private {

Error InjectPointerSigningFixupCode(llvm::Module &M,
                                    ExecutionPolicy execution_policy) {
  // If we cannot execute fixups, don't insert them.
  if (execution_policy == eExecutionPolicyNever)
    return Error::success();

  llvm::Triple T(M.getTargetTriple());

  // Bail out if we don't need pointer signing fixups.
  if (!T.isArm64e())
    return Error::success();

  // There is an older style of ptrauth support where there is a dedicated
  // global wrapper section called "llvm.ptrauth". If that is in use, we can
  // expect ConstantPointerAuth to not be involved. Detect that here and switch
  // implementations as needed. This should go away "eventually".
  bool ptrauth_section_found = false;
  for (auto &G : M.globals()) {
    if (G.getSection() == "llvm.ptrauth") {
      ptrauth_section_found = true;
      break;
    }
  }

  if (ptrauth_section_found)
    return HandleLLVMPtrauthSection(M, execution_policy);

  // Collect all ConstantPtrAuth expressions in global initializers.
  SmallVector<GlobalInitPtrAuthFixup> GlobalInitFixups;
  for (auto &G : M.globals()) {
    if (!G.hasInitializer())
      continue;
    SmallVector<unsigned> GEPPath;
    SmallVector<ExprStep> ExprPath;
    findGlobalInitPtrAuth(G.getInitializer(), G, GEPPath, ExprPath,
                          GlobalInitFixups);
  }

  // Collect all inline ConstantPtrAuth expressions for extern_weak globals in
  // functions.
  SmallVector<WeakInlinePtrAuthFixup> WeakInlineFixups;
  for (auto &F : M.functions()) {
    for (auto &BB : F) {
      for (auto &Inst : BB) {
        for (unsigned OpIdx = 0, E = Inst.getNumOperands(); OpIdx != E;
             OpIdx++) {
          auto *CPA = dyn_cast<ConstantPtrAuth>(Inst.getOperand(OpIdx));
          if (!CPA)
            continue;
          auto *GV = dyn_cast<GlobalValue>(CPA->getPointer());
          if (!GV || !GV->hasExternalWeakLinkage())
            continue;
          WeakInlineFixups.push_back({&Inst, OpIdx, CPA});
        }
      }
    }
  }

  if (GlobalInitFixups.empty() && WeakInlineFixups.empty())
    return Error::success();

  // Set up types and intrinsics.
  auto &Ctx = M.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *IntPtrTy = Type::getInt64Ty(Ctx);
  Function *BlendIntrinsic =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ptrauth_blend);
  Function *SignIntrinsic =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ptrauth_sign);

  // Create the fixup function.
  Function *FixupFn =
      Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                       GlobalValue::InternalLinkage, "ptrauth.sign", &M);
  FixupFn->insert(FixupFn->end(), BasicBlock::Create(Ctx));
  IRBuilder<> B(&FixupFn->back());

  for (auto &Fixup : GlobalInitFixups) {
    GlobalVariable *GV = Fixup.GV;
    ConstantPtrAuth *CPA = Fixup.CPA;

    // Null pointers must remain zero.
    if (isa<ConstantPointerNull>(CPA->getPointer())) {
      CPA->replaceAllUsesWith(CPA->getPointer());
      continue;
    }

    // Build a GEP to the location of the ConstantPtrAuth (or the expression
    // path to the ConstantPtrAuth) within the global.
    Value *Loc;
    if (Fixup.GEPPath.empty()) {
      Loc = GV;
    } else {
      SmallVector<Value *> GEPValues;
      GEPValues.push_back(ConstantInt::get(Int32Ty, 0));
      for (unsigned Idx : Fixup.GEPPath)
        GEPValues.push_back(ConstantInt::get(Int32Ty, Idx));
      Loc = B.CreateGEP(GV->getValueType(), GV, GEPValues);
    }

    Type *PtrTy = CPA->getType();

    // Compute the discriminator, blending with the address if needed.
    Value *Disc = CPA->getDiscriminator();
    if (CPA->hasAddressDiscriminator())
      Disc = B.CreateCall(BlendIntrinsic,
                          {B.CreatePointerCast(Loc, IntPtrTy), Disc});

    if (!Fixup.ExprPath.empty()) {
      // The CPA is wrapped in a ConstantExpr chain. Sign the CPA's pointer
      // directly and re-evaluate the expr chain.
      Value *SignedPtr = B.CreateCall(
          SignIntrinsic, {B.CreatePointerCast(CPA->getPointer(), IntPtrTy),
                          CPA->getKey(), Disc});
      Value *Result = B.CreateIntToPtr(SignedPtr, PtrTy);

      for (auto &Step : llvm::reverse(Fixup.ExprPath)) {
        Instruction *I = Step.CE->getAsInstruction();
        I->setOperand(Step.OperandIdx, Result);
        B.Insert(I);
        Result = I;
      }
      B.CreateStore(Result, Loc);
    } else {
      // There is no expression chain. Load and sign the pointer directly.
      Value *RawPtr = B.CreateLoad(PtrTy, Loc);
      Value *SignedPtr =
          B.CreateCall(SignIntrinsic, {B.CreatePointerCast(RawPtr, IntPtrTy),
                                       CPA->getKey(), Disc});
      B.CreateStore(B.CreateBitOrPointerCast(SignedPtr, PtrTy), Loc);
    }
    // Replace the ConstantPtrAuth in the initializer with the unsigned pointer.
    CPA->replaceAllUsesWith(CPA->getPointer());
  }

  // Close off the fixup function.
  B.CreateRetVoid();

  // Rewrite extern_weak inline CPA operands.
  for (auto &Fixup : WeakInlineFixups) {
    IRBuilder<> B(Fixup.Inst);
    ConstantPtrAuth *CPA = Fixup.CPA;
    Type *PtrTy = CPA->getType();

    Value *Disc = CPA->getDiscriminator();
    if (CPA->hasAddressDiscriminator()) {
      Value *AddrDisc =
          B.CreatePointerCast(CPA->getAddrDiscriminator(), IntPtrTy);
      Disc = B.CreateCall(BlendIntrinsic, {AddrDisc, Disc});
    }

    // Signing a pointer value of `0x0` yields a non-null but invalid pointer.
    // We'll emit a runtime guard around signing the pointer.
    Value *RawPtr = B.CreatePtrToInt(CPA->getPointer(), IntPtrTy);
    Value *NullCheck = B.CreateIsNull(RawPtr);
    Value *SignedPtr =
        B.CreateCall(SignIntrinsic, {RawPtr, CPA->getKey(), Disc});
    Value *Result =
        B.CreateSelect(NullCheck, Constant::getNullValue(IntPtrTy), SignedPtr);
    Fixup.Inst->setOperand(Fixup.OperandIdx, B.CreateIntToPtr(Result, PtrTy));
  }

  // Update the global ctors list to call the pointer fixup function first.
  auto *UInt8PtrTy = PointerType::getUnqual(Ctx);
  StructType *CtorType =
      StructType::get(Ctx, {Int32Ty, FixupFn->getType(), UInt8PtrTy});
  Constant *PtrFixupCtor =
      ConstantStruct::get(CtorType, {ConstantInt::get(Int32Ty, 0), FixupFn,
                                     Constant::getNullValue(UInt8PtrTy)});

  const char *LLVMGlobalCtorsName = "llvm.global_ctors";
  GlobalVariable *OldCtorList = M.getNamedGlobal(LLVMGlobalCtorsName);
  SmallVector<Constant *> CtorListArgs;
  CtorListArgs.push_back(PtrFixupCtor);

  if (OldCtorList) {
    // If the old ctors list has any uses then bail out: we do not know how to
    // rewrite them.
    if (OldCtorList->getNumUses() != 0) {
      std::string ErrStr;
      raw_string_ostream S(ErrStr);
      S << "Global ctors variable has users, so can not be rewritten to "
           "include pointer fixups: '"
        << *OldCtorList << "'";
      return make_error<StringError>(S.str(), inconvertibleErrorCode());
    }

    for (auto &Op : OldCtorList->getInitializer()->operands())
      CtorListArgs.push_back(cast<Constant>(Op.get()));
  }

  ArrayType *CtorListType = ArrayType::get(CtorType, CtorListArgs.size());
  Constant *CtorListInit = ConstantArray::get(CtorListType, CtorListArgs);

  GlobalVariable *NewCtorList = new GlobalVariable(
      M, CtorListType, false, GlobalValue::AppendingLinkage, CtorListInit);

  if (OldCtorList) {
    NewCtorList->takeName(OldCtorList);
    OldCtorList->eraseFromParent();
  } else
    NewCtorList->setName(LLVMGlobalCtorsName);

  return Error::success();
}

} // namespace lldb_private
