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
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace {
struct PtrAuthFixup {
  GlobalVariable *GV;
  ConstantPtrAuth *CPA;
  SmallVector<unsigned> Indices;
  PtrAuthFixup(GlobalVariable *GV, ConstantPtrAuth *CPA,
               const SmallVectorImpl<unsigned> &Indices)
      : GV(GV), CPA(CPA), Indices(Indices.begin(), Indices.end()) {}
};
} // namespace

/// Recursively walk a constant looking for ConstantPtrAuth expressions.
/// When found, record the global variable containing the ConstantPtrAuth and
/// the index path to reach it within the initializer.
static void findPtrAuth(Constant *C, GlobalVariable &GV,
                        SmallVectorImpl<unsigned> &Indices,
                        SmallVectorImpl<PtrAuthFixup> &Fixups) {
  if (auto *CPA = dyn_cast<ConstantPtrAuth>(C)) {
    Fixups.emplace_back(&GV, CPA, Indices);
    return;
  }
  for (unsigned I = 0, E = C->getNumOperands(); I != E; ++I) {
    if (auto *COp = dyn_cast<Constant>(C->getOperand(I))) {
      Indices.push_back(I);
      findPtrAuth(COp, GV, Indices, Fixups);
      Indices.pop_back();
    }
  }
}

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

  // Collect all ConstantPtrAuth expressions in global initializers.
  SmallVector<PtrAuthFixup> Fixups;
  for (auto &G : M.globals()) {
    if (!G.hasInitializer())
      continue;
    SmallVector<unsigned> Indices;
    findPtrAuth(G.getInitializer(), G, Indices, Fixups);
  }

  if (Fixups.empty())
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

  for (auto &Fixup : Fixups) {
    GlobalVariable *GV = Fixup.GV;
    ConstantPtrAuth *CPA = Fixup.CPA;

    // Null pointers must remain zero.
    if (isa<ConstantPointerNull>(CPA->getPointer())) {
      CPA->replaceAllUsesWith(CPA->getPointer());
      continue;
    }

    // Build a GEP to the location of the ConstantPtrAuth within the global.
    Value *Loc;
    if (Fixup.Indices.empty()) {
      Loc = GV;
    } else {
      SmallVector<Value *> GEPIndices;
      GEPIndices.push_back(ConstantInt::get(Int32Ty, 0));
      for (unsigned Idx : Fixup.Indices)
        GEPIndices.push_back(ConstantInt::get(Int32Ty, Idx));
      Loc = B.CreateGEP(GV->getValueType(), GV, GEPIndices);
    }

    Type *PtrTy = CPA->getType();

    // Load the raw (unsigned) pointer.
    Value *RawPtr = B.CreateLoad(PtrTy, Loc);

    // Compute the discriminator, blending with the address if needed.
    Value *Disc = CPA->getDiscriminator();
    if (CPA->hasAddressDiscriminator())
      Disc = B.CreateCall(BlendIntrinsic,
                          {B.CreatePointerCast(Loc, IntPtrTy), Disc});

    // Sign the pointer.
    Value *SignedPtr =
        B.CreateCall(SignIntrinsic, {B.CreatePointerCast(RawPtr, IntPtrTy),
                                     CPA->getKey(), Disc});

    // Store the signed pointer back.
    B.CreateStore(B.CreateBitOrPointerCast(SignedPtr, PtrTy), Loc);

    // Replace the ConstantPtrAuth in the initializer with the unsigned pointer.
    CPA->replaceAllUsesWith(CPA->getPointer());
  }

  // Close off the fixup function.
  B.CreateRetVoid();

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
