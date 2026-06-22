//===- CoopTensorLowering.cpp - matmul2d / cooperative_tensor glue --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CoopTensorLowering.h"
#include "PointerRepairUtil.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace llvm {
namespace metal {

void retagTensorOpsExternallyDefined(Module &M) {
  for (auto &F : M)
    if (F.isDeclaration() && F.getName().starts_with("__tensorops_impl_") &&
        F.getSection().empty())
      F.setSection(kExternallyDefinedSection);
}

Type *requiredTensorArgPointee(StringRef Name, unsigned ArgNo,
                               LLVMContext &Ctx) {
  Type *I8 = Type::getInt8Ty(Ctx);
  auto handleTy = [&]() -> Type * {
    StructType *TT = StructType::getTypeByName(Ctx, "struct._tensor_t");
    if (!TT)
      TT = StructType::create(Ctx, "struct._tensor_t");
    return TT;
  };
  bool IsInitSliceExtent =
      Name.starts_with("air.init_strided_private_tensor") ||
      Name.starts_with("air.slice_private_tensor") ||
      Name.starts_with("air.get_extent_private_tensor");
  if (IsInitSliceExtent) {
    bool IsHandle =
        (ArgNo == 0) ||
        (ArgNo == 1 && Name.starts_with("air.slice_private_tensor"));
    return IsHandle ? handleTy() : I8;
  }
  if (Name.starts_with("air.get_descriptor_size_tensor"))
    return I8;
  if (Name.starts_with("__tensorops_impl_matmul2d")) {
    if (ArgNo == 1 || ArgNo == 3 || ArgNo == 5)
      return I8;
    return nullptr;
  }
  return nullptr;
}

Type *tensorHandlePointee(StringRef CalleeName, unsigned ArgNo, Value *Ptr,
                          LLVMContext &Ctx) {

  if (CalleeName.starts_with("__tensorops_impl_matmul2d")) {
    if (ArgNo == 0) {
      if (auto *AI = dyn_cast<AllocaInst>(Ptr->stripPointerCasts()))
        return AI->getAllocatedType();
    }
    return Type::getInt8Ty(Ctx);
  }
  return requiredTensorArgPointee(CalleeName, ArgNo, Ctx);
}

void fixTensorRuntimeArgTypes(Module &M, PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto Calls = collectInsts<CallInst>(M, [](CallInst *) { return true; });
  for (auto *CI : Calls) {
    Function *Callee = CI->getCalledFunction();
    if (!Callee)
      continue;
    StringRef Name = Callee->getName();
    for (unsigned J = 0; J < CI->arg_size(); ++J) {
      Type *Want = requiredTensorArgPointee(Name, J, Ctx);
      if (!Want)
        continue;
      Value *Arg = CI->getArgOperand(J);
      if (!Arg->getType()->isPointerTy() || isa<BitCastInst>(Arg))
        continue;
      Type *Pointee = nullptr;
      if (auto *AI = dyn_cast<AllocaInst>(Arg->stripPointerCasts()))
        Pointee = AI->getAllocatedType();
      else
        Pointee = effectivePointee(Arg, PTM);
      if (Pointee && Pointee != Want) {
        auto *BC = cast<BitCastInst>(CastInst::Create(
            Instruction::BitCast, Arg, Arg->getType(), "", CI->getIterator()));
        PTM.set(BC, Want);
        CI->setArgOperand(J, BC);
      }
    }
  }
}

}
}
