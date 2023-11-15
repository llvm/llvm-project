//===- Utility.cpp ------ Collection of geneirc offloading utilities ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Value.h"

using namespace llvm;
using namespace llvm::offloading;

// TODO: Export this to the linker wrapper code registration.
static StructType *getEntryTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *EntryTy =
      StructType::getTypeByName(C, "struct.__tgt_offload_entry");
  if (!EntryTy)
    EntryTy = StructType::create(
        "struct.__tgt_offload_entry", PointerType::getUnqual(C),
        PointerType::getUnqual(C), M.getDataLayout().getIntPtrType(C),
        Type::getInt32Ty(C), Type::getInt32Ty(C));
  return EntryTy;
}

// TODO: Rework this interface to be more generic.
void offloading::emitOffloadingEntry(Module &M, Constant *Addr, StringRef Name,
                                     uint64_t Size, int32_t Flags,
                                     StringRef SectionName) {
  Type *Int8PtrTy = PointerType::getUnqual(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *SizeTy = M.getDataLayout().getIntPtrType(M.getContext());

  Constant *AddrName = ConstantDataArray::getString(M.getContext(), Name);

  // Create the constant string used to look up the symbol in the device.
  auto *Str =
      new llvm::GlobalVariable(M, AddrName->getType(), /*isConstant=*/true,
                               llvm::GlobalValue::InternalLinkage, AddrName,
                               ".omp_offloading.entry_name");
  Str->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  // Construct the offloading entry.
  Constant *EntryData[] = {
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(Addr, Int8PtrTy),
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(Str, Int8PtrTy),
      ConstantInt::get(SizeTy, Size),
      ConstantInt::get(Int32Ty, Flags),
      ConstantInt::get(Int32Ty, 0),
  };
  Constant *EntryInitializer = ConstantStruct::get(getEntryTy(M), EntryData);

  auto *Entry = new GlobalVariable(
      M, getEntryTy(M),
      /*isConstant=*/true, GlobalValue::WeakAnyLinkage, EntryInitializer,
      ".omp_offloading.entry." + Name, nullptr, GlobalValue::NotThreadLocal,
      M.getDataLayout().getDefaultGlobalsAddressSpace());

  // The entry has to be created in the section the linker expects it to be.
  Entry->setSection(SectionName);
  Entry->setAlignment(Align(1));
}
