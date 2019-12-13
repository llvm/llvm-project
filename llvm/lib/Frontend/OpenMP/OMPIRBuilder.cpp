//===- OpenMPIRBuilder.cpp - Builder for LLVM-IR for OpenMP directives ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the OpenMPIRBuilder class, which is used as a
/// convenient way to create LLVM instructions for OpenMP directives.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <sstream>

#define DEBUG_TYPE "openmp-ir-builder"

using namespace llvm;
using namespace omp;
using namespace types;

static cl::opt<bool>
    OptimisticAttributes("openmp-ir-builder-optimistic-attributes", cl::Hidden,
                         cl::desc("Use optimistic attributes describing "
                                  "'as-if' properties of runtime calls."),
                         cl::init(false));

void OpenMPIRBuilder::addAttributes(omp::RuntimeFunction FnID, Function &Fn) {
  LLVMContext &Ctx = Fn.getContext();

#define OMP_ATTRS_SET(VarName, AttrSet) AttributeSet VarName = AttrSet;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

  // Add attributes to the new declaration.
  switch (FnID) {
#define OMP_RTL_ATTRS(Enum, FnAttrSet, RetAttrSet, ArgAttrSets)                \
  case Enum:                                                                   \
    Fn.setAttributes(                                                          \
        AttributeList::get(Ctx, FnAttrSet, RetAttrSet, ArgAttrSets));          \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  default:
    // Attributes are optional.
    break;
  }
}

Function *OpenMPIRBuilder::getOrCreateRuntimeFunction(RuntimeFunction FnID) {
  Function *Fn = nullptr;

  // Try to find the declation in the module first.
  switch (FnID) {
#define OMP_RTL(Enum, Str, IsVarArg, ReturnType, ...)                          \
  case Enum:                                                                   \
    Fn = M.getFunction(Str);                                                   \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }

  if (!Fn) {
    // Create a new declaration if we need one.
    switch (FnID) {
#define OMP_RTL(Enum, Str, IsVarArg, ReturnType, ...)                          \
  case Enum:                                                                   \
    Fn = Function::Create(FunctionType::get(ReturnType,                        \
                                            ArrayRef<Type *>{__VA_ARGS__},     \
                                            IsVarArg),                         \
                          GlobalValue::ExternalLinkage, Str, M);               \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
    }

    addAttributes(FnID, *Fn);
  }

  assert(Fn && "Failed to create OpenMP runtime function");
  return Fn;
}

void OpenMPIRBuilder::initialize() { initializeTypes(M); }

Value *OpenMPIRBuilder::getOrCreateIdent(Constant *SrcLocStr,
                                         IdentFlag LocFlags) {
  // Enable "C-mode".
  LocFlags |= OMP_IDENT_FLAG_KMPC;

  GlobalVariable *&DefaultIdent = IdentMap[{SrcLocStr, uint64_t(LocFlags)}];
  if (!DefaultIdent) {
    Constant *I32Null = ConstantInt::getNullValue(Int32);
    Constant *IdentData[] = {I32Null,
                             ConstantInt::get(Int32, uint64_t(LocFlags)),
                             I32Null, I32Null, SrcLocStr};
    Constant *Initializer = ConstantStruct::get(
        cast<StructType>(IdentPtr->getPointerElementType()), IdentData);

    // Look for existing encoding of the location + flags, not needed but
    // minimizes the difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.getGlobalList())
      if (GV.getType() == IdentPtr && GV.hasInitializer())
        if (GV.getInitializer() == Initializer)
          return DefaultIdent = &GV;

    DefaultIdent = new GlobalVariable(M, IdentPtr->getPointerElementType(),
                                      /* isConstant = */ false,
                                      GlobalValue::PrivateLinkage, Initializer);
    DefaultIdent->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    DefaultIdent->setAlignment(Align(8));
  }
  return DefaultIdent;
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(StringRef LocStr) {
  Constant *&SrcLocStr = SrcLocStrMap[LocStr];
  if (!SrcLocStr) {
    Constant *Initializer =
        ConstantDataArray::getString(M.getContext(), LocStr);

    // Look for existing encoding of the location, not needed but minimizes the
    // difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.getGlobalList())
      if (GV.isConstant() && GV.hasInitializer() &&
          GV.getInitializer() == Initializer)
        return SrcLocStr = ConstantExpr::getPointerCast(&GV, Int8Ptr);

    SrcLocStr = Builder.CreateGlobalStringPtr(LocStr);
  }
  return SrcLocStr;
}

Constant *OpenMPIRBuilder::getOrCreateDefaultSrcLocStr() {
  return getOrCreateSrcLocStr(";unknown;unknown;0;0;;");
}

Constant *
OpenMPIRBuilder::getOrCreateSrcLocStr(const LocationDescription &Loc) {
  DILocation *DIL = Loc.DL.get();
  if (!DIL)
    return getOrCreateDefaultSrcLocStr();
  StringRef Filename =
      !DIL->getFilename().empty() ? DIL->getFilename() : M.getName();
  StringRef Function = DIL->getScope()->getSubprogram()->getName();
  Function =
      !Function.empty() ? Function : Loc.IP.getBlock()->getParent()->getName();
  std::string LineStr = std::to_string(DIL->getLine());
  std::string ColumnStr = std::to_string(DIL->getColumn());
  std::stringstream SrcLocStr;
  SrcLocStr << ";" << Filename.data() << ";" << Function.data() << ";"
            << LineStr << ";" << ColumnStr << ";;";
  return getOrCreateSrcLocStr(SrcLocStr.str());
}

Value *OpenMPIRBuilder::getOrCreateThreadID(Value *Ident) {
  return Builder.CreateCall(
      getOrCreateRuntimeFunction(OMPRTL___kmpc_global_thread_num), Ident,
      "omp_global_thread_num");
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::CreateBarrier(const LocationDescription &Loc, Directive DK,
                               bool ForceSimpleCall, bool CheckCancelFlag) {
  if (!updateToLocation(Loc))
    return Loc.IP;
  return emitBarrierImpl(Loc, DK, ForceSimpleCall, CheckCancelFlag);
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::emitBarrierImpl(const LocationDescription &Loc, Directive Kind,
                                 bool ForceSimpleCall, bool CheckCancelFlag) {
  // Build call __kmpc_cancel_barrier(loc, thread_id) or
  //            __kmpc_barrier(loc, thread_id);

  IdentFlag BarrierLocFlags;
  switch (Kind) {
  case OMPD_for:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_FOR;
    break;
  case OMPD_sections:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_SECTIONS;
    break;
  case OMPD_single:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_SINGLE;
    break;
  case OMPD_barrier:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_EXPL;
    break;
  default:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL;
    break;
  }

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Args[] = {getOrCreateIdent(SrcLocStr, BarrierLocFlags),
                   getOrCreateThreadID(getOrCreateIdent(SrcLocStr))};

  // If we are in a cancellable parallel region, barriers are cancellation
  // points.
  // TODO: Check why we would force simple calls or to ignore the cancel flag.
  bool UseCancelBarrier = !ForceSimpleCall && CancellationBlock;

  Value *Result = Builder.CreateCall(
      getOrCreateRuntimeFunction(UseCancelBarrier ? OMPRTL___kmpc_cancel_barrier
                                                  : OMPRTL___kmpc_barrier),
      Args);

  if (UseCancelBarrier && CheckCancelFlag) {
    // For a cancel barrier we create two new blocks.
    BasicBlock *BB = Builder.GetInsertBlock();
    BasicBlock *NonCancellationBlock = BasicBlock::Create(
        BB->getContext(), BB->getName() + ".cont", BB->getParent());

    // Jump to them based on the return value.
    Value *Cmp = Builder.CreateIsNull(Result);
    Builder.CreateCondBr(Cmp, NonCancellationBlock, CancellationBlock,
                         /* TODO weight */ nullptr, nullptr);

    Builder.SetInsertPoint(NonCancellationBlock);
    assert(CancellationBlock->getParent() == BB->getParent() &&
           "Unexpected cancellation block parent!");

    // TODO: This is a workaround for now, we always reset the cancellation
    // block until we manage it ourselves here.
    CancellationBlock = nullptr;
  }

  return Builder.saveIP();
}
