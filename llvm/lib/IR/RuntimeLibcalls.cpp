//===- RuntimeLibcalls.cpp - Interface for runtime libcalls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/xxhash.h"
#include "llvm/TargetParser/ARMTargetParser.h"

#define DEBUG_TYPE "runtime-libcalls-info"

using namespace llvm;
using namespace RTLIB;

#define GET_RUNTIME_LIBCALLS_INFO
#define GET_INIT_RUNTIME_LIBCALL_NAMES
#define GET_SET_TARGET_RUNTIME_LIBCALL_SETS
#define DEFINE_GET_LOOKUP_LIBCALL_IMPL_NAME
#include "llvm/IR/RuntimeLibcalls.inc"

RuntimeLibcallsInfo::RuntimeLibcallsInfo(const Module &M)
    : RuntimeLibcallsInfo(M.getTargetTriple()) {
  // TODO: Consider module flags
}

/// Set default libcall names. If a target wants to opt-out of a libcall it
/// should be placed here.
void RuntimeLibcallsInfo::initLibcalls(const Triple &TT,
                                       ExceptionHandling ExceptionModel,
                                       FloatABI::ABIType FloatABI,
                                       EABI EABIVersion, StringRef ABIName) {
  setTargetRuntimeLibcallSets(TT, ExceptionModel, FloatABI, EABIVersion,
                              ABIName);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
iota_range<RTLIB::LibcallImpl>
RuntimeLibcallsInfo::libcallImplNameHit(uint16_t NameOffsetEntry,
                                        uint16_t StrOffset) {
  int NumAliases = 1;
  for (uint16_t Entry : ArrayRef(RuntimeLibcallNameOffsetTable)
                            .drop_front(NameOffsetEntry + 1)) {
    if (Entry != StrOffset)
      break;
    ++NumAliases;
  }

  RTLIB::LibcallImpl ImplStart = static_cast<RTLIB::LibcallImpl>(
      &RuntimeLibcallNameOffsetTable[NameOffsetEntry] -
      &RuntimeLibcallNameOffsetTable[0]);
  return enum_seq(ImplStart,
                  static_cast<RTLIB::LibcallImpl>(ImplStart + NumAliases));
}

bool RuntimeLibcallsInfo::isAAPCS_ABI(const Triple &TT, StringRef ABIName) {
  const ARM::ARMABI TargetABI = ARM::computeTargetABI(TT, ABIName);
  return TargetABI == ARM::ARM_ABI_AAPCS || TargetABI == ARM::ARM_ABI_AAPCS16;
}

bool RuntimeLibcallsInfo::darwinHasExp10(const Triple &TT) {
  switch (TT.getOS()) {
  case Triple::MacOSX:
    return !TT.isMacOSXVersionLT(10, 9);
  case Triple::IOS:
    return !TT.isOSVersionLT(7, 0);
  case Triple::DriverKit:
  case Triple::TvOS:
  case Triple::WatchOS:
  case Triple::XROS:
  case Triple::BridgeOS:
    return true;
  default:
    return false;
  }
}

std::pair<FunctionType *, AttributeList>
RuntimeLibcallsInfo::getFunctionTy(LLVMContext &Ctx, const Triple &TT,
                                   const DataLayout &DL,
                                   RTLIB::LibcallImpl LibcallImpl) const {
  static constexpr Attribute::AttrKind CommonFnAttrs[] = {
      Attribute::NoCallback, Attribute::NoFree, Attribute::NoSync,
      Attribute::NoUnwind, Attribute::WillReturn};

  switch (LibcallImpl) {
  case RTLIB::impl___sincos_stret:
  case RTLIB::impl___sincosf_stret: {
    if (!darwinHasSinCosStret(TT)) // Non-darwin currently unexpected
      return {};

    Type *ScalarTy = LibcallImpl == RTLIB::impl___sincosf_stret
                         ? Type::getFloatTy(Ctx)
                         : Type::getDoubleTy(Ctx);

    AttrBuilder FuncAttrBuilder(Ctx);
    for (Attribute::AttrKind Attr : CommonFnAttrs)
      FuncAttrBuilder.addAttribute(Attr);

    const bool UseSret =
        TT.isX86_32() || ((TT.isARM() || TT.isThumb()) &&
                          ARM::computeTargetABI(TT) == ARM::ARM_ABI_APCS);

    FuncAttrBuilder.addMemoryAttr(MemoryEffects::argumentOrErrnoMemOnly(
        UseSret ? ModRefInfo::Mod : ModRefInfo::NoModRef, ModRefInfo::Mod));

    AttributeList Attrs;
    Attrs = Attrs.addFnAttributes(Ctx, FuncAttrBuilder);

    if (UseSret) {
      AttrBuilder AttrBuilder(Ctx);
      StructType *StructTy = StructType::get(ScalarTy, ScalarTy);
      AttrBuilder.addStructRetAttr(StructTy);
      AttrBuilder.addAlignmentAttr(DL.getABITypeAlign(StructTy));
      FunctionType *FuncTy = FunctionType::get(
          Type::getVoidTy(Ctx), {DL.getAllocaPtrType(Ctx), ScalarTy}, false);

      return {FuncTy, Attrs.addParamAttributes(Ctx, 0, AttrBuilder)};
    }

    Type *RetTy =
        LibcallImpl == RTLIB::impl___sincosf_stret && TT.isX86_64()
            ? static_cast<Type *>(FixedVectorType::get(ScalarTy, 2))
            : static_cast<Type *>(StructType::get(ScalarTy, ScalarTy));

    return {FunctionType::get(RetTy, {ScalarTy}, false), Attrs};
  }
  case RTLIB::impl_sqrtf:
  case RTLIB::impl_sqrt: {
    AttrBuilder FuncAttrBuilder(Ctx);

    for (Attribute::AttrKind Attr : CommonFnAttrs)
      FuncAttrBuilder.addAttribute(Attr);
    FuncAttrBuilder.addMemoryAttr(MemoryEffects::errnoMemOnly(ModRefInfo::Mod));

    AttributeList Attrs;
    Attrs = Attrs.addFnAttributes(Ctx, FuncAttrBuilder);

    Type *ScalarTy = LibcallImpl == RTLIB::impl_sqrtf ? Type::getFloatTy(Ctx)
                                                      : Type::getDoubleTy(Ctx);
    FunctionType *FuncTy = FunctionType::get(ScalarTy, {ScalarTy}, false);

    Attrs = Attrs.addRetAttribute(
        Ctx, Attribute::getWithNoFPClass(Ctx, fcNegInf | fcNegSubnormal |
                                                  fcNegNormal));
    return {FuncTy, Attrs};
  }
  default:
    return {};
  }

  return {};
}
