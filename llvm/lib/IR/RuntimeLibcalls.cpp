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
#include "llvm/IR/SystemLibraries.h"
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

RuntimeLibcallsInfo::RuntimeLibcallsInfo(const Triple &TT,
                                         ExceptionHandling ExceptionModel,
                                         FloatABI::ABIType FloatABI,
                                         EABI EABIVersion, StringRef ABIName,
                                         VectorLibrary VecLib) {
  // FIXME: The ExceptionModel parameter is to handle the field in
  // TargetOptions. This interface fails to distinguish the forced disable
  // case for targets which support exceptions by default. This should
  // probably be a module flag and removed from TargetOptions.
  if (ExceptionModel == ExceptionHandling::None)
    ExceptionModel = TT.getDefaultExceptionHandling();

  initLibcalls(TT, ExceptionModel, FloatABI, EABIVersion, ABIName);

  // TODO: Tablegen should generate these sets
  switch (VecLib) {
  case VectorLibrary::SLEEFGNUABI:
    for (RTLIB::LibcallImpl Impl :
         {RTLIB::impl__ZGVnN2vl8_modf, RTLIB::impl__ZGVnN4vl4_modff,
          RTLIB::impl__ZGVsNxvl8_modf, RTLIB::impl__ZGVsNxvl4_modff,
          RTLIB::impl__ZGVnN2vl8l8_sincos, RTLIB::impl__ZGVnN4vl4l4_sincosf,
          RTLIB::impl__ZGVsNxvl8l8_sincos, RTLIB::impl__ZGVsNxvl4l4_sincosf,
          RTLIB::impl__ZGVnN4vl4l4_sincospif, RTLIB::impl__ZGVnN2vl8l8_sincospi,
          RTLIB::impl__ZGVsNxvl4l4_sincospif,
          RTLIB::impl__ZGVsNxvl8l8_sincospi})
      setAvailable(Impl);
    break;
  case VectorLibrary::ArmPL:
    for (RTLIB::LibcallImpl Impl :
         {RTLIB::impl_armpl_vmodfq_f64, RTLIB::impl_armpl_vmodfq_f32,
          RTLIB::impl_armpl_svmodf_f64_x, RTLIB::impl_armpl_svmodf_f32_x,
          RTLIB::impl_armpl_vsincosq_f64, RTLIB::impl_armpl_vsincosq_f32,
          RTLIB::impl_armpl_svsincos_f64_x, RTLIB::impl_armpl_svsincos_f32_x,
          RTLIB::impl_armpl_vsincospiq_f32, RTLIB::impl_armpl_vsincospiq_f64,
          RTLIB::impl_armpl_svsincospi_f32_x,
          RTLIB::impl_armpl_svsincospi_f64_x})
      setAvailable(Impl);

    for (RTLIB::LibcallImpl Impl :
         {RTLIB::impl_armpl_vsincosq_f64, RTLIB::impl_armpl_vsincosq_f32})
      setLibcallImplCallingConv(Impl, CallingConv::AArch64_VectorCall);

    break;
  default:
    break;
  }
}

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
  static constexpr Attribute::AttrKind CommonPtrArgAttrs[] = {
      Attribute::NoAlias, Attribute::WriteOnly, Attribute::NonNull};

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
  case RTLIB::impl__ZGVnN2vl8_modf:
  case RTLIB::impl__ZGVnN4vl4_modff:
  case RTLIB::impl__ZGVsNxvl8_modf:
  case RTLIB::impl__ZGVsNxvl4_modff:
  case RTLIB::impl_armpl_vmodfq_f64:
  case RTLIB::impl_armpl_vmodfq_f32:
  case RTLIB::impl_armpl_svmodf_f64_x:
  case RTLIB::impl_armpl_svmodf_f32_x: {
    AttrBuilder FuncAttrBuilder(Ctx);

    bool IsF32 = LibcallImpl == RTLIB::impl__ZGVnN4vl4_modff ||
                 LibcallImpl == RTLIB::impl__ZGVsNxvl4_modff ||
                 LibcallImpl == RTLIB::impl_armpl_vmodfq_f32 ||
                 LibcallImpl == RTLIB::impl_armpl_svmodf_f32_x;

    bool IsScalable = LibcallImpl == RTLIB::impl__ZGVsNxvl8_modf ||
                      LibcallImpl == RTLIB::impl__ZGVsNxvl4_modff ||
                      LibcallImpl == RTLIB::impl_armpl_svmodf_f64_x ||
                      LibcallImpl == RTLIB::impl_armpl_svmodf_f32_x;

    Type *ScalarTy = IsF32 ? Type::getFloatTy(Ctx) : Type::getDoubleTy(Ctx);
    unsigned EC = IsF32 ? 4 : 2;
    VectorType *VecTy = VectorType::get(ScalarTy, EC, IsScalable);

    for (Attribute::AttrKind Attr : CommonFnAttrs)
      FuncAttrBuilder.addAttribute(Attr);
    FuncAttrBuilder.addMemoryAttr(MemoryEffects::argMemOnly(ModRefInfo::Mod));

    AttributeList Attrs;
    Attrs = Attrs.addFnAttributes(Ctx, FuncAttrBuilder);

    {
      AttrBuilder ArgAttrBuilder(Ctx);
      for (Attribute::AttrKind AK : CommonPtrArgAttrs)
        ArgAttrBuilder.addAttribute(AK);
      ArgAttrBuilder.addAlignmentAttr(DL.getABITypeAlign(VecTy));
      Attrs = Attrs.addParamAttributes(Ctx, 1, ArgAttrBuilder);
    }

    PointerType *PtrTy = PointerType::get(Ctx, 0);
    SmallVector<Type *, 4> ArgTys = {VecTy, PtrTy};
    if (hasVectorMaskArgument(LibcallImpl))
      ArgTys.push_back(VectorType::get(Type::getInt1Ty(Ctx), EC, IsScalable));

    return {FunctionType::get(VecTy, ArgTys, false), Attrs};
  }
  case RTLIB::impl__ZGVnN2vl8l8_sincos:
  case RTLIB::impl__ZGVnN4vl4l4_sincosf:
  case RTLIB::impl__ZGVsNxvl8l8_sincos:
  case RTLIB::impl__ZGVsNxvl4l4_sincosf:
  case RTLIB::impl_armpl_vsincosq_f64:
  case RTLIB::impl_armpl_vsincosq_f32:
  case RTLIB::impl_armpl_svsincos_f64_x:
  case RTLIB::impl_armpl_svsincos_f32_x:
  case RTLIB::impl__ZGVnN4vl4l4_sincospif:
  case RTLIB::impl__ZGVnN2vl8l8_sincospi:
  case RTLIB::impl__ZGVsNxvl4l4_sincospif:
  case RTLIB::impl__ZGVsNxvl8l8_sincospi:
  case RTLIB::impl_armpl_vsincospiq_f32:
  case RTLIB::impl_armpl_vsincospiq_f64:
  case RTLIB::impl_armpl_svsincospi_f32_x:
  case RTLIB::impl_armpl_svsincospi_f64_x: {
    AttrBuilder FuncAttrBuilder(Ctx);

    bool IsF32 = LibcallImpl == RTLIB::impl__ZGVnN4vl4l4_sincospif ||
                 LibcallImpl == RTLIB::impl__ZGVsNxvl4l4_sincospif ||
                 LibcallImpl == RTLIB::impl_armpl_vsincospiq_f32 ||
                 LibcallImpl == RTLIB::impl_armpl_svsincospi_f32_x ||
                 LibcallImpl == RTLIB::impl__ZGVnN4vl4l4_sincosf ||
                 LibcallImpl == RTLIB::impl__ZGVsNxvl4l4_sincosf ||
                 LibcallImpl == RTLIB::impl_armpl_vsincosq_f32 ||
                 LibcallImpl == RTLIB::impl_armpl_svsincos_f32_x;

    Type *ScalarTy = IsF32 ? Type::getFloatTy(Ctx) : Type::getDoubleTy(Ctx);
    unsigned EC = IsF32 ? 4 : 2;

    bool IsScalable = LibcallImpl == RTLIB::impl__ZGVsNxvl8l8_sincos ||
                      LibcallImpl == RTLIB::impl__ZGVsNxvl4l4_sincosf ||
                      LibcallImpl == RTLIB::impl_armpl_svsincos_f32_x ||
                      LibcallImpl == RTLIB::impl_armpl_svsincos_f64_x ||
                      LibcallImpl == RTLIB::impl__ZGVsNxvl4l4_sincospif ||
                      LibcallImpl == RTLIB::impl__ZGVsNxvl8l8_sincospi ||
                      LibcallImpl == RTLIB::impl_armpl_svsincospi_f32_x ||
                      LibcallImpl == RTLIB::impl_armpl_svsincospi_f64_x;
    VectorType *VecTy = VectorType::get(ScalarTy, EC, IsScalable);

    for (Attribute::AttrKind Attr : CommonFnAttrs)
      FuncAttrBuilder.addAttribute(Attr);
    FuncAttrBuilder.addMemoryAttr(MemoryEffects::argMemOnly(ModRefInfo::Mod));

    AttributeList Attrs;
    Attrs = Attrs.addFnAttributes(Ctx, FuncAttrBuilder);

    {
      AttrBuilder ArgAttrBuilder(Ctx);
      for (Attribute::AttrKind AK : CommonPtrArgAttrs)
        ArgAttrBuilder.addAttribute(AK);
      ArgAttrBuilder.addAlignmentAttr(DL.getABITypeAlign(VecTy));
      Attrs = Attrs.addParamAttributes(Ctx, 1, ArgAttrBuilder);
      Attrs = Attrs.addParamAttributes(Ctx, 2, ArgAttrBuilder);
    }

    PointerType *PtrTy = PointerType::get(Ctx, 0);
    SmallVector<Type *, 4> ArgTys = {VecTy, PtrTy, PtrTy};
    if (hasVectorMaskArgument(LibcallImpl))
      ArgTys.push_back(VectorType::get(Type::getInt1Ty(Ctx), EC, IsScalable));

    return {FunctionType::get(Type::getVoidTy(Ctx), ArgTys, false), Attrs};
  }
  default:
    return {};
  }

  return {};
}

bool RuntimeLibcallsInfo::hasVectorMaskArgument(RTLIB::LibcallImpl Impl) {
  /// FIXME: This should be generated by tablegen and support the argument at an
  /// arbitrary position
  switch (Impl) {
  case RTLIB::impl_armpl_svmodf_f64_x:
  case RTLIB::impl_armpl_svmodf_f32_x:
  case RTLIB::impl_armpl_svsincos_f32_x:
  case RTLIB::impl_armpl_svsincos_f64_x:
  case RTLIB::impl_armpl_svsincospi_f32_x:
  case RTLIB::impl_armpl_svsincospi_f64_x:
    return true;
  default:
    return false;
  }
}
