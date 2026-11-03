//===- RuntimeLibcalls.cpp - Interface for runtime libcalls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/SystemLibraries.h"
#include "llvm/IR/Type.h"
#include "llvm/TargetParser/ARMTargetParser.h"

#define DEBUG_TYPE "runtime-libcalls-info"

using namespace llvm;
using namespace RTLIB;

#define GET_RUNTIME_LIBCALLS_INFO
#define GET_INIT_RUNTIME_LIBCALL_NAMES
#define GET_INIT_RUNTIME_LIBCALL_SIGNATURES
#define GET_RUNTIME_LIBCALL_MASKED_VECTOR_LIBCALL_MAPPING
#define GET_SET_TARGET_RUNTIME_LIBCALL_SETS
#define DEFINE_GET_LOOKUP_LIBCALL_IMPL_NAME
#define GET_RUNTIME_LIBCALL_INTRINSIC_TO_LIBCALL
#include "llvm/IR/RuntimeLibcalls.inc"

static RuntimeLibrary getVectorRuntimeLibrary(VectorLibrary VecLib) {
  switch (VecLib) {
  case VectorLibrary::NoLibrary:
    break;
  case VectorLibrary::Accelerate:
    return RuntimeLibrary::accelerate;
  case VectorLibrary::DarwinLibSystemM:
    return RuntimeLibrary::darwin_libsystem_m;
  case VectorLibrary::LIBMVEC:
    return RuntimeLibrary::libmvec;
  case VectorLibrary::MASSV:
    return RuntimeLibrary::massv;
  case VectorLibrary::SVML:
    return RuntimeLibrary::svml;
  case VectorLibrary::SLEEFGNUABI:
    return RuntimeLibrary::sleefgnuabi;
  case VectorLibrary::ArmPL:
    return RuntimeLibrary::armpl;
  case VectorLibrary::AMDLIBM:
    return RuntimeLibrary::amdlibm;
  }

  llvm_unreachable("no RuntimeLibrary for VectorLibrary");
}

RuntimeLibcallsInfo::RuntimeLibcallsInfo(
    const Triple &TT, ExceptionHandling ExceptionModel,
    FloatABI::ABIType FloatABI, std::optional<LongDoubleFormat> LDblFormat,
    StringRef ABIName, VectorLibrary VecLib)
    : LongDoubleFmt(LDblFormat.value_or(TT.getDefaultLongDoubleFormat())) {
  // Only an unspecified model resolves to the triple default; None is left as
  // an explicit disable.
  if (ExceptionModel == ExceptionHandling::Default)
    ExceptionModel = TT.getDefaultExceptionHandling();

  initLibcalls(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFmt);

  if (VecLib != VectorLibrary::NoLibrary) {
    setAvailableLibraryLibcalls(getVectorRuntimeLibrary(VecLib), TT,
                                ExceptionModel, FloatABI, ABIName,
                                LongDoubleFmt);
  }
}

// TODO: Consider the remaining module flags.
RuntimeLibcallsInfo::RuntimeLibcallsInfo(const Module &M, StringRef ABIName,
                                         VectorLibrary VecLib)
    : RuntimeLibcallsInfo(M.getTargetTriple(), M.getExceptionModel(),
                          M.getFloatABI(), M.getLongDoubleFormat(), ABIName,
                          VecLib) {}

RTLIB::Libcall RuntimeLibcallsInfo::selectLongDoubleLibcall(
    RTLIB::Libcall F80, RTLIB::Libcall F128, RTLIB::Libcall PPCF128) const {
  switch (LongDoubleFmt) {
  case LongDoubleFormat::X87DoubleExtended:
    return F80;
  case LongDoubleFormat::IEEEquad:
    return F128;
  case LongDoubleFormat::PPCDoubleDouble:
    return PPCF128;
  case LongDoubleFormat::IEEEsingle:
  case LongDoubleFormat::IEEEdouble:
    // These long double formats aren't properly handled, but we also don't emit
    // calls for them. TODO: We could fix the avr_sin/avr_cos hack if we handled
    // the float case.
    return RTLIB::UNKNOWN_LIBCALL;
  }

  llvm_unreachable("unhandled long double format");
}

bool RuntimeLibcallsInfo::isLibraryAvailable(RuntimeLibrary Library) const {
  // TODO: Drive this from module-level state (e.g. the linked runtime). For now
  // every named library is reported as available.
  return true;
}

void RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(
    const Triple &TT, ExceptionHandling ExceptionModel,
    FloatABI::ABIType FloatABI, StringRef ABIName,
    LongDoubleFormat LongDoubleFormat) {
  CallingConv::ID DefaultCC = getDefaultLibcallCallingConv(
      TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat);
  if (DefaultCC != CallingConv::C)
    llvm::fill(LibcallImplCallingConvs, DefaultCC);

  for (RTLIB::RuntimeLibrary Library : getDefaultRuntimeLibraries(TT)) {
    if (isLibraryAvailable(Library))
      setAvailableLibraryLibcalls(Library, TT, ExceptionModel, FloatABI,
                                  ABIName, LongDoubleFormat);
  }
}

/// Set default libcall names. If a target wants to opt-out of a libcall it
/// should be placed here.
void RuntimeLibcallsInfo::initLibcalls(const Triple &TT,
                                       ExceptionHandling ExceptionModel,
                                       FloatABI::ABIType FloatABI,
                                       StringRef ABIName,
                                       LongDoubleFormat LongDoubleFormat) {
  setTargetRuntimeLibcallSets(TT, ExceptionModel, FloatABI, ABIName,
                              LongDoubleFormat);
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

// PowerPC64, Sparc64, SystemZ need signext/zeroext on i32 parameters and
// returns corresponding to C-level ints and unsigned ints.
static bool shouldExtI32(const Triple &T) {
  return T.isPPC64() || T.getArch() == Triple::sparcv9 ||
         T.getArch() == Triple::systemz;
}

// LoongArch, Mips, and riscv64, on the other hand, need signext on i32
// parameters corresponding to both signed and unsigned ints.
static bool shouldSignExtI32Param(const Triple &T) {
  return T.isLoongArch() || T.isMIPS() || T.isRISCV64();
}

// LoongArch and riscv64 need signext on i32 returns corresponding to both
// signed and unsigned ints.
static bool shouldSignExtI32Return(const Triple &T) {
  return T.isLoongArch() || T.isRISCV64();
}

void RuntimeLibcallsInfo::initExtensionsForTriple(bool &ShouldExtI32Param,
                                                  bool &ShouldExtI32Return,
                                                  bool &ShouldSignExtI32Param,
                                                  bool &ShouldSignExtI32Return,
                                                  const Triple &T) {
  ShouldExtI32Param = ShouldExtI32Return = shouldExtI32(T);
  ShouldSignExtI32Param = shouldSignExtI32Param(T);
  ShouldSignExtI32Return = shouldSignExtI32Return(T);
}

static bool shouldZExtBool(const Triple &T) {
  return !T.isAArch64() && !T.isOSDarwin();
}

void RuntimeLibcallsInfo::initBoolExtensionsForTriple(
    bool &ShouldZExtBoolParam, bool &ShouldZExtBoolReturn, const Triple &T) {
  ShouldZExtBoolParam = ShouldZExtBoolReturn = shouldZExtBool(T);
}

/// TODO: There is really no guarantee that sizeof(size_t) is equal to the index
/// size of the default address space. This matches TargetLibraryInfo and should
/// be kept in sync.
static IntegerType *getSizeTType(LLVMContext &Ctx, const DataLayout &DL) {
  return DL.getIndexType(Ctx, /*AddressSpace=*/0);
}

static unsigned getIntSize(const Triple &TT) {
  return RuntimeLibcallsInfo::getIntSize(TT);
}

static unsigned getLongSize(const Triple &TT, const DataLayout &DL) {
  if (TT.isOSWindowsOrUEFI() && !TT.isWindowsCygwinEnvironment())
    return 32;
  return std::max(32u, DL.getPointerSizeInBits(/*AS=*/0));
}

// TODO: NoCallback probably unsafe in general
static constexpr Attribute::AttrKind CommonFnAttrs[] = {
    Attribute::NoCallback, Attribute::NoFree, Attribute::NoSync,
    Attribute::NoUnwind, Attribute::WillReturn};

static Type *getLongDoubleType(LLVMContext &Ctx, LongDoubleFormat Format) {
  switch (Format) {
  case LongDoubleFormat::IEEEsingle:
    return Type::getFloatTy(Ctx);
  case LongDoubleFormat::IEEEdouble:
    return Type::getDoubleTy(Ctx);
  case LongDoubleFormat::X87DoubleExtended:
    return Type::getX86_FP80Ty(Ctx);
  case LongDoubleFormat::IEEEquad:
    return Type::getFP128Ty(Ctx);
  case LongDoubleFormat::PPCDoubleDouble:
    return Type::getPPC_FP128Ty(Ctx);
  }

  llvm_unreachable("unhandled long double format");
}

static Type *getInt128Type(LLVMContext &Ctx, const Triple &TT) {
  if (TT.isOSWindowsOrUEFI() && TT.isX86_64())
    return FixedVectorType::get(Type::getInt64Ty(Ctx), 2);
  return Type::getInt128Ty(Ctx);
}

static Type *convertToIRType(FuncArgTypeID ID, LLVMContext &Ctx,
                             const Triple &TT, const DataLayout &DL,
                             LongDoubleFormat LdblFormat) {
  // FIXME: Use the llvm/ABI library to get accurate IR types and attributes.
  switch (ID) {
  case Void:
    return Type::getVoidTy(Ctx);
  case Bool:
    return IntegerType::get(Ctx, 1);
  case Int8:
  case UInt8:
    return IntegerType::get(Ctx, 8);
  case Int16:
  case UInt16:
    return IntegerType::get(Ctx, 16);
  case Int32:
  case UInt32:
    return IntegerType::get(Ctx, 32);
  case Int:
  case UInt:
  case IntPlus:
  case UIntPlus:
  case IntX:
  case UIntX:
    return IntegerType::get(Ctx, getIntSize(TT));
  case Long:
  case ULong:
    return IntegerType::get(Ctx, getLongSize(TT, DL));
  case Int64:
  case UInt64:
  case LLong:
  case ULLong:
    return IntegerType::get(Ctx, 64);
  case Int128:
  case UInt128:
    return getInt128Type(Ctx, TT);
  case SizeT:
  case SSizeT:
    return getSizeTType(Ctx, DL);
  case CmpResult:
    if (TT.isAArch64() || TT.isAMDGPU() || TT.isWasm())
      return Type::getInt32Ty(Ctx);
    if (TT.isAVR())
      return Type::getInt8Ty(Ctx);
    return IntegerType::get(Ctx, TT.getArchPointerBitWidth());
  case Half:
    return Type::getHalfTy(Ctx);
  case BFloat:
    return Type::getBFloatTy(Ctx);
  case Flt:
  case Floating:
    return Type::getFloatTy(Ctx);
  case Dbl:
    return Type::getDoubleTy(Ctx);
  case LDbl:
    return getLongDoubleType(Ctx, LdblFormat);
  case X86FP80:
    return Type::getX86_FP80Ty(Ctx);
  case FP128:
    return Type::getFP128Ty(Ctx);
  case PPCFP128:
    return Type::getPPC_FP128Ty(Ctx);
  case Ptr:
    return PointerType::get(Ctx, 0);
  case V2F32:
    return FixedVectorType::get(Type::getFloatTy(Ctx), 2);
  case V4F32:
    return FixedVectorType::get(Type::getFloatTy(Ctx), 4);
  case V8F32:
    return FixedVectorType::get(Type::getFloatTy(Ctx), 8);
  case V16F32:
    return FixedVectorType::get(Type::getFloatTy(Ctx), 16);
  case NXV4F32:
    return ScalableVectorType::get(Type::getFloatTy(Ctx), 4);
  case V2F64:
    return FixedVectorType::get(Type::getDoubleTy(Ctx), 2);
  case V4F64:
    return FixedVectorType::get(Type::getDoubleTy(Ctx), 4);
  case V8F64:
    return FixedVectorType::get(Type::getDoubleTy(Ctx), 8);
  case NXV2F64:
    return ScalableVectorType::get(Type::getDoubleTy(Ctx), 2);
  case NXV2I1:
    return ScalableVectorType::get(Type::getInt1Ty(Ctx), 2);
  case NXV4I1:
    return ScalableVectorType::get(Type::getInt1Ty(Ctx), 4);
  case V2I32:
    return FixedVectorType::get(Type::getInt32Ty(Ctx), 2);
  case V4I32:
    return FixedVectorType::get(Type::getInt32Ty(Ctx), 4);
  case V8I32:
    return FixedVectorType::get(Type::getInt32Ty(Ctx), 8);
  case V16I32:
    return FixedVectorType::get(Type::getInt32Ty(Ctx), 16);
  case NXV2I32:
    return ScalableVectorType::get(Type::getInt32Ty(Ctx), 2);
  case NXV4I32:
    return ScalableVectorType::get(Type::getInt32Ty(Ctx), 4);
  default:
    return nullptr;
  }
}

static std::optional<bool> isSignedIntArgType(FuncArgTypeID ID) {
  switch (ID) {
  case Int8:
  case Int16:
  case Int32:
  case Int:
  case IntPlus:
  case Long:
  case Int64:
  case LLong:
  case Int128:
  case SSizeT:
    return true;
  case UInt8:
  case UInt16:
  case UInt32:
  case UInt:
  case UIntPlus:
  case ULong:
  case UIntX:
  case UInt64:
  case ULLong:
  case UInt128:
  case SizeT:
    return false;
  default:
    return std::nullopt;
  }
}

static Attribute::AttrKind getIntExtAttr(FuncArgTypeID ID, Type *Ty,
                                         const Triple &TT, bool IsReturn) {
  if (ID == Bool)
    return shouldZExtBool(TT) ? Attribute::ZExt : Attribute::None;

  std::optional<bool> Signed = isSignedIntArgType(ID);
  if (!Signed || !Ty->isIntegerTy())
    return Attribute::None;

  // All targets currently extend types narrower than int according to their
  // signedness, but this could become target dependent if a target needs it.
  unsigned Bits = Ty->getIntegerBitWidth();
  unsigned IntBits = getIntSize(TT);

  if (Bits < IntBits)
    return *Signed ? Attribute::SExt : Attribute::ZExt;
  if (Bits != 32 || IntBits != 32)
    return Attribute::None;
  if (shouldExtI32(TT))
    return *Signed ? Attribute::SExt : Attribute::ZExt;
  if (IsReturn ? shouldSignExtI32Return(TT) : shouldSignExtI32Param(TT))
    return Attribute::SExt;
  return Attribute::None;
}

static bool isFPVectorTy(Type *Ty) {
  return Ty->isVectorTy() && Ty->isFPOrFPVectorTy();
}

std::pair<FunctionType *, AttributeList>
RuntimeLibcallsInfo::getDefaultFunctionTy(
    LLVMContext &Ctx, const Triple &TT, const DataLayout &DL,
    RTLIB::LibcallImpl LibcallImpl) const {
  const FuncArgTypeID *ProtoTypes =
      &SignatureTable[SignatureOffset[LibcallImpl]];

  if (ProtoTypes[0] == NoFuncArgType)
    return {};

  AttributeList Attrs;

  FuncArgTypeID LastID = ProtoTypes[0];
  Type *RetTy = convertToIRType(LastID, Ctx, TT, DL, LongDoubleFmt);
  if (Attribute::AttrKind AK =
          getIntExtAttr(LastID, RetTy, TT, /*IsReturn=*/true))
    Attrs = Attrs.addRetAttribute(Ctx, AK);

  SmallVector<Type *, 4> ArgTys;
  bool IsVarArg = false;
  unsigned Idx = 1;
  for (FuncArgTypeID TyID = ProtoTypes[Idx]; TyID != NoFuncArgType;
       TyID = ProtoTypes[++Idx]) {
    if (TyID == Ellip) {
      // The ellipsis ends the protoype list so it must be followed by
      // NoFuncArgType.
      assert(ProtoTypes[Idx + 1] == NoFuncArgType);
      IsVarArg = true;
      break;
    }

    if (TyID == Same)
      TyID = LastID;
    else
      LastID = TyID;

    Type *ArgTy = convertToIRType(TyID, Ctx, TT, DL, LongDoubleFmt);
    ArgTys.push_back(ArgTy);
    if (Attribute::AttrKind AK =
            getIntExtAttr(TyID, ArgTy, TT, /*IsReturn=*/false))
      Attrs = Attrs.addParamAttribute(Ctx, Idx - 1, AK);
  }

  if (RetTy->isVectorTy() && !ArgTys.empty() && isFPVectorTy(ArgTys[0])) {
    AttrBuilder FuncAttrBuilder(Ctx);
    for (Attribute::AttrKind Attr : CommonFnAttrs)
      FuncAttrBuilder.addAttribute(Attr);
    Attrs = Attrs.addFnAttributes(Ctx, FuncAttrBuilder);
  }

  return {FunctionType::get(RetTy, ArgTys, IsVarArg), Attrs};
}

std::pair<FunctionType *, AttributeList>
RuntimeLibcallsInfo::getFunctionTy(LLVMContext &Ctx, const Triple &TT,
                                   const DataLayout &DL,
                                   RTLIB::LibcallImpl LibcallImpl) const {
  static constexpr Attribute::AttrKind MemoryFnAttrs[] = {
      Attribute::NoUnwind, Attribute::WillReturn};
  static constexpr Attribute::AttrKind CommonPtrArgAttrs[] = {
      Attribute::NoAlias, Attribute::WriteOnly, Attribute::NonNull};

  switch (LibcallImpl) {
  // FIXME: Should also include __sincospi[f]_stret
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

    FuncAttrBuilder.addMemoryAttr(MemoryEffects::argMemOnly(
        UseSret ? ModRefInfo::Mod : ModRefInfo::NoModRef));

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
  case RTLIB::impl_malloc:
  case RTLIB::impl_calloc: {
    AttrBuilder FuncAttrBuilder(Ctx);
    for (Attribute::AttrKind Attr : MemoryFnAttrs)
      FuncAttrBuilder.addAttribute(Attr);
    FuncAttrBuilder.addAttribute(Attribute::NoFree);

    AllocFnKind AllocKind = AllocFnKind::Alloc;
    if (LibcallImpl == RTLIB::impl_malloc)
      AllocKind |= AllocFnKind::Uninitialized;

    // TODO: Set memory attribute
    FuncAttrBuilder.addAllocKindAttr(AllocKind);
    FuncAttrBuilder.addAttribute("alloc-family", "malloc");
    FuncAttrBuilder.addAllocSizeAttr(0, LibcallImpl == RTLIB::impl_malloc
                                            ? std::nullopt
                                            : std::make_optional(1));

    AttributeList Attrs;
    Attrs = Attrs.addFnAttributes(Ctx, FuncAttrBuilder);

    {
      AttrBuilder ArgAttrBuilder(Ctx);
      for (Attribute::AttrKind AK : CommonPtrArgAttrs)
        ArgAttrBuilder.addAttribute(AK);

      Attrs = Attrs.addRetAttribute(Ctx, Attribute::NoUndef);
      Attrs = Attrs.addRetAttribute(Ctx, Attribute::NoAlias);
      Attrs = Attrs.addParamAttribute(Ctx, 0, Attribute::NoUndef);
      if (LibcallImpl == RTLIB::impl_calloc)
        Attrs = Attrs.addParamAttribute(Ctx, 1, Attribute::NoUndef);
    }

    IntegerType *SizeT = getSizeTType(Ctx, DL);
    PointerType *PtrTy = PointerType::get(Ctx, 0);
    SmallVector<Type *, 2> ArgTys = {SizeT};
    if (LibcallImpl == RTLIB::impl_calloc)
      ArgTys.push_back(SizeT);

    return {FunctionType::get(PtrTy, ArgTys, false), Attrs};
  }
  case RTLIB::impl_free: {
    // TODO: Set memory attribute
    AttrBuilder FuncAttrBuilder(Ctx);
    for (Attribute::AttrKind Attr : MemoryFnAttrs)
      FuncAttrBuilder.addAttribute(Attr);

    FuncAttrBuilder.addAllocKindAttr(AllocFnKind::Free);
    FuncAttrBuilder.addAttribute("alloc-family", "malloc");

    AttributeList Attrs;
    Attrs = Attrs.addFnAttributes(Ctx, FuncAttrBuilder);

    {
      AttrBuilder ArgAttrBuilder(Ctx);
      ArgAttrBuilder.addAttribute(Attribute::NoUndef);
      ArgAttrBuilder.addAttribute(Attribute::AllocatedPointer);
      ArgAttrBuilder.addCapturesAttr(CaptureInfo::none());
      Attrs = Attrs.addParamAttributes(Ctx, 0, ArgAttrBuilder);
    }

    return {FunctionType::get(Type::getVoidTy(Ctx), {PointerType::get(Ctx, 0)},
                              false),
            Attrs};
  }
  case RTLIB::impl___aeabi_idivmod:
  case RTLIB::impl___aeabi_uidivmod:
  case RTLIB::impl___aeabi_ldivmod:
  case RTLIB::impl___aeabi_uldivmod:
  case RTLIB::impl___rt_sdiv:
  case RTLIB::impl___rt_udiv:
  case RTLIB::impl___rt_sdiv64:
  case RTLIB::impl___rt_udiv64: {
    // The ARM AEABI (__aeabi_*divmod) and Windows (__rt_*div*) divmod functions
    // return both values modeled as an inreg { iN, iN } struct (quotient,
    // remainder). The __rt_*div* cases pass the arguments in opposite order,
    // though this doesn't affect the declaration.
    bool IsSigned;
    unsigned Bits;
    switch (LibcallImpl) {
    case RTLIB::impl___aeabi_idivmod:
    case RTLIB::impl___rt_sdiv:
      IsSigned = true;
      Bits = 32;
      break;
    case RTLIB::impl___aeabi_uidivmod:
    case RTLIB::impl___rt_udiv:
      IsSigned = false;
      Bits = 32;
      break;
    case RTLIB::impl___aeabi_ldivmod:
    case RTLIB::impl___rt_sdiv64:
      IsSigned = true;
      Bits = 64;
      break;
    case RTLIB::impl___aeabi_uldivmod:
    case RTLIB::impl___rt_udiv64:
      IsSigned = false;
      Bits = 64;
      break;
    default:
      llvm_unreachable("unexpected divmod libcall");
    }

    Type *IntTy = IntegerType::get(Ctx, Bits);
    StructType *RetTy = StructType::get(IntTy, IntTy);
    FunctionType *FuncTy = FunctionType::get(RetTy, {IntTy, IntTy}, false);

    AttrBuilder FuncAttrBuilder(Ctx);
    for (Attribute::AttrKind Attr : CommonFnAttrs)
      FuncAttrBuilder.addAttribute(Attr);
    FuncAttrBuilder.addMemoryAttr(MemoryEffects::none());

    AttributeList Attrs;
    Attrs = Attrs.addFnAttributes(Ctx, FuncAttrBuilder);

    Attribute::AttrKind ExtKind = IsSigned ? Attribute::SExt : Attribute::ZExt;
    Attrs = Attrs.addRetAttribute(Ctx, Attribute::InReg);
    Attrs = Attrs.addParamAttribute(Ctx, 0, ExtKind);
    Attrs = Attrs.addParamAttribute(Ctx, 1, ExtKind);

    return {FuncTy, Attrs};
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
    break;
  }

  auto [FuncTy, Attrs] = getDefaultFunctionTy(Ctx, TT, DL, LibcallImpl);
  if (!FuncTy || FuncTy->getNumParams() == 0 ||
      !isFPVectorTy(FuncTy->getParamType(0)))
    return {FuncTy, Attrs};

  Type *VecTy = FuncTy->getParamType(0);
  Align PtrAlign = DL.getABITypeAlign(VecTy->getScalarType());
  switch (LibcallImpl) {
  case RTLIB::impl_amd_vrd2_sincos:
  case RTLIB::impl_amd_vrd4_sincos:
  case RTLIB::impl_amd_vrd8_sincos:
  case RTLIB::impl_amd_vrs4_sincosf:
  case RTLIB::impl_amd_vrs8_sincosf:
  case RTLIB::impl_amd_vrs16_sincosf:
    PtrAlign = DL.getABITypeAlign(VecTy);
    break;
  default:
    break;
  }

  AttrBuilder ArgAttrBuilder(Ctx);
  for (Attribute::AttrKind AK : CommonPtrArgAttrs)
    ArgAttrBuilder.addAttribute(AK);
  ArgAttrBuilder.addAlignmentAttr(PtrAlign);

  bool HasPtrArg = false;
  for (auto [Idx, ParamTy] : enumerate(FuncTy->params())) {
    if (ParamTy->isPointerTy()) {
      Attrs = Attrs.addParamAttributes(Ctx, Idx, ArgAttrBuilder);
      HasPtrArg = true;
    }
  }

  if (HasPtrArg) {
    AttrBuilder FuncAttrBuilder(Ctx);
    for (Attribute::AttrKind Attr : CommonFnAttrs)
      FuncAttrBuilder.addAttribute(Attr);
    FuncAttrBuilder.addMemoryAttr(MemoryEffects::argMemOnly(ModRefInfo::Mod));
    Attrs = Attrs.addFnAttributes(Ctx, FuncAttrBuilder);
  }

  return {FuncTy, Attrs};
}
