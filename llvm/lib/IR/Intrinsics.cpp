//===-- Intrinsics.cpp - Intrinsic Function Handling ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions required for supporting intrinsic functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Intrinsics.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/IntrinsicsBPF.h"
#include "llvm/IR/IntrinsicsHexagon.h"
#include "llvm/IR/IntrinsicsLoongArch.h"
#include "llvm/IR/IntrinsicsMips.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/IntrinsicsPowerPC.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsS390.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/IntrinsicsVE.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/IntrinsicsXCore.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace llvm;

/// Table of string intrinsic names indexed by enum value.
#define GET_INTRINSIC_NAME_TABLE
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_NAME_TABLE

StringRef Intrinsic::getBaseName(ID id) {
  assert(id < num_intrinsics && "Invalid intrinsic ID!");
  return IntrinsicNameTable[IntrinsicNameOffsetTable[id]];
}

StringRef Intrinsic::getName(ID id) {
  assert(id < num_intrinsics && "Invalid intrinsic ID!");
  assert(!Intrinsic::isOverloaded(id) &&
         "This version of getName does not support overloading");
  return getBaseName(id);
}

/// Returns a stable mangling for the type specified for use in the name
/// mangling scheme used by 'any' types in intrinsic signatures.  The mangling
/// of named types is simply their name.  Manglings for unnamed types consist
/// of a prefix ('p' for pointers, 'a' for arrays, 'f_' for functions)
/// combined with the mangling of their component types.  A vararg function
/// type will have a suffix of 'vararg'.  Since function types can contain
/// other function types, we close a function type mangling with suffix 'f'
/// which can't be confused with it's prefix.  This ensures we don't have
/// collisions between two unrelated function types. Otherwise, you might
/// parse ffXX as f(fXX) or f(fX)X.  (X is a placeholder for any other type.)
/// The HasUnnamedType boolean is set if an unnamed type was encountered,
/// indicating that extra care must be taken to ensure a unique name.
static std::string getMangledTypeStr(Type *Ty, bool &HasUnnamedType) {
  std::string Result;
  if (PointerType *PTyp = dyn_cast<PointerType>(Ty)) {
    Result += "p" + utostr(PTyp->getAddressSpace());
  } else if (ArrayType *ATyp = dyn_cast<ArrayType>(Ty)) {
    Result += "a" + utostr(ATyp->getNumElements()) +
              getMangledTypeStr(ATyp->getElementType(), HasUnnamedType);
  } else if (StructType *STyp = dyn_cast<StructType>(Ty)) {
    if (!STyp->isLiteral()) {
      Result += "s_";
      if (STyp->hasName())
        Result += STyp->getName();
      else
        HasUnnamedType = true;
    } else {
      Result += "sl_";
      for (auto *Elem : STyp->elements())
        Result += getMangledTypeStr(Elem, HasUnnamedType);
    }
    // Ensure nested structs are distinguishable.
    Result += "s";
  } else if (FunctionType *FT = dyn_cast<FunctionType>(Ty)) {
    Result += "f_" + getMangledTypeStr(FT->getReturnType(), HasUnnamedType);
    for (size_t i = 0; i < FT->getNumParams(); i++)
      Result += getMangledTypeStr(FT->getParamType(i), HasUnnamedType);
    if (FT->isVarArg())
      Result += "vararg";
    // Ensure nested function types are distinguishable.
    Result += "f";
  } else if (VectorType *VTy = dyn_cast<VectorType>(Ty)) {
    ElementCount EC = VTy->getElementCount();
    if (EC.isScalable())
      Result += "nx";
    Result += "v" + utostr(EC.getKnownMinValue()) +
              getMangledTypeStr(VTy->getElementType(), HasUnnamedType);
  } else if (TargetExtType *TETy = dyn_cast<TargetExtType>(Ty)) {
    Result += "t";
    Result += TETy->getName();
    for (Type *ParamTy : TETy->type_params())
      Result += "_" + getMangledTypeStr(ParamTy, HasUnnamedType);
    for (unsigned IntParam : TETy->int_params())
      Result += "_" + utostr(IntParam);
    // Ensure nested target extension types are distinguishable.
    Result += "t";
  } else if (Ty) {
    switch (Ty->getTypeID()) {
    default:
      llvm_unreachable("Unhandled type");
    case Type::VoidTyID:
      Result += "isVoid";
      break;
    case Type::MetadataTyID:
      Result += "Metadata";
      break;
    case Type::HalfTyID:
      Result += "f16";
      break;
    case Type::BFloatTyID:
      Result += "bf16";
      break;
    case Type::FloatTyID:
      Result += "f32";
      break;
    case Type::DoubleTyID:
      Result += "f64";
      break;
    case Type::X86_FP80TyID:
      Result += "f80";
      break;
    case Type::FP128TyID:
      Result += "f128";
      break;
    case Type::PPC_FP128TyID:
      Result += "ppcf128";
      break;
    case Type::X86_AMXTyID:
      Result += "x86amx";
      break;
    case Type::IntegerTyID:
      Result += "i" + utostr(cast<IntegerType>(Ty)->getBitWidth());
      break;
    }
  }
  return Result;
}

static std::string getIntrinsicNameImpl(Intrinsic::ID Id, ArrayRef<Type *> Tys,
                                        Module *M, FunctionType *FT,
                                        bool EarlyModuleCheck) {

  assert(Id < Intrinsic::num_intrinsics && "Invalid intrinsic ID!");
  assert((Tys.empty() || Intrinsic::isOverloaded(Id)) &&
         "This version of getName is for overloaded intrinsics only");
  (void)EarlyModuleCheck;
  assert((!EarlyModuleCheck || M ||
          !any_of(Tys, [](Type *T) { return isa<PointerType>(T); })) &&
         "Intrinsic overloading on pointer types need to provide a Module");
  bool HasUnnamedType = false;
  std::string Result(Intrinsic::getBaseName(Id));
  for (Type *Ty : Tys)
    Result += "." + getMangledTypeStr(Ty, HasUnnamedType);
  if (HasUnnamedType) {
    assert(M && "unnamed types need a module");
    if (!FT)
      FT = Intrinsic::getType(M->getContext(), Id, Tys);
    else
      assert((FT == Intrinsic::getType(M->getContext(), Id, Tys)) &&
             "Provided FunctionType must match arguments");
    return M->getUniqueIntrinsicName(Result, Id, FT);
  }
  return Result;
}

std::string Intrinsic::getName(ID Id, ArrayRef<Type *> Tys, Module *M,
                               FunctionType *FT) {
  assert(M && "We need to have a Module");
  return getIntrinsicNameImpl(Id, Tys, M, FT, true);
}

std::string Intrinsic::getNameNoUnnamedTypes(ID Id, ArrayRef<Type *> Tys) {
  return getIntrinsicNameImpl(Id, Tys, nullptr, nullptr, false);
}

/// IIT_Info - These are enumerators that describe the entries returned by the
/// getIntrinsicInfoTableEntries function.
///
/// Defined in Intrinsics.td.
enum IIT_Info {
#define GET_INTRINSIC_IITINFO
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_IITINFO
};

static void
DecodeIITType(unsigned &NextElt, ArrayRef<unsigned char> Infos,
              IIT_Info LastInfo,
              SmallVectorImpl<Intrinsic::IITDescriptor> &OutputTable) {
  using namespace Intrinsic;

  bool IsScalableVector = (LastInfo == IIT_SCALABLE_VEC);

  IIT_Info Info = IIT_Info(Infos[NextElt++]);

  switch (Info) {
  case IIT_Done:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Void, 0));
    return;
  case IIT_VARARG:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::VarArg, 0));
    return;
  case IIT_MMX:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::MMX, 0));
    return;
  case IIT_AMX:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::AMX, 0));
    return;
  case IIT_TOKEN:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Token, 0));
    return;
  case IIT_METADATA:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Metadata, 0));
    return;
  case IIT_F16:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Half, 0));
    return;
  case IIT_BF16:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::BFloat, 0));
    return;
  case IIT_F32:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Float, 0));
    return;
  case IIT_F64:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Double, 0));
    return;
  case IIT_F128:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Quad, 0));
    return;
  case IIT_PPCF128:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::PPCQuad, 0));
    return;
  case IIT_I1:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 1));
    return;
  case IIT_I2:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 2));
    return;
  case IIT_I4:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 4));
    return;
  case IIT_AARCH64_SVCOUNT:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::AArch64Svcount, 0));
    return;
  case IIT_I8:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 8));
    return;
  case IIT_I16:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 16));
    return;
  case IIT_I32:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 32));
    return;
  case IIT_I64:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 64));
    return;
  case IIT_I128:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 128));
    return;
  case IIT_V1:
    OutputTable.push_back(IITDescriptor::getVector(1, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V2:
    OutputTable.push_back(IITDescriptor::getVector(2, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V3:
    OutputTable.push_back(IITDescriptor::getVector(3, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V4:
    OutputTable.push_back(IITDescriptor::getVector(4, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V6:
    OutputTable.push_back(IITDescriptor::getVector(6, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V8:
    OutputTable.push_back(IITDescriptor::getVector(8, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V10:
    OutputTable.push_back(IITDescriptor::getVector(10, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V16:
    OutputTable.push_back(IITDescriptor::getVector(16, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V32:
    OutputTable.push_back(IITDescriptor::getVector(32, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V64:
    OutputTable.push_back(IITDescriptor::getVector(64, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V128:
    OutputTable.push_back(IITDescriptor::getVector(128, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V256:
    OutputTable.push_back(IITDescriptor::getVector(256, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V512:
    OutputTable.push_back(IITDescriptor::getVector(512, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V1024:
    OutputTable.push_back(IITDescriptor::getVector(1024, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V2048:
    OutputTable.push_back(IITDescriptor::getVector(2048, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V4096:
    OutputTable.push_back(IITDescriptor::getVector(4096, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_EXTERNREF:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Pointer, 10));
    return;
  case IIT_FUNCREF:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Pointer, 20));
    return;
  case IIT_PTR:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Pointer, 0));
    return;
  case IIT_ANYPTR: // [ANYPTR addrspace]
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::Pointer, Infos[NextElt++]));
    return;
  case IIT_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Argument, ArgInfo));
    return;
  }
  case IIT_EXTEND_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::ExtendArgument, ArgInfo));
    return;
  }
  case IIT_TRUNC_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::TruncArgument, ArgInfo));
    return;
  }
  case IIT_ONE_NTH_ELTS_VEC_ARG: {
    unsigned short ArgNo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    unsigned short N = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::OneNthEltsVecArgument, N, ArgNo));
    return;
  }
  case IIT_SAME_VEC_WIDTH_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::SameVecWidthArgument, ArgInfo));
    return;
  }
  case IIT_VEC_OF_ANYPTRS_TO_ELT: {
    unsigned short ArgNo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    unsigned short RefNo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::VecOfAnyPtrsToElt, ArgNo, RefNo));
    return;
  }
  case IIT_EMPTYSTRUCT:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Struct, 0));
    return;
  case IIT_STRUCT: {
    unsigned StructElts = Infos[NextElt++] + 2;

    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::Struct, StructElts));

    for (unsigned i = 0; i != StructElts; ++i)
      DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  }
  case IIT_SUBDIVIDE2_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::Subdivide2Argument, ArgInfo));
    return;
  }
  case IIT_SUBDIVIDE4_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::Subdivide4Argument, ArgInfo));
    return;
  }
  case IIT_VEC_ELEMENT: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::VecElementArgument, ArgInfo));
    return;
  }
  case IIT_SCALABLE_VEC: {
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  }
  case IIT_VEC_OF_BITCASTS_TO_INT: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::VecOfBitcastsToInt, ArgInfo));
    return;
  }
  }
  llvm_unreachable("unhandled");
}

#define GET_INTRINSIC_GENERATOR_GLOBAL
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_GENERATOR_GLOBAL

void Intrinsic::getIntrinsicInfoTableEntries(
    ID id, SmallVectorImpl<IITDescriptor> &T) {
  static_assert(sizeof(IIT_Table[0]) == 2,
                "Expect 16-bit entries in IIT_Table");
  // Check to see if the intrinsic's type was expressible by the table.
  uint16_t TableVal = IIT_Table[id - 1];

  // Decode the TableVal into an array of IITValues.
  SmallVector<unsigned char> IITValues;
  ArrayRef<unsigned char> IITEntries;
  unsigned NextElt = 0;
  if (TableVal >> 15) {
    // This is an offset into the IIT_LongEncodingTable.
    IITEntries = IIT_LongEncodingTable;

    // Strip sentinel bit.
    NextElt = TableVal & 0x7fff;
  } else {
    // If the entry was encoded into a single word in the table itself, decode
    // it from an array of nibbles to an array of bytes.
    do {
      IITValues.push_back(TableVal & 0xF);
      TableVal >>= 4;
    } while (TableVal);

    IITEntries = IITValues;
    NextElt = 0;
  }

  // Okay, decode the table into the output vector of IITDescriptors.
  DecodeIITType(NextElt, IITEntries, IIT_Done, T);
  while (NextElt != IITEntries.size() && IITEntries[NextElt] != 0)
    DecodeIITType(NextElt, IITEntries, IIT_Done, T);
}

static Type *DecodeFixedType(ArrayRef<Intrinsic::IITDescriptor> &Infos,
                             ArrayRef<Type *> Tys, LLVMContext &Context) {
  using namespace Intrinsic;

  IITDescriptor D = Infos.front();
  Infos = Infos.slice(1);

  switch (D.Kind) {
  case IITDescriptor::Void:
    return Type::getVoidTy(Context);
  case IITDescriptor::VarArg:
    return Type::getVoidTy(Context);
  case IITDescriptor::MMX:
    return llvm::FixedVectorType::get(llvm::IntegerType::get(Context, 64), 1);
  case IITDescriptor::AMX:
    return Type::getX86_AMXTy(Context);
  case IITDescriptor::Token:
    return Type::getTokenTy(Context);
  case IITDescriptor::Metadata:
    return Type::getMetadataTy(Context);
  case IITDescriptor::Half:
    return Type::getHalfTy(Context);
  case IITDescriptor::BFloat:
    return Type::getBFloatTy(Context);
  case IITDescriptor::Float:
    return Type::getFloatTy(Context);
  case IITDescriptor::Double:
    return Type::getDoubleTy(Context);
  case IITDescriptor::Quad:
    return Type::getFP128Ty(Context);
  case IITDescriptor::PPCQuad:
    return Type::getPPC_FP128Ty(Context);
  case IITDescriptor::AArch64Svcount:
    return TargetExtType::get(Context, "aarch64.svcount");

  case IITDescriptor::Integer:
    return IntegerType::get(Context, D.Integer_Width);
  case IITDescriptor::Vector:
    return VectorType::get(DecodeFixedType(Infos, Tys, Context),
                           D.Vector_Width);
  case IITDescriptor::Pointer:
    return PointerType::get(Context, D.Pointer_AddressSpace);
  case IITDescriptor::Struct: {
    SmallVector<Type *, 8> Elts;
    for (unsigned i = 0, e = D.Struct_NumElements; i != e; ++i)
      Elts.push_back(DecodeFixedType(Infos, Tys, Context));
    return StructType::get(Context, Elts);
  }
  case IITDescriptor::Argument:
    return Tys[D.getArgumentNumber()];
  case IITDescriptor::ExtendArgument: {
    Type *Ty = Tys[D.getArgumentNumber()];
    if (VectorType *VTy = dyn_cast<VectorType>(Ty))
      return VectorType::getExtendedElementVectorType(VTy);

    return IntegerType::get(Context, 2 * cast<IntegerType>(Ty)->getBitWidth());
  }
  case IITDescriptor::TruncArgument: {
    Type *Ty = Tys[D.getArgumentNumber()];
    if (VectorType *VTy = dyn_cast<VectorType>(Ty))
      return VectorType::getTruncatedElementVectorType(VTy);

    IntegerType *ITy = cast<IntegerType>(Ty);
    assert(ITy->getBitWidth() % 2 == 0);
    return IntegerType::get(Context, ITy->getBitWidth() / 2);
  }
  case IITDescriptor::Subdivide2Argument:
  case IITDescriptor::Subdivide4Argument: {
    Type *Ty = Tys[D.getArgumentNumber()];
    VectorType *VTy = dyn_cast<VectorType>(Ty);
    assert(VTy && "Expected an argument of Vector Type");
    int SubDivs = D.Kind == IITDescriptor::Subdivide2Argument ? 1 : 2;
    return VectorType::getSubdividedVectorType(VTy, SubDivs);
  }
  case IITDescriptor::OneNthEltsVecArgument:
    return VectorType::getOneNthElementsVectorType(
        cast<VectorType>(Tys[D.getRefArgNumber()]), D.getVectorDivisor());
  case IITDescriptor::SameVecWidthArgument: {
    Type *EltTy = DecodeFixedType(Infos, Tys, Context);
    Type *Ty = Tys[D.getArgumentNumber()];
    if (auto *VTy = dyn_cast<VectorType>(Ty))
      return VectorType::get(EltTy, VTy->getElementCount());
    return EltTy;
  }
  case IITDescriptor::VecElementArgument: {
    Type *Ty = Tys[D.getArgumentNumber()];
    if (VectorType *VTy = dyn_cast<VectorType>(Ty))
      return VTy->getElementType();
    llvm_unreachable("Expected an argument of Vector Type");
  }
  case IITDescriptor::VecOfBitcastsToInt: {
    Type *Ty = Tys[D.getArgumentNumber()];
    VectorType *VTy = dyn_cast<VectorType>(Ty);
    assert(VTy && "Expected an argument of Vector Type");
    return VectorType::getInteger(VTy);
  }
  case IITDescriptor::VecOfAnyPtrsToElt:
    // Return the overloaded type (which determines the pointers address space)
    return Tys[D.getOverloadArgNumber()];
  }
  llvm_unreachable("unhandled");
}

FunctionType *Intrinsic::getType(LLVMContext &Context, ID id,
                                 ArrayRef<Type *> Tys) {
  SmallVector<IITDescriptor, 8> Table;
  getIntrinsicInfoTableEntries(id, Table);

  ArrayRef<IITDescriptor> TableRef = Table;
  Type *ResultTy = DecodeFixedType(TableRef, Tys, Context);

  SmallVector<Type *, 8> ArgTys;
  while (!TableRef.empty())
    ArgTys.push_back(DecodeFixedType(TableRef, Tys, Context));

  // DecodeFixedType returns Void for IITDescriptor::Void and
  // IITDescriptor::VarArg If we see void type as the type of the last argument,
  // it is vararg intrinsic
  if (!ArgTys.empty() && ArgTys.back()->isVoidTy()) {
    ArgTys.pop_back();
    return FunctionType::get(ResultTy, ArgTys, true);
  }
  return FunctionType::get(ResultTy, ArgTys, false);
}

bool Intrinsic::isOverloaded(ID id) {
#define GET_INTRINSIC_OVERLOAD_TABLE
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_OVERLOAD_TABLE
}

/// Table of per-target intrinsic name tables.
#define GET_INTRINSIC_TARGET_DATA
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_TARGET_DATA

bool Intrinsic::isTargetIntrinsic(Intrinsic::ID IID) {
  return IID > TargetInfos[0].Count;
}

/// Looks up Name in NameTable via binary search. NameTable must be sorted
/// and all entries must start with "llvm.".  If NameTable contains an exact
/// match for Name or a prefix of Name followed by a dot, its index in
/// NameTable is returned. Otherwise, -1 is returned.
static int lookupLLVMIntrinsicByName(ArrayRef<unsigned> NameOffsetTable,
                                     StringRef Name, StringRef Target = "") {
  assert(Name.starts_with("llvm.") && "Unexpected intrinsic prefix");
  assert(Name.drop_front(5).starts_with(Target) && "Unexpected target");

  // Do successive binary searches of the dotted name components. For
  // "llvm.gc.experimental.statepoint.p1i8.p1i32", we will find the range of
  // intrinsics starting with "llvm.gc", then "llvm.gc.experimental", then
  // "llvm.gc.experimental.statepoint", and then we will stop as the range is
  // size 1. During the search, we can skip the prefix that we already know is
  // identical. By using strncmp we consider names with differing suffixes to
  // be part of the equal range.
  size_t CmpEnd = 4; // Skip the "llvm" component.
  if (!Target.empty())
    CmpEnd += 1 + Target.size(); // skip the .target component.

  const unsigned *Low = NameOffsetTable.begin();
  const unsigned *High = NameOffsetTable.end();
  const unsigned *LastLow = Low;
  while (CmpEnd < Name.size() && High - Low > 0) {
    size_t CmpStart = CmpEnd;
    CmpEnd = Name.find('.', CmpStart + 1);
    CmpEnd = CmpEnd == StringRef::npos ? Name.size() : CmpEnd;
    auto Cmp = [CmpStart, CmpEnd](auto LHS, auto RHS) {
      // `equal_range` requires the comparison to work with either side being an
      // offset or the value. Detect which kind each side is to set up the
      // compared strings.
      const char *LHSStr;
      if constexpr (std::is_integral_v<decltype(LHS)>)
        LHSStr = IntrinsicNameTable.getCString(LHS);
      else
        LHSStr = LHS;

      const char *RHSStr;
      if constexpr (std::is_integral_v<decltype(RHS)>)
        RHSStr = IntrinsicNameTable.getCString(RHS);
      else
        RHSStr = RHS;

      return strncmp(LHSStr + CmpStart, RHSStr + CmpStart, CmpEnd - CmpStart) <
             0;
    };
    LastLow = Low;
    std::tie(Low, High) = std::equal_range(Low, High, Name.data(), Cmp);
  }
  if (High - Low > 0)
    LastLow = Low;

  if (LastLow == NameOffsetTable.end())
    return -1;
  StringRef NameFound = IntrinsicNameTable[*LastLow];
  if (Name == NameFound ||
      (Name.starts_with(NameFound) && Name[NameFound.size()] == '.'))
    return LastLow - NameOffsetTable.begin();
  return -1;
}

/// Find the segment of \c IntrinsicNameOffsetTable for intrinsics with the same
/// target as \c Name, or the generic table if \c Name is not target specific.
///
/// Returns the relevant slice of \c IntrinsicNameOffsetTable and the target
/// name.
static std::pair<ArrayRef<unsigned>, StringRef>
findTargetSubtable(StringRef Name) {
  assert(Name.starts_with("llvm."));

  ArrayRef<IntrinsicTargetInfo> Targets(TargetInfos);
  // Drop "llvm." and take the first dotted component. That will be the target
  // if this is target specific.
  StringRef Target = Name.drop_front(5).split('.').first;
  auto It = partition_point(
      Targets, [=](const IntrinsicTargetInfo &TI) { return TI.Name < Target; });
  // We've either found the target or just fall back to the generic set, which
  // is always first.
  const auto &TI = It != Targets.end() && It->Name == Target ? *It : Targets[0];
  return {ArrayRef(&IntrinsicNameOffsetTable[1] + TI.Offset, TI.Count),
          TI.Name};
}

/// This does the actual lookup of an intrinsic ID which matches the given
/// function name.
Intrinsic::ID Intrinsic::lookupIntrinsicID(StringRef Name) {
  auto [NameOffsetTable, Target] = findTargetSubtable(Name);
  int Idx = lookupLLVMIntrinsicByName(NameOffsetTable, Name, Target);
  if (Idx == -1)
    return Intrinsic::not_intrinsic;

  // Intrinsic IDs correspond to the location in IntrinsicNameTable, but we have
  // an index into a sub-table.
  int Adjust = NameOffsetTable.data() - IntrinsicNameOffsetTable;
  Intrinsic::ID ID = static_cast<Intrinsic::ID>(Idx + Adjust);

  // If the intrinsic is not overloaded, require an exact match. If it is
  // overloaded, require either exact or prefix match.
  const auto MatchSize = IntrinsicNameTable[NameOffsetTable[Idx]].size();
  assert(Name.size() >= MatchSize && "Expected either exact or prefix match");
  bool IsExactMatch = Name.size() == MatchSize;
  return IsExactMatch || Intrinsic::isOverloaded(ID) ? ID
                                                     : Intrinsic::not_intrinsic;
}

/// This defines the "Intrinsic::getAttributes(ID id)" method.
#define GET_INTRINSIC_ATTRIBUTES
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_ATTRIBUTES

Function *Intrinsic::getOrInsertDeclaration(Module *M, ID id,
                                            ArrayRef<Type *> Tys) {
  // There can never be multiple globals with the same name of different types,
  // because intrinsics must be a specific type.
  auto *FT = getType(M->getContext(), id, Tys);
  Function *F = cast<Function>(
      M->getOrInsertFunction(
           Tys.empty() ? getName(id) : getName(id, Tys, M, FT), FT)
          .getCallee());
  if (F->getFunctionType() == FT)
    return F;

  // It's possible that a declaration for this intrinsic already exists with an
  // incorrect signature, if the signature has changed, but this particular
  // declaration has not been auto-upgraded yet. In that case, rename the
  // invalid declaration and insert a new one with the correct signature. The
  // invalid declaration will get upgraded later.
  F->setName(F->getName() + ".invalid");
  return cast<Function>(
      M->getOrInsertFunction(
           Tys.empty() ? getName(id) : getName(id, Tys, M, FT), FT)
          .getCallee());
}

Function *Intrinsic::getDeclarationIfExists(const Module *M, ID id) {
  return M->getFunction(getName(id));
}

Function *Intrinsic::getDeclarationIfExists(Module *M, ID id,
                                            ArrayRef<Type *> Tys,
                                            FunctionType *FT) {
  return M->getFunction(getName(id, Tys, M, FT));
}

// This defines the "Intrinsic::getIntrinsicForClangBuiltin()" method.
#define GET_LLVM_INTRINSIC_FOR_CLANG_BUILTIN
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_LLVM_INTRINSIC_FOR_CLANG_BUILTIN

// This defines the "Intrinsic::getIntrinsicForMSBuiltin()" method.
#define GET_LLVM_INTRINSIC_FOR_MS_BUILTIN
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_LLVM_INTRINSIC_FOR_MS_BUILTIN

bool Intrinsic::isConstrainedFPIntrinsic(ID QID) {
  switch (QID) {
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC)                         \
  case Intrinsic::INTRINSIC:
#include "llvm/IR/ConstrainedOps.def"
#undef INSTRUCTION
    return true;
  default:
    return false;
  }
}

bool Intrinsic::hasConstrainedFPRoundingModeOperand(Intrinsic::ID QID) {
  switch (QID) {
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC)                         \
  case Intrinsic::INTRINSIC:                                                   \
    return ROUND_MODE == 1;
#include "llvm/IR/ConstrainedOps.def"
#undef INSTRUCTION
  default:
    return false;
  }
}

using DeferredIntrinsicMatchPair =
    std::pair<Type *, ArrayRef<Intrinsic::IITDescriptor>>;

static bool
matchIntrinsicType(Type *Ty, ArrayRef<Intrinsic::IITDescriptor> &Infos,
                   SmallVectorImpl<Type *> &ArgTys,
                   SmallVectorImpl<DeferredIntrinsicMatchPair> &DeferredChecks,
                   bool IsDeferredCheck) {
  using namespace Intrinsic;

  // If we ran out of descriptors, there are too many arguments.
  if (Infos.empty())
    return true;

  // Do this before slicing off the 'front' part
  auto InfosRef = Infos;
  auto DeferCheck = [&DeferredChecks, &InfosRef](Type *T) {
    DeferredChecks.emplace_back(T, InfosRef);
    return false;
  };

  IITDescriptor D = Infos.front();
  Infos = Infos.slice(1);

  switch (D.Kind) {
  case IITDescriptor::Void:
    return !Ty->isVoidTy();
  case IITDescriptor::VarArg:
    return true;
  case IITDescriptor::MMX: {
    FixedVectorType *VT = dyn_cast<FixedVectorType>(Ty);
    return !VT || VT->getNumElements() != 1 ||
           !VT->getElementType()->isIntegerTy(64);
  }
  case IITDescriptor::AMX:
    return !Ty->isX86_AMXTy();
  case IITDescriptor::Token:
    return !Ty->isTokenTy();
  case IITDescriptor::Metadata:
    return !Ty->isMetadataTy();
  case IITDescriptor::Half:
    return !Ty->isHalfTy();
  case IITDescriptor::BFloat:
    return !Ty->isBFloatTy();
  case IITDescriptor::Float:
    return !Ty->isFloatTy();
  case IITDescriptor::Double:
    return !Ty->isDoubleTy();
  case IITDescriptor::Quad:
    return !Ty->isFP128Ty();
  case IITDescriptor::PPCQuad:
    return !Ty->isPPC_FP128Ty();
  case IITDescriptor::Integer:
    return !Ty->isIntegerTy(D.Integer_Width);
  case IITDescriptor::AArch64Svcount:
    return !isa<TargetExtType>(Ty) ||
           cast<TargetExtType>(Ty)->getName() != "aarch64.svcount";
  case IITDescriptor::Vector: {
    VectorType *VT = dyn_cast<VectorType>(Ty);
    return !VT || VT->getElementCount() != D.Vector_Width ||
           matchIntrinsicType(VT->getElementType(), Infos, ArgTys,
                              DeferredChecks, IsDeferredCheck);
  }
  case IITDescriptor::Pointer: {
    PointerType *PT = dyn_cast<PointerType>(Ty);
    return !PT || PT->getAddressSpace() != D.Pointer_AddressSpace;
  }

  case IITDescriptor::Struct: {
    StructType *ST = dyn_cast<StructType>(Ty);
    if (!ST || !ST->isLiteral() || ST->isPacked() ||
        ST->getNumElements() != D.Struct_NumElements)
      return true;

    for (unsigned i = 0, e = D.Struct_NumElements; i != e; ++i)
      if (matchIntrinsicType(ST->getElementType(i), Infos, ArgTys,
                             DeferredChecks, IsDeferredCheck))
        return true;
    return false;
  }

  case IITDescriptor::Argument:
    // If this is the second occurrence of an argument,
    // verify that the later instance matches the previous instance.
    if (D.getArgumentNumber() < ArgTys.size())
      return Ty != ArgTys[D.getArgumentNumber()];

    if (D.getArgumentNumber() > ArgTys.size() ||
        D.getArgumentKind() == IITDescriptor::AK_MatchType)
      return IsDeferredCheck || DeferCheck(Ty);

    assert(D.getArgumentNumber() == ArgTys.size() && !IsDeferredCheck &&
           "Table consistency error");
    ArgTys.push_back(Ty);

    switch (D.getArgumentKind()) {
    case IITDescriptor::AK_Any:
      return false; // Success
    case IITDescriptor::AK_AnyInteger:
      return !Ty->isIntOrIntVectorTy();
    case IITDescriptor::AK_AnyFloat:
      return !Ty->isFPOrFPVectorTy();
    case IITDescriptor::AK_AnyVector:
      return !isa<VectorType>(Ty);
    case IITDescriptor::AK_AnyPointer:
      return !isa<PointerType>(Ty);
    default:
      break;
    }
    llvm_unreachable("all argument kinds not covered");

  case IITDescriptor::ExtendArgument: {
    // If this is a forward reference, defer the check for later.
    if (D.getArgumentNumber() >= ArgTys.size())
      return IsDeferredCheck || DeferCheck(Ty);

    Type *NewTy = ArgTys[D.getArgumentNumber()];
    if (VectorType *VTy = dyn_cast<VectorType>(NewTy))
      NewTy = VectorType::getExtendedElementVectorType(VTy);
    else if (IntegerType *ITy = dyn_cast<IntegerType>(NewTy))
      NewTy = IntegerType::get(ITy->getContext(), 2 * ITy->getBitWidth());
    else
      return true;

    return Ty != NewTy;
  }
  case IITDescriptor::TruncArgument: {
    // If this is a forward reference, defer the check for later.
    if (D.getArgumentNumber() >= ArgTys.size())
      return IsDeferredCheck || DeferCheck(Ty);

    Type *NewTy = ArgTys[D.getArgumentNumber()];
    if (VectorType *VTy = dyn_cast<VectorType>(NewTy))
      NewTy = VectorType::getTruncatedElementVectorType(VTy);
    else if (IntegerType *ITy = dyn_cast<IntegerType>(NewTy))
      NewTy = IntegerType::get(ITy->getContext(), ITy->getBitWidth() / 2);
    else
      return true;

    return Ty != NewTy;
  }
  case IITDescriptor::OneNthEltsVecArgument:
    // If this is a forward reference, defer the check for later.
    if (D.getRefArgNumber() >= ArgTys.size())
      return IsDeferredCheck || DeferCheck(Ty);
    return !isa<VectorType>(ArgTys[D.getRefArgNumber()]) ||
           VectorType::getOneNthElementsVectorType(
               cast<VectorType>(ArgTys[D.getRefArgNumber()]),
               D.getVectorDivisor()) != Ty;
  case IITDescriptor::SameVecWidthArgument: {
    if (D.getArgumentNumber() >= ArgTys.size()) {
      // Defer check and subsequent check for the vector element type.
      Infos = Infos.slice(1);
      return IsDeferredCheck || DeferCheck(Ty);
    }
    auto *ReferenceType = dyn_cast<VectorType>(ArgTys[D.getArgumentNumber()]);
    auto *ThisArgType = dyn_cast<VectorType>(Ty);
    // Both must be vectors of the same number of elements or neither.
    if ((ReferenceType != nullptr) != (ThisArgType != nullptr))
      return true;
    Type *EltTy = Ty;
    if (ThisArgType) {
      if (ReferenceType->getElementCount() != ThisArgType->getElementCount())
        return true;
      EltTy = ThisArgType->getElementType();
    }
    return matchIntrinsicType(EltTy, Infos, ArgTys, DeferredChecks,
                              IsDeferredCheck);
  }
  case IITDescriptor::VecOfAnyPtrsToElt: {
    unsigned RefArgNumber = D.getRefArgNumber();
    if (RefArgNumber >= ArgTys.size()) {
      if (IsDeferredCheck)
        return true;
      // If forward referencing, already add the pointer-vector type and
      // defer the checks for later.
      ArgTys.push_back(Ty);
      return DeferCheck(Ty);
    }

    if (!IsDeferredCheck) {
      assert(D.getOverloadArgNumber() == ArgTys.size() &&
             "Table consistency error");
      ArgTys.push_back(Ty);
    }

    // Verify the overloaded type "matches" the Ref type.
    // i.e. Ty is a vector with the same width as Ref.
    // Composed of pointers to the same element type as Ref.
    auto *ReferenceType = dyn_cast<VectorType>(ArgTys[RefArgNumber]);
    auto *ThisArgVecTy = dyn_cast<VectorType>(Ty);
    if (!ThisArgVecTy || !ReferenceType ||
        (ReferenceType->getElementCount() != ThisArgVecTy->getElementCount()))
      return true;
    return !ThisArgVecTy->getElementType()->isPointerTy();
  }
  case IITDescriptor::VecElementArgument: {
    if (D.getArgumentNumber() >= ArgTys.size())
      return IsDeferredCheck ? true : DeferCheck(Ty);
    auto *ReferenceType = dyn_cast<VectorType>(ArgTys[D.getArgumentNumber()]);
    return !ReferenceType || Ty != ReferenceType->getElementType();
  }
  case IITDescriptor::Subdivide2Argument:
  case IITDescriptor::Subdivide4Argument: {
    // If this is a forward reference, defer the check for later.
    if (D.getArgumentNumber() >= ArgTys.size())
      return IsDeferredCheck || DeferCheck(Ty);

    Type *NewTy = ArgTys[D.getArgumentNumber()];
    if (auto *VTy = dyn_cast<VectorType>(NewTy)) {
      int SubDivs = D.Kind == IITDescriptor::Subdivide2Argument ? 1 : 2;
      NewTy = VectorType::getSubdividedVectorType(VTy, SubDivs);
      return Ty != NewTy;
    }
    return true;
  }
  case IITDescriptor::VecOfBitcastsToInt: {
    if (D.getArgumentNumber() >= ArgTys.size())
      return IsDeferredCheck || DeferCheck(Ty);
    auto *ReferenceType = dyn_cast<VectorType>(ArgTys[D.getArgumentNumber()]);
    auto *ThisArgVecTy = dyn_cast<VectorType>(Ty);
    if (!ThisArgVecTy || !ReferenceType)
      return true;
    return ThisArgVecTy != VectorType::getInteger(ReferenceType);
  }
  }
  llvm_unreachable("unhandled");
}

Intrinsic::MatchIntrinsicTypesResult
Intrinsic::matchIntrinsicSignature(FunctionType *FTy,
                                   ArrayRef<Intrinsic::IITDescriptor> &Infos,
                                   SmallVectorImpl<Type *> &ArgTys) {
  SmallVector<DeferredIntrinsicMatchPair, 2> DeferredChecks;
  if (matchIntrinsicType(FTy->getReturnType(), Infos, ArgTys, DeferredChecks,
                         false))
    return MatchIntrinsicTypes_NoMatchRet;

  unsigned NumDeferredReturnChecks = DeferredChecks.size();

  for (auto *Ty : FTy->params())
    if (matchIntrinsicType(Ty, Infos, ArgTys, DeferredChecks, false))
      return MatchIntrinsicTypes_NoMatchArg;

  for (unsigned I = 0, E = DeferredChecks.size(); I != E; ++I) {
    DeferredIntrinsicMatchPair &Check = DeferredChecks[I];
    if (matchIntrinsicType(Check.first, Check.second, ArgTys, DeferredChecks,
                           true))
      return I < NumDeferredReturnChecks ? MatchIntrinsicTypes_NoMatchRet
                                         : MatchIntrinsicTypes_NoMatchArg;
  }

  return MatchIntrinsicTypes_Match;
}

bool Intrinsic::matchIntrinsicVarArg(
    bool isVarArg, ArrayRef<Intrinsic::IITDescriptor> &Infos) {
  // If there are no descriptors left, then it can't be a vararg.
  if (Infos.empty())
    return isVarArg;

  // There should be only one descriptor remaining at this point.
  if (Infos.size() != 1)
    return true;

  // Check and verify the descriptor.
  IITDescriptor D = Infos.front();
  Infos = Infos.slice(1);
  if (D.Kind == IITDescriptor::VarArg)
    return !isVarArg;

  return true;
}

bool Intrinsic::getIntrinsicSignature(Intrinsic::ID ID, FunctionType *FT,
                                      SmallVectorImpl<Type *> &ArgTys) {
  if (!ID)
    return false;

  SmallVector<Intrinsic::IITDescriptor, 8> Table;
  getIntrinsicInfoTableEntries(ID, Table);
  ArrayRef<Intrinsic::IITDescriptor> TableRef = Table;

  if (Intrinsic::matchIntrinsicSignature(FT, TableRef, ArgTys) !=
      Intrinsic::MatchIntrinsicTypesResult::MatchIntrinsicTypes_Match) {
    return false;
  }
  if (Intrinsic::matchIntrinsicVarArg(FT->isVarArg(), TableRef))
    return false;
  return true;
}

bool Intrinsic::getIntrinsicSignature(Function *F,
                                      SmallVectorImpl<Type *> &ArgTys) {
  return getIntrinsicSignature(F->getIntrinsicID(), F->getFunctionType(),
                               ArgTys);
}

std::optional<Function *> Intrinsic::remangleIntrinsicFunction(Function *F) {
  SmallVector<Type *, 4> ArgTys;
  if (!getIntrinsicSignature(F, ArgTys))
    return std::nullopt;

  Intrinsic::ID ID = F->getIntrinsicID();
  StringRef Name = F->getName();
  std::string WantedName =
      Intrinsic::getName(ID, ArgTys, F->getParent(), F->getFunctionType());
  if (Name == WantedName)
    return std::nullopt;

  Function *NewDecl = [&] {
    if (auto *ExistingGV = F->getParent()->getNamedValue(WantedName)) {
      if (auto *ExistingF = dyn_cast<Function>(ExistingGV))
        if (ExistingF->getFunctionType() == F->getFunctionType())
          return ExistingF;

      // The name already exists, but is not a function or has the wrong
      // prototype. Make place for the new one by renaming the old version.
      // Either this old version will be removed later on or the module is
      // invalid and we'll get an error.
      ExistingGV->setName(WantedName + ".renamed");
    }
    return Intrinsic::getOrInsertDeclaration(F->getParent(), ID, ArgTys);
  }();

  NewDecl->setCallingConv(F->getCallingConv());
  assert(NewDecl->getFunctionType() == F->getFunctionType() &&
         "Shouldn't change the signature");
  return NewDecl;
}

struct InterleaveIntrinsic {
  Intrinsic::ID Interleave, Deinterleave;
};

static InterleaveIntrinsic InterleaveIntrinsics[] = {
    {Intrinsic::vector_interleave2, Intrinsic::vector_deinterleave2},
    {Intrinsic::vector_interleave3, Intrinsic::vector_deinterleave3},
    {Intrinsic::vector_interleave4, Intrinsic::vector_deinterleave4},
    {Intrinsic::vector_interleave5, Intrinsic::vector_deinterleave5},
    {Intrinsic::vector_interleave6, Intrinsic::vector_deinterleave6},
    {Intrinsic::vector_interleave7, Intrinsic::vector_deinterleave7},
    {Intrinsic::vector_interleave8, Intrinsic::vector_deinterleave8},
};

Intrinsic::ID Intrinsic::getInterleaveIntrinsicID(unsigned Factor) {
  assert(Factor >= 2 && Factor <= 8 && "Unexpected factor");
  return InterleaveIntrinsics[Factor - 2].Interleave;
}

Intrinsic::ID Intrinsic::getDeinterleaveIntrinsicID(unsigned Factor) {
  assert(Factor >= 2 && Factor <= 8 && "Unexpected factor");
  return InterleaveIntrinsics[Factor - 2].Deinterleave;
}
