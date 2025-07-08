//===- Type.cpp - Implement the Type class --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Type class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Type.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include <cassert>
#include <utility>

using namespace llvm;

//===----------------------------------------------------------------------===//
//                         Type Class Implementation
//===----------------------------------------------------------------------===//

Type *Type::getPrimitiveType(LLVMContext &C, TypeID IDNumber) {
  switch (IDNumber) {
  case VoidTyID      : return getVoidTy(C);
  case HalfTyID      : return getHalfTy(C);
  case BFloatTyID    : return getBFloatTy(C);
  case FloatTyID     : return getFloatTy(C);
  case DoubleTyID    : return getDoubleTy(C);
  case X86_FP80TyID  : return getX86_FP80Ty(C);
  case FP128TyID     : return getFP128Ty(C);
  case PPC_FP128TyID : return getPPC_FP128Ty(C);
  case LabelTyID     : return getLabelTy(C);
  case MetadataTyID  : return getMetadataTy(C);
  case X86_AMXTyID   : return getX86_AMXTy(C);
  case TokenTyID     : return getTokenTy(C);
  default:
    return nullptr;
  }
}

bool Type::isIntegerTy(unsigned Bitwidth) const {
  return isIntegerTy() && cast<IntegerType>(this)->getBitWidth() == Bitwidth;
}

bool Type::isScalableTy(SmallPtrSetImpl<const Type *> &Visited) const {
  if (const auto *ATy = dyn_cast<ArrayType>(this))
    return ATy->getElementType()->isScalableTy(Visited);
  if (const auto *STy = dyn_cast<StructType>(this))
    return STy->isScalableTy(Visited);
  return getTypeID() == ScalableVectorTyID || isScalableTargetExtTy();
}

bool Type::isScalableTy() const {
  SmallPtrSet<const Type *, 4> Visited;
  return isScalableTy(Visited);
}

bool Type::containsNonGlobalTargetExtType(
    SmallPtrSetImpl<const Type *> &Visited) const {
  if (const auto *ATy = dyn_cast<ArrayType>(this))
    return ATy->getElementType()->containsNonGlobalTargetExtType(Visited);
  if (const auto *STy = dyn_cast<StructType>(this))
    return STy->containsNonGlobalTargetExtType(Visited);
  if (auto *TT = dyn_cast<TargetExtType>(this))
    return !TT->hasProperty(TargetExtType::CanBeGlobal);
  return false;
}

bool Type::containsNonGlobalTargetExtType() const {
  SmallPtrSet<const Type *, 4> Visited;
  return containsNonGlobalTargetExtType(Visited);
}

bool Type::containsNonLocalTargetExtType(
    SmallPtrSetImpl<const Type *> &Visited) const {
  if (const auto *ATy = dyn_cast<ArrayType>(this))
    return ATy->getElementType()->containsNonLocalTargetExtType(Visited);
  if (const auto *STy = dyn_cast<StructType>(this))
    return STy->containsNonLocalTargetExtType(Visited);
  if (auto *TT = dyn_cast<TargetExtType>(this))
    return !TT->hasProperty(TargetExtType::CanBeLocal);
  return false;
}

bool Type::containsNonLocalTargetExtType() const {
  SmallPtrSet<const Type *, 4> Visited;
  return containsNonLocalTargetExtType(Visited);
}

const fltSemantics &Type::getFltSemantics() const {
  switch (getTypeID()) {
  case HalfTyID: return APFloat::IEEEhalf();
  case BFloatTyID: return APFloat::BFloat();
  case FloatTyID: return APFloat::IEEEsingle();
  case DoubleTyID: return APFloat::IEEEdouble();
  case X86_FP80TyID: return APFloat::x87DoubleExtended();
  case FP128TyID: return APFloat::IEEEquad();
  case PPC_FP128TyID: return APFloat::PPCDoubleDouble();
  default: llvm_unreachable("Invalid floating type");
  }
}

bool Type::isScalableTargetExtTy() const {
  if (auto *TT = dyn_cast<TargetExtType>(this))
    return isa<ScalableVectorType>(TT->getLayoutType());
  return false;
}

Type *Type::getFloatingPointTy(LLVMContext &C, const fltSemantics &S) {
  Type *Ty;
  if (&S == &APFloat::IEEEhalf())
    Ty = Type::getHalfTy(C);
  else if (&S == &APFloat::BFloat())
    Ty = Type::getBFloatTy(C);
  else if (&S == &APFloat::IEEEsingle())
    Ty = Type::getFloatTy(C);
  else if (&S == &APFloat::IEEEdouble())
    Ty = Type::getDoubleTy(C);
  else if (&S == &APFloat::x87DoubleExtended())
    Ty = Type::getX86_FP80Ty(C);
  else if (&S == &APFloat::IEEEquad())
    Ty = Type::getFP128Ty(C);
  else {
    assert(&S == &APFloat::PPCDoubleDouble() && "Unknown FP format");
    Ty = Type::getPPC_FP128Ty(C);
  }
  return Ty;
}

bool Type::isRISCVVectorTupleTy() const {
  if (!isTargetExtTy())
    return false;

  return cast<TargetExtType>(this)->getName() == "riscv.vector.tuple";
}

bool Type::canLosslesslyBitCastTo(Type *Ty) const {
  // Identity cast means no change so return true
  if (this == Ty)
    return true;

  // They are not convertible unless they are at least first class types
  if (!this->isFirstClassType() || !Ty->isFirstClassType())
    return false;

  // Vector -> Vector conversions are always lossless if the two vector types
  // have the same size, otherwise not.
  if (isa<VectorType>(this) && isa<VectorType>(Ty))
    return getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits();

  //  8192-bit fixed width vector types can be losslessly converted to x86amx.
  if (((isa<FixedVectorType>(this)) && Ty->isX86_AMXTy()) &&
      getPrimitiveSizeInBits().getFixedValue() == 8192)
    return true;
  if ((isX86_AMXTy() && isa<FixedVectorType>(Ty)) &&
      Ty->getPrimitiveSizeInBits().getFixedValue() == 8192)
    return true;

  // Conservatively assume we can't losslessly convert between pointers with
  // different address spaces.
  return false;
}

bool Type::isEmptyTy() const {
  if (auto *ATy = dyn_cast<ArrayType>(this)) {
    unsigned NumElements = ATy->getNumElements();
    return NumElements == 0 || ATy->getElementType()->isEmptyTy();
  }

  if (auto *STy = dyn_cast<StructType>(this)) {
    unsigned NumElements = STy->getNumElements();
    for (unsigned i = 0; i < NumElements; ++i)
      if (!STy->getElementType(i)->isEmptyTy())
        return false;
    return true;
  }

  return false;
}

TypeSize Type::getPrimitiveSizeInBits() const {
  switch (getTypeID()) {
  case Type::HalfTyID:
    return TypeSize::getFixed(16);
  case Type::BFloatTyID:
    return TypeSize::getFixed(16);
  case Type::FloatTyID:
    return TypeSize::getFixed(32);
  case Type::DoubleTyID:
    return TypeSize::getFixed(64);
  case Type::X86_FP80TyID:
    return TypeSize::getFixed(80);
  case Type::FP128TyID:
    return TypeSize::getFixed(128);
  case Type::PPC_FP128TyID:
    return TypeSize::getFixed(128);
  case Type::X86_AMXTyID:
    return TypeSize::getFixed(8192);
  case Type::IntegerTyID:
    return TypeSize::getFixed(cast<IntegerType>(this)->getBitWidth());
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID: {
    const VectorType *VTy = cast<VectorType>(this);
    ElementCount EC = VTy->getElementCount();
    TypeSize ETS = VTy->getElementType()->getPrimitiveSizeInBits();
    assert(!ETS.isScalable() && "Vector type should have fixed-width elements");
    return {ETS.getFixedValue() * EC.getKnownMinValue(), EC.isScalable()};
  }
  default:
    return TypeSize::getFixed(0);
  }
}

unsigned Type::getScalarSizeInBits() const {
  // It is safe to assume that the scalar types have a fixed size.
  return getScalarType()->getPrimitiveSizeInBits().getFixedValue();
}

int Type::getFPMantissaWidth() const {
  if (auto *VTy = dyn_cast<VectorType>(this))
    return VTy->getElementType()->getFPMantissaWidth();
  assert(isFloatingPointTy() && "Not a floating point type!");
  if (getTypeID() == HalfTyID) return 11;
  if (getTypeID() == BFloatTyID) return 8;
  if (getTypeID() == FloatTyID) return 24;
  if (getTypeID() == DoubleTyID) return 53;
  if (getTypeID() == X86_FP80TyID) return 64;
  if (getTypeID() == FP128TyID) return 113;
  assert(getTypeID() == PPC_FP128TyID && "unknown fp type");
  return -1;
}

bool Type::isFirstClassType() const {
  switch (getTypeID()) {
    default:
      return true;
    case FunctionTyID:
    case VoidTyID:
      return false;
    case StructTyID: {
      auto *ST = cast<StructType>(this);
      return !ST->isOpaque();
    }
  }
}

bool Type::isSizedDerivedType(SmallPtrSetImpl<Type*> *Visited) const {
  if (auto *ATy = dyn_cast<ArrayType>(this))
    return ATy->getElementType()->isSized(Visited);

  if (auto *VTy = dyn_cast<VectorType>(this))
    return VTy->getElementType()->isSized(Visited);

  if (auto *TTy = dyn_cast<TargetExtType>(this))
    return TTy->getLayoutType()->isSized(Visited);

  return cast<StructType>(this)->isSized(Visited);
}

//===----------------------------------------------------------------------===//
//                          Primitive 'Type' data
//===----------------------------------------------------------------------===//

Type *Type::getVoidTy(LLVMContext &C) { return &C.pImpl->VoidTy; }
Type *Type::getLabelTy(LLVMContext &C) { return &C.pImpl->LabelTy; }
Type *Type::getHalfTy(LLVMContext &C) { return &C.pImpl->HalfTy; }
Type *Type::getBFloatTy(LLVMContext &C) { return &C.pImpl->BFloatTy; }
Type *Type::getFloatTy(LLVMContext &C) { return &C.pImpl->FloatTy; }
Type *Type::getDoubleTy(LLVMContext &C) { return &C.pImpl->DoubleTy; }
Type *Type::getMetadataTy(LLVMContext &C) { return &C.pImpl->MetadataTy; }
Type *Type::getTokenTy(LLVMContext &C) { return &C.pImpl->TokenTy; }
Type *Type::getX86_FP80Ty(LLVMContext &C) { return &C.pImpl->X86_FP80Ty; }
Type *Type::getFP128Ty(LLVMContext &C) { return &C.pImpl->FP128Ty; }
Type *Type::getPPC_FP128Ty(LLVMContext &C) { return &C.pImpl->PPC_FP128Ty; }
Type *Type::getX86_AMXTy(LLVMContext &C) { return &C.pImpl->X86_AMXTy; }

IntegerType *Type::getInt1Ty(LLVMContext &C) { return &C.pImpl->Int1Ty; }
IntegerType *Type::getInt8Ty(LLVMContext &C) { return &C.pImpl->Int8Ty; }
IntegerType *Type::getInt16Ty(LLVMContext &C) { return &C.pImpl->Int16Ty; }
IntegerType *Type::getInt32Ty(LLVMContext &C) { return &C.pImpl->Int32Ty; }
IntegerType *Type::getInt64Ty(LLVMContext &C) { return &C.pImpl->Int64Ty; }
IntegerType *Type::getInt128Ty(LLVMContext &C) { return &C.pImpl->Int128Ty; }

IntegerType *Type::getIntNTy(LLVMContext &C, unsigned N) {
  return IntegerType::get(C, N);
}

Type *Type::getWasm_ExternrefTy(LLVMContext &C) {
  // opaque pointer in addrspace(10)
  static PointerType *Ty = PointerType::get(C, 10);
  return Ty;
}

Type *Type::getWasm_FuncrefTy(LLVMContext &C) {
  // opaque pointer in addrspace(20)
  static PointerType *Ty = PointerType::get(C, 20);
  return Ty;
}

//===----------------------------------------------------------------------===//
//                       IntegerType Implementation
//===----------------------------------------------------------------------===//

IntegerType *IntegerType::get(LLVMContext &C, unsigned NumBits) {
  assert(NumBits >= MIN_INT_BITS && "bitwidth too small");
  assert(NumBits <= MAX_INT_BITS && "bitwidth too large");

  // Check for the built-in integer types
  switch (NumBits) {
  case   1: return cast<IntegerType>(Type::getInt1Ty(C));
  case   8: return cast<IntegerType>(Type::getInt8Ty(C));
  case  16: return cast<IntegerType>(Type::getInt16Ty(C));
  case  32: return cast<IntegerType>(Type::getInt32Ty(C));
  case  64: return cast<IntegerType>(Type::getInt64Ty(C));
  case 128: return cast<IntegerType>(Type::getInt128Ty(C));
  default:
    break;
  }

  IntegerType *&Entry = C.pImpl->IntegerTypes[NumBits];

  if (!Entry)
    Entry = new (C.pImpl->Alloc) IntegerType(C, NumBits);

  return Entry;
}

APInt IntegerType::getMask() const { return APInt::getAllOnes(getBitWidth()); }

//===----------------------------------------------------------------------===//
//                       FunctionType Implementation
//===----------------------------------------------------------------------===//

FunctionType::FunctionType(Type *Result, ArrayRef<Type*> Params,
                           bool IsVarArgs)
  : Type(Result->getContext(), FunctionTyID) {
  Type **SubTys = reinterpret_cast<Type**>(this+1);
  assert(isValidReturnType(Result) && "invalid return type for function");
  setSubclassData(IsVarArgs);

  SubTys[0] = Result;

  for (unsigned i = 0, e = Params.size(); i != e; ++i) {
    assert(isValidArgumentType(Params[i]) &&
           "Not a valid type for function argument!");
    SubTys[i+1] = Params[i];
  }

  ContainedTys = SubTys;
  NumContainedTys = Params.size() + 1; // + 1 for result type
}

// This is the factory function for the FunctionType class.
FunctionType *FunctionType::get(Type *ReturnType,
                                ArrayRef<Type*> Params, bool isVarArg) {
  LLVMContextImpl *pImpl = ReturnType->getContext().pImpl;
  const FunctionTypeKeyInfo::KeyTy Key(ReturnType, Params, isVarArg);
  FunctionType *FT;
  // Since we only want to allocate a fresh function type in case none is found
  // and we don't want to perform two lookups (one for checking if existent and
  // one for inserting the newly allocated one), here we instead lookup based on
  // Key and update the reference to the function type in-place to a newly
  // allocated one if not found.
  auto Insertion = pImpl->FunctionTypes.insert_as(nullptr, Key);
  if (Insertion.second) {
    // The function type was not found. Allocate one and update FunctionTypes
    // in-place.
    FT = (FunctionType *)pImpl->Alloc.Allocate(
        sizeof(FunctionType) + sizeof(Type *) * (Params.size() + 1),
        alignof(FunctionType));
    new (FT) FunctionType(ReturnType, Params, isVarArg);
    *Insertion.first = FT;
  } else {
    // The function type was found. Just return it.
    FT = *Insertion.first;
  }
  return FT;
}

FunctionType *FunctionType::get(Type *Result, bool isVarArg) {
  return get(Result, {}, isVarArg);
}

bool FunctionType::isValidReturnType(Type *RetTy) {
  return !RetTy->isFunctionTy() && !RetTy->isLabelTy() &&
  !RetTy->isMetadataTy();
}

bool FunctionType::isValidArgumentType(Type *ArgTy) {
  return ArgTy->isFirstClassType() && !ArgTy->isLabelTy();
}

//===----------------------------------------------------------------------===//
//                       StructType Implementation
//===----------------------------------------------------------------------===//

// Primitive Constructors.

StructType *StructType::get(LLVMContext &Context, ArrayRef<Type*> ETypes,
                            bool isPacked) {
  LLVMContextImpl *pImpl = Context.pImpl;
  const AnonStructTypeKeyInfo::KeyTy Key(ETypes, isPacked);

  StructType *ST;
  // Since we only want to allocate a fresh struct type in case none is found
  // and we don't want to perform two lookups (one for checking if existent and
  // one for inserting the newly allocated one), here we instead lookup based on
  // Key and update the reference to the struct type in-place to a newly
  // allocated one if not found.
  auto Insertion = pImpl->AnonStructTypes.insert_as(nullptr, Key);
  if (Insertion.second) {
    // The struct type was not found. Allocate one and update AnonStructTypes
    // in-place.
    ST = new (Context.pImpl->Alloc) StructType(Context);
    ST->setSubclassData(SCDB_IsLiteral);  // Literal struct.
    ST->setBody(ETypes, isPacked);
    *Insertion.first = ST;
  } else {
    // The struct type was found. Just return it.
    ST = *Insertion.first;
  }

  return ST;
}

bool StructType::isScalableTy(SmallPtrSetImpl<const Type *> &Visited) const {
  if ((getSubclassData() & SCDB_ContainsScalableVector) != 0)
    return true;

  if ((getSubclassData() & SCDB_NotContainsScalableVector) != 0)
    return false;

  if (!Visited.insert(this).second)
    return false;

  for (Type *Ty : elements()) {
    if (Ty->isScalableTy(Visited)) {
      const_cast<StructType *>(this)->setSubclassData(
          getSubclassData() | SCDB_ContainsScalableVector);
      return true;
    }
  }

  // For structures that are opaque, return false but do not set the
  // SCDB_NotContainsScalableVector flag since it may gain scalable vector type
  // when it becomes non-opaque.
  if (!isOpaque())
    const_cast<StructType *>(this)->setSubclassData(
        getSubclassData() | SCDB_NotContainsScalableVector);
  return false;
}

bool StructType::containsNonGlobalTargetExtType(
    SmallPtrSetImpl<const Type *> &Visited) const {
  if ((getSubclassData() & SCDB_ContainsNonGlobalTargetExtType) != 0)
    return true;

  if ((getSubclassData() & SCDB_NotContainsNonGlobalTargetExtType) != 0)
    return false;

  if (!Visited.insert(this).second)
    return false;

  for (Type *Ty : elements()) {
    if (Ty->containsNonGlobalTargetExtType(Visited)) {
      const_cast<StructType *>(this)->setSubclassData(
          getSubclassData() | SCDB_ContainsNonGlobalTargetExtType);
      return true;
    }
  }

  // For structures that are opaque, return false but do not set the
  // SCDB_NotContainsNonGlobalTargetExtType flag since it may gain non-global
  // target extension types when it becomes non-opaque.
  if (!isOpaque())
    const_cast<StructType *>(this)->setSubclassData(
        getSubclassData() | SCDB_NotContainsNonGlobalTargetExtType);
  return false;
}

bool StructType::containsNonLocalTargetExtType(
    SmallPtrSetImpl<const Type *> &Visited) const {
  if ((getSubclassData() & SCDB_ContainsNonLocalTargetExtType) != 0)
    return true;

  if ((getSubclassData() & SCDB_NotContainsNonLocalTargetExtType) != 0)
    return false;

  if (!Visited.insert(this).second)
    return false;

  for (Type *Ty : elements()) {
    if (Ty->containsNonLocalTargetExtType(Visited)) {
      const_cast<StructType *>(this)->setSubclassData(
          getSubclassData() | SCDB_ContainsNonLocalTargetExtType);
      return true;
    }
  }

  // For structures that are opaque, return false but do not set the
  // SCDB_NotContainsNonLocalTargetExtType flag since it may gain non-local
  // target extension types when it becomes non-opaque.
  if (!isOpaque())
    const_cast<StructType *>(this)->setSubclassData(
        getSubclassData() | SCDB_NotContainsNonLocalTargetExtType);
  return false;
}

bool StructType::containsHomogeneousScalableVectorTypes() const {
  if (getNumElements() <= 0 || !isa<ScalableVectorType>(elements().front()))
    return false;
  return containsHomogeneousTypes();
}

bool StructType::containsHomogeneousTypes() const {
  ArrayRef<Type *> ElementTys = elements();
  return !ElementTys.empty() && all_equal(ElementTys);
}

void StructType::setBody(ArrayRef<Type*> Elements, bool isPacked) {
  cantFail(setBodyOrError(Elements, isPacked));
}

Error StructType::setBodyOrError(ArrayRef<Type *> Elements, bool isPacked) {
  assert(isOpaque() && "Struct body already set!");

  if (auto E = checkBody(Elements))
    return E;

  setSubclassData(getSubclassData() | SCDB_HasBody);
  if (isPacked)
    setSubclassData(getSubclassData() | SCDB_Packed);

  NumContainedTys = Elements.size();
  ContainedTys = Elements.empty()
                     ? nullptr
                     : Elements.copy(getContext().pImpl->Alloc).data();

  return Error::success();
}

Error StructType::checkBody(ArrayRef<Type *> Elements) {
  SmallSetVector<Type *, 4> Worklist(Elements.begin(), Elements.end());
  for (unsigned I = 0; I < Worklist.size(); ++I) {
    Type *Ty = Worklist[I];
    if (Ty == this)
      return createStringError(Twine("identified structure type '") +
                               getName() + "' is recursive");
    Worklist.insert_range(Ty->subtypes());
  }
  return Error::success();
}

void StructType::setName(StringRef Name) {
  if (Name == getName()) return;

  StringMap<StructType *> &SymbolTable = getContext().pImpl->NamedStructTypes;

  using EntryTy = StringMap<StructType *>::MapEntryTy;

  // If this struct already had a name, remove its symbol table entry. Don't
  // delete the data yet because it may be part of the new name.
  if (SymbolTableEntry)
    SymbolTable.remove((EntryTy *)SymbolTableEntry);

  // If this is just removing the name, we're done.
  if (Name.empty()) {
    if (SymbolTableEntry) {
      // Delete the old string data.
      ((EntryTy *)SymbolTableEntry)->Destroy(SymbolTable.getAllocator());
      SymbolTableEntry = nullptr;
    }
    return;
  }

  // Look up the entry for the name.
  auto IterBool =
      getContext().pImpl->NamedStructTypes.insert(std::make_pair(Name, this));

  // While we have a name collision, try a random rename.
  if (!IterBool.second) {
    SmallString<64> TempStr(Name);
    TempStr.push_back('.');
    raw_svector_ostream TmpStream(TempStr);
    unsigned NameSize = Name.size();

    do {
      TempStr.resize(NameSize + 1);
      TmpStream << getContext().pImpl->NamedStructTypesUniqueID++;

      IterBool = getContext().pImpl->NamedStructTypes.insert(
          std::make_pair(TmpStream.str(), this));
    } while (!IterBool.second);
  }

  // Delete the old string data.
  if (SymbolTableEntry)
    ((EntryTy *)SymbolTableEntry)->Destroy(SymbolTable.getAllocator());
  SymbolTableEntry = &*IterBool.first;
}

//===----------------------------------------------------------------------===//
// StructType Helper functions.

StructType *StructType::create(LLVMContext &Context, StringRef Name) {
  StructType *ST = new (Context.pImpl->Alloc) StructType(Context);
  if (!Name.empty())
    ST->setName(Name);
  return ST;
}

StructType *StructType::get(LLVMContext &Context, bool isPacked) {
  return get(Context, {}, isPacked);
}

StructType *StructType::create(LLVMContext &Context, ArrayRef<Type*> Elements,
                               StringRef Name, bool isPacked) {
  StructType *ST = create(Context, Name);
  ST->setBody(Elements, isPacked);
  return ST;
}

StructType *StructType::create(LLVMContext &Context, ArrayRef<Type*> Elements) {
  return create(Context, Elements, StringRef());
}

StructType *StructType::create(LLVMContext &Context) {
  return create(Context, StringRef());
}

StructType *StructType::create(ArrayRef<Type*> Elements, StringRef Name,
                               bool isPacked) {
  assert(!Elements.empty() &&
         "This method may not be invoked with an empty list");
  return create(Elements[0]->getContext(), Elements, Name, isPacked);
}

StructType *StructType::create(ArrayRef<Type*> Elements) {
  assert(!Elements.empty() &&
         "This method may not be invoked with an empty list");
  return create(Elements[0]->getContext(), Elements, StringRef());
}

bool StructType::isSized(SmallPtrSetImpl<Type*> *Visited) const {
  if ((getSubclassData() & SCDB_IsSized) != 0)
    return true;
  if (isOpaque())
    return false;

  if (Visited && !Visited->insert(const_cast<StructType*>(this)).second)
    return false;

  // Okay, our struct is sized if all of the elements are, but if one of the
  // elements is opaque, the struct isn't sized *yet*, but may become sized in
  // the future, so just bail out without caching.
  // The ONLY special case inside a struct that is considered sized is when the
  // elements are homogeneous of a scalable vector type.
  if (containsHomogeneousScalableVectorTypes()) {
    const_cast<StructType *>(this)->setSubclassData(getSubclassData() |
                                                    SCDB_IsSized);
    return true;
  }
  for (Type *Ty : elements()) {
    // If the struct contains a scalable vector type, don't consider it sized.
    // This prevents it from being used in loads/stores/allocas/GEPs. The ONLY
    // special case right now is a structure of homogenous scalable vector
    // types and is handled by the if-statement before this for-loop.
    if (Ty->isScalableTy())
      return false;
    if (!Ty->isSized(Visited))
      return false;
  }

  // Here we cheat a bit and cast away const-ness. The goal is to memoize when
  // we find a sized type, as types can only move from opaque to sized, not the
  // other way.
  const_cast<StructType*>(this)->setSubclassData(
    getSubclassData() | SCDB_IsSized);
  return true;
}

StringRef StructType::getName() const {
  assert(!isLiteral() && "Literal structs never have names");
  if (!SymbolTableEntry) return StringRef();

  return ((StringMapEntry<StructType*> *)SymbolTableEntry)->getKey();
}

bool StructType::isValidElementType(Type *ElemTy) {
  return !ElemTy->isVoidTy() && !ElemTy->isLabelTy() &&
         !ElemTy->isMetadataTy() && !ElemTy->isFunctionTy() &&
         !ElemTy->isTokenTy();
}

bool StructType::isLayoutIdentical(StructType *Other) const {
  if (this == Other) return true;

  if (isPacked() != Other->isPacked())
    return false;

  return elements() == Other->elements();
}

Type *StructType::getTypeAtIndex(const Value *V) const {
  unsigned Idx = (unsigned)cast<Constant>(V)->getUniqueInteger().getZExtValue();
  assert(indexValid(Idx) && "Invalid structure index!");
  return getElementType(Idx);
}

bool StructType::indexValid(const Value *V) const {
  // Structure indexes require (vectors of) 32-bit integer constants.  In the
  // vector case all of the indices must be equal.
  if (!V->getType()->isIntOrIntVectorTy(32))
    return false;
  if (isa<ScalableVectorType>(V->getType()))
    return false;
  const Constant *C = dyn_cast<Constant>(V);
  if (C && V->getType()->isVectorTy())
    C = C->getSplatValue();
  const ConstantInt *CU = dyn_cast_or_null<ConstantInt>(C);
  return CU && CU->getZExtValue() < getNumElements();
}

StructType *StructType::getTypeByName(LLVMContext &C, StringRef Name) {
  return C.pImpl->NamedStructTypes.lookup(Name);
}

//===----------------------------------------------------------------------===//
//                           ArrayType Implementation
//===----------------------------------------------------------------------===//

ArrayType::ArrayType(Type *ElType, uint64_t NumEl)
    : Type(ElType->getContext(), ArrayTyID), ContainedType(ElType),
      NumElements(NumEl) {
  ContainedTys = &ContainedType;
  NumContainedTys = 1;
}

ArrayType *ArrayType::get(Type *ElementType, uint64_t NumElements) {
  assert(isValidElementType(ElementType) && "Invalid type for array element!");

  LLVMContextImpl *pImpl = ElementType->getContext().pImpl;
  ArrayType *&Entry =
    pImpl->ArrayTypes[std::make_pair(ElementType, NumElements)];

  if (!Entry)
    Entry = new (pImpl->Alloc) ArrayType(ElementType, NumElements);
  return Entry;
}

bool ArrayType::isValidElementType(Type *ElemTy) {
  return !ElemTy->isVoidTy() && !ElemTy->isLabelTy() &&
         !ElemTy->isMetadataTy() && !ElemTy->isFunctionTy() &&
         !ElemTy->isTokenTy() && !ElemTy->isX86_AMXTy();
}

//===----------------------------------------------------------------------===//
//                          VectorType Implementation
//===----------------------------------------------------------------------===//

VectorType::VectorType(Type *ElType, unsigned EQ, Type::TypeID TID)
    : Type(ElType->getContext(), TID), ContainedType(ElType),
      ElementQuantity(EQ) {
  ContainedTys = &ContainedType;
  NumContainedTys = 1;
}

VectorType *VectorType::get(Type *ElementType, ElementCount EC) {
  if (EC.isScalable())
    return ScalableVectorType::get(ElementType, EC.getKnownMinValue());
  else
    return FixedVectorType::get(ElementType, EC.getKnownMinValue());
}

bool VectorType::isValidElementType(Type *ElemTy) {
  if (ElemTy->isIntegerTy() || ElemTy->isFloatingPointTy() ||
      ElemTy->isPointerTy() || ElemTy->getTypeID() == TypedPointerTyID)
    return true;
  if (auto *TTy = dyn_cast<TargetExtType>(ElemTy))
    return TTy->hasProperty(TargetExtType::CanBeVectorElement);
  return false;
}

//===----------------------------------------------------------------------===//
//                        FixedVectorType Implementation
//===----------------------------------------------------------------------===//

FixedVectorType *FixedVectorType::get(Type *ElementType, unsigned NumElts) {
  assert(NumElts > 0 && "#Elements of a VectorType must be greater than 0");
  assert(isValidElementType(ElementType) && "Element type of a VectorType must "
                                            "be an integer, floating point, "
                                            "pointer type, or a valid target "
                                            "extension type.");

  auto EC = ElementCount::getFixed(NumElts);

  LLVMContextImpl *pImpl = ElementType->getContext().pImpl;
  VectorType *&Entry = ElementType->getContext()
                           .pImpl->VectorTypes[std::make_pair(ElementType, EC)];

  if (!Entry)
    Entry = new (pImpl->Alloc) FixedVectorType(ElementType, NumElts);
  return cast<FixedVectorType>(Entry);
}

//===----------------------------------------------------------------------===//
//                       ScalableVectorType Implementation
//===----------------------------------------------------------------------===//

ScalableVectorType *ScalableVectorType::get(Type *ElementType,
                                            unsigned MinNumElts) {
  assert(MinNumElts > 0 && "#Elements of a VectorType must be greater than 0");
  assert(isValidElementType(ElementType) && "Element type of a VectorType must "
                                            "be an integer, floating point, or "
                                            "pointer type.");

  auto EC = ElementCount::getScalable(MinNumElts);

  LLVMContextImpl *pImpl = ElementType->getContext().pImpl;
  VectorType *&Entry = ElementType->getContext()
                           .pImpl->VectorTypes[std::make_pair(ElementType, EC)];

  if (!Entry)
    Entry = new (pImpl->Alloc) ScalableVectorType(ElementType, MinNumElts);
  return cast<ScalableVectorType>(Entry);
}

//===----------------------------------------------------------------------===//
//                         PointerType Implementation
//===----------------------------------------------------------------------===//

PointerType *PointerType::get(Type *EltTy, unsigned AddressSpace) {
  assert(EltTy && "Can't get a pointer to <null> type!");
  assert(isValidElementType(EltTy) && "Invalid type for pointer element!");

  // Automatically convert typed pointers to opaque pointers.
  return get(EltTy->getContext(), AddressSpace);
}

PointerType *PointerType::get(LLVMContext &C, unsigned AddressSpace) {
  LLVMContextImpl *CImpl = C.pImpl;

  // Since AddressSpace #0 is the common case, we special case it.
  PointerType *&Entry = AddressSpace == 0 ? CImpl->AS0PointerType
                                          : CImpl->PointerTypes[AddressSpace];

  if (!Entry)
    Entry = new (CImpl->Alloc) PointerType(C, AddressSpace);
  return Entry;
}

PointerType::PointerType(LLVMContext &C, unsigned AddrSpace)
    : Type(C, PointerTyID) {
  setSubclassData(AddrSpace);
}

PointerType *Type::getPointerTo(unsigned AddrSpace) const {
  return PointerType::get(getContext(), AddrSpace);
}

bool PointerType::isValidElementType(Type *ElemTy) {
  return !ElemTy->isVoidTy() && !ElemTy->isLabelTy() &&
         !ElemTy->isMetadataTy() && !ElemTy->isTokenTy() &&
         !ElemTy->isX86_AMXTy();
}

bool PointerType::isLoadableOrStorableType(Type *ElemTy) {
  return isValidElementType(ElemTy) && !ElemTy->isFunctionTy();
}

//===----------------------------------------------------------------------===//
//                       TargetExtType Implementation
//===----------------------------------------------------------------------===//

TargetExtType::TargetExtType(LLVMContext &C, StringRef Name,
                             ArrayRef<Type *> Types, ArrayRef<unsigned> Ints)
    : Type(C, TargetExtTyID), Name(C.pImpl->Saver.save(Name)) {
  NumContainedTys = Types.size();

  // Parameter storage immediately follows the class in allocation.
  Type **Params = reinterpret_cast<Type **>(this + 1);
  ContainedTys = Params;
  for (Type *T : Types)
    *Params++ = T;

  setSubclassData(Ints.size());
  unsigned *IntParamSpace = reinterpret_cast<unsigned *>(Params);
  IntParams = IntParamSpace;
  for (unsigned IntParam : Ints)
    *IntParamSpace++ = IntParam;
}

TargetExtType *TargetExtType::get(LLVMContext &C, StringRef Name,
                                  ArrayRef<Type *> Types,
                                  ArrayRef<unsigned> Ints) {
  return cantFail(getOrError(C, Name, Types, Ints));
}

Expected<TargetExtType *> TargetExtType::getOrError(LLVMContext &C,
                                                    StringRef Name,
                                                    ArrayRef<Type *> Types,
                                                    ArrayRef<unsigned> Ints) {
  const TargetExtTypeKeyInfo::KeyTy Key(Name, Types, Ints);
  TargetExtType *TT;
  // Since we only want to allocate a fresh target type in case none is found
  // and we don't want to perform two lookups (one for checking if existent and
  // one for inserting the newly allocated one), here we instead lookup based on
  // Key and update the reference to the target type in-place to a newly
  // allocated one if not found.
  auto [Iter, Inserted] = C.pImpl->TargetExtTypes.insert_as(nullptr, Key);
  if (Inserted) {
    // The target type was not found. Allocate one and update TargetExtTypes
    // in-place.
    TT = (TargetExtType *)C.pImpl->Alloc.Allocate(
        sizeof(TargetExtType) + sizeof(Type *) * Types.size() +
            sizeof(unsigned) * Ints.size(),
        alignof(TargetExtType));
    new (TT) TargetExtType(C, Name, Types, Ints);
    *Iter = TT;
    return checkParams(TT);
  }

  // The target type was found. Just return it.
  return *Iter;
}

Expected<TargetExtType *> TargetExtType::checkParams(TargetExtType *TTy) {
  // Opaque types in the AArch64 name space.
  if (TTy->Name == "aarch64.svcount" &&
      (TTy->getNumTypeParameters() != 0 || TTy->getNumIntParameters() != 0))
    return createStringError(
        "target extension type aarch64.svcount should have no parameters");

  // Opaque types in the RISC-V name space.
  if (TTy->Name == "riscv.vector.tuple" &&
      (TTy->getNumTypeParameters() != 1 || TTy->getNumIntParameters() != 1))
    return createStringError(
        "target extension type riscv.vector.tuple should have one "
        "type parameter and one integer parameter");

  // Opaque types in the AMDGPU name space.
  if (TTy->Name == "amdgcn.named.barrier" &&
      (TTy->getNumTypeParameters() != 0 || TTy->getNumIntParameters() != 1)) {
    return createStringError("target extension type amdgcn.named.barrier "
                             "should have no type parameters "
                             "and one integer parameter");
  }

  return TTy;
}

namespace {
struct TargetTypeInfo {
  Type *LayoutType;
  uint64_t Properties;

  template <typename... ArgTys>
  TargetTypeInfo(Type *LayoutType, ArgTys... Properties)
      : LayoutType(LayoutType), Properties((0 | ... | Properties)) {
    assert((!(this->Properties & TargetExtType::CanBeVectorElement) ||
            LayoutType->isSized()) &&
           "Vector element type must be sized");
  }
};
} // anonymous namespace

static TargetTypeInfo getTargetTypeInfo(const TargetExtType *Ty) {
  LLVMContext &C = Ty->getContext();
  StringRef Name = Ty->getName();
  if (Name == "spirv.Image" || Name == "spirv.SignedImage")
    return TargetTypeInfo(PointerType::get(C, 0), TargetExtType::CanBeGlobal,
                          TargetExtType::CanBeLocal);
  if (Name == "spirv.Type") {
    assert(Ty->getNumIntParameters() == 3 &&
           "Wrong number of parameters for spirv.Type");

    auto Size = Ty->getIntParameter(1);
    auto Alignment = Ty->getIntParameter(2);

    llvm::Type *LayoutType = nullptr;
    if (Size > 0 && Alignment > 0) {
      LayoutType =
          ArrayType::get(Type::getIntNTy(C, Alignment), Size * 8 / Alignment);
    } else {
      // LLVM expects variables that can be allocated to have an alignment and
      // size. Default to using a 32-bit int as the layout type if none are
      // present.
      LayoutType = Type::getInt32Ty(C);
    }

    return TargetTypeInfo(LayoutType, TargetExtType::CanBeGlobal,
                          TargetExtType::CanBeLocal);
  }
  if (Name == "spirv.IntegralConstant" || Name == "spirv.Literal")
    return TargetTypeInfo(Type::getVoidTy(C));
  if (Name.starts_with("spirv."))
    return TargetTypeInfo(PointerType::get(C, 0), TargetExtType::HasZeroInit,
                          TargetExtType::CanBeGlobal,
                          TargetExtType::CanBeLocal);

  // Opaque types in the AArch64 name space.
  if (Name == "aarch64.svcount")
    return TargetTypeInfo(ScalableVectorType::get(Type::getInt1Ty(C), 16),
                          TargetExtType::HasZeroInit,
                          TargetExtType::CanBeLocal);

  // RISC-V vector tuple type. The layout is represented as the type that needs
  // the same number of vector registers(VREGS) as this tuple type, represented
  // as <vscale x (RVVBitsPerBlock * VREGS / 8) x i8>.
  if (Name == "riscv.vector.tuple") {
    unsigned TotalNumElts =
        std::max(cast<ScalableVectorType>(Ty->getTypeParameter(0))
                     ->getMinNumElements(),
                 RISCV::RVVBytesPerBlock) *
        Ty->getIntParameter(0);
    return TargetTypeInfo(
        ScalableVectorType::get(Type::getInt8Ty(C), TotalNumElts),
        TargetExtType::CanBeLocal, TargetExtType::HasZeroInit);
  }

  // DirectX resources
  if (Name.starts_with("dx."))
    return TargetTypeInfo(PointerType::get(C, 0), TargetExtType::CanBeGlobal,
                          TargetExtType::CanBeLocal);

  // Opaque types in the AMDGPU name space.
  if (Name == "amdgcn.named.barrier") {
    return TargetTypeInfo(FixedVectorType::get(Type::getInt32Ty(C), 4),
                          TargetExtType::CanBeGlobal);
  }

  // Type used to test vector element target extension property.
  // Can be removed once a public target extension type uses CanBeVectorElement.
  if (Name == "llvm.test.vectorelement") {
    return TargetTypeInfo(Type::getInt32Ty(C), TargetExtType::CanBeLocal,
                          TargetExtType::CanBeVectorElement);
  }

  return TargetTypeInfo(Type::getVoidTy(C));
}

Type *TargetExtType::getLayoutType() const {
  return getTargetTypeInfo(this).LayoutType;
}

bool TargetExtType::hasProperty(Property Prop) const {
  uint64_t Properties = getTargetTypeInfo(this).Properties;
  return (Properties & Prop) == Prop;
}
