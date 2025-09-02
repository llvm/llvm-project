//===- Record.cpp - Record implementation ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement the tablegen record classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Record.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TGTimer.h"
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "tblgen-records"

//===----------------------------------------------------------------------===//
//    Context
//===----------------------------------------------------------------------===//

namespace llvm {
namespace detail {
/// This class represents the internal implementation of the RecordKeeper.
/// It contains all of the contextual static state of the Record classes. It is
/// kept out-of-line to simplify dependencies, and also make it easier for
/// internal classes to access the uniquer state of the keeper.
struct RecordKeeperImpl {
  RecordKeeperImpl(RecordKeeper &RK)
      : SharedBitRecTy(RK), SharedIntRecTy(RK), SharedStringRecTy(RK),
        SharedDagRecTy(RK), AnyRecord(RK, {}), TheUnsetInit(RK),
        TrueBitInit(true, &SharedBitRecTy),
        FalseBitInit(false, &SharedBitRecTy), StringInitStringPool(Allocator),
        StringInitCodePool(Allocator), AnonCounter(0), LastRecordID(0) {}

  BumpPtrAllocator Allocator;
  std::vector<BitsRecTy *> SharedBitsRecTys;
  BitRecTy SharedBitRecTy;
  IntRecTy SharedIntRecTy;
  StringRecTy SharedStringRecTy;
  DagRecTy SharedDagRecTy;

  RecordRecTy AnyRecord;
  UnsetInit TheUnsetInit;
  BitInit TrueBitInit;
  BitInit FalseBitInit;

  FoldingSet<ArgumentInit> TheArgumentInitPool;
  FoldingSet<BitsInit> TheBitsInitPool;
  std::map<int64_t, IntInit *> TheIntInitPool;
  StringMap<const StringInit *, BumpPtrAllocator &> StringInitStringPool;
  StringMap<const StringInit *, BumpPtrAllocator &> StringInitCodePool;
  FoldingSet<ListInit> TheListInitPool;
  FoldingSet<UnOpInit> TheUnOpInitPool;
  FoldingSet<BinOpInit> TheBinOpInitPool;
  FoldingSet<TernOpInit> TheTernOpInitPool;
  FoldingSet<FoldOpInit> TheFoldOpInitPool;
  FoldingSet<IsAOpInit> TheIsAOpInitPool;
  FoldingSet<ExistsOpInit> TheExistsOpInitPool;
  FoldingSet<InstancesOpInit> TheInstancesOpInitPool;
  DenseMap<std::pair<const RecTy *, const Init *>, VarInit *> TheVarInitPool;
  DenseMap<std::pair<const TypedInit *, unsigned>, VarBitInit *>
      TheVarBitInitPool;
  FoldingSet<VarDefInit> TheVarDefInitPool;
  DenseMap<std::pair<const Init *, const StringInit *>, FieldInit *>
      TheFieldInitPool;
  FoldingSet<CondOpInit> TheCondOpInitPool;
  FoldingSet<DagInit> TheDagInitPool;
  FoldingSet<RecordRecTy> RecordTypePool;

  unsigned AnonCounter;
  unsigned LastRecordID;

  void dumpAllocationStats(raw_ostream &OS) const;
};
} // namespace detail
} // namespace llvm

void detail::RecordKeeperImpl::dumpAllocationStats(raw_ostream &OS) const {
  // Dump memory allocation related stats.
  OS << "TheArgumentInitPool size = " << TheArgumentInitPool.size() << '\n';
  OS << "TheBitsInitPool size = " << TheBitsInitPool.size() << '\n';
  OS << "TheIntInitPool size = " << TheIntInitPool.size() << '\n';
  OS << "StringInitStringPool size = " << StringInitStringPool.size() << '\n';
  OS << "StringInitCodePool size = " << StringInitCodePool.size() << '\n';
  OS << "TheListInitPool size = " << TheListInitPool.size() << '\n';
  OS << "TheUnOpInitPool size = " << TheUnOpInitPool.size() << '\n';
  OS << "TheBinOpInitPool size = " << TheBinOpInitPool.size() << '\n';
  OS << "TheTernOpInitPool size = " << TheTernOpInitPool.size() << '\n';
  OS << "TheFoldOpInitPool size = " << TheFoldOpInitPool.size() << '\n';
  OS << "TheIsAOpInitPool size = " << TheIsAOpInitPool.size() << '\n';
  OS << "TheExistsOpInitPool size = " << TheExistsOpInitPool.size() << '\n';
  OS << "TheCondOpInitPool size = " << TheCondOpInitPool.size() << '\n';
  OS << "TheDagInitPool size = " << TheDagInitPool.size() << '\n';
  OS << "RecordTypePool size = " << RecordTypePool.size() << '\n';
  OS << "TheVarInitPool size = " << TheVarInitPool.size() << '\n';
  OS << "TheVarBitInitPool size = " << TheVarBitInitPool.size() << '\n';
  OS << "TheVarDefInitPool size = " << TheVarDefInitPool.size() << '\n';
  OS << "TheFieldInitPool size = " << TheFieldInitPool.size() << '\n';
  OS << "Bytes allocated = " << Allocator.getBytesAllocated() << '\n';
  OS << "Total allocator memory = " << Allocator.getTotalMemory() << "\n\n";

  OS << "Number of records instantiated = " << LastRecordID << '\n';
  OS << "Number of anonymous records = " << AnonCounter << '\n';
}

//===----------------------------------------------------------------------===//
//    Type implementations
//===----------------------------------------------------------------------===//

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void RecTy::dump() const { print(errs()); }
#endif

const ListRecTy *RecTy::getListTy() const {
  if (!ListTy)
    ListTy = new (RK.getImpl().Allocator) ListRecTy(this);
  return ListTy;
}

bool RecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  assert(RHS && "NULL pointer");
  return Kind == RHS->getRecTyKind();
}

bool RecTy::typeIsA(const RecTy *RHS) const { return this == RHS; }

const BitRecTy *BitRecTy::get(RecordKeeper &RK) {
  return &RK.getImpl().SharedBitRecTy;
}

bool BitRecTy::typeIsConvertibleTo(const RecTy *RHS) const{
  if (RecTy::typeIsConvertibleTo(RHS) || RHS->getRecTyKind() == IntRecTyKind)
    return true;
  if (const auto *BitsTy = dyn_cast<BitsRecTy>(RHS))
    return BitsTy->getNumBits() == 1;
  return false;
}

const BitsRecTy *BitsRecTy::get(RecordKeeper &RK, unsigned Sz) {
  detail::RecordKeeperImpl &RKImpl = RK.getImpl();
  if (Sz >= RKImpl.SharedBitsRecTys.size())
    RKImpl.SharedBitsRecTys.resize(Sz + 1);
  BitsRecTy *&Ty = RKImpl.SharedBitsRecTys[Sz];
  if (!Ty)
    Ty = new (RKImpl.Allocator) BitsRecTy(RK, Sz);
  return Ty;
}

std::string BitsRecTy::getAsString() const {
  return "bits<" + utostr(Size) + ">";
}

bool BitsRecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  if (RecTy::typeIsConvertibleTo(RHS)) //argument and the sender are same type
    return cast<BitsRecTy>(RHS)->Size == Size;
  RecTyKind kind = RHS->getRecTyKind();
  return (kind == BitRecTyKind && Size == 1) || (kind == IntRecTyKind);
}

const IntRecTy *IntRecTy::get(RecordKeeper &RK) {
  return &RK.getImpl().SharedIntRecTy;
}

bool IntRecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  RecTyKind kind = RHS->getRecTyKind();
  return kind==BitRecTyKind || kind==BitsRecTyKind || kind==IntRecTyKind;
}

const StringRecTy *StringRecTy::get(RecordKeeper &RK) {
  return &RK.getImpl().SharedStringRecTy;
}

std::string StringRecTy::getAsString() const {
  return "string";
}

bool StringRecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  RecTyKind Kind = RHS->getRecTyKind();
  return Kind == StringRecTyKind;
}

std::string ListRecTy::getAsString() const {
  return "list<" + ElementTy->getAsString() + ">";
}

bool ListRecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  if (const auto *ListTy = dyn_cast<ListRecTy>(RHS))
    return ElementTy->typeIsConvertibleTo(ListTy->getElementType());
  return false;
}

bool ListRecTy::typeIsA(const RecTy *RHS) const {
  if (const auto *RHSl = dyn_cast<ListRecTy>(RHS))
    return getElementType()->typeIsA(RHSl->getElementType());
  return false;
}

const DagRecTy *DagRecTy::get(RecordKeeper &RK) {
  return &RK.getImpl().SharedDagRecTy;
}

std::string DagRecTy::getAsString() const {
  return "dag";
}

static void ProfileRecordRecTy(FoldingSetNodeID &ID,
                               ArrayRef<const Record *> Classes) {
  ID.AddInteger(Classes.size());
  for (const Record *R : Classes)
    ID.AddPointer(R);
}

RecordRecTy::RecordRecTy(RecordKeeper &RK, ArrayRef<const Record *> Classes)
    : RecTy(RecordRecTyKind, RK), NumClasses(Classes.size()) {
  llvm::uninitialized_copy(Classes, getTrailingObjects());
}

const RecordRecTy *RecordRecTy::get(RecordKeeper &RK,
                                    ArrayRef<const Record *> UnsortedClasses) {
  detail::RecordKeeperImpl &RKImpl = RK.getImpl();
  if (UnsortedClasses.empty())
    return &RKImpl.AnyRecord;

  FoldingSet<RecordRecTy> &ThePool = RKImpl.RecordTypePool;

  SmallVector<const Record *, 4> Classes(UnsortedClasses);
  llvm::sort(Classes, [](const Record *LHS, const Record *RHS) {
    return LHS->getNameInitAsString() < RHS->getNameInitAsString();
  });

  FoldingSetNodeID ID;
  ProfileRecordRecTy(ID, Classes);

  void *IP = nullptr;
  if (RecordRecTy *Ty = ThePool.FindNodeOrInsertPos(ID, IP))
    return Ty;

#ifndef NDEBUG
  // Check for redundancy.
  for (unsigned i = 0; i < Classes.size(); ++i) {
    for (unsigned j = 0; j < Classes.size(); ++j) {
      assert(i == j || !Classes[i]->isSubClassOf(Classes[j]));
    }
    assert(&Classes[0]->getRecords() == &Classes[i]->getRecords());
  }
#endif

  void *Mem = RKImpl.Allocator.Allocate(
      totalSizeToAlloc<const Record *>(Classes.size()), alignof(RecordRecTy));
  RecordRecTy *Ty = new (Mem) RecordRecTy(RK, Classes);
  ThePool.InsertNode(Ty, IP);
  return Ty;
}

const RecordRecTy *RecordRecTy::get(const Record *Class) {
  assert(Class && "unexpected null class");
  return get(Class->getRecords(), {Class});
}

void RecordRecTy::Profile(FoldingSetNodeID &ID) const {
  ProfileRecordRecTy(ID, getClasses());
}

std::string RecordRecTy::getAsString() const {
  if (NumClasses == 1)
    return getClasses()[0]->getNameInitAsString();

  std::string Str = "{";
  ListSeparator LS;
  for (const Record *R : getClasses()) {
    Str += LS;
    Str += R->getNameInitAsString();
  }
  Str += "}";
  return Str;
}

bool RecordRecTy::isSubClassOf(const Record *Class) const {
  return llvm::any_of(getClasses(), [Class](const Record *MySuperClass) {
    return MySuperClass == Class || MySuperClass->isSubClassOf(Class);
  });
}

bool RecordRecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  if (this == RHS)
    return true;

  const auto *RTy = dyn_cast<RecordRecTy>(RHS);
  if (!RTy)
    return false;

  return llvm::all_of(RTy->getClasses(), [this](const Record *TargetClass) {
    return isSubClassOf(TargetClass);
  });
}

bool RecordRecTy::typeIsA(const RecTy *RHS) const {
  return typeIsConvertibleTo(RHS);
}

static const RecordRecTy *resolveRecordTypes(const RecordRecTy *T1,
                                             const RecordRecTy *T2) {
  SmallVector<const Record *, 4> CommonSuperClasses;
  SmallVector<const Record *, 4> Stack(T1->getClasses());

  while (!Stack.empty()) {
    const Record *R = Stack.pop_back_val();

    if (T2->isSubClassOf(R))
      CommonSuperClasses.push_back(R);
    else
      llvm::append_range(Stack, make_first_range(R->getDirectSuperClasses()));
  }

  return RecordRecTy::get(T1->getRecordKeeper(), CommonSuperClasses);
}

const RecTy *llvm::resolveTypes(const RecTy *T1, const RecTy *T2) {
  if (T1 == T2)
    return T1;

  if (const auto *RecTy1 = dyn_cast<RecordRecTy>(T1)) {
    if (const auto *RecTy2 = dyn_cast<RecordRecTy>(T2))
      return resolveRecordTypes(RecTy1, RecTy2);
  }

  assert(T1 != nullptr && "Invalid record type");
  if (T1->typeIsConvertibleTo(T2))
    return T2;

  assert(T2 != nullptr && "Invalid record type");
  if (T2->typeIsConvertibleTo(T1))
    return T1;

  if (const auto *ListTy1 = dyn_cast<ListRecTy>(T1)) {
    if (const auto *ListTy2 = dyn_cast<ListRecTy>(T2)) {
      const RecTy *NewType =
          resolveTypes(ListTy1->getElementType(), ListTy2->getElementType());
      if (NewType)
        return NewType->getListTy();
    }
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
//    Initializer implementations
//===----------------------------------------------------------------------===//

void Init::anchor() {}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void Init::dump() const { return print(errs()); }
#endif

RecordKeeper &Init::getRecordKeeper() const {
  if (auto *TyInit = dyn_cast<TypedInit>(this))
    return TyInit->getType()->getRecordKeeper();
  if (auto *ArgInit = dyn_cast<ArgumentInit>(this))
    return ArgInit->getRecordKeeper();
  return cast<UnsetInit>(this)->getRecordKeeper();
}

UnsetInit *UnsetInit::get(RecordKeeper &RK) {
  return &RK.getImpl().TheUnsetInit;
}

const Init *UnsetInit::getCastTo(const RecTy *Ty) const { return this; }

const Init *UnsetInit::convertInitializerTo(const RecTy *Ty) const {
  return this;
}

static void ProfileArgumentInit(FoldingSetNodeID &ID, const Init *Value,
                                ArgAuxType Aux) {
  auto I = Aux.index();
  ID.AddInteger(I);
  if (I == ArgumentInit::Positional)
    ID.AddInteger(std::get<ArgumentInit::Positional>(Aux));
  if (I == ArgumentInit::Named)
    ID.AddPointer(std::get<ArgumentInit::Named>(Aux));
  ID.AddPointer(Value);
}

void ArgumentInit::Profile(FoldingSetNodeID &ID) const {
  ProfileArgumentInit(ID, Value, Aux);
}

const ArgumentInit *ArgumentInit::get(const Init *Value, ArgAuxType Aux) {
  FoldingSetNodeID ID;
  ProfileArgumentInit(ID, Value, Aux);

  RecordKeeper &RK = Value->getRecordKeeper();
  detail::RecordKeeperImpl &RKImpl = RK.getImpl();
  void *IP = nullptr;
  if (const ArgumentInit *I =
          RKImpl.TheArgumentInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  ArgumentInit *I = new (RKImpl.Allocator) ArgumentInit(Value, Aux);
  RKImpl.TheArgumentInitPool.InsertNode(I, IP);
  return I;
}

const Init *ArgumentInit::resolveReferences(Resolver &R) const {
  const Init *NewValue = Value->resolveReferences(R);
  if (NewValue != Value)
    return cloneWithValue(NewValue);

  return this;
}

BitInit *BitInit::get(RecordKeeper &RK, bool V) {
  return V ? &RK.getImpl().TrueBitInit : &RK.getImpl().FalseBitInit;
}

const Init *BitInit::convertInitializerTo(const RecTy *Ty) const {
  if (isa<BitRecTy>(Ty))
    return this;

  if (isa<IntRecTy>(Ty))
    return IntInit::get(getRecordKeeper(), getValue());

  if (auto *BRT = dyn_cast<BitsRecTy>(Ty)) {
    // Can only convert single bit.
    if (BRT->getNumBits() == 1)
      return BitsInit::get(getRecordKeeper(), this);
  }

  return nullptr;
}

static void ProfileBitsInit(FoldingSetNodeID &ID,
                            ArrayRef<const Init *> Range) {
  ID.AddInteger(Range.size());

  for (const Init *I : Range)
    ID.AddPointer(I);
}

BitsInit::BitsInit(RecordKeeper &RK, ArrayRef<const Init *> Bits)
    : TypedInit(IK_BitsInit, BitsRecTy::get(RK, Bits.size())),
      NumBits(Bits.size()) {
  llvm::uninitialized_copy(Bits, getTrailingObjects());
}

BitsInit *BitsInit::get(RecordKeeper &RK, ArrayRef<const Init *> Bits) {
  FoldingSetNodeID ID;
  ProfileBitsInit(ID, Bits);

  detail::RecordKeeperImpl &RKImpl = RK.getImpl();
  void *IP = nullptr;
  if (BitsInit *I = RKImpl.TheBitsInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  void *Mem = RKImpl.Allocator.Allocate(
      totalSizeToAlloc<const Init *>(Bits.size()), alignof(BitsInit));
  BitsInit *I = new (Mem) BitsInit(RK, Bits);
  RKImpl.TheBitsInitPool.InsertNode(I, IP);
  return I;
}

void BitsInit::Profile(FoldingSetNodeID &ID) const {
  ProfileBitsInit(ID, getBits());
}

const Init *BitsInit::convertInitializerTo(const RecTy *Ty) const {
  if (isa<BitRecTy>(Ty)) {
    if (getNumBits() != 1) return nullptr; // Only accept if just one bit!
    return getBit(0);
  }

  if (auto *BRT = dyn_cast<BitsRecTy>(Ty)) {
    // If the number of bits is right, return it. Otherwise we need to expand
    // or truncate.
    if (getNumBits() != BRT->getNumBits()) return nullptr;
    return this;
  }

  if (isa<IntRecTy>(Ty)) {
    std::optional<int64_t> Result = convertInitializerToInt();
    if (Result)
      return IntInit::get(getRecordKeeper(), *Result);
  }

  return nullptr;
}

std::optional<int64_t> BitsInit::convertInitializerToInt() const {
  int64_t Result = 0;
  for (auto [Idx, InitV] : enumerate(getBits()))
    if (auto *Bit = dyn_cast<BitInit>(InitV))
      Result |= static_cast<int64_t>(Bit->getValue()) << Idx;
    else
      return std::nullopt;
  return Result;
}

const Init *
BitsInit::convertInitializerBitRange(ArrayRef<unsigned> Bits) const {
  SmallVector<const Init *, 16> NewBits(Bits.size());

  for (auto [Bit, NewBit] : zip_equal(Bits, NewBits)) {
    if (Bit >= getNumBits())
      return nullptr;
    NewBit = getBit(Bit);
  }
  return BitsInit::get(getRecordKeeper(), NewBits);
}

bool BitsInit::isComplete() const {
  return all_of(getBits(), [](const Init *Bit) { return Bit->isComplete(); });
}
bool BitsInit::allInComplete() const {
  return all_of(getBits(), [](const Init *Bit) { return !Bit->isComplete(); });
}
bool BitsInit::isConcrete() const {
  return all_of(getBits(), [](const Init *Bit) { return Bit->isConcrete(); });
}

std::string BitsInit::getAsString() const {
  std::string Result = "{ ";
  ListSeparator LS;
  for (const Init *Bit : reverse(getBits())) {
    Result += LS;
    if (Bit)
      Result += Bit->getAsString();
    else
      Result += "*";
  }
  return Result + " }";
}

// resolveReferences - If there are any field references that refer to fields
// that have been filled in, we can propagate the values now.
const Init *BitsInit::resolveReferences(Resolver &R) const {
  bool Changed = false;
  SmallVector<const Init *, 16> NewBits(getNumBits());

  const Init *CachedBitVarRef = nullptr;
  const Init *CachedBitVarResolved = nullptr;

  for (auto [CurBit, NewBit] : zip_equal(getBits(), NewBits)) {
    NewBit = CurBit;

    if (const auto *CurBitVar = dyn_cast<VarBitInit>(CurBit)) {
      if (CurBitVar->getBitVar() != CachedBitVarRef) {
        CachedBitVarRef = CurBitVar->getBitVar();
        CachedBitVarResolved = CachedBitVarRef->resolveReferences(R);
      }
      assert(CachedBitVarResolved && "Unresolved bitvar reference");
      NewBit = CachedBitVarResolved->getBit(CurBitVar->getBitNum());
    } else {
      // getBit(0) implicitly converts int and bits<1> values to bit.
      NewBit = CurBit->resolveReferences(R)->getBit(0);
    }

    if (isa<UnsetInit>(NewBit) && R.keepUnsetBits())
      NewBit = CurBit;
    Changed |= CurBit != NewBit;
  }

  if (Changed)
    return BitsInit::get(getRecordKeeper(), NewBits);

  return this;
}

IntInit *IntInit::get(RecordKeeper &RK, int64_t V) {
  IntInit *&I = RK.getImpl().TheIntInitPool[V];
  if (!I)
    I = new (RK.getImpl().Allocator) IntInit(RK, V);
  return I;
}

std::string IntInit::getAsString() const {
  return itostr(Value);
}

static bool canFitInBitfield(int64_t Value, unsigned NumBits) {
  // For example, with NumBits == 4, we permit Values from [-7 .. 15].
  return (NumBits >= sizeof(Value) * 8) ||
         (Value >> NumBits == 0) || (Value >> (NumBits-1) == -1);
}

const Init *IntInit::convertInitializerTo(const RecTy *Ty) const {
  if (isa<IntRecTy>(Ty))
    return this;

  if (isa<BitRecTy>(Ty)) {
    int64_t Val = getValue();
    if (Val != 0 && Val != 1) return nullptr;  // Only accept 0 or 1 for a bit!
    return BitInit::get(getRecordKeeper(), Val != 0);
  }

  if (const auto *BRT = dyn_cast<BitsRecTy>(Ty)) {
    int64_t Value = getValue();
    // Make sure this bitfield is large enough to hold the integer value.
    if (!canFitInBitfield(Value, BRT->getNumBits()))
      return nullptr;

    SmallVector<const Init *, 16> NewBits(BRT->getNumBits());
    for (unsigned i = 0; i != BRT->getNumBits(); ++i)
      NewBits[i] =
          BitInit::get(getRecordKeeper(), Value & ((i < 64) ? (1LL << i) : 0));

    return BitsInit::get(getRecordKeeper(), NewBits);
  }

  return nullptr;
}

const Init *IntInit::convertInitializerBitRange(ArrayRef<unsigned> Bits) const {
  SmallVector<const Init *, 16> NewBits(Bits.size());

  for (auto [Bit, NewBit] : zip_equal(Bits, NewBits)) {
    if (Bit >= 64)
      return nullptr;

    NewBit = BitInit::get(getRecordKeeper(), Value & (INT64_C(1) << Bit));
  }
  return BitsInit::get(getRecordKeeper(), NewBits);
}

AnonymousNameInit *AnonymousNameInit::get(RecordKeeper &RK, unsigned V) {
  return new (RK.getImpl().Allocator) AnonymousNameInit(RK, V);
}

const StringInit *AnonymousNameInit::getNameInit() const {
  return StringInit::get(getRecordKeeper(), getAsString());
}

std::string AnonymousNameInit::getAsString() const {
  return "anonymous_" + utostr(Value);
}

const Init *AnonymousNameInit::resolveReferences(Resolver &R) const {
  auto *Old = this;
  auto *New = R.resolve(Old);
  New = New ? New : Old;
  if (R.isFinal())
    if (const auto *Anonymous = dyn_cast<AnonymousNameInit>(New))
      return Anonymous->getNameInit();
  return New;
}

const StringInit *StringInit::get(RecordKeeper &RK, StringRef V,
                                  StringFormat Fmt) {
  detail::RecordKeeperImpl &RKImpl = RK.getImpl();
  auto &InitMap = Fmt == SF_String ? RKImpl.StringInitStringPool
                                   : RKImpl.StringInitCodePool;
  auto &Entry = *InitMap.try_emplace(V, nullptr).first;
  if (!Entry.second)
    Entry.second = new (RKImpl.Allocator) StringInit(RK, Entry.getKey(), Fmt);
  return Entry.second;
}

const Init *StringInit::convertInitializerTo(const RecTy *Ty) const {
  if (isa<StringRecTy>(Ty))
    return this;

  return nullptr;
}

static void ProfileListInit(FoldingSetNodeID &ID,
                            ArrayRef<const Init *> Elements,
                            const RecTy *EltTy) {
  ID.AddInteger(Elements.size());
  ID.AddPointer(EltTy);

  for (const Init *E : Elements)
    ID.AddPointer(E);
}

ListInit::ListInit(ArrayRef<const Init *> Elements, const RecTy *EltTy)
    : TypedInit(IK_ListInit, ListRecTy::get(EltTy)),
      NumElements(Elements.size()) {
  llvm::uninitialized_copy(Elements, getTrailingObjects());
}

const ListInit *ListInit::get(ArrayRef<const Init *> Elements,
                              const RecTy *EltTy) {
  FoldingSetNodeID ID;
  ProfileListInit(ID, Elements, EltTy);

  detail::RecordKeeperImpl &RK = EltTy->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (const ListInit *I = RK.TheListInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  assert(Elements.empty() || !isa<TypedInit>(Elements[0]) ||
         cast<TypedInit>(Elements[0])->getType()->typeIsConvertibleTo(EltTy));

  void *Mem = RK.Allocator.Allocate(
      totalSizeToAlloc<const Init *>(Elements.size()), alignof(ListInit));
  ListInit *I = new (Mem) ListInit(Elements, EltTy);
  RK.TheListInitPool.InsertNode(I, IP);
  return I;
}

void ListInit::Profile(FoldingSetNodeID &ID) const {
  const RecTy *EltTy = cast<ListRecTy>(getType())->getElementType();
  ProfileListInit(ID, getElements(), EltTy);
}

const Init *ListInit::convertInitializerTo(const RecTy *Ty) const {
  if (getType() == Ty)
    return this;

  if (const auto *LRT = dyn_cast<ListRecTy>(Ty)) {
    SmallVector<const Init *, 8> Elements;
    Elements.reserve(size());

    // Verify that all of the elements of the list are subclasses of the
    // appropriate class!
    bool Changed = false;
    const RecTy *ElementType = LRT->getElementType();
    for (const Init *I : getElements())
      if (const Init *CI = I->convertInitializerTo(ElementType)) {
        Elements.push_back(CI);
        if (CI != I)
          Changed = true;
      } else {
        return nullptr;
      }

    if (!Changed)
      return this;
    return ListInit::get(Elements, ElementType);
  }

  return nullptr;
}

const Record *ListInit::getElementAsRecord(unsigned Idx) const {
  const auto *DI = dyn_cast<DefInit>(getElement(Idx));
  if (!DI)
    PrintFatalError("Expected record in list!");
  return DI->getDef();
}

const Init *ListInit::resolveReferences(Resolver &R) const {
  SmallVector<const Init *, 8> Resolved;
  Resolved.reserve(size());
  bool Changed = false;

  for (const Init *CurElt : getElements()) {
    const Init *E = CurElt->resolveReferences(R);
    Changed |= E != CurElt;
    Resolved.push_back(E);
  }

  if (Changed)
    return ListInit::get(Resolved, getElementType());
  return this;
}

bool ListInit::isComplete() const {
  return all_of(*this,
                [](const Init *Element) { return Element->isComplete(); });
}

bool ListInit::isConcrete() const {
  return all_of(*this,
                [](const Init *Element) { return Element->isConcrete(); });
}

std::string ListInit::getAsString() const {
  std::string Result = "[";
  ListSeparator LS;
  for (const Init *Element : *this) {
    Result += LS;
    Result += Element->getAsString();
  }
  return Result + "]";
}

const Init *OpInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get(getRecordKeeper()))
    return this;
  return VarBitInit::get(this, Bit);
}

static void ProfileUnOpInit(FoldingSetNodeID &ID, unsigned Opcode,
                            const Init *Op, const RecTy *Type) {
  ID.AddInteger(Opcode);
  ID.AddPointer(Op);
  ID.AddPointer(Type);
}

const UnOpInit *UnOpInit::get(UnaryOp Opc, const Init *LHS, const RecTy *Type) {
  FoldingSetNodeID ID;
  ProfileUnOpInit(ID, Opc, LHS, Type);

  detail::RecordKeeperImpl &RK = Type->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (const UnOpInit *I = RK.TheUnOpInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  UnOpInit *I = new (RK.Allocator) UnOpInit(Opc, LHS, Type);
  RK.TheUnOpInitPool.InsertNode(I, IP);
  return I;
}

void UnOpInit::Profile(FoldingSetNodeID &ID) const {
  ProfileUnOpInit(ID, getOpcode(), getOperand(), getType());
}

const Init *UnOpInit::Fold(const Record *CurRec, bool IsFinal) const {
  RecordKeeper &RK = getRecordKeeper();
  switch (getOpcode()) {
  case REPR:
    if (LHS->isConcrete()) {
      // If it is a Record, print the full content.
      if (const auto *Def = dyn_cast<DefInit>(LHS)) {
        std::string S;
        raw_string_ostream OS(S);
        OS << *Def->getDef();
        return StringInit::get(RK, S);
      } else {
        // Otherwise, print the value of the variable.
        //
        // NOTE: we could recursively !repr the elements of a list,
        // but that could produce a lot of output when printing a
        // defset.
        return StringInit::get(RK, LHS->getAsString());
      }
    }
    break;
  case TOLOWER:
    if (const auto *LHSs = dyn_cast<StringInit>(LHS))
      return StringInit::get(RK, LHSs->getValue().lower());
    break;
  case TOUPPER:
    if (const auto *LHSs = dyn_cast<StringInit>(LHS))
      return StringInit::get(RK, LHSs->getValue().upper());
    break;
  case CAST:
    if (isa<StringRecTy>(getType())) {
      if (const auto *LHSs = dyn_cast<StringInit>(LHS))
        return LHSs;

      if (const auto *LHSd = dyn_cast<DefInit>(LHS))
        return StringInit::get(RK, LHSd->getAsString());

      if (const auto *LHSi = dyn_cast_or_null<IntInit>(
              LHS->convertInitializerTo(IntRecTy::get(RK))))
        return StringInit::get(RK, LHSi->getAsString());

    } else if (isa<RecordRecTy>(getType())) {
      if (const auto *Name = dyn_cast<StringInit>(LHS)) {
        const Record *D = RK.getDef(Name->getValue());
        if (!D && CurRec) {
          // Self-references are allowed, but their resolution is delayed until
          // the final resolve to ensure that we get the correct type for them.
          auto *Anonymous = dyn_cast<AnonymousNameInit>(CurRec->getNameInit());
          if (Name == CurRec->getNameInit() ||
              (Anonymous && Name == Anonymous->getNameInit())) {
            if (!IsFinal)
              break;
            D = CurRec;
          }
        }

        auto PrintFatalErrorHelper = [CurRec](const Twine &T) {
          if (CurRec)
            PrintFatalError(CurRec->getLoc(), T);
          else
            PrintFatalError(T);
        };

        if (!D) {
          if (IsFinal) {
            PrintFatalErrorHelper(Twine("Undefined reference to record: '") +
                                  Name->getValue() + "'\n");
          }
          break;
        }

        DefInit *DI = D->getDefInit();
        if (!DI->getType()->typeIsA(getType())) {
          PrintFatalErrorHelper(Twine("Expected type '") +
                                getType()->getAsString() + "', got '" +
                                DI->getType()->getAsString() + "' in: " +
                                getAsString() + "\n");
        }
        return DI;
      }
    }

    if (const Init *NewInit = LHS->convertInitializerTo(getType()))
      return NewInit;
    break;

  case INITIALIZED:
    if (isa<UnsetInit>(LHS))
      return IntInit::get(RK, 0);
    if (LHS->isConcrete())
      return IntInit::get(RK, 1);
    break;

  case NOT:
    if (const auto *LHSi = dyn_cast_or_null<IntInit>(
            LHS->convertInitializerTo(IntRecTy::get(RK))))
      return IntInit::get(RK, LHSi->getValue() ? 0 : 1);
    break;

  case HEAD:
    if (const auto *LHSl = dyn_cast<ListInit>(LHS)) {
      assert(!LHSl->empty() && "Empty list in head");
      return LHSl->getElement(0);
    }
    break;

  case TAIL:
    if (const auto *LHSl = dyn_cast<ListInit>(LHS)) {
      assert(!LHSl->empty() && "Empty list in tail");
      // Note the slice(1). We can't just pass the result of getElements()
      // directly.
      return ListInit::get(LHSl->getElements().slice(1),
                           LHSl->getElementType());
    }
    break;

  case SIZE:
    if (const auto *LHSl = dyn_cast<ListInit>(LHS))
      return IntInit::get(RK, LHSl->size());
    if (const auto *LHSd = dyn_cast<DagInit>(LHS))
      return IntInit::get(RK, LHSd->arg_size());
    if (const auto *LHSs = dyn_cast<StringInit>(LHS))
      return IntInit::get(RK, LHSs->getValue().size());
    break;

  case EMPTY:
    if (const auto *LHSl = dyn_cast<ListInit>(LHS))
      return IntInit::get(RK, LHSl->empty());
    if (const auto *LHSd = dyn_cast<DagInit>(LHS))
      return IntInit::get(RK, LHSd->arg_empty());
    if (const auto *LHSs = dyn_cast<StringInit>(LHS))
      return IntInit::get(RK, LHSs->getValue().empty());
    break;

  case GETDAGOP:
    if (const auto *Dag = dyn_cast<DagInit>(LHS)) {
      // TI is not necessarily a def due to the late resolution in multiclasses,
      // but has to be a TypedInit.
      auto *TI = cast<TypedInit>(Dag->getOperator());
      if (!TI->getType()->typeIsA(getType())) {
        PrintFatalError(CurRec->getLoc(),
                        Twine("Expected type '") + getType()->getAsString() +
                            "', got '" + TI->getType()->getAsString() +
                            "' in: " + getAsString() + "\n");
      } else {
        return Dag->getOperator();
      }
    }
    break;

  case GETDAGOPNAME:
    if (const auto *Dag = dyn_cast<DagInit>(LHS)) {
      return Dag->getName();
    }
    break;

  case LOG2:
    if (const auto *LHSi = dyn_cast_or_null<IntInit>(
            LHS->convertInitializerTo(IntRecTy::get(RK)))) {
      int64_t LHSv = LHSi->getValue();
      if (LHSv <= 0) {
        PrintFatalError(CurRec->getLoc(),
                        "Illegal operation: logtwo is undefined "
                        "on arguments less than or equal to 0");
      } else {
        uint64_t Log = Log2_64(LHSv);
        assert(Log <= INT64_MAX &&
               "Log of an int64_t must be smaller than INT64_MAX");
        return IntInit::get(RK, static_cast<int64_t>(Log));
      }
    }
    break;

  case LISTFLATTEN:
    if (const auto *LHSList = dyn_cast<ListInit>(LHS)) {
      const auto *InnerListTy = dyn_cast<ListRecTy>(LHSList->getElementType());
      // list of non-lists, !listflatten() is a NOP.
      if (!InnerListTy)
        return LHS;

      auto Flatten =
          [](const ListInit *List) -> std::optional<std::vector<const Init *>> {
        std::vector<const Init *> Flattened;
        // Concatenate elements of all the inner lists.
        for (const Init *InnerInit : List->getElements()) {
          const auto *InnerList = dyn_cast<ListInit>(InnerInit);
          if (!InnerList)
            return std::nullopt;
          llvm::append_range(Flattened, InnerList->getElements());
        };
        return Flattened;
      };

      auto Flattened = Flatten(LHSList);
      if (Flattened)
        return ListInit::get(*Flattened, InnerListTy->getElementType());
    }
    break;
  }
  return this;
}

const Init *UnOpInit::resolveReferences(Resolver &R) const {
  const Init *lhs = LHS->resolveReferences(R);

  if (LHS != lhs || (R.isFinal() && getOpcode() == CAST))
    return (UnOpInit::get(getOpcode(), lhs, getType()))
        ->Fold(R.getCurrentRecord(), R.isFinal());
  return this;
}

std::string UnOpInit::getAsString() const {
  std::string Result;
  switch (getOpcode()) {
  case CAST: Result = "!cast<" + getType()->getAsString() + ">"; break;
  case NOT: Result = "!not"; break;
  case HEAD: Result = "!head"; break;
  case TAIL: Result = "!tail"; break;
  case SIZE: Result = "!size"; break;
  case EMPTY: Result = "!empty"; break;
  case GETDAGOP: Result = "!getdagop"; break;
  case GETDAGOPNAME:
    Result = "!getdagopname";
    break;
  case LOG2 : Result = "!logtwo"; break;
  case LISTFLATTEN:
    Result = "!listflatten";
    break;
  case REPR:
    Result = "!repr";
    break;
  case TOLOWER:
    Result = "!tolower";
    break;
  case TOUPPER:
    Result = "!toupper";
    break;
  case INITIALIZED:
    Result = "!initialized";
    break;
  }
  return Result + "(" + LHS->getAsString() + ")";
}

static void ProfileBinOpInit(FoldingSetNodeID &ID, unsigned Opcode,
                             const Init *LHS, const Init *RHS,
                             const RecTy *Type) {
  ID.AddInteger(Opcode);
  ID.AddPointer(LHS);
  ID.AddPointer(RHS);
  ID.AddPointer(Type);
}

const BinOpInit *BinOpInit::get(BinaryOp Opc, const Init *LHS, const Init *RHS,
                                const RecTy *Type) {
  FoldingSetNodeID ID;
  ProfileBinOpInit(ID, Opc, LHS, RHS, Type);

  detail::RecordKeeperImpl &RK = LHS->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (const BinOpInit *I = RK.TheBinOpInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  BinOpInit *I = new (RK.Allocator) BinOpInit(Opc, LHS, RHS, Type);
  RK.TheBinOpInitPool.InsertNode(I, IP);
  return I;
}

void BinOpInit::Profile(FoldingSetNodeID &ID) const {
  ProfileBinOpInit(ID, getOpcode(), getLHS(), getRHS(), getType());
}

static const StringInit *ConcatStringInits(const StringInit *I0,
                                           const StringInit *I1) {
  SmallString<80> Concat(I0->getValue());
  Concat.append(I1->getValue());
  return StringInit::get(
      I0->getRecordKeeper(), Concat,
      StringInit::determineFormat(I0->getFormat(), I1->getFormat()));
}

static const StringInit *interleaveStringList(const ListInit *List,
                                              const StringInit *Delim) {
  if (List->size() == 0)
    return StringInit::get(List->getRecordKeeper(), "");
  const auto *Element = dyn_cast<StringInit>(List->getElement(0));
  if (!Element)
    return nullptr;
  SmallString<80> Result(Element->getValue());
  StringInit::StringFormat Fmt = StringInit::SF_String;

  for (const Init *Elem : List->getElements().drop_front()) {
    Result.append(Delim->getValue());
    const auto *Element = dyn_cast<StringInit>(Elem);
    if (!Element)
      return nullptr;
    Result.append(Element->getValue());
    Fmt = StringInit::determineFormat(Fmt, Element->getFormat());
  }
  return StringInit::get(List->getRecordKeeper(), Result, Fmt);
}

static const StringInit *interleaveIntList(const ListInit *List,
                                           const StringInit *Delim) {
  RecordKeeper &RK = List->getRecordKeeper();
  if (List->size() == 0)
    return StringInit::get(RK, "");
  const auto *Element = dyn_cast_or_null<IntInit>(
      List->getElement(0)->convertInitializerTo(IntRecTy::get(RK)));
  if (!Element)
    return nullptr;
  SmallString<80> Result(Element->getAsString());

  for (const Init *Elem : List->getElements().drop_front()) {
    Result.append(Delim->getValue());
    const auto *Element = dyn_cast_or_null<IntInit>(
        Elem->convertInitializerTo(IntRecTy::get(RK)));
    if (!Element)
      return nullptr;
    Result.append(Element->getAsString());
  }
  return StringInit::get(RK, Result);
}

const Init *BinOpInit::getStrConcat(const Init *I0, const Init *I1) {
  // Shortcut for the common case of concatenating two strings.
  if (const auto *I0s = dyn_cast<StringInit>(I0))
    if (const auto *I1s = dyn_cast<StringInit>(I1))
      return ConcatStringInits(I0s, I1s);
  return BinOpInit::get(BinOpInit::STRCONCAT, I0, I1,
                        StringRecTy::get(I0->getRecordKeeper()));
}

static const ListInit *ConcatListInits(const ListInit *LHS,
                                       const ListInit *RHS) {
  SmallVector<const Init *, 8> Args;
  llvm::append_range(Args, *LHS);
  llvm::append_range(Args, *RHS);
  return ListInit::get(Args, LHS->getElementType());
}

const Init *BinOpInit::getListConcat(const TypedInit *LHS, const Init *RHS) {
  assert(isa<ListRecTy>(LHS->getType()) && "First arg must be a list");

  // Shortcut for the common case of concatenating two lists.
  if (const auto *LHSList = dyn_cast<ListInit>(LHS))
    if (const auto *RHSList = dyn_cast<ListInit>(RHS))
      return ConcatListInits(LHSList, RHSList);
  return BinOpInit::get(BinOpInit::LISTCONCAT, LHS, RHS, LHS->getType());
}

std::optional<bool> BinOpInit::CompareInit(unsigned Opc, const Init *LHS,
                                           const Init *RHS) const {
  // First see if we have two bit, bits, or int.
  const auto *LHSi = dyn_cast_or_null<IntInit>(
      LHS->convertInitializerTo(IntRecTy::get(getRecordKeeper())));
  const auto *RHSi = dyn_cast_or_null<IntInit>(
      RHS->convertInitializerTo(IntRecTy::get(getRecordKeeper())));

  if (LHSi && RHSi) {
    bool Result;
    switch (Opc) {
    case EQ:
      Result = LHSi->getValue() == RHSi->getValue();
      break;
    case NE:
      Result = LHSi->getValue() != RHSi->getValue();
      break;
    case LE:
      Result = LHSi->getValue() <= RHSi->getValue();
      break;
    case LT:
      Result = LHSi->getValue() < RHSi->getValue();
      break;
    case GE:
      Result = LHSi->getValue() >= RHSi->getValue();
      break;
    case GT:
      Result = LHSi->getValue() > RHSi->getValue();
      break;
    default:
      llvm_unreachable("unhandled comparison");
    }
    return Result;
  }

  // Next try strings.
  const auto *LHSs = dyn_cast<StringInit>(LHS);
  const auto *RHSs = dyn_cast<StringInit>(RHS);

  if (LHSs && RHSs) {
    bool Result;
    switch (Opc) {
    case EQ:
      Result = LHSs->getValue() == RHSs->getValue();
      break;
    case NE:
      Result = LHSs->getValue() != RHSs->getValue();
      break;
    case LE:
      Result = LHSs->getValue() <= RHSs->getValue();
      break;
    case LT:
      Result = LHSs->getValue() < RHSs->getValue();
      break;
    case GE:
      Result = LHSs->getValue() >= RHSs->getValue();
      break;
    case GT:
      Result = LHSs->getValue() > RHSs->getValue();
      break;
    default:
      llvm_unreachable("unhandled comparison");
    }
    return Result;
  }

  // Finally, !eq and !ne can be used with records.
  if (Opc == EQ || Opc == NE) {
    const auto *LHSd = dyn_cast<DefInit>(LHS);
    const auto *RHSd = dyn_cast<DefInit>(RHS);
    if (LHSd && RHSd)
      return (Opc == EQ) ? LHSd == RHSd : LHSd != RHSd;
  }

  return std::nullopt;
}

static std::optional<unsigned>
getDagArgNoByKey(const DagInit *Dag, const Init *Key, std::string &Error) {
  // Accessor by index
  if (const auto *Idx = dyn_cast<IntInit>(Key)) {
    int64_t Pos = Idx->getValue();
    if (Pos < 0) {
      // The index is negative.
      Error =
          (Twine("index ") + std::to_string(Pos) + Twine(" is negative")).str();
      return std::nullopt;
    }
    if (Pos >= Dag->getNumArgs()) {
      // The index is out-of-range.
      Error = (Twine("index ") + std::to_string(Pos) +
               " is out of range (dag has " +
               std::to_string(Dag->getNumArgs()) + " arguments)")
                  .str();
      return std::nullopt;
    }
    return Pos;
  }
  assert(isa<StringInit>(Key));
  // Accessor by name
  const auto *Name = dyn_cast<StringInit>(Key);
  auto ArgNo = Dag->getArgNo(Name->getValue());
  if (!ArgNo) {
    // The key is not found.
    Error = (Twine("key '") + Name->getValue() + Twine("' is not found")).str();
    return std::nullopt;
  }
  return *ArgNo;
}

const Init *BinOpInit::Fold(const Record *CurRec) const {
  switch (getOpcode()) {
  case CONCAT: {
    const auto *LHSs = dyn_cast<DagInit>(LHS);
    const auto *RHSs = dyn_cast<DagInit>(RHS);
    if (LHSs && RHSs) {
      const auto *LOp = dyn_cast<DefInit>(LHSs->getOperator());
      const auto *ROp = dyn_cast<DefInit>(RHSs->getOperator());
      if ((!LOp && !isa<UnsetInit>(LHSs->getOperator())) ||
          (!ROp && !isa<UnsetInit>(RHSs->getOperator())))
        break;
      if (LOp && ROp && LOp->getDef() != ROp->getDef()) {
        PrintFatalError(Twine("Concatenated Dag operators do not match: '") +
                        LHSs->getAsString() + "' vs. '" + RHSs->getAsString() +
                        "'");
      }
      const Init *Op = LOp ? LOp : ROp;
      if (!Op)
        Op = UnsetInit::get(getRecordKeeper());

      SmallVector<std::pair<const Init *, const StringInit *>, 8> Args;
      llvm::append_range(Args, LHSs->getArgAndNames());
      llvm::append_range(Args, RHSs->getArgAndNames());
      // Use the name of the LHS DAG if it's set, otherwise the name of the RHS.
      const auto *NameInit = LHSs->getName();
      if (!NameInit)
        NameInit = RHSs->getName();
      return DagInit::get(Op, NameInit, Args);
    }
    break;
  }
  case MATCH: {
    const auto *StrInit = dyn_cast<StringInit>(LHS);
    if (!StrInit)
      return this;

    const auto *RegexInit = dyn_cast<StringInit>(RHS);
    if (!RegexInit)
      return this;

    StringRef RegexStr = RegexInit->getValue();
    llvm::Regex Matcher(RegexStr);
    if (!Matcher.isValid())
      PrintFatalError(Twine("invalid regex '") + RegexStr + Twine("'"));

    return BitInit::get(LHS->getRecordKeeper(),
                        Matcher.match(StrInit->getValue()));
  }
  case LISTCONCAT: {
    const auto *LHSs = dyn_cast<ListInit>(LHS);
    const auto *RHSs = dyn_cast<ListInit>(RHS);
    if (LHSs && RHSs) {
      SmallVector<const Init *, 8> Args;
      llvm::append_range(Args, *LHSs);
      llvm::append_range(Args, *RHSs);
      return ListInit::get(Args, LHSs->getElementType());
    }
    break;
  }
  case LISTSPLAT: {
    const auto *Value = dyn_cast<TypedInit>(LHS);
    const auto *Size = dyn_cast<IntInit>(RHS);
    if (Value && Size) {
      SmallVector<const Init *, 8> Args(Size->getValue(), Value);
      return ListInit::get(Args, Value->getType());
    }
    break;
  }
  case LISTREMOVE: {
    const auto *LHSs = dyn_cast<ListInit>(LHS);
    const auto *RHSs = dyn_cast<ListInit>(RHS);
    if (LHSs && RHSs) {
      SmallVector<const Init *, 8> Args;
      for (const Init *EltLHS : *LHSs) {
        bool Found = false;
        for (const Init *EltRHS : *RHSs) {
          if (std::optional<bool> Result = CompareInit(EQ, EltLHS, EltRHS)) {
            if (*Result) {
              Found = true;
              break;
            }
          }
        }
        if (!Found)
          Args.push_back(EltLHS);
      }
      return ListInit::get(Args, LHSs->getElementType());
    }
    break;
  }
  case LISTELEM: {
    const auto *TheList = dyn_cast<ListInit>(LHS);
    const auto *Idx = dyn_cast<IntInit>(RHS);
    if (!TheList || !Idx)
      break;
    auto i = Idx->getValue();
    if (i < 0 || i >= (ssize_t)TheList->size())
      break;
    return TheList->getElement(i);
  }
  case LISTSLICE: {
    const auto *TheList = dyn_cast<ListInit>(LHS);
    const auto *SliceIdxs = dyn_cast<ListInit>(RHS);
    if (!TheList || !SliceIdxs)
      break;
    SmallVector<const Init *, 8> Args;
    Args.reserve(SliceIdxs->size());
    for (auto *I : *SliceIdxs) {
      auto *II = dyn_cast<IntInit>(I);
      if (!II)
        goto unresolved;
      auto i = II->getValue();
      if (i < 0 || i >= (ssize_t)TheList->size())
        goto unresolved;
      Args.push_back(TheList->getElement(i));
    }
    return ListInit::get(Args, TheList->getElementType());
  }
  case RANGEC: {
    const auto *LHSi = dyn_cast<IntInit>(LHS);
    const auto *RHSi = dyn_cast<IntInit>(RHS);
    if (!LHSi || !RHSi)
      break;

    int64_t Start = LHSi->getValue();
    int64_t End = RHSi->getValue();
    SmallVector<const Init *, 8> Args;
    if (getOpcode() == RANGEC) {
      // Closed interval
      if (Start <= End) {
        // Ascending order
        Args.reserve(End - Start + 1);
        for (auto i = Start; i <= End; ++i)
          Args.push_back(IntInit::get(getRecordKeeper(), i));
      } else {
        // Descending order
        Args.reserve(Start - End + 1);
        for (auto i = Start; i >= End; --i)
          Args.push_back(IntInit::get(getRecordKeeper(), i));
      }
    } else if (Start < End) {
      // Half-open interval (excludes `End`)
      Args.reserve(End - Start);
      for (auto i = Start; i < End; ++i)
        Args.push_back(IntInit::get(getRecordKeeper(), i));
    } else {
      // Empty set
    }
    return ListInit::get(Args, LHSi->getType());
  }
  case STRCONCAT: {
    const auto *LHSs = dyn_cast<StringInit>(LHS);
    const auto *RHSs = dyn_cast<StringInit>(RHS);
    if (LHSs && RHSs)
      return ConcatStringInits(LHSs, RHSs);
    break;
  }
  case INTERLEAVE: {
    const auto *List = dyn_cast<ListInit>(LHS);
    const auto *Delim = dyn_cast<StringInit>(RHS);
    if (List && Delim) {
      const StringInit *Result;
      if (isa<StringRecTy>(List->getElementType()))
        Result = interleaveStringList(List, Delim);
      else
        Result = interleaveIntList(List, Delim);
      if (Result)
        return Result;
    }
    break;
  }
  case EQ:
  case NE:
  case LE:
  case LT:
  case GE:
  case GT: {
    if (std::optional<bool> Result = CompareInit(getOpcode(), LHS, RHS))
      return BitInit::get(getRecordKeeper(), *Result);
    break;
  }
  case GETDAGARG: {
    const auto *Dag = dyn_cast<DagInit>(LHS);
    if (Dag && isa<IntInit, StringInit>(RHS)) {
      std::string Error;
      auto ArgNo = getDagArgNoByKey(Dag, RHS, Error);
      if (!ArgNo)
        PrintFatalError(CurRec->getLoc(), "!getdagarg " + Error);

      assert(*ArgNo < Dag->getNumArgs());

      const Init *Arg = Dag->getArg(*ArgNo);
      if (const auto *TI = dyn_cast<TypedInit>(Arg))
        if (!TI->getType()->typeIsConvertibleTo(getType()))
          return UnsetInit::get(Dag->getRecordKeeper());
      return Arg;
    }
    break;
  }
  case GETDAGNAME: {
    const auto *Dag = dyn_cast<DagInit>(LHS);
    const auto *Idx = dyn_cast<IntInit>(RHS);
    if (Dag && Idx) {
      int64_t Pos = Idx->getValue();
      if (Pos < 0 || Pos >= Dag->getNumArgs()) {
        // The index is out-of-range.
        PrintError(CurRec->getLoc(),
                   Twine("!getdagname index is out of range 0...") +
                       std::to_string(Dag->getNumArgs() - 1) + ": " +
                       std::to_string(Pos));
      }
      const Init *ArgName = Dag->getArgName(Pos);
      if (!ArgName)
        return UnsetInit::get(getRecordKeeper());
      return ArgName;
    }
    break;
  }
  case SETDAGOP: {
    const auto *Dag = dyn_cast<DagInit>(LHS);
    const auto *Op = dyn_cast<DefInit>(RHS);
    if (Dag && Op)
      return DagInit::get(Op, Dag->getArgs(), Dag->getArgNames());
    break;
  }
  case SETDAGOPNAME: {
    const auto *Dag = dyn_cast<DagInit>(LHS);
    const auto *Op = dyn_cast<StringInit>(RHS);
    if (Dag && Op)
      return DagInit::get(Dag->getOperator(), Op, Dag->getArgs(),
                          Dag->getArgNames());
    break;
  }
  case ADD:
  case SUB:
  case MUL:
  case DIV:
  case AND:
  case OR:
  case XOR:
  case SHL:
  case SRA:
  case SRL: {
    const auto *LHSi = dyn_cast_or_null<IntInit>(
        LHS->convertInitializerTo(IntRecTy::get(getRecordKeeper())));
    const auto *RHSi = dyn_cast_or_null<IntInit>(
        RHS->convertInitializerTo(IntRecTy::get(getRecordKeeper())));
    if (LHSi && RHSi) {
      int64_t LHSv = LHSi->getValue(), RHSv = RHSi->getValue();
      int64_t Result;
      switch (getOpcode()) {
      default: llvm_unreachable("Bad opcode!");
      case ADD: Result = LHSv + RHSv; break;
      case SUB: Result = LHSv - RHSv; break;
      case MUL: Result = LHSv * RHSv; break;
      case DIV:
        if (RHSv == 0)
          PrintFatalError(CurRec->getLoc(),
                          "Illegal operation: division by zero");
        else if (LHSv == INT64_MIN && RHSv == -1)
          PrintFatalError(CurRec->getLoc(),
                          "Illegal operation: INT64_MIN / -1");
        else
          Result = LHSv / RHSv;
        break;
      case AND: Result = LHSv & RHSv; break;
      case OR:  Result = LHSv | RHSv; break;
      case XOR: Result = LHSv ^ RHSv; break;
      case SHL:
        if (RHSv < 0 || RHSv >= 64)
          PrintFatalError(CurRec->getLoc(),
                          "Illegal operation: out of bounds shift");
        Result = (uint64_t)LHSv << (uint64_t)RHSv;
        break;
      case SRA:
        if (RHSv < 0 || RHSv >= 64)
          PrintFatalError(CurRec->getLoc(),
                          "Illegal operation: out of bounds shift");
        Result = LHSv >> (uint64_t)RHSv;
        break;
      case SRL:
        if (RHSv < 0 || RHSv >= 64)
          PrintFatalError(CurRec->getLoc(),
                          "Illegal operation: out of bounds shift");
        Result = (uint64_t)LHSv >> (uint64_t)RHSv;
        break;
      }
      return IntInit::get(getRecordKeeper(), Result);
    }
    break;
  }
  }
unresolved:
  return this;
}

const Init *BinOpInit::resolveReferences(Resolver &R) const {
  const Init *NewLHS = LHS->resolveReferences(R);

  unsigned Opc = getOpcode();
  if (Opc == AND || Opc == OR) {
    // Short-circuit. Regardless whether this is a logical or bitwise
    // AND/OR.
    // Ideally we could also short-circuit `!or(true, ...)`, but it's
    // difficult to do it right without knowing if rest of the operands
    // are all `bit` or not. Therefore, we're only implementing a relatively
    // limited version of short-circuit against all ones (`true` is casted
    // to 1 rather than all ones before we evaluate `!or`).
    if (const auto *LHSi = dyn_cast_or_null<IntInit>(
            NewLHS->convertInitializerTo(IntRecTy::get(getRecordKeeper())))) {
      if ((Opc == AND && !LHSi->getValue()) ||
          (Opc == OR && LHSi->getValue() == -1))
        return LHSi;
    }
  }

  const Init *NewRHS = RHS->resolveReferences(R);

  if (LHS != NewLHS || RHS != NewRHS)
    return (BinOpInit::get(getOpcode(), NewLHS, NewRHS, getType()))
        ->Fold(R.getCurrentRecord());
  return this;
}

std::string BinOpInit::getAsString() const {
  std::string Result;
  switch (getOpcode()) {
  case LISTELEM:
  case LISTSLICE:
    return LHS->getAsString() + "[" + RHS->getAsString() + "]";
  case RANGEC:
    return LHS->getAsString() + "..." + RHS->getAsString();
  case CONCAT: Result = "!con"; break;
  case MATCH:
    Result = "!match";
    break;
  case ADD: Result = "!add"; break;
  case SUB: Result = "!sub"; break;
  case MUL: Result = "!mul"; break;
  case DIV: Result = "!div"; break;
  case AND: Result = "!and"; break;
  case OR: Result = "!or"; break;
  case XOR: Result = "!xor"; break;
  case SHL: Result = "!shl"; break;
  case SRA: Result = "!sra"; break;
  case SRL: Result = "!srl"; break;
  case EQ: Result = "!eq"; break;
  case NE: Result = "!ne"; break;
  case LE: Result = "!le"; break;
  case LT: Result = "!lt"; break;
  case GE: Result = "!ge"; break;
  case GT: Result = "!gt"; break;
  case LISTCONCAT: Result = "!listconcat"; break;
  case LISTSPLAT: Result = "!listsplat"; break;
  case LISTREMOVE:
    Result = "!listremove";
    break;
  case STRCONCAT: Result = "!strconcat"; break;
  case INTERLEAVE: Result = "!interleave"; break;
  case SETDAGOP: Result = "!setdagop"; break;
  case SETDAGOPNAME:
    Result = "!setdagopname";
    break;
  case GETDAGARG:
    Result = "!getdagarg<" + getType()->getAsString() + ">";
    break;
  case GETDAGNAME:
    Result = "!getdagname";
    break;
  }
  return Result + "(" + LHS->getAsString() + ", " + RHS->getAsString() + ")";
}

static void ProfileTernOpInit(FoldingSetNodeID &ID, unsigned Opcode,
                              const Init *LHS, const Init *MHS, const Init *RHS,
                              const RecTy *Type) {
  ID.AddInteger(Opcode);
  ID.AddPointer(LHS);
  ID.AddPointer(MHS);
  ID.AddPointer(RHS);
  ID.AddPointer(Type);
}

const TernOpInit *TernOpInit::get(TernaryOp Opc, const Init *LHS,
                                  const Init *MHS, const Init *RHS,
                                  const RecTy *Type) {
  FoldingSetNodeID ID;
  ProfileTernOpInit(ID, Opc, LHS, MHS, RHS, Type);

  detail::RecordKeeperImpl &RK = LHS->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (TernOpInit *I = RK.TheTernOpInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  TernOpInit *I = new (RK.Allocator) TernOpInit(Opc, LHS, MHS, RHS, Type);
  RK.TheTernOpInitPool.InsertNode(I, IP);
  return I;
}

void TernOpInit::Profile(FoldingSetNodeID &ID) const {
  ProfileTernOpInit(ID, getOpcode(), getLHS(), getMHS(), getRHS(), getType());
}

static const Init *ItemApply(const Init *LHS, const Init *MHSe, const Init *RHS,
                             const Record *CurRec) {
  MapResolver R(CurRec);
  R.set(LHS, MHSe);
  return RHS->resolveReferences(R);
}

static const Init *ForeachDagApply(const Init *LHS, const DagInit *MHSd,
                                   const Init *RHS, const Record *CurRec) {
  bool Change = false;
  const Init *Val = ItemApply(LHS, MHSd->getOperator(), RHS, CurRec);
  if (Val != MHSd->getOperator())
    Change = true;

  SmallVector<std::pair<const Init *, const StringInit *>, 8> NewArgs;
  for (auto [Arg, ArgName] : MHSd->getArgAndNames()) {
    const Init *NewArg;

    if (const auto *Argd = dyn_cast<DagInit>(Arg))
      NewArg = ForeachDagApply(LHS, Argd, RHS, CurRec);
    else
      NewArg = ItemApply(LHS, Arg, RHS, CurRec);

    NewArgs.emplace_back(NewArg, ArgName);
    if (Arg != NewArg)
      Change = true;
  }

  if (Change)
    return DagInit::get(Val, MHSd->getName(), NewArgs);
  return MHSd;
}

// Applies RHS to all elements of MHS, using LHS as a temp variable.
static const Init *ForeachHelper(const Init *LHS, const Init *MHS,
                                 const Init *RHS, const RecTy *Type,
                                 const Record *CurRec) {
  if (const auto *MHSd = dyn_cast<DagInit>(MHS))
    return ForeachDagApply(LHS, MHSd, RHS, CurRec);

  if (const auto *MHSl = dyn_cast<ListInit>(MHS)) {
    SmallVector<const Init *, 8> NewList(MHSl->begin(), MHSl->end());

    for (const Init *&Item : NewList) {
      const Init *NewItem = ItemApply(LHS, Item, RHS, CurRec);
      if (NewItem != Item)
        Item = NewItem;
    }
    return ListInit::get(NewList, cast<ListRecTy>(Type)->getElementType());
  }

  return nullptr;
}

// Evaluates RHS for all elements of MHS, using LHS as a temp variable.
// Creates a new list with the elements that evaluated to true.
static const Init *FilterHelper(const Init *LHS, const Init *MHS,
                                const Init *RHS, const RecTy *Type,
                                const Record *CurRec) {
  if (const auto *MHSl = dyn_cast<ListInit>(MHS)) {
    SmallVector<const Init *, 8> NewList;

    for (const Init *Item : MHSl->getElements()) {
      const Init *Include = ItemApply(LHS, Item, RHS, CurRec);
      if (!Include)
        return nullptr;
      if (const auto *IncludeInt =
              dyn_cast_or_null<IntInit>(Include->convertInitializerTo(
                  IntRecTy::get(LHS->getRecordKeeper())))) {
        if (IncludeInt->getValue())
          NewList.push_back(Item);
      } else {
        return nullptr;
      }
    }
    return ListInit::get(NewList, cast<ListRecTy>(Type)->getElementType());
  }

  return nullptr;
}

const Init *TernOpInit::Fold(const Record *CurRec) const {
  RecordKeeper &RK = getRecordKeeper();
  switch (getOpcode()) {
  case SUBST: {
    const auto *LHSd = dyn_cast<DefInit>(LHS);
    const auto *LHSv = dyn_cast<VarInit>(LHS);
    const auto *LHSs = dyn_cast<StringInit>(LHS);

    const auto *MHSd = dyn_cast<DefInit>(MHS);
    const auto *MHSv = dyn_cast<VarInit>(MHS);
    const auto *MHSs = dyn_cast<StringInit>(MHS);

    const auto *RHSd = dyn_cast<DefInit>(RHS);
    const auto *RHSv = dyn_cast<VarInit>(RHS);
    const auto *RHSs = dyn_cast<StringInit>(RHS);

    if (LHSd && MHSd && RHSd) {
      const Record *Val = RHSd->getDef();
      if (LHSd->getAsString() == RHSd->getAsString())
        Val = MHSd->getDef();
      return Val->getDefInit();
    }
    if (LHSv && MHSv && RHSv) {
      std::string Val = RHSv->getName().str();
      if (LHSv->getAsString() == RHSv->getAsString())
        Val = MHSv->getName().str();
      return VarInit::get(Val, getType());
    }
    if (LHSs && MHSs && RHSs) {
      std::string Val = RHSs->getValue().str();

      std::string::size_type Idx = 0;
      while (true) {
        std::string::size_type Found = Val.find(LHSs->getValue(), Idx);
        if (Found == std::string::npos)
          break;
        Val.replace(Found, LHSs->getValue().size(), MHSs->getValue().str());
        Idx = Found + MHSs->getValue().size();
      }

      return StringInit::get(RK, Val);
    }
    break;
  }

  case FOREACH: {
    if (const Init *Result = ForeachHelper(LHS, MHS, RHS, getType(), CurRec))
      return Result;
    break;
  }

  case FILTER: {
    if (const Init *Result = FilterHelper(LHS, MHS, RHS, getType(), CurRec))
      return Result;
    break;
  }

  case IF: {
    if (const auto *LHSi = dyn_cast_or_null<IntInit>(
            LHS->convertInitializerTo(IntRecTy::get(RK)))) {
      if (LHSi->getValue())
        return MHS;
      return RHS;
    }
    break;
  }

  case DAG: {
    const auto *MHSl = dyn_cast<ListInit>(MHS);
    const auto *RHSl = dyn_cast<ListInit>(RHS);
    bool MHSok = MHSl || isa<UnsetInit>(MHS);
    bool RHSok = RHSl || isa<UnsetInit>(RHS);

    if (isa<UnsetInit>(MHS) && isa<UnsetInit>(RHS))
      break; // Typically prevented by the parser, but might happen with template args

    if (MHSok && RHSok && (!MHSl || !RHSl || MHSl->size() == RHSl->size())) {
      SmallVector<std::pair<const Init *, const StringInit *>, 8> Children;
      unsigned Size = MHSl ? MHSl->size() : RHSl->size();
      for (unsigned i = 0; i != Size; ++i) {
        const Init *Node = MHSl ? MHSl->getElement(i) : UnsetInit::get(RK);
        const Init *Name = RHSl ? RHSl->getElement(i) : UnsetInit::get(RK);
        if (!isa<StringInit>(Name) && !isa<UnsetInit>(Name))
          return this;
        Children.emplace_back(Node, dyn_cast<StringInit>(Name));
      }
      return DagInit::get(LHS, Children);
    }
    break;
  }

  case RANGE: {
    const auto *LHSi = dyn_cast<IntInit>(LHS);
    const auto *MHSi = dyn_cast<IntInit>(MHS);
    const auto *RHSi = dyn_cast<IntInit>(RHS);
    if (!LHSi || !MHSi || !RHSi)
      break;

    auto Start = LHSi->getValue();
    auto End = MHSi->getValue();
    auto Step = RHSi->getValue();
    if (Step == 0)
      PrintError(CurRec->getLoc(), "Step of !range can't be 0");

    SmallVector<const Init *, 8> Args;
    if (Start < End && Step > 0) {
      Args.reserve((End - Start) / Step);
      for (auto I = Start; I < End; I += Step)
        Args.push_back(IntInit::get(getRecordKeeper(), I));
    } else if (Start > End && Step < 0) {
      Args.reserve((Start - End) / -Step);
      for (auto I = Start; I > End; I += Step)
        Args.push_back(IntInit::get(getRecordKeeper(), I));
    } else {
      // Empty set
    }
    return ListInit::get(Args, LHSi->getType());
  }

  case SUBSTR: {
    const auto *LHSs = dyn_cast<StringInit>(LHS);
    const auto *MHSi = dyn_cast<IntInit>(MHS);
    const auto *RHSi = dyn_cast<IntInit>(RHS);
    if (LHSs && MHSi && RHSi) {
      int64_t StringSize = LHSs->getValue().size();
      int64_t Start = MHSi->getValue();
      int64_t Length = RHSi->getValue();
      if (Start < 0 || Start > StringSize)
        PrintError(CurRec->getLoc(),
                   Twine("!substr start position is out of range 0...") +
                       std::to_string(StringSize) + ": " +
                       std::to_string(Start));
      if (Length < 0)
        PrintError(CurRec->getLoc(), "!substr length must be nonnegative");
      return StringInit::get(RK, LHSs->getValue().substr(Start, Length),
                             LHSs->getFormat());
    }
    break;
  }

  case FIND: {
    const auto *LHSs = dyn_cast<StringInit>(LHS);
    const auto *MHSs = dyn_cast<StringInit>(MHS);
    const auto *RHSi = dyn_cast<IntInit>(RHS);
    if (LHSs && MHSs && RHSi) {
      int64_t SourceSize = LHSs->getValue().size();
      int64_t Start = RHSi->getValue();
      if (Start < 0 || Start > SourceSize)
        PrintError(CurRec->getLoc(),
                   Twine("!find start position is out of range 0...") +
                       std::to_string(SourceSize) + ": " +
                       std::to_string(Start));
      auto I = LHSs->getValue().find(MHSs->getValue(), Start);
      if (I == std::string::npos)
        return IntInit::get(RK, -1);
      return IntInit::get(RK, I);
    }
    break;
  }

  case SETDAGARG: {
    const auto *Dag = dyn_cast<DagInit>(LHS);
    if (Dag && isa<IntInit, StringInit>(MHS)) {
      std::string Error;
      auto ArgNo = getDagArgNoByKey(Dag, MHS, Error);
      if (!ArgNo)
        PrintFatalError(CurRec->getLoc(), "!setdagarg " + Error);

      assert(*ArgNo < Dag->getNumArgs());

      SmallVector<const Init *, 8> Args(Dag->getArgs());
      Args[*ArgNo] = RHS;
      return DagInit::get(Dag->getOperator(), Dag->getName(), Args,
                          Dag->getArgNames());
    }
    break;
  }

  case SETDAGNAME: {
    const auto *Dag = dyn_cast<DagInit>(LHS);
    if (Dag && isa<IntInit, StringInit>(MHS)) {
      std::string Error;
      auto ArgNo = getDagArgNoByKey(Dag, MHS, Error);
      if (!ArgNo)
        PrintFatalError(CurRec->getLoc(), "!setdagname " + Error);

      assert(*ArgNo < Dag->getNumArgs());

      SmallVector<const StringInit *, 8> Names(Dag->getArgNames());
      Names[*ArgNo] = dyn_cast<StringInit>(RHS);
      return DagInit::get(Dag->getOperator(), Dag->getName(), Dag->getArgs(),
                          Names);
    }
    break;
  }
  }

  return this;
}

const Init *TernOpInit::resolveReferences(Resolver &R) const {
  const Init *lhs = LHS->resolveReferences(R);

  if (getOpcode() == IF && lhs != LHS) {
    if (const auto *Value = dyn_cast_or_null<IntInit>(
            lhs->convertInitializerTo(IntRecTy::get(getRecordKeeper())))) {
      // Short-circuit
      if (Value->getValue())
        return MHS->resolveReferences(R);
      return RHS->resolveReferences(R);
    }
  }

  const Init *mhs = MHS->resolveReferences(R);
  const Init *rhs;

  if (getOpcode() == FOREACH || getOpcode() == FILTER) {
    ShadowResolver SR(R);
    SR.addShadow(lhs);
    rhs = RHS->resolveReferences(SR);
  } else {
    rhs = RHS->resolveReferences(R);
  }

  if (LHS != lhs || MHS != mhs || RHS != rhs)
    return (TernOpInit::get(getOpcode(), lhs, mhs, rhs, getType()))
        ->Fold(R.getCurrentRecord());
  return this;
}

std::string TernOpInit::getAsString() const {
  std::string Result;
  bool UnquotedLHS = false;
  switch (getOpcode()) {
  case DAG: Result = "!dag"; break;
  case FILTER: Result = "!filter"; UnquotedLHS = true; break;
  case FOREACH: Result = "!foreach"; UnquotedLHS = true; break;
  case IF: Result = "!if"; break;
  case RANGE:
    Result = "!range";
    break;
  case SUBST: Result = "!subst"; break;
  case SUBSTR: Result = "!substr"; break;
  case FIND: Result = "!find"; break;
  case SETDAGARG:
    Result = "!setdagarg";
    break;
  case SETDAGNAME:
    Result = "!setdagname";
    break;
  }
  return (Result + "(" +
          (UnquotedLHS ? LHS->getAsUnquotedString() : LHS->getAsString()) +
          ", " + MHS->getAsString() + ", " + RHS->getAsString() + ")");
}

static void ProfileFoldOpInit(FoldingSetNodeID &ID, const Init *Start,
                              const Init *List, const Init *A, const Init *B,
                              const Init *Expr, const RecTy *Type) {
  ID.AddPointer(Start);
  ID.AddPointer(List);
  ID.AddPointer(A);
  ID.AddPointer(B);
  ID.AddPointer(Expr);
  ID.AddPointer(Type);
}

const FoldOpInit *FoldOpInit::get(const Init *Start, const Init *List,
                                  const Init *A, const Init *B,
                                  const Init *Expr, const RecTy *Type) {
  FoldingSetNodeID ID;
  ProfileFoldOpInit(ID, Start, List, A, B, Expr, Type);

  detail::RecordKeeperImpl &RK = Start->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (const FoldOpInit *I = RK.TheFoldOpInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  FoldOpInit *I = new (RK.Allocator) FoldOpInit(Start, List, A, B, Expr, Type);
  RK.TheFoldOpInitPool.InsertNode(I, IP);
  return I;
}

void FoldOpInit::Profile(FoldingSetNodeID &ID) const {
  ProfileFoldOpInit(ID, Start, List, A, B, Expr, getType());
}

const Init *FoldOpInit::Fold(const Record *CurRec) const {
  if (const auto *LI = dyn_cast<ListInit>(List)) {
    const Init *Accum = Start;
    for (const Init *Elt : *LI) {
      MapResolver R(CurRec);
      R.set(A, Accum);
      R.set(B, Elt);
      Accum = Expr->resolveReferences(R);
    }
    return Accum;
  }
  return this;
}

const Init *FoldOpInit::resolveReferences(Resolver &R) const {
  const Init *NewStart = Start->resolveReferences(R);
  const Init *NewList = List->resolveReferences(R);
  ShadowResolver SR(R);
  SR.addShadow(A);
  SR.addShadow(B);
  const Init *NewExpr = Expr->resolveReferences(SR);

  if (Start == NewStart && List == NewList && Expr == NewExpr)
    return this;

  return get(NewStart, NewList, A, B, NewExpr, getType())
      ->Fold(R.getCurrentRecord());
}

const Init *FoldOpInit::getBit(unsigned Bit) const {
  return VarBitInit::get(this, Bit);
}

std::string FoldOpInit::getAsString() const {
  return (Twine("!foldl(") + Start->getAsString() + ", " + List->getAsString() +
          ", " + A->getAsUnquotedString() + ", " + B->getAsUnquotedString() +
          ", " + Expr->getAsString() + ")")
      .str();
}

static void ProfileIsAOpInit(FoldingSetNodeID &ID, const RecTy *CheckType,
                             const Init *Expr) {
  ID.AddPointer(CheckType);
  ID.AddPointer(Expr);
}

const IsAOpInit *IsAOpInit::get(const RecTy *CheckType, const Init *Expr) {

  FoldingSetNodeID ID;
  ProfileIsAOpInit(ID, CheckType, Expr);

  detail::RecordKeeperImpl &RK = Expr->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (const IsAOpInit *I = RK.TheIsAOpInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  IsAOpInit *I = new (RK.Allocator) IsAOpInit(CheckType, Expr);
  RK.TheIsAOpInitPool.InsertNode(I, IP);
  return I;
}

void IsAOpInit::Profile(FoldingSetNodeID &ID) const {
  ProfileIsAOpInit(ID, CheckType, Expr);
}

const Init *IsAOpInit::Fold() const {
  if (const auto *TI = dyn_cast<TypedInit>(Expr)) {
    // Is the expression type known to be (a subclass of) the desired type?
    if (TI->getType()->typeIsConvertibleTo(CheckType))
      return IntInit::get(getRecordKeeper(), 1);

    if (isa<RecordRecTy>(CheckType)) {
      // If the target type is not a subclass of the expression type once the
      // expression has been made concrete, or if the expression has fully
      // resolved to a record, we know that it can't be of the required type.
      if ((!CheckType->typeIsConvertibleTo(TI->getType()) &&
           Expr->isConcrete()) ||
          isa<DefInit>(Expr))
        return IntInit::get(getRecordKeeper(), 0);
    } else {
      // We treat non-record types as not castable.
      return IntInit::get(getRecordKeeper(), 0);
    }
  }
  return this;
}

const Init *IsAOpInit::resolveReferences(Resolver &R) const {
  const Init *NewExpr = Expr->resolveReferences(R);
  if (Expr != NewExpr)
    return get(CheckType, NewExpr)->Fold();
  return this;
}

const Init *IsAOpInit::getBit(unsigned Bit) const {
  return VarBitInit::get(this, Bit);
}

std::string IsAOpInit::getAsString() const {
  return (Twine("!isa<") + CheckType->getAsString() + ">(" +
          Expr->getAsString() + ")")
      .str();
}

static void ProfileExistsOpInit(FoldingSetNodeID &ID, const RecTy *CheckType,
                                const Init *Expr) {
  ID.AddPointer(CheckType);
  ID.AddPointer(Expr);
}

const ExistsOpInit *ExistsOpInit::get(const RecTy *CheckType,
                                      const Init *Expr) {
  FoldingSetNodeID ID;
  ProfileExistsOpInit(ID, CheckType, Expr);

  detail::RecordKeeperImpl &RK = Expr->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (const ExistsOpInit *I =
          RK.TheExistsOpInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  ExistsOpInit *I = new (RK.Allocator) ExistsOpInit(CheckType, Expr);
  RK.TheExistsOpInitPool.InsertNode(I, IP);
  return I;
}

void ExistsOpInit::Profile(FoldingSetNodeID &ID) const {
  ProfileExistsOpInit(ID, CheckType, Expr);
}

const Init *ExistsOpInit::Fold(const Record *CurRec, bool IsFinal) const {
  if (const auto *Name = dyn_cast<StringInit>(Expr)) {
    // Look up all defined records to see if we can find one.
    const Record *D = CheckType->getRecordKeeper().getDef(Name->getValue());
    if (D) {
      // Check if types are compatible.
      return IntInit::get(getRecordKeeper(),
                          D->getDefInit()->getType()->typeIsA(CheckType));
    }

    if (CurRec) {
      // Self-references are allowed, but their resolution is delayed until
      // the final resolve to ensure that we get the correct type for them.
      auto *Anonymous = dyn_cast<AnonymousNameInit>(CurRec->getNameInit());
      if (Name == CurRec->getNameInit() ||
          (Anonymous && Name == Anonymous->getNameInit())) {
        if (!IsFinal)
          return this;

        // No doubt that there exists a record, so we should check if types are
        // compatible.
        return IntInit::get(getRecordKeeper(),
                            CurRec->getType()->typeIsA(CheckType));
      }
    }

    if (IsFinal)
      return IntInit::get(getRecordKeeper(), 0);
  }
  return this;
}

const Init *ExistsOpInit::resolveReferences(Resolver &R) const {
  const Init *NewExpr = Expr->resolveReferences(R);
  if (Expr != NewExpr || R.isFinal())
    return get(CheckType, NewExpr)->Fold(R.getCurrentRecord(), R.isFinal());
  return this;
}

const Init *ExistsOpInit::getBit(unsigned Bit) const {
  return VarBitInit::get(this, Bit);
}

std::string ExistsOpInit::getAsString() const {
  return (Twine("!exists<") + CheckType->getAsString() + ">(" +
          Expr->getAsString() + ")")
      .str();
}

static void ProfileInstancesOpInit(FoldingSetNodeID &ID, const RecTy *Type,
                                   const Init *Regex) {
  ID.AddPointer(Type);
  ID.AddPointer(Regex);
}

const InstancesOpInit *InstancesOpInit::get(const RecTy *Type,
                                            const Init *Regex) {
  FoldingSetNodeID ID;
  ProfileInstancesOpInit(ID, Type, Regex);

  detail::RecordKeeperImpl &RK = Regex->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (const InstancesOpInit *I =
          RK.TheInstancesOpInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  InstancesOpInit *I = new (RK.Allocator) InstancesOpInit(Type, Regex);
  RK.TheInstancesOpInitPool.InsertNode(I, IP);
  return I;
}

void InstancesOpInit::Profile(FoldingSetNodeID &ID) const {
  ProfileInstancesOpInit(ID, Type, Regex);
}

const Init *InstancesOpInit::Fold(const Record *CurRec, bool IsFinal) const {
  if (CurRec && !IsFinal)
    return this;

  const auto *RegexInit = dyn_cast<StringInit>(Regex);
  if (!RegexInit)
    return this;

  StringRef RegexStr = RegexInit->getValue();
  llvm::Regex Matcher(RegexStr);
  if (!Matcher.isValid())
    PrintFatalError(Twine("invalid regex '") + RegexStr + Twine("'"));

  const RecordKeeper &RK = Type->getRecordKeeper();
  SmallVector<Init *, 8> Selected;
  for (auto &Def : RK.getAllDerivedDefinitionsIfDefined(Type->getAsString()))
    if (Matcher.match(Def->getName()))
      Selected.push_back(Def->getDefInit());

  return ListInit::get(Selected, Type);
}

const Init *InstancesOpInit::resolveReferences(Resolver &R) const {
  const Init *NewRegex = Regex->resolveReferences(R);
  if (Regex != NewRegex || R.isFinal())
    return get(Type, NewRegex)->Fold(R.getCurrentRecord(), R.isFinal());
  return this;
}

const Init *InstancesOpInit::getBit(unsigned Bit) const {
  return VarBitInit::get(this, Bit);
}

std::string InstancesOpInit::getAsString() const {
  return "!instances<" + Type->getAsString() + ">(" + Regex->getAsString() +
         ")";
}

const RecTy *TypedInit::getFieldType(const StringInit *FieldName) const {
  if (const auto *RecordType = dyn_cast<RecordRecTy>(getType())) {
    for (const Record *Rec : RecordType->getClasses()) {
      if (const RecordVal *Field = Rec->getValue(FieldName))
        return Field->getType();
    }
  }
  return nullptr;
}

const Init *TypedInit::convertInitializerTo(const RecTy *Ty) const {
  if (getType() == Ty || getType()->typeIsA(Ty))
    return this;

  if (isa<BitRecTy>(getType()) && isa<BitsRecTy>(Ty) &&
      cast<BitsRecTy>(Ty)->getNumBits() == 1)
    return BitsInit::get(getRecordKeeper(), {this});

  return nullptr;
}

const Init *
TypedInit::convertInitializerBitRange(ArrayRef<unsigned> Bits) const {
  const auto *T = dyn_cast<BitsRecTy>(getType());
  if (!T) return nullptr;  // Cannot subscript a non-bits variable.
  unsigned NumBits = T->getNumBits();

  SmallVector<const Init *, 16> NewBits;
  NewBits.reserve(Bits.size());
  for (unsigned Bit : Bits) {
    if (Bit >= NumBits)
      return nullptr;

    NewBits.push_back(VarBitInit::get(this, Bit));
  }
  return BitsInit::get(getRecordKeeper(), NewBits);
}

const Init *TypedInit::getCastTo(const RecTy *Ty) const {
  // Handle the common case quickly
  if (getType() == Ty || getType()->typeIsA(Ty))
    return this;

  if (const Init *Converted = convertInitializerTo(Ty)) {
    assert(!isa<TypedInit>(Converted) ||
           cast<TypedInit>(Converted)->getType()->typeIsA(Ty));
    return Converted;
  }

  if (!getType()->typeIsConvertibleTo(Ty))
    return nullptr;

  return UnOpInit::get(UnOpInit::CAST, this, Ty)->Fold(nullptr);
}

const VarInit *VarInit::get(StringRef VN, const RecTy *T) {
  const Init *Value = StringInit::get(T->getRecordKeeper(), VN);
  return VarInit::get(Value, T);
}

const VarInit *VarInit::get(const Init *VN, const RecTy *T) {
  detail::RecordKeeperImpl &RK = T->getRecordKeeper().getImpl();
  VarInit *&I = RK.TheVarInitPool[{T, VN}];
  if (!I)
    I = new (RK.Allocator) VarInit(VN, T);
  return I;
}

StringRef VarInit::getName() const {
  const auto *NameString = cast<StringInit>(getNameInit());
  return NameString->getValue();
}

const Init *VarInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get(getRecordKeeper()))
    return this;
  return VarBitInit::get(this, Bit);
}

const Init *VarInit::resolveReferences(Resolver &R) const {
  if (const Init *Val = R.resolve(VarName))
    return Val;
  return this;
}

const VarBitInit *VarBitInit::get(const TypedInit *T, unsigned B) {
  detail::RecordKeeperImpl &RK = T->getRecordKeeper().getImpl();
  VarBitInit *&I = RK.TheVarBitInitPool[{T, B}];
  if (!I)
    I = new (RK.Allocator) VarBitInit(T, B);
  return I;
}

std::string VarBitInit::getAsString() const {
  return TI->getAsString() + "{" + utostr(Bit) + "}";
}

const Init *VarBitInit::resolveReferences(Resolver &R) const {
  const Init *I = TI->resolveReferences(R);
  if (TI != I)
    return I->getBit(getBitNum());

  return this;
}

DefInit::DefInit(const Record *D)
    : TypedInit(IK_DefInit, D->getType()), Def(D) {}

const Init *DefInit::convertInitializerTo(const RecTy *Ty) const {
  if (auto *RRT = dyn_cast<RecordRecTy>(Ty))
    if (getType()->typeIsConvertibleTo(RRT))
      return this;
  return nullptr;
}

const RecTy *DefInit::getFieldType(const StringInit *FieldName) const {
  if (const RecordVal *RV = Def->getValue(FieldName))
    return RV->getType();
  return nullptr;
}

std::string DefInit::getAsString() const { return Def->getName().str(); }

static void ProfileVarDefInit(FoldingSetNodeID &ID, const Record *Class,
                              ArrayRef<const ArgumentInit *> Args) {
  ID.AddInteger(Args.size());
  ID.AddPointer(Class);

  for (const Init *I : Args)
    ID.AddPointer(I);
}

VarDefInit::VarDefInit(SMLoc Loc, const Record *Class,
                       ArrayRef<const ArgumentInit *> Args)
    : TypedInit(IK_VarDefInit, RecordRecTy::get(Class)), Loc(Loc), Class(Class),
      NumArgs(Args.size()) {
  llvm::uninitialized_copy(Args, getTrailingObjects());
}

const VarDefInit *VarDefInit::get(SMLoc Loc, const Record *Class,
                                  ArrayRef<const ArgumentInit *> Args) {
  FoldingSetNodeID ID;
  ProfileVarDefInit(ID, Class, Args);

  detail::RecordKeeperImpl &RK = Class->getRecords().getImpl();
  void *IP = nullptr;
  if (const VarDefInit *I = RK.TheVarDefInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  void *Mem = RK.Allocator.Allocate(
      totalSizeToAlloc<const ArgumentInit *>(Args.size()), alignof(VarDefInit));
  VarDefInit *I = new (Mem) VarDefInit(Loc, Class, Args);
  RK.TheVarDefInitPool.InsertNode(I, IP);
  return I;
}

void VarDefInit::Profile(FoldingSetNodeID &ID) const {
  ProfileVarDefInit(ID, Class, args());
}

const DefInit *VarDefInit::instantiate() {
  if (Def)
    return Def;

  RecordKeeper &Records = Class->getRecords();
  auto NewRecOwner = std::make_unique<Record>(
      Records.getNewAnonymousName(), Loc, Records, Record::RK_AnonymousDef);
  Record *NewRec = NewRecOwner.get();

  // Copy values from class to instance
  for (const RecordVal &Val : Class->getValues())
    NewRec->addValue(Val);

  // Copy assertions from class to instance.
  NewRec->appendAssertions(Class);

  // Copy dumps from class to instance.
  NewRec->appendDumps(Class);

  // Substitute and resolve template arguments
  ArrayRef<const Init *> TArgs = Class->getTemplateArgs();
  MapResolver R(NewRec);

  for (const Init *Arg : TArgs) {
    R.set(Arg, NewRec->getValue(Arg)->getValue());
    NewRec->removeValue(Arg);
  }

  for (auto *Arg : args()) {
    if (Arg->isPositional())
      R.set(TArgs[Arg->getIndex()], Arg->getValue());
    if (Arg->isNamed())
      R.set(Arg->getName(), Arg->getValue());
  }

  NewRec->resolveReferences(R);

  // Add superclass.
  NewRec->addDirectSuperClass(
      Class, SMRange(Class->getLoc().back(), Class->getLoc().back()));

  // Resolve internal references and store in record keeper
  NewRec->resolveReferences();
  Records.addDef(std::move(NewRecOwner));

  // Check the assertions.
  NewRec->checkRecordAssertions();

  // Check the assertions.
  NewRec->emitRecordDumps();

  return Def = NewRec->getDefInit();
}

const Init *VarDefInit::resolveReferences(Resolver &R) const {
  TrackUnresolvedResolver UR(&R);
  bool Changed = false;
  SmallVector<const ArgumentInit *, 8> NewArgs;
  NewArgs.reserve(args_size());

  for (const ArgumentInit *Arg : args()) {
    const auto *NewArg = cast<ArgumentInit>(Arg->resolveReferences(UR));
    NewArgs.push_back(NewArg);
    Changed |= NewArg != Arg;
  }

  if (Changed) {
    auto *New = VarDefInit::get(Loc, Class, NewArgs);
    if (!UR.foundUnresolved())
      return const_cast<VarDefInit *>(New)->instantiate();
    return New;
  }
  return this;
}

const Init *VarDefInit::Fold() const {
  if (Def)
    return Def;

  TrackUnresolvedResolver R;
  for (const Init *Arg : args())
    Arg->resolveReferences(R);

  if (!R.foundUnresolved())
    return const_cast<VarDefInit *>(this)->instantiate();
  return this;
}

std::string VarDefInit::getAsString() const {
  std::string Result = Class->getNameInitAsString() + "<";
  ListSeparator LS;
  for (const Init *Arg : args()) {
    Result += LS;
    Result += Arg->getAsString();
  }
  return Result + ">";
}

const FieldInit *FieldInit::get(const Init *R, const StringInit *FN) {
  detail::RecordKeeperImpl &RK = R->getRecordKeeper().getImpl();
  FieldInit *&I = RK.TheFieldInitPool[{R, FN}];
  if (!I)
    I = new (RK.Allocator) FieldInit(R, FN);
  return I;
}

const Init *FieldInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get(getRecordKeeper()))
    return this;
  return VarBitInit::get(this, Bit);
}

const Init *FieldInit::resolveReferences(Resolver &R) const {
  const Init *NewRec = Rec->resolveReferences(R);
  if (NewRec != Rec)
    return FieldInit::get(NewRec, FieldName)->Fold(R.getCurrentRecord());
  return this;
}

const Init *FieldInit::Fold(const Record *CurRec) const {
  if (const auto *DI = dyn_cast<DefInit>(Rec)) {
    const Record *Def = DI->getDef();
    if (Def == CurRec)
      PrintFatalError(CurRec->getLoc(),
                      Twine("Attempting to access field '") +
                      FieldName->getAsUnquotedString() + "' of '" +
                      Rec->getAsString() + "' is a forbidden self-reference");
    const Init *FieldVal = Def->getValue(FieldName)->getValue();
    if (FieldVal->isConcrete())
      return FieldVal;
  }
  return this;
}

bool FieldInit::isConcrete() const {
  if (const auto *DI = dyn_cast<DefInit>(Rec)) {
    const Init *FieldVal = DI->getDef()->getValue(FieldName)->getValue();
    return FieldVal->isConcrete();
  }
  return false;
}

static void ProfileCondOpInit(FoldingSetNodeID &ID,
                              ArrayRef<const Init *> Conds,
                              ArrayRef<const Init *> Vals,
                              const RecTy *ValType) {
  assert(Conds.size() == Vals.size() &&
         "Number of conditions and values must match!");
  ID.AddPointer(ValType);

  for (const auto &[Cond, Val] : zip(Conds, Vals)) {
    ID.AddPointer(Cond);
    ID.AddPointer(Val);
  }
}

CondOpInit::CondOpInit(ArrayRef<const Init *> Conds,
                       ArrayRef<const Init *> Values, const RecTy *Type)
    : TypedInit(IK_CondOpInit, Type), NumConds(Conds.size()), ValType(Type) {
  const Init **TrailingObjects = getTrailingObjects();
  llvm::uninitialized_copy(Conds, TrailingObjects);
  llvm::uninitialized_copy(Values, TrailingObjects + NumConds);
}

void CondOpInit::Profile(FoldingSetNodeID &ID) const {
  ProfileCondOpInit(ID, getConds(), getVals(), ValType);
}

const CondOpInit *CondOpInit::get(ArrayRef<const Init *> Conds,
                                  ArrayRef<const Init *> Values,
                                  const RecTy *Ty) {
  assert(Conds.size() == Values.size() &&
         "Number of conditions and values must match!");

  FoldingSetNodeID ID;
  ProfileCondOpInit(ID, Conds, Values, Ty);

  detail::RecordKeeperImpl &RK = Ty->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (const CondOpInit *I = RK.TheCondOpInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  void *Mem = RK.Allocator.Allocate(
      totalSizeToAlloc<const Init *>(2 * Conds.size()), alignof(CondOpInit));
  CondOpInit *I = new (Mem) CondOpInit(Conds, Values, Ty);
  RK.TheCondOpInitPool.InsertNode(I, IP);
  return I;
}

const Init *CondOpInit::resolveReferences(Resolver &R) const {
  SmallVector<const Init *, 4> NewConds;
  SmallVector<const Init *, 4> NewVals;

  bool Changed = false;
  for (auto [Cond, Val] : getCondAndVals()) {
    const Init *NewCond = Cond->resolveReferences(R);
    NewConds.push_back(NewCond);
    Changed |= NewCond != Cond;

    const Init *NewVal = Val->resolveReferences(R);
    NewVals.push_back(NewVal);
    Changed |= NewVal != Val;
  }

  if (Changed)
    return (CondOpInit::get(NewConds, NewVals,
            getValType()))->Fold(R.getCurrentRecord());

  return this;
}

const Init *CondOpInit::Fold(const Record *CurRec) const {
  RecordKeeper &RK = getRecordKeeper();
  for (auto [Cond, Val] : getCondAndVals()) {
    if (const auto *CondI = dyn_cast_or_null<IntInit>(
            Cond->convertInitializerTo(IntRecTy::get(RK)))) {
      if (CondI->getValue())
        return Val->convertInitializerTo(getValType());
    } else {
      return this;
    }
  }

  PrintFatalError(CurRec->getLoc(),
                  CurRec->getNameInitAsString() +
                  " does not have any true condition in:" +
                  this->getAsString());
  return nullptr;
}

bool CondOpInit::isConcrete() const {
  return all_of(getCondAndVals(), [](const auto &Pair) {
    return std::get<0>(Pair)->isConcrete() && std::get<1>(Pair)->isConcrete();
  });
}

bool CondOpInit::isComplete() const {
  return all_of(getCondAndVals(), [](const auto &Pair) {
    return std::get<0>(Pair)->isComplete() && std::get<1>(Pair)->isComplete();
  });
}

std::string CondOpInit::getAsString() const {
  std::string Result = "!cond(";
  ListSeparator LS;
  for (auto [Cond, Val] : getCondAndVals()) {
    Result += LS;
    Result += Cond->getAsString() + ": ";
    Result += Val->getAsString();
  }
  return Result + ")";
}

const Init *CondOpInit::getBit(unsigned Bit) const {
  return VarBitInit::get(this, Bit);
}

static void ProfileDagInit(FoldingSetNodeID &ID, const Init *V,
                           const StringInit *VN, ArrayRef<const Init *> Args,
                           ArrayRef<const StringInit *> ArgNames) {
  ID.AddPointer(V);
  ID.AddPointer(VN);

  for (auto [Arg, Name] : zip_equal(Args, ArgNames)) {
    ID.AddPointer(Arg);
    ID.AddPointer(Name);
  }
}

DagInit::DagInit(const Init *V, const StringInit *VN,
                 ArrayRef<const Init *> Args,
                 ArrayRef<const StringInit *> ArgNames)
    : TypedInit(IK_DagInit, DagRecTy::get(V->getRecordKeeper())), Val(V),
      ValName(VN), NumArgs(Args.size()) {
  llvm::uninitialized_copy(Args, getTrailingObjects<const Init *>());
  llvm::uninitialized_copy(ArgNames, getTrailingObjects<const StringInit *>());
}

const DagInit *DagInit::get(const Init *V, const StringInit *VN,
                            ArrayRef<const Init *> Args,
                            ArrayRef<const StringInit *> ArgNames) {
  assert(Args.size() == ArgNames.size() &&
         "Number of DAG args and arg names must match!");

  FoldingSetNodeID ID;
  ProfileDagInit(ID, V, VN, Args, ArgNames);

  detail::RecordKeeperImpl &RK = V->getRecordKeeper().getImpl();
  void *IP = nullptr;
  if (const DagInit *I = RK.TheDagInitPool.FindNodeOrInsertPos(ID, IP))
    return I;

  void *Mem =
      RK.Allocator.Allocate(totalSizeToAlloc<const Init *, const StringInit *>(
                                Args.size(), ArgNames.size()),
                            alignof(DagInit));
  DagInit *I = new (Mem) DagInit(V, VN, Args, ArgNames);
  RK.TheDagInitPool.InsertNode(I, IP);
  return I;
}

const DagInit *DagInit::get(
    const Init *V, const StringInit *VN,
    ArrayRef<std::pair<const Init *, const StringInit *>> ArgAndNames) {
  SmallVector<const Init *, 8> Args(make_first_range(ArgAndNames));
  SmallVector<const StringInit *, 8> Names(make_second_range(ArgAndNames));
  return DagInit::get(V, VN, Args, Names);
}

void DagInit::Profile(FoldingSetNodeID &ID) const {
  ProfileDagInit(ID, Val, ValName, getArgs(), getArgNames());
}

const Record *DagInit::getOperatorAsDef(ArrayRef<SMLoc> Loc) const {
  if (const auto *DefI = dyn_cast<DefInit>(Val))
    return DefI->getDef();
  PrintFatalError(Loc, "Expected record as operator");
  return nullptr;
}

std::optional<unsigned> DagInit::getArgNo(StringRef Name) const {
  ArrayRef<const StringInit *> ArgNames = getArgNames();
  auto It = llvm::find_if(ArgNames, [Name](const StringInit *ArgName) {
    return ArgName && ArgName->getValue() == Name;
  });
  if (It == ArgNames.end())
    return std::nullopt;
  return std::distance(ArgNames.begin(), It);
}

const Init *DagInit::resolveReferences(Resolver &R) const {
  SmallVector<const Init *, 8> NewArgs;
  NewArgs.reserve(arg_size());
  bool ArgsChanged = false;
  for (const Init *Arg : getArgs()) {
    const Init *NewArg = Arg->resolveReferences(R);
    NewArgs.push_back(NewArg);
    ArgsChanged |= NewArg != Arg;
  }

  const Init *Op = Val->resolveReferences(R);
  if (Op != Val || ArgsChanged)
    return DagInit::get(Op, ValName, NewArgs, getArgNames());

  return this;
}

bool DagInit::isConcrete() const {
  if (!Val->isConcrete())
    return false;
  return all_of(getArgs(), [](const Init *Elt) { return Elt->isConcrete(); });
}

std::string DagInit::getAsString() const {
  std::string Result = "(" + Val->getAsString();
  if (ValName)
    Result += ":$" + ValName->getAsUnquotedString();
  if (!arg_empty()) {
    Result += " ";
    ListSeparator LS;
    for (auto [Arg, Name] : getArgAndNames()) {
      Result += LS;
      Result += Arg->getAsString();
      if (Name)
        Result += ":$" + Name->getAsUnquotedString();
    }
  }
  return Result + ")";
}

//===----------------------------------------------------------------------===//
//    Other implementations
//===----------------------------------------------------------------------===//

RecordVal::RecordVal(const Init *N, const RecTy *T, FieldKind K)
    : Name(N), TyAndKind(T, K) {
  setValue(UnsetInit::get(N->getRecordKeeper()));
  assert(Value && "Cannot create unset value for current type!");
}

// This constructor accepts the same arguments as the above, but also
// a source location.
RecordVal::RecordVal(const Init *N, SMLoc Loc, const RecTy *T, FieldKind K)
    : Name(N), Loc(Loc), TyAndKind(T, K) {
  setValue(UnsetInit::get(N->getRecordKeeper()));
  assert(Value && "Cannot create unset value for current type!");
}

StringRef RecordVal::getName() const {
  return cast<StringInit>(getNameInit())->getValue();
}

std::string RecordVal::getPrintType() const {
  if (getType() == StringRecTy::get(getRecordKeeper())) {
    if (const auto *StrInit = dyn_cast<StringInit>(Value)) {
      if (StrInit->hasCodeFormat())
        return "code";
      else
        return "string";
    } else {
      return "string";
    }
  } else {
    return TyAndKind.getPointer()->getAsString();
  }
}

bool RecordVal::setValue(const Init *V) {
  if (!V) {
    Value = nullptr;
    return false;
  }

  Value = V->getCastTo(getType());
  if (!Value)
    return true;

  assert(!isa<TypedInit>(Value) ||
         cast<TypedInit>(Value)->getType()->typeIsA(getType()));
  if (const auto *BTy = dyn_cast<BitsRecTy>(getType())) {
    if (isa<BitsInit>(Value))
      return false;
    SmallVector<const Init *, 64> Bits(BTy->getNumBits());
    for (unsigned I = 0, E = BTy->getNumBits(); I < E; ++I)
      Bits[I] = Value->getBit(I);
    Value = BitsInit::get(V->getRecordKeeper(), Bits);
  }

  return false;
}

// This version of setValue takes a source location and resets the
// location in the RecordVal.
bool RecordVal::setValue(const Init *V, SMLoc NewLoc) {
  Loc = NewLoc;
  return setValue(V);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void RecordVal::dump() const { errs() << *this; }
#endif

void RecordVal::print(raw_ostream &OS, bool PrintSem) const {
  if (isNonconcreteOK()) OS << "field ";
  OS << getPrintType() << " " << getNameInitAsString();

  if (getValue())
    OS << " = " << *getValue();

  if (PrintSem) OS << ";\n";
}

void Record::updateClassLoc(SMLoc Loc) {
  assert(Locs.size() == 1);
  ForwardDeclarationLocs.push_back(Locs.front());

  Locs.clear();
  Locs.push_back(Loc);
}

void Record::checkName() {
  // Ensure the record name has string type.
  const auto *TypedName = cast<const TypedInit>(Name);
  if (!isa<StringRecTy>(TypedName->getType()))
    PrintFatalError(getLoc(), Twine("Record name '") + Name->getAsString() +
                                  "' is not a string!");
}

const RecordRecTy *Record::getType() const {
  SmallVector<const Record *> DirectSCs(
      make_first_range(getDirectSuperClasses()));
  return RecordRecTy::get(TrackedRecords, DirectSCs);
}

DefInit *Record::getDefInit() const {
  if (!CorrespondingDefInit) {
    CorrespondingDefInit =
        new (TrackedRecords.getImpl().Allocator) DefInit(this);
  }
  return CorrespondingDefInit;
}

unsigned Record::getNewUID(RecordKeeper &RK) {
  return RK.getImpl().LastRecordID++;
}

void Record::setName(const Init *NewName) {
  Name = NewName;
  checkName();
  // DO NOT resolve record values to the name at this point because
  // there might be default values for arguments of this def. Those
  // arguments might not have been resolved yet so we don't want to
  // prematurely assume values for those arguments were not passed to
  // this def.
  //
  // Nonetheless, it may be that some of this Record's values
  // reference the record name. Indeed, the reason for having the
  // record name be an Init is to provide this flexibility. The extra
  // resolve steps after completely instantiating defs takes care of
  // this. See TGParser::ParseDef and TGParser::ParseDefm.
}

void Record::resolveReferences(Resolver &R, const RecordVal *SkipVal) {
  const Init *OldName = getNameInit();
  const Init *NewName = Name->resolveReferences(R);
  if (NewName != OldName) {
    // Re-register with RecordKeeper.
    setName(NewName);
  }

  // Resolve the field values.
  for (RecordVal &Value : Values) {
    if (SkipVal == &Value) // Skip resolve the same field as the given one
      continue;
    if (const Init *V = Value.getValue()) {
      const Init *VR = V->resolveReferences(R);
      if (Value.setValue(VR)) {
        std::string Type;
        if (const auto *VRT = dyn_cast<TypedInit>(VR))
          Type =
              (Twine("of type '") + VRT->getType()->getAsString() + "' ").str();
        PrintFatalError(
            getLoc(),
            Twine("Invalid value ") + Type + "found when setting field '" +
                Value.getNameInitAsString() + "' of type '" +
                Value.getType()->getAsString() +
                "' after resolving references: " + VR->getAsUnquotedString() +
                "\n");
      }
    }
  }

  // Resolve the assertion expressions.
  for (AssertionInfo &Assertion : Assertions) {
    const Init *Value = Assertion.Condition->resolveReferences(R);
    Assertion.Condition = Value;
    Value = Assertion.Message->resolveReferences(R);
    Assertion.Message = Value;
  }
  // Resolve the dump expressions.
  for (DumpInfo &Dump : Dumps) {
    const Init *Value = Dump.Message->resolveReferences(R);
    Dump.Message = Value;
  }
}

void Record::resolveReferences(const Init *NewName) {
  RecordResolver R(*this);
  R.setName(NewName);
  R.setFinal(true);
  resolveReferences(R);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void Record::dump() const { errs() << *this; }
#endif

raw_ostream &llvm::operator<<(raw_ostream &OS, const Record &R) {
  OS << R.getNameInitAsString();

  ArrayRef<const Init *> TArgs = R.getTemplateArgs();
  if (!TArgs.empty()) {
    OS << "<";
    ListSeparator LS;
    for (const Init *TA : TArgs) {
      const RecordVal *RV = R.getValue(TA);
      assert(RV && "Template argument record not found??");
      OS << LS;
      RV->print(OS, false);
    }
    OS << ">";
  }

  OS << " {";
  std::vector<const Record *> SCs = R.getSuperClasses();
  if (!SCs.empty()) {
    OS << "\t//";
    for (const Record *SC : SCs)
      OS << " " << SC->getNameInitAsString();
  }
  OS << "\n";

  for (const RecordVal &Val : R.getValues())
    if (Val.isNonconcreteOK() && !R.isTemplateArg(Val.getNameInit()))
      OS << Val;
  for (const RecordVal &Val : R.getValues())
    if (!Val.isNonconcreteOK() && !R.isTemplateArg(Val.getNameInit()))
      OS << Val;

  return OS << "}\n";
}

SMLoc Record::getFieldLoc(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");
  return R->getLoc();
}

const Init *Record::getValueInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");
  return R->getValue();
}

StringRef Record::getValueAsString(StringRef FieldName) const {
  const Init *I = getValueInit(FieldName);
  if (const auto *SI = dyn_cast<StringInit>(I))
    return SI->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" + FieldName +
                                "' exists but does not have a string value");
}

std::optional<StringRef>
Record::getValueAsOptionalString(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    return std::nullopt;
  if (isa<UnsetInit>(R->getValue()))
    return std::nullopt;

  if (const auto *SI = dyn_cast<StringInit>(R->getValue()))
    return SI->getValue();

  PrintFatalError(getLoc(),
                  "Record `" + getName() + "', ` field `" + FieldName +
                      "' exists but does not have a string initializer!");
}

const BitsInit *Record::getValueAsBitsInit(StringRef FieldName) const {
  const Init *I = getValueInit(FieldName);
  if (const auto *BI = dyn_cast<BitsInit>(I))
    return BI;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" + FieldName +
                                "' exists but does not have a bits value");
}

const ListInit *Record::getValueAsListInit(StringRef FieldName) const {
  const Init *I = getValueInit(FieldName);
  if (const auto *LI = dyn_cast<ListInit>(I))
    return LI;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" + FieldName +
                                "' exists but does not have a list value");
}

std::vector<const Record *>
Record::getValueAsListOfDefs(StringRef FieldName) const {
  const ListInit *List = getValueAsListInit(FieldName);
  std::vector<const Record *> Defs;
  for (const Init *I : List->getElements()) {
    if (const auto *DI = dyn_cast<DefInit>(I))
      Defs.push_back(DI->getDef());
    else
      PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
                                    FieldName +
                                    "' list is not entirely DefInit!");
  }
  return Defs;
}

int64_t Record::getValueAsInt(StringRef FieldName) const {
  const Init *I = getValueInit(FieldName);
  if (const auto *II = dyn_cast<IntInit>(I))
    return II->getValue();
  PrintFatalError(
      getLoc(),
      Twine("Record `") + getName() + "', field `" + FieldName +
          "' exists but does not have an int value: " + I->getAsString());
}

std::vector<int64_t>
Record::getValueAsListOfInts(StringRef FieldName) const {
  const ListInit *List = getValueAsListInit(FieldName);
  std::vector<int64_t> Ints;
  for (const Init *I : List->getElements()) {
    if (const auto *II = dyn_cast<IntInit>(I))
      Ints.push_back(II->getValue());
    else
      PrintFatalError(getLoc(),
                      Twine("Record `") + getName() + "', field `" + FieldName +
                          "' exists but does not have a list of ints value: " +
                          I->getAsString());
  }
  return Ints;
}

std::vector<StringRef>
Record::getValueAsListOfStrings(StringRef FieldName) const {
  const ListInit *List = getValueAsListInit(FieldName);
  std::vector<StringRef> Strings;
  for (const Init *I : List->getElements()) {
    if (const auto *SI = dyn_cast<StringInit>(I))
      Strings.push_back(SI->getValue());
    else
      PrintFatalError(getLoc(),
                      Twine("Record `") + getName() + "', field `" + FieldName +
                          "' exists but does not have a list of strings value: " +
                          I->getAsString());
  }
  return Strings;
}

const Record *Record::getValueAsDef(StringRef FieldName) const {
  const Init *I = getValueInit(FieldName);
  if (const auto *DI = dyn_cast<DefInit>(I))
    return DI->getDef();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a def initializer!");
}

const Record *Record::getValueAsOptionalDef(StringRef FieldName) const {
  const Init *I = getValueInit(FieldName);
  if (const auto *DI = dyn_cast<DefInit>(I))
    return DI->getDef();
  if (isa<UnsetInit>(I))
    return nullptr;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have either a def initializer or '?'!");
}

bool Record::getValueAsBit(StringRef FieldName) const {
  const Init *I = getValueInit(FieldName);
  if (const auto *BI = dyn_cast<BitInit>(I))
    return BI->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a bit initializer!");
}

bool Record::getValueAsBitOrUnset(StringRef FieldName, bool &Unset) const {
  const Init *I = getValueInit(FieldName);
  if (isa<UnsetInit>(I)) {
    Unset = true;
    return false;
  }
  Unset = false;
  if (const auto *BI = dyn_cast<BitInit>(I))
    return BI->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a bit initializer!");
}

const DagInit *Record::getValueAsDag(StringRef FieldName) const {
  const Init *I = getValueInit(FieldName);
  if (const auto *DI = dyn_cast<DagInit>(I))
    return DI;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a dag initializer!");
}

// Check all record assertions: For each one, resolve the condition
// and message, then call CheckAssert().
// Note: The condition and message are probably already resolved,
//       but resolving again allows calls before records are resolved.
void Record::checkRecordAssertions() {
  RecordResolver R(*this);
  R.setFinal(true);

  bool AnyFailed = false;
  for (const auto &Assertion : getAssertions()) {
    const Init *Condition = Assertion.Condition->resolveReferences(R);
    const Init *Message = Assertion.Message->resolveReferences(R);
    AnyFailed |= CheckAssert(Assertion.Loc, Condition, Message);
  }

  if (!AnyFailed)
    return;

  // If any of the record assertions failed, print some context that will
  // help see where the record that caused these assert failures is defined.
  PrintError(this, "assertion failed in this record");
}

void Record::emitRecordDumps() {
  RecordResolver R(*this);
  R.setFinal(true);

  for (const DumpInfo &Dump : getDumps()) {
    const Init *Message = Dump.Message->resolveReferences(R);
    dumpMessage(Dump.Loc, Message);
  }
}

// Report a warning if the record has unused template arguments.
void Record::checkUnusedTemplateArgs() {
  for (const Init *TA : getTemplateArgs()) {
    const RecordVal *Arg = getValue(TA);
    if (!Arg->isUsed())
      PrintWarning(Arg->getLoc(),
                   "unused template argument: " + Twine(Arg->getName()));
  }
}

RecordKeeper::RecordKeeper()
    : Impl(std::make_unique<detail::RecordKeeperImpl>(*this)),
      Timer(std::make_unique<TGTimer>()) {}

RecordKeeper::~RecordKeeper() = default;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void RecordKeeper::dump() const { errs() << *this; }
#endif

raw_ostream &llvm::operator<<(raw_ostream &OS, const RecordKeeper &RK) {
  OS << "------------- Classes -----------------\n";
  for (const auto &[_, C] : RK.getClasses())
    OS << "class " << *C;

  OS << "------------- Defs -----------------\n";
  for (const auto &[_, D] : RK.getDefs())
    OS << "def " << *D;
  return OS;
}

/// GetNewAnonymousName - Generate a unique anonymous name that can be used as
/// an identifier.
const Init *RecordKeeper::getNewAnonymousName() {
  return AnonymousNameInit::get(*this, getImpl().AnonCounter++);
}

ArrayRef<const Record *>
RecordKeeper::getAllDerivedDefinitions(StringRef ClassName) const {
  // We cache the record vectors for single classes. Many backends request
  // the same vectors multiple times.
  auto [Iter, Inserted] = Cache.try_emplace(ClassName.str());
  if (Inserted)
    Iter->second = getAllDerivedDefinitions(ArrayRef(ClassName));
  return Iter->second;
}

std::vector<const Record *>
RecordKeeper::getAllDerivedDefinitions(ArrayRef<StringRef> ClassNames) const {
  SmallVector<const Record *, 2> ClassRecs;
  std::vector<const Record *> Defs;

  assert(ClassNames.size() > 0 && "At least one class must be passed.");
  for (StringRef ClassName : ClassNames) {
    const Record *Class = getClass(ClassName);
    if (!Class)
      PrintFatalError("The class '" + ClassName + "' is not defined\n");
    ClassRecs.push_back(Class);
  }

  for (const auto &OneDef : getDefs()) {
    if (all_of(ClassRecs, [&OneDef](const Record *Class) {
          return OneDef.second->isSubClassOf(Class);
        }))
      Defs.push_back(OneDef.second.get());
  }
  llvm::sort(Defs, LessRecord());
  return Defs;
}

ArrayRef<const Record *>
RecordKeeper::getAllDerivedDefinitionsIfDefined(StringRef ClassName) const {
  if (getClass(ClassName))
    return getAllDerivedDefinitions(ClassName);
  return Cache[""];
}

void RecordKeeper::dumpAllocationStats(raw_ostream &OS) const {
  Impl->dumpAllocationStats(OS);
}

const Init *MapResolver::resolve(const Init *VarName) {
  auto It = Map.find(VarName);
  if (It == Map.end())
    return nullptr;

  const Init *I = It->second.V;

  if (!It->second.Resolved && Map.size() > 1) {
    // Resolve mutual references among the mapped variables, but prevent
    // infinite recursion.
    Map.erase(It);
    I = I->resolveReferences(*this);
    Map[VarName] = {I, true};
  }

  return I;
}

const Init *RecordResolver::resolve(const Init *VarName) {
  const Init *Val = Cache.lookup(VarName);
  if (Val)
    return Val;

  if (llvm::is_contained(Stack, VarName))
    return nullptr; // prevent infinite recursion

  if (const RecordVal *RV = getCurrentRecord()->getValue(VarName)) {
    if (!isa<UnsetInit>(RV->getValue())) {
      Val = RV->getValue();
      Stack.push_back(VarName);
      Val = Val->resolveReferences(*this);
      Stack.pop_back();
    }
  } else if (Name && VarName == getCurrentRecord()->getNameInit()) {
    Stack.push_back(VarName);
    Val = Name->resolveReferences(*this);
    Stack.pop_back();
  }

  Cache[VarName] = Val;
  return Val;
}

const Init *TrackUnresolvedResolver::resolve(const Init *VarName) {
  const Init *I = nullptr;

  if (R) {
    I = R->resolve(VarName);
    if (I && !FoundUnresolved) {
      // Do not recurse into the resolved initializer, as that would change
      // the behavior of the resolver we're delegating, but do check to see
      // if there are unresolved variables remaining.
      TrackUnresolvedResolver Sub;
      I->resolveReferences(Sub);
      FoundUnresolved |= Sub.FoundUnresolved;
    }
  }

  if (!I)
    FoundUnresolved = true;
  return I;
}

const Init *HasReferenceResolver::resolve(const Init *VarName) {
  if (VarName == VarNameToTrack)
    Found = true;
  return nullptr;
}
