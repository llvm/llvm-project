//===- Record.cpp - Record implementation ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the tablegen record classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/Error.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//    std::string wrapper for DenseMap purposes
//===----------------------------------------------------------------------===//

namespace llvm {

/// TableGenStringKey - This is a wrapper for std::string suitable for
/// using as a key to a DenseMap.  Because there isn't a particularly
/// good way to indicate tombstone or empty keys for strings, we want
/// to wrap std::string to indicate that this is a "special" string
/// not expected to take on certain values (those of the tombstone and
/// empty keys).  This makes things a little safer as it clarifies
/// that DenseMap is really not appropriate for general strings.

class TableGenStringKey {
public:
  TableGenStringKey(const std::string &str) : data(str) {}
  TableGenStringKey(const char *str) : data(str) {}

  const std::string &str() const { return data; }

  friend hash_code hash_value(const TableGenStringKey &Value) {
    using llvm::hash_value;
    return hash_value(Value.str());
  }
private:
  std::string data;
};

/// Specialize DenseMapInfo for TableGenStringKey.
template<> struct DenseMapInfo<TableGenStringKey> {
  static inline TableGenStringKey getEmptyKey() {
    TableGenStringKey Empty("<<<EMPTY KEY>>>");
    return Empty;
  }
  static inline TableGenStringKey getTombstoneKey() {
    TableGenStringKey Tombstone("<<<TOMBSTONE KEY>>>");
    return Tombstone;
  }
  static unsigned getHashValue(const TableGenStringKey& Val) {
    using llvm::hash_value;
    return hash_value(Val);
  }
  static bool isEqual(const TableGenStringKey& LHS,
                      const TableGenStringKey& RHS) {
    return LHS.str() == RHS.str();
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
//    Type implementations
//===----------------------------------------------------------------------===//

BitRecTy BitRecTy::Shared;
IntRecTy IntRecTy::Shared;
StringRecTy StringRecTy::Shared;
DagRecTy DagRecTy::Shared;

void RecTy::anchor() { }
void RecTy::dump() const { print(errs()); }

ListRecTy *RecTy::getListTy() {
  if (!ListTy)
    ListTy = new ListRecTy(this);
  return ListTy;
}

Init *BitRecTy::convertValue(BitsInit *BI) {
  if (BI->getNumBits() != 1) return 0; // Only accept if just one bit!
  return BI->getBit(0);
}

bool BitRecTy::baseClassOf(const BitsRecTy *RHS) const {
  return RHS->getNumBits() == 1;
}

Init *BitRecTy::convertValue(IntInit *II) {
  int64_t Val = II->getValue();
  if (Val != 0 && Val != 1) return 0;  // Only accept 0 or 1 for a bit!

  return BitInit::get(Val != 0);
}

Init *BitRecTy::convertValue(TypedInit *VI) {
  RecTy *Ty = VI->getType();
  if (isa<BitRecTy>(Ty) || isa<BitsRecTy>(Ty) || isa<IntRecTy>(Ty))
    return VI;  // Accept variable if it is already of bit type!
  return 0;
}

BitsRecTy *BitsRecTy::get(unsigned Sz) {
  static std::vector<BitsRecTy*> Shared;
  if (Sz >= Shared.size())
    Shared.resize(Sz + 1);
  BitsRecTy *&Ty = Shared[Sz];
  if (!Ty)
    Ty = new BitsRecTy(Sz);
  return Ty;
}

std::string BitsRecTy::getAsString() const {
  return "bits<" + utostr(Size) + ">";
}

Init *BitsRecTy::convertValue(UnsetInit *UI) {
  SmallVector<Init *, 16> NewBits(Size);

  for (unsigned i = 0; i != Size; ++i)
    NewBits[i] = UnsetInit::get();

  return BitsInit::get(NewBits);
}

Init *BitsRecTy::convertValue(BitInit *UI) {
  if (Size != 1) return 0;  // Can only convert single bit.
          return BitsInit::get(UI);
}

/// canFitInBitfield - Return true if the number of bits is large enough to hold
/// the integer value.
static bool canFitInBitfield(int64_t Value, unsigned NumBits) {
  // For example, with NumBits == 4, we permit Values from [-7 .. 15].
  return (NumBits >= sizeof(Value) * 8) ||
         (Value >> NumBits == 0) || (Value >> (NumBits-1) == -1);
}

/// convertValue from Int initializer to bits type: Split the integer up into the
/// appropriate bits.
///
Init *BitsRecTy::convertValue(IntInit *II) {
  int64_t Value = II->getValue();
  // Make sure this bitfield is large enough to hold the integer value.
  if (!canFitInBitfield(Value, Size))
    return 0;

  SmallVector<Init *, 16> NewBits(Size);

  for (unsigned i = 0; i != Size; ++i)
    NewBits[i] = BitInit::get(Value & (1LL << i));

  return BitsInit::get(NewBits);
}

Init *BitsRecTy::convertValue(BitsInit *BI) {
  // If the number of bits is right, return it.  Otherwise we need to expand or
  // truncate.
  if (BI->getNumBits() == Size) return BI;
  return 0;
}

Init *BitsRecTy::convertValue(TypedInit *VI) {
  if (Size == 1 && isa<BitRecTy>(VI->getType()))
    return BitsInit::get(VI);

  if (VI->getType()->typeIsConvertibleTo(this)) {
    SmallVector<Init *, 16> NewBits(Size);

    for (unsigned i = 0; i != Size; ++i)
      NewBits[i] = VarBitInit::get(VI, i);
    return BitsInit::get(NewBits);
  }

  return 0;
}

Init *IntRecTy::convertValue(BitInit *BI) {
  return IntInit::get(BI->getValue());
}

Init *IntRecTy::convertValue(BitsInit *BI) {
  int64_t Result = 0;
  for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i)
    if (BitInit *Bit = dyn_cast<BitInit>(BI->getBit(i))) {
      Result |= Bit->getValue() << i;
    } else {
      return 0;
    }
  return IntInit::get(Result);
}

Init *IntRecTy::convertValue(TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;  // Accept variable if already of the right type!
  return 0;
}

Init *StringRecTy::convertValue(UnOpInit *BO) {
  if (BO->getOpcode() == UnOpInit::CAST) {
    Init *L = BO->getOperand()->convertInitializerTo(this);
    if (L == 0) return 0;
    if (L != BO->getOperand())
      return UnOpInit::get(UnOpInit::CAST, L, new StringRecTy);
    return BO;
  }

  return convertValue((TypedInit*)BO);
}

Init *StringRecTy::convertValue(BinOpInit *BO) {
  if (BO->getOpcode() == BinOpInit::STRCONCAT) {
    Init *L = BO->getLHS()->convertInitializerTo(this);
    Init *R = BO->getRHS()->convertInitializerTo(this);
    if (L == 0 || R == 0) return 0;
    if (L != BO->getLHS() || R != BO->getRHS())
      return BinOpInit::get(BinOpInit::STRCONCAT, L, R, new StringRecTy);
    return BO;
  }

  return convertValue((TypedInit*)BO);
}


Init *StringRecTy::convertValue(TypedInit *TI) {
  if (isa<StringRecTy>(TI->getType()))
    return TI;  // Accept variable if already of the right type!
  return 0;
}

std::string ListRecTy::getAsString() const {
  return "list<" + Ty->getAsString() + ">";
}

Init *ListRecTy::convertValue(ListInit *LI) {
  std::vector<Init*> Elements;

  // Verify that all of the elements of the list are subclasses of the
  // appropriate class!
  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    if (Init *CI = LI->getElement(i)->convertInitializerTo(Ty))
      Elements.push_back(CI);
    else
      return 0;

  if (!isa<ListRecTy>(LI->getType()))
    return 0;

  return ListInit::get(Elements, this);
}

Init *ListRecTy::convertValue(TypedInit *TI) {
  // Ensure that TI is compatible with our class.
  if (ListRecTy *LRT = dyn_cast<ListRecTy>(TI->getType()))
    if (LRT->getElementType()->typeIsConvertibleTo(getElementType()))
      return TI;
  return 0;
}

Init *DagRecTy::convertValue(TypedInit *TI) {
  if (TI->getType()->typeIsConvertibleTo(this))
    return TI;
  return 0;
}

Init *DagRecTy::convertValue(UnOpInit *BO) {
  if (BO->getOpcode() == UnOpInit::CAST) {
    Init *L = BO->getOperand()->convertInitializerTo(this);
    if (L == 0) return 0;
    if (L != BO->getOperand())
      return UnOpInit::get(UnOpInit::CAST, L, new DagRecTy);
    return BO;
  }
  return 0;
}

Init *DagRecTy::convertValue(BinOpInit *BO) {
  if (BO->getOpcode() == BinOpInit::CONCAT) {
    Init *L = BO->getLHS()->convertInitializerTo(this);
    Init *R = BO->getRHS()->convertInitializerTo(this);
    if (L == 0 || R == 0) return 0;
    if (L != BO->getLHS() || R != BO->getRHS())
      return BinOpInit::get(BinOpInit::CONCAT, L, R, new DagRecTy);
    return BO;
  }
  return 0;
}

RecordRecTy *RecordRecTy::get(Record *R) {
  return dyn_cast<RecordRecTy>(R->getDefInit()->getType());
}

std::string RecordRecTy::getAsString() const {
  return Rec->getName();
}

Init *RecordRecTy::convertValue(DefInit *DI) {
  // Ensure that DI is a subclass of Rec.
  if (!DI->getDef()->isSubClassOf(Rec))
    return 0;
  return DI;
}

Init *RecordRecTy::convertValue(TypedInit *TI) {
  // Ensure that TI is compatible with Rec.
  if (RecordRecTy *RRT = dyn_cast<RecordRecTy>(TI->getType()))
    if (RRT->getRecord()->isSubClassOf(getRecord()) ||
        RRT->getRecord() == getRecord())
      return TI;
  return 0;
}

bool RecordRecTy::baseClassOf(const RecordRecTy *RHS) const {
  if (Rec == RHS->getRecord() || RHS->getRecord()->isSubClassOf(Rec))
    return true;

  const std::vector<Record*> &SC = Rec->getSuperClasses();
  for (unsigned i = 0, e = SC.size(); i != e; ++i)
    if (RHS->getRecord()->isSubClassOf(SC[i]))
      return true;

  return false;
}

/// resolveTypes - Find a common type that T1 and T2 convert to.
/// Return 0 if no such type exists.
///
RecTy *llvm::resolveTypes(RecTy *T1, RecTy *T2) {
  if (T1->typeIsConvertibleTo(T2))
    return T2;
  if (T2->typeIsConvertibleTo(T1))
    return T1;

  // If one is a Record type, check superclasses
  if (RecordRecTy *RecTy1 = dyn_cast<RecordRecTy>(T1)) {
    // See if T2 inherits from a type T1 also inherits from
    const std::vector<Record *> &T1SuperClasses =
      RecTy1->getRecord()->getSuperClasses();
    for(std::vector<Record *>::const_iterator i = T1SuperClasses.begin(),
          iend = T1SuperClasses.end();
        i != iend;
        ++i) {
      RecordRecTy *SuperRecTy1 = RecordRecTy::get(*i);
      RecTy *NewType1 = resolveTypes(SuperRecTy1, T2);
      if (NewType1 != 0) {
        if (NewType1 != SuperRecTy1) {
          delete SuperRecTy1;
        }
        return NewType1;
      }
    }
  }
  if (RecordRecTy *RecTy2 = dyn_cast<RecordRecTy>(T2)) {
    // See if T1 inherits from a type T2 also inherits from
    const std::vector<Record *> &T2SuperClasses =
      RecTy2->getRecord()->getSuperClasses();
    for (std::vector<Record *>::const_iterator i = T2SuperClasses.begin(),
          iend = T2SuperClasses.end();
        i != iend;
        ++i) {
      RecordRecTy *SuperRecTy2 = RecordRecTy::get(*i);
      RecTy *NewType2 = resolveTypes(T1, SuperRecTy2);
      if (NewType2 != 0) {
        if (NewType2 != SuperRecTy2) {
          delete SuperRecTy2;
        }
        return NewType2;
      }
    }
  }
  return 0;
}


//===----------------------------------------------------------------------===//
//    Initializer implementations
//===----------------------------------------------------------------------===//

void Init::anchor() { }
void Init::dump() const { return print(errs()); }

void UnsetInit::anchor() { }

UnsetInit *UnsetInit::get() {
  static UnsetInit TheInit;
  return &TheInit;
}

void BitInit::anchor() { }

BitInit *BitInit::get(bool V) {
  static BitInit True(true);
  static BitInit False(false);

  return V ? &True : &False;
}

static void
ProfileBitsInit(FoldingSetNodeID &ID, ArrayRef<Init *> Range) {
  ID.AddInteger(Range.size());

  for (ArrayRef<Init *>::iterator i = Range.begin(),
         iend = Range.end();
       i != iend;
       ++i)
    ID.AddPointer(*i);
}

BitsInit *BitsInit::get(ArrayRef<Init *> Range) {
  typedef FoldingSet<BitsInit> Pool;
  static Pool ThePool;  

  FoldingSetNodeID ID;
  ProfileBitsInit(ID, Range);

  void *IP = 0;
  if (BitsInit *I = ThePool.FindNodeOrInsertPos(ID, IP))
    return I;

  BitsInit *I = new BitsInit(Range);
  ThePool.InsertNode(I, IP);

  return I;
}

void BitsInit::Profile(FoldingSetNodeID &ID) const {
  ProfileBitsInit(ID, Bits);
}

Init *
BitsInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
  SmallVector<Init *, 16> NewBits(Bits.size());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= getNumBits())
      return 0;
    NewBits[i] = getBit(Bits[i]);
  }
  return BitsInit::get(NewBits);
}

std::string BitsInit::getAsString() const {
  std::string Result = "{ ";
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    if (i) Result += ", ";
    if (Init *Bit = getBit(e-i-1))
      Result += Bit->getAsString();
    else
      Result += "*";
  }
  return Result + " }";
}

// Fix bit initializer to preserve the behavior that bit reference from a unset
// bits initializer will resolve into VarBitInit to keep the field name and bit
// number used in targets with fixed insn length.
static Init *fixBitInit(const RecordVal *RV, Init *Before, Init *After) {
  if (RV || After != UnsetInit::get())
    return After;
  return Before;
}

// resolveReferences - If there are any field references that refer to fields
// that have been filled in, we can propagate the values now.
//
Init *BitsInit::resolveReferences(Record &R, const RecordVal *RV) const {
  bool Changed = false;
  SmallVector<Init *, 16> NewBits(getNumBits());

  Init *CachedInit = 0;
  Init *CachedBitVar = 0;
  bool CachedBitVarChanged = false;

  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    Init *CurBit = Bits[i];
    Init *CurBitVar = CurBit->getBitVar();

    NewBits[i] = CurBit;

    if (CurBitVar == CachedBitVar) {
      if (CachedBitVarChanged) {
        Init *Bit = CachedInit->getBit(CurBit->getBitNum());
        NewBits[i] = fixBitInit(RV, CurBit, Bit);
      }
      continue;
    }
    CachedBitVar = CurBitVar;
    CachedBitVarChanged = false;

    Init *B;
    do {
      B = CurBitVar;
      CurBitVar = CurBitVar->resolveReferences(R, RV);
      CachedBitVarChanged |= B != CurBitVar;
      Changed |= B != CurBitVar;
    } while (B != CurBitVar);
    CachedInit = CurBitVar;

    if (CachedBitVarChanged) {
      Init *Bit = CurBitVar->getBit(CurBit->getBitNum());
      NewBits[i] = fixBitInit(RV, CurBit, Bit);
    }
  }

  if (Changed)
    return BitsInit::get(NewBits);

  return const_cast<BitsInit *>(this);
}

IntInit *IntInit::get(int64_t V) {
  typedef DenseMap<int64_t, IntInit *> Pool;
  static Pool ThePool;

  IntInit *&I = ThePool[V];
  if (!I) I = new IntInit(V);
  return I;
}

std::string IntInit::getAsString() const {
  return itostr(Value);
}

Init *
IntInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
  SmallVector<Init *, 16> NewBits(Bits.size());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= 64)
      return 0;

    NewBits[i] = BitInit::get(Value & (INT64_C(1) << Bits[i]));
  }
  return BitsInit::get(NewBits);
}

void StringInit::anchor() { }

StringInit *StringInit::get(StringRef V) {
  typedef StringMap<StringInit *> Pool;
  static Pool ThePool;

  StringInit *&I = ThePool[V];
  if (!I) I = new StringInit(V);
  return I;
}

static void ProfileListInit(FoldingSetNodeID &ID,
                            ArrayRef<Init *> Range,
                            RecTy *EltTy) {
  ID.AddInteger(Range.size());
  ID.AddPointer(EltTy);

  for (ArrayRef<Init *>::iterator i = Range.begin(),
         iend = Range.end();
       i != iend;
       ++i)
    ID.AddPointer(*i);
}

ListInit *ListInit::get(ArrayRef<Init *> Range, RecTy *EltTy) {
  typedef FoldingSet<ListInit> Pool;
  static Pool ThePool;

  // Just use the FoldingSetNodeID to compute a hash.  Use a DenseMap
  // for actual storage.
  FoldingSetNodeID ID;
  ProfileListInit(ID, Range, EltTy);

  void *IP = 0;
  if (ListInit *I = ThePool.FindNodeOrInsertPos(ID, IP))
    return I;

  ListInit *I = new ListInit(Range, EltTy);
  ThePool.InsertNode(I, IP);
  return I;
}

void ListInit::Profile(FoldingSetNodeID &ID) const {
  ListRecTy *ListType = dyn_cast<ListRecTy>(getType());
  assert(ListType && "Bad type for ListInit!");
  RecTy *EltTy = ListType->getElementType();

  ProfileListInit(ID, Values, EltTy);
}

Init *
ListInit::convertInitListSlice(const std::vector<unsigned> &Elements) const {
  std::vector<Init*> Vals;
  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    if (Elements[i] >= getSize())
      return 0;
    Vals.push_back(getElement(Elements[i]));
  }
  return ListInit::get(Vals, getType());
}

Record *ListInit::getElementAsRecord(unsigned i) const {
  assert(i < Values.size() && "List element index out of range!");
  DefInit *DI = dyn_cast<DefInit>(Values[i]);
  if (DI == 0)
    PrintFatalError("Expected record in list!");
  return DI->getDef();
}

Init *ListInit::resolveReferences(Record &R, const RecordVal *RV) const {
  std::vector<Init*> Resolved;
  Resolved.reserve(getSize());
  bool Changed = false;

  for (unsigned i = 0, e = getSize(); i != e; ++i) {
    Init *E;
    Init *CurElt = getElement(i);

    do {
      E = CurElt;
      CurElt = CurElt->resolveReferences(R, RV);
      Changed |= E != CurElt;
    } while (E != CurElt);
    Resolved.push_back(E);
  }

  if (Changed)
    return ListInit::get(Resolved, getType());
  return const_cast<ListInit *>(this);
}

Init *ListInit::resolveListElementReference(Record &R, const RecordVal *IRV,
                                            unsigned Elt) const {
  if (Elt >= getSize())
    return 0;  // Out of range reference.
  Init *E = getElement(Elt);
  // If the element is set to some value, or if we are resolving a reference
  // to a specific variable and that variable is explicitly unset, then
  // replace the VarListElementInit with it.
  if (IRV || !isa<UnsetInit>(E))
    return E;
  return 0;
}

std::string ListInit::getAsString() const {
  std::string Result = "[";
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (i) Result += ", ";
    Result += Values[i]->getAsString();
  }
  return Result + "]";
}

Init *OpInit::resolveListElementReference(Record &R, const RecordVal *IRV,
                                          unsigned Elt) const {
  Init *Resolved = resolveReferences(R, IRV);
  OpInit *OResolved = dyn_cast<OpInit>(Resolved);
  if (OResolved) {
    Resolved = OResolved->Fold(&R, 0);
  }

  if (Resolved != this) {
    TypedInit *Typed = dyn_cast<TypedInit>(Resolved);
    assert(Typed && "Expected typed init for list reference");
    if (Typed) {
      Init *New = Typed->resolveListElementReference(R, IRV, Elt);
      if (New)
        return New;
      return VarListElementInit::get(Typed, Elt);
    }
  }

  return 0;
}

Init *OpInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get())
    return const_cast<OpInit*>(this);
  return VarBitInit::get(const_cast<OpInit*>(this), Bit);
}

UnOpInit *UnOpInit::get(UnaryOp opc, Init *lhs, RecTy *Type) {
  typedef std::pair<std::pair<unsigned, Init *>, RecTy *> Key;

  typedef DenseMap<Key, UnOpInit *> Pool;
  static Pool ThePool;  

  Key TheKey(std::make_pair(std::make_pair(opc, lhs), Type));

  UnOpInit *&I = ThePool[TheKey];
  if (!I) I = new UnOpInit(opc, lhs, Type);
  return I;
}

Init *UnOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) const {
  switch (getOpcode()) {
  case CAST: {
    if (getType()->getAsString() == "string") {
      if (StringInit *LHSs = dyn_cast<StringInit>(LHS))
        return LHSs;

      if (DefInit *LHSd = dyn_cast<DefInit>(LHS))
        return StringInit::get(LHSd->getDef()->getName());

      if (IntInit *LHSi = dyn_cast<IntInit>(LHS))
        return StringInit::get(LHSi->getAsString());
    } else {
      if (StringInit *LHSs = dyn_cast<StringInit>(LHS)) {
        std::string Name = LHSs->getValue();

        // From TGParser::ParseIDValue
        if (CurRec) {
          if (const RecordVal *RV = CurRec->getValue(Name)) {
            if (RV->getType() != getType())
              PrintFatalError("type mismatch in cast");
            return VarInit::get(Name, RV->getType());
          }

          Init *TemplateArgName = QualifyName(*CurRec, CurMultiClass, Name,
                                              ":");
      
          if (CurRec->isTemplateArg(TemplateArgName)) {
            const RecordVal *RV = CurRec->getValue(TemplateArgName);
            assert(RV && "Template arg doesn't exist??");

            if (RV->getType() != getType())
              PrintFatalError("type mismatch in cast");

            return VarInit::get(TemplateArgName, RV->getType());
          }
        }

        if (CurMultiClass) {
          Init *MCName = QualifyName(CurMultiClass->Rec, CurMultiClass, Name, "::");

          if (CurMultiClass->Rec.isTemplateArg(MCName)) {
            const RecordVal *RV = CurMultiClass->Rec.getValue(MCName);
            assert(RV && "Template arg doesn't exist??");

            if (RV->getType() != getType())
              PrintFatalError("type mismatch in cast");

            return VarInit::get(MCName, RV->getType());
          }
        }

        if (Record *D = (CurRec->getRecords()).getDef(Name))
          return DefInit::get(D);

        PrintFatalError(CurRec->getLoc(),
                        "Undefined reference:'" + Name + "'\n");
      }
    }
    break;
  }
  case HEAD: {
    if (ListInit *LHSl = dyn_cast<ListInit>(LHS)) {
      if (LHSl->getSize() == 0) {
        assert(0 && "Empty list in car");
        return 0;
      }
      return LHSl->getElement(0);
    }
    break;
  }
  case TAIL: {
    if (ListInit *LHSl = dyn_cast<ListInit>(LHS)) {
      if (LHSl->getSize() == 0) {
        assert(0 && "Empty list in cdr");
        return 0;
      }
      // Note the +1.  We can't just pass the result of getValues()
      // directly.
      ArrayRef<Init *>::iterator begin = LHSl->getValues().begin()+1;
      ArrayRef<Init *>::iterator end   = LHSl->getValues().end();
      ListInit *Result =
        ListInit::get(ArrayRef<Init *>(begin, end - begin),
                      LHSl->getType());
      return Result;
    }
    break;
  }
  case EMPTY: {
    if (ListInit *LHSl = dyn_cast<ListInit>(LHS)) {
      if (LHSl->getSize() == 0) {
        return IntInit::get(1);
      } else {
        return IntInit::get(0);
      }
    }
    if (StringInit *LHSs = dyn_cast<StringInit>(LHS)) {
      if (LHSs->getValue().empty()) {
        return IntInit::get(1);
      } else {
        return IntInit::get(0);
      }
    }

    break;
  }
  }
  return const_cast<UnOpInit *>(this);
}

Init *UnOpInit::resolveReferences(Record &R, const RecordVal *RV) const {
  Init *lhs = LHS->resolveReferences(R, RV);

  if (LHS != lhs)
    return (UnOpInit::get(getOpcode(), lhs, getType()))->Fold(&R, 0);
  return Fold(&R, 0);
}

std::string UnOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case CAST: Result = "!cast<" + getType()->getAsString() + ">"; break;
  case HEAD: Result = "!head"; break;
  case TAIL: Result = "!tail"; break;
  case EMPTY: Result = "!empty"; break;
  }
  return Result + "(" + LHS->getAsString() + ")";
}

BinOpInit *BinOpInit::get(BinaryOp opc, Init *lhs,
                          Init *rhs, RecTy *Type) {
  typedef std::pair<
    std::pair<std::pair<unsigned, Init *>, Init *>,
    RecTy *
    > Key;

  typedef DenseMap<Key, BinOpInit *> Pool;
  static Pool ThePool;  

  Key TheKey(std::make_pair(std::make_pair(std::make_pair(opc, lhs), rhs),
                            Type));

  BinOpInit *&I = ThePool[TheKey];
  if (!I) I = new BinOpInit(opc, lhs, rhs, Type);
  return I;
}

Init *BinOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) const {
  switch (getOpcode()) {
  case CONCAT: {
    DagInit *LHSs = dyn_cast<DagInit>(LHS);
    DagInit *RHSs = dyn_cast<DagInit>(RHS);
    if (LHSs && RHSs) {
      DefInit *LOp = dyn_cast<DefInit>(LHSs->getOperator());
      DefInit *ROp = dyn_cast<DefInit>(RHSs->getOperator());
      if (LOp == 0 || ROp == 0 || LOp->getDef() != ROp->getDef())
        PrintFatalError("Concated Dag operators do not match!");
      std::vector<Init*> Args;
      std::vector<std::string> ArgNames;
      for (unsigned i = 0, e = LHSs->getNumArgs(); i != e; ++i) {
        Args.push_back(LHSs->getArg(i));
        ArgNames.push_back(LHSs->getArgName(i));
      }
      for (unsigned i = 0, e = RHSs->getNumArgs(); i != e; ++i) {
        Args.push_back(RHSs->getArg(i));
        ArgNames.push_back(RHSs->getArgName(i));
      }
      return DagInit::get(LHSs->getOperator(), "", Args, ArgNames);
    }
    break;
  }
  case STRCONCAT: {
    StringInit *LHSs = dyn_cast<StringInit>(LHS);
    StringInit *RHSs = dyn_cast<StringInit>(RHS);
    if (LHSs && RHSs)
      return StringInit::get(LHSs->getValue() + RHSs->getValue());
    break;
  }
  case EQ: {
    // try to fold eq comparison for 'bit' and 'int', otherwise fallback
    // to string objects.
    IntInit *L =
      dyn_cast_or_null<IntInit>(LHS->convertInitializerTo(IntRecTy::get()));
    IntInit *R =
      dyn_cast_or_null<IntInit>(RHS->convertInitializerTo(IntRecTy::get()));

    if (L && R)
      return IntInit::get(L->getValue() == R->getValue());

    StringInit *LHSs = dyn_cast<StringInit>(LHS);
    StringInit *RHSs = dyn_cast<StringInit>(RHS);

    // Make sure we've resolved
    if (LHSs && RHSs)
      return IntInit::get(LHSs->getValue() == RHSs->getValue());

    break;
  }
  case SHL:
  case SRA:
  case SRL: {
    IntInit *LHSi = dyn_cast<IntInit>(LHS);
    IntInit *RHSi = dyn_cast<IntInit>(RHS);
    if (LHSi && RHSi) {
      int64_t LHSv = LHSi->getValue(), RHSv = RHSi->getValue();
      int64_t Result;
      switch (getOpcode()) {
      default: llvm_unreachable("Bad opcode!");
      case SHL: Result = LHSv << RHSv; break;
      case SRA: Result = LHSv >> RHSv; break;
      case SRL: Result = (uint64_t)LHSv >> (uint64_t)RHSv; break;
      }
      return IntInit::get(Result);
    }
    break;
  }
  }
  return const_cast<BinOpInit *>(this);
}

Init *BinOpInit::resolveReferences(Record &R, const RecordVal *RV) const {
  Init *lhs = LHS->resolveReferences(R, RV);
  Init *rhs = RHS->resolveReferences(R, RV);

  if (LHS != lhs || RHS != rhs)
    return (BinOpInit::get(getOpcode(), lhs, rhs, getType()))->Fold(&R, 0);
  return Fold(&R, 0);
}

std::string BinOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case CONCAT: Result = "!con"; break;
  case SHL: Result = "!shl"; break;
  case SRA: Result = "!sra"; break;
  case SRL: Result = "!srl"; break;
  case EQ: Result = "!eq"; break;
  case STRCONCAT: Result = "!strconcat"; break;
  }
  return Result + "(" + LHS->getAsString() + ", " + RHS->getAsString() + ")";
}

TernOpInit *TernOpInit::get(TernaryOp opc, Init *lhs,
                                  Init *mhs, Init *rhs,
                                  RecTy *Type) {
  typedef std::pair<
    std::pair<
      std::pair<std::pair<unsigned, RecTy *>, Init *>,
      Init *
      >,
    Init *
    > Key;

  typedef DenseMap<Key, TernOpInit *> Pool;
  static Pool ThePool;

  Key TheKey(std::make_pair(std::make_pair(std::make_pair(std::make_pair(opc,
                                                                         Type),
                                                          lhs),
                                           mhs),
                            rhs));

  TernOpInit *&I = ThePool[TheKey];
  if (!I) I = new TernOpInit(opc, lhs, mhs, rhs, Type);
  return I;
}

static Init *ForeachHelper(Init *LHS, Init *MHS, Init *RHS, RecTy *Type,
                           Record *CurRec, MultiClass *CurMultiClass);

static Init *EvaluateOperation(OpInit *RHSo, Init *LHS, Init *Arg,
                               RecTy *Type, Record *CurRec,
                               MultiClass *CurMultiClass) {
  std::vector<Init *> NewOperands;

  TypedInit *TArg = dyn_cast<TypedInit>(Arg);

  // If this is a dag, recurse
  if (TArg && TArg->getType()->getAsString() == "dag") {
    Init *Result = ForeachHelper(LHS, Arg, RHSo, Type,
                                 CurRec, CurMultiClass);
    if (Result != 0) {
      return Result;
    } else {
      return 0;
    }
  }

  for (int i = 0; i < RHSo->getNumOperands(); ++i) {
    OpInit *RHSoo = dyn_cast<OpInit>(RHSo->getOperand(i));

    if (RHSoo) {
      Init *Result = EvaluateOperation(RHSoo, LHS, Arg,
                                       Type, CurRec, CurMultiClass);
      if (Result != 0) {
        NewOperands.push_back(Result);
      } else {
        NewOperands.push_back(Arg);
      }
    } else if (LHS->getAsString() == RHSo->getOperand(i)->getAsString()) {
      NewOperands.push_back(Arg);
    } else {
      NewOperands.push_back(RHSo->getOperand(i));
    }
  }

  // Now run the operator and use its result as the new leaf
  const OpInit *NewOp = RHSo->clone(NewOperands);
  Init *NewVal = NewOp->Fold(CurRec, CurMultiClass);
  if (NewVal != NewOp)
    return NewVal;

  return 0;
}

static Init *ForeachHelper(Init *LHS, Init *MHS, Init *RHS, RecTy *Type,
                           Record *CurRec, MultiClass *CurMultiClass) {
  DagInit *MHSd = dyn_cast<DagInit>(MHS);
  ListInit *MHSl = dyn_cast<ListInit>(MHS);

  OpInit *RHSo = dyn_cast<OpInit>(RHS);

  if (!RHSo) {
    PrintFatalError(CurRec->getLoc(), "!foreach requires an operator\n");
  }

  TypedInit *LHSt = dyn_cast<TypedInit>(LHS);

  if (!LHSt)
    PrintFatalError(CurRec->getLoc(), "!foreach requires typed variable\n");

  if ((MHSd && isa<DagRecTy>(Type)) || (MHSl && isa<ListRecTy>(Type))) {
    if (MHSd) {
      Init *Val = MHSd->getOperator();
      Init *Result = EvaluateOperation(RHSo, LHS, Val,
                                       Type, CurRec, CurMultiClass);
      if (Result != 0) {
        Val = Result;
      }

      std::vector<std::pair<Init *, std::string> > args;
      for (unsigned int i = 0; i < MHSd->getNumArgs(); ++i) {
        Init *Arg;
        std::string ArgName;
        Arg = MHSd->getArg(i);
        ArgName = MHSd->getArgName(i);

        // Process args
        Init *Result = EvaluateOperation(RHSo, LHS, Arg, Type,
                                         CurRec, CurMultiClass);
        if (Result != 0) {
          Arg = Result;
        }

        // TODO: Process arg names
        args.push_back(std::make_pair(Arg, ArgName));
      }

      return DagInit::get(Val, "", args);
    }
    if (MHSl) {
      std::vector<Init *> NewOperands;
      std::vector<Init *> NewList(MHSl->begin(), MHSl->end());

      for (std::vector<Init *>::iterator li = NewList.begin(),
             liend = NewList.end();
           li != liend;
           ++li) {
        Init *Item = *li;
        NewOperands.clear();
        for(int i = 0; i < RHSo->getNumOperands(); ++i) {
          // First, replace the foreach variable with the list item
          if (LHS->getAsString() == RHSo->getOperand(i)->getAsString()) {
            NewOperands.push_back(Item);
          } else {
            NewOperands.push_back(RHSo->getOperand(i));
          }
        }

        // Now run the operator and use its result as the new list item
        const OpInit *NewOp = RHSo->clone(NewOperands);
        Init *NewItem = NewOp->Fold(CurRec, CurMultiClass);
        if (NewItem != NewOp)
          *li = NewItem;
      }
      return ListInit::get(NewList, MHSl->getType());
    }
  }
  return 0;
}

Init *TernOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) const {
  switch (getOpcode()) {
  case SUBST: {
    DefInit *LHSd = dyn_cast<DefInit>(LHS);
    VarInit *LHSv = dyn_cast<VarInit>(LHS);
    StringInit *LHSs = dyn_cast<StringInit>(LHS);

    DefInit *MHSd = dyn_cast<DefInit>(MHS);
    VarInit *MHSv = dyn_cast<VarInit>(MHS);
    StringInit *MHSs = dyn_cast<StringInit>(MHS);

    DefInit *RHSd = dyn_cast<DefInit>(RHS);
    VarInit *RHSv = dyn_cast<VarInit>(RHS);
    StringInit *RHSs = dyn_cast<StringInit>(RHS);

    if ((LHSd && MHSd && RHSd)
        || (LHSv && MHSv && RHSv)
        || (LHSs && MHSs && RHSs)) {
      if (RHSd) {
        Record *Val = RHSd->getDef();
        if (LHSd->getAsString() == RHSd->getAsString()) {
          Val = MHSd->getDef();
        }
        return DefInit::get(Val);
      }
      if (RHSv) {
        std::string Val = RHSv->getName();
        if (LHSv->getAsString() == RHSv->getAsString()) {
          Val = MHSv->getName();
        }
        return VarInit::get(Val, getType());
      }
      if (RHSs) {
        std::string Val = RHSs->getValue();

        std::string::size_type found;
        std::string::size_type idx = 0;
        do {
          found = Val.find(LHSs->getValue(), idx);
          if (found != std::string::npos) {
            Val.replace(found, LHSs->getValue().size(), MHSs->getValue());
          }
          idx = found +  MHSs->getValue().size();
        } while (found != std::string::npos);

        return StringInit::get(Val);
      }
    }
    break;
  }

  case FOREACH: {
    Init *Result = ForeachHelper(LHS, MHS, RHS, getType(),
                                 CurRec, CurMultiClass);
    if (Result != 0) {
      return Result;
    }
    break;
  }

  case IF: {
    IntInit *LHSi = dyn_cast<IntInit>(LHS);
    if (Init *I = LHS->convertInitializerTo(IntRecTy::get()))
      LHSi = dyn_cast<IntInit>(I);
    if (LHSi) {
      if (LHSi->getValue()) {
        return MHS;
      } else {
        return RHS;
      }
    }
    break;
  }
  }

  return const_cast<TernOpInit *>(this);
}

Init *TernOpInit::resolveReferences(Record &R,
                                    const RecordVal *RV) const {
  Init *lhs = LHS->resolveReferences(R, RV);

  if (Opc == IF && lhs != LHS) {
    IntInit *Value = dyn_cast<IntInit>(lhs);
    if (Init *I = lhs->convertInitializerTo(IntRecTy::get()))
      Value = dyn_cast<IntInit>(I);
    if (Value != 0) {
      // Short-circuit
      if (Value->getValue()) {
        Init *mhs = MHS->resolveReferences(R, RV);
        return (TernOpInit::get(getOpcode(), lhs, mhs,
                                RHS, getType()))->Fold(&R, 0);
      } else {
        Init *rhs = RHS->resolveReferences(R, RV);
        return (TernOpInit::get(getOpcode(), lhs, MHS,
                                rhs, getType()))->Fold(&R, 0);
      }
    }
  }

  Init *mhs = MHS->resolveReferences(R, RV);
  Init *rhs = RHS->resolveReferences(R, RV);

  if (LHS != lhs || MHS != mhs || RHS != rhs)
    return (TernOpInit::get(getOpcode(), lhs, mhs, rhs,
                            getType()))->Fold(&R, 0);
  return Fold(&R, 0);
}

std::string TernOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case SUBST: Result = "!subst"; break;
  case FOREACH: Result = "!foreach"; break;
  case IF: Result = "!if"; break;
 }
  return Result + "(" + LHS->getAsString() + ", " + MHS->getAsString() + ", "
    + RHS->getAsString() + ")";
}

RecTy *TypedInit::getFieldType(const std::string &FieldName) const {
  if (RecordRecTy *RecordType = dyn_cast<RecordRecTy>(getType()))
    if (RecordVal *Field = RecordType->getRecord()->getValue(FieldName))
      return Field->getType();
  return 0;
}

Init *
TypedInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
  BitsRecTy *T = dyn_cast<BitsRecTy>(getType());
  if (T == 0) return 0;  // Cannot subscript a non-bits variable.
  unsigned NumBits = T->getNumBits();

  SmallVector<Init *, 16> NewBits(Bits.size());
  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= NumBits)
      return 0;

    NewBits[i] = VarBitInit::get(const_cast<TypedInit *>(this), Bits[i]);
  }
  return BitsInit::get(NewBits);
}

Init *
TypedInit::convertInitListSlice(const std::vector<unsigned> &Elements) const {
  ListRecTy *T = dyn_cast<ListRecTy>(getType());
  if (T == 0) return 0;  // Cannot subscript a non-list variable.

  if (Elements.size() == 1)
    return VarListElementInit::get(const_cast<TypedInit *>(this), Elements[0]);

  std::vector<Init*> ListInits;
  ListInits.reserve(Elements.size());
  for (unsigned i = 0, e = Elements.size(); i != e; ++i)
    ListInits.push_back(VarListElementInit::get(const_cast<TypedInit *>(this),
                                                Elements[i]));
  return ListInit::get(ListInits, T);
}


VarInit *VarInit::get(const std::string &VN, RecTy *T) {
  Init *Value = StringInit::get(VN);
  return VarInit::get(Value, T);
}

VarInit *VarInit::get(Init *VN, RecTy *T) {
  typedef std::pair<RecTy *, Init *> Key;
  typedef DenseMap<Key, VarInit *> Pool;
  static Pool ThePool;

  Key TheKey(std::make_pair(T, VN));

  VarInit *&I = ThePool[TheKey];
  if (!I) I = new VarInit(VN, T);
  return I;
}

const std::string &VarInit::getName() const {
  StringInit *NameString = dyn_cast<StringInit>(getNameInit());
  assert(NameString && "VarInit name is not a string!");
  return NameString->getValue();
}

Init *VarInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get())
    return const_cast<VarInit*>(this);
  return VarBitInit::get(const_cast<VarInit*>(this), Bit);
}

Init *VarInit::resolveListElementReference(Record &R,
                                           const RecordVal *IRV,
                                           unsigned Elt) const {
  if (R.isTemplateArg(getNameInit())) return 0;
  if (IRV && IRV->getNameInit() != getNameInit()) return 0;

  RecordVal *RV = R.getValue(getNameInit());
  assert(RV && "Reference to a non-existent variable?");
  ListInit *LI = dyn_cast<ListInit>(RV->getValue());
  if (!LI) {
    TypedInit *VI = dyn_cast<TypedInit>(RV->getValue());
    assert(VI && "Invalid list element!");
    return VarListElementInit::get(VI, Elt);
  }

  if (Elt >= LI->getSize())
    return 0;  // Out of range reference.
  Init *E = LI->getElement(Elt);
  // If the element is set to some value, or if we are resolving a reference
  // to a specific variable and that variable is explicitly unset, then
  // replace the VarListElementInit with it.
  if (IRV || !isa<UnsetInit>(E))
    return E;
  return 0;
}


RecTy *VarInit::getFieldType(const std::string &FieldName) const {
  if (RecordRecTy *RTy = dyn_cast<RecordRecTy>(getType()))
    if (const RecordVal *RV = RTy->getRecord()->getValue(FieldName))
      return RV->getType();
  return 0;
}

Init *VarInit::getFieldInit(Record &R, const RecordVal *RV,
                            const std::string &FieldName) const {
  if (isa<RecordRecTy>(getType()))
    if (const RecordVal *Val = R.getValue(VarName)) {
      if (RV != Val && (RV || isa<UnsetInit>(Val->getValue())))
        return 0;
      Init *TheInit = Val->getValue();
      assert(TheInit != this && "Infinite loop detected!");
      if (Init *I = TheInit->getFieldInit(R, RV, FieldName))
        return I;
      else
        return 0;
    }
  return 0;
}

/// resolveReferences - This method is used by classes that refer to other
/// variables which may not be defined at the time the expression is formed.
/// If a value is set for the variable later, this method will be called on
/// users of the value to allow the value to propagate out.
///
Init *VarInit::resolveReferences(Record &R, const RecordVal *RV) const {
  if (RecordVal *Val = R.getValue(VarName))
    if (RV == Val || (RV == 0 && !isa<UnsetInit>(Val->getValue())))
      return Val->getValue();
  return const_cast<VarInit *>(this);
}

VarBitInit *VarBitInit::get(TypedInit *T, unsigned B) {
  typedef std::pair<TypedInit *, unsigned> Key;
  typedef DenseMap<Key, VarBitInit *> Pool;

  static Pool ThePool;

  Key TheKey(std::make_pair(T, B));

  VarBitInit *&I = ThePool[TheKey];
  if (!I) I = new VarBitInit(T, B);
  return I;
}

std::string VarBitInit::getAsString() const {
   return TI->getAsString() + "{" + utostr(Bit) + "}";
}

Init *VarBitInit::resolveReferences(Record &R, const RecordVal *RV) const {
  Init *I = TI->resolveReferences(R, RV);
  if (TI != I)
    return I->getBit(getBitNum());

  return const_cast<VarBitInit*>(this);
}

VarListElementInit *VarListElementInit::get(TypedInit *T,
                                            unsigned E) {
  typedef std::pair<TypedInit *, unsigned> Key;
  typedef DenseMap<Key, VarListElementInit *> Pool;

  static Pool ThePool;

  Key TheKey(std::make_pair(T, E));

  VarListElementInit *&I = ThePool[TheKey];
  if (!I) I = new VarListElementInit(T, E);
  return I;
}

std::string VarListElementInit::getAsString() const {
  return TI->getAsString() + "[" + utostr(Element) + "]";
}

Init *
VarListElementInit::resolveReferences(Record &R, const RecordVal *RV) const {
  if (Init *I = getVariable()->resolveListElementReference(R, RV,
                                                           getElementNum()))
    return I;
  return const_cast<VarListElementInit *>(this);
}

Init *VarListElementInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get())
    return const_cast<VarListElementInit*>(this);
  return VarBitInit::get(const_cast<VarListElementInit*>(this), Bit);
}

Init *VarListElementInit:: resolveListElementReference(Record &R,
                                                       const RecordVal *RV,
                                                       unsigned Elt) const {
  Init *Result = TI->resolveListElementReference(R, RV, Element);
  
  if (Result) {
    if (TypedInit *TInit = dyn_cast<TypedInit>(Result)) {
      Init *Result2 = TInit->resolveListElementReference(R, RV, Elt);
      if (Result2) return Result2;
      return new VarListElementInit(TInit, Elt);
    }
    return Result;
  }
 
  return 0;
}

DefInit *DefInit::get(Record *R) {
  return R->getDefInit();
}

RecTy *DefInit::getFieldType(const std::string &FieldName) const {
  if (const RecordVal *RV = Def->getValue(FieldName))
    return RV->getType();
  return 0;
}

Init *DefInit::getFieldInit(Record &R, const RecordVal *RV,
                            const std::string &FieldName) const {
  return Def->getValue(FieldName)->getValue();
}


std::string DefInit::getAsString() const {
  return Def->getName();
}

FieldInit *FieldInit::get(Init *R, const std::string &FN) {
  typedef std::pair<Init *, TableGenStringKey> Key;
  typedef DenseMap<Key, FieldInit *> Pool;
  static Pool ThePool;  

  Key TheKey(std::make_pair(R, FN));

  FieldInit *&I = ThePool[TheKey];
  if (!I) I = new FieldInit(R, FN);
  return I;
}

Init *FieldInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get())
    return const_cast<FieldInit*>(this);
  return VarBitInit::get(const_cast<FieldInit*>(this), Bit);
}

Init *FieldInit::resolveListElementReference(Record &R, const RecordVal *RV,
                                             unsigned Elt) const {
  if (Init *ListVal = Rec->getFieldInit(R, RV, FieldName))
    if (ListInit *LI = dyn_cast<ListInit>(ListVal)) {
      if (Elt >= LI->getSize()) return 0;
      Init *E = LI->getElement(Elt);

      // If the element is set to some value, or if we are resolving a
      // reference to a specific variable and that variable is explicitly
      // unset, then replace the VarListElementInit with it.
      if (RV || !isa<UnsetInit>(E))
        return E;
    }
  return 0;
}

Init *FieldInit::resolveReferences(Record &R, const RecordVal *RV) const {
  Init *NewRec = RV ? Rec->resolveReferences(R, RV) : Rec;

  Init *BitsVal = NewRec->getFieldInit(R, RV, FieldName);
  if (BitsVal) {
    Init *BVR = BitsVal->resolveReferences(R, RV);
    return BVR->isComplete() ? BVR : const_cast<FieldInit *>(this);
  }

  if (NewRec != Rec) {
    return FieldInit::get(NewRec, FieldName);
  }
  return const_cast<FieldInit *>(this);
}

void ProfileDagInit(FoldingSetNodeID &ID,
                    Init *V,
                    const std::string &VN,
                    ArrayRef<Init *> ArgRange,
                    ArrayRef<std::string> NameRange) {
  ID.AddPointer(V);
  ID.AddString(VN);

  ArrayRef<Init *>::iterator Arg  = ArgRange.begin();
  ArrayRef<std::string>::iterator  Name = NameRange.begin();
  while (Arg != ArgRange.end()) {
    assert(Name != NameRange.end() && "Arg name underflow!");
    ID.AddPointer(*Arg++);
    ID.AddString(*Name++);
  }
  assert(Name == NameRange.end() && "Arg name overflow!");
}

DagInit *
DagInit::get(Init *V, const std::string &VN,
             ArrayRef<Init *> ArgRange,
             ArrayRef<std::string> NameRange) {
  typedef FoldingSet<DagInit> Pool;
  static Pool ThePool;  

  FoldingSetNodeID ID;
  ProfileDagInit(ID, V, VN, ArgRange, NameRange);

  void *IP = 0;
  if (DagInit *I = ThePool.FindNodeOrInsertPos(ID, IP))
    return I;

  DagInit *I = new DagInit(V, VN, ArgRange, NameRange);
  ThePool.InsertNode(I, IP);

  return I;
}

DagInit *
DagInit::get(Init *V, const std::string &VN,
             const std::vector<std::pair<Init*, std::string> > &args) {
  typedef std::pair<Init*, std::string> PairType;

  std::vector<Init *> Args;
  std::vector<std::string> Names;

  for (std::vector<PairType>::const_iterator i = args.begin(),
         iend = args.end();
       i != iend;
       ++i) {
    Args.push_back(i->first);
    Names.push_back(i->second);
  }

  return DagInit::get(V, VN, Args, Names);
}

void DagInit::Profile(FoldingSetNodeID &ID) const {
  ProfileDagInit(ID, Val, ValName, Args, ArgNames);
}

Init *DagInit::resolveReferences(Record &R, const RecordVal *RV) const {
  std::vector<Init*> NewArgs;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    NewArgs.push_back(Args[i]->resolveReferences(R, RV));

  Init *Op = Val->resolveReferences(R, RV);

  if (Args != NewArgs || Op != Val)
    return DagInit::get(Op, ValName, NewArgs, ArgNames);

  return const_cast<DagInit *>(this);
}


std::string DagInit::getAsString() const {
  std::string Result = "(" + Val->getAsString();
  if (!ValName.empty())
    Result += ":" + ValName;
  if (Args.size()) {
    Result += " " + Args[0]->getAsString();
    if (!ArgNames[0].empty()) Result += ":$" + ArgNames[0];
    for (unsigned i = 1, e = Args.size(); i != e; ++i) {
      Result += ", " + Args[i]->getAsString();
      if (!ArgNames[i].empty()) Result += ":$" + ArgNames[i];
    }
  }
  return Result + ")";
}


//===----------------------------------------------------------------------===//
//    Other implementations
//===----------------------------------------------------------------------===//

RecordVal::RecordVal(Init *N, RecTy *T, unsigned P)
  : Name(N), Ty(T), Prefix(P) {
  Value = Ty->convertValue(UnsetInit::get());
  assert(Value && "Cannot create unset value for current type!");
}

RecordVal::RecordVal(const std::string &N, RecTy *T, unsigned P)
  : Name(StringInit::get(N)), Ty(T), Prefix(P) {
  Value = Ty->convertValue(UnsetInit::get());
  assert(Value && "Cannot create unset value for current type!");
}

const std::string &RecordVal::getName() const {
  StringInit *NameString = dyn_cast<StringInit>(Name);
  assert(NameString && "RecordVal name is not a string!");
  return NameString->getValue();
}

void RecordVal::dump() const { errs() << *this; }

void RecordVal::print(raw_ostream &OS, bool PrintSem) const {
  if (getPrefix()) OS << "field ";
  OS << *getType() << " " << getNameInitAsString();

  if (getValue())
    OS << " = " << *getValue();

  if (PrintSem) OS << ";\n";
}

unsigned Record::LastID = 0;

void Record::init() {
  checkName();

  // Every record potentially has a def at the top.  This value is
  // replaced with the top-level def name at instantiation time.
  RecordVal DN("NAME", StringRecTy::get(), 0);
  addValue(DN);
}

void Record::checkName() {
  // Ensure the record name has string type.
  const TypedInit *TypedName = dyn_cast<const TypedInit>(Name);
  assert(TypedName && "Record name is not typed!");
  RecTy *Type = TypedName->getType();
  if (!isa<StringRecTy>(Type))
    PrintFatalError(getLoc(), "Record name is not a string!");
}

DefInit *Record::getDefInit() {
  if (!TheInit)
    TheInit = new DefInit(this, new RecordRecTy(this));
  return TheInit;
}

const std::string &Record::getName() const {
  const StringInit *NameString = dyn_cast<StringInit>(Name);
  assert(NameString && "Record name is not a string!");
  return NameString->getValue();
}

void Record::setName(Init *NewName) {
  if (TrackedRecords.getDef(Name->getAsUnquotedString()) == this) {
    TrackedRecords.removeDef(Name->getAsUnquotedString());
    TrackedRecords.addDef(this);
  } else if (TrackedRecords.getClass(Name->getAsUnquotedString()) == this) {
    TrackedRecords.removeClass(Name->getAsUnquotedString());
    TrackedRecords.addClass(this);
  }  // Otherwise this isn't yet registered.
  Name = NewName;
  checkName();
  // DO NOT resolve record values to the name at this point because
  // there might be default values for arguments of this def.  Those
  // arguments might not have been resolved yet so we don't want to
  // prematurely assume values for those arguments were not passed to
  // this def.
  //
  // Nonetheless, it may be that some of this Record's values
  // reference the record name.  Indeed, the reason for having the
  // record name be an Init is to provide this flexibility.  The extra
  // resolve steps after completely instantiating defs takes care of
  // this.  See TGParser::ParseDef and TGParser::ParseDefm.
}

void Record::setName(const std::string &Name) {
  setName(StringInit::get(Name));
}

/// resolveReferencesTo - If anything in this record refers to RV, replace the
/// reference to RV with the RHS of RV.  If RV is null, we resolve all possible
/// references.
void Record::resolveReferencesTo(const RecordVal *RV) {
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (RV == &Values[i]) // Skip resolve the same field as the given one
      continue;
    if (Init *V = Values[i].getValue())
      if (Values[i].setValue(V->resolveReferences(*this, RV)))
        PrintFatalError(getLoc(), "Invalid value is found when setting '"
                      + Values[i].getNameInitAsString()
                      + "' after resolving references"
                      + (RV ? " against '" + RV->getNameInitAsString()
                              + "' of ("
                              + RV->getValue()->getAsUnquotedString() + ")"
                            : "")
                      + "\n");
  }
  Init *OldName = getNameInit();
  Init *NewName = Name->resolveReferences(*this, RV);
  if (NewName != OldName) {
    // Re-register with RecordKeeper.
    setName(NewName);
  }
}

void Record::dump() const { errs() << *this; }

raw_ostream &llvm::operator<<(raw_ostream &OS, const Record &R) {
  OS << R.getNameInitAsString();

  const std::vector<Init *> &TArgs = R.getTemplateArgs();
  if (!TArgs.empty()) {
    OS << "<";
    for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
      if (i) OS << ", ";
      const RecordVal *RV = R.getValue(TArgs[i]);
      assert(RV && "Template argument record not found??");
      RV->print(OS, false);
    }
    OS << ">";
  }

  OS << " {";
  const std::vector<Record*> &SC = R.getSuperClasses();
  if (!SC.empty()) {
    OS << "\t//";
    for (unsigned i = 0, e = SC.size(); i != e; ++i)
      OS << " " << SC[i]->getNameInitAsString();
  }
  OS << "\n";

  const std::vector<RecordVal> &Vals = R.getValues();
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    if (Vals[i].getPrefix() && !R.isTemplateArg(Vals[i].getName()))
      OS << Vals[i];
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    if (!Vals[i].getPrefix() && !R.isTemplateArg(Vals[i].getName()))
      OS << Vals[i];

  return OS << "}\n";
}

/// getValueInit - Return the initializer for a value with the specified name,
/// or abort if the field does not exist.
///
Init *Record::getValueInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");
  return R->getValue();
}


/// getValueAsString - This method looks up the specified field and returns its
/// value as a string, aborts if the field does not exist or if
/// the value is not a string.
///
std::string Record::getValueAsString(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");

  if (StringInit *SI = dyn_cast<StringInit>(R->getValue()))
    return SI->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName.str() + "' does not have a string initializer!");
}

/// getValueAsBitsInit - This method looks up the specified field and returns
/// its value as a BitsInit, aborts if the field does not exist or if
/// the value is not the right type.
///
BitsInit *Record::getValueAsBitsInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");

  if (BitsInit *BI = dyn_cast<BitsInit>(R->getValue()))
    return BI;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName.str() + "' does not have a BitsInit initializer!");
}

/// getValueAsListInit - This method looks up the specified field and returns
/// its value as a ListInit, aborting if the field does not exist or if
/// the value is not the right type.
///
ListInit *Record::getValueAsListInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");

  if (ListInit *LI = dyn_cast<ListInit>(R->getValue()))
    return LI;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName.str() + "' does not have a list initializer!");
}

/// getValueAsListOfDefs - This method looks up the specified field and returns
/// its value as a vector of records, aborting if the field does not exist
/// or if the value is not the right type.
///
std::vector<Record*>
Record::getValueAsListOfDefs(StringRef FieldName) const {
  ListInit *List = getValueAsListInit(FieldName);
  std::vector<Record*> Defs;
  for (unsigned i = 0; i < List->getSize(); i++) {
    if (DefInit *DI = dyn_cast<DefInit>(List->getElement(i))) {
      Defs.push_back(DI->getDef());
    } else {
      PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
        FieldName.str() + "' list is not entirely DefInit!");
    }
  }
  return Defs;
}

/// getValueAsInt - This method looks up the specified field and returns its
/// value as an int64_t, aborting if the field does not exist or if the value
/// is not the right type.
///
int64_t Record::getValueAsInt(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");

  if (IntInit *II = dyn_cast<IntInit>(R->getValue()))
    return II->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName.str() + "' does not have an int initializer!");
}

/// getValueAsListOfInts - This method looks up the specified field and returns
/// its value as a vector of integers, aborting if the field does not exist or
/// if the value is not the right type.
///
std::vector<int64_t>
Record::getValueAsListOfInts(StringRef FieldName) const {
  ListInit *List = getValueAsListInit(FieldName);
  std::vector<int64_t> Ints;
  for (unsigned i = 0; i < List->getSize(); i++) {
    if (IntInit *II = dyn_cast<IntInit>(List->getElement(i))) {
      Ints.push_back(II->getValue());
    } else {
      PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
        FieldName.str() + "' does not have a list of ints initializer!");
    }
  }
  return Ints;
}

/// getValueAsListOfStrings - This method looks up the specified field and
/// returns its value as a vector of strings, aborting if the field does not
/// exist or if the value is not the right type.
///
std::vector<std::string>
Record::getValueAsListOfStrings(StringRef FieldName) const {
  ListInit *List = getValueAsListInit(FieldName);
  std::vector<std::string> Strings;
  for (unsigned i = 0; i < List->getSize(); i++) {
    if (StringInit *II = dyn_cast<StringInit>(List->getElement(i))) {
      Strings.push_back(II->getValue());
    } else {
      PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
        FieldName.str() + "' does not have a list of strings initializer!");
    }
  }
  return Strings;
}

/// getValueAsDef - This method looks up the specified field and returns its
/// value as a Record, aborting if the field does not exist or if the value
/// is not the right type.
///
Record *Record::getValueAsDef(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");

  if (DefInit *DI = dyn_cast<DefInit>(R->getValue()))
    return DI->getDef();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName.str() + "' does not have a def initializer!");
}

/// getValueAsBit - This method looks up the specified field and returns its
/// value as a bit, aborting if the field does not exist or if the value is
/// not the right type.
///
bool Record::getValueAsBit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");

  if (BitInit *BI = dyn_cast<BitInit>(R->getValue()))
    return BI->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName.str() + "' does not have a bit initializer!");
}

bool Record::getValueAsBitOrUnset(StringRef FieldName, bool &Unset) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");

  if (R->getValue() == UnsetInit::get()) {
    Unset = true;
    return false;
  }
  Unset = false;
  if (BitInit *BI = dyn_cast<BitInit>(R->getValue()))
    return BI->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName.str() + "' does not have a bit initializer!");
}

/// getValueAsDag - This method looks up the specified field and returns its
/// value as an Dag, aborting if the field does not exist or if the value is
/// not the right type.
///
DagInit *Record::getValueAsDag(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (R == 0 || R->getValue() == 0)
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");

  if (DagInit *DI = dyn_cast<DagInit>(R->getValue()))
    return DI;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName.str() + "' does not have a dag initializer!");
}


void MultiClass::dump() const {
  errs() << "Record:\n";
  Rec.dump();

  errs() << "Defs:\n";
  for (RecordVector::const_iterator r = DefPrototypes.begin(),
         rend = DefPrototypes.end();
       r != rend;
       ++r) {
    (*r)->dump();
  }
}


void RecordKeeper::dump() const { errs() << *this; }

raw_ostream &llvm::operator<<(raw_ostream &OS, const RecordKeeper &RK) {
  OS << "------------- Classes -----------------\n";
  const std::map<std::string, Record*> &Classes = RK.getClasses();
  for (std::map<std::string, Record*>::const_iterator I = Classes.begin(),
         E = Classes.end(); I != E; ++I)
    OS << "class " << *I->second;

  OS << "------------- Defs -----------------\n";
  const std::map<std::string, Record*> &Defs = RK.getDefs();
  for (std::map<std::string, Record*>::const_iterator I = Defs.begin(),
         E = Defs.end(); I != E; ++I)
    OS << "def " << *I->second;
  return OS;
}


/// getAllDerivedDefinitions - This method returns all concrete definitions
/// that derive from the specified class name.  If a class with the specified
/// name does not exist, an error is printed and true is returned.
std::vector<Record*>
RecordKeeper::getAllDerivedDefinitions(const std::string &ClassName) const {
  Record *Class = getClass(ClassName);
  if (!Class)
    PrintFatalError("ERROR: Couldn't find the `" + ClassName + "' class!\n");

  std::vector<Record*> Defs;
  for (std::map<std::string, Record*>::const_iterator I = getDefs().begin(),
         E = getDefs().end(); I != E; ++I)
    if (I->second->isSubClassOf(Class))
      Defs.push_back(I->second);

  return Defs;
}

/// QualifyName - Return an Init with a qualifier prefix referring
/// to CurRec's name.
Init *llvm::QualifyName(Record &CurRec, MultiClass *CurMultiClass,
                        Init *Name, const std::string &Scoper) {
  RecTy *Type = dyn_cast<TypedInit>(Name)->getType();

  BinOpInit *NewName =
    BinOpInit::get(BinOpInit::STRCONCAT, 
                      BinOpInit::get(BinOpInit::STRCONCAT,
                                        CurRec.getNameInit(),
                                        StringInit::get(Scoper),
                                        Type)->Fold(&CurRec, CurMultiClass),
                      Name,
                      Type);

  if (CurMultiClass && Scoper != "::") {
    NewName =
      BinOpInit::get(BinOpInit::STRCONCAT, 
                        BinOpInit::get(BinOpInit::STRCONCAT,
                                          CurMultiClass->Rec.getNameInit(),
                                          StringInit::get("::"),
                                          Type)->Fold(&CurRec, CurMultiClass),
                        NewName->Fold(&CurRec, CurMultiClass),
                        Type);
  }

  return NewName->Fold(&CurRec, CurMultiClass);
}

/// QualifyName - Return an Init with a qualifier prefix referring
/// to CurRec's name.
Init *llvm::QualifyName(Record &CurRec, MultiClass *CurMultiClass,
                        const std::string &Name,
                        const std::string &Scoper) {
  return QualifyName(CurRec, CurMultiClass, StringInit::get(Name), Scoper);
}
