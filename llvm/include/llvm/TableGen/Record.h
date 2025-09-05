//===- llvm/TableGen/Record.h - Classes for Table Records -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the main TableGen data structures, including the TableGen
// types, values, and high-level data structures.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_RECORD_H
#define LLVM_TABLEGEN_RECORD_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/TrailingObjects.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace llvm {
namespace detail {
struct RecordKeeperImpl;
} // namespace detail

class ListRecTy;
class Record;
class RecordKeeper;
class RecordVal;
class Resolver;
class StringInit;
class TypedInit;
class TGTimer;

//===----------------------------------------------------------------------===//
//  Type Classes
//===----------------------------------------------------------------------===//

class RecTy {
public:
  /// Subclass discriminator (for dyn_cast<> et al.)
  enum RecTyKind {
    BitRecTyKind,
    BitsRecTyKind,
    IntRecTyKind,
    StringRecTyKind,
    ListRecTyKind,
    DagRecTyKind,
    RecordRecTyKind
  };

private:
  RecTyKind Kind;
  /// The RecordKeeper that uniqued this Type.
  RecordKeeper &RK;
  /// ListRecTy of the list that has elements of this type. Its a cache that
  /// is populated on demand.
  mutable const ListRecTy *ListTy = nullptr;

public:
  RecTy(RecTyKind K, RecordKeeper &RK) : Kind(K), RK(RK) {}
  virtual ~RecTy() = default;

  RecTyKind getRecTyKind() const { return Kind; }

  /// Return the RecordKeeper that uniqued this Type.
  RecordKeeper &getRecordKeeper() const { return RK; }

  virtual std::string getAsString() const = 0;
  void print(raw_ostream &OS) const { OS << getAsString(); }
  void dump() const;

  /// Return true if all values of 'this' type can be converted to the specified
  /// type.
  virtual bool typeIsConvertibleTo(const RecTy *RHS) const;

  /// Return true if 'this' type is equal to or a subtype of RHS. For example,
  /// a bit set is not an int, but they are convertible.
  virtual bool typeIsA(const RecTy *RHS) const;

  /// Returns the type representing list<thistype>.
  const ListRecTy *getListTy() const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const RecTy &Ty) {
  Ty.print(OS);
  return OS;
}

/// 'bit' - Represent a single bit
class BitRecTy : public RecTy {
  friend detail::RecordKeeperImpl;

  BitRecTy(RecordKeeper &RK) : RecTy(BitRecTyKind, RK) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == BitRecTyKind;
  }

  static const BitRecTy *get(RecordKeeper &RK);

  std::string getAsString() const override { return "bit"; }

  bool typeIsConvertibleTo(const RecTy *RHS) const override;
};

/// 'bits<n>' - Represent a fixed number of bits
class BitsRecTy : public RecTy {
  unsigned Size;

  explicit BitsRecTy(RecordKeeper &RK, unsigned Sz)
      : RecTy(BitsRecTyKind, RK), Size(Sz) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == BitsRecTyKind;
  }

  static const BitsRecTy *get(RecordKeeper &RK, unsigned Sz);

  unsigned getNumBits() const { return Size; }

  std::string getAsString() const override;

  bool typeIsConvertibleTo(const RecTy *RHS) const override;
};

/// 'int' - Represent an integer value of no particular size
class IntRecTy : public RecTy {
  friend detail::RecordKeeperImpl;

  IntRecTy(RecordKeeper &RK) : RecTy(IntRecTyKind, RK) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == IntRecTyKind;
  }

  static const IntRecTy *get(RecordKeeper &RK);

  std::string getAsString() const override { return "int"; }

  bool typeIsConvertibleTo(const RecTy *RHS) const override;
};

/// 'string' - Represent an string value
class StringRecTy : public RecTy {
  friend detail::RecordKeeperImpl;

  StringRecTy(RecordKeeper &RK) : RecTy(StringRecTyKind, RK) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == StringRecTyKind;
  }

  static const StringRecTy *get(RecordKeeper &RK);

  std::string getAsString() const override;

  bool typeIsConvertibleTo(const RecTy *RHS) const override;
};

/// 'list<Ty>' - Represent a list of element values, all of which must be of
/// the specified type. The type is stored in ElementTy.
class ListRecTy : public RecTy {
  friend const ListRecTy *RecTy::getListTy() const;

  const RecTy *ElementTy;

  explicit ListRecTy(const RecTy *T)
      : RecTy(ListRecTyKind, T->getRecordKeeper()), ElementTy(T) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == ListRecTyKind;
  }

  static const ListRecTy *get(const RecTy *T) { return T->getListTy(); }
  const RecTy *getElementType() const { return ElementTy; }

  std::string getAsString() const override;

  bool typeIsConvertibleTo(const RecTy *RHS) const override;

  bool typeIsA(const RecTy *RHS) const override;
};

/// 'dag' - Represent a dag fragment
class DagRecTy : public RecTy {
  friend detail::RecordKeeperImpl;

  DagRecTy(RecordKeeper &RK) : RecTy(DagRecTyKind, RK) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == DagRecTyKind;
  }

  static const DagRecTy *get(RecordKeeper &RK);

  std::string getAsString() const override;
};

/// '[classname]' - Type of record values that have zero or more superclasses.
///
/// The list of superclasses is non-redundant, i.e. only contains classes that
/// are not the superclass of some other listed class.
class RecordRecTy final : public RecTy,
                          public FoldingSetNode,
                          private TrailingObjects<RecordRecTy, const Record *> {
  friend TrailingObjects;
  friend class Record;
  friend detail::RecordKeeperImpl;

  unsigned NumClasses;

  explicit RecordRecTy(RecordKeeper &RK, ArrayRef<const Record *> Classes);

public:
  RecordRecTy(const RecordRecTy &) = delete;
  RecordRecTy &operator=(const RecordRecTy &) = delete;

  // Do not use sized deallocation due to trailing objects.
  void operator delete(void *Ptr) { ::operator delete(Ptr); }

  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == RecordRecTyKind;
  }

  /// Get the record type with the given non-redundant list of superclasses.
  static const RecordRecTy *get(RecordKeeper &RK,
                                ArrayRef<const Record *> Classes);
  static const RecordRecTy *get(const Record *Class);

  void Profile(FoldingSetNodeID &ID) const;

  ArrayRef<const Record *> getClasses() const {
    return getTrailingObjects(NumClasses);
  }

  using const_record_iterator = const Record *const *;

  const_record_iterator classes_begin() const { return getClasses().begin(); }
  const_record_iterator classes_end() const { return getClasses().end(); }

  std::string getAsString() const override;

  bool isSubClassOf(const Record *Class) const;
  bool typeIsConvertibleTo(const RecTy *RHS) const override;

  bool typeIsA(const RecTy *RHS) const override;
};

/// Find a common type that T1 and T2 convert to.
/// Return 0 if no such type exists.
const RecTy *resolveTypes(const RecTy *T1, const RecTy *T2);

//===----------------------------------------------------------------------===//
//  Initializer Classes
//===----------------------------------------------------------------------===//

class Init {
protected:
  /// Discriminator enum (for isa<>, dyn_cast<>, et al.)
  ///
  /// This enum is laid out by a preorder traversal of the inheritance
  /// hierarchy, and does not contain an entry for abstract classes, as per
  /// the recommendation in docs/HowToSetUpLLVMStyleRTTI.rst.
  ///
  /// We also explicitly include "first" and "last" values for each
  /// interior node of the inheritance tree, to make it easier to read the
  /// corresponding classof().
  ///
  /// We could pack these a bit tighter by not having the IK_FirstXXXInit
  /// and IK_LastXXXInit be their own values, but that would degrade
  /// readability for really no benefit.
  enum InitKind : uint8_t {
    IK_First, // unused; silence a spurious warning
    IK_FirstTypedInit,
    IK_BitInit,
    IK_BitsInit,
    IK_DagInit,
    IK_DefInit,
    IK_FieldInit,
    IK_IntInit,
    IK_ListInit,
    IK_FirstOpInit,
    IK_BinOpInit,
    IK_TernOpInit,
    IK_UnOpInit,
    IK_LastOpInit,
    IK_CondOpInit,
    IK_FoldOpInit,
    IK_IsAOpInit,
    IK_ExistsOpInit,
    IK_InstancesOpInit,
    IK_AnonymousNameInit,
    IK_StringInit,
    IK_VarInit,
    IK_VarBitInit,
    IK_VarDefInit,
    IK_LastTypedInit,
    IK_UnsetInit,
    IK_ArgumentInit,
  };

private:
  const InitKind Kind;

protected:
  uint8_t Opc; // Used by UnOpInit, BinOpInit, and TernOpInit

private:
  virtual void anchor();

public:
  /// Get the kind (type) of the value.
  InitKind getKind() const { return Kind; }

  /// Get the record keeper that initialized this Init.
  RecordKeeper &getRecordKeeper() const;

protected:
  explicit Init(InitKind K, uint8_t Opc = 0) : Kind(K), Opc(Opc) {}

public:
  Init(const Init &) = delete;
  Init &operator=(const Init &) = delete;
  virtual ~Init() = default;

  /// Is this a complete value with no unset (uninitialized) subvalues?
  virtual bool isComplete() const { return true; }

  /// Is this a concrete and fully resolved value without any references or
  /// stuck operations? Unset values are concrete.
  virtual bool isConcrete() const { return false; }

  /// Print this value.
  void print(raw_ostream &OS) const { OS << getAsString(); }

  /// Convert this value to a literal form.
  virtual std::string getAsString() const = 0;

  /// Convert this value to a literal form,
  /// without adding quotes around a string.
  virtual std::string getAsUnquotedString() const { return getAsString(); }

  /// Debugging method that may be called through a debugger; just
  /// invokes print on stderr.
  void dump() const;

  /// If this value is convertible to type \p Ty, return a value whose
  /// type is \p Ty, generating a !cast operation if required.
  /// Otherwise, return null.
  virtual const Init *getCastTo(const RecTy *Ty) const = 0;

  /// Convert to a value whose type is \p Ty, or return null if this
  /// is not possible. This can happen if the value's type is convertible
  /// to \p Ty, but there are unresolved references.
  virtual const Init *convertInitializerTo(const RecTy *Ty) const = 0;

  /// This function is used to implement the bit range
  /// selection operator. Given a value, it selects the specified bits,
  /// returning them as a new \p Init of type \p bits. If it is not legal
  /// to use the bit selection operator on this value, null is returned.
  virtual const Init *
  convertInitializerBitRange(ArrayRef<unsigned> Bits) const {
    return nullptr;
  }

  /// This function is used to implement the FieldInit class.
  /// Implementors of this method should return the type of the named
  /// field if they are of type record.
  virtual const RecTy *getFieldType(const StringInit *FieldName) const {
    return nullptr;
  }

  /// This function is used by classes that refer to other
  /// variables which may not be defined at the time the expression is formed.
  /// If a value is set for the variable later, this method will be called on
  /// users of the value to allow the value to propagate out.
  virtual const Init *resolveReferences(Resolver &R) const { return this; }

  /// Get the \p Init value of the specified bit.
  virtual const Init *getBit(unsigned Bit) const = 0;
};

inline raw_ostream &operator<<(raw_ostream &OS, const Init &I) {
  I.print(OS); return OS;
}

/// This is the common superclass of types that have a specific,
/// explicit type, stored in ValueTy.
class TypedInit : public Init {
  const RecTy *ValueTy;

protected:
  explicit TypedInit(InitKind K, const RecTy *T, uint8_t Opc = 0)
      : Init(K, Opc), ValueTy(T) {}

public:
  TypedInit(const TypedInit &) = delete;
  TypedInit &operator=(const TypedInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() >= IK_FirstTypedInit &&
           I->getKind() <= IK_LastTypedInit;
  }

  /// Get the type of the Init as a RecTy.
  const RecTy *getType() const { return ValueTy; }

  /// Get the record keeper that initialized this Init.
  RecordKeeper &getRecordKeeper() const { return ValueTy->getRecordKeeper(); }

  const Init *getCastTo(const RecTy *Ty) const override;
  const Init *convertInitializerTo(const RecTy *Ty) const override;

  const Init *
  convertInitializerBitRange(ArrayRef<unsigned> Bits) const override;

  /// This method is used to implement the FieldInit class.
  /// Implementors of this method should return the type of the named field if
  /// they are of type record.
  const RecTy *getFieldType(const StringInit *FieldName) const override;
};

/// '?' - Represents an uninitialized value.
class UnsetInit final : public Init {
  friend detail::RecordKeeperImpl;

  /// The record keeper that initialized this Init.
  RecordKeeper &RK;

  UnsetInit(RecordKeeper &RK) : Init(IK_UnsetInit), RK(RK) {}

public:
  UnsetInit(const UnsetInit &) = delete;
  UnsetInit &operator=(const UnsetInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_UnsetInit;
  }

  /// Get the singleton unset Init.
  static UnsetInit *get(RecordKeeper &RK);

  /// Get the record keeper that initialized this Init.
  RecordKeeper &getRecordKeeper() const { return RK; }

  const Init *getCastTo(const RecTy *Ty) const override;
  const Init *convertInitializerTo(const RecTy *Ty) const override;

  const Init *getBit(unsigned Bit) const override { return this; }

  /// Is this a complete value with no unset (uninitialized) subvalues?
  bool isComplete() const override { return false; }

  bool isConcrete() const override { return true; }

  /// Get the string representation of the Init.
  std::string getAsString() const override { return "?"; }
};

// Represent an argument.
using ArgAuxType = std::variant<unsigned, const Init *>;
class ArgumentInit final : public Init, public FoldingSetNode {
public:
  enum Kind {
    Positional,
    Named,
  };

private:
  const Init *Value;
  ArgAuxType Aux;

protected:
  explicit ArgumentInit(const Init *Value, ArgAuxType Aux)
      : Init(IK_ArgumentInit), Value(Value), Aux(Aux) {}

public:
  ArgumentInit(const ArgumentInit &) = delete;
  ArgumentInit &operator=(const ArgumentInit &) = delete;

  static bool classof(const Init *I) { return I->getKind() == IK_ArgumentInit; }

  RecordKeeper &getRecordKeeper() const { return Value->getRecordKeeper(); }

  static const ArgumentInit *get(const Init *Value, ArgAuxType Aux);

  bool isPositional() const { return Aux.index() == Positional; }
  bool isNamed() const { return Aux.index() == Named; }

  const Init *getValue() const { return Value; }
  unsigned getIndex() const {
    assert(isPositional() && "Should be positional!");
    return std::get<Positional>(Aux);
  }
  const Init *getName() const {
    assert(isNamed() && "Should be named!");
    return std::get<Named>(Aux);
  }
  const ArgumentInit *cloneWithValue(const Init *Value) const {
    return get(Value, Aux);
  }

  void Profile(FoldingSetNodeID &ID) const;

  const Init *resolveReferences(Resolver &R) const override;
  std::string getAsString() const override {
    if (isPositional())
      return utostr(getIndex()) + ": " + Value->getAsString();
    if (isNamed())
      return getName()->getAsString() + ": " + Value->getAsString();
    llvm_unreachable("Unsupported argument type!");
    return "";
  }

  bool isComplete() const override { return false; }
  bool isConcrete() const override { return false; }
  const Init *getBit(unsigned Bit) const override { return Value->getBit(Bit); }
  const Init *getCastTo(const RecTy *Ty) const override {
    return Value->getCastTo(Ty);
  }
  const Init *convertInitializerTo(const RecTy *Ty) const override {
    return Value->convertInitializerTo(Ty);
  }
};

/// 'true'/'false' - Represent a concrete initializer for a bit.
class BitInit final : public TypedInit {
  friend detail::RecordKeeperImpl;

  bool Value;

  explicit BitInit(bool V, const RecTy *T)
      : TypedInit(IK_BitInit, T), Value(V) {}

public:
  BitInit(const BitInit &) = delete;
  BitInit &operator=(BitInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_BitInit;
  }

  static BitInit *get(RecordKeeper &RK, bool V);

  bool getValue() const { return Value; }

  const Init *convertInitializerTo(const RecTy *Ty) const override;

  const Init *getBit(unsigned Bit) const override {
    assert(Bit < 1 && "Bit index out of range!");
    return this;
  }

  bool isConcrete() const override { return true; }
  std::string getAsString() const override { return Value ? "1" : "0"; }
};

/// '{ a, b, c }' - Represents an initializer for a BitsRecTy value.
/// It contains a vector of bits, whose size is determined by the type.
class BitsInit final : public TypedInit,
                       public FoldingSetNode,
                       private TrailingObjects<BitsInit, const Init *> {
  friend TrailingObjects;
  unsigned NumBits;

  BitsInit(RecordKeeper &RK, ArrayRef<const Init *> Bits);

public:
  BitsInit(const BitsInit &) = delete;
  BitsInit &operator=(const BitsInit &) = delete;

  // Do not use sized deallocation due to trailing objects.
  void operator delete(void *Ptr) { ::operator delete(Ptr); }

  static bool classof(const Init *I) {
    return I->getKind() == IK_BitsInit;
  }

  static BitsInit *get(RecordKeeper &RK, ArrayRef<const Init *> Range);

  void Profile(FoldingSetNodeID &ID) const;

  unsigned getNumBits() const { return NumBits; }

  const Init *convertInitializerTo(const RecTy *Ty) const override;
  const Init *
  convertInitializerBitRange(ArrayRef<unsigned> Bits) const override;
  std::optional<int64_t> convertInitializerToInt() const;

  // Returns the set of known bits as a 64-bit integer.
  uint64_t convertKnownBitsToInt() const;

  bool isComplete() const override;
  bool allInComplete() const;
  bool isConcrete() const override;
  std::string getAsString() const override;

  const Init *resolveReferences(Resolver &R) const override;

  ArrayRef<const Init *> getBits() const { return getTrailingObjects(NumBits); }

  const Init *getBit(unsigned Bit) const override { return getBits()[Bit]; }
};

/// '7' - Represent an initialization by a literal integer value.
class IntInit final : public TypedInit {
  int64_t Value;

  explicit IntInit(RecordKeeper &RK, int64_t V)
      : TypedInit(IK_IntInit, IntRecTy::get(RK)), Value(V) {}

public:
  IntInit(const IntInit &) = delete;
  IntInit &operator=(const IntInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_IntInit;
  }

  static IntInit *get(RecordKeeper &RK, int64_t V);

  int64_t getValue() const { return Value; }

  const Init *convertInitializerTo(const RecTy *Ty) const override;
  const Init *
  convertInitializerBitRange(ArrayRef<unsigned> Bits) const override;

  bool isConcrete() const override { return true; }
  std::string getAsString() const override;

  const Init *getBit(unsigned Bit) const override {
    return BitInit::get(getRecordKeeper(), (Value & (1ULL << Bit)) != 0);
  }
};

/// "anonymous_n" - Represent an anonymous record name
class AnonymousNameInit final : public TypedInit {
  unsigned Value;

  explicit AnonymousNameInit(RecordKeeper &RK, unsigned V)
      : TypedInit(IK_AnonymousNameInit, StringRecTy::get(RK)), Value(V) {}

public:
  AnonymousNameInit(const AnonymousNameInit &) = delete;
  AnonymousNameInit &operator=(const AnonymousNameInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_AnonymousNameInit;
  }

  static AnonymousNameInit *get(RecordKeeper &RK, unsigned);

  unsigned getValue() const { return Value; }

  const StringInit *getNameInit() const;

  std::string getAsString() const override;

  const Init *resolveReferences(Resolver &R) const override;

  const Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off string");
  }
};

/// "foo" - Represent an initialization by a string value.
class StringInit final : public TypedInit {
public:
  enum StringFormat {
    SF_String, // Format as "text"
    SF_Code,   // Format as [{text}]
  };

private:
  StringRef Value;
  StringFormat Format;

  explicit StringInit(RecordKeeper &RK, StringRef V, StringFormat Fmt)
      : TypedInit(IK_StringInit, StringRecTy::get(RK)), Value(V), Format(Fmt) {}

public:
  StringInit(const StringInit &) = delete;
  StringInit &operator=(const StringInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_StringInit;
  }

  static const StringInit *get(RecordKeeper &RK, StringRef,
                               StringFormat Fmt = SF_String);

  static StringFormat determineFormat(StringFormat Fmt1, StringFormat Fmt2) {
    return (Fmt1 == SF_Code || Fmt2 == SF_Code) ? SF_Code : SF_String;
  }

  StringRef getValue() const { return Value; }
  StringFormat getFormat() const { return Format; }
  bool hasCodeFormat() const { return Format == SF_Code; }

  const Init *convertInitializerTo(const RecTy *Ty) const override;

  bool isConcrete() const override { return true; }

  std::string getAsString() const override {
    if (Format == SF_String)
      return "\"" + Value.str() + "\"";
    else
      return "[{" + Value.str() + "}]";
  }

  std::string getAsUnquotedString() const override { return Value.str(); }

  const Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off string");
  }
};

/// [AL, AH, CL] - Represent a list of defs
///
class ListInit final : public TypedInit,
                       public FoldingSetNode,
                       private TrailingObjects<ListInit, const Init *> {
  friend TrailingObjects;
  unsigned NumElements;

public:
  using const_iterator = const Init *const *;

private:
  explicit ListInit(ArrayRef<const Init *> Elements, const RecTy *EltTy);

public:
  ListInit(const ListInit &) = delete;
  ListInit &operator=(const ListInit &) = delete;

  // Do not use sized deallocation due to trailing objects.
  void operator delete(void *Ptr) { ::operator delete(Ptr); }

  static bool classof(const Init *I) {
    return I->getKind() == IK_ListInit;
  }
  static const ListInit *get(ArrayRef<const Init *> Range, const RecTy *EltTy);

  void Profile(FoldingSetNodeID &ID) const;

  ArrayRef<const Init *> getElements() const {
    return ArrayRef(getTrailingObjects(), NumElements);
  }

  LLVM_DEPRECATED("Use getElements instead", "getElements")
  ArrayRef<const Init *> getValues() const { return getElements(); }

  const Init *getElement(unsigned Idx) const { return getElements()[Idx]; }

  const RecTy *getElementType() const {
    return cast<ListRecTy>(getType())->getElementType();
  }

  const Record *getElementAsRecord(unsigned Idx) const;

  const Init *convertInitializerTo(const RecTy *Ty) const override;

  /// This method is used by classes that refer to other
  /// variables which may not be defined at the time they expression is formed.
  /// If a value is set for the variable later, this method will be called on
  /// users of the value to allow the value to propagate out.
  ///
  const Init *resolveReferences(Resolver &R) const override;

  bool isComplete() const override;
  bool isConcrete() const override;
  std::string getAsString() const override;

  const_iterator begin() const { return getElements().begin(); }
  const_iterator end() const { return getElements().end(); }

  size_t size() const { return NumElements; }
  bool empty() const { return NumElements == 0; }

  const Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off list");
  }
};

/// Base class for operators
///
class OpInit : public TypedInit {
protected:
  explicit OpInit(InitKind K, const RecTy *Type, uint8_t Opc)
      : TypedInit(K, Type, Opc) {}

public:
  OpInit(const OpInit &) = delete;
  OpInit &operator=(OpInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() >= IK_FirstOpInit &&
           I->getKind() <= IK_LastOpInit;
  }

  const Init *getBit(unsigned Bit) const final;
};

/// !op (X) - Transform an init.
///
class UnOpInit final : public OpInit, public FoldingSetNode {
public:
  enum UnaryOp : uint8_t {
    TOLOWER,
    TOUPPER,
    CAST,
    NOT,
    HEAD,
    TAIL,
    SIZE,
    EMPTY,
    GETDAGOP,
    GETDAGOPNAME,
    LOG2,
    REPR,
    LISTFLATTEN,
    INITIALIZED,
  };

private:
  const Init *LHS;

  UnOpInit(UnaryOp opc, const Init *lhs, const RecTy *Type)
      : OpInit(IK_UnOpInit, Type, opc), LHS(lhs) {}

public:
  UnOpInit(const UnOpInit &) = delete;
  UnOpInit &operator=(const UnOpInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_UnOpInit;
  }

  static const UnOpInit *get(UnaryOp opc, const Init *lhs, const RecTy *Type);

  void Profile(FoldingSetNodeID &ID) const;

  UnaryOp getOpcode() const { return (UnaryOp)Opc; }
  const Init *getOperand() const { return LHS; }

  // Fold - If possible, fold this to a simpler init. Return this if not
  // possible to fold.
  const Init *Fold(const Record *CurRec, bool IsFinal = false) const;

  const Init *resolveReferences(Resolver &R) const override;

  std::string getAsString() const override;
};

/// !op (X, Y) - Combine two inits.
class BinOpInit final : public OpInit, public FoldingSetNode {
public:
  enum BinaryOp : uint8_t {
    ADD,
    SUB,
    MUL,
    DIV,
    AND,
    OR,
    XOR,
    SHL,
    SRA,
    SRL,
    LISTCONCAT,
    LISTSPLAT,
    LISTREMOVE,
    LISTELEM,
    LISTSLICE,
    RANGEC,
    STRCONCAT,
    INTERLEAVE,
    CONCAT,
    MATCH,
    EQ,
    NE,
    LE,
    LT,
    GE,
    GT,
    GETDAGARG,
    GETDAGNAME,
    SETDAGOP,
    SETDAGOPNAME
  };

private:
  const Init *LHS, *RHS;

  BinOpInit(BinaryOp opc, const Init *lhs, const Init *rhs, const RecTy *Type)
      : OpInit(IK_BinOpInit, Type, opc), LHS(lhs), RHS(rhs) {}

public:
  BinOpInit(const BinOpInit &) = delete;
  BinOpInit &operator=(const BinOpInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_BinOpInit;
  }

  static const BinOpInit *get(BinaryOp opc, const Init *lhs, const Init *rhs,
                              const RecTy *Type);
  static const Init *getStrConcat(const Init *lhs, const Init *rhs);
  static const Init *getListConcat(const TypedInit *lhs, const Init *rhs);

  void Profile(FoldingSetNodeID &ID) const;

  BinaryOp getOpcode() const { return (BinaryOp)Opc; }
  const Init *getLHS() const { return LHS; }
  const Init *getRHS() const { return RHS; }

  std::optional<bool> CompareInit(unsigned Opc, const Init *LHS,
                                  const Init *RHS) const;

  // Fold - If possible, fold this to a simpler init. Return this if not
  // possible to fold.
  const Init *Fold(const Record *CurRec) const;

  const Init *resolveReferences(Resolver &R) const override;

  std::string getAsString() const override;
};

/// !op (X, Y, Z) - Combine two inits.
class TernOpInit final : public OpInit, public FoldingSetNode {
public:
  enum TernaryOp : uint8_t {
    SUBST,
    FOREACH,
    FILTER,
    IF,
    DAG,
    RANGE,
    SUBSTR,
    FIND,
    SETDAGARG,
    SETDAGNAME,
  };

private:
  const Init *LHS, *MHS, *RHS;

  TernOpInit(TernaryOp opc, const Init *lhs, const Init *mhs, const Init *rhs,
             const RecTy *Type)
      : OpInit(IK_TernOpInit, Type, opc), LHS(lhs), MHS(mhs), RHS(rhs) {}

public:
  TernOpInit(const TernOpInit &) = delete;
  TernOpInit &operator=(const TernOpInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_TernOpInit;
  }

  static const TernOpInit *get(TernaryOp opc, const Init *lhs, const Init *mhs,
                               const Init *rhs, const RecTy *Type);

  void Profile(FoldingSetNodeID &ID) const;

  TernaryOp getOpcode() const { return (TernaryOp)Opc; }
  const Init *getLHS() const { return LHS; }
  const Init *getMHS() const { return MHS; }
  const Init *getRHS() const { return RHS; }

  // Fold - If possible, fold this to a simpler init. Return this if not
  // possible to fold.
  const Init *Fold(const Record *CurRec) const;

  bool isComplete() const override {
    return LHS->isComplete() && MHS->isComplete() && RHS->isComplete();
  }

  const Init *resolveReferences(Resolver &R) const override;

  std::string getAsString() const override;
};

/// !cond(condition_1: value1, ... , condition_n: value)
/// Selects the first value for which condition is true.
/// Otherwise reports an error.
class CondOpInit final : public TypedInit,
                         public FoldingSetNode,
                         private TrailingObjects<CondOpInit, const Init *> {
  friend TrailingObjects;
  unsigned NumConds;
  const RecTy *ValType;

  CondOpInit(ArrayRef<const Init *> Conds, ArrayRef<const Init *> Values,
             const RecTy *Type);

public:
  CondOpInit(const CondOpInit &) = delete;
  CondOpInit &operator=(const CondOpInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_CondOpInit;
  }

  static const CondOpInit *get(ArrayRef<const Init *> Conds,
                               ArrayRef<const Init *> Values,
                               const RecTy *Type);

  void Profile(FoldingSetNodeID &ID) const;

  const RecTy *getValType() const { return ValType; }

  unsigned getNumConds() const { return NumConds; }

  const Init *getCond(unsigned Num) const { return getConds()[Num]; }

  const Init *getVal(unsigned Num) const { return getVals()[Num]; }

  ArrayRef<const Init *> getConds() const {
    return getTrailingObjects(NumConds);
  }

  ArrayRef<const Init *> getVals() const {
    return ArrayRef(getTrailingObjects() + NumConds, NumConds);
  }

  auto getCondAndVals() const { return zip_equal(getConds(), getVals()); }

  const Init *Fold(const Record *CurRec) const;

  const Init *resolveReferences(Resolver &R) const override;

  bool isConcrete() const override;
  bool isComplete() const override;
  std::string getAsString() const override;

  using const_case_iterator = SmallVectorImpl<const Init *>::const_iterator;
  using const_val_iterator = SmallVectorImpl<const Init *>::const_iterator;

  inline const_case_iterator  arg_begin() const { return getConds().begin(); }
  inline const_case_iterator  arg_end  () const { return getConds().end(); }

  inline size_t              case_size () const { return NumConds; }
  inline bool                case_empty() const { return NumConds == 0; }

  inline const_val_iterator name_begin() const { return getVals().begin();}
  inline const_val_iterator name_end  () const { return getVals().end(); }

  inline size_t              val_size () const { return NumConds; }
  inline bool                val_empty() const { return NumConds == 0; }

  const Init *getBit(unsigned Bit) const override;
};

/// !foldl (a, b, expr, start, lst) - Fold over a list.
class FoldOpInit final : public TypedInit, public FoldingSetNode {
private:
  const Init *Start, *List, *A, *B, *Expr;

  FoldOpInit(const Init *Start, const Init *List, const Init *A, const Init *B,
             const Init *Expr, const RecTy *Type)
      : TypedInit(IK_FoldOpInit, Type), Start(Start), List(List), A(A), B(B),
        Expr(Expr) {}

public:
  FoldOpInit(const FoldOpInit &) = delete;
  FoldOpInit &operator=(const FoldOpInit &) = delete;

  static bool classof(const Init *I) { return I->getKind() == IK_FoldOpInit; }

  static const FoldOpInit *get(const Init *Start, const Init *List,
                               const Init *A, const Init *B, const Init *Expr,
                               const RecTy *Type);

  void Profile(FoldingSetNodeID &ID) const;

  // Fold - If possible, fold this to a simpler init. Return this if not
  // possible to fold.
  const Init *Fold(const Record *CurRec) const;

  bool isComplete() const override { return false; }

  const Init *resolveReferences(Resolver &R) const override;

  const Init *getBit(unsigned Bit) const override;

  std::string getAsString() const override;
};

/// !isa<type>(expr) - Dynamically determine the type of an expression.
class IsAOpInit final : public TypedInit, public FoldingSetNode {
private:
  const RecTy *CheckType;
  const Init *Expr;

  IsAOpInit(const RecTy *CheckType, const Init *Expr)
      : TypedInit(IK_IsAOpInit, IntRecTy::get(CheckType->getRecordKeeper())),
        CheckType(CheckType), Expr(Expr) {}

public:
  IsAOpInit(const IsAOpInit &) = delete;
  IsAOpInit &operator=(const IsAOpInit &) = delete;

  static bool classof(const Init *I) { return I->getKind() == IK_IsAOpInit; }

  static const IsAOpInit *get(const RecTy *CheckType, const Init *Expr);

  void Profile(FoldingSetNodeID &ID) const;

  // Fold - If possible, fold this to a simpler init. Return this if not
  // possible to fold.
  const Init *Fold() const;

  bool isComplete() const override { return false; }

  const Init *resolveReferences(Resolver &R) const override;

  const Init *getBit(unsigned Bit) const override;

  std::string getAsString() const override;
};

/// !exists<type>(expr) - Dynamically determine if a record of `type` named
/// `expr` exists.
class ExistsOpInit final : public TypedInit, public FoldingSetNode {
private:
  const RecTy *CheckType;
  const Init *Expr;

  ExistsOpInit(const RecTy *CheckType, const Init *Expr)
      : TypedInit(IK_ExistsOpInit, IntRecTy::get(CheckType->getRecordKeeper())),
        CheckType(CheckType), Expr(Expr) {}

public:
  ExistsOpInit(const ExistsOpInit &) = delete;
  ExistsOpInit &operator=(const ExistsOpInit &) = delete;

  static bool classof(const Init *I) { return I->getKind() == IK_ExistsOpInit; }

  static const ExistsOpInit *get(const RecTy *CheckType, const Init *Expr);

  void Profile(FoldingSetNodeID &ID) const;

  // Fold - If possible, fold this to a simpler init. Return this if not
  // possible to fold.
  const Init *Fold(const Record *CurRec, bool IsFinal = false) const;

  bool isComplete() const override { return false; }

  const Init *resolveReferences(Resolver &R) const override;

  const Init *getBit(unsigned Bit) const override;

  std::string getAsString() const override;
};

/// !instances<type>([regex]) - Produces a list of records whose type is `type`.
/// If `regex` is provided, only records whose name matches the regular
/// expression `regex` will be included.
class InstancesOpInit final : public TypedInit, public FoldingSetNode {
private:
  const RecTy *Type;
  const Init *Regex;

  InstancesOpInit(const RecTy *Type, const Init *Regex)
      : TypedInit(IK_InstancesOpInit, ListRecTy::get(Type)), Type(Type),
        Regex(Regex) {}

public:
  InstancesOpInit(const InstancesOpInit &) = delete;
  InstancesOpInit &operator=(const InstancesOpInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_InstancesOpInit;
  }

  static const InstancesOpInit *get(const RecTy *Type, const Init *Regex);

  void Profile(FoldingSetNodeID &ID) const;

  const Init *Fold(const Record *CurRec, bool IsFinal = false) const;

  bool isComplete() const override { return false; }

  const Init *resolveReferences(Resolver &R) const override;

  const Init *getBit(unsigned Bit) const override;

  std::string getAsString() const override;
};

/// 'Opcode' - Represent a reference to an entire variable object.
class VarInit final : public TypedInit {
  const Init *VarName;

  explicit VarInit(const Init *VN, const RecTy *T)
      : TypedInit(IK_VarInit, T), VarName(VN) {}

public:
  VarInit(const VarInit &) = delete;
  VarInit &operator=(const VarInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_VarInit;
  }

  static const VarInit *get(StringRef VN, const RecTy *T);
  static const VarInit *get(const Init *VN, const RecTy *T);

  StringRef getName() const;
  const Init *getNameInit() const { return VarName; }

  std::string getNameInitAsString() const {
    return getNameInit()->getAsUnquotedString();
  }

  /// This method is used by classes that refer to other
  /// variables which may not be defined at the time they expression is formed.
  /// If a value is set for the variable later, this method will be called on
  /// users of the value to allow the value to propagate out.
  ///
  const Init *resolveReferences(Resolver &R) const override;

  const Init *getBit(unsigned Bit) const override;

  std::string getAsString() const override { return std::string(getName()); }
};

/// Opcode{0} - Represent access to one bit of a variable or field.
class VarBitInit final : public TypedInit {
  const TypedInit *TI;
  unsigned Bit;

  VarBitInit(const TypedInit *T, unsigned B)
      : TypedInit(IK_VarBitInit, BitRecTy::get(T->getRecordKeeper())), TI(T),
        Bit(B) {
    assert(T->getType() &&
           (isa<IntRecTy>(T->getType()) ||
            (isa<BitsRecTy>(T->getType()) &&
             cast<BitsRecTy>(T->getType())->getNumBits() > B)) &&
           "Illegal VarBitInit expression!");
  }

public:
  VarBitInit(const VarBitInit &) = delete;
  VarBitInit &operator=(const VarBitInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_VarBitInit;
  }

  static const VarBitInit *get(const TypedInit *T, unsigned B);

  const Init *getBitVar() const { return TI; }
  unsigned getBitNum() const { return Bit; }

  std::string getAsString() const override;
  const Init *resolveReferences(Resolver &R) const override;

  const Init *getBit(unsigned B) const override {
    assert(B < 1 && "Bit index out of range!");
    return this;
  }
};

/// AL - Represent a reference to a 'def' in the description
class DefInit final : public TypedInit {
  friend class Record;

  const Record *Def;

  explicit DefInit(const Record *D);

public:
  DefInit(const DefInit &) = delete;
  DefInit &operator=(const DefInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_DefInit;
  }

  const Init *convertInitializerTo(const RecTy *Ty) const override;

  const Record *getDef() const { return Def; }

  const RecTy *getFieldType(const StringInit *FieldName) const override;

  bool isConcrete() const override { return true; }
  std::string getAsString() const override;

  const Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off def");
  }
};

/// classname<targs...> - Represent an uninstantiated anonymous class
/// instantiation.
class VarDefInit final
    : public TypedInit,
      public FoldingSetNode,
      private TrailingObjects<VarDefInit, const ArgumentInit *> {
  friend TrailingObjects;
  SMLoc Loc;
  const Record *Class;
  const DefInit *Def = nullptr; // after instantiation
  unsigned NumArgs;

  explicit VarDefInit(SMLoc Loc, const Record *Class,
                      ArrayRef<const ArgumentInit *> Args);

  const DefInit *instantiate();

public:
  VarDefInit(const VarDefInit &) = delete;
  VarDefInit &operator=(const VarDefInit &) = delete;

  // Do not use sized deallocation due to trailing objects.
  void operator delete(void *Ptr) { ::operator delete(Ptr); }

  static bool classof(const Init *I) {
    return I->getKind() == IK_VarDefInit;
  }
  static const VarDefInit *get(SMLoc Loc, const Record *Class,
                               ArrayRef<const ArgumentInit *> Args);

  void Profile(FoldingSetNodeID &ID) const;

  const Init *resolveReferences(Resolver &R) const override;
  const Init *Fold() const;

  std::string getAsString() const override;

  const ArgumentInit *getArg(unsigned i) const { return args()[i]; }

  using const_iterator = const ArgumentInit *const *;

  const_iterator args_begin() const { return args().begin(); }
  const_iterator args_end() const { return args().end(); }

  size_t         args_size () const { return NumArgs; }
  bool           args_empty() const { return NumArgs == 0; }

  ArrayRef<const ArgumentInit *> args() const {
    return getTrailingObjects(NumArgs);
  }

  const Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off anonymous def");
  }
};

/// X.Y - Represent a reference to a subfield of a variable
class FieldInit final : public TypedInit {
  const Init *Rec;             // Record we are referring to
  const StringInit *FieldName; // Field we are accessing

  FieldInit(const Init *R, const StringInit *FN)
      : TypedInit(IK_FieldInit, R->getFieldType(FN)), Rec(R), FieldName(FN) {
#ifndef NDEBUG
    if (!getType()) {
      llvm::errs() << "In Record = " << Rec->getAsString()
                   << ", got FieldName = " << *FieldName
                   << " with non-record type!\n";
      llvm_unreachable("FieldInit with non-record type!");
    }
#endif
  }

public:
  FieldInit(const FieldInit &) = delete;
  FieldInit &operator=(const FieldInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_FieldInit;
  }

  static const FieldInit *get(const Init *R, const StringInit *FN);

  const Init *getRecord() const { return Rec; }
  const StringInit *getFieldName() const { return FieldName; }

  const Init *getBit(unsigned Bit) const override;

  const Init *resolveReferences(Resolver &R) const override;
  const Init *Fold(const Record *CurRec) const;

  bool isConcrete() const override;
  std::string getAsString() const override {
    return Rec->getAsString() + "." + FieldName->getValue().str();
  }
};

/// (v a, b) - Represent a DAG tree value. DAG inits are required
/// to have at least one value then a (possibly empty) list of arguments. Each
/// argument can have a name associated with it.
class DagInit final
    : public TypedInit,
      public FoldingSetNode,
      private TrailingObjects<DagInit, const Init *, const StringInit *> {
  friend TrailingObjects;

  const Init *Val;
  const StringInit *ValName;
  unsigned NumArgs;

  DagInit(const Init *V, const StringInit *VN, ArrayRef<const Init *> Args,
          ArrayRef<const StringInit *> ArgNames);

  size_t numTrailingObjects(OverloadToken<const Init *>) const {
    return NumArgs;
  }

public:
  DagInit(const DagInit &) = delete;
  DagInit &operator=(const DagInit &) = delete;

  static bool classof(const Init *I) {
    return I->getKind() == IK_DagInit;
  }

  static const DagInit *get(const Init *V, const StringInit *VN,
                            ArrayRef<const Init *> Args,
                            ArrayRef<const StringInit *> ArgNames);

  static const DagInit *get(const Init *V, ArrayRef<const Init *> Args,
                            ArrayRef<const StringInit *> ArgNames) {
    return DagInit::get(V, nullptr, Args, ArgNames);
  }

  static const DagInit *
  get(const Init *V, const StringInit *VN,
      ArrayRef<std::pair<const Init *, const StringInit *>> ArgAndNames);

  static const DagInit *
  get(const Init *V,
      ArrayRef<std::pair<const Init *, const StringInit *>> ArgAndNames) {
    return DagInit::get(V, nullptr, ArgAndNames);
  }

  void Profile(FoldingSetNodeID &ID) const;

  const Init *getOperator() const { return Val; }
  const Record *getOperatorAsDef(ArrayRef<SMLoc> Loc) const;

  const StringInit *getName() const { return ValName; }

  StringRef getNameStr() const {
    return ValName ? ValName->getValue() : StringRef();
  }

  unsigned getNumArgs() const { return NumArgs; }

  const Init *getArg(unsigned Num) const { return getArgs()[Num]; }

  /// This method looks up the specified argument name and returns its argument
  /// number or std::nullopt if that argument name does not exist.
  std::optional<unsigned> getArgNo(StringRef Name) const;

  const StringInit *getArgName(unsigned Num) const {
    return getArgNames()[Num];
  }

  StringRef getArgNameStr(unsigned Num) const {
    const StringInit *Init = getArgName(Num);
    return Init ? Init->getValue() : StringRef();
  }

  ArrayRef<const Init *> getArgs() const {
    return getTrailingObjects<const Init *>(NumArgs);
  }

  ArrayRef<const StringInit *> getArgNames() const {
    return getTrailingObjects<const StringInit *>(NumArgs);
  }

  // Return a range of std::pair.
  auto getArgAndNames() const {
    auto Zip = llvm::zip_equal(getArgs(), getArgNames());
    using EltTy = decltype(*adl_begin(Zip));
    return llvm::map_range(Zip, [](const EltTy &E) {
      return std::make_pair(std::get<0>(E), std::get<1>(E));
    });
  }

  const Init *resolveReferences(Resolver &R) const override;

  bool isConcrete() const override;
  std::string getAsString() const override;

  using const_arg_iterator = SmallVectorImpl<const Init *>::const_iterator;
  using const_name_iterator =
      SmallVectorImpl<const StringInit *>::const_iterator;

  inline const_arg_iterator  arg_begin() const { return getArgs().begin(); }
  inline const_arg_iterator  arg_end  () const { return getArgs().end(); }

  inline size_t              arg_size () const { return NumArgs; }
  inline bool                arg_empty() const { return NumArgs == 0; }

  inline const_name_iterator name_begin() const { return getArgNames().begin();}
  inline const_name_iterator name_end  () const { return getArgNames().end(); }

  const Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off dag");
  }
};

//===----------------------------------------------------------------------===//
//  High-Level Classes
//===----------------------------------------------------------------------===//

/// This class represents a field in a record, including its name, type,
/// value, and source location.
class RecordVal {
  friend class Record;

public:
  enum FieldKind {
    FK_Normal,        // A normal record field.
    FK_NonconcreteOK, // A field that can be nonconcrete ('field' keyword).
    FK_TemplateArg,   // A template argument.
  };

private:
  const Init *Name;
  SMLoc Loc; // Source location of definition of name.
  PointerIntPair<const RecTy *, 2, FieldKind> TyAndKind;
  const Init *Value;
  bool IsUsed = false;

  /// Reference locations to this record value.
  SmallVector<SMRange, 0> ReferenceLocs;

public:
  RecordVal(const Init *N, const RecTy *T, FieldKind K);
  RecordVal(const Init *N, SMLoc Loc, const RecTy *T, FieldKind K);

  /// Get the record keeper used to unique this value.
  RecordKeeper &getRecordKeeper() const { return Name->getRecordKeeper(); }

  /// Get the name of the field as a StringRef.
  StringRef getName() const;

  /// Get the name of the field as an Init.
  const Init *getNameInit() const { return Name; }

  /// Get the name of the field as a std::string.
  std::string getNameInitAsString() const {
    return getNameInit()->getAsUnquotedString();
  }

  /// Get the source location of the point where the field was defined.
  const SMLoc &getLoc() const { return Loc; }

  /// Is this a field where nonconcrete values are okay?
  bool isNonconcreteOK() const {
    return TyAndKind.getInt() == FK_NonconcreteOK;
  }

  /// Is this a template argument?
  bool isTemplateArg() const {
    return TyAndKind.getInt() == FK_TemplateArg;
  }

  /// Get the type of the field value as a RecTy.
  const RecTy *getType() const { return TyAndKind.getPointer(); }

  /// Get the type of the field for printing purposes.
  std::string getPrintType() const;

  /// Get the value of the field as an Init.
  const Init *getValue() const { return Value; }

  /// Set the value of the field from an Init.
  bool setValue(const Init *V);

  /// Set the value and source location of the field.
  bool setValue(const Init *V, SMLoc NewLoc);

  /// Add a reference to this record value.
  void addReferenceLoc(SMRange Loc) { ReferenceLocs.push_back(Loc); }

  /// Return the references of this record value.
  ArrayRef<SMRange> getReferenceLocs() const { return ReferenceLocs; }

  /// Whether this value is used. Useful for reporting warnings, for example
  /// when a template argument is unused.
  void setUsed(bool Used) { IsUsed = Used; }
  bool isUsed() const { return IsUsed; }

  void dump() const;

  /// Print the value to an output stream, possibly with a semicolon.
  void print(raw_ostream &OS, bool PrintSem = true) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const RecordVal &RV) {
  RV.print(OS << "  ");
  return OS;
}

class Record {
public:
  struct AssertionInfo {
    SMLoc Loc;
    const Init *Condition;
    const Init *Message;

    // User-defined constructor to support std::make_unique(). It can be
    // removed in C++20 when braced initialization is supported.
    AssertionInfo(SMLoc Loc, const Init *Condition, const Init *Message)
        : Loc(Loc), Condition(Condition), Message(Message) {}
  };

  struct DumpInfo {
    SMLoc Loc;
    const Init *Message;

    // User-defined constructor to support std::make_unique(). It can be
    // removed in C++20 when braced initialization is supported.
    DumpInfo(SMLoc Loc, const Init *Message) : Loc(Loc), Message(Message) {}
  };

  enum RecordKind { RK_Def, RK_AnonymousDef, RK_Class, RK_MultiClass };

private:
  const Init *Name;
  // Location where record was instantiated, followed by the location of
  // multiclass prototypes used, and finally by the locations of references to
  // this record.
  SmallVector<SMLoc, 4> Locs;
  SmallVector<SMLoc, 0> ForwardDeclarationLocs;
  mutable SmallVector<SMRange, 0> ReferenceLocs;
  SmallVector<const Init *, 0> TemplateArgs;
  SmallVector<RecordVal, 0> Values;
  SmallVector<AssertionInfo, 0> Assertions;
  SmallVector<DumpInfo, 0> Dumps;

  // Direct superclasses, which are roots of the inheritance forest (yes, it
  // must be a forest; diamond-shaped inheritance is not allowed).
  SmallVector<std::pair<const Record *, SMRange>, 0> DirectSuperClasses;

  // Tracks Record instances. Not owned by Record.
  RecordKeeper &TrackedRecords;

  // The DefInit corresponding to this record.
  mutable DefInit *CorrespondingDefInit = nullptr;

  // Unique record ID.
  unsigned ID;

  RecordKind Kind;

  void checkName();

public:
  // Constructs a record.
  explicit Record(const Init *N, ArrayRef<SMLoc> locs, RecordKeeper &records,
                  RecordKind Kind = RK_Def)
      : Name(N), Locs(locs), TrackedRecords(records),
        ID(getNewUID(N->getRecordKeeper())), Kind(Kind) {
    checkName();
  }

  explicit Record(StringRef N, ArrayRef<SMLoc> locs, RecordKeeper &records,
                  RecordKind Kind = RK_Def)
      : Record(StringInit::get(records, N), locs, records, Kind) {}

  // When copy-constructing a Record, we must still guarantee a globally unique
  // ID number. Don't copy CorrespondingDefInit either, since it's owned by the
  // original record. All other fields can be copied normally.
  Record(const Record &O)
      : Name(O.Name), Locs(O.Locs), TemplateArgs(O.TemplateArgs),
        Values(O.Values), Assertions(O.Assertions),
        DirectSuperClasses(O.DirectSuperClasses),
        TrackedRecords(O.TrackedRecords), ID(getNewUID(O.getRecords())),
        Kind(O.Kind) {}

  static unsigned getNewUID(RecordKeeper &RK);

  unsigned getID() const { return ID; }

  StringRef getName() const { return cast<StringInit>(Name)->getValue(); }

  const Init *getNameInit() const { return Name; }

  std::string getNameInitAsString() const {
    return getNameInit()->getAsUnquotedString();
  }

  void setName(const Init *Name); // Also updates RecordKeeper.

  ArrayRef<SMLoc> getLoc() const { return Locs; }
  void appendLoc(SMLoc Loc) { Locs.push_back(Loc); }

  ArrayRef<SMLoc> getForwardDeclarationLocs() const {
    return ForwardDeclarationLocs;
  }

  /// Add a reference to this record value.
  void appendReferenceLoc(SMRange Loc) const { ReferenceLocs.push_back(Loc); }

  /// Return the references of this record value.
  ArrayRef<SMRange> getReferenceLocs() const { return ReferenceLocs; }

  // Update a class location when encountering a (re-)definition.
  void updateClassLoc(SMLoc Loc);

  // Make the type that this record should have based on its superclasses.
  const RecordRecTy *getType() const;

  /// get the corresponding DefInit.
  DefInit *getDefInit() const;

  bool isClass() const { return Kind == RK_Class; }

  bool isMultiClass() const { return Kind == RK_MultiClass; }

  bool isAnonymous() const { return Kind == RK_AnonymousDef; }

  ArrayRef<const Init *> getTemplateArgs() const { return TemplateArgs; }

  ArrayRef<RecordVal> getValues() const { return Values; }

  ArrayRef<AssertionInfo> getAssertions() const { return Assertions; }
  ArrayRef<DumpInfo> getDumps() const { return Dumps; }

  /// Append all superclasses in post-order to \p Classes.
  void getSuperClasses(std::vector<const Record *> &Classes) const {
    for (const Record *SC : make_first_range(DirectSuperClasses)) {
      SC->getSuperClasses(Classes);
      Classes.push_back(SC);
    }
  }

  /// Return all superclasses in post-order.
  std::vector<const Record *> getSuperClasses() const {
    std::vector<const Record *> Classes;
    getSuperClasses(Classes);
    return Classes;
  }

  /// Determine whether this record has the specified direct superclass.
  bool hasDirectSuperClass(const Record *SuperClass) const {
    return is_contained(make_first_range(DirectSuperClasses), SuperClass);
  }

  /// Return the direct superclasses of this record.
  ArrayRef<std::pair<const Record *, SMRange>> getDirectSuperClasses() const {
    return DirectSuperClasses;
  }

  bool isTemplateArg(const Init *Name) const {
    return llvm::is_contained(TemplateArgs, Name);
  }

  const RecordVal *getValue(const Init *Name) const {
    for (const RecordVal &Val : Values)
      if (Val.Name == Name) return &Val;
    return nullptr;
  }

  const RecordVal *getValue(StringRef Name) const {
    return getValue(StringInit::get(getRecords(), Name));
  }

  RecordVal *getValue(const Init *Name) {
    return const_cast<RecordVal *>(
        static_cast<const Record *>(this)->getValue(Name));
  }

  RecordVal *getValue(StringRef Name) {
    return const_cast<RecordVal *>(
        static_cast<const Record *>(this)->getValue(Name));
  }

  void addTemplateArg(const Init *Name) {
    assert(!isTemplateArg(Name) && "Template arg already defined!");
    TemplateArgs.push_back(Name);
  }

  void addValue(const RecordVal &RV) {
    assert(getValue(RV.getNameInit()) == nullptr && "Value already added!");
    Values.push_back(RV);
  }

  void removeValue(const Init *Name) {
    auto It = llvm::find_if(
        Values, [Name](const RecordVal &V) { return V.getNameInit() == Name; });
    if (It == Values.end())
      llvm_unreachable("Cannot remove an entry that does not exist!");
    Values.erase(It);
  }

  void removeValue(StringRef Name) {
    removeValue(StringInit::get(getRecords(), Name));
  }

  void addAssertion(SMLoc Loc, const Init *Condition, const Init *Message) {
    Assertions.push_back(AssertionInfo(Loc, Condition, Message));
  }

  void addDump(SMLoc Loc, const Init *Message) {
    Dumps.push_back(DumpInfo(Loc, Message));
  }

  void appendAssertions(const Record *Rec) {
    Assertions.append(Rec->Assertions);
  }

  void appendDumps(const Record *Rec) { Dumps.append(Rec->Dumps); }

  void checkRecordAssertions();
  void emitRecordDumps();
  void checkUnusedTemplateArgs();

  bool isSubClassOf(const Record *R) const {
    for (const Record *SC : make_first_range(DirectSuperClasses)) {
      if (SC == R || SC->isSubClassOf(R))
        return true;
    }
    return false;
  }

  bool isSubClassOf(StringRef Name) const {
    for (const Record *SC : make_first_range(DirectSuperClasses)) {
      if (const auto *SI = dyn_cast<StringInit>(SC->getNameInit())) {
        if (SI->getValue() == Name)
          return true;
      } else if (SC->getNameInitAsString() == Name) {
        return true;
      }
      if (SC->isSubClassOf(Name))
        return true;
    }
    return false;
  }

  void addDirectSuperClass(const Record *R, SMRange Range) {
    assert(!CorrespondingDefInit &&
           "changing type of record after it has been referenced");
    assert(!isSubClassOf(R) && "Already subclassing record!");
    DirectSuperClasses.emplace_back(R, Range);
  }

  /// If there are any field references that refer to fields that have been
  /// filled in, we can propagate the values now.
  ///
  /// This is a final resolve: any error messages, e.g. due to undefined !cast
  /// references, are generated now.
  void resolveReferences(const Init *NewName = nullptr);

  /// Apply the resolver to the name of the record as well as to the
  /// initializers of all fields of the record except SkipVal.
  ///
  /// The resolver should not resolve any of the fields itself, to avoid
  /// recursion / infinite loops.
  void resolveReferences(Resolver &R, const RecordVal *SkipVal = nullptr);

  RecordKeeper &getRecords() const {
    return TrackedRecords;
  }

  void dump() const;

  //===--------------------------------------------------------------------===//
  // High-level methods useful to tablegen back-ends
  //

  /// Return the source location for the named field.
  SMLoc getFieldLoc(StringRef FieldName) const;

  /// Return the initializer for a value with the specified name, or throw an
  /// exception if the field does not exist.
  const Init *getValueInit(StringRef FieldName) const;

  /// Return true if the named field is unset.
  bool isValueUnset(StringRef FieldName) const {
    return isa<UnsetInit>(getValueInit(FieldName));
  }

  /// This method looks up the specified field and returns its value as a
  /// string, throwing an exception if the field does not exist or if the value
  /// is not a string.
  StringRef getValueAsString(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a
  /// string, throwing an exception if the value is not a string and
  /// std::nullopt if the field does not exist.
  std::optional<StringRef> getValueAsOptionalString(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a
  /// BitsInit, throwing an exception if the field does not exist or if the
  /// value is not the right type.
  const BitsInit *getValueAsBitsInit(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a
  /// ListInit, throwing an exception if the field does not exist or if the
  /// value is not the right type.
  const ListInit *getValueAsListInit(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a
  /// vector of records, throwing an exception if the field does not exist or
  /// if the value is not the right type.
  std::vector<const Record *> getValueAsListOfDefs(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a
  /// vector of integers, throwing an exception if the field does not exist or
  /// if the value is not the right type.
  std::vector<int64_t> getValueAsListOfInts(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a
  /// vector of strings, throwing an exception if the field does not exist or
  /// if the value is not the right type.
  std::vector<StringRef> getValueAsListOfStrings(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a
  /// Record, throwing an exception if the field does not exist or if the value
  /// is not the right type.
  const Record *getValueAsDef(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a
  /// Record, returning null if the field exists but is "uninitialized" (i.e.
  /// set to `?`), and throwing an exception if the field does not exist or if
  /// its value is not the right type.
  const Record *getValueAsOptionalDef(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a bit,
  /// throwing an exception if the field does not exist or if the value is not
  /// the right type.
  bool getValueAsBit(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as a bit.
  /// If the field is unset, sets Unset to true and returns false.
  bool getValueAsBitOrUnset(StringRef FieldName, bool &Unset) const;

  /// This method looks up the specified field and returns its value as an
  /// int64_t, throwing an exception if the field does not exist or if the
  /// value is not the right type.
  int64_t getValueAsInt(StringRef FieldName) const;

  /// This method looks up the specified field and returns its value as an Dag,
  /// throwing an exception if the field does not exist or if the value is not
  /// the right type.
  const DagInit *getValueAsDag(StringRef FieldName) const;
};

raw_ostream &operator<<(raw_ostream &OS, const Record &R);

class RecordKeeper {
  using RecordMap = std::map<std::string, std::unique_ptr<Record>, std::less<>>;
  using GlobalMap = std::map<std::string, const Init *, std::less<>>;

public:
  RecordKeeper();
  ~RecordKeeper();

  /// Return the internal implementation of the RecordKeeper.
  detail::RecordKeeperImpl &getImpl() { return *Impl; }

  /// Get the main TableGen input file's name.
  StringRef getInputFilename() const { return InputFilename; }

  /// Get the map of classes.
  const RecordMap &getClasses() const { return Classes; }

  /// Get the map of records (defs).
  const RecordMap &getDefs() const { return Defs; }

  /// Get the map of global variables.
  const GlobalMap &getGlobals() const { return ExtraGlobals; }

  /// Get the class with the specified name.
  const Record *getClass(StringRef Name) const {
    auto I = Classes.find(Name);
    return I == Classes.end() ? nullptr : I->second.get();
  }

  /// Get the concrete record with the specified name.
  const Record *getDef(StringRef Name) const {
    auto I = Defs.find(Name);
    return I == Defs.end() ? nullptr : I->second.get();
  }

  /// Get the \p Init value of the specified global variable.
  const Init *getGlobal(StringRef Name) const {
    if (const Record *R = getDef(Name))
      return R->getDefInit();
    auto It = ExtraGlobals.find(Name);
    return It == ExtraGlobals.end() ? nullptr : It->second;
  }

  void saveInputFilename(std::string Filename) {
    InputFilename = std::move(Filename);
  }

  void addClass(std::unique_ptr<Record> R) {
    bool Ins =
        Classes.try_emplace(std::string(R->getName()), std::move(R)).second;
    (void)Ins;
    assert(Ins && "Class already exists");
  }

  void addDef(std::unique_ptr<Record> R) {
    bool Ins = Defs.try_emplace(std::string(R->getName()), std::move(R)).second;
    (void)Ins;
    assert(Ins && "Record already exists");
    // Clear cache
    if (!Cache.empty())
      Cache.clear();
  }

  void addExtraGlobal(StringRef Name, const Init *I) {
    bool Ins = ExtraGlobals.try_emplace(std::string(Name), I).second;
    (void)Ins;
    assert(!getDef(Name));
    assert(Ins && "Global already exists");
  }

  const Init *getNewAnonymousName();

  TGTimer &getTimer() const { return *Timer; }

  //===--------------------------------------------------------------------===//
  // High-level helper methods, useful for tablegen backends.

  /// Get all the concrete records that inherit from the one specified
  /// class. The class must be defined.
  ArrayRef<const Record *> getAllDerivedDefinitions(StringRef ClassName) const;

  /// Get all the concrete records that inherit from all the specified
  /// classes. The classes must be defined.
  std::vector<const Record *>
  getAllDerivedDefinitions(ArrayRef<StringRef> ClassNames) const;

  /// Get all the concrete records that inherit from specified class, if the
  /// class is defined. Returns an empty vector if the class is not defined.
  ArrayRef<const Record *>
  getAllDerivedDefinitionsIfDefined(StringRef ClassName) const;

  void dump() const;

  void dumpAllocationStats(raw_ostream &OS) const;

private:
  RecordKeeper(RecordKeeper &&) = delete;
  RecordKeeper(const RecordKeeper &) = delete;
  RecordKeeper &operator=(RecordKeeper &&) = delete;
  RecordKeeper &operator=(const RecordKeeper &) = delete;

  std::string InputFilename;
  RecordMap Classes, Defs;
  mutable std::map<std::string, std::vector<const Record *>> Cache;
  GlobalMap ExtraGlobals;

  /// The internal uniquer implementation of the RecordKeeper.
  std::unique_ptr<detail::RecordKeeperImpl> Impl;
  std::unique_ptr<TGTimer> Timer;
};

/// Sorting predicate to sort record pointers by name.
struct LessRecord {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getName().compare_numeric(Rec2->getName()) < 0;
  }
};

/// Sorting predicate to sort record pointers by their
/// unique ID. If you just need a deterministic order, use this, since it
/// just compares two `unsigned`; the other sorting predicates require
/// string manipulation.
struct LessRecordByID {
  bool operator()(const Record *LHS, const Record *RHS) const {
    return LHS->getID() < RHS->getID();
  }
};

/// Sorting predicate to sort record pointers by their Name field.
struct LessRecordFieldName {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getValueAsString("Name") < Rec2->getValueAsString("Name");
  }
};

struct LessRecordRegister {
  struct RecordParts {
    SmallVector<std::pair< bool, StringRef>, 4> Parts;

    RecordParts(StringRef Rec) {
      if (Rec.empty())
        return;

      size_t Len = 0;
      const char *Start = Rec.data();
      const char *Curr = Start;
      bool IsDigitPart = isDigit(Curr[0]);
      for (size_t I = 0, E = Rec.size(); I != E; ++I, ++Len) {
        bool IsDigit = isDigit(Curr[I]);
        if (IsDigit != IsDigitPart) {
          Parts.emplace_back(IsDigitPart, StringRef(Start, Len));
          Len = 0;
          Start = &Curr[I];
          IsDigitPart = isDigit(Curr[I]);
        }
      }
      // Push the last part.
      Parts.emplace_back(IsDigitPart, StringRef(Start, Len));
    }

    size_t size() { return Parts.size(); }

    std::pair<bool, StringRef> getPart(size_t Idx) { return Parts[Idx]; }
  };

  bool operator()(const Record *Rec1, const Record *Rec2) const {
    int64_t LHSPositionOrder = Rec1->getValueAsInt("PositionOrder");
    int64_t RHSPositionOrder = Rec2->getValueAsInt("PositionOrder");
    if (LHSPositionOrder != RHSPositionOrder)
      return LHSPositionOrder < RHSPositionOrder;

    RecordParts LHSParts(StringRef(Rec1->getName()));
    RecordParts RHSParts(StringRef(Rec2->getName()));

    size_t LHSNumParts = LHSParts.size();
    size_t RHSNumParts = RHSParts.size();
    assert (LHSNumParts && RHSNumParts && "Expected at least one part!");

    if (LHSNumParts != RHSNumParts)
      return LHSNumParts < RHSNumParts;

    // We expect the registers to be of the form [_a-zA-Z]+([0-9]*[_a-zA-Z]*)*.
    for (size_t I = 0, E = LHSNumParts; I < E; I+=2) {
      std::pair<bool, StringRef> LHSPart = LHSParts.getPart(I);
      std::pair<bool, StringRef> RHSPart = RHSParts.getPart(I);
      // Expect even part to always be alpha.
      assert (LHSPart.first == false && RHSPart.first == false &&
              "Expected both parts to be alpha.");
      if (int Res = LHSPart.second.compare(RHSPart.second))
        return Res < 0;
    }
    for (size_t I = 1, E = LHSNumParts; I < E; I+=2) {
      std::pair<bool, StringRef> LHSPart = LHSParts.getPart(I);
      std::pair<bool, StringRef> RHSPart = RHSParts.getPart(I);
      // Expect odd part to always be numeric.
      assert (LHSPart.first == true && RHSPart.first == true &&
              "Expected both parts to be numeric.");
      if (LHSPart.second.size() != RHSPart.second.size())
        return LHSPart.second.size() < RHSPart.second.size();

      unsigned LHSVal, RHSVal;

      bool LHSFailed = LHSPart.second.getAsInteger(10, LHSVal); (void)LHSFailed;
      assert(!LHSFailed && "Unable to convert LHS to integer.");
      bool RHSFailed = RHSPart.second.getAsInteger(10, RHSVal); (void)RHSFailed;
      assert(!RHSFailed && "Unable to convert RHS to integer.");

      if (LHSVal != RHSVal)
        return LHSVal < RHSVal;
    }
    return LHSNumParts < RHSNumParts;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const RecordKeeper &RK);

//===----------------------------------------------------------------------===//
//  Resolvers
//===----------------------------------------------------------------------===//

/// Interface for looking up the initializer for a variable name, used by
/// Init::resolveReferences.
class Resolver {
  const Record *CurRec;
  bool IsFinal = false;

public:
  explicit Resolver(const Record *CurRec) : CurRec(CurRec) {}
  virtual ~Resolver() = default;

  const Record *getCurrentRecord() const { return CurRec; }

  /// Return the initializer for the given variable name (should normally be a
  /// StringInit), or nullptr if the name could not be resolved.
  virtual const Init *resolve(const Init *VarName) = 0;

  // Whether bits in a BitsInit should stay unresolved if resolving them would
  // result in a ? (UnsetInit). This behavior is used to represent instruction
  // encodings by keeping references to unset variables within a record.
  virtual bool keepUnsetBits() const { return false; }

  // Whether this is the final resolve step before adding a record to the
  // RecordKeeper. Error reporting during resolve and related constant folding
  // should only happen when this is true.
  bool isFinal() const { return IsFinal; }

  void setFinal(bool Final) { IsFinal = Final; }
};

/// Resolve arbitrary mappings.
class MapResolver final : public Resolver {
  struct MappedValue {
    const Init *V;
    bool Resolved;

    MappedValue() : V(nullptr), Resolved(false) {}
    MappedValue(const Init *V, bool Resolved) : V(V), Resolved(Resolved) {}
  };

  DenseMap<const Init *, MappedValue> Map;

public:
  explicit MapResolver(const Record *CurRec = nullptr) : Resolver(CurRec) {}

  void set(const Init *Key, const Init *Value) { Map[Key] = {Value, false}; }

  bool isComplete(Init *VarName) const {
    auto It = Map.find(VarName);
    assert(It != Map.end() && "key must be present in map");
    return It->second.V->isComplete();
  }

  const Init *resolve(const Init *VarName) override;
};

/// Resolve all variables from a record except for unset variables.
class RecordResolver final : public Resolver {
  DenseMap<const Init *, const Init *> Cache;
  SmallVector<const Init *, 4> Stack;
  const Init *Name = nullptr;

public:
  explicit RecordResolver(const Record &R) : Resolver(&R) {}

  void setName(const Init *NewName) { Name = NewName; }

  const Init *resolve(const Init *VarName) override;

  bool keepUnsetBits() const override { return true; }
};

/// Delegate resolving to a sub-resolver, but shadow some variable names.
class ShadowResolver final : public Resolver {
  Resolver &R;
  DenseSet<const Init *> Shadowed;

public:
  explicit ShadowResolver(Resolver &R)
      : Resolver(R.getCurrentRecord()), R(R) {
    setFinal(R.isFinal());
  }

  void addShadow(const Init *Key) { Shadowed.insert(Key); }

  const Init *resolve(const Init *VarName) override {
    if (Shadowed.count(VarName))
      return nullptr;
    return R.resolve(VarName);
  }
};

/// (Optionally) delegate resolving to a sub-resolver, and keep track whether
/// there were unresolved references.
class TrackUnresolvedResolver final : public Resolver {
  Resolver *R;
  bool FoundUnresolved = false;

public:
  explicit TrackUnresolvedResolver(Resolver *R = nullptr)
      : Resolver(R ? R->getCurrentRecord() : nullptr), R(R) {}

  bool foundUnresolved() const { return FoundUnresolved; }

  const Init *resolve(const Init *VarName) override;
};

/// Do not resolve anything, but keep track of whether a given variable was
/// referenced.
class HasReferenceResolver final : public Resolver {
  const Init *VarNameToTrack;
  bool Found = false;

public:
  explicit HasReferenceResolver(const Init *VarNameToTrack)
      : Resolver(nullptr), VarNameToTrack(VarNameToTrack) {}

  bool found() const { return Found; }

  const Init *resolve(const Init *VarName) override;
};

void EmitDetailedRecords(const RecordKeeper &RK, raw_ostream &OS);
void EmitJSON(const RecordKeeper &RK, raw_ostream &OS);

} // end namespace llvm

#endif // LLVM_TABLEGEN_RECORD_H
