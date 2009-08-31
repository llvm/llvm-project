//===- Record.h - Classes to represent Table Records ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the main TableGen data structures, including the TableGen
// types, values, and high-level data structures.
//
//===----------------------------------------------------------------------===//

#ifndef RECORD_H
#define RECORD_H

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace llvm {
class raw_ostream;
  
// RecTy subclasses.
class BitRecTy;
class BitsRecTy;
class IntRecTy;
class StringRecTy;
class ListRecTy;
class CodeRecTy;
class DagRecTy;
class RecordRecTy;

// Init subclasses.
struct Init;
class UnsetInit;
class BitInit;
class BitsInit;
class IntInit;
class StringInit;
class CodeInit;
class ListInit;
class UnOpInit;
class BinOpInit;
class TernOpInit;
class DefInit;
class DagInit;
class TypedInit;
class VarInit;
class FieldInit;
class VarBitInit;
class VarListElementInit;

// Other classes.
class Record;
class RecordVal;
struct MultiClass;

//===----------------------------------------------------------------------===//
//  Type Classes
//===----------------------------------------------------------------------===//

struct RecTy {
  virtual ~RecTy() {}

  virtual std::string getAsString() const = 0;
  void print(raw_ostream &OS) const { OS << getAsString(); }
  void dump() const;

  /// typeIsConvertibleTo - Return true if all values of 'this' type can be
  /// converted to the specified type.
  virtual bool typeIsConvertibleTo(const RecTy *RHS) const = 0;

public:   // These methods should only be called from subclasses of Init
  virtual Init *convertValue( UnsetInit *UI) { return 0; }
  virtual Init *convertValue(   BitInit *BI) { return 0; }
  virtual Init *convertValue(  BitsInit *BI) { return 0; }
  virtual Init *convertValue(   IntInit *II) { return 0; }
  virtual Init *convertValue(StringInit *SI) { return 0; }
  virtual Init *convertValue(  ListInit *LI) { return 0; }
  virtual Init *convertValue( UnOpInit *UI) {
    return convertValue((TypedInit*)UI);
  }
  virtual Init *convertValue( BinOpInit *UI) {
    return convertValue((TypedInit*)UI);
  }
  virtual Init *convertValue( TernOpInit *UI) {
    return convertValue((TypedInit*)UI);
  }
  virtual Init *convertValue(  CodeInit *CI) { return 0; }
  virtual Init *convertValue(VarBitInit *VB) { return 0; }
  virtual Init *convertValue(   DefInit *DI) { return 0; }
  virtual Init *convertValue(   DagInit *DI) { return 0; }
  virtual Init *convertValue( TypedInit *TI) { return 0; }
  virtual Init *convertValue(   VarInit *VI) {
    return convertValue((TypedInit*)VI);
  }
  virtual Init *convertValue( FieldInit *FI) {
    return convertValue((TypedInit*)FI);
  }

public:   // These methods should only be called by subclasses of RecTy.
  // baseClassOf - These virtual methods should be overloaded to return true iff
  // all values of type 'RHS' can be converted to the 'this' type.
  virtual bool baseClassOf(const BitRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const BitsRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const IntRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const StringRecTy *RHS) const { return false; }
  virtual bool baseClassOf(const ListRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const CodeRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const DagRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const RecordRecTy *RHS) const { return false; }
};

inline raw_ostream &operator<<(raw_ostream &OS, const RecTy &Ty) {
  Ty.print(OS);
  return OS;
}


/// BitRecTy - 'bit' - Represent a single bit
///
class BitRecTy : public RecTy {
public:
  virtual Init *convertValue( UnsetInit *UI) { return (Init*)UI; }
  virtual Init *convertValue(   BitInit *BI) { return (Init*)BI; }
  virtual Init *convertValue(  BitsInit *BI);
  virtual Init *convertValue(   IntInit *II);
  virtual Init *convertValue(StringInit *SI) { return 0; }
  virtual Init *convertValue(  ListInit *LI) { return 0; }
  virtual Init *convertValue(  CodeInit *CI) { return 0; }
  virtual Init *convertValue(VarBitInit *VB) { return (Init*)VB; }
  virtual Init *convertValue(   DefInit *DI) { return 0; }
  virtual Init *convertValue(   DagInit *DI) { return 0; }
  virtual Init *convertValue( UnOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( BinOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TernOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TypedInit *TI);
  virtual Init *convertValue(   VarInit *VI) { return RecTy::convertValue(VI);}
  virtual Init *convertValue( FieldInit *FI) { return RecTy::convertValue(FI);}

  std::string getAsString() const { return "bit"; }

  bool typeIsConvertibleTo(const RecTy *RHS) const {
    return RHS->baseClassOf(this);
  }
  virtual bool baseClassOf(const BitRecTy    *RHS) const { return true; }
  virtual bool baseClassOf(const BitsRecTy   *RHS) const;
  virtual bool baseClassOf(const IntRecTy    *RHS) const { return true; }
  virtual bool baseClassOf(const StringRecTy *RHS) const { return false; }
  virtual bool baseClassOf(const ListRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const CodeRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const DagRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const RecordRecTy *RHS) const { return false; }

};


// BitsRecTy - 'bits<n>' - Represent a fixed number of bits
/// BitsRecTy - 'bits&lt;n&gt;' - Represent a fixed number of bits
///
class BitsRecTy : public RecTy {
  unsigned Size;
public:
  explicit BitsRecTy(unsigned Sz) : Size(Sz) {}

  unsigned getNumBits() const { return Size; }

  virtual Init *convertValue( UnsetInit *UI);
  virtual Init *convertValue(   BitInit *UI);
  virtual Init *convertValue(  BitsInit *BI);
  virtual Init *convertValue(   IntInit *II);
  virtual Init *convertValue(StringInit *SI) { return 0; }
  virtual Init *convertValue(  ListInit *LI) { return 0; }
  virtual Init *convertValue(  CodeInit *CI) { return 0; }
  virtual Init *convertValue(VarBitInit *VB) { return 0; }
  virtual Init *convertValue(   DefInit *DI) { return 0; }
  virtual Init *convertValue(   DagInit *DI) { return 0; }
  virtual Init *convertValue( UnOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( BinOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TernOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TypedInit *TI);
  virtual Init *convertValue(   VarInit *VI) { return RecTy::convertValue(VI);}
  virtual Init *convertValue( FieldInit *FI) { return RecTy::convertValue(FI);}

  std::string getAsString() const;

  bool typeIsConvertibleTo(const RecTy *RHS) const {
    return RHS->baseClassOf(this);
  }
  virtual bool baseClassOf(const BitRecTy    *RHS) const { return Size == 1; }
  virtual bool baseClassOf(const BitsRecTy   *RHS) const {
    return RHS->Size == Size;
  }
  virtual bool baseClassOf(const IntRecTy    *RHS) const { return true; }
  virtual bool baseClassOf(const StringRecTy *RHS) const { return false; }
  virtual bool baseClassOf(const ListRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const CodeRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const DagRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const RecordRecTy *RHS) const { return false; }

};


/// IntRecTy - 'int' - Represent an integer value of no particular size
///
class IntRecTy : public RecTy {
public:
  virtual Init *convertValue( UnsetInit *UI) { return (Init*)UI; }
  virtual Init *convertValue(   BitInit *BI);
  virtual Init *convertValue(  BitsInit *BI);
  virtual Init *convertValue(   IntInit *II) { return (Init*)II; }
  virtual Init *convertValue(StringInit *SI) { return 0; }
  virtual Init *convertValue(  ListInit *LI) { return 0; }
  virtual Init *convertValue(  CodeInit *CI) { return 0; }
  virtual Init *convertValue(VarBitInit *VB) { return 0; }
  virtual Init *convertValue(   DefInit *DI) { return 0; }
  virtual Init *convertValue(   DagInit *DI) { return 0; }
  virtual Init *convertValue( UnOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( BinOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TernOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TypedInit *TI);
  virtual Init *convertValue(   VarInit *VI) { return RecTy::convertValue(VI);}
  virtual Init *convertValue( FieldInit *FI) { return RecTy::convertValue(FI);}

  std::string getAsString() const { return "int"; }

  bool typeIsConvertibleTo(const RecTy *RHS) const {
    return RHS->baseClassOf(this);
  }

  virtual bool baseClassOf(const BitRecTy    *RHS) const { return true; }
  virtual bool baseClassOf(const BitsRecTy   *RHS) const { return true; }
  virtual bool baseClassOf(const IntRecTy    *RHS) const { return true; }
  virtual bool baseClassOf(const StringRecTy *RHS) const { return false; }
  virtual bool baseClassOf(const ListRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const CodeRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const DagRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const RecordRecTy *RHS) const { return false; }

};

/// StringRecTy - 'string' - Represent an string value
///
class StringRecTy : public RecTy {
public:
  virtual Init *convertValue( UnsetInit *UI) { return (Init*)UI; }
  virtual Init *convertValue(   BitInit *BI) { return 0; }
  virtual Init *convertValue(  BitsInit *BI) { return 0; }
  virtual Init *convertValue(   IntInit *II) { return 0; }
  virtual Init *convertValue(StringInit *SI) { return (Init*)SI; }
  virtual Init *convertValue(  ListInit *LI) { return 0; }
  virtual Init *convertValue( UnOpInit *BO);
  virtual Init *convertValue( BinOpInit *BO);
  virtual Init *convertValue( TernOpInit *BO) { return RecTy::convertValue(BO);}

  virtual Init *convertValue(  CodeInit *CI) { return 0; }
  virtual Init *convertValue(VarBitInit *VB) { return 0; }
  virtual Init *convertValue(   DefInit *DI) { return 0; }
  virtual Init *convertValue(   DagInit *DI) { return 0; }
  virtual Init *convertValue( TypedInit *TI);
  virtual Init *convertValue(   VarInit *VI) { return RecTy::convertValue(VI);}
  virtual Init *convertValue( FieldInit *FI) { return RecTy::convertValue(FI);}

  std::string getAsString() const { return "string"; }

  bool typeIsConvertibleTo(const RecTy *RHS) const {
    return RHS->baseClassOf(this);
  }

  virtual bool baseClassOf(const BitRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const BitsRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const IntRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const StringRecTy *RHS) const { return true; }
  virtual bool baseClassOf(const ListRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const CodeRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const DagRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const RecordRecTy *RHS) const { return false; }
};

// ListRecTy - 'list<Ty>' - Represent a list of values, all of which must be of
// the specified type.
/// ListRecTy - 'list&lt;Ty&gt;' - Represent a list of values, all of which must
/// be of the specified type.
///
class ListRecTy : public RecTy {
  RecTy *Ty;
public:
  explicit ListRecTy(RecTy *T) : Ty(T) {}

  RecTy *getElementType() const { return Ty; }

  virtual Init *convertValue( UnsetInit *UI) { return (Init*)UI; }
  virtual Init *convertValue(   BitInit *BI) { return 0; }
  virtual Init *convertValue(  BitsInit *BI) { return 0; }
  virtual Init *convertValue(   IntInit *II) { return 0; }
  virtual Init *convertValue(StringInit *SI) { return 0; }
  virtual Init *convertValue(  ListInit *LI);
  virtual Init *convertValue(  CodeInit *CI) { return 0; }
  virtual Init *convertValue(VarBitInit *VB) { return 0; }
  virtual Init *convertValue(   DefInit *DI) { return 0; }
  virtual Init *convertValue(   DagInit *DI) { return 0; }
  virtual Init *convertValue( UnOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( BinOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TernOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TypedInit *TI);
  virtual Init *convertValue(   VarInit *VI) { return RecTy::convertValue(VI);}
  virtual Init *convertValue( FieldInit *FI) { return RecTy::convertValue(FI);}

  std::string getAsString() const;

  bool typeIsConvertibleTo(const RecTy *RHS) const {
    return RHS->baseClassOf(this);
  }

  virtual bool baseClassOf(const BitRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const BitsRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const IntRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const StringRecTy *RHS) const { return false; }
  virtual bool baseClassOf(const ListRecTy   *RHS) const {
    return RHS->getElementType()->typeIsConvertibleTo(Ty);
  }
  virtual bool baseClassOf(const CodeRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const DagRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const RecordRecTy *RHS) const { return false; }
};

/// CodeRecTy - 'code' - Represent an code fragment, function or method.
///
class CodeRecTy : public RecTy {
public:
  virtual Init *convertValue( UnsetInit *UI) { return (Init*)UI; }
  virtual Init *convertValue(   BitInit *BI) { return 0; }
  virtual Init *convertValue(  BitsInit *BI) { return 0; }
  virtual Init *convertValue(   IntInit *II) { return 0; }
  virtual Init *convertValue(StringInit *SI) { return 0; }
  virtual Init *convertValue(  ListInit *LI) { return 0; }
  virtual Init *convertValue(  CodeInit *CI) { return (Init*)CI; }
  virtual Init *convertValue(VarBitInit *VB) { return 0; }
  virtual Init *convertValue(   DefInit *DI) { return 0; }
  virtual Init *convertValue(   DagInit *DI) { return 0; }
  virtual Init *convertValue( UnOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( BinOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TernOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TypedInit *TI);
  virtual Init *convertValue(   VarInit *VI) { return RecTy::convertValue(VI);}
  virtual Init *convertValue( FieldInit *FI) { return RecTy::convertValue(FI);}

  std::string getAsString() const { return "code"; }

  bool typeIsConvertibleTo(const RecTy *RHS) const {
    return RHS->baseClassOf(this);
  }
  virtual bool baseClassOf(const BitRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const BitsRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const IntRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const StringRecTy *RHS) const { return false; }
  virtual bool baseClassOf(const ListRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const CodeRecTy   *RHS) const { return true; }
  virtual bool baseClassOf(const DagRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const RecordRecTy *RHS) const { return false; }
};

/// DagRecTy - 'dag' - Represent a dag fragment
///
class DagRecTy : public RecTy {
public:
  virtual Init *convertValue( UnsetInit *UI) { return (Init*)UI; }
  virtual Init *convertValue(   BitInit *BI) { return 0; }
  virtual Init *convertValue(  BitsInit *BI) { return 0; }
  virtual Init *convertValue(   IntInit *II) { return 0; }
  virtual Init *convertValue(StringInit *SI) { return 0; }
  virtual Init *convertValue(  ListInit *LI) { return 0; }
  virtual Init *convertValue(  CodeInit *CI) { return 0; }
  virtual Init *convertValue(VarBitInit *VB) { return 0; }
  virtual Init *convertValue(   DefInit *DI) { return 0; }
  virtual Init *convertValue( UnOpInit *BO);
  virtual Init *convertValue( BinOpInit *BO);
  virtual Init *convertValue( TernOpInit *BO) { return RecTy::convertValue(BO);}
  virtual Init *convertValue(   DagInit *CI) { return (Init*)CI; }
  virtual Init *convertValue( TypedInit *TI);
  virtual Init *convertValue(   VarInit *VI) { return RecTy::convertValue(VI);}
  virtual Init *convertValue( FieldInit *FI) { return RecTy::convertValue(FI);}

  std::string getAsString() const { return "dag"; }

  bool typeIsConvertibleTo(const RecTy *RHS) const {
    return RHS->baseClassOf(this);
  }

  virtual bool baseClassOf(const BitRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const BitsRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const IntRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const StringRecTy *RHS) const { return false; }
  virtual bool baseClassOf(const ListRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const CodeRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const DagRecTy    *RHS) const { return true; }
  virtual bool baseClassOf(const RecordRecTy *RHS) const { return false; }
};


/// RecordRecTy - '[classname]' - Represent an instance of a class, such as:
/// (R32 X = EAX).
///
class RecordRecTy : public RecTy {
  Record *Rec;
public:
  explicit RecordRecTy(Record *R) : Rec(R) {}

  Record *getRecord() const { return Rec; }

  virtual Init *convertValue( UnsetInit *UI) { return (Init*)UI; }
  virtual Init *convertValue(   BitInit *BI) { return 0; }
  virtual Init *convertValue(  BitsInit *BI) { return 0; }
  virtual Init *convertValue(   IntInit *II) { return 0; }
  virtual Init *convertValue(StringInit *SI) { return 0; }
  virtual Init *convertValue(  ListInit *LI) { return 0; }
  virtual Init *convertValue(  CodeInit *CI) { return 0; }
  virtual Init *convertValue(VarBitInit *VB) { return 0; }
  virtual Init *convertValue( UnOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( BinOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue( TernOpInit *UI) { return RecTy::convertValue(UI);}
  virtual Init *convertValue(   DefInit *DI);
  virtual Init *convertValue(   DagInit *DI) { return 0; }
  virtual Init *convertValue( TypedInit *VI);
  virtual Init *convertValue(   VarInit *VI) { return RecTy::convertValue(VI);}
  virtual Init *convertValue( FieldInit *FI) { return RecTy::convertValue(FI);}

  std::string getAsString() const;

  bool typeIsConvertibleTo(const RecTy *RHS) const {
    return RHS->baseClassOf(this);
  }
  virtual bool baseClassOf(const BitRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const BitsRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const IntRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const StringRecTy *RHS) const { return false; }
  virtual bool baseClassOf(const ListRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const CodeRecTy   *RHS) const { return false; }
  virtual bool baseClassOf(const DagRecTy    *RHS) const { return false; }
  virtual bool baseClassOf(const RecordRecTy *RHS) const;
};

/// resolveTypes - Find a common type that T1 and T2 convert to.  
/// Return 0 if no such type exists.
///
RecTy *resolveTypes(RecTy *T1, RecTy *T2);

//===----------------------------------------------------------------------===//
//  Initializer Classes
//===----------------------------------------------------------------------===//

struct Init {
  virtual ~Init() {}

  /// isComplete - This virtual method should be overridden by values that may
  /// not be completely specified yet.
  virtual bool isComplete() const { return true; }

  /// print - Print out this value.
  void print(raw_ostream &OS) const { OS << getAsString(); }

  /// getAsString - Convert this value to a string form.
  virtual std::string getAsString() const = 0;

  /// dump - Debugging method that may be called through a debugger, just
  /// invokes print on stderr.
  void dump() const;

  /// convertInitializerTo - This virtual function is a simple call-back
  /// function that should be overridden to call the appropriate
  /// RecTy::convertValue method.
  ///
  virtual Init *convertInitializerTo(RecTy *Ty) = 0;

  /// convertInitializerBitRange - This method is used to implement the bitrange
  /// selection operator.  Given an initializer, it selects the specified bits
  /// out, returning them as a new init of bits type.  If it is not legal to use
  /// the bit subscript operator on this initializer, return null.
  ///
  virtual Init *convertInitializerBitRange(const std::vector<unsigned> &Bits) {
    return 0;
  }

  /// convertInitListSlice - This method is used to implement the list slice
  /// selection operator.  Given an initializer, it selects the specified list
  /// elements, returning them as a new init of list type.  If it is not legal
  /// to take a slice of this, return null.
  ///
  virtual Init *convertInitListSlice(const std::vector<unsigned> &Elements) {
    return 0;
  }

  /// getFieldType - This method is used to implement the FieldInit class.
  /// Implementors of this method should return the type of the named field if
  /// they are of record type.
  ///
  virtual RecTy *getFieldType(const std::string &FieldName) const { return 0; }

  /// getFieldInit - This method complements getFieldType to return the
  /// initializer for the specified field.  If getFieldType returns non-null
  /// this method should return non-null, otherwise it returns null.
  ///
  virtual Init *getFieldInit(Record &R, const std::string &FieldName) const {
    return 0;
  }

  /// resolveReferences - This method is used by classes that refer to other
  /// variables which may not be defined at the time they expression is formed.
  /// If a value is set for the variable later, this method will be called on
  /// users of the value to allow the value to propagate out.
  ///
  virtual Init *resolveReferences(Record &R, const RecordVal *RV) {
    return this;
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const Init &I) {
  I.print(OS); return OS;
}

/// TypedInit - This is the common super-class of types that have a specific,
/// explicit, type.
///
class TypedInit : public Init {
  RecTy *Ty;
public:
  explicit TypedInit(RecTy *T) : Ty(T) {}

  RecTy *getType() const { return Ty; }

  virtual Init *convertInitializerBitRange(const std::vector<unsigned> &Bits);
  virtual Init *convertInitListSlice(const std::vector<unsigned> &Elements);

  /// resolveBitReference - This method is used to implement
  /// VarBitInit::resolveReferences.  If the bit is able to be resolved, we
  /// simply return the resolved value, otherwise we return null.
  ///
  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit) = 0;

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt) = 0;
};


/// UnsetInit - ? - Represents an uninitialized value
///
class UnsetInit : public Init {
public:
  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  virtual bool isComplete() const { return false; }
  virtual std::string getAsString() const { return "?"; }
};


/// BitInit - true/false - Represent a concrete initializer for a bit.
///
class BitInit : public Init {
  bool Value;
public:
  explicit BitInit(bool V) : Value(V) {}

  bool getValue() const { return Value; }

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  virtual std::string getAsString() const { return Value ? "1" : "0"; }
};

/// BitsInit - { a, b, c } - Represents an initializer for a BitsRecTy value.
/// It contains a vector of bits, whose size is determined by the type.
///
class BitsInit : public Init {
  std::vector<Init*> Bits;
public:
  explicit BitsInit(unsigned Size) : Bits(Size) {}

  unsigned getNumBits() const { return Bits.size(); }

  Init *getBit(unsigned Bit) const {
    assert(Bit < Bits.size() && "Bit index out of range!");
    return Bits[Bit];
  }
  void setBit(unsigned Bit, Init *V) {
    assert(Bit < Bits.size() && "Bit index out of range!");
    assert(Bits[Bit] == 0 && "Bit already set!");
    Bits[Bit] = V;
  }

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }
  virtual Init *convertInitializerBitRange(const std::vector<unsigned> &Bits);

  virtual bool isComplete() const {
    for (unsigned i = 0; i != getNumBits(); ++i)
      if (!getBit(i)->isComplete()) return false;
    return true;
  }
  virtual std::string getAsString() const;

  virtual Init *resolveReferences(Record &R, const RecordVal *RV);

  // printXX - Print this bitstream with the specified format, returning true if
  // it is not possible.
  bool printInHex(raw_ostream &OS) const;
  bool printAsVariable(raw_ostream &OS) const;
  bool printAsUnset(raw_ostream &OS) const;
};


/// IntInit - 7 - Represent an initalization by a literal integer value.
///
class IntInit : public TypedInit {
  int64_t Value;
public:
  explicit IntInit(int64_t V) : TypedInit(new IntRecTy), Value(V) {}

  int64_t getValue() const { return Value; }

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }
  virtual Init *convertInitializerBitRange(const std::vector<unsigned> &Bits);

  virtual std::string getAsString() const;

  /// resolveBitReference - This method is used to implement
  /// VarBitInit::resolveReferences.  If the bit is able to be resolved, we
  /// simply return the resolved value, otherwise we return null.
  ///
  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit) {
    assert(0 && "Illegal bit reference off int");
    return 0;
  }

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt) {
    assert(0 && "Illegal element reference off int");
    return 0;
  }
};


/// StringInit - "foo" - Represent an initialization by a string value.
///
class StringInit : public TypedInit {
  std::string Value;
public:
  explicit StringInit(const std::string &V)
    : TypedInit(new StringRecTy), Value(V) {}

  const std::string &getValue() const { return Value; }

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  virtual std::string getAsString() const { return "\"" + Value + "\""; }

  /// resolveBitReference - This method is used to implement
  /// VarBitInit::resolveReferences.  If the bit is able to be resolved, we
  /// simply return the resolved value, otherwise we return null.
  ///
  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit) {
    assert(0 && "Illegal bit reference off string");
    return 0;
  }

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt) {
    assert(0 && "Illegal element reference off string");
    return 0;
  }
};

/// CodeInit - "[{...}]" - Represent a code fragment.
///
class CodeInit : public Init {
  std::string Value;
public:
  explicit CodeInit(const std::string &V) : Value(V) {}

  const std::string getValue() const { return Value; }

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  virtual std::string getAsString() const { return "[{" + Value + "}]"; }
};

/// ListInit - [AL, AH, CL] - Represent a list of defs
///
class ListInit : public TypedInit {
  std::vector<Init*> Values;
public:
  typedef std::vector<Init*>::iterator       iterator;
  typedef std::vector<Init*>::const_iterator const_iterator;

  explicit ListInit(std::vector<Init*> &Vs, RecTy *EltTy)
    : TypedInit(new ListRecTy(EltTy)) {
    Values.swap(Vs);
  }
  explicit ListInit(iterator Start, iterator End, RecTy *EltTy)
      : TypedInit(new ListRecTy(EltTy)), Values(Start, End) {}

  unsigned getSize() const { return Values.size(); }
  Init *getElement(unsigned i) const {
    assert(i < Values.size() && "List element index out of range!");
    return Values[i];
  }

  Record *getElementAsRecord(unsigned i) const;
  
  Init *convertInitListSlice(const std::vector<unsigned> &Elements);

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  /// resolveReferences - This method is used by classes that refer to other
  /// variables which may not be defined at the time they expression is formed.
  /// If a value is set for the variable later, this method will be called on
  /// users of the value to allow the value to propagate out.
  ///
  virtual Init *resolveReferences(Record &R, const RecordVal *RV);

  virtual std::string getAsString() const;

  inline iterator       begin()       { return Values.begin(); }
  inline const_iterator begin() const { return Values.begin(); }
  inline iterator       end  ()       { return Values.end();   }
  inline const_iterator end  () const { return Values.end();   }

  inline size_t         size () const { return Values.size();  }
  inline bool           empty() const { return Values.empty(); }

  /// resolveBitReference - This method is used to implement
  /// VarBitInit::resolveReferences.  If the bit is able to be resolved, we
  /// simply return the resolved value, otherwise we return null.
  ///
  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit) {
    assert(0 && "Illegal bit reference off list");
    return 0;
  }

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt);
};


/// OpInit - Base class for operators
///
class OpInit : public TypedInit {
public:
  OpInit(RecTy *Type) : TypedInit(Type) {}

  // Clone - Clone this operator, replacing arguments with the new list
  virtual OpInit *clone(std::vector<Init *> &Operands) = 0;

  virtual int getNumOperands() const = 0;
  virtual Init *getOperand(int i) = 0;

  // Fold - If possible, fold this to a simpler init.  Return this if not
  // possible to fold.
  virtual Init *Fold(Record *CurRec, MultiClass *CurMultiClass) = 0;

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }
  
  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit);
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt);
};


/// UnOpInit - !op (X) - Transform an init.
///
class UnOpInit : public OpInit {
public:
  enum UnaryOp { CAST, CAR, CDR, LNULL };
private:
  UnaryOp Opc;
  Init *LHS;
public:
  UnOpInit(UnaryOp opc, Init *lhs, RecTy *Type) :
      OpInit(Type), Opc(opc), LHS(lhs) {
  }

  // Clone - Clone this operator, replacing arguments with the new list
  virtual OpInit *clone(std::vector<Init *> &Operands) {
    assert(Operands.size() == 1 &&
           "Wrong number of operands for unary operation");
    return new UnOpInit(getOpcode(), *Operands.begin(), getType());
  }

  int getNumOperands() const { return 1; }
  Init *getOperand(int i) {
    assert(i == 0 && "Invalid operand id for unary operator");
    return getOperand();
  }
  
  UnaryOp getOpcode() const { return Opc; }
  Init *getOperand() const { return LHS; }

  // Fold - If possible, fold this to a simpler init.  Return this if not
  // possible to fold.
  Init *Fold(Record *CurRec, MultiClass *CurMultiClass);

  virtual Init *resolveReferences(Record &R, const RecordVal *RV);
  
  /// getFieldType - This method is used to implement the FieldInit class.
  /// Implementors of this method should return the type of the named field if
  /// they are of record type.
  ///
  virtual RecTy *getFieldType(const std::string &FieldName) const;

  virtual std::string getAsString() const;
};

/// BinOpInit - !op (X, Y) - Combine two inits.
///
class BinOpInit : public OpInit {
public:
  enum BinaryOp { SHL, SRA, SRL, STRCONCAT, CONCAT, NAMECONCAT };
private:
  BinaryOp Opc;
  Init *LHS, *RHS;
public:
  BinOpInit(BinaryOp opc, Init *lhs, Init *rhs, RecTy *Type) :
      OpInit(Type), Opc(opc), LHS(lhs), RHS(rhs) {
  }
  
  // Clone - Clone this operator, replacing arguments with the new list
  virtual OpInit *clone(std::vector<Init *> &Operands) {
    assert(Operands.size() == 2 &&
           "Wrong number of operands for binary operation");
    return new BinOpInit(getOpcode(), Operands[0], Operands[1], getType());
  }

  int getNumOperands() const { return 2; }
  Init *getOperand(int i) {
    assert((i == 0 || i == 1) && "Invalid operand id for binary operator");
    if (i == 0) {
      return getLHS();
    }
    else {
      return getRHS();
    }
  }

  BinaryOp getOpcode() const { return Opc; }
  Init *getLHS() const { return LHS; }
  Init *getRHS() const { return RHS; }

  // Fold - If possible, fold this to a simpler init.  Return this if not
  // possible to fold.
  Init *Fold(Record *CurRec, MultiClass *CurMultiClass);

  virtual Init *resolveReferences(Record &R, const RecordVal *RV);
  
  virtual std::string getAsString() const;
};

/// TernOpInit - !op (X, Y, Z) - Combine two inits.
///
class TernOpInit : public OpInit {
public:
  enum TernaryOp { SUBST, FOREACH, IF };
private:
  TernaryOp Opc;
  Init *LHS, *MHS, *RHS;
public:
  TernOpInit(TernaryOp opc, Init *lhs, Init *mhs, Init *rhs, RecTy *Type) :
      OpInit(Type), Opc(opc), LHS(lhs), MHS(mhs), RHS(rhs) {
  }
  
  // Clone - Clone this operator, replacing arguments with the new list
  virtual OpInit *clone(std::vector<Init *> &Operands) {
    assert(Operands.size() == 3 &&
           "Wrong number of operands for ternary operation");
    return new TernOpInit(getOpcode(), Operands[0], Operands[1], Operands[2],
                          getType());
  }

  int getNumOperands() const { return 3; }
  Init *getOperand(int i) {
    assert((i == 0 || i == 1 || i == 2) &&
           "Invalid operand id for ternary operator");
    if (i == 0) {
      return getLHS();
    }
    else if (i == 1) {
      return getMHS();
    }
    else {
      return getRHS();
    }
  }

  TernaryOp getOpcode() const { return Opc; }
  Init *getLHS() const { return LHS; }
  Init *getMHS() const { return MHS; }
  Init *getRHS() const { return RHS; }

  // Fold - If possible, fold this to a simpler init.  Return this if not
  // possible to fold.
  Init *Fold(Record *CurRec, MultiClass *CurMultiClass);
  
  virtual Init *resolveReferences(Record &R, const RecordVal *RV);
  
  virtual std::string getAsString() const;
};


/// VarInit - 'Opcode' - Represent a reference to an entire variable object.
///
class VarInit : public TypedInit {
  std::string VarName;
public:
  explicit VarInit(const std::string &VN, RecTy *T)
    : TypedInit(T), VarName(VN) {}

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  const std::string &getName() const { return VarName; }

  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit);
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt);

  virtual RecTy *getFieldType(const std::string &FieldName) const;
  virtual Init *getFieldInit(Record &R, const std::string &FieldName) const;

  /// resolveReferences - This method is used by classes that refer to other
  /// variables which may not be defined at the time they expression is formed.
  /// If a value is set for the variable later, this method will be called on
  /// users of the value to allow the value to propagate out.
  ///
  virtual Init *resolveReferences(Record &R, const RecordVal *RV);

  virtual std::string getAsString() const { return VarName; }
};


/// VarBitInit - Opcode{0} - Represent access to one bit of a variable or field.
///
class VarBitInit : public Init {
  TypedInit *TI;
  unsigned Bit;
public:
  VarBitInit(TypedInit *T, unsigned B) : TI(T), Bit(B) {
    assert(T->getType() && dynamic_cast<BitsRecTy*>(T->getType()) &&
           ((BitsRecTy*)T->getType())->getNumBits() > B &&
           "Illegal VarBitInit expression!");
  }

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  TypedInit *getVariable() const { return TI; }
  unsigned getBitNum() const { return Bit; }

  virtual std::string getAsString() const;
  virtual Init *resolveReferences(Record &R, const RecordVal *RV);
};

/// VarListElementInit - List[4] - Represent access to one element of a var or
/// field.
class VarListElementInit : public TypedInit {
  TypedInit *TI;
  unsigned Element;
public:
  VarListElementInit(TypedInit *T, unsigned E)
    : TypedInit(dynamic_cast<ListRecTy*>(T->getType())->getElementType()),
                TI(T), Element(E) {
    assert(T->getType() && dynamic_cast<ListRecTy*>(T->getType()) &&
           "Illegal VarBitInit expression!");
  }

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  TypedInit *getVariable() const { return TI; }
  unsigned getElementNum() const { return Element; }

  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit);

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt);

  virtual std::string getAsString() const;
  virtual Init *resolveReferences(Record &R, const RecordVal *RV);
};

/// DefInit - AL - Represent a reference to a 'def' in the description
///
class DefInit : public TypedInit {
  Record *Def;
public:
  explicit DefInit(Record *D) : TypedInit(new RecordRecTy(D)), Def(D) {}

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  Record *getDef() const { return Def; }

  //virtual Init *convertInitializerBitRange(const std::vector<unsigned> &Bits);

  virtual RecTy *getFieldType(const std::string &FieldName) const;
  virtual Init *getFieldInit(Record &R, const std::string &FieldName) const;

  virtual std::string getAsString() const;

  /// resolveBitReference - This method is used to implement
  /// VarBitInit::resolveReferences.  If the bit is able to be resolved, we
  /// simply return the resolved value, otherwise we return null.
  ///
  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit) {
    assert(0 && "Illegal bit reference off def");
    return 0;
  }

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt) {
    assert(0 && "Illegal element reference off def");
    return 0;
  }
};


/// FieldInit - X.Y - Represent a reference to a subfield of a variable
///
class FieldInit : public TypedInit {
  Init *Rec;                // Record we are referring to
  std::string FieldName;    // Field we are accessing
public:
  FieldInit(Init *R, const std::string &FN)
    : TypedInit(R->getFieldType(FN)), Rec(R), FieldName(FN) {
    assert(getType() && "FieldInit with non-record type!");
  }

  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit);
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt);

  virtual Init *resolveReferences(Record &R, const RecordVal *RV);

  virtual std::string getAsString() const {
    return Rec->getAsString() + "." + FieldName;
  }
};

/// DagInit - (v a, b) - Represent a DAG tree value.  DAG inits are required
/// to have at least one value then a (possibly empty) list of arguments.  Each
/// argument can have a name associated with it.
///
class DagInit : public TypedInit {
  Init *Val;
  std::string ValName;
  std::vector<Init*> Args;
  std::vector<std::string> ArgNames;
public:
  DagInit(Init *V, std::string VN, 
          const std::vector<std::pair<Init*, std::string> > &args)
    : TypedInit(new DagRecTy), Val(V), ValName(VN) {
    Args.reserve(args.size());
    ArgNames.reserve(args.size());
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      Args.push_back(args[i].first);
      ArgNames.push_back(args[i].second);
    }
  }
  DagInit(Init *V, std::string VN, const std::vector<Init*> &args, 
          const std::vector<std::string> &argNames)
  : TypedInit(new DagRecTy), Val(V), ValName(VN), Args(args), ArgNames(argNames) {
  }
  
  virtual Init *convertInitializerTo(RecTy *Ty) {
    return Ty->convertValue(this);
  }

  Init *getOperator() const { return Val; }

  const std::string &getName() const { return ValName; }

  unsigned getNumArgs() const { return Args.size(); }
  Init *getArg(unsigned Num) const {
    assert(Num < Args.size() && "Arg number out of range!");
    return Args[Num];
  }
  const std::string &getArgName(unsigned Num) const {
    assert(Num < ArgNames.size() && "Arg number out of range!");
    return ArgNames[Num];
  }

  void setArg(unsigned Num, Init *I) {
    assert(Num < Args.size() && "Arg number out of range!");
    Args[Num] = I;
  }
  
  virtual Init *resolveReferences(Record &R, const RecordVal *RV);

  virtual std::string getAsString() const;

  typedef std::vector<Init*>::iterator             arg_iterator;
  typedef std::vector<Init*>::const_iterator       const_arg_iterator;
  typedef std::vector<std::string>::iterator       name_iterator;
  typedef std::vector<std::string>::const_iterator const_name_iterator;

  inline arg_iterator        arg_begin()       { return Args.begin(); }
  inline const_arg_iterator  arg_begin() const { return Args.begin(); }
  inline arg_iterator        arg_end  ()       { return Args.end();   }
  inline const_arg_iterator  arg_end  () const { return Args.end();   }

  inline size_t              arg_size () const { return Args.size();  }
  inline bool                arg_empty() const { return Args.empty(); }

  inline name_iterator       name_begin()       { return ArgNames.begin(); }
  inline const_name_iterator name_begin() const { return ArgNames.begin(); }
  inline name_iterator       name_end  ()       { return ArgNames.end();   }
  inline const_name_iterator name_end  () const { return ArgNames.end();   }

  inline size_t              name_size () const { return ArgNames.size();  }
  inline bool                name_empty() const { return ArgNames.empty(); }

  virtual Init *resolveBitReference(Record &R, const RecordVal *RV,
                                    unsigned Bit) {
    assert(0 && "Illegal bit reference off dag");
    return 0;
  }
  
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt) {
    assert(0 && "Illegal element reference off dag");
    return 0;
  }
  
};

//===----------------------------------------------------------------------===//
//  High-Level Classes
//===----------------------------------------------------------------------===//

class RecordVal {
  std::string Name;
  RecTy *Ty;
  unsigned Prefix;
  Init *Value;
public:
  RecordVal(const std::string &N, RecTy *T, unsigned P);

  const std::string &getName() const { return Name; }

  unsigned getPrefix() const { return Prefix; }
  RecTy *getType() const { return Ty; }
  Init *getValue() const { return Value; }

  bool setValue(Init *V) {
    if (V) {
      Value = V->convertInitializerTo(Ty);
      return Value == 0;
    }
    Value = 0;
    return false;
  }

  void dump() const;
  void print(raw_ostream &OS, bool PrintSem = true) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const RecordVal &RV) {
  RV.print(OS << "  ");
  return OS;
}

class Record {
  static unsigned LastID;

  // Unique record ID.
  unsigned ID;
  std::string Name;
  SMLoc Loc;
  std::vector<std::string> TemplateArgs;
  std::vector<RecordVal> Values;
  std::vector<Record*> SuperClasses;
public:

  explicit Record(const std::string &N, SMLoc loc) : 
    ID(LastID++), Name(N), Loc(loc) {}
  ~Record() {}
  
  unsigned getID() const { return ID; }

  const std::string &getName() const { return Name; }
  void setName(const std::string &Name);  // Also updates RecordKeeper.
  
  SMLoc getLoc() const { return Loc; }
  
  const std::vector<std::string> &getTemplateArgs() const {
    return TemplateArgs;
  }
  const std::vector<RecordVal> &getValues() const { return Values; }
  const std::vector<Record*>   &getSuperClasses() const { return SuperClasses; }

  bool isTemplateArg(const std::string &Name) const {
    for (unsigned i = 0, e = TemplateArgs.size(); i != e; ++i)
      if (TemplateArgs[i] == Name) return true;
    return false;
  }

  const RecordVal *getValue(const std::string &Name) const {
    for (unsigned i = 0, e = Values.size(); i != e; ++i)
      if (Values[i].getName() == Name) return &Values[i];
    return 0;
  }
  RecordVal *getValue(const std::string &Name) {
    for (unsigned i = 0, e = Values.size(); i != e; ++i)
      if (Values[i].getName() == Name) return &Values[i];
    return 0;
  }

  void addTemplateArg(const std::string &Name) {
    assert(!isTemplateArg(Name) && "Template arg already defined!");
    TemplateArgs.push_back(Name);
  }

  void addValue(const RecordVal &RV) {
    assert(getValue(RV.getName()) == 0 && "Value already added!");
    Values.push_back(RV);
  }

  void removeValue(const std::string &Name) {
    assert(getValue(Name) && "Cannot remove an entry that does not exist!");
    for (unsigned i = 0, e = Values.size(); i != e; ++i)
      if (Values[i].getName() == Name) {
        Values.erase(Values.begin()+i);
        return;
      }
    assert(0 && "Name does not exist in record!");
  }

  bool isSubClassOf(const Record *R) const {
    for (unsigned i = 0, e = SuperClasses.size(); i != e; ++i)
      if (SuperClasses[i] == R)
        return true;
    return false;
  }

  bool isSubClassOf(const std::string &Name) const {
    for (unsigned i = 0, e = SuperClasses.size(); i != e; ++i)
      if (SuperClasses[i]->getName() == Name)
        return true;
    return false;
  }

  void addSuperClass(Record *R) {
    assert(!isSubClassOf(R) && "Already subclassing record!");
    SuperClasses.push_back(R);
  }

  /// resolveReferences - If there are any field references that refer to fields
  /// that have been filled in, we can propagate the values now.
  ///
  void resolveReferences() { resolveReferencesTo(0); }

  /// resolveReferencesTo - If anything in this record refers to RV, replace the
  /// reference to RV with the RHS of RV.  If RV is null, we resolve all
  /// possible references.
  void resolveReferencesTo(const RecordVal *RV);

  void dump() const;

  //===--------------------------------------------------------------------===//
  // High-level methods useful to tablegen back-ends
  //

  /// getValueInit - Return the initializer for a value with the specified name,
  /// or throw an exception if the field does not exist.
  ///
  Init *getValueInit(const std::string &FieldName) const;

  /// getValueAsString - This method looks up the specified field and returns
  /// its value as a string, throwing an exception if the field does not exist
  /// or if the value is not a string.
  ///
  std::string getValueAsString(const std::string &FieldName) const;

  /// getValueAsBitsInit - This method looks up the specified field and returns
  /// its value as a BitsInit, throwing an exception if the field does not exist
  /// or if the value is not the right type.
  ///
  BitsInit *getValueAsBitsInit(const std::string &FieldName) const;

  /// getValueAsListInit - This method looks up the specified field and returns
  /// its value as a ListInit, throwing an exception if the field does not exist
  /// or if the value is not the right type.
  ///
  ListInit *getValueAsListInit(const std::string &FieldName) const;

  /// getValueAsListOfDefs - This method looks up the specified field and
  /// returns its value as a vector of records, throwing an exception if the
  /// field does not exist or if the value is not the right type.
  ///
  std::vector<Record*> getValueAsListOfDefs(const std::string &FieldName) const;

  /// getValueAsListOfInts - This method looks up the specified field and returns
  /// its value as a vector of integers, throwing an exception if the field does
  /// not exist or if the value is not the right type.
  ///
  std::vector<int64_t> getValueAsListOfInts(const std::string &FieldName) const;
  
  /// getValueAsDef - This method looks up the specified field and returns its
  /// value as a Record, throwing an exception if the field does not exist or if
  /// the value is not the right type.
  ///
  Record *getValueAsDef(const std::string &FieldName) const;

  /// getValueAsBit - This method looks up the specified field and returns its
  /// value as a bit, throwing an exception if the field does not exist or if
  /// the value is not the right type.
  ///
  bool getValueAsBit(const std::string &FieldName) const;

  /// getValueAsInt - This method looks up the specified field and returns its
  /// value as an int64_t, throwing an exception if the field does not exist or
  /// if the value is not the right type.
  ///
  int64_t getValueAsInt(const std::string &FieldName) const;

  /// getValueAsDag - This method looks up the specified field and returns its
  /// value as an Dag, throwing an exception if the field does not exist or if
  /// the value is not the right type.
  ///
  DagInit *getValueAsDag(const std::string &FieldName) const;
  
  /// getValueAsCode - This method looks up the specified field and returns
  /// its value as the string data in a CodeInit, throwing an exception if the
  /// field does not exist or if the value is not a code object.
  ///
  std::string getValueAsCode(const std::string &FieldName) const;
};

raw_ostream &operator<<(raw_ostream &OS, const Record &R);

struct MultiClass {
  Record Rec;  // Placeholder for template args and Name.
  typedef std::vector<Record*> RecordVector;
  RecordVector DefPrototypes;

  void dump() const;

  MultiClass(const std::string &Name, SMLoc Loc) : Rec(Name, Loc) {}
};

class RecordKeeper {
  std::map<std::string, Record*> Classes, Defs;
public:
  ~RecordKeeper() {
    for (std::map<std::string, Record*>::iterator I = Classes.begin(),
           E = Classes.end(); I != E; ++I)
      delete I->second;
    for (std::map<std::string, Record*>::iterator I = Defs.begin(),
           E = Defs.end(); I != E; ++I)
      delete I->second;
  }

  const std::map<std::string, Record*> &getClasses() const { return Classes; }
  const std::map<std::string, Record*> &getDefs() const { return Defs; }

  Record *getClass(const std::string &Name) const {
    std::map<std::string, Record*>::const_iterator I = Classes.find(Name);
    return I == Classes.end() ? 0 : I->second;
  }
  Record *getDef(const std::string &Name) const {
    std::map<std::string, Record*>::const_iterator I = Defs.find(Name);
    return I == Defs.end() ? 0 : I->second;
  }
  void addClass(Record *R) {
    assert(getClass(R->getName()) == 0 && "Class already exists!");
    Classes.insert(std::make_pair(R->getName(), R));
  }
  void addDef(Record *R) {
    assert(getDef(R->getName()) == 0 && "Def already exists!");
    Defs.insert(std::make_pair(R->getName(), R));
  }

  /// removeClass - Remove, but do not delete, the specified record.
  ///
  void removeClass(const std::string &Name) {
    assert(Classes.count(Name) && "Class does not exist!");
    Classes.erase(Name);
  }
  /// removeDef - Remove, but do not delete, the specified record.
  ///
  void removeDef(const std::string &Name) {
    assert(Defs.count(Name) && "Def does not exist!");
    Defs.erase(Name);
  }
  
  //===--------------------------------------------------------------------===//
  // High-level helper methods, useful for tablegen backends...

  /// getAllDerivedDefinitions - This method returns all concrete definitions
  /// that derive from the specified class name.  If a class with the specified
  /// name does not exist, an exception is thrown.
  std::vector<Record*>
  getAllDerivedDefinitions(const std::string &ClassName) const;


  void dump() const;
};

/// LessRecord - Sorting predicate to sort record pointers by name.
///
struct LessRecord {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getName() < Rec2->getName();
  }
};

/// LessRecordFieldName - Sorting predicate to sort record pointers by their 
/// name field.
///
struct LessRecordFieldName {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getValueAsString("Name") < Rec2->getValueAsString("Name");
  }
};


class TGError {
  SMLoc Loc;
  std::string Message;
public:
  TGError(SMLoc loc, const std::string &message) : Loc(loc), Message(message) {}
  
  SMLoc getLoc() const { return Loc; }
  const std::string &getMessage() const { return Message; }
};
  
  
raw_ostream &operator<<(raw_ostream &OS, const RecordKeeper &RK);

extern RecordKeeper Records;

void PrintError(SMLoc ErrorLoc, const std::string &Msg);

  
} // End llvm namespace

#endif
