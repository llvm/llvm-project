//===-- Constants.cpp - Implement Constant nodes --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Constant* classes...
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "ConstantFold.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <map>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                              Constant Class
//===----------------------------------------------------------------------===//

void Constant::destroyConstantImpl() {
  // When a Constant is destroyed, there may be lingering
  // references to the constant by other constants in the constant pool.  These
  // constants are implicitly dependent on the module that is being deleted,
  // but they don't know that.  Because we only find out when the CPV is
  // deleted, we must now notify all of our users (that should only be
  // Constants) that they are, in fact, invalid now and should be deleted.
  //
  while (!use_empty()) {
    Value *V = use_back();
#ifndef NDEBUG      // Only in -g mode...
    if (!isa<Constant>(V))
      DOUT << "While deleting: " << *this
           << "\n\nUse still stuck around after Def is destroyed: "
           << *V << "\n\n";
#endif
    assert(isa<Constant>(V) && "References remain to Constant being destroyed");
    Constant *CV = cast<Constant>(V);
    CV->destroyConstant();

    // The constant should remove itself from our use list...
    assert((use_empty() || use_back() != V) && "Constant not removed!");
  }

  // Value has no outstanding references it is safe to delete it now...
  delete this;
}

/// canTrap - Return true if evaluation of this constant could trap.  This is
/// true for things like constant expressions that could divide by zero.
bool Constant::canTrap() const {
  assert(getType()->isFirstClassType() && "Cannot evaluate aggregate vals!");
  // The only thing that could possibly trap are constant exprs.
  const ConstantExpr *CE = dyn_cast<ConstantExpr>(this);
  if (!CE) return false;
  
  // ConstantExpr traps if any operands can trap. 
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (getOperand(i)->canTrap()) 
      return true;

  // Otherwise, only specific operations can trap.
  switch (CE->getOpcode()) {
  default:
    return false;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
    // Div and rem can trap if the RHS is not known to be non-zero.
    if (!isa<ConstantInt>(getOperand(1)) || getOperand(1)->isNullValue())
      return true;
    return false;
  }
}

/// ContaintsRelocations - Return true if the constant value contains
/// relocations which cannot be resolved at compile time.
bool Constant::ContainsRelocations() const {
  if (isa<GlobalValue>(this))
    return true;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (getOperand(i)->ContainsRelocations())
      return true;
  return false;
}

// Static constructor to create a '0' constant of arbitrary type...
Constant *Constant::getNullValue(const Type *Ty) {
  static uint64_t zero[2] = {0, 0};
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID:
    return ConstantInt::get(Ty, 0);
  case Type::FloatTyID:
    return ConstantFP::get(APFloat(APInt(32, 0)));
  case Type::DoubleTyID:
    return ConstantFP::get(APFloat(APInt(64, 0)));
  case Type::X86_FP80TyID:
    return ConstantFP::get(APFloat(APInt(80, 2, zero)));
  case Type::FP128TyID:
    return ConstantFP::get(APFloat(APInt(128, 2, zero), true));
  case Type::PPC_FP128TyID:
    return ConstantFP::get(APFloat(APInt(128, 2, zero)));
  case Type::PointerTyID:
    return ConstantPointerNull::get(cast<PointerType>(Ty));
  case Type::StructTyID:
  case Type::ArrayTyID:
  case Type::VectorTyID:
    return ConstantAggregateZero::get(Ty);
  default:
    // Function, Label, or Opaque type?
    assert(!"Cannot create a null constant of that type!");
    return 0;
  }
}

Constant *Constant::getAllOnesValue(const Type *Ty) {
  if (const IntegerType* ITy = dyn_cast<IntegerType>(Ty))
    return ConstantInt::get(APInt::getAllOnesValue(ITy->getBitWidth()));
  return ConstantVector::getAllOnesValue(cast<VectorType>(Ty));
}

// Static constructor to create an integral constant with all bits set
ConstantInt *ConstantInt::getAllOnesValue(const Type *Ty) {
  if (const IntegerType* ITy = dyn_cast<IntegerType>(Ty))
    return ConstantInt::get(APInt::getAllOnesValue(ITy->getBitWidth()));
  return 0;
}

/// @returns the value for a vector integer constant of the given type that
/// has all its bits set to true.
/// @brief Get the all ones value
ConstantVector *ConstantVector::getAllOnesValue(const VectorType *Ty) {
  std::vector<Constant*> Elts;
  Elts.resize(Ty->getNumElements(),
              ConstantInt::getAllOnesValue(Ty->getElementType()));
  assert(Elts[0] && "Not a vector integer type!");
  return cast<ConstantVector>(ConstantVector::get(Elts));
}


/// getVectorElements - This method, which is only valid on constant of vector
/// type, returns the elements of the vector in the specified smallvector.
/// This handles breaking down a vector undef into undef elements, etc.  For
/// constant exprs and other cases we can't handle, we return an empty vector.
void Constant::getVectorElements(SmallVectorImpl<Constant*> &Elts) const {
  assert(isa<VectorType>(getType()) && "Not a vector constant!");
  
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(this)) {
    for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i)
      Elts.push_back(CV->getOperand(i));
    return;
  }
  
  const VectorType *VT = cast<VectorType>(getType());
  if (isa<ConstantAggregateZero>(this)) {
    Elts.assign(VT->getNumElements(), 
                Constant::getNullValue(VT->getElementType()));
    return;
  }
  
  if (isa<UndefValue>(this)) {
    Elts.assign(VT->getNumElements(), UndefValue::get(VT->getElementType()));
    return;
  }
  
  // Unknown type, must be constant expr etc.
}



//===----------------------------------------------------------------------===//
//                                ConstantInt
//===----------------------------------------------------------------------===//

ConstantInt::ConstantInt(const IntegerType *Ty, const APInt& V)
  : Constant(Ty, ConstantIntVal, 0, 0), Val(V) {
  assert(V.getBitWidth() == Ty->getBitWidth() && "Invalid constant for type");
}

ConstantInt *ConstantInt::TheTrueVal = 0;
ConstantInt *ConstantInt::TheFalseVal = 0;

namespace llvm {
  void CleanupTrueFalse(void *) {
    ConstantInt::ResetTrueFalse();
  }
}

static ManagedCleanup<llvm::CleanupTrueFalse> TrueFalseCleanup;

ConstantInt *ConstantInt::CreateTrueFalseVals(bool WhichOne) {
  assert(TheTrueVal == 0 && TheFalseVal == 0);
  TheTrueVal  = get(Type::Int1Ty, 1);
  TheFalseVal = get(Type::Int1Ty, 0);
  
  // Ensure that llvm_shutdown nulls out TheTrueVal/TheFalseVal.
  TrueFalseCleanup.Register();
  
  return WhichOne ? TheTrueVal : TheFalseVal;
}


namespace {
  struct DenseMapAPIntKeyInfo {
    struct KeyTy {
      APInt val;
      const Type* type;
      KeyTy(const APInt& V, const Type* Ty) : val(V), type(Ty) {}
      KeyTy(const KeyTy& that) : val(that.val), type(that.type) {}
      bool operator==(const KeyTy& that) const {
        return type == that.type && this->val == that.val;
      }
      bool operator!=(const KeyTy& that) const {
        return !this->operator==(that);
      }
    };
    static inline KeyTy getEmptyKey() { return KeyTy(APInt(1,0), 0); }
    static inline KeyTy getTombstoneKey() { return KeyTy(APInt(1,1), 0); }
    static unsigned getHashValue(const KeyTy &Key) {
      return DenseMapInfo<void*>::getHashValue(Key.type) ^ 
        Key.val.getHashValue();
    }
    static bool isEqual(const KeyTy &LHS, const KeyTy &RHS) {
      return LHS == RHS;
    }
    static bool isPod() { return false; }
  };
}


typedef DenseMap<DenseMapAPIntKeyInfo::KeyTy, ConstantInt*, 
                 DenseMapAPIntKeyInfo> IntMapTy;
static ManagedStatic<IntMapTy> IntConstants;

ConstantInt *ConstantInt::get(const Type *Ty, uint64_t V, bool isSigned) {
  const IntegerType *ITy = cast<IntegerType>(Ty);
  return get(APInt(ITy->getBitWidth(), V, isSigned));
}

// Get a ConstantInt from an APInt. Note that the value stored in the DenseMap 
// as the key, is a DenseMapAPIntKeyInfo::KeyTy which has provided the
// operator== and operator!= to ensure that the DenseMap doesn't attempt to
// compare APInt's of different widths, which would violate an APInt class
// invariant which generates an assertion.
ConstantInt *ConstantInt::get(const APInt& V) {
  // Get the corresponding integer type for the bit width of the value.
  const IntegerType *ITy = IntegerType::get(V.getBitWidth());
  // get an existing value or the insertion position
  DenseMapAPIntKeyInfo::KeyTy Key(V, ITy);
  ConstantInt *&Slot = (*IntConstants)[Key]; 
  // if it exists, return it.
  if (Slot)
    return Slot;
  // otherwise create a new one, insert it, and return it.
  return Slot = new ConstantInt(ITy, V);
}

//===----------------------------------------------------------------------===//
//                                ConstantFP
//===----------------------------------------------------------------------===//

static const fltSemantics *TypeToFloatSemantics(const Type *Ty) {
  if (Ty == Type::FloatTy)
    return &APFloat::IEEEsingle;
  if (Ty == Type::DoubleTy)
    return &APFloat::IEEEdouble;
  if (Ty == Type::X86_FP80Ty)
    return &APFloat::x87DoubleExtended;
  else if (Ty == Type::FP128Ty)
    return &APFloat::IEEEquad;
  
  assert(Ty == Type::PPC_FP128Ty && "Unknown FP format");
  return &APFloat::PPCDoubleDouble;
}

ConstantFP::ConstantFP(const Type *Ty, const APFloat& V)
  : Constant(Ty, ConstantFPVal, 0, 0), Val(V) {
  assert(&V.getSemantics() == TypeToFloatSemantics(Ty) &&
         "FP type Mismatch");
}

bool ConstantFP::isNullValue() const {
  return Val.isZero() && !Val.isNegative();
}

ConstantFP *ConstantFP::getNegativeZero(const Type *Ty) {
  APFloat apf = cast <ConstantFP>(Constant::getNullValue(Ty))->getValueAPF();
  apf.changeSign();
  return ConstantFP::get(apf);
}

bool ConstantFP::isExactlyValue(const APFloat& V) const {
  return Val.bitwiseIsEqual(V);
}

namespace {
  struct DenseMapAPFloatKeyInfo {
    struct KeyTy {
      APFloat val;
      KeyTy(const APFloat& V) : val(V){}
      KeyTy(const KeyTy& that) : val(that.val) {}
      bool operator==(const KeyTy& that) const {
        return this->val.bitwiseIsEqual(that.val);
      }
      bool operator!=(const KeyTy& that) const {
        return !this->operator==(that);
      }
    };
    static inline KeyTy getEmptyKey() { 
      return KeyTy(APFloat(APFloat::Bogus,1));
    }
    static inline KeyTy getTombstoneKey() { 
      return KeyTy(APFloat(APFloat::Bogus,2)); 
    }
    static unsigned getHashValue(const KeyTy &Key) {
      return Key.val.getHashValue();
    }
    static bool isEqual(const KeyTy &LHS, const KeyTy &RHS) {
      return LHS == RHS;
    }
    static bool isPod() { return false; }
  };
}

//---- ConstantFP::get() implementation...
//
typedef DenseMap<DenseMapAPFloatKeyInfo::KeyTy, ConstantFP*, 
                 DenseMapAPFloatKeyInfo> FPMapTy;

static ManagedStatic<FPMapTy> FPConstants;

ConstantFP *ConstantFP::get(const APFloat &V) {
  DenseMapAPFloatKeyInfo::KeyTy Key(V);
  ConstantFP *&Slot = (*FPConstants)[Key];
  if (Slot) return Slot;
  
  const Type *Ty;
  if (&V.getSemantics() == &APFloat::IEEEsingle)
    Ty = Type::FloatTy;
  else if (&V.getSemantics() == &APFloat::IEEEdouble)
    Ty = Type::DoubleTy;
  else if (&V.getSemantics() == &APFloat::x87DoubleExtended)
    Ty = Type::X86_FP80Ty;
  else if (&V.getSemantics() == &APFloat::IEEEquad)
    Ty = Type::FP128Ty;
  else {
    assert(&V.getSemantics() == &APFloat::PPCDoubleDouble&&"Unknown FP format");
    Ty = Type::PPC_FP128Ty;
  }
  
  return Slot = new ConstantFP(Ty, V);
}

/// get() - This returns a constant fp for the specified value in the
/// specified type.  This should only be used for simple constant values like
/// 2.0/1.0 etc, that are known-valid both as double and as the target format.
ConstantFP *ConstantFP::get(const Type *Ty, double V) {
  APFloat FV(V);
  bool ignored;
  FV.convert(*TypeToFloatSemantics(Ty), APFloat::rmNearestTiesToEven, &ignored);
  return get(FV);
}

//===----------------------------------------------------------------------===//
//                            ConstantXXX Classes
//===----------------------------------------------------------------------===//


ConstantArray::ConstantArray(const ArrayType *T,
                             const std::vector<Constant*> &V)
  : Constant(T, ConstantArrayVal,
             OperandTraits<ConstantArray>::op_end(this) - V.size(),
             V.size()) {
  assert(V.size() == T->getNumElements() &&
         "Invalid initializer vector for constant array");
  Use *OL = OperandList;
  for (std::vector<Constant*>::const_iterator I = V.begin(), E = V.end();
       I != E; ++I, ++OL) {
    Constant *C = *I;
    assert((C->getType() == T->getElementType() ||
            (T->isAbstract() &&
             C->getType()->getTypeID() == T->getElementType()->getTypeID())) &&
           "Initializer for array element doesn't match array element type!");
    *OL = C;
  }
}


ConstantStruct::ConstantStruct(const StructType *T,
                               const std::vector<Constant*> &V)
  : Constant(T, ConstantStructVal,
             OperandTraits<ConstantStruct>::op_end(this) - V.size(),
             V.size()) {
  assert(V.size() == T->getNumElements() &&
         "Invalid initializer vector for constant structure");
  Use *OL = OperandList;
  for (std::vector<Constant*>::const_iterator I = V.begin(), E = V.end();
       I != E; ++I, ++OL) {
    Constant *C = *I;
    assert((C->getType() == T->getElementType(I-V.begin()) ||
            ((T->getElementType(I-V.begin())->isAbstract() ||
              C->getType()->isAbstract()) &&
             T->getElementType(I-V.begin())->getTypeID() == 
                   C->getType()->getTypeID())) &&
           "Initializer for struct element doesn't match struct element type!");
    *OL = C;
  }
}


ConstantVector::ConstantVector(const VectorType *T,
                               const std::vector<Constant*> &V)
  : Constant(T, ConstantVectorVal,
             OperandTraits<ConstantVector>::op_end(this) - V.size(),
             V.size()) {
  Use *OL = OperandList;
    for (std::vector<Constant*>::const_iterator I = V.begin(), E = V.end();
         I != E; ++I, ++OL) {
      Constant *C = *I;
      assert((C->getType() == T->getElementType() ||
            (T->isAbstract() &&
             C->getType()->getTypeID() == T->getElementType()->getTypeID())) &&
           "Initializer for vector element doesn't match vector element type!");
    *OL = C;
  }
}


namespace llvm {
// We declare several classes private to this file, so use an anonymous
// namespace
namespace {

/// UnaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement unary constant exprs.
class VISIBILITY_HIDDEN UnaryConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  UnaryConstantExpr(unsigned Opcode, Constant *C, const Type *Ty)
    : ConstantExpr(Ty, Opcode, &Op<0>(), 1) {
    Op<0>() = C;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// BinaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement binary constant exprs.
class VISIBILITY_HIDDEN BinaryConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  BinaryConstantExpr(unsigned Opcode, Constant *C1, Constant *C2)
    : ConstantExpr(C1->getType(), Opcode, &Op<0>(), 2) {
    Op<0>() = C1;
    Op<1>() = C2;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// SelectConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement select constant exprs.
class VISIBILITY_HIDDEN SelectConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }
  SelectConstantExpr(Constant *C1, Constant *C2, Constant *C3)
    : ConstantExpr(C2->getType(), Instruction::Select, &Op<0>(), 3) {
    Op<0>() = C1;
    Op<1>() = C2;
    Op<2>() = C3;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// ExtractElementConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// extractelement constant exprs.
class VISIBILITY_HIDDEN ExtractElementConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  ExtractElementConstantExpr(Constant *C1, Constant *C2)
    : ConstantExpr(cast<VectorType>(C1->getType())->getElementType(), 
                   Instruction::ExtractElement, &Op<0>(), 2) {
    Op<0>() = C1;
    Op<1>() = C2;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// InsertElementConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// insertelement constant exprs.
class VISIBILITY_HIDDEN InsertElementConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }
  InsertElementConstantExpr(Constant *C1, Constant *C2, Constant *C3)
    : ConstantExpr(C1->getType(), Instruction::InsertElement, 
                   &Op<0>(), 3) {
    Op<0>() = C1;
    Op<1>() = C2;
    Op<2>() = C3;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// ShuffleVectorConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// shufflevector constant exprs.
class VISIBILITY_HIDDEN ShuffleVectorConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }
  ShuffleVectorConstantExpr(Constant *C1, Constant *C2, Constant *C3)
  : ConstantExpr(C1->getType(), Instruction::ShuffleVector, 
                 &Op<0>(), 3) {
    Op<0>() = C1;
    Op<1>() = C2;
    Op<2>() = C3;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// ExtractValueConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// extractvalue constant exprs.
class VISIBILITY_HIDDEN ExtractValueConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  ExtractValueConstantExpr(Constant *Agg,
                           const SmallVector<unsigned, 4> &IdxList,
                           const Type *DestTy)
    : ConstantExpr(DestTy, Instruction::ExtractValue, &Op<0>(), 1),
      Indices(IdxList) {
    Op<0>() = Agg;
  }

  /// Indices - These identify which value to extract.
  const SmallVector<unsigned, 4> Indices;

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// InsertValueConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// insertvalue constant exprs.
class VISIBILITY_HIDDEN InsertValueConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  InsertValueConstantExpr(Constant *Agg, Constant *Val,
                          const SmallVector<unsigned, 4> &IdxList,
                          const Type *DestTy)
    : ConstantExpr(DestTy, Instruction::InsertValue, &Op<0>(), 2),
      Indices(IdxList) {
    Op<0>() = Agg;
    Op<1>() = Val;
  }

  /// Indices - These identify the position for the insertion.
  const SmallVector<unsigned, 4> Indices;

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};


/// GetElementPtrConstantExpr - This class is private to Constants.cpp, and is
/// used behind the scenes to implement getelementpr constant exprs.
class VISIBILITY_HIDDEN GetElementPtrConstantExpr : public ConstantExpr {
  GetElementPtrConstantExpr(Constant *C, const std::vector<Constant*> &IdxList,
                            const Type *DestTy);
public:
  static GetElementPtrConstantExpr *Create(Constant *C,
                                           const std::vector<Constant*>&IdxList,
                                           const Type *DestTy) {
    return new(IdxList.size() + 1)
      GetElementPtrConstantExpr(C, IdxList, DestTy);
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

// CompareConstantExpr - This class is private to Constants.cpp, and is used
// behind the scenes to implement ICmp and FCmp constant expressions. This is
// needed in order to store the predicate value for these instructions.
struct VISIBILITY_HIDDEN CompareConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  unsigned short predicate;
  CompareConstantExpr(const Type *ty, Instruction::OtherOps opc,
                      unsigned short pred,  Constant* LHS, Constant* RHS)
    : ConstantExpr(ty, opc, &Op<0>(), 2), predicate(pred) {
    Op<0>() = LHS;
    Op<1>() = RHS;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

} // end anonymous namespace

template <>
struct OperandTraits<UnaryConstantExpr> : FixedNumOperandTraits<1> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(UnaryConstantExpr, Value)

template <>
struct OperandTraits<BinaryConstantExpr> : FixedNumOperandTraits<2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(BinaryConstantExpr, Value)

template <>
struct OperandTraits<SelectConstantExpr> : FixedNumOperandTraits<3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(SelectConstantExpr, Value)

template <>
struct OperandTraits<ExtractElementConstantExpr> : FixedNumOperandTraits<2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractElementConstantExpr, Value)

template <>
struct OperandTraits<InsertElementConstantExpr> : FixedNumOperandTraits<3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertElementConstantExpr, Value)

template <>
struct OperandTraits<ShuffleVectorConstantExpr> : FixedNumOperandTraits<3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ShuffleVectorConstantExpr, Value)

template <>
struct OperandTraits<ExtractValueConstantExpr> : FixedNumOperandTraits<1> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractValueConstantExpr, Value)

template <>
struct OperandTraits<InsertValueConstantExpr> : FixedNumOperandTraits<2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertValueConstantExpr, Value)

template <>
struct OperandTraits<GetElementPtrConstantExpr> : VariadicOperandTraits<1> {
};

GetElementPtrConstantExpr::GetElementPtrConstantExpr
  (Constant *C,
   const std::vector<Constant*> &IdxList,
   const Type *DestTy)
    : ConstantExpr(DestTy, Instruction::GetElementPtr,
                   OperandTraits<GetElementPtrConstantExpr>::op_end(this)
                   - (IdxList.size()+1),
                   IdxList.size()+1) {
  OperandList[0] = C;
  for (unsigned i = 0, E = IdxList.size(); i != E; ++i)
    OperandList[i+1] = IdxList[i];
}

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GetElementPtrConstantExpr, Value)


template <>
struct OperandTraits<CompareConstantExpr> : FixedNumOperandTraits<2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CompareConstantExpr, Value)


} // End llvm namespace


// Utility function for determining if a ConstantExpr is a CastOp or not. This
// can't be inline because we don't want to #include Instruction.h into
// Constant.h
bool ConstantExpr::isCast() const {
  return Instruction::isCast(getOpcode());
}

bool ConstantExpr::isCompare() const {
  return getOpcode() == Instruction::ICmp || getOpcode() == Instruction::FCmp ||
         getOpcode() == Instruction::VICmp || getOpcode() == Instruction::VFCmp;
}

bool ConstantExpr::hasIndices() const {
  return getOpcode() == Instruction::ExtractValue ||
         getOpcode() == Instruction::InsertValue;
}

const SmallVector<unsigned, 4> &ConstantExpr::getIndices() const {
  if (const ExtractValueConstantExpr *EVCE =
        dyn_cast<ExtractValueConstantExpr>(this))
    return EVCE->Indices;

  return cast<InsertValueConstantExpr>(this)->Indices;
}

/// ConstantExpr::get* - Return some common constants without having to
/// specify the full Instruction::OPCODE identifier.
///
Constant *ConstantExpr::getNeg(Constant *C) {
  return get(Instruction::Sub,
             ConstantExpr::getZeroValueForNegationExpr(C->getType()),
             C);
}
Constant *ConstantExpr::getNot(Constant *C) {
  assert((isa<IntegerType>(C->getType()) ||
            cast<VectorType>(C->getType())->getElementType()->isInteger()) &&
          "Cannot NOT a nonintegral value!");
  return get(Instruction::Xor, C,
             Constant::getAllOnesValue(C->getType()));
}
Constant *ConstantExpr::getAdd(Constant *C1, Constant *C2) {
  return get(Instruction::Add, C1, C2);
}
Constant *ConstantExpr::getSub(Constant *C1, Constant *C2) {
  return get(Instruction::Sub, C1, C2);
}
Constant *ConstantExpr::getMul(Constant *C1, Constant *C2) {
  return get(Instruction::Mul, C1, C2);
}
Constant *ConstantExpr::getUDiv(Constant *C1, Constant *C2) {
  return get(Instruction::UDiv, C1, C2);
}
Constant *ConstantExpr::getSDiv(Constant *C1, Constant *C2) {
  return get(Instruction::SDiv, C1, C2);
}
Constant *ConstantExpr::getFDiv(Constant *C1, Constant *C2) {
  return get(Instruction::FDiv, C1, C2);
}
Constant *ConstantExpr::getURem(Constant *C1, Constant *C2) {
  return get(Instruction::URem, C1, C2);
}
Constant *ConstantExpr::getSRem(Constant *C1, Constant *C2) {
  return get(Instruction::SRem, C1, C2);
}
Constant *ConstantExpr::getFRem(Constant *C1, Constant *C2) {
  return get(Instruction::FRem, C1, C2);
}
Constant *ConstantExpr::getAnd(Constant *C1, Constant *C2) {
  return get(Instruction::And, C1, C2);
}
Constant *ConstantExpr::getOr(Constant *C1, Constant *C2) {
  return get(Instruction::Or, C1, C2);
}
Constant *ConstantExpr::getXor(Constant *C1, Constant *C2) {
  return get(Instruction::Xor, C1, C2);
}
unsigned ConstantExpr::getPredicate() const {
  assert(getOpcode() == Instruction::FCmp || 
         getOpcode() == Instruction::ICmp ||
         getOpcode() == Instruction::VFCmp ||
         getOpcode() == Instruction::VICmp);
  return ((const CompareConstantExpr*)this)->predicate;
}
Constant *ConstantExpr::getShl(Constant *C1, Constant *C2) {
  return get(Instruction::Shl, C1, C2);
}
Constant *ConstantExpr::getLShr(Constant *C1, Constant *C2) {
  return get(Instruction::LShr, C1, C2);
}
Constant *ConstantExpr::getAShr(Constant *C1, Constant *C2) {
  return get(Instruction::AShr, C1, C2);
}

/// getWithOperandReplaced - Return a constant expression identical to this
/// one, but with the specified operand set to the specified value.
Constant *
ConstantExpr::getWithOperandReplaced(unsigned OpNo, Constant *Op) const {
  assert(OpNo < getNumOperands() && "Operand num is out of range!");
  assert(Op->getType() == getOperand(OpNo)->getType() &&
         "Replacing operand with value of different type!");
  if (getOperand(OpNo) == Op)
    return const_cast<ConstantExpr*>(this);
  
  Constant *Op0, *Op1, *Op2;
  switch (getOpcode()) {
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    return ConstantExpr::getCast(getOpcode(), Op, getType());
  case Instruction::Select:
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    Op2 = (OpNo == 2) ? Op : getOperand(2);
    return ConstantExpr::getSelect(Op0, Op1, Op2);
  case Instruction::InsertElement:
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    Op2 = (OpNo == 2) ? Op : getOperand(2);
    return ConstantExpr::getInsertElement(Op0, Op1, Op2);
  case Instruction::ExtractElement:
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    return ConstantExpr::getExtractElement(Op0, Op1);
  case Instruction::ShuffleVector:
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    Op2 = (OpNo == 2) ? Op : getOperand(2);
    return ConstantExpr::getShuffleVector(Op0, Op1, Op2);
  case Instruction::GetElementPtr: {
    SmallVector<Constant*, 8> Ops;
    Ops.resize(getNumOperands()-1);
    for (unsigned i = 1, e = getNumOperands(); i != e; ++i)
      Ops[i-1] = getOperand(i);
    if (OpNo == 0)
      return ConstantExpr::getGetElementPtr(Op, &Ops[0], Ops.size());
    Ops[OpNo-1] = Op;
    return ConstantExpr::getGetElementPtr(getOperand(0), &Ops[0], Ops.size());
  }
  default:
    assert(getNumOperands() == 2 && "Must be binary operator?");
    Op0 = (OpNo == 0) ? Op : getOperand(0);
    Op1 = (OpNo == 1) ? Op : getOperand(1);
    return ConstantExpr::get(getOpcode(), Op0, Op1);
  }
}

/// getWithOperands - This returns the current constant expression with the
/// operands replaced with the specified values.  The specified operands must
/// match count and type with the existing ones.
Constant *ConstantExpr::
getWithOperands(Constant* const *Ops, unsigned NumOps) const {
  assert(NumOps == getNumOperands() && "Operand count mismatch!");
  bool AnyChange = false;
  for (unsigned i = 0; i != NumOps; ++i) {
    assert(Ops[i]->getType() == getOperand(i)->getType() &&
           "Operand type mismatch!");
    AnyChange |= Ops[i] != getOperand(i);
  }
  if (!AnyChange)  // No operands changed, return self.
    return const_cast<ConstantExpr*>(this);

  switch (getOpcode()) {
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    return ConstantExpr::getCast(getOpcode(), Ops[0], getType());
  case Instruction::Select:
    return ConstantExpr::getSelect(Ops[0], Ops[1], Ops[2]);
  case Instruction::InsertElement:
    return ConstantExpr::getInsertElement(Ops[0], Ops[1], Ops[2]);
  case Instruction::ExtractElement:
    return ConstantExpr::getExtractElement(Ops[0], Ops[1]);
  case Instruction::ShuffleVector:
    return ConstantExpr::getShuffleVector(Ops[0], Ops[1], Ops[2]);
  case Instruction::GetElementPtr:
    return ConstantExpr::getGetElementPtr(Ops[0], &Ops[1], NumOps-1);
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::VICmp:
  case Instruction::VFCmp:
    return ConstantExpr::getCompare(getPredicate(), Ops[0], Ops[1]);
  default:
    assert(getNumOperands() == 2 && "Must be binary operator?");
    return ConstantExpr::get(getOpcode(), Ops[0], Ops[1]);
  }
}


//===----------------------------------------------------------------------===//
//                      isValueValidForType implementations

bool ConstantInt::isValueValidForType(const Type *Ty, uint64_t Val) {
  unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth(); // assert okay
  if (Ty == Type::Int1Ty)
    return Val == 0 || Val == 1;
  if (NumBits >= 64)
    return true; // always true, has to fit in largest type
  uint64_t Max = (1ll << NumBits) - 1;
  return Val <= Max;
}

bool ConstantInt::isValueValidForType(const Type *Ty, int64_t Val) {
  unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth(); // assert okay
  if (Ty == Type::Int1Ty)
    return Val == 0 || Val == 1 || Val == -1;
  if (NumBits >= 64)
    return true; // always true, has to fit in largest type
  int64_t Min = -(1ll << (NumBits-1));
  int64_t Max = (1ll << (NumBits-1)) - 1;
  return (Val >= Min && Val <= Max);
}

bool ConstantFP::isValueValidForType(const Type *Ty, const APFloat& Val) {
  // convert modifies in place, so make a copy.
  APFloat Val2 = APFloat(Val);
  bool losesInfo;
  switch (Ty->getTypeID()) {
  default:
    return false;         // These can't be represented as floating point!

  // FIXME rounding mode needs to be more flexible
  case Type::FloatTyID: {
    if (&Val2.getSemantics() == &APFloat::IEEEsingle)
      return true;
    Val2.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven, &losesInfo);
    return !losesInfo;
  }
  case Type::DoubleTyID: {
    if (&Val2.getSemantics() == &APFloat::IEEEsingle ||
        &Val2.getSemantics() == &APFloat::IEEEdouble)
      return true;
    Val2.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &losesInfo);
    return !losesInfo;
  }
  case Type::X86_FP80TyID:
    return &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           &Val2.getSemantics() == &APFloat::x87DoubleExtended;
  case Type::FP128TyID:
    return &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           &Val2.getSemantics() == &APFloat::IEEEquad;
  case Type::PPC_FP128TyID:
    return &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           &Val2.getSemantics() == &APFloat::PPCDoubleDouble;
  }
}

//===----------------------------------------------------------------------===//
//                      Factory Function Implementation


// The number of operands for each ConstantCreator::create method is
// determined by the ConstantTraits template.
// ConstantCreator - A class that is used to create constants by
// ValueMap*.  This class should be partially specialized if there is
// something strange that needs to be done to interface to the ctor for the
// constant.
//
namespace llvm {
  template<class ValType>
  struct ConstantTraits;

  template<typename T, typename Alloc>
  struct VISIBILITY_HIDDEN ConstantTraits< std::vector<T, Alloc> > {
    static unsigned uses(const std::vector<T, Alloc>& v) {
      return v.size();
    }
  };

  template<class ConstantClass, class TypeClass, class ValType>
  struct VISIBILITY_HIDDEN ConstantCreator {
    static ConstantClass *create(const TypeClass *Ty, const ValType &V) {
      return new(ConstantTraits<ValType>::uses(V)) ConstantClass(Ty, V);
    }
  };

  template<class ConstantClass, class TypeClass>
  struct VISIBILITY_HIDDEN ConvertConstantType {
    static void convert(ConstantClass *OldC, const TypeClass *NewTy) {
      assert(0 && "This type cannot be converted!\n");
      abort();
    }
  };

  template<class ValType, class TypeClass, class ConstantClass,
           bool HasLargeKey = false  /*true for arrays and structs*/ >
  class VISIBILITY_HIDDEN ValueMap : public AbstractTypeUser {
  public:
    typedef std::pair<const Type*, ValType> MapKey;
    typedef std::map<MapKey, Constant *> MapTy;
    typedef std::map<Constant*, typename MapTy::iterator> InverseMapTy;
    typedef std::map<const Type*, typename MapTy::iterator> AbstractTypeMapTy;
  private:
    /// Map - This is the main map from the element descriptor to the Constants.
    /// This is the primary way we avoid creating two of the same shape
    /// constant.
    MapTy Map;
    
    /// InverseMap - If "HasLargeKey" is true, this contains an inverse mapping
    /// from the constants to their element in Map.  This is important for
    /// removal of constants from the array, which would otherwise have to scan
    /// through the map with very large keys.
    InverseMapTy InverseMap;

    /// AbstractTypeMap - Map for abstract type constants.
    ///
    AbstractTypeMapTy AbstractTypeMap;

  public:
    typename MapTy::iterator map_end() { return Map.end(); }
    
    /// InsertOrGetItem - Return an iterator for the specified element.
    /// If the element exists in the map, the returned iterator points to the
    /// entry and Exists=true.  If not, the iterator points to the newly
    /// inserted entry and returns Exists=false.  Newly inserted entries have
    /// I->second == 0, and should be filled in.
    typename MapTy::iterator InsertOrGetItem(std::pair<MapKey, Constant *>
                                   &InsertVal,
                                   bool &Exists) {
      std::pair<typename MapTy::iterator, bool> IP = Map.insert(InsertVal);
      Exists = !IP.second;
      return IP.first;
    }
    
private:
    typename MapTy::iterator FindExistingElement(ConstantClass *CP) {
      if (HasLargeKey) {
        typename InverseMapTy::iterator IMI = InverseMap.find(CP);
        assert(IMI != InverseMap.end() && IMI->second != Map.end() &&
               IMI->second->second == CP &&
               "InverseMap corrupt!");
        return IMI->second;
      }
      
      typename MapTy::iterator I =
        Map.find(MapKey(static_cast<const TypeClass*>(CP->getRawType()),
                        getValType(CP)));
      if (I == Map.end() || I->second != CP) {
        // FIXME: This should not use a linear scan.  If this gets to be a
        // performance problem, someone should look at this.
        for (I = Map.begin(); I != Map.end() && I->second != CP; ++I)
          /* empty */;
      }
      return I;
    }
public:
    
    /// getOrCreate - Return the specified constant from the map, creating it if
    /// necessary.
    ConstantClass *getOrCreate(const TypeClass *Ty, const ValType &V) {
      MapKey Lookup(Ty, V);
      typename MapTy::iterator I = Map.find(Lookup);
      // Is it in the map?      
      if (I != Map.end())
        return static_cast<ConstantClass *>(I->second);  

      // If no preexisting value, create one now...
      ConstantClass *Result =
        ConstantCreator<ConstantClass,TypeClass,ValType>::create(Ty, V);

      assert(Result->getType() == Ty && "Type specified is not correct!");
      I = Map.insert(I, std::make_pair(MapKey(Ty, V), Result));

      if (HasLargeKey)  // Remember the reverse mapping if needed.
        InverseMap.insert(std::make_pair(Result, I));
      
      // If the type of the constant is abstract, make sure that an entry exists
      // for it in the AbstractTypeMap.
      if (Ty->isAbstract()) {
        typename AbstractTypeMapTy::iterator TI = AbstractTypeMap.find(Ty);

        if (TI == AbstractTypeMap.end()) {
          // Add ourselves to the ATU list of the type.
          cast<DerivedType>(Ty)->addAbstractTypeUser(this);

          AbstractTypeMap.insert(TI, std::make_pair(Ty, I));
        }
      }
      return Result;
    }

    void remove(ConstantClass *CP) {
      typename MapTy::iterator I = FindExistingElement(CP);
      assert(I != Map.end() && "Constant not found in constant table!");
      assert(I->second == CP && "Didn't find correct element?");

      if (HasLargeKey)  // Remember the reverse mapping if needed.
        InverseMap.erase(CP);
      
      // Now that we found the entry, make sure this isn't the entry that
      // the AbstractTypeMap points to.
      const TypeClass *Ty = static_cast<const TypeClass *>(I->first.first);
      if (Ty->isAbstract()) {
        assert(AbstractTypeMap.count(Ty) &&
               "Abstract type not in AbstractTypeMap?");
        typename MapTy::iterator &ATMEntryIt = AbstractTypeMap[Ty];
        if (ATMEntryIt == I) {
          // Yes, we are removing the representative entry for this type.
          // See if there are any other entries of the same type.
          typename MapTy::iterator TmpIt = ATMEntryIt;

          // First check the entry before this one...
          if (TmpIt != Map.begin()) {
            --TmpIt;
            if (TmpIt->first.first != Ty) // Not the same type, move back...
              ++TmpIt;
          }

          // If we didn't find the same type, try to move forward...
          if (TmpIt == ATMEntryIt) {
            ++TmpIt;
            if (TmpIt == Map.end() || TmpIt->first.first != Ty)
              --TmpIt;   // No entry afterwards with the same type
          }

          // If there is another entry in the map of the same abstract type,
          // update the AbstractTypeMap entry now.
          if (TmpIt != ATMEntryIt) {
            ATMEntryIt = TmpIt;
          } else {
            // Otherwise, we are removing the last instance of this type
            // from the table.  Remove from the ATM, and from user list.
            cast<DerivedType>(Ty)->removeAbstractTypeUser(this);
            AbstractTypeMap.erase(Ty);
          }
        }
      }

      Map.erase(I);
    }

    
    /// MoveConstantToNewSlot - If we are about to change C to be the element
    /// specified by I, update our internal data structures to reflect this
    /// fact.
    void MoveConstantToNewSlot(ConstantClass *C, typename MapTy::iterator I) {
      // First, remove the old location of the specified constant in the map.
      typename MapTy::iterator OldI = FindExistingElement(C);
      assert(OldI != Map.end() && "Constant not found in constant table!");
      assert(OldI->second == C && "Didn't find correct element?");
      
      // If this constant is the representative element for its abstract type,
      // update the AbstractTypeMap so that the representative element is I.
      if (C->getType()->isAbstract()) {
        typename AbstractTypeMapTy::iterator ATI =
            AbstractTypeMap.find(C->getType());
        assert(ATI != AbstractTypeMap.end() &&
               "Abstract type not in AbstractTypeMap?");
        if (ATI->second == OldI)
          ATI->second = I;
      }
      
      // Remove the old entry from the map.
      Map.erase(OldI);
      
      // Update the inverse map so that we know that this constant is now
      // located at descriptor I.
      if (HasLargeKey) {
        assert(I->second == C && "Bad inversemap entry!");
        InverseMap[C] = I;
      }
    }
    
    void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
      typename AbstractTypeMapTy::iterator I =
        AbstractTypeMap.find(cast<Type>(OldTy));

      assert(I != AbstractTypeMap.end() &&
             "Abstract type not in AbstractTypeMap?");

      // Convert a constant at a time until the last one is gone.  The last one
      // leaving will remove() itself, causing the AbstractTypeMapEntry to be
      // eliminated eventually.
      do {
        ConvertConstantType<ConstantClass,
                            TypeClass>::convert(
                                static_cast<ConstantClass *>(I->second->second),
                                                cast<TypeClass>(NewTy));

        I = AbstractTypeMap.find(cast<Type>(OldTy));
      } while (I != AbstractTypeMap.end());
    }

    // If the type became concrete without being refined to any other existing
    // type, we just remove ourselves from the ATU list.
    void typeBecameConcrete(const DerivedType *AbsTy) {
      AbsTy->removeAbstractTypeUser(this);
    }

    void dump() const {
      DOUT << "Constant.cpp: ValueMap\n";
    }
  };
}



//---- ConstantAggregateZero::get() implementation...
//
namespace llvm {
  // ConstantAggregateZero does not take extra "value" argument...
  template<class ValType>
  struct ConstantCreator<ConstantAggregateZero, Type, ValType> {
    static ConstantAggregateZero *create(const Type *Ty, const ValType &V){
      return new ConstantAggregateZero(Ty);
    }
  };

  template<>
  struct ConvertConstantType<ConstantAggregateZero, Type> {
    static void convert(ConstantAggregateZero *OldC, const Type *NewTy) {
      // Make everyone now use a constant of the new type...
      Constant *New = ConstantAggregateZero::get(NewTy);
      assert(New != OldC && "Didn't replace constant??");
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();     // This constant is now dead, destroy it.
    }
  };
}

static ManagedStatic<ValueMap<char, Type, 
                              ConstantAggregateZero> > AggZeroConstants;

static char getValType(ConstantAggregateZero *CPZ) { return 0; }

ConstantAggregateZero *ConstantAggregateZero::get(const Type *Ty) {
  assert((isa<StructType>(Ty) || isa<ArrayType>(Ty) || isa<VectorType>(Ty)) &&
         "Cannot create an aggregate zero of non-aggregate type!");
  return AggZeroConstants->getOrCreate(Ty, 0);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantAggregateZero::destroyConstant() {
  AggZeroConstants->remove(this);
  destroyConstantImpl();
}

//---- ConstantArray::get() implementation...
//
namespace llvm {
  template<>
  struct ConvertConstantType<ConstantArray, ArrayType> {
    static void convert(ConstantArray *OldC, const ArrayType *NewTy) {
      // Make everyone now use a constant of the new type...
      std::vector<Constant*> C;
      for (unsigned i = 0, e = OldC->getNumOperands(); i != e; ++i)
        C.push_back(cast<Constant>(OldC->getOperand(i)));
      Constant *New = ConstantArray::get(NewTy, C);
      assert(New != OldC && "Didn't replace constant??");
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();    // This constant is now dead, destroy it.
    }
  };
}

static std::vector<Constant*> getValType(ConstantArray *CA) {
  std::vector<Constant*> Elements;
  Elements.reserve(CA->getNumOperands());
  for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
    Elements.push_back(cast<Constant>(CA->getOperand(i)));
  return Elements;
}

typedef ValueMap<std::vector<Constant*>, ArrayType, 
                 ConstantArray, true /*largekey*/> ArrayConstantsTy;
static ManagedStatic<ArrayConstantsTy> ArrayConstants;

Constant *ConstantArray::get(const ArrayType *Ty,
                             const std::vector<Constant*> &V) {
  // If this is an all-zero array, return a ConstantAggregateZero object
  if (!V.empty()) {
    Constant *C = V[0];
    if (!C->isNullValue())
      return ArrayConstants->getOrCreate(Ty, V);
    for (unsigned i = 1, e = V.size(); i != e; ++i)
      if (V[i] != C)
        return ArrayConstants->getOrCreate(Ty, V);
  }
  return ConstantAggregateZero::get(Ty);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantArray::destroyConstant() {
  ArrayConstants->remove(this);
  destroyConstantImpl();
}

/// ConstantArray::get(const string&) - Return an array that is initialized to
/// contain the specified string.  If length is zero then a null terminator is 
/// added to the specified string so that it may be used in a natural way. 
/// Otherwise, the length parameter specifies how much of the string to use 
/// and it won't be null terminated.
///
Constant *ConstantArray::get(const std::string &Str, bool AddNull) {
  std::vector<Constant*> ElementVals;
  for (unsigned i = 0; i < Str.length(); ++i)
    ElementVals.push_back(ConstantInt::get(Type::Int8Ty, Str[i]));

  // Add a null terminator to the string...
  if (AddNull) {
    ElementVals.push_back(ConstantInt::get(Type::Int8Ty, 0));
  }

  ArrayType *ATy = ArrayType::get(Type::Int8Ty, ElementVals.size());
  return ConstantArray::get(ATy, ElementVals);
}

/// isString - This method returns true if the array is an array of i8, and 
/// if the elements of the array are all ConstantInt's.
bool ConstantArray::isString() const {
  // Check the element type for i8...
  if (getType()->getElementType() != Type::Int8Ty)
    return false;
  // Check the elements to make sure they are all integers, not constant
  // expressions.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (!isa<ConstantInt>(getOperand(i)))
      return false;
  return true;
}

/// isCString - This method returns true if the array is a string (see
/// isString) and it ends in a null byte \0 and does not contains any other
/// null bytes except its terminator.
bool ConstantArray::isCString() const {
  // Check the element type for i8...
  if (getType()->getElementType() != Type::Int8Ty)
    return false;
  Constant *Zero = Constant::getNullValue(getOperand(0)->getType());
  // Last element must be a null.
  if (getOperand(getNumOperands()-1) != Zero)
    return false;
  // Other elements must be non-null integers.
  for (unsigned i = 0, e = getNumOperands()-1; i != e; ++i) {
    if (!isa<ConstantInt>(getOperand(i)))
      return false;
    if (getOperand(i) == Zero)
      return false;
  }
  return true;
}


// getAsString - If the sub-element type of this array is i8
// then this method converts the array to an std::string and returns it.
// Otherwise, it asserts out.
//
std::string ConstantArray::getAsString() const {
  assert(isString() && "Not a string!");
  std::string Result;
  Result.reserve(getNumOperands());
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    Result.push_back((char)cast<ConstantInt>(getOperand(i))->getZExtValue());
  return Result;
}


//---- ConstantStruct::get() implementation...
//

namespace llvm {
  template<>
  struct ConvertConstantType<ConstantStruct, StructType> {
    static void convert(ConstantStruct *OldC, const StructType *NewTy) {
      // Make everyone now use a constant of the new type...
      std::vector<Constant*> C;
      for (unsigned i = 0, e = OldC->getNumOperands(); i != e; ++i)
        C.push_back(cast<Constant>(OldC->getOperand(i)));
      Constant *New = ConstantStruct::get(NewTy, C);
      assert(New != OldC && "Didn't replace constant??");

      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();    // This constant is now dead, destroy it.
    }
  };
}

typedef ValueMap<std::vector<Constant*>, StructType,
                 ConstantStruct, true /*largekey*/> StructConstantsTy;
static ManagedStatic<StructConstantsTy> StructConstants;

static std::vector<Constant*> getValType(ConstantStruct *CS) {
  std::vector<Constant*> Elements;
  Elements.reserve(CS->getNumOperands());
  for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i)
    Elements.push_back(cast<Constant>(CS->getOperand(i)));
  return Elements;
}

Constant *ConstantStruct::get(const StructType *Ty,
                              const std::vector<Constant*> &V) {
  // Create a ConstantAggregateZero value if all elements are zeros...
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    if (!V[i]->isNullValue())
      return StructConstants->getOrCreate(Ty, V);

  return ConstantAggregateZero::get(Ty);
}

Constant *ConstantStruct::get(const std::vector<Constant*> &V, bool packed) {
  std::vector<const Type*> StructEls;
  StructEls.reserve(V.size());
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    StructEls.push_back(V[i]->getType());
  return get(StructType::get(StructEls, packed), V);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantStruct::destroyConstant() {
  StructConstants->remove(this);
  destroyConstantImpl();
}

//---- ConstantVector::get() implementation...
//
namespace llvm {
  template<>
  struct ConvertConstantType<ConstantVector, VectorType> {
    static void convert(ConstantVector *OldC, const VectorType *NewTy) {
      // Make everyone now use a constant of the new type...
      std::vector<Constant*> C;
      for (unsigned i = 0, e = OldC->getNumOperands(); i != e; ++i)
        C.push_back(cast<Constant>(OldC->getOperand(i)));
      Constant *New = ConstantVector::get(NewTy, C);
      assert(New != OldC && "Didn't replace constant??");
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();    // This constant is now dead, destroy it.
    }
  };
}

static std::vector<Constant*> getValType(ConstantVector *CP) {
  std::vector<Constant*> Elements;
  Elements.reserve(CP->getNumOperands());
  for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
    Elements.push_back(CP->getOperand(i));
  return Elements;
}

static ManagedStatic<ValueMap<std::vector<Constant*>, VectorType,
                              ConstantVector> > VectorConstants;

Constant *ConstantVector::get(const VectorType *Ty,
                              const std::vector<Constant*> &V) {
  assert(!V.empty() && "Vectors can't be empty");
  // If this is an all-undef or alll-zero vector, return a
  // ConstantAggregateZero or UndefValue.
  Constant *C = V[0];
  bool isZero = C->isNullValue();
  bool isUndef = isa<UndefValue>(C);

  if (isZero || isUndef) {
    for (unsigned i = 1, e = V.size(); i != e; ++i)
      if (V[i] != C) {
        isZero = isUndef = false;
        break;
      }
  }
  
  if (isZero)
    return ConstantAggregateZero::get(Ty);
  if (isUndef)
    return UndefValue::get(Ty);
  return VectorConstants->getOrCreate(Ty, V);
}

Constant *ConstantVector::get(const std::vector<Constant*> &V) {
  assert(!V.empty() && "Cannot infer type if V is empty");
  return get(VectorType::get(V.front()->getType(),V.size()), V);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantVector::destroyConstant() {
  VectorConstants->remove(this);
  destroyConstantImpl();
}

/// This function will return true iff every element in this vector constant
/// is set to all ones.
/// @returns true iff this constant's emements are all set to all ones.
/// @brief Determine if the value is all ones.
bool ConstantVector::isAllOnesValue() const {
  // Check out first element.
  const Constant *Elt = getOperand(0);
  const ConstantInt *CI = dyn_cast<ConstantInt>(Elt);
  if (!CI || !CI->isAllOnesValue()) return false;
  // Then make sure all remaining elements point to the same value.
  for (unsigned I = 1, E = getNumOperands(); I < E; ++I) {
    if (getOperand(I) != Elt) return false;
  }
  return true;
}

/// getSplatValue - If this is a splat constant, where all of the
/// elements have the same value, return that value. Otherwise return null.
Constant *ConstantVector::getSplatValue() {
  // Check out first element.
  Constant *Elt = getOperand(0);
  // Then make sure all remaining elements point to the same value.
  for (unsigned I = 1, E = getNumOperands(); I < E; ++I)
    if (getOperand(I) != Elt) return 0;
  return Elt;
}

//---- ConstantPointerNull::get() implementation...
//

namespace llvm {
  // ConstantPointerNull does not take extra "value" argument...
  template<class ValType>
  struct ConstantCreator<ConstantPointerNull, PointerType, ValType> {
    static ConstantPointerNull *create(const PointerType *Ty, const ValType &V){
      return new ConstantPointerNull(Ty);
    }
  };

  template<>
  struct ConvertConstantType<ConstantPointerNull, PointerType> {
    static void convert(ConstantPointerNull *OldC, const PointerType *NewTy) {
      // Make everyone now use a constant of the new type...
      Constant *New = ConstantPointerNull::get(NewTy);
      assert(New != OldC && "Didn't replace constant??");
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();     // This constant is now dead, destroy it.
    }
  };
}

static ManagedStatic<ValueMap<char, PointerType, 
                              ConstantPointerNull> > NullPtrConstants;

static char getValType(ConstantPointerNull *) {
  return 0;
}


ConstantPointerNull *ConstantPointerNull::get(const PointerType *Ty) {
  return NullPtrConstants->getOrCreate(Ty, 0);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantPointerNull::destroyConstant() {
  NullPtrConstants->remove(this);
  destroyConstantImpl();
}


//---- UndefValue::get() implementation...
//

namespace llvm {
  // UndefValue does not take extra "value" argument...
  template<class ValType>
  struct ConstantCreator<UndefValue, Type, ValType> {
    static UndefValue *create(const Type *Ty, const ValType &V) {
      return new UndefValue(Ty);
    }
  };

  template<>
  struct ConvertConstantType<UndefValue, Type> {
    static void convert(UndefValue *OldC, const Type *NewTy) {
      // Make everyone now use a constant of the new type.
      Constant *New = UndefValue::get(NewTy);
      assert(New != OldC && "Didn't replace constant??");
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();     // This constant is now dead, destroy it.
    }
  };
}

static ManagedStatic<ValueMap<char, Type, UndefValue> > UndefValueConstants;

static char getValType(UndefValue *) {
  return 0;
}


UndefValue *UndefValue::get(const Type *Ty) {
  return UndefValueConstants->getOrCreate(Ty, 0);
}

// destroyConstant - Remove the constant from the constant table.
//
void UndefValue::destroyConstant() {
  UndefValueConstants->remove(this);
  destroyConstantImpl();
}


//---- ConstantExpr::get() implementations...
//

namespace {

struct ExprMapKeyType {
  typedef SmallVector<unsigned, 4> IndexList;

  ExprMapKeyType(unsigned opc,
      const std::vector<Constant*> &ops,
      unsigned short pred = 0,
      const IndexList &inds = IndexList())
        : opcode(opc), predicate(pred), operands(ops), indices(inds) {}
  uint16_t opcode;
  uint16_t predicate;
  std::vector<Constant*> operands;
  IndexList indices;
  bool operator==(const ExprMapKeyType& that) const {
    return this->opcode == that.opcode &&
           this->predicate == that.predicate &&
           this->operands == that.operands &&
           this->indices == that.indices;
  }
  bool operator<(const ExprMapKeyType & that) const {
    return this->opcode < that.opcode ||
      (this->opcode == that.opcode && this->predicate < that.predicate) ||
      (this->opcode == that.opcode && this->predicate == that.predicate &&
       this->operands < that.operands) ||
      (this->opcode == that.opcode && this->predicate == that.predicate &&
       this->operands == that.operands && this->indices < that.indices);
  }

  bool operator!=(const ExprMapKeyType& that) const {
    return !(*this == that);
  }
};

}

namespace llvm {
  template<>
  struct ConstantCreator<ConstantExpr, Type, ExprMapKeyType> {
    static ConstantExpr *create(const Type *Ty, const ExprMapKeyType &V,
        unsigned short pred = 0) {
      if (Instruction::isCast(V.opcode))
        return new UnaryConstantExpr(V.opcode, V.operands[0], Ty);
      if ((V.opcode >= Instruction::BinaryOpsBegin &&
           V.opcode < Instruction::BinaryOpsEnd))
        return new BinaryConstantExpr(V.opcode, V.operands[0], V.operands[1]);
      if (V.opcode == Instruction::Select)
        return new SelectConstantExpr(V.operands[0], V.operands[1], 
                                      V.operands[2]);
      if (V.opcode == Instruction::ExtractElement)
        return new ExtractElementConstantExpr(V.operands[0], V.operands[1]);
      if (V.opcode == Instruction::InsertElement)
        return new InsertElementConstantExpr(V.operands[0], V.operands[1],
                                             V.operands[2]);
      if (V.opcode == Instruction::ShuffleVector)
        return new ShuffleVectorConstantExpr(V.operands[0], V.operands[1],
                                             V.operands[2]);
      if (V.opcode == Instruction::InsertValue)
        return new InsertValueConstantExpr(V.operands[0], V.operands[1],
                                           V.indices, Ty);
      if (V.opcode == Instruction::ExtractValue)
        return new ExtractValueConstantExpr(V.operands[0], V.indices, Ty);
      if (V.opcode == Instruction::GetElementPtr) {
        std::vector<Constant*> IdxList(V.operands.begin()+1, V.operands.end());
        return GetElementPtrConstantExpr::Create(V.operands[0], IdxList, Ty);
      }

      // The compare instructions are weird. We have to encode the predicate
      // value and it is combined with the instruction opcode by multiplying
      // the opcode by one hundred. We must decode this to get the predicate.
      if (V.opcode == Instruction::ICmp)
        return new CompareConstantExpr(Ty, Instruction::ICmp, V.predicate, 
                                       V.operands[0], V.operands[1]);
      if (V.opcode == Instruction::FCmp) 
        return new CompareConstantExpr(Ty, Instruction::FCmp, V.predicate, 
                                       V.operands[0], V.operands[1]);
      if (V.opcode == Instruction::VICmp)
        return new CompareConstantExpr(Ty, Instruction::VICmp, V.predicate, 
                                       V.operands[0], V.operands[1]);
      if (V.opcode == Instruction::VFCmp) 
        return new CompareConstantExpr(Ty, Instruction::VFCmp, V.predicate, 
                                       V.operands[0], V.operands[1]);
      assert(0 && "Invalid ConstantExpr!");
      return 0;
    }
  };

  template<>
  struct ConvertConstantType<ConstantExpr, Type> {
    static void convert(ConstantExpr *OldC, const Type *NewTy) {
      Constant *New;
      switch (OldC->getOpcode()) {
      case Instruction::Trunc:
      case Instruction::ZExt:
      case Instruction::SExt:
      case Instruction::FPTrunc:
      case Instruction::FPExt:
      case Instruction::UIToFP:
      case Instruction::SIToFP:
      case Instruction::FPToUI:
      case Instruction::FPToSI:
      case Instruction::PtrToInt:
      case Instruction::IntToPtr:
      case Instruction::BitCast:
        New = ConstantExpr::getCast(OldC->getOpcode(), OldC->getOperand(0), 
                                    NewTy);
        break;
      case Instruction::Select:
        New = ConstantExpr::getSelectTy(NewTy, OldC->getOperand(0),
                                        OldC->getOperand(1),
                                        OldC->getOperand(2));
        break;
      default:
        assert(OldC->getOpcode() >= Instruction::BinaryOpsBegin &&
               OldC->getOpcode() <  Instruction::BinaryOpsEnd);
        New = ConstantExpr::getTy(NewTy, OldC->getOpcode(), OldC->getOperand(0),
                                  OldC->getOperand(1));
        break;
      case Instruction::GetElementPtr:
        // Make everyone now use a constant of the new type...
        std::vector<Value*> Idx(OldC->op_begin()+1, OldC->op_end());
        New = ConstantExpr::getGetElementPtrTy(NewTy, OldC->getOperand(0),
                                               &Idx[0], Idx.size());
        break;
      }

      assert(New != OldC && "Didn't replace constant??");
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();    // This constant is now dead, destroy it.
    }
  };
} // end namespace llvm


static ExprMapKeyType getValType(ConstantExpr *CE) {
  std::vector<Constant*> Operands;
  Operands.reserve(CE->getNumOperands());
  for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i)
    Operands.push_back(cast<Constant>(CE->getOperand(i)));
  return ExprMapKeyType(CE->getOpcode(), Operands, 
      CE->isCompare() ? CE->getPredicate() : 0,
      CE->hasIndices() ?
        CE->getIndices() : SmallVector<unsigned, 4>());
}

static ManagedStatic<ValueMap<ExprMapKeyType, Type,
                              ConstantExpr> > ExprConstants;

/// This is a utility function to handle folding of casts and lookup of the
/// cast in the ExprConstants map. It is used by the various get* methods below.
static inline Constant *getFoldedCast(
  Instruction::CastOps opc, Constant *C, const Type *Ty) {
  assert(Ty->isFirstClassType() && "Cannot cast to an aggregate type!");
  // Fold a few common cases
  if (Constant *FC = ConstantFoldCastInstruction(opc, C, Ty))
    return FC;

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> argVec(1, C);
  ExprMapKeyType Key(opc, argVec);
  return ExprConstants->getOrCreate(Ty, Key);
}
 
Constant *ConstantExpr::getCast(unsigned oc, Constant *C, const Type *Ty) {
  Instruction::CastOps opc = Instruction::CastOps(oc);
  assert(Instruction::isCast(opc) && "opcode out of range");
  assert(C && Ty && "Null arguments to getCast");
  assert(Ty->isFirstClassType() && "Cannot cast to an aggregate type!");

  switch (opc) {
    default:
      assert(0 && "Invalid cast opcode");
      break;
    case Instruction::Trunc:    return getTrunc(C, Ty);
    case Instruction::ZExt:     return getZExt(C, Ty);
    case Instruction::SExt:     return getSExt(C, Ty);
    case Instruction::FPTrunc:  return getFPTrunc(C, Ty);
    case Instruction::FPExt:    return getFPExtend(C, Ty);
    case Instruction::UIToFP:   return getUIToFP(C, Ty);
    case Instruction::SIToFP:   return getSIToFP(C, Ty);
    case Instruction::FPToUI:   return getFPToUI(C, Ty);
    case Instruction::FPToSI:   return getFPToSI(C, Ty);
    case Instruction::PtrToInt: return getPtrToInt(C, Ty);
    case Instruction::IntToPtr: return getIntToPtr(C, Ty);
    case Instruction::BitCast:  return getBitCast(C, Ty);
  }
  return 0;
} 

Constant *ConstantExpr::getZExtOrBitCast(Constant *C, const Type *Ty) {
  if (C->getType()->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return getCast(Instruction::BitCast, C, Ty);
  return getCast(Instruction::ZExt, C, Ty);
}

Constant *ConstantExpr::getSExtOrBitCast(Constant *C, const Type *Ty) {
  if (C->getType()->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return getCast(Instruction::BitCast, C, Ty);
  return getCast(Instruction::SExt, C, Ty);
}

Constant *ConstantExpr::getTruncOrBitCast(Constant *C, const Type *Ty) {
  if (C->getType()->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return getCast(Instruction::BitCast, C, Ty);
  return getCast(Instruction::Trunc, C, Ty);
}

Constant *ConstantExpr::getPointerCast(Constant *S, const Type *Ty) {
  assert(isa<PointerType>(S->getType()) && "Invalid cast");
  assert((Ty->isInteger() || isa<PointerType>(Ty)) && "Invalid cast");

  if (Ty->isInteger())
    return getCast(Instruction::PtrToInt, S, Ty);
  return getCast(Instruction::BitCast, S, Ty);
}

Constant *ConstantExpr::getIntegerCast(Constant *C, const Type *Ty, 
                                       bool isSigned) {
  assert(C->getType()->isInteger() && Ty->isInteger() && "Invalid cast");
  unsigned SrcBits = C->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = Ty->getPrimitiveSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return getCast(opcode, C, Ty);
}

Constant *ConstantExpr::getFPCast(Constant *C, const Type *Ty) {
  assert(C->getType()->isFloatingPoint() && Ty->isFloatingPoint() && 
         "Invalid cast");
  unsigned SrcBits = C->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = Ty->getPrimitiveSizeInBits();
  if (SrcBits == DstBits)
    return C; // Avoid a useless cast
  Instruction::CastOps opcode =
     (SrcBits > DstBits ? Instruction::FPTrunc : Instruction::FPExt);
  return getCast(opcode, C, Ty);
}

Constant *ConstantExpr::getTrunc(Constant *C, const Type *Ty) {
  assert(C->getType()->isInteger() && "Trunc operand must be integer");
  assert(Ty->isInteger() && "Trunc produces only integral");
  assert(C->getType()->getPrimitiveSizeInBits() > Ty->getPrimitiveSizeInBits()&&
         "SrcTy must be larger than DestTy for Trunc!");

  return getFoldedCast(Instruction::Trunc, C, Ty);
}

Constant *ConstantExpr::getSExt(Constant *C, const Type *Ty) {
  assert(C->getType()->isInteger() && "SEXt operand must be integral");
  assert(Ty->isInteger() && "SExt produces only integer");
  assert(C->getType()->getPrimitiveSizeInBits() < Ty->getPrimitiveSizeInBits()&&
         "SrcTy must be smaller than DestTy for SExt!");

  return getFoldedCast(Instruction::SExt, C, Ty);
}

Constant *ConstantExpr::getZExt(Constant *C, const Type *Ty) {
  assert(C->getType()->isInteger() && "ZEXt operand must be integral");
  assert(Ty->isInteger() && "ZExt produces only integer");
  assert(C->getType()->getPrimitiveSizeInBits() < Ty->getPrimitiveSizeInBits()&&
         "SrcTy must be smaller than DestTy for ZExt!");

  return getFoldedCast(Instruction::ZExt, C, Ty);
}

Constant *ConstantExpr::getFPTrunc(Constant *C, const Type *Ty) {
  assert(C->getType()->isFloatingPoint() && Ty->isFloatingPoint() &&
         C->getType()->getPrimitiveSizeInBits() > Ty->getPrimitiveSizeInBits()&&
         "This is an illegal floating point truncation!");
  return getFoldedCast(Instruction::FPTrunc, C, Ty);
}

Constant *ConstantExpr::getFPExtend(Constant *C, const Type *Ty) {
  assert(C->getType()->isFloatingPoint() && Ty->isFloatingPoint() &&
         C->getType()->getPrimitiveSizeInBits() < Ty->getPrimitiveSizeInBits()&&
         "This is an illegal floating point extension!");
  return getFoldedCast(Instruction::FPExt, C, Ty);
}

Constant *ConstantExpr::getUIToFP(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVector() && Ty->isFPOrFPVector() &&
         "This is an illegal uint to floating point cast!");
  return getFoldedCast(Instruction::UIToFP, C, Ty);
}

Constant *ConstantExpr::getSIToFP(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVector() && Ty->isFPOrFPVector() &&
         "This is an illegal sint to floating point cast!");
  return getFoldedCast(Instruction::SIToFP, C, Ty);
}

Constant *ConstantExpr::getFPToUI(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVector() && Ty->isIntOrIntVector() &&
         "This is an illegal floating point to uint cast!");
  return getFoldedCast(Instruction::FPToUI, C, Ty);
}

Constant *ConstantExpr::getFPToSI(Constant *C, const Type *Ty) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVector() && Ty->isIntOrIntVector() &&
         "This is an illegal floating point to sint cast!");
  return getFoldedCast(Instruction::FPToSI, C, Ty);
}

Constant *ConstantExpr::getPtrToInt(Constant *C, const Type *DstTy) {
  assert(isa<PointerType>(C->getType()) && "PtrToInt source must be pointer");
  assert(DstTy->isInteger() && "PtrToInt destination must be integral");
  return getFoldedCast(Instruction::PtrToInt, C, DstTy);
}

Constant *ConstantExpr::getIntToPtr(Constant *C, const Type *DstTy) {
  assert(C->getType()->isInteger() && "IntToPtr source must be integral");
  assert(isa<PointerType>(DstTy) && "IntToPtr destination must be a pointer");
  return getFoldedCast(Instruction::IntToPtr, C, DstTy);
}

Constant *ConstantExpr::getBitCast(Constant *C, const Type *DstTy) {
  // BitCast implies a no-op cast of type only. No bits change.  However, you 
  // can't cast pointers to anything but pointers.
#ifndef NDEBUG
  const Type *SrcTy = C->getType();
  assert((isa<PointerType>(SrcTy) == isa<PointerType>(DstTy)) &&
         "BitCast cannot cast pointer to non-pointer and vice versa");

  // Now we know we're not dealing with mismatched pointer casts (ptr->nonptr
  // or nonptr->ptr). For all the other types, the cast is okay if source and 
  // destination bit widths are identical.
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DstBitSize = DstTy->getPrimitiveSizeInBits();
#endif
  assert(SrcBitSize == DstBitSize && "BitCast requies types of same width");
  return getFoldedCast(Instruction::BitCast, C, DstTy);
}

Constant *ConstantExpr::getSizeOf(const Type *Ty) {
  // sizeof is implemented as: (i64) gep (Ty*)null, 1
  Constant *GEPIdx = ConstantInt::get(Type::Int32Ty, 1);
  Constant *GEP =
    getGetElementPtr(getNullValue(PointerType::getUnqual(Ty)), &GEPIdx, 1);
  return getCast(Instruction::PtrToInt, GEP, Type::Int64Ty);
}

Constant *ConstantExpr::getTy(const Type *ReqTy, unsigned Opcode,
                              Constant *C1, Constant *C2) {
  // Check the operands for consistency first
  assert(Opcode >= Instruction::BinaryOpsBegin &&
         Opcode <  Instruction::BinaryOpsEnd   &&
         "Invalid opcode in binary constant expression");
  assert(C1->getType() == C2->getType() &&
         "Operand types in binary constant expression should match");

  if (ReqTy == C1->getType() || ReqTy == Type::Int1Ty)
    if (Constant *FC = ConstantFoldBinaryInstruction(Opcode, C1, C2))
      return FC;          // Fold a few common cases...

  std::vector<Constant*> argVec(1, C1); argVec.push_back(C2);
  ExprMapKeyType Key(Opcode, argVec);
  return ExprConstants->getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getCompareTy(unsigned short predicate,
                                     Constant *C1, Constant *C2) {
  bool isVectorType = C1->getType()->getTypeID() == Type::VectorTyID;
  switch (predicate) {
    default: assert(0 && "Invalid CmpInst predicate");
    case CmpInst::FCMP_FALSE: case CmpInst::FCMP_OEQ: case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_OGE:   case CmpInst::FCMP_OLT: case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_ONE:   case CmpInst::FCMP_ORD: case CmpInst::FCMP_UNO:
    case CmpInst::FCMP_UEQ:   case CmpInst::FCMP_UGT: case CmpInst::FCMP_UGE:
    case CmpInst::FCMP_ULT:   case CmpInst::FCMP_ULE: case CmpInst::FCMP_UNE:
    case CmpInst::FCMP_TRUE:
      return isVectorType ? getVFCmp(predicate, C1, C2) 
                          : getFCmp(predicate, C1, C2);
    case CmpInst::ICMP_EQ:  case CmpInst::ICMP_NE:  case CmpInst::ICMP_UGT:
    case CmpInst::ICMP_UGE: case CmpInst::ICMP_ULT: case CmpInst::ICMP_ULE:
    case CmpInst::ICMP_SGT: case CmpInst::ICMP_SGE: case CmpInst::ICMP_SLT:
    case CmpInst::ICMP_SLE:
      return isVectorType ? getVICmp(predicate, C1, C2)
                          : getICmp(predicate, C1, C2);
  }
}

Constant *ConstantExpr::get(unsigned Opcode, Constant *C1, Constant *C2) {
#ifndef NDEBUG
  switch (Opcode) {
  case Instruction::Add: 
  case Instruction::Sub:
  case Instruction::Mul: 
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert((C1->getType()->isInteger() || C1->getType()->isFloatingPoint() ||
            isa<VectorType>(C1->getType())) &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::UDiv: 
  case Instruction::SDiv: 
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert((C1->getType()->isInteger() || (isa<VectorType>(C1->getType()) &&
      cast<VectorType>(C1->getType())->getElementType()->isInteger())) &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::FDiv:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert((C1->getType()->isFloatingPoint() || (isa<VectorType>(C1->getType())
      && cast<VectorType>(C1->getType())->getElementType()->isFloatingPoint())) 
      && "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::URem: 
  case Instruction::SRem: 
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert((C1->getType()->isInteger() || (isa<VectorType>(C1->getType()) &&
      cast<VectorType>(C1->getType())->getElementType()->isInteger())) &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::FRem:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert((C1->getType()->isFloatingPoint() || (isa<VectorType>(C1->getType())
      && cast<VectorType>(C1->getType())->getElementType()->isFloatingPoint())) 
      && "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert((C1->getType()->isInteger() || isa<VectorType>(C1->getType())) &&
           "Tried to create a logical operation on a non-integral type!");
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isInteger() &&
           "Tried to create a shift operation on a non-integer type!");
    break;
  default:
    break;
  }
#endif

  return getTy(C1->getType(), Opcode, C1, C2);
}

Constant *ConstantExpr::getCompare(unsigned short pred, 
                            Constant *C1, Constant *C2) {
  assert(C1->getType() == C2->getType() && "Op types should be identical!");
  return getCompareTy(pred, C1, C2);
}

Constant *ConstantExpr::getSelectTy(const Type *ReqTy, Constant *C,
                                    Constant *V1, Constant *V2) {
  assert(!SelectInst::areInvalidOperands(C, V1, V2)&&"Invalid select operands");

  if (ReqTy == V1->getType())
    if (Constant *SC = ConstantFoldSelectInstruction(C, V1, V2))
      return SC;        // Fold common cases

  std::vector<Constant*> argVec(3, C);
  argVec[1] = V1;
  argVec[2] = V2;
  ExprMapKeyType Key(Instruction::Select, argVec);
  return ExprConstants->getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getGetElementPtrTy(const Type *ReqTy, Constant *C,
                                           Value* const *Idxs,
                                           unsigned NumIdx) {
  assert(GetElementPtrInst::getIndexedType(C->getType(), Idxs,
                                           Idxs+NumIdx) ==
         cast<PointerType>(ReqTy)->getElementType() &&
         "GEP indices invalid!");

  if (Constant *FC = ConstantFoldGetElementPtr(C, (Constant**)Idxs, NumIdx))
    return FC;          // Fold a few common cases...

  assert(isa<PointerType>(C->getType()) &&
         "Non-pointer type for constant GetElementPtr expression");
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.reserve(NumIdx+1);
  ArgVec.push_back(C);
  for (unsigned i = 0; i != NumIdx; ++i)
    ArgVec.push_back(cast<Constant>(Idxs[i]));
  const ExprMapKeyType Key(Instruction::GetElementPtr, ArgVec);
  return ExprConstants->getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getGetElementPtr(Constant *C, Value* const *Idxs,
                                         unsigned NumIdx) {
  // Get the result type of the getelementptr!
  const Type *Ty = 
    GetElementPtrInst::getIndexedType(C->getType(), Idxs, Idxs+NumIdx);
  assert(Ty && "GEP indices invalid!");
  unsigned As = cast<PointerType>(C->getType())->getAddressSpace();
  return getGetElementPtrTy(PointerType::get(Ty, As), C, Idxs, NumIdx);
}

Constant *ConstantExpr::getGetElementPtr(Constant *C, Constant* const *Idxs,
                                         unsigned NumIdx) {
  return getGetElementPtr(C, (Value* const *)Idxs, NumIdx);
}


Constant *
ConstantExpr::getICmp(unsigned short pred, Constant* LHS, Constant* RHS) {
  assert(LHS->getType() == RHS->getType());
  assert(pred >= ICmpInst::FIRST_ICMP_PREDICATE && 
         pred <= ICmpInst::LAST_ICMP_PREDICATE && "Invalid ICmp Predicate");

  if (Constant *FC = ConstantFoldCompareInstruction(pred, LHS, RHS))
    return FC;          // Fold a few common cases...

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.push_back(LHS);
  ArgVec.push_back(RHS);
  // Get the key type with both the opcode and predicate
  const ExprMapKeyType Key(Instruction::ICmp, ArgVec, pred);
  return ExprConstants->getOrCreate(Type::Int1Ty, Key);
}

Constant *
ConstantExpr::getFCmp(unsigned short pred, Constant* LHS, Constant* RHS) {
  assert(LHS->getType() == RHS->getType());
  assert(pred <= FCmpInst::LAST_FCMP_PREDICATE && "Invalid FCmp Predicate");

  if (Constant *FC = ConstantFoldCompareInstruction(pred, LHS, RHS))
    return FC;          // Fold a few common cases...

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.push_back(LHS);
  ArgVec.push_back(RHS);
  // Get the key type with both the opcode and predicate
  const ExprMapKeyType Key(Instruction::FCmp, ArgVec, pred);
  return ExprConstants->getOrCreate(Type::Int1Ty, Key);
}

Constant *
ConstantExpr::getVICmp(unsigned short pred, Constant* LHS, Constant* RHS) {
  assert(isa<VectorType>(LHS->getType()) && LHS->getType() == RHS->getType() &&
         "Tried to create vicmp operation on non-vector type!");
  assert(pred >= ICmpInst::FIRST_ICMP_PREDICATE && 
         pred <= ICmpInst::LAST_ICMP_PREDICATE && "Invalid VICmp Predicate");

  const VectorType *VTy = cast<VectorType>(LHS->getType());
  const Type *EltTy = VTy->getElementType();
  unsigned NumElts = VTy->getNumElements();

  // See if we can fold the element-wise comparison of the LHS and RHS.
  SmallVector<Constant *, 16> LHSElts, RHSElts;
  LHS->getVectorElements(LHSElts);
  RHS->getVectorElements(RHSElts);
                    
  if (!LHSElts.empty() && !RHSElts.empty()) {
    SmallVector<Constant *, 16> Elts;
    for (unsigned i = 0; i != NumElts; ++i) {
      Constant *FC = ConstantFoldCompareInstruction(pred, LHSElts[i],
                                                    RHSElts[i]);
      if (ConstantInt *FCI = dyn_cast_or_null<ConstantInt>(FC)) {
        if (FCI->getZExtValue())
          Elts.push_back(ConstantInt::getAllOnesValue(EltTy));
        else
          Elts.push_back(ConstantInt::get(EltTy, 0ULL));
      } else if (FC && isa<UndefValue>(FC)) {
        Elts.push_back(UndefValue::get(EltTy));
      } else {
        break;
      }
    }
    if (Elts.size() == NumElts)
      return ConstantVector::get(&Elts[0], Elts.size());
  }

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.push_back(LHS);
  ArgVec.push_back(RHS);
  // Get the key type with both the opcode and predicate
  const ExprMapKeyType Key(Instruction::VICmp, ArgVec, pred);
  return ExprConstants->getOrCreate(LHS->getType(), Key);
}

Constant *
ConstantExpr::getVFCmp(unsigned short pred, Constant* LHS, Constant* RHS) {
  assert(isa<VectorType>(LHS->getType()) &&
         "Tried to create vfcmp operation on non-vector type!");
  assert(LHS->getType() == RHS->getType());
  assert(pred <= FCmpInst::LAST_FCMP_PREDICATE && "Invalid VFCmp Predicate");

  const VectorType *VTy = cast<VectorType>(LHS->getType());
  unsigned NumElts = VTy->getNumElements();
  const Type *EltTy = VTy->getElementType();
  const Type *REltTy = IntegerType::get(EltTy->getPrimitiveSizeInBits());
  const Type *ResultTy = VectorType::get(REltTy, NumElts);

  // See if we can fold the element-wise comparison of the LHS and RHS.
  SmallVector<Constant *, 16> LHSElts, RHSElts;
  LHS->getVectorElements(LHSElts);
  RHS->getVectorElements(RHSElts);
  
  if (!LHSElts.empty() && !RHSElts.empty()) {
    SmallVector<Constant *, 16> Elts;
    for (unsigned i = 0; i != NumElts; ++i) {
      Constant *FC = ConstantFoldCompareInstruction(pred, LHSElts[i],
                                                    RHSElts[i]);
      if (ConstantInt *FCI = dyn_cast_or_null<ConstantInt>(FC)) {
        if (FCI->getZExtValue())
          Elts.push_back(ConstantInt::getAllOnesValue(REltTy));
        else
          Elts.push_back(ConstantInt::get(REltTy, 0ULL));
      } else if (FC && isa<UndefValue>(FC)) {
        Elts.push_back(UndefValue::get(REltTy));
      } else {
        break;
      }
    }
    if (Elts.size() == NumElts)
      return ConstantVector::get(&Elts[0], Elts.size());
  }

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.push_back(LHS);
  ArgVec.push_back(RHS);
  // Get the key type with both the opcode and predicate
  const ExprMapKeyType Key(Instruction::VFCmp, ArgVec, pred);
  return ExprConstants->getOrCreate(ResultTy, Key);
}

Constant *ConstantExpr::getExtractElementTy(const Type *ReqTy, Constant *Val,
                                            Constant *Idx) {
  if (Constant *FC = ConstantFoldExtractElementInstruction(Val, Idx))
    return FC;          // Fold a few common cases...
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec(1, Val);
  ArgVec.push_back(Idx);
  const ExprMapKeyType Key(Instruction::ExtractElement,ArgVec);
  return ExprConstants->getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getExtractElement(Constant *Val, Constant *Idx) {
  assert(isa<VectorType>(Val->getType()) &&
         "Tried to create extractelement operation on non-vector type!");
  assert(Idx->getType() == Type::Int32Ty &&
         "Extractelement index must be i32 type!");
  return getExtractElementTy(cast<VectorType>(Val->getType())->getElementType(),
                             Val, Idx);
}

Constant *ConstantExpr::getInsertElementTy(const Type *ReqTy, Constant *Val,
                                           Constant *Elt, Constant *Idx) {
  if (Constant *FC = ConstantFoldInsertElementInstruction(Val, Elt, Idx))
    return FC;          // Fold a few common cases...
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec(1, Val);
  ArgVec.push_back(Elt);
  ArgVec.push_back(Idx);
  const ExprMapKeyType Key(Instruction::InsertElement,ArgVec);
  return ExprConstants->getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getInsertElement(Constant *Val, Constant *Elt, 
                                         Constant *Idx) {
  assert(isa<VectorType>(Val->getType()) &&
         "Tried to create insertelement operation on non-vector type!");
  assert(Elt->getType() == cast<VectorType>(Val->getType())->getElementType()
         && "Insertelement types must match!");
  assert(Idx->getType() == Type::Int32Ty &&
         "Insertelement index must be i32 type!");
  return getInsertElementTy(Val->getType(), Val, Elt, Idx);
}

Constant *ConstantExpr::getShuffleVectorTy(const Type *ReqTy, Constant *V1,
                                           Constant *V2, Constant *Mask) {
  if (Constant *FC = ConstantFoldShuffleVectorInstruction(V1, V2, Mask))
    return FC;          // Fold a few common cases...
  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec(1, V1);
  ArgVec.push_back(V2);
  ArgVec.push_back(Mask);
  const ExprMapKeyType Key(Instruction::ShuffleVector,ArgVec);
  return ExprConstants->getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getShuffleVector(Constant *V1, Constant *V2, 
                                         Constant *Mask) {
  assert(ShuffleVectorInst::isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector constant expr operands!");
  return getShuffleVectorTy(V1->getType(), V1, V2, Mask);
}

Constant *ConstantExpr::getInsertValueTy(const Type *ReqTy, Constant *Agg,
                                         Constant *Val,
                                        const unsigned *Idxs, unsigned NumIdx) {
  assert(ExtractValueInst::getIndexedType(Agg->getType(), Idxs,
                                          Idxs+NumIdx) == Val->getType() &&
         "insertvalue indices invalid!");
  assert(Agg->getType() == ReqTy &&
         "insertvalue type invalid!");
  assert(Agg->getType()->isFirstClassType() &&
         "Non-first-class type for constant InsertValue expression");
  Constant *FC = ConstantFoldInsertValueInstruction(Agg, Val, Idxs, NumIdx);
  assert(FC && "InsertValue constant expr couldn't be folded!");
  return FC;
}

Constant *ConstantExpr::getInsertValue(Constant *Agg, Constant *Val,
                                     const unsigned *IdxList, unsigned NumIdx) {
  assert(Agg->getType()->isFirstClassType() &&
         "Tried to create insertelement operation on non-first-class type!");

  const Type *ReqTy = Agg->getType();
#ifndef NDEBUG
  const Type *ValTy =
    ExtractValueInst::getIndexedType(Agg->getType(), IdxList, IdxList+NumIdx);
#endif
  assert(ValTy == Val->getType() && "insertvalue indices invalid!");
  return getInsertValueTy(ReqTy, Agg, Val, IdxList, NumIdx);
}

Constant *ConstantExpr::getExtractValueTy(const Type *ReqTy, Constant *Agg,
                                        const unsigned *Idxs, unsigned NumIdx) {
  assert(ExtractValueInst::getIndexedType(Agg->getType(), Idxs,
                                          Idxs+NumIdx) == ReqTy &&
         "extractvalue indices invalid!");
  assert(Agg->getType()->isFirstClassType() &&
         "Non-first-class type for constant extractvalue expression");
  Constant *FC = ConstantFoldExtractValueInstruction(Agg, Idxs, NumIdx);
  assert(FC && "ExtractValue constant expr couldn't be folded!");
  return FC;
}

Constant *ConstantExpr::getExtractValue(Constant *Agg,
                                     const unsigned *IdxList, unsigned NumIdx) {
  assert(Agg->getType()->isFirstClassType() &&
         "Tried to create extractelement operation on non-first-class type!");

  const Type *ReqTy =
    ExtractValueInst::getIndexedType(Agg->getType(), IdxList, IdxList+NumIdx);
  assert(ReqTy && "extractvalue indices invalid!");
  return getExtractValueTy(ReqTy, Agg, IdxList, NumIdx);
}

Constant *ConstantExpr::getZeroValueForNegationExpr(const Type *Ty) {
  if (const VectorType *PTy = dyn_cast<VectorType>(Ty))
    if (PTy->getElementType()->isFloatingPoint()) {
      std::vector<Constant*> zeros(PTy->getNumElements(),
                           ConstantFP::getNegativeZero(PTy->getElementType()));
      return ConstantVector::get(PTy, zeros);
    }

  if (Ty->isFloatingPoint()) 
    return ConstantFP::getNegativeZero(Ty);

  return Constant::getNullValue(Ty);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantExpr::destroyConstant() {
  ExprConstants->remove(this);
  destroyConstantImpl();
}

const char *ConstantExpr::getOpcodeName() const {
  return Instruction::getOpcodeName(getOpcode());
}

//===----------------------------------------------------------------------===//
//                replaceUsesOfWithOnConstant implementations

/// replaceUsesOfWithOnConstant - Update this constant array to change uses of
/// 'From' to be uses of 'To'.  This must update the uniquing data structures
/// etc.
///
/// Note that we intentionally replace all uses of From with To here.  Consider
/// a large array that uses 'From' 1000 times.  By handling this case all here,
/// ConstantArray::replaceUsesOfWithOnConstant is only invoked once, and that
/// single invocation handles all 1000 uses.  Handling them one at a time would
/// work, but would be really slow because it would have to unique each updated
/// array instance.
void ConstantArray::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  std::pair<ArrayConstantsTy::MapKey, Constant*> Lookup;
  Lookup.first.first = getType();
  Lookup.second = this;

  std::vector<Constant*> &Values = Lookup.first.second;
  Values.reserve(getNumOperands());  // Build replacement array.

  // Fill values with the modified operands of the constant array.  Also, 
  // compute whether this turns into an all-zeros array.
  bool isAllZeros = false;
  unsigned NumUpdated = 0;
  if (!ToC->isNullValue()) {
    for (Use *O = OperandList, *E = OperandList+getNumOperands(); O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      if (Val == From) {
        Val = ToC;
        ++NumUpdated;
      }
      Values.push_back(Val);
    }
  } else {
    isAllZeros = true;
    for (Use *O = OperandList, *E = OperandList+getNumOperands(); O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      if (Val == From) {
        Val = ToC;
        ++NumUpdated;
      }
      Values.push_back(Val);
      if (isAllZeros) isAllZeros = Val->isNullValue();
    }
  }
  
  Constant *Replacement = 0;
  if (isAllZeros) {
    Replacement = ConstantAggregateZero::get(getType());
  } else {
    // Check to see if we have this array type already.
    bool Exists;
    ArrayConstantsTy::MapTy::iterator I =
      ArrayConstants->InsertOrGetItem(Lookup, Exists);
    
    if (Exists) {
      Replacement = I->second;
    } else {
      // Okay, the new shape doesn't exist in the system yet.  Instead of
      // creating a new constant array, inserting it, replaceallusesof'ing the
      // old with the new, then deleting the old... just update the current one
      // in place!
      ArrayConstants->MoveConstantToNewSlot(this, I);
      
      // Update to the new value.  Optimize for the case when we have a single
      // operand that we're changing, but handle bulk updates efficiently.
      if (NumUpdated == 1) {
        unsigned OperandToUpdate = U-OperandList;
        assert(getOperand(OperandToUpdate) == From &&
               "ReplaceAllUsesWith broken!");
        setOperand(OperandToUpdate, ToC);
      } else {
        for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
          if (getOperand(i) == From)
            setOperand(i, ToC);
      }
      return;
    }
  }
 
  // Otherwise, I do need to replace this with an existing value.
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  uncheckedReplaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

void ConstantStruct::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                 Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  unsigned OperandToUpdate = U-OperandList;
  assert(getOperand(OperandToUpdate) == From && "ReplaceAllUsesWith broken!");

  std::pair<StructConstantsTy::MapKey, Constant*> Lookup;
  Lookup.first.first = getType();
  Lookup.second = this;
  std::vector<Constant*> &Values = Lookup.first.second;
  Values.reserve(getNumOperands());  // Build replacement struct.
  
  
  // Fill values with the modified operands of the constant struct.  Also, 
  // compute whether this turns into an all-zeros struct.
  bool isAllZeros = false;
  if (!ToC->isNullValue()) {
    for (Use *O = OperandList, *E = OperandList+getNumOperands(); O != E; ++O)
      Values.push_back(cast<Constant>(O->get()));
  } else {
    isAllZeros = true;
    for (Use *O = OperandList, *E = OperandList+getNumOperands(); O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      Values.push_back(Val);
      if (isAllZeros) isAllZeros = Val->isNullValue();
    }
  }
  Values[OperandToUpdate] = ToC;
  
  Constant *Replacement = 0;
  if (isAllZeros) {
    Replacement = ConstantAggregateZero::get(getType());
  } else {
    // Check to see if we have this array type already.
    bool Exists;
    StructConstantsTy::MapTy::iterator I =
      StructConstants->InsertOrGetItem(Lookup, Exists);
    
    if (Exists) {
      Replacement = I->second;
    } else {
      // Okay, the new shape doesn't exist in the system yet.  Instead of
      // creating a new constant struct, inserting it, replaceallusesof'ing the
      // old with the new, then deleting the old... just update the current one
      // in place!
      StructConstants->MoveConstantToNewSlot(this, I);
      
      // Update to the new value.
      setOperand(OperandToUpdate, ToC);
      return;
    }
  }
  
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  uncheckedReplaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

void ConstantVector::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                 Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  
  std::vector<Constant*> Values;
  Values.reserve(getNumOperands());  // Build replacement array...
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    Constant *Val = getOperand(i);
    if (Val == From) Val = cast<Constant>(To);
    Values.push_back(Val);
  }
  
  Constant *Replacement = ConstantVector::get(getType(), Values);
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  uncheckedReplaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

void ConstantExpr::replaceUsesOfWithOnConstant(Value *From, Value *ToV,
                                               Use *U) {
  assert(isa<Constant>(ToV) && "Cannot make Constant refer to non-constant!");
  Constant *To = cast<Constant>(ToV);
  
  Constant *Replacement = 0;
  if (getOpcode() == Instruction::GetElementPtr) {
    SmallVector<Constant*, 8> Indices;
    Constant *Pointer = getOperand(0);
    Indices.reserve(getNumOperands()-1);
    if (Pointer == From) Pointer = To;
    
    for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
      Constant *Val = getOperand(i);
      if (Val == From) Val = To;
      Indices.push_back(Val);
    }
    Replacement = ConstantExpr::getGetElementPtr(Pointer,
                                                 &Indices[0], Indices.size());
  } else if (getOpcode() == Instruction::ExtractValue) {
    Constant *Agg = getOperand(0);
    if (Agg == From) Agg = To;
    
    const SmallVector<unsigned, 4> &Indices = getIndices();
    Replacement = ConstantExpr::getExtractValue(Agg,
                                                &Indices[0], Indices.size());
  } else if (getOpcode() == Instruction::InsertValue) {
    Constant *Agg = getOperand(0);
    Constant *Val = getOperand(1);
    if (Agg == From) Agg = To;
    if (Val == From) Val = To;
    
    const SmallVector<unsigned, 4> &Indices = getIndices();
    Replacement = ConstantExpr::getInsertValue(Agg, Val,
                                               &Indices[0], Indices.size());
  } else if (isCast()) {
    assert(getOperand(0) == From && "Cast only has one use!");
    Replacement = ConstantExpr::getCast(getOpcode(), To, getType());
  } else if (getOpcode() == Instruction::Select) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    Constant *C3 = getOperand(2);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    if (C3 == From) C3 = To;
    Replacement = ConstantExpr::getSelect(C1, C2, C3);
  } else if (getOpcode() == Instruction::ExtractElement) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    Replacement = ConstantExpr::getExtractElement(C1, C2);
  } else if (getOpcode() == Instruction::InsertElement) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    Constant *C3 = getOperand(1);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    if (C3 == From) C3 = To;
    Replacement = ConstantExpr::getInsertElement(C1, C2, C3);
  } else if (getOpcode() == Instruction::ShuffleVector) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    Constant *C3 = getOperand(2);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    if (C3 == From) C3 = To;
    Replacement = ConstantExpr::getShuffleVector(C1, C2, C3);
  } else if (isCompare()) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    if (getOpcode() == Instruction::ICmp)
      Replacement = ConstantExpr::getICmp(getPredicate(), C1, C2);
    else if (getOpcode() == Instruction::FCmp)
      Replacement = ConstantExpr::getFCmp(getPredicate(), C1, C2);
    else if (getOpcode() == Instruction::VICmp)
      Replacement = ConstantExpr::getVICmp(getPredicate(), C1, C2);
    else {
      assert(getOpcode() == Instruction::VFCmp);
      Replacement = ConstantExpr::getVFCmp(getPredicate(), C1, C2);
    }
  } else if (getNumOperands() == 2) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    Replacement = ConstantExpr::get(getOpcode(), C1, C2);
  } else {
    assert(0 && "Unknown ConstantExpr type!");
    return;
  }
  
  assert(Replacement != this && "I didn't contain From!");
  
  // Everyone using this now uses the replacement.
  uncheckedReplaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}
