//===-- ConstantsContext.h - Constants-related Context Interals -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines various helper methods and classes used by
// LLVMContextImpl for creating and managing constants.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTANTSCONTEXT_H
#define LLVM_CONSTANTSCONTEXT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Operator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace llvm {
template<class ValType>
struct ConstantTraits;

/// UnaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement unary constant exprs.
class UnaryConstantExpr : public ConstantExpr {
  virtual void anchor();
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  UnaryConstantExpr(unsigned Opcode, Constant *C, Type *Ty)
    : ConstantExpr(Ty, Opcode, &Op<0>(), 1) {
    Op<0>() = C;
  }
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// BinaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement binary constant exprs.
class BinaryConstantExpr : public ConstantExpr {
  virtual void anchor();
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  BinaryConstantExpr(unsigned Opcode, Constant *C1, Constant *C2,
                     unsigned Flags)
    : ConstantExpr(C1->getType(), Opcode, &Op<0>(), 2) {
    Op<0>() = C1;
    Op<1>() = C2;
    SubclassOptionalData = Flags;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// SelectConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement select constant exprs.
class SelectConstantExpr : public ConstantExpr {
  virtual void anchor();
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
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
class ExtractElementConstantExpr : public ConstantExpr {
  virtual void anchor();
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
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
class InsertElementConstantExpr : public ConstantExpr {
  virtual void anchor();
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
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
class ShuffleVectorConstantExpr : public ConstantExpr {
  virtual void anchor();
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }
  ShuffleVectorConstantExpr(Constant *C1, Constant *C2, Constant *C3)
  : ConstantExpr(VectorType::get(
                   cast<VectorType>(C1->getType())->getElementType(),
                   cast<VectorType>(C3->getType())->getNumElements()),
                 Instruction::ShuffleVector, 
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
class ExtractValueConstantExpr : public ConstantExpr {
  virtual void anchor();
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  ExtractValueConstantExpr(Constant *Agg,
                           const SmallVector<unsigned, 4> &IdxList,
                           Type *DestTy)
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
class InsertValueConstantExpr : public ConstantExpr {
  virtual void anchor();
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  InsertValueConstantExpr(Constant *Agg, Constant *Val,
                          const SmallVector<unsigned, 4> &IdxList,
                          Type *DestTy)
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
class GetElementPtrConstantExpr : public ConstantExpr {
  virtual void anchor();
  GetElementPtrConstantExpr(Constant *C, ArrayRef<Constant*> IdxList,
                            Type *DestTy);
public:
  static GetElementPtrConstantExpr *Create(Constant *C,
                                           ArrayRef<Constant*> IdxList,
                                           Type *DestTy,
                                           unsigned Flags) {
    GetElementPtrConstantExpr *Result =
      new(IdxList.size() + 1) GetElementPtrConstantExpr(C, IdxList, DestTy);
    Result->SubclassOptionalData = Flags;
    return Result;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

// CompareConstantExpr - This class is private to Constants.cpp, and is used
// behind the scenes to implement ICmp and FCmp constant expressions. This is
// needed in order to store the predicate value for these instructions.
class CompareConstantExpr : public ConstantExpr {
  virtual void anchor();
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  unsigned short predicate;
  CompareConstantExpr(Type *ty, Instruction::OtherOps opc,
                      unsigned short pred,  Constant* LHS, Constant* RHS)
    : ConstantExpr(ty, opc, &Op<0>(), 2), predicate(pred) {
    Op<0>() = LHS;
    Op<1>() = RHS;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

template <>
struct OperandTraits<UnaryConstantExpr> :
  public FixedNumOperandTraits<UnaryConstantExpr, 1> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(UnaryConstantExpr, Value)

template <>
struct OperandTraits<BinaryConstantExpr> :
  public FixedNumOperandTraits<BinaryConstantExpr, 2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(BinaryConstantExpr, Value)

template <>
struct OperandTraits<SelectConstantExpr> :
  public FixedNumOperandTraits<SelectConstantExpr, 3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(SelectConstantExpr, Value)

template <>
struct OperandTraits<ExtractElementConstantExpr> :
  public FixedNumOperandTraits<ExtractElementConstantExpr, 2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractElementConstantExpr, Value)

template <>
struct OperandTraits<InsertElementConstantExpr> :
  public FixedNumOperandTraits<InsertElementConstantExpr, 3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertElementConstantExpr, Value)

template <>
struct OperandTraits<ShuffleVectorConstantExpr> :
    public FixedNumOperandTraits<ShuffleVectorConstantExpr, 3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ShuffleVectorConstantExpr, Value)

template <>
struct OperandTraits<ExtractValueConstantExpr> :
  public FixedNumOperandTraits<ExtractValueConstantExpr, 1> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractValueConstantExpr, Value)

template <>
struct OperandTraits<InsertValueConstantExpr> :
  public FixedNumOperandTraits<InsertValueConstantExpr, 2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertValueConstantExpr, Value)

template <>
struct OperandTraits<GetElementPtrConstantExpr> :
  public VariadicOperandTraits<GetElementPtrConstantExpr, 1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GetElementPtrConstantExpr, Value)


template <>
struct OperandTraits<CompareConstantExpr> :
  public FixedNumOperandTraits<CompareConstantExpr, 2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CompareConstantExpr, Value)

struct ExprMapKeyType {
  ExprMapKeyType(unsigned opc,
      ArrayRef<Constant*> ops,
      unsigned short flags = 0,
      unsigned short optionalflags = 0,
      ArrayRef<unsigned> inds = ArrayRef<unsigned>())
        : opcode(opc), subclassoptionaldata(optionalflags), subclassdata(flags),
        operands(ops.begin(), ops.end()), indices(inds.begin(), inds.end()) {}
  uint8_t opcode;
  uint8_t subclassoptionaldata;
  uint16_t subclassdata;
  std::vector<Constant*> operands;
  SmallVector<unsigned, 4> indices;
  bool operator==(const ExprMapKeyType& that) const {
    return this->opcode == that.opcode &&
           this->subclassdata == that.subclassdata &&
           this->subclassoptionaldata == that.subclassoptionaldata &&
           this->operands == that.operands &&
           this->indices == that.indices;
  }
  bool operator<(const ExprMapKeyType & that) const {
    if (this->opcode != that.opcode) return this->opcode < that.opcode;
    if (this->operands != that.operands) return this->operands < that.operands;
    if (this->subclassdata != that.subclassdata)
      return this->subclassdata < that.subclassdata;
    if (this->subclassoptionaldata != that.subclassoptionaldata)
      return this->subclassoptionaldata < that.subclassoptionaldata;
    if (this->indices != that.indices) return this->indices < that.indices;
    return false;
  }

  bool operator!=(const ExprMapKeyType& that) const {
    return !(*this == that);
  }
};

struct InlineAsmKeyType {
  InlineAsmKeyType(StringRef AsmString,
                   StringRef Constraints, bool hasSideEffects,
                   bool isAlignStack, InlineAsm::AsmDialect asmDialect)
    : asm_string(AsmString), constraints(Constraints),
      has_side_effects(hasSideEffects), is_align_stack(isAlignStack),
      asm_dialect(asmDialect) {}
  std::string asm_string;
  std::string constraints;
  bool has_side_effects;
  bool is_align_stack;
  InlineAsm::AsmDialect asm_dialect;
  bool operator==(const InlineAsmKeyType& that) const {
    return this->asm_string == that.asm_string &&
           this->constraints == that.constraints &&
           this->has_side_effects == that.has_side_effects &&
           this->is_align_stack == that.is_align_stack &&
           this->asm_dialect == that.asm_dialect;
  }
  bool operator<(const InlineAsmKeyType& that) const {
    if (this->asm_string != that.asm_string)
      return this->asm_string < that.asm_string;
    if (this->constraints != that.constraints)
      return this->constraints < that.constraints;
    if (this->has_side_effects != that.has_side_effects)
      return this->has_side_effects < that.has_side_effects;
    if (this->is_align_stack != that.is_align_stack)
      return this->is_align_stack < that.is_align_stack;
    if (this->asm_dialect != that.asm_dialect)
      return this->asm_dialect < that.asm_dialect;
    return false;
  }

  bool operator!=(const InlineAsmKeyType& that) const {
    return !(*this == that);
  }
};

// The number of operands for each ConstantCreator::create method is
// determined by the ConstantTraits template.
// ConstantCreator - A class that is used to create constants by
// ConstantUniqueMap*.  This class should be partially specialized if there is
// something strange that needs to be done to interface to the ctor for the
// constant.
//
template<typename T, typename Alloc>
struct ConstantTraits< std::vector<T, Alloc> > {
  static unsigned uses(const std::vector<T, Alloc>& v) {
    return v.size();
  }
};

template<>
struct ConstantTraits<Constant *> {
  static unsigned uses(Constant * const & v) {
    return 1;
  }
};

template<class ConstantClass, class TypeClass, class ValType>
struct ConstantCreator {
  static ConstantClass *create(TypeClass *Ty, const ValType &V) {
    return new(ConstantTraits<ValType>::uses(V)) ConstantClass(Ty, V);
  }
};

template<class ConstantClass, class TypeClass>
struct ConstantArrayCreator {
  static ConstantClass *create(TypeClass *Ty, ArrayRef<Constant*> V) {
    return new(V.size()) ConstantClass(Ty, V);
  }
};

template<class ConstantClass>
struct ConstantKeyData {
  typedef void ValType;
  static ValType getValType(ConstantClass *C) {
    llvm_unreachable("Unknown Constant type!");
  }
};

template<>
struct ConstantCreator<ConstantExpr, Type, ExprMapKeyType> {
  static ConstantExpr *create(Type *Ty, const ExprMapKeyType &V,
      unsigned short pred = 0) {
    if (Instruction::isCast(V.opcode))
      return new UnaryConstantExpr(V.opcode, V.operands[0], Ty);
    if ((V.opcode >= Instruction::BinaryOpsBegin &&
         V.opcode < Instruction::BinaryOpsEnd))
      return new BinaryConstantExpr(V.opcode, V.operands[0], V.operands[1],
                                    V.subclassoptionaldata);
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
      return GetElementPtrConstantExpr::Create(V.operands[0], IdxList, Ty,
                                               V.subclassoptionaldata);
    }

    // The compare instructions are weird. We have to encode the predicate
    // value and it is combined with the instruction opcode by multiplying
    // the opcode by one hundred. We must decode this to get the predicate.
    if (V.opcode == Instruction::ICmp)
      return new CompareConstantExpr(Ty, Instruction::ICmp, V.subclassdata,
                                     V.operands[0], V.operands[1]);
    if (V.opcode == Instruction::FCmp) 
      return new CompareConstantExpr(Ty, Instruction::FCmp, V.subclassdata,
                                     V.operands[0], V.operands[1]);
    llvm_unreachable("Invalid ConstantExpr!");
  }
};

template<>
struct ConstantKeyData<ConstantExpr> {
  typedef ExprMapKeyType ValType;
  static ValType getValType(ConstantExpr *CE) {
    std::vector<Constant*> Operands;
    Operands.reserve(CE->getNumOperands());
    for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i)
      Operands.push_back(cast<Constant>(CE->getOperand(i)));
    return ExprMapKeyType(CE->getOpcode(), Operands,
        CE->isCompare() ? CE->getPredicate() : 0,
        CE->getRawSubclassOptionalData(),
        CE->hasIndices() ?
          CE->getIndices() : ArrayRef<unsigned>());
  }
};

template<>
struct ConstantCreator<InlineAsm, PointerType, InlineAsmKeyType> {
  static InlineAsm *create(PointerType *Ty, const InlineAsmKeyType &Key) {
    return new InlineAsm(Ty, Key.asm_string, Key.constraints,
                         Key.has_side_effects, Key.is_align_stack,
                         Key.asm_dialect);
  }
};

template<>
struct ConstantKeyData<InlineAsm> {
  typedef InlineAsmKeyType ValType;
  static ValType getValType(InlineAsm *Asm) {
    return InlineAsmKeyType(Asm->getAsmString(), Asm->getConstraintString(),
                            Asm->hasSideEffects(), Asm->isAlignStack(),
                            Asm->getDialect());
  }
};

template<class ValType, class ValRefType, class TypeClass, class ConstantClass,
         bool HasLargeKey = false /*true for arrays and structs*/ >
class ConstantUniqueMap {
public:
  typedef std::pair<TypeClass*, ValType> MapKey;
  typedef std::map<MapKey, ConstantClass *> MapTy;
  typedef std::map<ConstantClass *, typename MapTy::iterator> InverseMapTy;
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

public:
  typename MapTy::iterator map_begin() { return Map.begin(); }
  typename MapTy::iterator map_end() { return Map.end(); }

  void freeConstants() {
    for (typename MapTy::iterator I=Map.begin(), E=Map.end();
         I != E; ++I) {
      // Asserts that use_empty().
      delete I->second;
    }
  }
    
  /// InsertOrGetItem - Return an iterator for the specified element.
  /// If the element exists in the map, the returned iterator points to the
  /// entry and Exists=true.  If not, the iterator points to the newly
  /// inserted entry and returns Exists=false.  Newly inserted entries have
  /// I->second == 0, and should be filled in.
  typename MapTy::iterator InsertOrGetItem(std::pair<MapKey, ConstantClass *>
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
      Map.find(MapKey(static_cast<TypeClass*>(CP->getType()),
                      ConstantKeyData<ConstantClass>::getValType(CP)));
    if (I == Map.end() || I->second != CP) {
      // FIXME: This should not use a linear scan.  If this gets to be a
      // performance problem, someone should look at this.
      for (I = Map.begin(); I != Map.end() && I->second != CP; ++I)
        /* empty */;
    }
    return I;
  }

  ConstantClass *Create(TypeClass *Ty, ValRefType V,
                        typename MapTy::iterator I) {
    ConstantClass* Result =
      ConstantCreator<ConstantClass,TypeClass,ValType>::create(Ty, V);

    assert(Result->getType() == Ty && "Type specified is not correct!");
    I = Map.insert(I, std::make_pair(MapKey(Ty, V), Result));

    if (HasLargeKey)  // Remember the reverse mapping if needed.
      InverseMap.insert(std::make_pair(Result, I));

    return Result;
  }
public:
    
  /// getOrCreate - Return the specified constant from the map, creating it if
  /// necessary.
  ConstantClass *getOrCreate(TypeClass *Ty, ValRefType V) {
    MapKey Lookup(Ty, V);
    ConstantClass* Result = 0;
    
    typename MapTy::iterator I = Map.find(Lookup);
    // Is it in the map?  
    if (I != Map.end())
      Result = I->second;
        
    if (!Result) {
      // If no preexisting value, create one now...
      Result = Create(Ty, V, I);
    }
        
    return Result;
  }

  void remove(ConstantClass *CP) {
    typename MapTy::iterator I = FindExistingElement(CP);
    assert(I != Map.end() && "Constant not found in constant table!");
    assert(I->second == CP && "Didn't find correct element?");

    if (HasLargeKey)  // Remember the reverse mapping if needed.
      InverseMap.erase(CP);

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
      
     // Remove the old entry from the map.
    Map.erase(OldI);
    
    // Update the inverse map so that we know that this constant is now
    // located at descriptor I.
    if (HasLargeKey) {
      assert(I->second == C && "Bad inversemap entry!");
      InverseMap[C] = I;
    }
  }

  void dump() const {
    DEBUG(dbgs() << "Constant.cpp: ConstantUniqueMap\n");
  }
};

// Unique map for aggregate constants
template<class TypeClass, class ConstantClass>
class ConstantAggrUniqueMap {
public:
  typedef ArrayRef<Constant*> Operands;
  typedef std::pair<TypeClass*, Operands> LookupKey;
private:
  struct MapInfo {
    typedef DenseMapInfo<ConstantClass*> ConstantClassInfo;
    typedef DenseMapInfo<Constant*> ConstantInfo;
    typedef DenseMapInfo<TypeClass*> TypeClassInfo;
    static inline ConstantClass* getEmptyKey() {
      return ConstantClassInfo::getEmptyKey();
    }
    static inline ConstantClass* getTombstoneKey() {
      return ConstantClassInfo::getTombstoneKey();
    }
    static unsigned getHashValue(const ConstantClass *CP) {
      SmallVector<Constant*, 8> CPOperands;
      CPOperands.reserve(CP->getNumOperands());
      for (unsigned I = 0, E = CP->getNumOperands(); I < E; ++I)
        CPOperands.push_back(CP->getOperand(I));
      return getHashValue(LookupKey(CP->getType(), CPOperands));
    }
    static bool isEqual(const ConstantClass *LHS, const ConstantClass *RHS) {
      return LHS == RHS;
    }
    static unsigned getHashValue(const LookupKey &Val) {
      return hash_combine(Val.first, hash_combine_range(Val.second.begin(),
                                                        Val.second.end()));
    }
    static bool isEqual(const LookupKey &LHS, const ConstantClass *RHS) {
      if (RHS == getEmptyKey() || RHS == getTombstoneKey())
        return false;
      if (LHS.first != RHS->getType()
          || LHS.second.size() != RHS->getNumOperands())
        return false;
      for (unsigned I = 0, E = RHS->getNumOperands(); I < E; ++I) {
        if (LHS.second[I] != RHS->getOperand(I))
          return false;
      }
      return true;
    }
  };
public:
  typedef DenseMap<ConstantClass *, char, MapInfo> MapTy;

private:
  /// Map - This is the main map from the element descriptor to the Constants.
  /// This is the primary way we avoid creating two of the same shape
  /// constant.
  MapTy Map;

public:
  typename MapTy::iterator map_begin() { return Map.begin(); }
  typename MapTy::iterator map_end() { return Map.end(); }

  void freeConstants() {
    for (typename MapTy::iterator I=Map.begin(), E=Map.end();
         I != E; ++I) {
      // Asserts that use_empty().
      delete I->first;
    }
  }

private:
  typename MapTy::iterator findExistingElement(ConstantClass *CP) {
    return Map.find(CP);
  }

  ConstantClass *Create(TypeClass *Ty, Operands V, typename MapTy::iterator I) {
    ConstantClass* Result =
      ConstantArrayCreator<ConstantClass,TypeClass>::create(Ty, V);

    assert(Result->getType() == Ty && "Type specified is not correct!");
    Map[Result] = '\0';

    return Result;
  }
public:

  /// getOrCreate - Return the specified constant from the map, creating it if
  /// necessary.
  ConstantClass *getOrCreate(TypeClass *Ty, Operands V) {
    LookupKey Lookup(Ty, V);
    ConstantClass* Result = 0;

    typename MapTy::iterator I = Map.find_as(Lookup);
    // Is it in the map?
    if (I != Map.end())
      Result = I->first;

    if (!Result) {
      // If no preexisting value, create one now...
      Result = Create(Ty, V, I);
    }

    return Result;
  }

  /// Find the constant by lookup key.
  typename MapTy::iterator find(LookupKey Lookup) {
    return Map.find_as(Lookup);
  }

  /// Insert the constant into its proper slot.
  void insert(ConstantClass *CP) {
    Map[CP] = '\0';
  }

  /// Remove this constant from the map
  void remove(ConstantClass *CP) {
    typename MapTy::iterator I = findExistingElement(CP);
    assert(I != Map.end() && "Constant not found in constant table!");
    assert(I->first == CP && "Didn't find correct element?");
    Map.erase(I);
  }

  void dump() const {
    DEBUG(dbgs() << "Constant.cpp: ConstantUniqueMap\n");
  }
};

}

#endif
