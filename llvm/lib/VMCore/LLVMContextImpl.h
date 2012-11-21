//===-- LLVMContextImpl.h - The LLVMContextImpl opaque class ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declares LLVMContextImpl, the opaque implementation 
//  of LLVMContext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLVMCONTEXT_IMPL_H
#define LLVM_LLVMCONTEXT_IMPL_H

#include "llvm/LLVMContext.h"
#include "AttributesImpl.h"
#include "ConstantsContext.h"
#include "LeaksContext.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Metadata.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Hashing.h"
#include <vector>

namespace llvm {

class ConstantInt;
class ConstantFP;
class LLVMContext;
class Type;
class Value;

struct DenseMapAPIntKeyInfo {
  struct KeyTy {
    APInt val;
    Type* type;
    KeyTy(const APInt& V, Type* Ty) : val(V), type(Ty) {}
    KeyTy(const KeyTy& that) : val(that.val), type(that.type) {}
    bool operator==(const KeyTy& that) const {
      return type == that.type && this->val == that.val;
    }
    bool operator!=(const KeyTy& that) const {
      return !this->operator==(that);
    }
    friend hash_code hash_value(const KeyTy &Key) {
      return hash_combine(Key.type, Key.val);
    }
  };
  static inline KeyTy getEmptyKey() { return KeyTy(APInt(1,0), 0); }
  static inline KeyTy getTombstoneKey() { return KeyTy(APInt(1,1), 0); }
  static unsigned getHashValue(const KeyTy &Key) {
    return static_cast<unsigned>(hash_value(Key));
  }
  static bool isEqual(const KeyTy &LHS, const KeyTy &RHS) {
    return LHS == RHS;
  }
};

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
    friend hash_code hash_value(const KeyTy &Key) {
      return hash_combine(Key.val);
    }
  };
  static inline KeyTy getEmptyKey() { 
    return KeyTy(APFloat(APFloat::Bogus,1));
  }
  static inline KeyTy getTombstoneKey() { 
    return KeyTy(APFloat(APFloat::Bogus,2)); 
  }
  static unsigned getHashValue(const KeyTy &Key) {
    return static_cast<unsigned>(hash_value(Key));
  }
  static bool isEqual(const KeyTy &LHS, const KeyTy &RHS) {
    return LHS == RHS;
  }
};

struct AnonStructTypeKeyInfo {
  struct KeyTy {
    ArrayRef<Type*> ETypes;
    bool isPacked;
    KeyTy(const ArrayRef<Type*>& E, bool P) :
      ETypes(E), isPacked(P) {}
    KeyTy(const KeyTy& that) :
      ETypes(that.ETypes), isPacked(that.isPacked) {}
    KeyTy(const StructType* ST) :
      ETypes(ArrayRef<Type*>(ST->element_begin(), ST->element_end())),
      isPacked(ST->isPacked()) {}
    bool operator==(const KeyTy& that) const {
      if (isPacked != that.isPacked)
        return false;
      if (ETypes != that.ETypes)
        return false;
      return true;
    }
    bool operator!=(const KeyTy& that) const {
      return !this->operator==(that);
    }
  };
  static inline StructType* getEmptyKey() {
    return DenseMapInfo<StructType*>::getEmptyKey();
  }
  static inline StructType* getTombstoneKey() {
    return DenseMapInfo<StructType*>::getTombstoneKey();
  }
  static unsigned getHashValue(const KeyTy& Key) {
    return hash_combine(hash_combine_range(Key.ETypes.begin(),
                                           Key.ETypes.end()),
                        Key.isPacked);
  }
  static unsigned getHashValue(const StructType *ST) {
    return getHashValue(KeyTy(ST));
  }
  static bool isEqual(const KeyTy& LHS, const StructType *RHS) {
    if (RHS == getEmptyKey() || RHS == getTombstoneKey())
      return false;
    return LHS == KeyTy(RHS);
  }
  static bool isEqual(const StructType *LHS, const StructType *RHS) {
    return LHS == RHS;
  }
};

struct FunctionTypeKeyInfo {
  struct KeyTy {
    const Type *ReturnType;
    ArrayRef<Type*> Params;
    bool isVarArg;
    KeyTy(const Type* R, const ArrayRef<Type*>& P, bool V) :
      ReturnType(R), Params(P), isVarArg(V) {}
    KeyTy(const KeyTy& that) :
      ReturnType(that.ReturnType),
      Params(that.Params),
      isVarArg(that.isVarArg) {}
    KeyTy(const FunctionType* FT) :
      ReturnType(FT->getReturnType()),
      Params(ArrayRef<Type*>(FT->param_begin(), FT->param_end())),
      isVarArg(FT->isVarArg()) {}
    bool operator==(const KeyTy& that) const {
      if (ReturnType != that.ReturnType)
        return false;
      if (isVarArg != that.isVarArg)
        return false;
      if (Params != that.Params)
        return false;
      return true;
    }
    bool operator!=(const KeyTy& that) const {
      return !this->operator==(that);
    }
  };
  static inline FunctionType* getEmptyKey() {
    return DenseMapInfo<FunctionType*>::getEmptyKey();
  }
  static inline FunctionType* getTombstoneKey() {
    return DenseMapInfo<FunctionType*>::getTombstoneKey();
  }
  static unsigned getHashValue(const KeyTy& Key) {
    return hash_combine(Key.ReturnType,
                        hash_combine_range(Key.Params.begin(),
                                           Key.Params.end()),
                        Key.isVarArg);
  }
  static unsigned getHashValue(const FunctionType *FT) {
    return getHashValue(KeyTy(FT));
  }
  static bool isEqual(const KeyTy& LHS, const FunctionType *RHS) {
    if (RHS == getEmptyKey() || RHS == getTombstoneKey())
      return false;
    return LHS == KeyTy(RHS);
  }
  static bool isEqual(const FunctionType *LHS, const FunctionType *RHS) {
    return LHS == RHS;
  }
};

// Provide a FoldingSetTrait::Equals specialization for MDNode that can use a
// shortcut to avoid comparing all operands.
template<> struct FoldingSetTrait<MDNode> : DefaultFoldingSetTrait<MDNode> {
  static bool Equals(const MDNode &X, const FoldingSetNodeID &ID,
                     unsigned IDHash, FoldingSetNodeID &TempID) {
    assert(!X.isNotUniqued() && "Non-uniqued MDNode in FoldingSet?");
    // First, check if the cached hashes match.  If they don't we can skip the
    // expensive operand walk.
    if (X.Hash != IDHash)
      return false;

    // If they match we have to compare the operands.
    X.Profile(TempID);
    return TempID == ID;
  }
  static unsigned ComputeHash(const MDNode &X, FoldingSetNodeID &) {
    return X.Hash; // Return cached hash.
  }
};

/// DebugRecVH - This is a CallbackVH used to keep the Scope -> index maps
/// up to date as MDNodes mutate.  This class is implemented in DebugLoc.cpp.
class DebugRecVH : public CallbackVH {
  /// Ctx - This is the LLVM Context being referenced.
  LLVMContextImpl *Ctx;
  
  /// Idx - The index into either ScopeRecordIdx or ScopeInlinedAtRecords that
  /// this reference lives in.  If this is zero, then it represents a
  /// non-canonical entry that has no DenseMap value.  This can happen due to
  /// RAUW.
  int Idx;
public:
  DebugRecVH(MDNode *n, LLVMContextImpl *ctx, int idx)
    : CallbackVH(n), Ctx(ctx), Idx(idx) {}
  
  MDNode *get() const {
    return cast_or_null<MDNode>(getValPtr());
  }
  
  virtual void deleted();
  virtual void allUsesReplacedWith(Value *VNew);
};
  
class LLVMContextImpl {
public:
  /// OwnedModules - The set of modules instantiated in this context, and which
  /// will be automatically deleted if this context is deleted.
  SmallPtrSet<Module*, 4> OwnedModules;
  
  LLVMContext::InlineAsmDiagHandlerTy InlineAsmDiagHandler;
  void *InlineAsmDiagContext;
  
  typedef DenseMap<DenseMapAPIntKeyInfo::KeyTy, ConstantInt*, 
                         DenseMapAPIntKeyInfo> IntMapTy;
  IntMapTy IntConstants;
  
  typedef DenseMap<DenseMapAPFloatKeyInfo::KeyTy, ConstantFP*, 
                         DenseMapAPFloatKeyInfo> FPMapTy;
  FPMapTy FPConstants;

  FoldingSet<AttributesImpl> AttrsSet;
  
  StringMap<Value*> MDStringCache;

  FoldingSet<MDNode> MDNodeSet;

  // MDNodes may be uniqued or not uniqued.  When they're not uniqued, they
  // aren't in the MDNodeSet, but they're still shared between objects, so no
  // one object can destroy them.  This set allows us to at least destroy them
  // on Context destruction.
  SmallPtrSet<MDNode*, 1> NonUniquedMDNodes;
  
  DenseMap<Type*, ConstantAggregateZero*> CAZConstants;

  typedef ConstantAggrUniqueMap<ArrayType, ConstantArray> ArrayConstantsTy;
  ArrayConstantsTy ArrayConstants;
  
  typedef ConstantAggrUniqueMap<StructType, ConstantStruct> StructConstantsTy;
  StructConstantsTy StructConstants;
  
  typedef ConstantAggrUniqueMap<VectorType, ConstantVector> VectorConstantsTy;
  VectorConstantsTy VectorConstants;
  
  DenseMap<PointerType*, ConstantPointerNull*> CPNConstants;

  DenseMap<Type*, UndefValue*> UVConstants;
  
  StringMap<ConstantDataSequential*> CDSConstants;

  
  DenseMap<std::pair<Function*, BasicBlock*> , BlockAddress*> BlockAddresses;
  ConstantUniqueMap<ExprMapKeyType, const ExprMapKeyType&, Type, ConstantExpr>
    ExprConstants;

  ConstantUniqueMap<InlineAsmKeyType, const InlineAsmKeyType&, PointerType,
                    InlineAsm> InlineAsms;
  
  ConstantInt *TheTrueVal;
  ConstantInt *TheFalseVal;
  
  LeakDetectorImpl<Value> LLVMObjects;
  
  // Basic type instances.
  Type VoidTy, LabelTy, HalfTy, FloatTy, DoubleTy, MetadataTy;
  Type X86_FP80Ty, FP128Ty, PPC_FP128Ty, X86_MMXTy;
  IntegerType Int1Ty, Int8Ty, Int16Ty, Int32Ty, Int64Ty;

  
  /// TypeAllocator - All dynamically allocated types are allocated from this.
  /// They live forever until the context is torn down.
  BumpPtrAllocator TypeAllocator;
  
  DenseMap<unsigned, IntegerType*> IntegerTypes;
  
  typedef DenseMap<FunctionType*, bool, FunctionTypeKeyInfo> FunctionTypeMap;
  FunctionTypeMap FunctionTypes;
  typedef DenseMap<StructType*, bool, AnonStructTypeKeyInfo> StructTypeMap;
  StructTypeMap AnonStructTypes;
  StringMap<StructType*> NamedStructTypes;
  unsigned NamedStructTypesUniqueID;
    
  DenseMap<std::pair<Type *, uint64_t>, ArrayType*> ArrayTypes;
  DenseMap<std::pair<Type *, unsigned>, VectorType*> VectorTypes;
  DenseMap<Type*, PointerType*> PointerTypes;  // Pointers in AddrSpace = 0
  DenseMap<std::pair<Type*, unsigned>, PointerType*> ASPointerTypes;


  /// ValueHandles - This map keeps track of all of the value handles that are
  /// watching a Value*.  The Value::HasValueHandle bit is used to know
  // whether or not a value has an entry in this map.
  typedef DenseMap<Value*, ValueHandleBase*> ValueHandlesTy;
  ValueHandlesTy ValueHandles;
  
  /// CustomMDKindNames - Map to hold the metadata string to ID mapping.
  StringMap<unsigned> CustomMDKindNames;
  
  typedef std::pair<unsigned, TrackingVH<MDNode> > MDPairTy;
  typedef SmallVector<MDPairTy, 2> MDMapTy;

  /// MetadataStore - Collection of per-instruction metadata used in this
  /// context.
  DenseMap<const Instruction *, MDMapTy> MetadataStore;
  
  /// ScopeRecordIdx - This is the index in ScopeRecords for an MDNode scope
  /// entry with no "inlined at" element.
  DenseMap<MDNode*, int> ScopeRecordIdx;
  
  /// ScopeRecords - These are the actual mdnodes (in a value handle) for an
  /// index.  The ValueHandle ensures that ScopeRecordIdx stays up to date if
  /// the MDNode is RAUW'd.
  std::vector<DebugRecVH> ScopeRecords;
  
  /// ScopeInlinedAtIdx - This is the index in ScopeInlinedAtRecords for an
  /// scope/inlined-at pair.
  DenseMap<std::pair<MDNode*, MDNode*>, int> ScopeInlinedAtIdx;
  
  /// ScopeInlinedAtRecords - These are the actual mdnodes (in value handles)
  /// for an index.  The ValueHandle ensures that ScopeINlinedAtIdx stays up
  /// to date.
  std::vector<std::pair<DebugRecVH, DebugRecVH> > ScopeInlinedAtRecords;
  
  int getOrAddScopeRecordIdxEntry(MDNode *N, int ExistingIdx);
  int getOrAddScopeInlinedAtIdxEntry(MDNode *Scope, MDNode *IA,int ExistingIdx);
  
  LLVMContextImpl(LLVMContext &C);
  ~LLVMContextImpl();
};

}

#endif
