//===- SPIRVTypeAnalysis.h -----------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis links a type information to every register/pointer, allowing
// us to legalize type mismatches when required (graphical SPIR-V pointers for
// ex).
//
//===----------------------------------------------------------------------===//

#include "SPIRVTypeAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/InitializePasses.h"

#include <optional>
#include <queue>

#define DEBUG_TYPE "spirv-type-analysis"

using namespace llvm;

namespace llvm {
void initializeSPIRVTypeAnalysisWrapperPassPass(PassRegistry &);
} // namespace llvm

INITIALIZE_PASS_BEGIN(SPIRVTypeAnalysisWrapperPass, "type-analysis",
                      "SPIRV type analysis", true, true)
INITIALIZE_PASS_END(SPIRVTypeAnalysisWrapperPass, "type-region",
                    "SPIRV type analysis", true, true)

namespace llvm {
namespace SPIRV {
namespace {} // anonymous namespace

// Returns true this type contains no opaque pointers (recursively).
bool TypeInfo::isOpaqueType(const Type *T) {
  if (T->isPointerTy())
    return true;

  if (const ArrayType *AT = dyn_cast<ArrayType>(T))
    return TypeInfo::isOpaqueType(AT->getElementType());
  if (const VectorType *VT = dyn_cast<VectorType>(T))
    return TypeInfo::isOpaqueType(VT->getElementType());

  return false;
}

class TypeAnalyzer {

public:
  TypeAnalyzer(Module &M)
      : M(M), TypeMap(new DenseMap<const Value *, Type *>()) {}

  TypeInfo analyze() {
    for (const Function &F : M) {
      for (const BasicBlock &BB : F) {
        for (const Value &V : BB) {
          if (!deduceElementType(&V))
            IncompleteTypeDefinition.insert(&V);
        }
      }
    }

    size_t IncompleteCount;
    do {
      IncompleteCount = IncompleteTypeDefinition.size();
      for (const Value *Item : IncompleteTypeDefinition) {
        if (deduceElementType(Item)) {
          IncompleteTypeDefinition.erase(Item);
          break;
        }
      }
    } while (IncompleteTypeDefinition.size() < IncompleteCount);

    return TypeInfo(TypeMap);
  }

private:
  Type *getMappedType(const Value *V) {
    auto It = TypeMap->find(V);
    if (It == TypeMap->end())
      return nullptr;
    return It->second;
  }

  bool typeContainsType(Type *Wrapper, Type *Needle) {
    if (Wrapper == Needle)
      return true;

    TypedPointerType *LP = dyn_cast<TypedPointerType>(Wrapper);
    TypedPointerType *RP = dyn_cast<TypedPointerType>(Needle);
    if (LP && RP)
      return typeContainsType(LP->getElementType(), RP->getElementType());

    if (StructType *ST = dyn_cast<StructType>(Wrapper))
      return typeContainsType(ST->getElementType(0), Needle);
    if (ArrayType *AT = dyn_cast<ArrayType>(Wrapper))
      return typeContainsType(AT->getElementType(), Needle);
    if (VectorType *VT = dyn_cast<VectorType>(Wrapper))
      return typeContainsType(VT->getElementType(), Needle);

    return false;
  }

  Type *resolveTypeConflict(TypedPointerType *A, TypedPointerType *B) {
    if (typeContainsType(A->getElementType(), B->getElementType()))
      return A;
    if (typeContainsType(B->getElementType(), A->getElementType()))
      return B;
    return nullptr;
  }

  void propagateType(Type *DeducedType, const Value *V) {
    assert(!TypeInfo::isOpaqueType(DeducedType));

    auto It = TypeMap->find(V);
    // The value type has already been deduced.
    if (It != TypeMap->end()) {
      // There is no conflict.
      if (DeducedType == It->second)
        return;

      TypedPointerType *DeducedPtrType =
          dyn_cast<TypedPointerType>(DeducedType);
      TypedPointerType *KnownPtrType = dyn_cast<TypedPointerType>(It->second);
      // Cannot resolve conflict on non-pointer types.
      if (!DeducedPtrType || !KnownPtrType) {
        assert(0); // FIXME: shall I ignore, fail, crash?
        return;
      }

      DeducedType = resolveTypeConflict(DeducedPtrType, KnownPtrType);
      if (!DeducedType)
        return;
      (*TypeMap)[V] = DeducedType;
    }

    if (const Constant *C = dyn_cast<Constant>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const Argument *C = dyn_cast<Argument>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const AllocaInst *C = dyn_cast<AllocaInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const CallInst *C = dyn_cast<CallInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const GetElementPtrInst *C = dyn_cast<GetElementPtrInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const LoadInst *C = dyn_cast<LoadInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const ReturnInst *C = dyn_cast<ReturnInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const StoreInst *C = dyn_cast<StoreInst>(V))
      propagateTypeDetails(DeducedType, C);
    else
      llvm_unreachable("FIXME: unsupported instruction");

    // for (const User *U : V->users())
    //   if (TypeMap->find(U) == TypeMap->end())
    //     deduceElementType(U);

    // X(CallInst, V);
    // X(GetElementPtrInst, V);
    // X(LoadInst, V);
    // X(ReturnInst, V);
    // X(StoreInst, V);
    //  TODO:  GlobalValue
    //  TODO: addrspacecast
    //  TODO: bitcast
    //  TODO: AtomicCmpXchgInst
    //  TODO: AtomicRMWInst
    //  TODO: PHINode
    //  TODO: SelectInst
    //  TODO: CallInst
  }

  void propagateTypeDetails(Type *DeducedType, const Constant *C) {
    (*TypeMap)[C] = DeducedType;
  }

  void propagateTypeDetails(Type *DeducedType, const Argument *A) {
    (*TypeMap)[A] = DeducedType;

    unsigned ArgNo = A->getArgNo();
    for (const User *U : A->getParent()->users()) {
      const CallInst *CI = cast<CallInst>(U);
      propagateType(DeducedType, CI->getOperand(ArgNo));
    }
  }

  void propagateTypeDetails(Type *DeducedType, const LoadInst *I) {
    TypeMap->try_emplace(I, DeducedType);
    propagateType(TypedPointerType::get(DeducedType, 0),
                  I->getPointerOperand());
  }

  void propagateTypeDetails(Type *DeducedType, const StoreInst *I) {
    TypeMap->try_emplace(I, DeducedType);
  }

  void propagateTypeDetails(Type *DeducedType, const AllocaInst *I) {
    Type *StoredType = cast<TypedPointerType>(DeducedType)->getElementType();
    if (ArrayType *AT = dyn_cast<ArrayType>(I->getAllocatedType())) {
      Type *NewType = ArrayType::get(StoredType, AT->getNumElements());
      TypeMap->try_emplace(I, NewType);
      return;
    }

    TypeMap->try_emplace(I, DeducedType);
  }

  void propagateTypeDetails(Type *DeducedType, const GetElementPtrInst *GEP) {
    if (!TypeInfo::isOpaqueType(GEP->getSourceElementType())) {
      // If the source is non-opaque, the result must be non-opaque (subset of
      // source). If not, this means the GEP is using the wrong base-type. We
      // don't support this.
      assert(!TypeInfo::isOpaqueType(GEP->getResultElementType()));
      // If the result is non-opaque, this means each use was non-opaque, and
      // thus we shouldn't have a deduction mismatch. If this happens, something
      // is wrong with this analysis.
      TypedPointerType *DeducedPtr = cast<TypedPointerType>(DeducedType);
      assert(GEP->getResultElementType() == DeducedPtr->getElementType());
      propagateType(TypedPointerType::get(GEP->getSourceElementType(), 0),
                    GEP->getPointerOperand());
      return;
    }

    TypeMap->try_emplace(GEP, DeducedType);

    // The source type is opaque. We might be able to deduce more info from the
    // new result type.
    Type *NewType = DeducedType;
    std::vector<Type *> Types = {GEP->getSourceElementType()};
    std::vector<uint64_t> Indices;
    for (const Use &U : GEP->indices())
      Indices.push_back(cast<ConstantInt>(&*U)->getZExtValue());

    for (unsigned I = 1; I < Indices.size(); ++I)
      Types.push_back(
          GetElementPtrInst::getTypeAtIndex(Types[I - 1], Indices[I]));

    for (unsigned I = 1; I < GEP->getNumIndices(); ++I) {
      unsigned Index = GEP->getNumIndices() - 1 - I;
      Type *T = Types[Index];
      if (T->isPointerTy())
        return;

      if (ArrayType *AT = dyn_cast<ArrayType>(T))
        NewType = ArrayType::get(NewType, AT->getNumElements());
      else if (VectorType *VT = dyn_cast<VectorType>(T))
        NewType = VectorType::get(NewType, VT->getElementCount());
      else if (StructType *ST = dyn_cast<StructType>(T))
        assert(0 && "Opaque struct types are not supported.");
      else
        llvm_unreachable("Unsupported aggregate type?");
    }

    // The first index of a GEP is indexing from the passed pointer. So we need
    // to add one layer.
    NewType = TypedPointerType::get(NewType, 0);

    propagateType(NewType, GEP->getPointerOperand());
  }

  void propagateTypeDetails(Type *DeducedType, const CallInst *CI) {
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI))
      llvm_unreachable("Not implemented");
    // return deduceType(II);

    FunctionType *FT = CI->getFunctionType();
    if (!TypeInfo::isOpaqueType(FT->getReturnType())) {
      assert(FT->getReturnType() == DeducedType);
      return;
    }

    TypeMap->try_emplace(CI, DeducedType);
    propagateType(DeducedType, CI->getCalledFunction());

    for (const BasicBlock &BB : *CI->getCalledFunction())
      for (const Instruction &I : BB)
        if (const ReturnInst *RI = dyn_cast<ReturnInst>(&I))
          propagateType(DeducedType, RI);
  }

  void propagateTypeDetails(Type *DeducedType, const ReturnInst *RI) {
    Value *RV = RI->getReturnValue();
    assert(RV || DeducedType->isVoidTy());

    TypeMap->try_emplace(RI, DeducedType);
    propagateType(DeducedType, RV);
  }

  bool deduceElementType(const Value *V) {
    assert(V != nullptr);

    auto It = TypeMap->find(V);
    if (It != TypeMap->end())
      return true;

#define X(Type, Value)                                                         \
  if (auto *Casted = dyn_cast<Type>(Value))                                    \
    return deduceType(Casted);

    X(AllocaInst, V);
    X(CallInst, V);
    X(GetElementPtrInst, V);
    X(LoadInst, V);
    X(ReturnInst, V);
    X(StoreInst, V);
    // TODO:  GlobalValue
    // TODO: addrspacecast
    // TODO: bitcast
    // TODO: AtomicCmpXchgInst
    // TODO: AtomicRMWInst
    // TODO: PHINode
    // TODO: SelectInst
    // TODO: CallInst
#undef X

    llvm_unreachable("FIXME: unsupported instruction");
    return false;
  }

  bool deduceType(const AllocaInst *I) {
    if (TypeInfo::isOpaqueType(I->getAllocatedType()))
      return false;
    TypeMap->try_emplace(I, TypedPointerType::get(I->getAllocatedType(), 0));
    return true;
  }

  bool deduceType(const LoadInst *I) {
    // First case: the loaded type is complete: we can assign the result type.
    if (!TypeInfo::isOpaqueType(I->getType())) {
      TypeMap->try_emplace(I, I->getType());
      propagateType(TypedPointerType::get(I->getType(), 0),
                    I->getPointerOperand());
      return true;
    }

    // The pointer operand is non-opaque, we can deduce the loaded type.
    if (Type *PointerOperandTy = getMappedType(I->getPointerOperand())) {
      // FIXME: only supports pointer of pointers for now.
      Type *ElementType =
          cast<TypedPointerType>(PointerOperandTy)->getElementType();
      assert(!ElementType->isPointerTy());
      TypeMap->try_emplace(I, ElementType);
      return true;
    }

    return false;
  }

  Type *getDeducedType(const Value *V) {
    auto It = TypeMap->find(V);
    return It == TypeMap->end() ? nullptr : It->second;
  }

  bool deduceType(const StoreInst *I) {
    Type *ValueType = I->getValueOperand()->getType();
    Type *SourceType = getDeducedType(I->getPointerOperand());
    bool isOpaqueType = TypeInfo::isOpaqueType(ValueType);

    if (isOpaqueType && !SourceType)
      return false;

    if (isOpaqueType)
      ValueType = cast<TypedPointerType>(SourceType)->getElementType();
    propagateType(ValueType, I->getValueOperand());
    propagateType(TypedPointerType::get(ValueType, 0), I->getPointerOperand());
    return true;
  }

  bool deduceType(const ReturnInst *I) {
    Value *RV = I->getReturnValue();
    if (nullptr == RV) {
      TypeMap->try_emplace(I, Type::getVoidTy(I->getContext()));
      return true;
    }

    Type *T = TypeInfo::isOpaqueType(RV->getType()) ? getDeducedType(RV)
                                                    : RV->getType();
    if (nullptr == T)
      return false;

    TypeMap->try_emplace(I, T);
    propagateType(T, I->getFunction());
    return true;
  }

  bool deduceType(const GetElementPtrInst *I) {
    if (TypeInfo::isOpaqueType(I->getResultElementType()))
      return false;

    Type *T = TypedPointerType::get(I->getResultElementType(), 0);
    // TypeMap->try_emplace(I, T);
    propagateType(T, I);
    return true;
  }

  bool deduceType(const CallInst *CI) {
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI))
      llvm_unreachable("Not implemented");
    // return deduceType(II);

    Type *ReturnType = CI->getFunctionType()->getReturnType();
    ReturnType = TypeInfo::isOpaqueType(ReturnType)
                     ? getDeducedType(CI->getCalledFunction())
                     : ReturnType;
    if (nullptr == ReturnType)
      return false;

    TypeMap->try_emplace(CI, ReturnType);
    propagateType(ReturnType, CI->getCalledFunction());
    return true;
  }

public:
  Module &M;
  DenseMap<const Value *, Type *> *TypeMap;
  std::unordered_set<const Value *> IncompleteTypeDefinition;
};

TypeInfo getTypeInfo(Module &M) {
  TypeAnalyzer Analyzer(M);
  return Analyzer.analyze();
}

} // namespace SPIRV

char SPIRVTypeAnalysisWrapperPass::ID = 0;

SPIRVTypeAnalysisWrapperPass::SPIRVTypeAnalysisWrapperPass() : ModulePass(ID) {}

bool SPIRVTypeAnalysisWrapperPass::runOnModule(Module &M) {
  TI = SPIRV::getTypeInfo(M);
  return false;
}

SPIRVTypeAnalysis::Result SPIRVTypeAnalysis::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  return SPIRV::getTypeInfo(M);
}

AnalysisKey SPIRVTypeAnalysis::Key;

} // namespace llvm
