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

  if (const TypedPointerType *TPT = dyn_cast<TypedPointerType>(T))
    return TypeInfo::isOpaqueType(TPT->getElementType());
  if (const ArrayType *AT = dyn_cast<ArrayType>(T))
    return TypeInfo::isOpaqueType(AT->getElementType());
  if (const VectorType *VT = dyn_cast<VectorType>(T))
    return TypeInfo::isOpaqueType(VT->getElementType());
  if (const StructType *ST = dyn_cast<StructType>(T)) {
    for (Type *ET : ST->elements())
      if (TypeInfo::isOpaqueType(ET))
        return true;

  }

  return false;
}

class TypeAnalyzer {

public:
  TypeAnalyzer(Module &M)
      : M(M), TypeMap(new DenseMap<const Value *, Type *>()) {}

  TypeInfo analyze() {
    // Queuing all the values.
    WorkList.clear();
    for (const Function &F : M) {
      for (const Argument& A : F.args())
        WorkList.emplace(&A, nullptr);
      for (const BasicBlock &BB : F)
        for (const Value &V : BB)
          WorkList.emplace(&V, nullptr);
    }
    for (auto It = M.global_begin(); It != M.global_end(); ++It)
      WorkList.emplace(&*It, nullptr);


    while (WorkList.size() != 0) {
      bool foundOneType = false;
      auto It = WorkList.begin();
      while (It != WorkList.end()) {
        const Value *V = It->first;
        Type *OldType = It->second;
        Type *NewType = deduceElementType(It->first);

        // Type deduction failed. Not enough information.
        if (!NewType && !OldType) {
          ++It;
          continue;
        }

        if (NewType == OldType || !NewType) {
          It = WorkList.erase(It);
          continue;
        }

        foundOneType = true;
        assignType(NewType, It->first);
        It->second = NewType;
        propagateType(It->first);
        for (const User *U: It->first->users())
          WorkList.emplace(U, getDeducedType(U));
        It = WorkList.erase(It);
        break;
      }

      // No progress was made, no enough info to resolve types,
      // we can stop.
      if (!foundOneType)
        break;
    }

    return TypeInfo(TypeMap);
  }

private:
  Type *getDeducedType(const Value *V) {
    auto It = TypeMap->find(V);
    return It == TypeMap->end() ? nullptr : It->second;
  }

  Type* getElementType(Type *T, const ArrayRef<unsigned>& Indices) {
    Type *Output = T;
    for (unsigned I : Indices) {
      if (ArrayType *AT = dyn_cast<ArrayType>(T))
        Output = AT->getElementType();
      else if (VectorType *VT = dyn_cast<VectorType>(T))
        Output = VT->getElementType();
      else if (StructType *ST = dyn_cast<StructType>(T))
        Output = ST->getElementType(I);
      else
        llvm_unreachable("Cannot index into this type.");
    }
    return Output;
  }

  template <typename U>
  std::vector<Type *> gatherElementTypes(Type *T, const ArrayRef<U>& Indices) {
    std::vector<Type*> Output;
    Output.push_back(T);

    for (unsigned I : Indices) {
      if (ArrayType *AT = dyn_cast<ArrayType>(T))
        T = AT->getElementType();
      else if (VectorType *VT = dyn_cast<VectorType>(T))
        T = VT->getElementType();
      else if (StructType *ST = dyn_cast<StructType>(T))
        T = ST->getElementType(I);
      else if (TypedPointerType *TP = dyn_cast<TypedPointerType>(T))
        T = TP->getElementType();
      else
        break;
      Output.push_back(T);
    }

    return Output;
  }

  template <typename U>
  Type* deduceTypeFromChain(Type *ElementType, Type *BaseType, const ArrayRef<U>& Indices) {
    if (!TypeInfo::isOpaqueType(BaseType))
      return BaseType;

    if (TypeInfo::isOpaqueType(ElementType))
      return nullptr;

    Type *Output = ElementType;
    std::vector<Type*> Types = gatherElementTypes(BaseType, Indices);
    for (unsigned I = 0; I < Indices.size(); ++I) {
      unsigned TypeIndex = Indices.size() - I - 1;
      // The entier chain cannot be walked (opaque pointer reached). Meaning we don't
      // have enough info to rebuild the whole chain.
      if (TypeIndex >= Types.size())
        return nullptr;

      Type *CT = Types[TypeIndex];

      if (ArrayType *AT = dyn_cast<ArrayType>(CT)) {
        Output = ArrayType::get(Output, AT->getNumElements());
        continue;
      }

      if (VectorType *VT = dyn_cast<VectorType>(CT)) {
        Output = VectorType::get(Output, VT->getElementCount());
        continue;
      }

      if (TypedPointerType *TP = dyn_cast<TypedPointerType>(CT)) {
        Output = TP->getElementType();
        continue;
      }

      if (dyn_cast<PointerType>(CT))
        return nullptr;

      StructType *ST = cast<StructType>(CT);
      if (!TypeInfo::isOpaqueType(ST)) {
        Output = ST;
        continue;
      }

      std::vector<Type*> ElementTypes(ST->element_begin(), ST->element_end());
      ElementTypes[Indices[I]] = Output;
      for (Type *ET : ElementTypes)
        if (TypeInfo::isOpaqueType(ET))
          return nullptr;
      if (ST->hasName())
        Output = StructType::create(ElementTypes, ST->getName(), ST->isPacked());
      else
        Output = StructType::get(ST->getContext(), ElementTypes, ST->isPacked());
    }

    return Output;
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

#if 0
  Type *resolveTypeConflict(Type *A, Type *B) {

  }

  Type *resolveTypeConflict(TypedPointerType *A, TypedPointerType *B) {
    if (typeContainsType(A->getElementType(), B->getElementType()))
      return A;
    if (typeContainsType(B->getElementType(), A->getElementType()))
      return B;
    return nullptr;
  }
#endif

  void assignType(Type *T, const Value *V) {
    assert(!TypeInfo::isOpaqueType(T));

    auto It = TypeMap->find(V);
    if (It == TypeMap->end()) {
      TypeMap->try_emplace(V, T);
      return;
    }

    // There is no conflict.
    if (T == It->second)
      return;

    Type *Solution = refineType(T, It->second);
#if 0
    TypedPointerType *LHS = dyn_cast<TypedPointerType>(T);
    TypedPointerType *RHS = dyn_cast<TypedPointerType>(It->second);
    assert(LHS && RHS && "Non pointer type conflict not handled.");
    Type *Solution = resolveTypeConflict(LHS, RHS);
#endif
    assert(Solution);
    It->second = Solution;
  }

  void propagateType(const Value *V) {
    assert(!TypeInfo::isOpaqueType(getDeducedType(V)));

    if (const AllocaInst *AI = dyn_cast<AllocaInst>(V))
      propagateTypeDetails(AI);
    else if (const LoadInst *LI = dyn_cast<LoadInst>(V))
      propagateTypeDetails(LI);
    else if (const ReturnInst *RI = dyn_cast<ReturnInst>(V))
      propagateTypeDetails(RI);
    else
      llvm_unreachable("FIXME: unsupported instruction");
#if 0
    if (const GlobalVariable *C = dyn_cast<GlobalVariable>(V))
      propagateTypeDetails(C);
    else if (const Constant *C = dyn_cast<Constant>(V))
      propagateTypeDetails(C);
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
    else if (const FreezeInst *C = dyn_cast<FreezeInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const ReturnInst *C = dyn_cast<ReturnInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const StoreInst *C = dyn_cast<StoreInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const GlobalVariable *C = dyn_cast<GlobalVariable>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const AtomicCmpXchgInst *C = dyn_cast<AtomicCmpXchgInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const AtomicRMWInst *C = dyn_cast<AtomicRMWInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const ExtractValueInst *C = dyn_cast<ExtractValueInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const InsertValueInst *C = dyn_cast<InsertValueInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const AddrSpaceCastInst *C = dyn_cast<AddrSpaceCastInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const SelectInst *C = dyn_cast<SelectInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const PHINode *C = dyn_cast<PHINode>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const CastInst *C = dyn_cast<CastInst>(V))
      propagateTypeDetails(DeducedType, C);
    else if (const BinaryOperator *C = dyn_cast<BinaryOperator>(V))
      propagateTypeDetails(DeducedType, C);
    else
      llvm_unreachable("FIXME: unsupported instruction");
#endif
  }

  void propagateTypeDetails(const AllocaInst *I) {
    return;
    //assignType(TypedPointerType::get(getDeducedType(I), I->getPointerAddressSpace()), I->getPointerOperand());
    //WorkList.insert(I->getPointerOperand());
  }

  void propagateTypeDetails(const LoadInst *I) {
    Type *T = TypedPointerType::get(getDeducedType(I), I->getPointerAddressSpace());
    assignType(T, I->getPointerOperand());
    WorkList.emplace(I->getPointerOperand(), nullptr);
  }

  void propagateTypeDetails(const ReturnInst *I) {
  }

#if 0
  void propagateTypeDetails(Type *DeducedType, const Constant *C) {
    assignType(DeducedType, C);
  }

  void propagateTypeDetails(Type *DeducedType, const Argument *A) {
    assignType(DeducedType, A);

    unsigned ArgNo = A->getArgNo();
    for (const User *U : A->getParent()->users()) {
      const CallInst *CI = cast<CallInst>(U);
      propagateType(DeducedType, CI->getOperand(ArgNo));
    }
  }

  void propagateTypeDetails(Type *DeducedType, const LoadInst *I) {
    assignType(DeducedType, I);
    propagateType(TypedPointerType::get(DeducedType, I->getPointerAddressSpace()),
                  I->getPointerOperand());
  }

  void propagateTypeDetails(Type *DeducedType, const StoreInst *I) {
    assignType(DeducedType, I);
  }

  void propagateTypeDetails(Type *DeducedType, const AllocaInst *I) {
    Type *StoredType = cast<TypedPointerType>(DeducedType)->getElementType();
    if (ArrayType *AT = dyn_cast<ArrayType>(I->getAllocatedType())) {
      Type *NewType = ArrayType::get(StoredType, AT->getNumElements());
      assignType(NewType, I);
      return;
    }

    assignType(DeducedType, I);
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
      propagateType(TypedPointerType::get(GEP->getSourceElementType(), GEP->getAddressSpace()),
                    GEP->getPointerOperand());
      return;
    }

    assignType(DeducedType, GEP);

    std::vector<uint64_t> Indices;
    // The first index of a GEP is indexing from the passed pointer. Skipping
    // the first index for type deduction.
    for (unsigned I = 1; I < GEP->getNumIndices(); ++I)
      Indices.push_back(cast<ConstantInt>(&*(GEP->idx_begin() + I))->getZExtValue());
    Type *BT = deduceTypeFromChain<uint64_t>(DeducedType, GEP->getSourceElementType(), Indices);
    if (!BT)
      return;

    // Adding back the first indirection to the deduced type.
    Type *NewType = TypedPointerType::get(BT, GEP->getAddressSpace());
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

    assignType(DeducedType, CI);
    propagateType(DeducedType, CI->getCalledFunction());

    for (const BasicBlock &BB : *CI->getCalledFunction())
      for (const Instruction &I : BB)
        if (const ReturnInst *RI = dyn_cast<ReturnInst>(&I))
          propagateType(DeducedType, RI);
  }

  void propagateTypeDetails(Type *DeducedType, const ReturnInst *RI) {
    Value *RV = RI->getReturnValue();
    assert(RV || DeducedType->isVoidTy());

    assignType(DeducedType, RI);
    propagateType(DeducedType, RV);
  }

  void propagateTypeDetails(Type *DeducedType, const GlobalVariable *GV) {
    assignType(DeducedType, GV);
    if (!GV->hasInitializer())
      return;
    TypedPointerType *TP = cast<TypedPointerType>(DeducedType);
    assert(!TypeInfo::isOpaqueType(TP));
    propagateType(TP->getElementType(), GV->getInitializer());
  }

  void propagateTypeDetails(Type *DeducedType, const AtomicCmpXchgInst *A) {
    StructType *ST = cast<StructType>(DeducedType);
    assignType(DeducedType, A);

    propagateType(TypedPointerType::get(ST->getElementType(0), A->getPointerAddressSpace()), A->getPointerOperand());
    propagateType(ST->getElementType(0), A->getCompareOperand());
    propagateType(ST->getElementType(0), A->getNewValOperand());
  }

  void propagateTypeDetails(Type *DeducedType, const ExtractValueInst *EI) {
    assignType(DeducedType, EI);

    if (getDeducedType(EI->getAggregateOperand()))
      return;

    Type *BT = deduceTypeFromChain(DeducedType, EI->getAggregateOperand()->getType(), EI->getIndices());
    if (!BT)
      return;
    propagateType(BT, EI->getAggregateOperand());
  }

  void propagateTypeDetails(Type *DeducedType, const InsertValueInst *II) {
    assignType(DeducedType, II);

    // The source aggregate hasn't been deduced, we can propagate the type we now have.
    if (!getDeducedType(II->getAggregateOperand()))
      propagateType(DeducedType, II->getAggregateOperand());

    // The value operand can be deduced from the result operand now.
    if (!getDeducedType(II->getInsertedValueOperand()))
      propagateType(getElementType(DeducedType, II->getIndices()), II->getInsertedValueOperand());
  }

  void propagateTypeDetails(Type *DeducedType, const AddrSpaceCastInst *ACI) {
    assignType(DeducedType, ACI);

    TypedPointerType *TPT = cast<TypedPointerType>(DeducedType);
    Type *Result = TypedPointerType::get(TPT->getElementType(), ACI->getSrcAddressSpace());
    propagateType(Result, ACI->getPointerOperand());
  }

  void propagateTypeDetails(Type *DeducedType, const SelectInst *SI) {
    assignType(DeducedType, SI);
    propagateType(DeducedType, SI->getTrueValue());
    propagateType(DeducedType, SI->getFalseValue());
  }

  void propagateTypeDetails(Type *DeducedType, const CastInst *SI) {
    assignType(DeducedType, SI);

    if (!TypeInfo::isOpaqueType(SI->getSrcTy()))
      propagateType(SI->getSrcTy(), SI->getOperand(0));
  }

  void propagateTypeDetails(Type *DeducedType, const PHINode *PHI) {
    assignType(DeducedType, PHI);

    for (Value *V : PHI->incoming_values())
      propagateType(DeducedType, V);
  }

  void propagateTypeDetails(Type *DeducedType, const AtomicRMWInst *I) {
    assignType(DeducedType, I);
    propagateType(TypedPointerType::get(DeducedType, I->getPointerAddressSpace()), I->getPointerOperand());
  }

  void propagateTypeDetails(Type *DeducedType, const BinaryOperator *I) {
    assignType(DeducedType, I);
    propagateType(DeducedType, I->getOperand(0));
    propagateType(DeducedType, I->getOperand(1));
  }

  void propagateTypeDetails(Type *DeducedType, const FreezeInst *I) {
    assignType(DeducedType, I);
    propagateType(DeducedType, I->getOperand(0));
  }
#endif

  Type *deduceElementType(const Value *V) {
    assert(V != nullptr);

#define X(ToType, Value)                                                       \
  if (auto *Casted = dyn_cast<ToType>(Value))                                  \
    return deduceType(Casted)
    //Type *O = deduceType(Casted);                                              \
  //  return O ? O : getDeducedType(Casted);                                     \
  //}

    X(AllocaInst, V);
    X(LoadInst, V);
    X(ReturnInst, V);
#if 0
    X(Argument, V);
    X(AddrSpaceCastInst, V);
    X(AtomicCmpXchgInst, V);
    X(ExtractValueInst, V);
    X(InsertValueInst, V);
    X(GetElementPtrInst, V);
    X(StoreInst, V);
    X(InsertElementInst, V);
    X(ExtractElementInst, V);
    X(SelectInst, V);
    X(PHINode, V);
    X(ShuffleVectorInst, V);
    X(AtomicRMWInst, V);
    X(FreezeInst, V);
    X(FenceInst, V);

    X(BranchInst, V);
    X(CallInst, V);
    X(CastInst, V);
    X(CmpInst, V);
    X(GlobalVariable, V);
    X(SwitchInst, V);
    X(UnreachableInst, V);

    X(BinaryOperator, V);
    X(UnaryOperator, V);
    // TODO: shufflevector
#undef X
#endif

    V->dump();
    llvm_unreachable("FIXME: unsupported instruction");
    return nullptr;
  }

  Type *refineType(Type *A, Type *B) {
    assert(A && B);

    if (A == B)
      return A;

    PointerType *PTA = dyn_cast<PointerType>(A);
    PointerType *PTB = dyn_cast<PointerType>(B);
    TypedPointerType *TPTA = dyn_cast<TypedPointerType>(A);
    TypedPointerType *TPTB = dyn_cast<TypedPointerType>(B);
    ArrayType *ATA = dyn_cast<ArrayType>(A);
    ArrayType *ATB = dyn_cast<ArrayType>(B);

    // Case 1: one is a pointer, the other a typed pointer/array.
    // we cannot go deeper on the left side, meaning we can stop.
    if (PTA && (TPTB || ATB))
      return B;
    if (PTB && (TPTA || ATA))
      return A;

    assert(!PTA && !PTB);

    // Case 2: Both are arrays/types pointers, moving the resolution to the element type.
    if (TPTA && TPTB) {
      assert(TPTA->getAddressSpace() == TPTB->getAddressSpace());
      return TypedPointerType::get(refineType(TPTA->getElementType(), TPTB->getElementType()), TPTA->getAddressSpace());
    }
    if (ATA && ATB) {
      assert(ATA->getNumElements() == ATB->getNumElements());
      return ArrayType::get(refineType(ATA->getElementType(), ATB->getElementType()), ATA->getNumElements());
    }

    // Case 3: One is a pointer, the other an array. The array being more constraint,
    // we keep it, and resolve the element type recursively.
    if (TPTA && ATB) {
      return ArrayType::get(refineType(TPTA->getElementType(), ATB->getElementType()), ATB->getNumElements());
    }
    if (TPTB && ATA) {
      return ArrayType::get(refineType(TPTB->getElementType(), ATA->getElementType()), ATA->getNumElements());
    }

    StructType *STA = dyn_cast<StructType>(A);
    StructType *STB = dyn_cast<StructType>(B);
    if ((TPTA && STB) || (TPTB && STA)) {
      StructType *ST = TPTA ? STB : STA;
      TypedPointerType *TPT = TPTA ? TPTA : TPTB;

      std::vector<Type*> ElementTypes(ST->element_begin(), ST->element_end());
      ElementTypes[0] = refineType(ST->getElementType(0), TPT);
      if (ST->hasName())
        return StructType::create(ElementTypes, ST->getName(), ST->isPacked());
      return StructType::get(ST->getContext(), ElementTypes, ST->isPacked());
    }

    if (STA && STB) {
      std::vector<Type*> ElementTypes;
      assert(STA->getNumElements() == STB->getNumElements());
      for (unsigned I = 0; I < STA->getNumElements(); ++I)
        ElementTypes.push_back(refineType(STA->getElementType(I), STB->getElementType(I)));
      return StructType::get(STA->getContext(), ElementTypes, STA->isPacked());
    }

    if (typeContainsType(A, B))
      return A;
    if (typeContainsType(B, A))
      return B;

    assert(0);
    return nullptr;
  }

  Type *deduceType(const AllocaInst *I) {

    Type *ReturnType = nullptr;
    if (dyn_cast<ArrayType>(I->getAllocatedType()))
      ReturnType = I->getAllocatedType();
    else
      ReturnType = TypedPointerType::get(I->getAllocatedType(), I->getAddressSpace());

    // Simple case: type is not opaque, we return the ground truth.
    if (!TypeInfo::isOpaqueType(ReturnType))
      return ReturnType;

    Type *DeducedType = getDeducedType(I);
    if (!DeducedType)
      return nullptr;

    TypedPointerType *TP = dyn_cast<TypedPointerType>(DeducedType);
    Type *Solution = refineType(ReturnType, TP);
    if (TypeInfo::isOpaqueType(Solution))
      return nullptr;
    return Solution;
  }

  Type *deduceType(const LoadInst *LI) {
    // First case: the loaded type is complete: we can assign the result type.
    if (!TypeInfo::isOpaqueType(LI->getType()))
      return LI->getType();

#if 0
    // This is not certain. Because the loaded size will define the level of unpacking,
    // this is not a trivial thing.
    // The pointer operand is non-opaque, we can deduce the loaded type.
    Type *PointerOperandTy = getDeducedType(LI->getPointerOperand());
    if (!PointerOperandTy)
      return nullptr;

    Type *ElementType = nullptr;
    if (auto *TPT = dyn_cast<TypedPointerType>(PointerOperandTy))
      ElementType = TPT->getElementType();
    else
      ElementType = cast<ArrayType>(PointerOperandTy)->getElementType();

    Type *DeducedLoadType = LI;
    if (ElementType == LI->getType())
      return ElementType;

#endif
    return nullptr;
  }

  Type *deduceType(const ReturnInst *RI) {
    Value *RV = RI->getReturnValue();
    if (!RV)
      return Type::getVoidTy(RI->getContext());

    if (TypeInfo::isOpaqueType(RV->getType()))
      return nullptr;
    return RV->getType();
  }

#if 0
  bool deduceType(const Argument *A) {
    if (TypeInfo::isOpaqueType(A->getType()))
      return false;
    TypeMap->try_emplace(A, A->getType());
    return true;
  }

  bool deduceType(const AllocaInst *I) {
    if (TypeInfo::isOpaqueType(I->getAllocatedType()))
      return false;
    TypeMap->try_emplace(I, TypedPointerType::get(I->getAllocatedType(), I->getAddressSpace()));
    return true;
  }

  bool deduceType(const AtomicRMWInst *I) {
    Type *VT = I->getValOperand()->getType();
    assert(!TypeInfo::isOpaqueType(VT));
    TypeMap->try_emplace(I, VT);
    propagateType(TypedPointerType::get(VT, I->getPointerAddressSpace()), I->getPointerOperand());
    return true;
  }

  bool deduceType(const BranchInst *I) {
    TypeMap->try_emplace(I, Type::getVoidTy(I->getContext()));
    return true;
  }

  bool deduceType(const SwitchInst *I) {
    TypeMap->try_emplace(I, Type::getVoidTy(I->getContext()));
    return true;
  }

  bool deduceType(const UnreachableInst *I) {
    TypeMap->try_emplace(I, Type::getVoidTy(I->getContext()));
    return true;
  }

  bool deduceType(const FenceInst *I) {
    TypeMap->try_emplace(I, Type::getVoidTy(I->getContext()));
    return true;
  }

  bool deduceType(const AddrSpaceCastInst *ACI) {
    Type *PT = getDeducedType(ACI->getPointerOperand());
    if (!PT)
      return false;

    TypedPointerType *TPT = cast<TypedPointerType>(PT);
    Type *Result = TypedPointerType::get(TPT->getElementType(), ACI->getDestAddressSpace());
    TypeMap->try_emplace(ACI, Result);
    return true;
  }

  bool deduceType(const CmpInst *CI) {
    assert(!TypeInfo::isOpaqueType(CI->getType()));
    TypeMap->try_emplace(CI, CI->getType());
    return true;
  }

  bool deduceType(const LoadInst *I) {
    // First case: the loaded type is complete: we can assign the result type.
    if (!TypeInfo::isOpaqueType(I->getType())) {
      TypeMap->try_emplace(I, I->getType());
      propagateType(TypedPointerType::get(I->getType(), I->getPointerAddressSpace()),
                    I->getPointerOperand());
      return true;
    }

    // The pointer operand is non-opaque, we can geduce the loaded type.
    if (Type *PointerOperandTy = getDeducedType(I->getPointerOperand())) {
      Type *ElementType = cast<TypedPointerType>(PointerOperandTy)->getElementType();
      TypeMap->try_emplace(I, ElementType);
      return true;
    }

    return false;
  }

  bool deduceType(const CastInst *I) {
    if (TypeInfo::isOpaqueType(I->getDestTy()))
      return false;
    TypeMap->try_emplace(I, I->getDestTy());
    return true;
  }

  bool deduceType(const AtomicCmpXchgInst *I) {
    Type *VT = nullptr;

    if (!TypeInfo::isOpaqueType(I->getType())) {
      // The cmpxchg instruction returns an aggregate with 2 values:
      //  - the original value
      //  - a bit indicating if the store occured or not.
      StructType *ST = cast<StructType>(I->getType());
      VT = ST->getElementType(0);
    }

    if (!VT)
      VT = getDeducedType(I->getPointerOperand());

    if (!VT)
      VT = getDeducedType(I->getCompareOperand());

    if (!VT)
      return false;

    StructType *RT = StructType::get(VT, IntegerType::getInt1Ty(I->getContext()));

    TypeMap->try_emplace(I, RT);
    propagateType(TypedPointerType::get(VT, I->getPointerAddressSpace()), I->getPointerOperand());
    propagateType(VT, I->getNewValOperand());
    propagateType(VT, I->getCompareOperand());

    return true;
  }

  bool deduceType(const StoreInst *I) {
    Type *VT = nullptr;
    // We can get the type from the value operand.
    if (!TypeInfo::isOpaqueType(I->getValueOperand()->getType()))
        VT = I->getValueOperand()->getType();

    // If not, we can maybe get it from the pointer operands.
    if (!VT) {
      Type *Source = getDeducedType(I->getPointerOperand());
      if (Source)
        VT = cast<TypedPointerType>(Source)->getElementType();
    }

    // Otherwise, nothing to deduce.
    if (!VT)
      return false;

    propagateType(VT, I->getValueOperand());
    propagateType(TypedPointerType::get(VT, I->getPointerAddressSpace()), I->getPointerOperand());
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
    if (!TypeInfo::isOpaqueType(I->getResultElementType())) {
      Type *T = TypedPointerType::get(I->getResultElementType(), I->getAddressSpace());
      TypeMap->try_emplace(I, T);

      assert(!TypeInfo::isOpaqueType(I->getSourceElementType()));
      propagateType(TypedPointerType::get(I->getSourceElementType(), I->getAddressSpace()), I->getPointerOperand());
      return true;
    }

    Type *DeducedBase = getDeducedType(I->getPointerOperand());
    if (!DeducedBase)
      return false;
    DeducedBase = cast<TypedPointerType>(DeducedBase)->getElementType();

    return false;
  }

  bool deduceType(const CallInst *CI) {
    FunctionType *FT = CI->getFunctionType();
    for (unsigned I = 0; I < FT->getNumParams(); ++I) {
      Type *ParamType = FT->getParamType(I);
      if (TypeInfo::isOpaqueType(ParamType))
        continue;
      // Variadic functions may have more arguments than the type defines types,
      // so it should be OK to limit I to be FT->getNumParams() and use it to index
      // in getArgOperand.
      propagateType(ParamType, CI->getArgOperand(I));
    }

    if (!TypeInfo::isOpaqueType(CI->getType())) {
      TypeMap->try_emplace(CI, CI->getType());
      return true;
    }

    if (!CI->getCalledFunction())
      return false;

    Type *ReturnType = getDeducedType(CI->getCalledFunction());
    if (ReturnType)
      TypeMap->try_emplace(CI, ReturnType);
    return ReturnType != nullptr;
  }

  bool deduceType(const GlobalVariable *GV) {
    // Simple case: the global stores a non-opaque value. We can directly deduce the type.
    if (!TypeInfo::isOpaqueType(GV->getValueType())) {
      TypeMap->try_emplace(GV, TypedPointerType::get(GV->getValueType(), GV->getType()->getAddressSpace()));
      return true;
    }

    // Global variable type is opaque, and has no initializer we can deduce the type with.
    const Constant *C = GV->hasInitializer() ? GV->getInitializer() : nullptr;
    if (C == nullptr)
      return false;

    // Type is opaque, and the initializer has no deduced type, we cannot do anything either.
    Type *DeducedValueType = getDeducedType(C);
    if (DeducedValueType == nullptr)
      return false;

    TypeMap->try_emplace(GV, TypedPointerType::get(DeducedValueType, GV->getType()->getAddressSpace()));
    return true;
  }

  bool deduceType(const ExtractValueInst *I) {
    Type *T = I->getType();
    if (!TypeInfo::isOpaqueType(T)) {
      TypeMap->try_emplace(I, T);
      return true;
    }

    T = getDeducedType(I->getAggregateOperand());
    if (!T)
      return false;

    T = getElementType(T, I->getIndices());
    TypeMap->try_emplace(I, T);
    return true;
  }

  bool deduceType(const InsertValueInst *I) {
    Type *T = I->getType();
    if (!TypeInfo::isOpaqueType(T)) {
      TypeMap->try_emplace(I, T);
      return true;
    }

    T = getDeducedType(I->getAggregateOperand());
    if (!T) {
      TypeMap->try_emplace(I, T);
      return true;
    }

    return false;
  }

  bool deduceType(const InsertElementInst *I) {
    assert(!TypeInfo::isOpaqueType(I->getType()));
    TypeMap->try_emplace(I, I->getType());
    propagateType(I->getType(), I->getOperand(0));
    return true;
  }

  bool deduceType(const ExtractElementInst *I) {
    assert(!TypeInfo::isOpaqueType(I->getType()));
    TypeMap->try_emplace(I, I->getType());
    propagateType(I->getVectorOperandType(), I->getVectorOperand());
    return true;
  }

  bool deduceType(const ShuffleVectorInst *I) {
    assert(!TypeInfo::isOpaqueType(I->getType()));
    TypeMap->try_emplace(I, I->getType());
    //propagateType(I->getType(), I->getOperand(0));
    //propagateType(I->getType(), I->getOperand(1));
    return true;
  }

  bool deduceType(const SelectInst *I) {
    Type *T = nullptr;
    if (!TypeInfo::isOpaqueType(I->getType()))
      T = I->getType();
    if (!T)
      T = getDeducedType(I->getTrueValue());
    if (!T)
      T = getDeducedType(I->getFalseValue());

    if (!T)
      return false;

    TypeMap->try_emplace(I, T);
    propagateType(T, I->getTrueValue());
    propagateType(T, I->getFalseValue());
    return true;
  }

  bool deduceType(const BinaryOperator *I) {
    DenseSet<Instruction::BinaryOps> SupportedOpcodes({
      Instruction::BinaryOps::Add,
      Instruction::BinaryOps::FAdd,
      Instruction::BinaryOps::Sub,
      Instruction::BinaryOps::FSub,
      Instruction::BinaryOps::Mul,
      Instruction::BinaryOps::FMul,
      Instruction::BinaryOps::UDiv,
      Instruction::BinaryOps::SDiv,
      Instruction::BinaryOps::FDiv,
      Instruction::BinaryOps::URem,
      Instruction::BinaryOps::SRem,
      Instruction::BinaryOps::FRem,
      Instruction::BinaryOps::Shl,
      Instruction::BinaryOps::LShr,
      Instruction::BinaryOps::AShr,
      Instruction::BinaryOps::And,
      Instruction::BinaryOps::Or,
      Instruction::BinaryOps::Xor
    });

    assert(SupportedOpcodes.contains(I->getOpcode()));
    Type *T = I->getType();
    if (TypeInfo::isOpaqueType(T))
      return false;

    TypeMap->try_emplace(I, T);
    propagateType(T, I->getOperand(0));
    propagateType(T, I->getOperand(1));
    return true;
  }

  bool deduceType(const UnaryOperator *I) {
    assert(I->getOpcode() == Instruction::UnaryOps::FNeg);
    Type *T = I->getType();
    if (TypeInfo::isOpaqueType(T))
      return false;

    TypeMap->try_emplace(I, T);
    propagateType(T, I->getOperand(0));
    return true;
  }

  bool deduceType(const PHINode *I) {
    Type *T = TypeInfo::isOpaqueType(I->getType()) ? nullptr : I->getType();
    for (Value *V : I->incoming_values()) {
      if (T)
        break;
     T = getDeducedType(V);
    }

    if (!T)
      return false;

    TypeMap->try_emplace(I, T);
    for (Value *V : I->incoming_values())
      propagateType(T, V);
    return true;
  }

  bool deduceType(const FreezeInst *I) {
    Type *T = I->getType();
    if (TypeInfo::isOpaqueType(T))
      T = getDeducedType(I);

    if (!T)
      return false;

    TypeMap->try_emplace(I, T);
    propagateType(T, I->getOperand(0));
    return true;
  }
#endif

public:
  Module &M;
  DenseMap<const Value *, Type *> *TypeMap;
  std::unordered_map<const Value*, Type*> WorkList;
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
