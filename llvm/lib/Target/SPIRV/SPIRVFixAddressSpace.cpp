//===-- SPIRVFixAddressSpace.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass is Vulkan specific as Logical SPIR-V doesn't support pointer
// cast.
//
// In LLVM IR, global and local variables are by default in the same address
// space. In SPIR-V, global and local variables live in a different address
// space (storage class in the SPIR-V parlance).
//
// This means the following function cannot be lowered to SPIR-V:
//   static int global = 0;
//   int& return_ref(int& ref, bool select) {
//    return select ? ref : global;
//   }
//
// A solution is to force inline, but this would prevent us from emitting SPIR-V
// libraries. Another solution is to move all globals to local variables, but
// this also blocks libraries. The last solution is to replace all local
// variables with global variables. This is possible because Vulkan SPIR-V
// completely forbids static recursion.
//
// This pass replace all alloca instruction with a new global variable.
// In addition, it moves all such allocations into the `Private` address space.
//
// After this pass, no variable or pointer should reference the default address
// space.
//
// Note:
//  LLVM IR has address spaces for functions, but SPIR-V doesn't. In addition,
//  Vulkan disallow function pointers and indirect jump, meaning we could never
//  have a pointer storing the function address. For this reason, functions are
//  left in the default address space, but all pointer operands to the default
//  AS are rewritten to point to the AS `Private`. This kind of blind rewrite
//  simplifies the code, but can only work with those assumptions.
//===----------------------------------------------------------------------===//

#include "Analysis/SPIRVConvergenceRegionAnalysis.h"
#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include <queue>
#include <stack>
#include <type_traits>
#include <unordered_set>

using namespace llvm;
using namespace SPIRV;

namespace llvm {
void initializeSPIRVFixAddressSpacePass(PassRegistry &);
} // namespace llvm

namespace {

constexpr unsigned GlobalAddressSpace =
    storageClassToAddressSpace(SPIRV::StorageClass::Private);

// Returns true if the given type or any subtype contains a pointer to the
// default address space.
bool typeRequiresConversion(Type *T) {
  if (T->isPointerTy())
    return T->getPointerAddressSpace() == 0;

  if (T->isArrayTy())
    return typeRequiresConversion(T->getArrayElementType());
  if (T->isStructTy()) {
    for (unsigned I = 0; I < T->getStructNumElements(); ++I)
      if (typeRequiresConversion(T->getStructElementType(I)))
        return true;
    return false;
  }
  if (FunctionType *FT = dyn_cast<FunctionType>(T)) {
    if (typeRequiresConversion(FT->getReturnType()))
      return true;
    for (Type *TP : FT->params())
      if (typeRequiresConversion(TP))
        return true;
    return false;
  }

  return false;
}

class SPIRVFixAddressSpace : public InstVisitor<SPIRVFixAddressSpace>,
                             public ModulePass {

  // Types are supposed to be unique. When using Type::get, there is a lookup to
  // only create types on demand. StructType::get only allows creating literal
  // structs, meaning we would lose the type name. This forces us to use
  // StructType::create, which doesn't deduplicates. This means we must bring
  // our own old-type/new-type map to prevent creating distinct types when we
  // shouldn't.
  std::unordered_map<Type *, Type *> ConvertedTypes;

private:
  // If the passed type or subtype contains a pointer to AS 0, get or create a
  // new type with all pointer changed to address space `Private`. The functions
  // below are overloads depending on the input type.
  PointerType *convertPointerType(PointerType *PT);
  ArrayType *convertArrayType(ArrayType *AT);
  StructType *convertStructType(StructType *ST);
  VectorType *convertVectorType(VectorType *VT);
  FunctionType *convertFunctionType(FunctionType *FT);

  // Get or create a new type if `T` or any of its subtype is a pointer to the
  // address space 0. All pointers in the returned type points to the address
  // space `Private`. If `T` was already converted once, the cached converted
  // type is returned and no additional type is created.
  Type *convertType(Type *T);

  // If `C` type or subtype contains any pointer to the address space 0, returns
  // a new constant with a fixed type.
  Constant *convertConstant(Constant *C);

  // See `convertConstant`. Those functions are overloads to handle specific
  // constant types.
  Constant *convertConstantAggregate(ConstantAggregate *CA);
  ConstantData *convertConstantData(ConstantData *CD);

  // If the passes global variable is in the default address space, replace it
  // with a global in the `Private` address space (=SPIR-V storage class). Does
  // not modify globals in a different address space (resources for ex). Returns
  // true if the global was replaced.
  bool rewriteGlobalVariable(Module &M, GlobalVariable *GV);

  // Modifies the given function by replacing all alloca by a global variable.
  // This function requires the alloca allocation size to be static:
  //  - Vulkan doesn't support VLA in local variables. (See
  //  VUID-StandaloneSpirv-OpTypeRuntimeArray-04680).
  //  - HLSL doesn't allow VLA.
  // Returns true if the function was modified.
  bool replaceAlloca(Function &F);

  // Mutate the types and operands of `F` to make sure no referenced type has a
  // pointer to the default address space. This function does not propagate the
  // type changes, hence if not used carefully, this could generate invalid IR.
  // Returns true if the function was modified.
  bool blindlyMutateTypes(Function &F);

  // Modifies any GEP instruction in the given function to only use
  // ptr addrspace(10) instead of pointers to the default address space.
  // Returns true if the function was modified.
  bool rewriteGEP(Function &F);

  // Checks all instructions in F, and make sure no pointer to the default
  // address space or local variable remains. This function assumes all other
  // functions/globals undergo the same treatment. Not calling this function on
  // all the module functions could yield to invalid IR. Returns true if the
  // function has been modified.
  bool fixInstructions(Function &F);

  // Checks if any type/subtype in the return value or a parameter is a ptr to
  // the default address space. If such pointer is found, recreate the function
  // replacing it with a `ptr addrspace(10)`. This function replaces all uses of
  // the function return value/argument/declaration with the new version, but
  // does not propagate changes further. If the rest of the instruction is not
  // cleaned-up, this can produce invalid IR. Returns true if the function has
  // been replaced.
  bool rewriteFunctionParameters(Module &M, Function *F);

  // Fix all functions in the given module.
  // Returns true if any of the module functions have been modified.
  // If the module is modified, new global variables could have been added.
  // This function only modified global variables referenced by at least one
  // function. Returns true if the module was modified.
  bool fixFunctions(Module &M);

  // Fix all the globals in the given module, even if not referenced by any
  // function.
  bool fixGlobals(Module &M);

public:
  static char ID;

  SPIRVFixAddressSpace() : ModulePass(ID) {
    initializeSPIRVFixAddressSpacePass(*PassRegistry::getPassRegistry());
  };

  virtual bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ModulePass::getAnalysisUsage(AU);
  }
};

PointerType *SPIRVFixAddressSpace::convertPointerType(PointerType *PT) {
  if (PT->getPointerAddressSpace() != 0)
    return PT;

  auto It = ConvertedTypes.find(PT);
  if (It != ConvertedTypes.end())
    return cast<PointerType>(It->second);

  PointerType *NewType = PointerType::get(PT->getContext(), GlobalAddressSpace);
  ConvertedTypes.emplace(PT, NewType);
  return NewType;
}

ArrayType *SPIRVFixAddressSpace::convertArrayType(ArrayType *AT) {
  if (!typeRequiresConversion(AT))
    return AT;

  auto It = ConvertedTypes.find(AT);
  if (It != ConvertedTypes.end())
    return cast<ArrayType>(It->second);

  Type *ElementType = convertType(AT->getElementType());
  ArrayType *NewType = ArrayType::get(ElementType, AT->getNumElements());
  ConvertedTypes.emplace(AT, NewType);
  return NewType;
}

StructType *SPIRVFixAddressSpace::convertStructType(StructType *ST) {
  if (!typeRequiresConversion(ST))
    return ST;
  std::vector<Type *> Elements;
  Elements.resize(ST->getNumElements());
  for (unsigned I = 0; I < Elements.size(); ++I)
    Elements[I] = convertType(ST->getElementType(I));

  auto It = ConvertedTypes.find(ST);
  if (It != ConvertedTypes.end())
    return cast<StructType>(It->second);

  if (!ST->hasName())
    return StructType::get(ST->getContext(), Elements);

  std::string OldName = ST->getName().str();
  ST->setName(OldName + ".old");

  StructType *NewType = StructType::create(ST->getContext(), Elements, OldName);
  ConvertedTypes.emplace(ST, NewType);
  return NewType;
}

VectorType *SPIRVFixAddressSpace::convertVectorType(VectorType *VT) {
  if (!typeRequiresConversion(VT))
    return VT;

  auto It = ConvertedTypes.find(VT);
  if (It != ConvertedTypes.end())
    return cast<VectorType>(It->second);

  Type *ElementType = convertType(VT->getElementType());
  VectorType *NewType = VectorType::get(ElementType, VT->getElementCount());
  ConvertedTypes.emplace(VT, NewType);
  return NewType;
}

FunctionType *SPIRVFixAddressSpace::convertFunctionType(FunctionType *FT) {
  if (!typeRequiresConversion(FT))
    return FT;

  auto It = ConvertedTypes.find(FT);
  if (It != ConvertedTypes.end())
    return cast<FunctionType>(It->second);

  Type *ReturnType = FT->getReturnType();
  std::vector<Type *> Params;
  Params.reserve(FT->getNumParams());
  for (Type *P : FT->params())
    Params.push_back(convertType(P));

  FunctionType *NewType = FunctionType::get(ReturnType, Params, FT->isVarArg());
  ConvertedTypes.emplace(FT, NewType);
  return NewType;
}

// Replace all references of the default address space in `T` with
// the `Private` SPIR-V address space, recreating the type is required.
// Returns the new type if recreated, `T` otherwise.
Type *SPIRVFixAddressSpace::convertType(Type *T) {
  if (PointerType *PT = dyn_cast<PointerType>(T))
    return convertPointerType(PT);

  if (ArrayType *AT = dyn_cast<ArrayType>(T))
    return convertArrayType(AT);

  if (VectorType *VT = dyn_cast<VectorType>(T))
    return convertVectorType(VT);

  if (StructType *ST = dyn_cast<StructType>(T))
    return convertStructType(ST);

  if (FunctionType *FT = dyn_cast<FunctionType>(T))
    return convertFunctionType(FT);

  if (isa<TargetExtType>(T))
    return T;

  if (T == Type::getTokenTy(T->getContext()))
    return T;

  if (T == Type::getLabelTy(T->getContext()))
    return T;

  // TypedPointerType: not implemented on purpose.

  // Make sure pointers & vectors are handled above.
  // All other single-value types don't address space conversion.
  assert(!T->isPointerTy() && !T->isVectorTy());
  if (T->isSingleValueType())
    return T;

  llvm_unreachable("Unsupported type for address space fixup.");
}

Constant *
SPIRVFixAddressSpace::convertConstantAggregate(ConstantAggregate *CA) {
  Type *NewType = convertType(CA->getType());
  std::vector<Constant *> Elements;
  Elements.resize(CA->getNumOperands());
  for (unsigned I = 0; I < CA->getNumOperands(); ++I)
    Elements[I] = convertConstant(cast<Constant>(CA->getOperand(I)));

  if (isa<ConstantArray>(CA))
    return ConstantArray::get(cast<ArrayType>(NewType), Elements);
  else if (isa<ConstantStruct>(CA))
    return ConstantStruct::get(cast<StructType>(NewType), Elements);
  return ConstantVector::get(Elements);
}

ConstantData *SPIRVFixAddressSpace::convertConstantData(ConstantData *CD) {
  if (!typeRequiresConversion(CD->getType()))
    return CD;

  if (ConstantPointerNull *CPN = dyn_cast<ConstantPointerNull>(CD))
    return ConstantPointerNull::get(convertPointerType(CPN->getType()));
  report_fatal_error("Unsupported ConstantData type.");
}

// Replace all references of the default address space in `C` with the
// SPIR-V private address space, recreating the constant, and/or modifying the
// type if required.
Constant *SPIRVFixAddressSpace::convertConstant(Constant *C) {
  if (!typeRequiresConversion(C->getType()))
    return C;

  if (ConstantAggregate *CA = dyn_cast<ConstantAggregate>(C))
    return convertConstantAggregate(CA);
  if (ConstantData *CD = dyn_cast<ConstantData>(C))
    return convertConstantData(CD);
  llvm_unreachable("Unsupported constant type.");
}

bool SPIRVFixAddressSpace::rewriteGlobalVariable(Module &M,
                                                 GlobalVariable *GV) {
  if (GV->getAddressSpace() != 0)
    return false;

  Type *NewType = GV->getValueType();
  if (typeRequiresConversion(GV->getValueType()))
    NewType = convertType(NewType);

  std::string OldName = GV->getName().str();
  GV->setName(OldName + ".dead");
  GlobalVariable *NewGV = new GlobalVariable(
      M, NewType,
      /* isConstant= */ false, GV->getLinkage(),
      convertConstant(GV->getInitializer()), OldName,
      /* insertBefore= */ GV, GV->getThreadLocalMode(), GlobalAddressSpace);

  std::vector<User *> ToFix(GV->user_begin(), GV->user_end());
  for (auto *User : ToFix) {
    if (Constant *C = dyn_cast<Constant>(User))
      C->handleOperandChange(GV, NewGV);
    else
      User->replaceUsesOfWith(GV, NewGV);
  }
  M.eraseGlobalVariable(GV);
  return true;
}

bool SPIRVFixAddressSpace::replaceAlloca(Function &F) {
  std::unordered_set<Instruction *> DeadInstructions;

  for (auto &BB : F) {
    for (auto &I : BB) {
      AllocaInst *AI = dyn_cast<AllocaInst>(&I);
      if (!AI)
        continue;

      // Vulkan doesn't support VLA in local variables. (See
      // VUID-StandaloneSpirv-OpTypeRuntimeArray-04680). HLSL doesn't allow VLA,
      // meaning we should not encounter this for now, but it another frontend
      // is used, we may hit this case.
      assert(isa<ConstantInt>(AI->getArraySize()));

      Type *NewType = convertType(AI->getAllocatedType());
      GlobalVariable *NewGV = new GlobalVariable(
          *F.getParent(), NewType,
          /* isConstant= */ false, GlobalValue::LinkageTypes::InternalLinkage,
          Constant::getNullValue(NewType), F.getName() + ".local",
          /* insertBefore= */ nullptr,
          GlobalValue::ThreadLocalMode::NotThreadLocal, GlobalAddressSpace);

      std::vector<User *> ToFix(AI->user_begin(), AI->user_end());
      for (auto *User : ToFix)
        User->replaceUsesOfWith(AI, NewGV);
      DeadInstructions.insert(AI);
    }
  }

  for (auto *I : DeadInstructions)
    I->eraseFromParent();
  return DeadInstructions.size() != 0;
}

bool SPIRVFixAddressSpace::blindlyMutateTypes(Function &F) {
  bool Modified = false;

  for (auto &BB : F) {
    for (auto &I : BB) {
      for (auto &Op : I.operands()) {
        if (isa<Function>(Op.get()) || isa<BlockAddress>(Op.get()))
          continue;

        Type *NewType = convertType(Op->getType());
        Op->mutateType(NewType);
        Modified = true;
      }

      if (typeRequiresConversion(I.getType())) {
        Type *NewType = convertType(I.getType());
        I.mutateType(NewType);
        Modified = true;
      }
    }
  }

  return Modified;
}

bool SPIRVFixAddressSpace::rewriteGEP(Function &F) {
  std::unordered_set<Instruction *> DeadInstructions;

  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *GEP = dyn_cast<GetElementPtrInst>(&I);
      if (!GEP)
        continue;

      Type *SourceType = convertType(GEP->getSourceElementType());

      IRBuilder<NoFolder> B(GEP->getParent(), NoFolder());
      B.SetInsertPoint(GEP);
      std::vector<Value *> Indices(GEP->idx_begin(), GEP->idx_end());
      auto *NewInstr =
          B.CreateGEP(SourceType, GEP->getPointerOperand(), Indices,
                      GEP->getName(), GEP->getNoWrapFlags());
      GEP->replaceAllUsesWith(NewInstr);
      DeadInstructions.insert(GEP);
    }
  }

  for (auto *I : DeadInstructions)
    I->eraseFromParent();
  return DeadInstructions.size() != 0;
}

bool SPIRVFixAddressSpace::fixInstructions(Function &F) {
  bool Modified = false;

  Modified |= replaceAlloca(F);
  Modified |= blindlyMutateTypes(F);
  Modified |= rewriteGEP(F);

  return Modified;
}

bool SPIRVFixAddressSpace::rewriteFunctionParameters(Module &M, Function *F) {
  if (F->isDeclaration())
    return false;

  FunctionType *NewType = convertFunctionType(F->getFunctionType());
  if (NewType == F->getFunctionType())
    return false;

  std::string OldName = F->getName().str();
  F->setName(OldName + ".dead");
  Function *NewFunction = Function::Create(NewType, F->getLinkage(),
                                           /* AddressSpace= */ 0, OldName);
  NewFunction->copyAttributesFrom(F);
  NewFunction->copyMetadata(F, 0);
  NewFunction->setIsNewDbgInfoFormat(F->IsNewDbgInfoFormat);
  M.getFunctionList().insert(F->getIterator(), NewFunction);

  std::vector<User *> ToFix(F->user_begin(), F->user_end());
  for (auto *User : ToFix) {
    User->replaceUsesOfWith(F, NewFunction);
    CallBase *CB = dyn_cast<CallBase>(User);
    if (!CB)
      continue;
    CB->mutateFunctionType(NewType);
  }

  for (size_t I = 0; I < NewFunction->arg_size(); ++I) {
    Argument *OldArgument = F->getArg(I);
    Argument *NewArgument = NewFunction->getArg(I);
    NewArgument->setName(OldArgument->getName());
    std::vector<User *> ToFix(OldArgument->user_begin(),
                              OldArgument->user_end());
    for (auto *User : ToFix)
      User->replaceUsesOfWith(OldArgument, NewArgument);
  }

  NewFunction->splice(NewFunction->begin(), F);
  F->eraseFromParent();
  return true;
}

bool SPIRVFixAddressSpace::fixFunctions(Module &M) {
  bool Modified = false;
  std::vector<Function *> WorkList;

  WorkList.reserve(M.size());
  for (Function &F : M)
    WorkList.push_back(&F);

  // If a function has a ptr argument/return value, we must
  // rewrite its definition to use ptr addrspace(10) instead.
  for (Function *F : WorkList)
    Modified |= rewriteFunctionParameters(M, F);

  // Once the function declarations are fixed, we must check each instruction,
  // and replace any use of `ptr` with `ptr addrspace(10)`.
  for (Function &F : M)
    Modified |= fixInstructions(F);

  return Modified;
}

bool SPIRVFixAddressSpace::fixGlobals(Module &M) {
  bool Modified = false;
  std::vector<GlobalVariable *> WorkList;

  WorkList.reserve(M.global_size());
  for (GlobalVariable &GV : M.globals())
    WorkList.push_back(&GV);

  for (GlobalVariable *GV : WorkList)
    Modified |= rewriteGlobalVariable(M, GV);

  return Modified;
}

} // anonymous namespace

bool SPIRVFixAddressSpace::runOnModule(Module &M) {
  bool Modified = false;
  Modified |= fixGlobals(M);
  Modified |= fixFunctions(M);
  return Modified;
}

char SPIRVFixAddressSpace::ID = 0;

INITIALIZE_PASS_BEGIN(SPIRVFixAddressSpace, "spirv-fix-address-space",
                      "Fixup address space", false, false)

INITIALIZE_PASS_END(SPIRVFixAddressSpace, "spirv-fix-address-space",
                    "Fixup address space", false, false)

ModulePass *llvm::createSPIRVFixAddressSpacePass() {
  return new SPIRVFixAddressSpace();
}
