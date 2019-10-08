//===------ BPFAbstractMemberAccess.cpp - Abstracting Member Accesses -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass abstracted struct/union member accesses in order to support
// compile-once run-everywhere (CO-RE). The CO-RE intends to compile the program
// which can run on different kernels. In particular, if bpf program tries to
// access a particular kernel data structure member, the details of the
// intermediate member access will be remembered so bpf loader can do
// necessary adjustment right before program loading.
//
// For example,
//
//   struct s {
//     int a;
//     int b;
//   };
//   struct t {
//     struct s c;
//     int d;
//   };
//   struct t e;
//
// For the member access e.c.b, the compiler will generate code
//   &e + 4
//
// The compile-once run-everywhere instead generates the following code
//   r = 4
//   &e + r
// The "4" in "r = 4" can be changed based on a particular kernel version.
// For example, on a particular kernel version, if struct s is changed to
//
//   struct s {
//     int new_field;
//     int a;
//     int b;
//   }
//
// By repeating the member access on the host, the bpf loader can
// adjust "r = 4" as "r = 8".
//
// This feature relies on the following three intrinsic calls:
//   addr = preserve_array_access_index(base, dimension, index)
//   addr = preserve_union_access_index(base, di_index)
//          !llvm.preserve.access.index <union_ditype>
//   addr = preserve_struct_access_index(base, gep_index, di_index)
//          !llvm.preserve.access.index <struct_ditype>
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFCORE.h"
#include "BPFTargetMachine.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <stack>

#define DEBUG_TYPE "bpf-abstract-member-access"

namespace llvm {
const std::string BPFCoreSharedInfo::AmaAttr = "btf_ama";
const std::string BPFCoreSharedInfo::PatchableExtSecName =
    ".BPF.patchable_externs";
} // namespace llvm

using namespace llvm;

namespace {

class BPFAbstractMemberAccess final : public ModulePass {
  StringRef getPassName() const override {
    return "BPF Abstract Member Access";
  }

  bool runOnModule(Module &M) override;

public:
  static char ID;
  BPFAbstractMemberAccess() : ModulePass(ID) {}

  struct CallInfo {
    uint32_t Kind;
    uint32_t AccessIndex;
    MDNode *Metadata;
    Value *Base;
  };

private:
  enum : uint32_t {
    BPFPreserveArrayAI = 1,
    BPFPreserveUnionAI = 2,
    BPFPreserveStructAI = 3,
  };

  std::map<std::string, GlobalVariable *> GEPGlobals;
  // A map to link preserve_*_access_index instrinsic calls.
  std::map<CallInst *, std::pair<CallInst *, CallInfo>> AIChain;
  // A map to hold all the base preserve_*_access_index instrinsic calls.
  // The base call is not an input of any other preserve_*_access_index
  // intrinsics.
  std::map<CallInst *, CallInfo> BaseAICalls;

  bool doTransformation(Module &M);

  void traceAICall(CallInst *Call, CallInfo &ParentInfo);
  void traceBitCast(BitCastInst *BitCast, CallInst *Parent,
                    CallInfo &ParentInfo);
  void traceGEP(GetElementPtrInst *GEP, CallInst *Parent,
                CallInfo &ParentInfo);
  void collectAICallChains(Module &M, Function &F);

  bool IsPreserveDIAccessIndexCall(const CallInst *Call, CallInfo &Cinfo);
  bool IsValidAIChain(const MDNode *ParentMeta, uint32_t ParentAI,
                      const MDNode *ChildMeta);
  bool removePreserveAccessIndexIntrinsic(Module &M);
  void replaceWithGEP(std::vector<CallInst *> &CallList,
                      uint32_t NumOfZerosIndex, uint32_t DIIndex);

  Value *computeBaseAndAccessKey(CallInst *Call, CallInfo &CInfo,
                                 std::string &AccessKey, MDNode *&BaseMeta);
  uint64_t getConstant(const Value *IndexValue);
  bool transformGEPChain(Module &M, CallInst *Call, CallInfo &CInfo);
};
} // End anonymous namespace

char BPFAbstractMemberAccess::ID = 0;
INITIALIZE_PASS(BPFAbstractMemberAccess, DEBUG_TYPE,
                "abstracting struct/union member accessees", false, false)

ModulePass *llvm::createBPFAbstractMemberAccess() {
  return new BPFAbstractMemberAccess();
}

bool BPFAbstractMemberAccess::runOnModule(Module &M) {
  LLVM_DEBUG(dbgs() << "********** Abstract Member Accesses **********\n");

  // Bail out if no debug info.
  if (M.debug_compile_units().empty())
    return false;

  return doTransformation(M);
}

static bool SkipDIDerivedTag(unsigned Tag) {
  if (Tag != dwarf::DW_TAG_typedef && Tag != dwarf::DW_TAG_const_type &&
      Tag != dwarf::DW_TAG_volatile_type &&
      Tag != dwarf::DW_TAG_restrict_type &&
      Tag != dwarf::DW_TAG_member)
     return false;
  return true;
}

static DIType * stripQualifiers(DIType *Ty) {
  while (auto *DTy = dyn_cast<DIDerivedType>(Ty)) {
    if (!SkipDIDerivedTag(DTy->getTag()))
      break;
    Ty = DTy->getBaseType();
  }
  return Ty;
}

static const DIType * stripQualifiers(const DIType *Ty) {
  while (auto *DTy = dyn_cast<DIDerivedType>(Ty)) {
    if (!SkipDIDerivedTag(DTy->getTag()))
      break;
    Ty = DTy->getBaseType();
  }
  return Ty;
}

static uint32_t calcArraySize(const DICompositeType *CTy, uint32_t StartDim) {
  DINodeArray Elements = CTy->getElements();
  uint32_t DimSize = 1;
  for (uint32_t I = StartDim; I < Elements.size(); ++I) {
    if (auto *Element = dyn_cast_or_null<DINode>(Elements[I]))
      if (Element->getTag() == dwarf::DW_TAG_subrange_type) {
        const DISubrange *SR = cast<DISubrange>(Element);
        auto *CI = SR->getCount().dyn_cast<ConstantInt *>();
        DimSize *= CI->getSExtValue();
      }
  }

  return DimSize;
}

/// Check whether a call is a preserve_*_access_index intrinsic call or not.
bool BPFAbstractMemberAccess::IsPreserveDIAccessIndexCall(const CallInst *Call,
                                                          CallInfo &CInfo) {
  if (!Call)
    return false;

  const auto *GV = dyn_cast<GlobalValue>(Call->getCalledValue());
  if (!GV)
    return false;
  if (GV->getName().startswith("llvm.preserve.array.access.index")) {
    CInfo.Kind = BPFPreserveArrayAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.array.access.index intrinsic");
    CInfo.AccessIndex = getConstant(Call->getArgOperand(2));
    CInfo.Base = Call->getArgOperand(0);
    return true;
  }
  if (GV->getName().startswith("llvm.preserve.union.access.index")) {
    CInfo.Kind = BPFPreserveUnionAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.union.access.index intrinsic");
    CInfo.AccessIndex = getConstant(Call->getArgOperand(1));
    CInfo.Base = Call->getArgOperand(0);
    return true;
  }
  if (GV->getName().startswith("llvm.preserve.struct.access.index")) {
    CInfo.Kind = BPFPreserveStructAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.struct.access.index intrinsic");
    CInfo.AccessIndex = getConstant(Call->getArgOperand(2));
    CInfo.Base = Call->getArgOperand(0);
    return true;
  }

  return false;
}

void BPFAbstractMemberAccess::replaceWithGEP(std::vector<CallInst *> &CallList,
                                             uint32_t DimensionIndex,
                                             uint32_t GEPIndex) {
  for (auto Call : CallList) {
    uint32_t Dimension = 1;
    if (DimensionIndex > 0)
      Dimension = getConstant(Call->getArgOperand(DimensionIndex));

    Constant *Zero =
        ConstantInt::get(Type::getInt32Ty(Call->getParent()->getContext()), 0);
    SmallVector<Value *, 4> IdxList;
    for (unsigned I = 0; I < Dimension; ++I)
      IdxList.push_back(Zero);
    IdxList.push_back(Call->getArgOperand(GEPIndex));

    auto *GEP = GetElementPtrInst::CreateInBounds(Call->getArgOperand(0),
                                                  IdxList, "", Call);
    Call->replaceAllUsesWith(GEP);
    Call->eraseFromParent();
  }
}

bool BPFAbstractMemberAccess::removePreserveAccessIndexIntrinsic(Module &M) {
  std::vector<CallInst *> PreserveArrayIndexCalls;
  std::vector<CallInst *> PreserveUnionIndexCalls;
  std::vector<CallInst *> PreserveStructIndexCalls;
  bool Found = false;

  for (Function &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *Call = dyn_cast<CallInst>(&I);
        CallInfo CInfo;
        if (!IsPreserveDIAccessIndexCall(Call, CInfo))
          continue;

        Found = true;
        if (CInfo.Kind == BPFPreserveArrayAI)
          PreserveArrayIndexCalls.push_back(Call);
        else if (CInfo.Kind == BPFPreserveUnionAI)
          PreserveUnionIndexCalls.push_back(Call);
        else
          PreserveStructIndexCalls.push_back(Call);
      }

  // do the following transformation:
  // . addr = preserve_array_access_index(base, dimension, index)
  //   is transformed to
  //     addr = GEP(base, dimenion's zero's, index)
  // . addr = preserve_union_access_index(base, di_index)
  //   is transformed to
  //     addr = base, i.e., all usages of "addr" are replaced by "base".
  // . addr = preserve_struct_access_index(base, gep_index, di_index)
  //   is transformed to
  //     addr = GEP(base, 0, gep_index)
  replaceWithGEP(PreserveArrayIndexCalls, 1, 2);
  replaceWithGEP(PreserveStructIndexCalls, 0, 1);
  for (auto Call : PreserveUnionIndexCalls) {
    Call->replaceAllUsesWith(Call->getArgOperand(0));
    Call->eraseFromParent();
  }

  return Found;
}

/// Check whether the access index chain is valid. We check
/// here because there may be type casts between two
/// access indexes. We want to ensure memory access still valid.
bool BPFAbstractMemberAccess::IsValidAIChain(const MDNode *ParentType,
                                             uint32_t ParentAI,
                                             const MDNode *ChildType) {
  const DIType *PType = stripQualifiers(cast<DIType>(ParentType));
  const DIType *CType = stripQualifiers(cast<DIType>(ChildType));

  // Child is a derived/pointer type, which is due to type casting.
  // Pointer type cannot be in the middle of chain.
  if (isa<DIDerivedType>(CType))
    return false;

  // Parent is a pointer type.
  if (const auto *PtrTy = dyn_cast<DIDerivedType>(PType)) {
    if (PtrTy->getTag() != dwarf::DW_TAG_pointer_type)
      return false;
    return stripQualifiers(PtrTy->getBaseType()) == CType;
  }

  // Otherwise, struct/union/array types
  const auto *PTy = dyn_cast<DICompositeType>(PType);
  const auto *CTy = dyn_cast<DICompositeType>(CType);
  assert(PTy && CTy && "ParentType or ChildType is null or not composite");

  uint32_t PTyTag = PTy->getTag();
  assert(PTyTag == dwarf::DW_TAG_array_type ||
         PTyTag == dwarf::DW_TAG_structure_type ||
         PTyTag == dwarf::DW_TAG_union_type);

  uint32_t CTyTag = CTy->getTag();
  assert(CTyTag == dwarf::DW_TAG_array_type ||
         CTyTag == dwarf::DW_TAG_structure_type ||
         CTyTag == dwarf::DW_TAG_union_type);

  // Multi dimensional arrays, base element should be the same
  if (PTyTag == dwarf::DW_TAG_array_type && PTyTag == CTyTag)
    return PTy->getBaseType() == CTy->getBaseType();

  DIType *Ty;
  if (PTyTag == dwarf::DW_TAG_array_type)
    Ty = PTy->getBaseType();
  else
    Ty = dyn_cast<DIType>(PTy->getElements()[ParentAI]);

  return dyn_cast<DICompositeType>(stripQualifiers(Ty)) == CTy;
}

void BPFAbstractMemberAccess::traceAICall(CallInst *Call,
                                          CallInfo &ParentInfo) {
  for (User *U : Call->users()) {
    Instruction *Inst = dyn_cast<Instruction>(U);
    if (!Inst)
      continue;

    if (auto *BI = dyn_cast<BitCastInst>(Inst)) {
      traceBitCast(BI, Call, ParentInfo);
    } else if (auto *CI = dyn_cast<CallInst>(Inst)) {
      CallInfo ChildInfo;

      if (IsPreserveDIAccessIndexCall(CI, ChildInfo) &&
          IsValidAIChain(ParentInfo.Metadata, ParentInfo.AccessIndex,
                         ChildInfo.Metadata)) {
        AIChain[CI] = std::make_pair(Call, ParentInfo);
        traceAICall(CI, ChildInfo);
      } else {
        BaseAICalls[Call] = ParentInfo;
      }
    } else if (auto *GI = dyn_cast<GetElementPtrInst>(Inst)) {
      if (GI->hasAllZeroIndices())
        traceGEP(GI, Call, ParentInfo);
      else
        BaseAICalls[Call] = ParentInfo;
    } else {
      BaseAICalls[Call] = ParentInfo;
    }
  }
}

void BPFAbstractMemberAccess::traceBitCast(BitCastInst *BitCast,
                                           CallInst *Parent,
                                           CallInfo &ParentInfo) {
  for (User *U : BitCast->users()) {
    Instruction *Inst = dyn_cast<Instruction>(U);
    if (!Inst)
      continue;

    if (auto *BI = dyn_cast<BitCastInst>(Inst)) {
      traceBitCast(BI, Parent, ParentInfo);
    } else if (auto *CI = dyn_cast<CallInst>(Inst)) {
      CallInfo ChildInfo;
      if (IsPreserveDIAccessIndexCall(CI, ChildInfo) &&
          IsValidAIChain(ParentInfo.Metadata, ParentInfo.AccessIndex,
                         ChildInfo.Metadata)) {
        AIChain[CI] = std::make_pair(Parent, ParentInfo);
        traceAICall(CI, ChildInfo);
      } else {
        BaseAICalls[Parent] = ParentInfo;
      }
    } else if (auto *GI = dyn_cast<GetElementPtrInst>(Inst)) {
      if (GI->hasAllZeroIndices())
        traceGEP(GI, Parent, ParentInfo);
      else
        BaseAICalls[Parent] = ParentInfo;
    } else {
      BaseAICalls[Parent] = ParentInfo;
    }
  }
}

void BPFAbstractMemberAccess::traceGEP(GetElementPtrInst *GEP, CallInst *Parent,
                                       CallInfo &ParentInfo) {
  for (User *U : GEP->users()) {
    Instruction *Inst = dyn_cast<Instruction>(U);
    if (!Inst)
      continue;

    if (auto *BI = dyn_cast<BitCastInst>(Inst)) {
      traceBitCast(BI, Parent, ParentInfo);
    } else if (auto *CI = dyn_cast<CallInst>(Inst)) {
      CallInfo ChildInfo;
      if (IsPreserveDIAccessIndexCall(CI, ChildInfo) &&
          IsValidAIChain(ParentInfo.Metadata, ParentInfo.AccessIndex,
                         ChildInfo.Metadata)) {
        AIChain[CI] = std::make_pair(Parent, ParentInfo);
        traceAICall(CI, ChildInfo);
      } else {
        BaseAICalls[Parent] = ParentInfo;
      }
    } else if (auto *GI = dyn_cast<GetElementPtrInst>(Inst)) {
      if (GI->hasAllZeroIndices())
        traceGEP(GI, Parent, ParentInfo);
      else
        BaseAICalls[Parent] = ParentInfo;
    } else {
      BaseAICalls[Parent] = ParentInfo;
    }
  }
}

void BPFAbstractMemberAccess::collectAICallChains(Module &M, Function &F) {
  AIChain.clear();
  BaseAICalls.clear();

  for (auto &BB : F)
    for (auto &I : BB) {
      CallInfo CInfo;
      auto *Call = dyn_cast<CallInst>(&I);
      if (!IsPreserveDIAccessIndexCall(Call, CInfo) ||
          AIChain.find(Call) != AIChain.end())
        continue;

      traceAICall(Call, CInfo);
    }
}

uint64_t BPFAbstractMemberAccess::getConstant(const Value *IndexValue) {
  const ConstantInt *CV = dyn_cast<ConstantInt>(IndexValue);
  assert(CV);
  return CV->getValue().getZExtValue();
}

/// Compute the base of the whole preserve_*_access_index chains, i.e., the base
/// pointer of the first preserve_*_access_index call, and construct the access
/// string, which will be the name of a global variable.
Value *BPFAbstractMemberAccess::computeBaseAndAccessKey(CallInst *Call,
                                                        CallInfo &CInfo,
                                                        std::string &AccessKey,
                                                        MDNode *&TypeMeta) {
  Value *Base = nullptr;
  std::string TypeName;
  std::stack<std::pair<CallInst *, CallInfo>> CallStack;

  // Put the access chain into a stack with the top as the head of the chain.
  while (Call) {
    CallStack.push(std::make_pair(Call, CInfo));
    CInfo = AIChain[Call].second;
    Call = AIChain[Call].first;
  }

  // The access offset from the base of the head of chain is also
  // calculated here as all debuginfo types are available.

  // Get type name and calculate the first index.
  // We only want to get type name from structure or union.
  // If user wants a relocation like
  //    int *p; ... __builtin_preserve_access_index(&p[4]) ...
  // or
  //    int a[10][20]; ... __builtin_preserve_access_index(&a[2][3]) ...
  // we will skip them.
  uint32_t FirstIndex = 0;
  uint32_t AccessOffset = 0;
  while (CallStack.size()) {
    auto StackElem = CallStack.top();
    Call = StackElem.first;
    CInfo = StackElem.second;

    if (!Base)
      Base = CInfo.Base;

    DIType *Ty = stripQualifiers(cast<DIType>(CInfo.Metadata));
    if (CInfo.Kind == BPFPreserveUnionAI ||
        CInfo.Kind == BPFPreserveStructAI) {
      // struct or union type
      TypeName = Ty->getName();
      TypeMeta = Ty;
      AccessOffset += FirstIndex * Ty->getSizeInBits() >> 3;
      break;
    }

    // Array entries will always be consumed for accumulative initial index.
    CallStack.pop();

    // BPFPreserveArrayAI
    uint64_t AccessIndex = CInfo.AccessIndex;

    DIType *BaseTy = nullptr;
    bool CheckElemType = false;
    if (const auto *CTy = dyn_cast<DICompositeType>(Ty)) {
      // array type
      assert(CTy->getTag() == dwarf::DW_TAG_array_type);


      FirstIndex += AccessIndex * calcArraySize(CTy, 1);
      BaseTy = stripQualifiers(CTy->getBaseType());
      CheckElemType = CTy->getElements().size() == 1;
    } else {
      // pointer type
      auto *DTy = cast<DIDerivedType>(Ty);
      assert(DTy->getTag() == dwarf::DW_TAG_pointer_type);

      BaseTy = stripQualifiers(DTy->getBaseType());
      CTy = dyn_cast<DICompositeType>(BaseTy);
      if (!CTy) {
        CheckElemType = true;
      } else if (CTy->getTag() != dwarf::DW_TAG_array_type) {
        FirstIndex += AccessIndex;
        CheckElemType = true;
      } else {
        FirstIndex += AccessIndex * calcArraySize(CTy, 0);
      }
    }

    if (CheckElemType) {
      auto *CTy = dyn_cast<DICompositeType>(BaseTy);
      if (!CTy)
        return nullptr;

      unsigned CTag = CTy->getTag();
      if (CTag != dwarf::DW_TAG_structure_type && CTag != dwarf::DW_TAG_union_type)
        return nullptr;
      else
        TypeName = CTy->getName();
      TypeMeta = CTy;
      AccessOffset += FirstIndex * CTy->getSizeInBits() >> 3;
      break;
    }
  }
  assert(TypeName.size());
  AccessKey += std::to_string(FirstIndex);

  // Traverse the rest of access chain to complete offset calculation
  // and access key construction.
  while (CallStack.size()) {
    auto StackElem = CallStack.top();
    CInfo = StackElem.second;
    CallStack.pop();

    // Access Index
    uint64_t AccessIndex = CInfo.AccessIndex;
    AccessKey += ":" + std::to_string(AccessIndex);

    MDNode *MDN = CInfo.Metadata;
    // At this stage, it cannot be pointer type.
    auto *CTy = cast<DICompositeType>(stripQualifiers(cast<DIType>(MDN)));
    uint32_t Tag = CTy->getTag();
    if (Tag == dwarf::DW_TAG_structure_type) {
      auto *MemberTy = cast<DIDerivedType>(CTy->getElements()[AccessIndex]);
      AccessOffset += MemberTy->getOffsetInBits() >> 3;
    } else if (Tag == dwarf::DW_TAG_array_type) {
      auto *EltTy = stripQualifiers(CTy->getBaseType());
      AccessOffset += AccessIndex * calcArraySize(CTy, 1) *
                      EltTy->getSizeInBits() >> 3;
    }
  }

  // Access key is the type name + access string, uniquely identifying
  // one kernel memory access.
  AccessKey = TypeName + ":" + std::to_string(AccessOffset) + "$" + AccessKey;

  return Base;
}

/// Call/Kind is the base preserve_*_access_index() call. Attempts to do
/// transformation to a chain of relocable GEPs.
bool BPFAbstractMemberAccess::transformGEPChain(Module &M, CallInst *Call,
                                                CallInfo &CInfo) {
  std::string AccessKey;
  MDNode *TypeMeta;
  Value *Base =
      computeBaseAndAccessKey(Call, CInfo, AccessKey, TypeMeta);
  if (!Base)
    return false;

  // Do the transformation
  // For any original GEP Call and Base %2 like
  //   %4 = bitcast %struct.net_device** %dev1 to i64*
  // it is transformed to:
  //   %6 = load sk_buff:50:$0:0:0:2:0
  //   %7 = bitcast %struct.sk_buff* %2 to i8*
  //   %8 = getelementptr i8, i8* %7, %6
  //   %9 = bitcast i8* %8 to i64*
  //   using %9 instead of %4
  // The original Call inst is removed.
  BasicBlock *BB = Call->getParent();
  GlobalVariable *GV;

  if (GEPGlobals.find(AccessKey) == GEPGlobals.end()) {
    GV = new GlobalVariable(M, Type::getInt64Ty(BB->getContext()), false,
                            GlobalVariable::ExternalLinkage, NULL, AccessKey);
    GV->addAttribute(BPFCoreSharedInfo::AmaAttr);
    GV->setMetadata(LLVMContext::MD_preserve_access_index, TypeMeta);
    GEPGlobals[AccessKey] = GV;
  } else {
    GV = GEPGlobals[AccessKey];
  }

  // Load the global variable.
  auto *LDInst = new LoadInst(Type::getInt64Ty(BB->getContext()), GV);
  BB->getInstList().insert(Call->getIterator(), LDInst);

  // Generate a BitCast
  auto *BCInst = new BitCastInst(Base, Type::getInt8PtrTy(BB->getContext()));
  BB->getInstList().insert(Call->getIterator(), BCInst);

  // Generate a GetElementPtr
  auto *GEP = GetElementPtrInst::Create(Type::getInt8Ty(BB->getContext()),
                                        BCInst, LDInst);
  BB->getInstList().insert(Call->getIterator(), GEP);

  // Generate a BitCast
  auto *BCInst2 = new BitCastInst(GEP, Call->getType());
  BB->getInstList().insert(Call->getIterator(), BCInst2);

  Call->replaceAllUsesWith(BCInst2);
  Call->eraseFromParent();

  return true;
}

bool BPFAbstractMemberAccess::doTransformation(Module &M) {
  bool Transformed = false;

  for (Function &F : M) {
    // Collect PreserveDIAccessIndex Intrinsic call chains.
    // The call chains will be used to generate the access
    // patterns similar to GEP.
    collectAICallChains(M, F);

    for (auto &C : BaseAICalls)
      Transformed = transformGEPChain(M, C.first, C.second) || Transformed;
  }

  return removePreserveAccessIndexIntrinsic(M) || Transformed;
}
