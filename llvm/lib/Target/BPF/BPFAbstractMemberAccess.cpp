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

private:
  enum : uint32_t {
    BPFPreserveArrayAI = 1,
    BPFPreserveUnionAI = 2,
    BPFPreserveStructAI = 3,
  };

  std::map<std::string, GlobalVariable *> GEPGlobals;
  // A map to link preserve_*_access_index instrinsic calls.
  std::map<CallInst *, std::pair<CallInst *, uint32_t>> AIChain;
  // A map to hold all the base preserve_*_access_index instrinsic calls.
  // The base call is not an input of any other preserve_*_access_index
  // intrinsics.
  std::map<CallInst *, uint32_t> BaseAICalls;

  bool doTransformation(Module &M);

  void traceAICall(CallInst *Call, uint32_t Kind);
  void traceBitCast(BitCastInst *BitCast, CallInst *Parent, uint32_t Kind);
  void traceGEP(GetElementPtrInst *GEP, CallInst *Parent, uint32_t Kind);
  void collectAICallChains(Module &M, Function &F);

  bool IsPreserveDIAccessIndexCall(const CallInst *Call, uint32_t &Kind);
  bool removePreserveAccessIndexIntrinsic(Module &M);
  void replaceWithGEP(std::vector<CallInst *> &CallList,
                      uint32_t NumOfZerosIndex, uint32_t DIIndex);

  Value *computeBaseAndAccessKey(CallInst *Call, std::string &AccessKey,
                                 uint32_t Kind, MDNode *&TypeMeta);
  bool getAccessIndex(const Value *IndexValue, uint64_t &AccessIndex);
  bool transformGEPChain(Module &M, CallInst *Call, uint32_t Kind);
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
  if (empty(M.debug_compile_units()))
    return false;

  return doTransformation(M);
}

/// Check whether a call is a preserve_*_access_index intrinsic call or not.
bool BPFAbstractMemberAccess::IsPreserveDIAccessIndexCall(const CallInst *Call,
                                                          uint32_t &Kind) {
  if (!Call)
    return false;

  const auto *GV = dyn_cast<GlobalValue>(Call->getCalledValue());
  if (!GV)
    return false;
  if (GV->getName().startswith("llvm.preserve.array.access.index")) {
    Kind = BPFPreserveArrayAI;
    return true;
  }
  if (GV->getName().startswith("llvm.preserve.union.access.index")) {
    Kind = BPFPreserveUnionAI;
    return true;
  }
  if (GV->getName().startswith("llvm.preserve.struct.access.index")) {
    Kind = BPFPreserveStructAI;
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
      Dimension = cast<ConstantInt>(Call->getArgOperand(DimensionIndex))
                      ->getZExtValue();

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
        uint32_t Kind;
        if (!IsPreserveDIAccessIndexCall(Call, Kind))
          continue;

        Found = true;
        if (Kind == BPFPreserveArrayAI)
          PreserveArrayIndexCalls.push_back(Call);
        else if (Kind == BPFPreserveUnionAI)
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

void BPFAbstractMemberAccess::traceAICall(CallInst *Call, uint32_t Kind) {
  for (User *U : Call->users()) {
    Instruction *Inst = dyn_cast<Instruction>(U);
    if (!Inst)
      continue;

    if (auto *BI = dyn_cast<BitCastInst>(Inst)) {
      traceBitCast(BI, Call, Kind);
    } else if (auto *CI = dyn_cast<CallInst>(Inst)) {
      uint32_t CIKind;
      if (IsPreserveDIAccessIndexCall(CI, CIKind)) {
        AIChain[CI] = std::make_pair(Call, Kind);
        traceAICall(CI, CIKind);
      } else {
        BaseAICalls[Call] = Kind;
      }
    } else if (auto *GI = dyn_cast<GetElementPtrInst>(Inst)) {
      if (GI->hasAllZeroIndices())
        traceGEP(GI, Call, Kind);
      else
        BaseAICalls[Call] = Kind;
    }
  }
}

void BPFAbstractMemberAccess::traceBitCast(BitCastInst *BitCast,
                                           CallInst *Parent, uint32_t Kind) {
  for (User *U : BitCast->users()) {
    Instruction *Inst = dyn_cast<Instruction>(U);
    if (!Inst)
      continue;

    if (auto *BI = dyn_cast<BitCastInst>(Inst)) {
      traceBitCast(BI, Parent, Kind);
    } else if (auto *CI = dyn_cast<CallInst>(Inst)) {
      uint32_t CIKind;
      if (IsPreserveDIAccessIndexCall(CI, CIKind)) {
        AIChain[CI] = std::make_pair(Parent, Kind);
        traceAICall(CI, CIKind);
      } else {
        BaseAICalls[Parent] = Kind;
      }
    } else if (auto *GI = dyn_cast<GetElementPtrInst>(Inst)) {
      if (GI->hasAllZeroIndices())
        traceGEP(GI, Parent, Kind);
      else
        BaseAICalls[Parent] = Kind;
    }
  }
}

void BPFAbstractMemberAccess::traceGEP(GetElementPtrInst *GEP, CallInst *Parent,
                                       uint32_t Kind) {
  for (User *U : GEP->users()) {
    Instruction *Inst = dyn_cast<Instruction>(U);
    if (!Inst)
      continue;

    if (auto *BI = dyn_cast<BitCastInst>(Inst)) {
      traceBitCast(BI, Parent, Kind);
    } else if (auto *CI = dyn_cast<CallInst>(Inst)) {
      uint32_t CIKind;
      if (IsPreserveDIAccessIndexCall(CI, CIKind)) {
        AIChain[CI] = std::make_pair(Parent, Kind);
        traceAICall(CI, CIKind);
      } else {
        BaseAICalls[Parent] = Kind;
      }
    } else if (auto *GI = dyn_cast<GetElementPtrInst>(Inst)) {
      if (GI->hasAllZeroIndices())
        traceGEP(GI, Parent, Kind);
      else
        BaseAICalls[Parent] = Kind;
    }
  }
}

void BPFAbstractMemberAccess::collectAICallChains(Module &M, Function &F) {
  AIChain.clear();
  BaseAICalls.clear();

  for (auto &BB : F)
    for (auto &I : BB) {
      uint32_t Kind;
      auto *Call = dyn_cast<CallInst>(&I);
      if (!IsPreserveDIAccessIndexCall(Call, Kind) ||
          AIChain.find(Call) != AIChain.end())
        continue;

      traceAICall(Call, Kind);
    }
}

/// Get access index from the preserve_*_access_index intrinsic calls.
bool BPFAbstractMemberAccess::getAccessIndex(const Value *IndexValue,
                                             uint64_t &AccessIndex) {
  const ConstantInt *CV = dyn_cast<ConstantInt>(IndexValue);
  if (!CV)
    return false;

  AccessIndex = CV->getValue().getZExtValue();
  return true;
}

/// Compute the base of the whole preserve_*_access_index chains, i.e., the base
/// pointer of the first preserve_*_access_index call, and construct the access
/// string, which will be the name of a global variable.
Value *BPFAbstractMemberAccess::computeBaseAndAccessKey(CallInst *Call,
                                                        std::string &AccessKey,
                                                        uint32_t Kind,
                                                        MDNode *&TypeMeta) {
  Value *Base = nullptr;
  std::vector<uint64_t> AccessIndices;
  uint64_t TypeNameIndex = 0;
  std::string LastTypeName;

  while (Call) {
    // Base of original corresponding GEP
    Base = Call->getArgOperand(0);

    // Type Name
    std::string TypeName;
    MDNode *MDN;
    if (Kind == BPFPreserveUnionAI || Kind == BPFPreserveStructAI) {
      MDN = Call->getMetadata(LLVMContext::MD_preserve_access_index);
      if (!MDN)
        return nullptr;

      DIType *Ty = dyn_cast<DIType>(MDN);
      if (!Ty)
        return nullptr;

      TypeName = Ty->getName();
    }

    // Access Index
    uint64_t AccessIndex;
    uint32_t ArgIndex = (Kind == BPFPreserveUnionAI) ? 1 : 2;
    if (!getAccessIndex(Call->getArgOperand(ArgIndex), AccessIndex))
      return nullptr;

    AccessIndices.push_back(AccessIndex);
    if (TypeName.size()) {
      TypeNameIndex = AccessIndices.size() - 1;
      LastTypeName = TypeName;
      TypeMeta = MDN;
    }

    Kind = AIChain[Call].second;
    Call = AIChain[Call].first;
  }

  // The intial type name is required.
  // FIXME: if the initial type access is an array index, e.g.,
  // &a[3].b.c, only one dimentional array is supported.
  if (!LastTypeName.size() || AccessIndices.size() > TypeNameIndex + 2)
    return nullptr;

  // Construct the type string AccessKey.
  for (unsigned I = 0; I < AccessIndices.size(); ++I)
    AccessKey = std::to_string(AccessIndices[I]) + ":" + AccessKey;

  if (TypeNameIndex == AccessIndices.size() - 1)
    AccessKey = "0:" + AccessKey;

  // Access key is the type name + access string, uniquely identifying
  // one kernel memory access.
  AccessKey = LastTypeName + ":" + AccessKey;

  return Base;
}

/// Call/Kind is the base preserve_*_access_index() call. Attempts to do
/// transformation to a chain of relocable GEPs.
bool BPFAbstractMemberAccess::transformGEPChain(Module &M, CallInst *Call,
                                                uint32_t Kind) {
  std::string AccessKey;
  MDNode *TypeMeta = nullptr;
  Value *Base =
      computeBaseAndAccessKey(Call, AccessKey, Kind, TypeMeta);
  if (!Base)
    return false;

  // Do the transformation
  // For any original GEP Call and Base %2 like
  //   %4 = bitcast %struct.net_device** %dev1 to i64*
  // it is transformed to:
  //   %6 = load __BTF_0:sk_buff:0:0:2:0:
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
    // Set the metadata (debuginfo types) for the global.
    if (TypeMeta)
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
