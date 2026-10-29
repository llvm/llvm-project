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
// Bitfield member access needs special attention. User cannot take the
// address of a bitfield acceess. To facilitate kernel verifier
// for easy bitfield code optimization, a new clang intrinsic is introduced:
//   uint32_t __builtin_preserve_field_info(member_access, info_kind)
// In IR, a chain with two (or more) intrinsic calls will be generated:
//   ...
//   addr = preserve_struct_access_index(base, 1, 1) !struct s
//   uint32_t result = bpf_preserve_field_info(addr, info_kind)
//
// Suppose the info_kind is FIELD_SIGNEDNESS,
// The above two IR intrinsics will be replaced with
// a relocatable insn:
//   signness = /* signness of member_access */
// and signness can be changed by bpf loader based on the
// types on the host.
//
// User can also test whether a field exists or not with
//   uint32_t result = bpf_preserve_field_info(member_access, FIELD_EXISTENCE)
// The field will be always available (result = 1) during initial
// compilation, but bpf loader can patch with the correct value
// on the target host where the member_access may or may not be available
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFCORE.h"
#include "BPFTargetMachine.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/BTF/BTF.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsBPF.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <optional>

#define DEBUG_TYPE "bpf-abstract-member-access"

namespace llvm {
uint32_t BPFCoreSharedInfo::SeqNum;

BPFCoreSharedInfo::AmaRelocation
BPFCoreSharedInfo::AmaRelocation::decode(StringRef Name) {
  // Type names may contain ':' and '$'. Only the suffix has a fixed format.
  auto [Prefix, AccessString] = Name.rsplit('$');
  auto [NameAndKind, PatchString] = Prefix.rsplit(':');
  StringRef KindString = NameAndKind.rsplit(':').second;
  uint32_t Kind;
  int64_t PatchImm;
  if (AccessString.empty() || KindString.getAsInteger(10, Kind) ||
      PatchString.getAsInteger(10, PatchImm))
    reportFatalInternalError(Twine("Invalid BPF CO-RE relocation global ") +
                             Name);
  return {static_cast<BTF::PatchableRelocKind>(Kind), PatchImm, AccessString};
}

Instruction *BPFCoreSharedInfo::insertPassThrough(Module *M, BasicBlock *BB,
                                                  Instruction *Input,
                                                  Instruction *Before) {
  Function *Fn = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::bpf_passthrough, {Input->getType(), Input->getType()});
  Constant *SeqNumVal = ConstantInt::get(Type::getInt32Ty(BB->getContext()),
                                         BPFCoreSharedInfo::SeqNum++);

  auto *NewInst = CallInst::Create(Fn, {SeqNumVal, Input});
  NewInst->insertBefore(Before->getIterator());
  return NewInst;
}
} // namespace llvm

using namespace llvm;

namespace {
class BPFAbstractMemberAccess final {
public:
  BPFAbstractMemberAccess(BPFTargetMachine *TM) : TM(TM) {}

  bool run(Function &F);

  struct CallInfo {
    uint32_t Kind;
    uint32_t AccessIndex;
    MaybeAlign RecordAlignment;
    MDNode *Metadata;
    WeakTrackingVH Base;
  };

private:
  enum : uint32_t {
    BPFPreserveArrayAI = 1,
    BPFPreserveUnionAI = 2,
    BPFPreserveStructAI = 3,
    BPFPreserveFieldInfoAI = 4,
  };

  TargetMachine *TM;
  const DataLayout *DL = nullptr;
  Module *M = nullptr;

  // A map to link preserve_*_access_index intrinsic calls.
  std::map<CallInst *, std::pair<CallInst *, CallInfo>> AIChain;
  // A map to hold all the base preserve_*_access_index intrinsic calls.
  // The base call is not an input of any other preserve_*
  // intrinsics.
  // Iterated below, so the order can't come from the addresses.
  SmallMapVector<CallInst *, CallInfo, 4> BaseAICalls;
  // A map to hold <AnonRecord, TypeDef> relationships
  std::map<DICompositeType *, DIDerivedType *> AnonRecords;

  void CheckAnonRecordType(DIDerivedType *ParentTy, DIType *Ty);
  void CheckCompositeType(DIDerivedType *ParentTy, DICompositeType *CTy);
  void CheckDerivedType(DIDerivedType *ParentTy, DIDerivedType *DTy);
  void ResetMetadata(struct CallInfo &CInfo);

  bool doTransformation(Function &F);

  void traceAICall(CallInst *Call, CallInfo &ParentInfo);
  void traceBitCast(BitCastInst *BitCast, CallInst *Parent,
                    CallInfo &ParentInfo);
  void traceGEP(GetElementPtrInst *GEP, CallInst *Parent,
                CallInfo &ParentInfo);
  void collectAICallChains(Function &F);

  bool IsPreserveDIAccessIndexCall(const CallInst *Call, CallInfo &Cinfo);
  bool IsValidAIChain(const MDNode *ParentMeta, uint32_t ParentAI,
                      const MDNode *ChildMeta);
  bool removePreserveAccessIndexIntrinsic(Function &F);
  void GetStorageBitRange(DIDerivedType *MemberTy, Align RecordAlignment,
                          uint32_t &StartBitOffset, uint32_t &EndBitOffset);
  uint32_t GetFieldInfo(uint32_t InfoKind, DICompositeType *CTy,
                        uint32_t AccessIndex, uint32_t PatchImm,
                        MaybeAlign RecordAlignment);

  struct AccessRoot {
    DIType *Type;
    uint32_t Index;
  };
  static std::optional<AccessRoot>
  consumeAccessRoot(SmallVectorImpl<CallInfo> &CallStack);
  Value *computeFieldAccess(CallInst *Call, const CallInfo &Terminal,
                            std::string &RelocSuffix, DIType *&RootType);
  struct EnumAccess {
    size_t Index;
    StringRef Value;
  };
  static EnumAccess computeEnumAccess(const CallInst *Call, const DIType *Ty);
  DIType *computeTypeAccess(CallInst *Call, const CallInfo &CInfo,
                            std::string &RelocSuffix);
  bool transformGEPChain(CallInst *Call, const CallInfo &CInfo);
};

} // End anonymous namespace

bool BPFAbstractMemberAccess::run(Function &F) {
  LLVM_DEBUG(dbgs() << "********** Abstract Member Accesses **********\n");

  M = F.getParent();
  if (!M)
    return false;

  // Bail out if no debug info.
  if (M->debug_compile_units().empty())
    return false;

  // For each argument/return/local_variable type, trace the type
  // pattern like '[derived_type]* [composite_type]' to check
  // and remember (anon record -> typedef) relations where the
  // anon record is defined as
  //   typedef [const/volatile/restrict]* [anon record]
  DISubprogram *SP = F.getSubprogram();
  if (SP && SP->isDefinition()) {
    for (DIType *Ty: SP->getType()->getTypeArray())
      CheckAnonRecordType(nullptr, Ty);
    for (const MDNode *DN : SP->getRetainedNodes()) {
      if (const auto *DV = dyn_cast<DILocalVariable>(DN))
        CheckAnonRecordType(nullptr, DV->getType());
    }
  }

  DL = &M->getDataLayout();
  return doTransformation(F);
}

void BPFAbstractMemberAccess::ResetMetadata(struct CallInfo &CInfo) {
  if (auto Ty = dyn_cast<DICompositeType>(CInfo.Metadata)) {
    auto It = AnonRecords.find(Ty);
    if (It != AnonRecords.end() && It->second != nullptr)
      CInfo.Metadata = It->second;
  }
}

void BPFAbstractMemberAccess::CheckCompositeType(DIDerivedType *ParentTy,
                                                 DICompositeType *CTy) {
  if (!CTy->getName().empty() || !ParentTy ||
      ParentTy->getTag() != dwarf::DW_TAG_typedef)
    return;

  auto [It, Inserted] = AnonRecords.try_emplace(CTy, ParentTy);
  // Two or more typedef's may point to the same anon record.
  // If this is the case, set the typedef DIType to be nullptr
  // to indicate the duplication case.
  if (!Inserted && It->second != ParentTy)
    It->second = nullptr;
}

void BPFAbstractMemberAccess::CheckDerivedType(DIDerivedType *ParentTy,
                                               DIDerivedType *DTy) {
  DIType *BaseType = DTy->getBaseType();
  if (!BaseType)
    return;

  unsigned Tag = DTy->getTag();
  if (Tag == dwarf::DW_TAG_pointer_type)
    CheckAnonRecordType(nullptr, BaseType);
  else if (Tag == dwarf::DW_TAG_typedef)
    CheckAnonRecordType(DTy, BaseType);
  else
    CheckAnonRecordType(ParentTy, BaseType);
}

void BPFAbstractMemberAccess::CheckAnonRecordType(DIDerivedType *ParentTy,
                                                  DIType *Ty) {
  if (!Ty)
    return;

  if (auto *CTy = dyn_cast<DICompositeType>(Ty))
    return CheckCompositeType(ParentTy, CTy);
  else if (auto *DTy = dyn_cast<DIDerivedType>(Ty))
    return CheckDerivedType(ParentTy, DTy);
}

static bool SkipDIDerivedTag(unsigned Tag, bool skipTypedef) {
  if (Tag != dwarf::DW_TAG_typedef && Tag != dwarf::DW_TAG_const_type &&
      Tag != dwarf::DW_TAG_volatile_type &&
      Tag != dwarf::DW_TAG_restrict_type &&
      Tag != dwarf::DW_TAG_member)
    return false;
  if (Tag == dwarf::DW_TAG_typedef && !skipTypedef)
    return false;
  return true;
}

static DIType * stripQualifiers(DIType *Ty, bool skipTypedef = true) {
  while (auto *DTy = dyn_cast<DIDerivedType>(Ty)) {
    if (!SkipDIDerivedTag(DTy->getTag(), skipTypedef))
      break;
    Ty = DTy->getBaseType();
  }
  return Ty;
}

static const DIType * stripQualifiers(const DIType *Ty) {
  while (auto *DTy = dyn_cast<DIDerivedType>(Ty)) {
    if (!SkipDIDerivedTag(DTy->getTag(), true))
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
        auto *CI = dyn_cast<ConstantInt *>(SR->getCount());
        DimSize *= CI->getSExtValue();
      }
  }

  return DimSize;
}

static Type *getBaseElementType(const CallInst *Call) {
  // Element type is stored in an elementtype() attribute on the first param.
  return Call->getParamElementType(0);
}

static uint64_t getConstant(const Value *IndexValue) {
  const ConstantInt *CV = dyn_cast<ConstantInt>(IndexValue);
  assert(CV);
  return CV->getValue().getZExtValue();
}

/// Check whether a call is a preserve_*_access_index intrinsic call or not.
bool BPFAbstractMemberAccess::IsPreserveDIAccessIndexCall(const CallInst *Call,
                                                          CallInfo &CInfo) {
  if (!Call)
    return false;

  const auto *GV = dyn_cast<GlobalValue>(Call->getCalledOperand());
  if (!GV)
    return false;
  if (GV->getName().starts_with("llvm.preserve.array.access.index")) {
    CInfo.Kind = BPFPreserveArrayAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.array.access.index intrinsic");
    CInfo.AccessIndex = getConstant(Call->getArgOperand(2));
    CInfo.Base = Call->getArgOperand(0);
    CInfo.RecordAlignment = DL->getABITypeAlign(getBaseElementType(Call));
    return true;
  }
  if (GV->getName().starts_with("llvm.preserve.union.access.index")) {
    CInfo.Kind = BPFPreserveUnionAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.union.access.index intrinsic");
    ResetMetadata(CInfo);
    CInfo.AccessIndex = getConstant(Call->getArgOperand(1));
    CInfo.Base = Call->getArgOperand(0);
    return true;
  }
  if (GV->getName().starts_with("llvm.preserve.struct.access.index")) {
    CInfo.Kind = BPFPreserveStructAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.struct.access.index intrinsic");
    ResetMetadata(CInfo);
    CInfo.AccessIndex = getConstant(Call->getArgOperand(2));
    CInfo.Base = Call->getArgOperand(0);
    CInfo.RecordAlignment = DL->getABITypeAlign(getBaseElementType(Call));
    return true;
  }
  if (GV->getName().starts_with("llvm.bpf.preserve.field.info")) {
    CInfo.Kind = BPFPreserveFieldInfoAI;
    CInfo.Metadata = nullptr;
    // Check validity of info_kind as clang did not check this.
    uint64_t InfoKind = getConstant(Call->getArgOperand(1));
    if (InfoKind >= BTF::MAX_FIELD_RELOC_KIND)
      report_fatal_error("Incorrect info_kind for llvm.bpf.preserve.field.info intrinsic");
    CInfo.AccessIndex = InfoKind;
    return true;
  }
  if (GV->getName().starts_with("llvm.bpf.preserve.type.info")) {
    CInfo.Kind = BPFPreserveFieldInfoAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.type.info intrinsic");
    uint64_t Flag = getConstant(Call->getArgOperand(1));
    if (Flag >= BPFCoreSharedInfo::MAX_PRESERVE_TYPE_INFO_FLAG)
      report_fatal_error("Incorrect flag for llvm.bpf.preserve.type.info intrinsic");
    if (Flag == BPFCoreSharedInfo::PRESERVE_TYPE_INFO_EXISTENCE)
      CInfo.AccessIndex = BTF::TYPE_EXISTENCE;
    else if (Flag == BPFCoreSharedInfo::PRESERVE_TYPE_INFO_MATCH)
      CInfo.AccessIndex = BTF::TYPE_MATCH;
    else
      CInfo.AccessIndex = BTF::TYPE_SIZE;
    return true;
  }
  if (GV->getName().starts_with("llvm.bpf.preserve.enum.value")) {
    CInfo.Kind = BPFPreserveFieldInfoAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.enum.value intrinsic");
    uint64_t Flag = getConstant(Call->getArgOperand(2));
    if (Flag >= BPFCoreSharedInfo::MAX_PRESERVE_ENUM_VALUE_FLAG)
      report_fatal_error("Incorrect flag for llvm.bpf.preserve.enum.value intrinsic");
    if (Flag == BPFCoreSharedInfo::PRESERVE_ENUM_VALUE_EXISTENCE)
      CInfo.AccessIndex = BTF::ENUM_VALUE_EXISTENCE;
    else
      CInfo.AccessIndex = BTF::ENUM_VALUE;
    return true;
  }

  return false;
}

static void replaceWithGEP(CallInst *Call, uint32_t DimensionIndex,
                           uint32_t GEPIndex) {
  uint32_t Dimension = 1;
  if (DimensionIndex > 0)
    Dimension = getConstant(Call->getArgOperand(DimensionIndex));

  Constant *Zero =
      ConstantInt::get(Type::getInt32Ty(Call->getParent()->getContext()), 0);
  SmallVector<Value *, 4> IdxList(Dimension, Zero);
  IdxList.push_back(Call->getArgOperand(GEPIndex));

  auto *GEP = GetElementPtrInst::CreateInBounds(getBaseElementType(Call),
                                                Call->getArgOperand(0), IdxList,
                                                "", Call->getIterator());
  Call->replaceAllUsesWith(GEP);
  Call->eraseFromParent();
}

void BPFCoreSharedInfo::removeArrayAccessCall(CallInst *Call) {
  replaceWithGEP(Call, 1, 2);
}

void BPFCoreSharedInfo::removeStructAccessCall(CallInst *Call) {
  replaceWithGEP(Call, 0, 1);
}

void BPFCoreSharedInfo::removeUnionAccessCall(CallInst *Call) {
  Call->replaceAllUsesWith(Call->getArgOperand(0));
  Call->eraseFromParent();
}

bool BPFAbstractMemberAccess::removePreserveAccessIndexIntrinsic(Function &F) {
  std::vector<CallInst *> PreserveArrayIndexCalls;
  std::vector<CallInst *> PreserveUnionIndexCalls;
  std::vector<CallInst *> PreserveStructIndexCalls;
  std::vector<CallInst *> PreserveFieldInfoCalls;
  bool Found = false;

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
      else if (CInfo.Kind == BPFPreserveFieldInfoAI)
        PreserveFieldInfoCalls.push_back(Call);
      else
        PreserveStructIndexCalls.push_back(Call);
    }

  // A preserve_field_info(), preserve_type_info() or preserve_enum_value()
  // call left here has an unused result, so it is not relocated. It has no
  // side effects, so just remove it.
  for (CallInst *Call : PreserveFieldInfoCalls)
    if (Call->use_empty())
      Call->eraseFromParent();

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
  for (CallInst *Call : PreserveArrayIndexCalls)
    BPFCoreSharedInfo::removeArrayAccessCall(Call);
  for (CallInst *Call : PreserveStructIndexCalls)
    BPFCoreSharedInfo::removeStructAccessCall(Call);
  for (CallInst *Call : PreserveUnionIndexCalls)
    BPFCoreSharedInfo::removeUnionAccessCall(Call);

  return Found;
}

/// Check whether the access index chain is valid. We check
/// here because there may be type casts between two
/// access indexes. We want to ensure memory access still valid.
bool BPFAbstractMemberAccess::IsValidAIChain(const MDNode *ParentType,
                                             uint32_t ParentAI,
                                             const MDNode *ChildType) {
  if (!ChildType)
    return true; // preserve_field_info, no type comparison needed.

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

void BPFAbstractMemberAccess::collectAICallChains(Function &F) {
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

/// Get the start and the end of storage offset for \p MemberTy.
void BPFAbstractMemberAccess::GetStorageBitRange(DIDerivedType *MemberTy,
                                                 Align RecordAlignment,
                                                 uint32_t &StartBitOffset,
                                                 uint32_t &EndBitOffset) {
  uint32_t MemberBitSize = MemberTy->getSizeInBits();
  uint32_t MemberBitOffset = MemberTy->getOffsetInBits();

  if (RecordAlignment > 8) {
    // If the Bits are within an aligned 8-byte, set the RecordAlignment
    // to 8, other report the fatal error.
    if (MemberBitOffset / 64 != (MemberBitOffset + MemberBitSize) / 64)
      report_fatal_error("Unsupported field expression for llvm.bpf.preserve.field.info, "
                         "requiring too big alignment");
    RecordAlignment = Align(8);
  }

  uint32_t AlignBits = RecordAlignment.value() * 8;
  if (MemberBitSize > AlignBits)
    report_fatal_error("Unsupported field expression for llvm.bpf.preserve.field.info, "
                       "bitfield size greater than record alignment");

  StartBitOffset = MemberBitOffset & ~(AlignBits - 1);
  if ((StartBitOffset + AlignBits) < (MemberBitOffset + MemberBitSize))
    report_fatal_error("Unsupported field expression for llvm.bpf.preserve.field.info, "
                       "cross alignment boundary");
  EndBitOffset = StartBitOffset + AlignBits;
}

uint32_t BPFAbstractMemberAccess::GetFieldInfo(uint32_t InfoKind,
                                               DICompositeType *CTy,
                                               uint32_t AccessIndex,
                                               uint32_t PatchImm,
                                               MaybeAlign RecordAlignment) {
  if (InfoKind == BTF::FIELD_EXISTENCE)
    return 1;

  uint32_t Tag = CTy->getTag();
  if (InfoKind == BTF::FIELD_BYTE_OFFSET) {
    if (Tag == dwarf::DW_TAG_array_type) {
      auto *EltTy = stripQualifiers(CTy->getBaseType());
      PatchImm += AccessIndex * calcArraySize(CTy, 1) *
                  (EltTy->getSizeInBits() >> 3);
    } else if (Tag == dwarf::DW_TAG_structure_type) {
      auto *MemberTy = cast<DIDerivedType>(CTy->getElements()[AccessIndex]);
      if (!MemberTy->isBitField()) {
        PatchImm += MemberTy->getOffsetInBits() >> 3;
      } else {
        unsigned SBitOffset, NextSBitOffset;
        GetStorageBitRange(MemberTy, *RecordAlignment, SBitOffset,
                           NextSBitOffset);
        PatchImm += SBitOffset >> 3;
      }
    }
    return PatchImm;
  }

  if (InfoKind == BTF::FIELD_BYTE_SIZE) {
    if (Tag == dwarf::DW_TAG_array_type) {
      auto *EltTy = stripQualifiers(CTy->getBaseType());
      return calcArraySize(CTy, 1) * (EltTy->getSizeInBits() >> 3);
    } else {
      auto *MemberTy = cast<DIDerivedType>(CTy->getElements()[AccessIndex]);
      uint32_t SizeInBits = MemberTy->getSizeInBits();
      if (!MemberTy->isBitField())
        return SizeInBits >> 3;

      unsigned SBitOffset, NextSBitOffset;
      GetStorageBitRange(MemberTy, *RecordAlignment, SBitOffset,
                         NextSBitOffset);
      SizeInBits = NextSBitOffset - SBitOffset;
      if (SizeInBits & (SizeInBits - 1))
        report_fatal_error("Unsupported field expression for llvm.bpf.preserve.field.info");
      return SizeInBits >> 3;
    }
  }

  if (InfoKind == BTF::FIELD_SIGNEDNESS) {
    const DIType *BaseTy;
    if (Tag == dwarf::DW_TAG_array_type) {
      // Signedness only checked when final array elements are accessed.
      if (CTy->getElements().size() != 1)
        report_fatal_error("Invalid array expression for llvm.bpf.preserve.field.info");
      BaseTy = stripQualifiers(CTy->getBaseType());
    } else {
      auto *MemberTy = cast<DIDerivedType>(CTy->getElements()[AccessIndex]);
      BaseTy = stripQualifiers(MemberTy->getBaseType());
    }

    // Only basic types and enum types have signedness.
    const auto *BTy = dyn_cast<DIBasicType>(BaseTy);
    while (!BTy) {
      const auto *CompTy = dyn_cast<DICompositeType>(BaseTy);
      // Report an error if the field expression does not have signedness.
      if (!CompTy || CompTy->getTag() != dwarf::DW_TAG_enumeration_type)
        report_fatal_error("Invalid field expression for llvm.bpf.preserve.field.info");
      BaseTy = stripQualifiers(CompTy->getBaseType());
      BTy = dyn_cast<DIBasicType>(BaseTy);
    }
    uint32_t Encoding = BTy->getEncoding();
    return (Encoding == dwarf::DW_ATE_signed || Encoding == dwarf::DW_ATE_signed_char);
  }

  if (InfoKind == BTF::FIELD_LSHIFT_U64) {
    // The value is loaded into a value with FIELD_BYTE_SIZE size,
    // and then zero or sign extended to U64.
    // FIELD_LSHIFT_U64 and FIELD_RSHIFT_U64 are operations
    // to extract the original value.
    const Triple &Triple = TM->getTargetTriple();
    DIDerivedType *MemberTy = nullptr;
    bool IsBitField = false;
    uint32_t SizeInBits;

    if (Tag == dwarf::DW_TAG_array_type) {
      auto *EltTy = stripQualifiers(CTy->getBaseType());
      SizeInBits = calcArraySize(CTy, 1) * EltTy->getSizeInBits();
    } else {
      MemberTy = cast<DIDerivedType>(CTy->getElements()[AccessIndex]);
      SizeInBits = MemberTy->getSizeInBits();
      IsBitField = MemberTy->isBitField();
    }

    if (!IsBitField) {
      if (SizeInBits > 64)
        report_fatal_error("too big field size for llvm.bpf.preserve.field.info");
      return 64 - SizeInBits;
    }

    unsigned SBitOffset, NextSBitOffset;
    GetStorageBitRange(MemberTy, *RecordAlignment, SBitOffset, NextSBitOffset);
    if (NextSBitOffset - SBitOffset > 64)
      report_fatal_error("too big field size for llvm.bpf.preserve.field.info");

    unsigned OffsetInBits = MemberTy->getOffsetInBits();
    if (Triple.getArch() == Triple::bpfel)
      return SBitOffset + 64 - OffsetInBits - SizeInBits;
    else
      return OffsetInBits + 64 - NextSBitOffset;
  }

  if (InfoKind == BTF::FIELD_RSHIFT_U64) {
    DIDerivedType *MemberTy = nullptr;
    bool IsBitField = false;
    uint32_t SizeInBits;
    if (Tag == dwarf::DW_TAG_array_type) {
      auto *EltTy = stripQualifiers(CTy->getBaseType());
      SizeInBits = calcArraySize(CTy, 1) * EltTy->getSizeInBits();
    } else {
      MemberTy = cast<DIDerivedType>(CTy->getElements()[AccessIndex]);
      SizeInBits = MemberTy->getSizeInBits();
      IsBitField = MemberTy->isBitField();
    }

    if (!IsBitField) {
      if (SizeInBits > 64)
        report_fatal_error("too big field size for llvm.bpf.preserve.field.info");
      return 64 - SizeInBits;
    }

    unsigned SBitOffset, NextSBitOffset;
    GetStorageBitRange(MemberTy, *RecordAlignment, SBitOffset, NextSBitOffset);
    if (NextSBitOffset - SBitOffset > 64)
      report_fatal_error("too big field size for llvm.bpf.preserve.field.info");

    return 64 - SizeInBits;
  }

  llvm_unreachable("Unknown llvm.bpf.preserve.field.info info kind");
}

/// Consume leading array accesses, leaving the first record access on the
/// stack.
std::optional<BPFAbstractMemberAccess::AccessRoot>
BPFAbstractMemberAccess::consumeAccessRoot(
    SmallVectorImpl<CallInfo> &CallStack) {
  uint32_t FirstIndex = 0;
  for (;;) {
    const CallInfo &CInfo = CallStack.back();
    auto *Metadata = cast<DIType>(CInfo.Metadata);
    switch (CInfo.Kind) {
    case BPFPreserveStructAI:
    case BPFPreserveUnionAI:
      // A typedef may be the only name the loader can match for this record.
      return AccessRoot{stripQualifiers(Metadata, /*skipTypedef=*/false),
                        FirstIndex};
    case BPFPreserveArrayAI:
      break;
    default:
      llvm_unreachable("Expected an array, struct, or union access");
    }

    const uint32_t Index = CInfo.AccessIndex;
    CallStack.pop_back();
    auto *Ty = stripQualifiers(Metadata);
    DIType *ElementTy;
    switch (Ty->getTag()) {
    case dwarf::DW_TAG_array_type: {
      auto *Array = cast<DICompositeType>(Ty);
      FirstIndex += Index * calcArraySize(Array, 1);
      if (Array->getElements().size() != 1 && !CallStack.empty())
        continue;
      ElementTy = stripQualifiers(Array->getBaseType());
      break;
    }
    case dwarf::DW_TAG_pointer_type: {
      auto *Pointee = stripQualifiers(cast<DIDerivedType>(Ty)->getBaseType());
      if (Pointee->getTag() != dwarf::DW_TAG_array_type) {
        FirstIndex += Index;
        ElementTy = Pointee;
        break;
      }
      auto *Array = cast<DICompositeType>(Pointee);
      FirstIndex += Index * calcArraySize(Array, 0);
      if (!CallStack.empty())
        continue;
      ElementTy = stripQualifiers(Array->getBaseType());
      break;
    }
    default:
      llvm_unreachable("Expected an array or pointer type");
    }

    // A chain ending at a subarray already has its index in units of the
    // innermost element, which can supply the relocation root.
    switch (ElementTy->getTag()) {
    case dwarf::DW_TAG_structure_type:
    case dwarf::DW_TAG_union_type:
      return AccessRoot{ElementTy, FirstIndex};
    default:
      return std::nullopt;
    }
  }
}

static std::string formatAmaSuffix(uint32_t Kind, int64_t PatchImm,
                                   const Twine &AccessString) {
  return (":" + Twine(Kind) + ":" + Twine(PatchImm) + "$" + AccessString).str();
}

/// Compute the base pointer, root type and relocation suffix of a field access.
Value *BPFAbstractMemberAccess::computeFieldAccess(CallInst *Call,
                                                   const CallInfo &Terminal,
                                                   std::string &RelocSuffix,
                                                   DIType *&RootType) {
  // Put the access chain into a stack with the top as the head of the chain.
  SmallVector<CallInfo, 4> CallStack{Terminal};
  for (auto It = AIChain.find(Call); It != AIChain.end();
       It = AIChain.find(It->second.first))
    CallStack.push_back(It->second.second);

  Value *const Base = CallStack.back().Base;
  const auto Root = consumeAccessRoot(CallStack);
  const bool IsFieldInfo = Terminal.Kind == BPFPreserveFieldInfoAI;
  if (!Root) {
    if (IsFieldInfo)
      report_fatal_error(
          "Invalid field access for llvm.preserve.field.info intrinsic");
    // With no record root, accesses such as &p[4] (int *p) or &a[2][3]
    // (int a[10][20]) lower to ordinary GEPs.
    return nullptr;
  }
  assert(!Root->Type->getName().empty());
  const uint32_t InfoKind =
      IsFieldInfo ? Terminal.AccessIndex : BTF::FIELD_BYTE_OFFSET;
  uint32_t PatchImm =
      Root->Index * (stripQualifiers(Root->Type)->getSizeInBits() >> 3);
  std::string AccessString = std::to_string(Root->Index);

  // Build the access string and compute the local byte offset or requested
  // field property while each access still has its debug type.
  ArrayRef<CallInfo> Fields = CallStack;
  if (IsFieldInfo)
    Fields = Fields.drop_front();
  while (!Fields.empty()) {
    const CallInfo &CInfo = Fields.back();
    Fields = Fields.drop_back();

    // Access Index
    uint64_t AccessIndex = CInfo.AccessIndex;
    MDNode *MDN = CInfo.Metadata;
    // At this stage, it cannot be pointer type.
    auto *CTy = cast<DICompositeType>(stripQualifiers(cast<DIType>(MDN)));

    uint64_t BTFIndex = AccessIndex;
    if (CTy->getTag() == dwarf::DW_TAG_structure_type) {
      DINodeArray Elements = CTy->getElements();
      uint64_t Offset = getBTFRecordElementOffset(Elements[AccessIndex]);
      // Find this element's position in the stable offset order without
      // sorting the whole record for every CO-RE access.
      BTFIndex = 0;
      for (unsigned I = 0; I < Elements.size(); ++I) {
        uint64_t ElementOffset = getBTFRecordElementOffset(Elements[I]);
        if (ElementOffset < Offset ||
            (ElementOffset == Offset && I < AccessIndex))
          ++BTFIndex;
      }
    }
    AccessString += ":" + std::to_string(BTFIndex);

    // Earlier accesses contribute byte offsets. Only the final field supplies
    // the property requested by preserve_field_info.
    const uint32_t FieldKind =
        Fields.empty() ? InfoKind : BTF::FIELD_BYTE_OFFSET;
    PatchImm = GetFieldInfo(FieldKind, CTy, AccessIndex, PatchImm,
                            CInfo.RecordAlignment);
  }

  if (InfoKind == BTF::FIELD_EXISTENCE)
    PatchImm = 1;

  RootType = Root->Type;
  RelocSuffix = formatAmaSuffix(InfoKind, PatchImm, AccessString);

  return Base;
}

BPFAbstractMemberAccess::EnumAccess
BPFAbstractMemberAccess::computeEnumAccess(const CallInst *Call,
                                           const DIType *Ty) {
  // The argument can be a global or a GEP into a global.
  const auto *GV =
      cast<GlobalVariable>(Call->getArgOperand(1)->stripPointerCasts());
  assert(GV->hasInitializer());
  const auto *DA = cast<ConstantDataArray>(GV->getInitializer());
  assert(DA->isString());
  StringRef ValueStr = DA->getAsString();
  size_t Separator = ValueStr.find(':');
  StringRef Enumerator = ValueStr.substr(0, Separator);
  StringRef Value = ValueStr.substr(Separator + 1);

  const auto *CTy = cast<DICompositeType>(stripQualifiers(Ty));
  assert(CTy->getTag() == dwarf::DW_TAG_enumeration_type);
  for (auto [Index, Element] : enumerate(CTy->getElements()))
    if (cast<DIEnumerator>(Element)->getName() == Enumerator)
      return {Index, Value};
  return {0, Value};
}

DIType *BPFAbstractMemberAccess::computeTypeAccess(CallInst *Call,
                                                   const CallInfo &CInfo,
                                                   std::string &RelocSuffix) {
  DIType *Ty = stripQualifiers(cast<DIType>(CInfo.Metadata), false);
  assert(!Ty->getName().empty());

  int64_t PatchImm = 1;
  size_t AccessIndex = 0;
  switch (CInfo.AccessIndex) {
  case BTF::TYPE_EXISTENCE:
  case BTF::TYPE_MATCH:
    break;
  case BTF::TYPE_SIZE:
    // typedef debuginfo type has size 0, get the eventual base type.
    PatchImm = stripQualifiers(Ty)->getSizeInBits() / 8;
    break;
  case BTF::ENUM_VALUE_EXISTENCE:
    AccessIndex = computeEnumAccess(Call, Ty).Index;
    break;
  case BTF::ENUM_VALUE: {
    auto [Index, Value] = computeEnumAccess(Call, Ty);
    PatchImm = std::stoll(Value.str());
    AccessIndex = Index;
    break;
  }
  default:
    llvm_unreachable("Unexpected BPF type relocation kind");
  }

  RelocSuffix =
      formatAmaSuffix(CInfo.AccessIndex, PatchImm, Twine(AccessIndex));

  return Ty;
}

static GlobalVariable *getOrCreateAmaGlobal(Module &M, DIType *Root,
                                            IntegerType *ValueType,
                                            StringRef RelocSuffix) {
  const std::string Prefix = ("llvm." + Root->getName()).str();
  std::string Name = (Twine(Prefix) + RelocSuffix).str();
  for (unsigned N = 1; GlobalValue *Existing = M.getNamedValue(Name); ++N) {
    auto *GV = dyn_cast<GlobalVariable>(Existing);
    if (GV && GV->isDeclaration() && GV->hasExternalLinkage() &&
        GV->hasAttribute(BPFCoreSharedInfo::AmaAttr) &&
        GV->getValueType() == ValueType &&
        GV->getMetadata(LLVMContext::MD_preserve_access_index) == Root)
      return GV;
    // Same-named roots and different load widths need distinct globals. Keep
    // the discriminator before the suffix that the backend will decode.
    Name = (Twine(Prefix) + "." + Twine(N) + RelocSuffix).str();
  }

  auto *GV = new GlobalVariable(M, ValueType, false,
                                GlobalValue::ExternalLinkage, nullptr, Name);
  GV->addAttribute(BPFCoreSharedInfo::AmaAttr);
  GV->setMetadata(LLVMContext::MD_preserve_access_index, Root);
  return GV;
}

/// Call/Kind is the base preserve_*_access_index() call. Attempts to do
/// transformation to a chain of relocable GEPs.
bool BPFAbstractMemberAccess::transformGEPChain(CallInst *Call,
                                                const CallInfo &CInfo) {
  std::string RelocSuffix;
  DIType *Root;
  Value *Base = nullptr;

  const bool IsFieldInfo = CInfo.Kind == BPFPreserveFieldInfoAI;
  if (IsFieldInfo && CInfo.Metadata) {
    Root = computeTypeAccess(Call, CInfo, RelocSuffix);
  } else {
    Base = computeFieldAccess(Call, CInfo, RelocSuffix, Root);
    if (!Base)
      return false;
  }

  BasicBlock *BB = Call->getParent();
  auto *ValueType = IsFieldInfo ? cast<IntegerType>(Call->getType())
                                : Type::getInt64Ty(Call->getContext());
  auto *GV = getOrCreateAmaGlobal(*M, Root, ValueType, RelocSuffix);
  Instruction *Input = new LoadInst(ValueType, GV, "", Call->getIterator());
  if (!IsFieldInfo) {
    auto *GEP = GetElementPtrInst::Create(Type::getInt8Ty(Call->getContext()),
                                          Base, Input);
    GEP->insertBefore(Call->getIterator());
    Input = GEP;
  }

  // For the following code,
  //    Block0:
  //      ...
  //      if (...) goto Block1 else ...
  //    Block1:
  //      %6 = load llvm.sk_buff:0:50$0:0:0:2:0
  //      %8 = getelementptr i8, i8* %2, %6
  //      ...
  //      goto CommonExit
  //    Block2:
  //      ...
  //      if (...) goto Block3 else ...
  //    Block3:
  //      %6 = load llvm.bpf_map:0:40$0:0:0:2:0
  //      %8 = getelementptr i8, i8* %2, %6
  //      ...
  //      goto CommonExit
  //    CommonExit
  // SimplifyCFG may generate:
  //    Block0:
  //      ...
  //      if (...) goto Block_Common else ...
  //     Block2:
  //       ...
  //      if (...) goto Block_Common else ...
  //    Block_Common:
  //      PHI = [llvm.sk_buff:0:50$0:0:0:2:0, llvm.bpf_map:0:40$0:0:0:2:0]
  //      %6 = load PHI
  //      %8 = getelementptr i8, i8* %2, %6
  //      ...
  //      goto CommonExit
  //  For the above code, we cannot perform proper relocation since
  //  "load PHI" has two possible relocations.
  //
  // To prevent above tail merging, we use __builtin_bpf_passthrough()
  // where one of its parameters is a seq_num. Since two
  // __builtin_bpf_passthrough() funcs will always have different seq_num,
  // tail merging cannot happen. The __builtin_bpf_passthrough() will be
  // removed in the beginning of Target IR passes.
  //
  // This approach is also used in other places when global var
  // representing a relocation is used.
  Instruction *PassThroughInst =
      BPFCoreSharedInfo::insertPassThrough(M, BB, Input, Call);
  Call->replaceAllUsesWith(PassThroughInst);
  Call->eraseFromParent();

  return true;
}

bool BPFAbstractMemberAccess::doTransformation(Function &F) {
  bool Transformed = false;

  // Collect PreserveDIAccessIndex Intrinsic call chains.
  // The call chains will be used to generate the access
  // patterns similar to GEP.
  collectAICallChains(F);

  for (auto &C : BaseAICalls)
    Transformed = transformGEPChain(C.first, C.second) || Transformed;

  return removePreserveAccessIndexIntrinsic(F) || Transformed;
}

PreservedAnalyses
BPFAbstractMemberAccessPass::run(Function &F, FunctionAnalysisManager &AM) {
  return BPFAbstractMemberAccess(TM).run(F) ? PreservedAnalyses::none()
                                            : PreservedAnalyses::all();
}
