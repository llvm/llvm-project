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
constexpr StringRef BPFCoreSharedInfo::AmaAttr;
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
  TargetMachine *TM;
  // Add optional BPFTargetMachine parameter so that BPF backend can add the phase
  // with target machine to find out the endianness. The default constructor (without
  // parameters) is used by the pass manager for managing purposes.
  BPFAbstractMemberAccess(BPFTargetMachine *TM = nullptr) : ModulePass(ID), TM(TM) {}

  struct CallInfo {
    uint32_t Kind;
    uint32_t AccessIndex;
    Align RecordAlignment;
    MDNode *Metadata;
    Value *Base;
  };
  typedef std::stack<std::pair<CallInst *, CallInfo>> CallInfoStack;

private:
  enum : uint32_t {
    BPFPreserveArrayAI = 1,
    BPFPreserveUnionAI = 2,
    BPFPreserveStructAI = 3,
    BPFPreserveFieldInfoAI = 4,
  };

  const DataLayout *DL = nullptr;

  std::map<std::string, GlobalVariable *> GEPGlobals;
  // A map to link preserve_*_access_index instrinsic calls.
  std::map<CallInst *, std::pair<CallInst *, CallInfo>> AIChain;
  // A map to hold all the base preserve_*_access_index instrinsic calls.
  // The base call is not an input of any other preserve_*
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
  bool HasPreserveFieldInfoCall(CallInfoStack &CallStack);
  void GetStorageBitRange(DIDerivedType *MemberTy, Align RecordAlignment,
                          uint32_t &StartBitOffset, uint32_t &EndBitOffset);
  uint32_t GetFieldInfo(uint32_t InfoKind, DICompositeType *CTy,
                        uint32_t AccessIndex, uint32_t PatchImm,
                        Align RecordAlignment);

  Value *computeBaseAndAccessKey(CallInst *Call, CallInfo &CInfo,
                                 std::string &AccessKey, MDNode *&BaseMeta);
  uint64_t getConstant(const Value *IndexValue);
  bool transformGEPChain(Module &M, CallInst *Call, CallInfo &CInfo);
};
} // End anonymous namespace

char BPFAbstractMemberAccess::ID = 0;
INITIALIZE_PASS(BPFAbstractMemberAccess, DEBUG_TYPE,
                "abstracting struct/union member accessees", false, false)

ModulePass *llvm::createBPFAbstractMemberAccess(BPFTargetMachine *TM) {
  return new BPFAbstractMemberAccess(TM);
}

bool BPFAbstractMemberAccess::runOnModule(Module &M) {
  LLVM_DEBUG(dbgs() << "********** Abstract Member Accesses **********\n");

  // Bail out if no debug info.
  if (M.debug_compile_units().empty())
    return false;

  DL = &M.getDataLayout();
  return doTransformation(M);
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

  const auto *GV = dyn_cast<GlobalValue>(Call->getCalledOperand());
  if (!GV)
    return false;
  if (GV->getName().startswith("llvm.preserve.array.access.index")) {
    CInfo.Kind = BPFPreserveArrayAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.array.access.index intrinsic");
    CInfo.AccessIndex = getConstant(Call->getArgOperand(2));
    CInfo.Base = Call->getArgOperand(0);
    CInfo.RecordAlignment =
        DL->getABITypeAlign(CInfo.Base->getType()->getPointerElementType());
    return true;
  }
  if (GV->getName().startswith("llvm.preserve.union.access.index")) {
    CInfo.Kind = BPFPreserveUnionAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.union.access.index intrinsic");
    CInfo.AccessIndex = getConstant(Call->getArgOperand(1));
    CInfo.Base = Call->getArgOperand(0);
    CInfo.RecordAlignment =
        DL->getABITypeAlign(CInfo.Base->getType()->getPointerElementType());
    return true;
  }
  if (GV->getName().startswith("llvm.preserve.struct.access.index")) {
    CInfo.Kind = BPFPreserveStructAI;
    CInfo.Metadata = Call->getMetadata(LLVMContext::MD_preserve_access_index);
    if (!CInfo.Metadata)
      report_fatal_error("Missing metadata for llvm.preserve.struct.access.index intrinsic");
    CInfo.AccessIndex = getConstant(Call->getArgOperand(2));
    CInfo.Base = Call->getArgOperand(0);
    CInfo.RecordAlignment =
        DL->getABITypeAlign(CInfo.Base->getType()->getPointerElementType());
    return true;
  }
  if (GV->getName().startswith("llvm.bpf.preserve.field.info")) {
    CInfo.Kind = BPFPreserveFieldInfoAI;
    CInfo.Metadata = nullptr;
    // Check validity of info_kind as clang did not check this.
    uint64_t InfoKind = getConstant(Call->getArgOperand(1));
    if (InfoKind >= BPFCoreSharedInfo::MAX_FIELD_RELOC_KIND)
      report_fatal_error("Incorrect info_kind for llvm.bpf.preserve.field.info intrinsic");
    CInfo.AccessIndex = InfoKind;
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

/// Get the start and the end of storage offset for \p MemberTy.
void BPFAbstractMemberAccess::GetStorageBitRange(DIDerivedType *MemberTy,
                                                 Align RecordAlignment,
                                                 uint32_t &StartBitOffset,
                                                 uint32_t &EndBitOffset) {
  uint32_t MemberBitSize = MemberTy->getSizeInBits();
  uint32_t MemberBitOffset = MemberTy->getOffsetInBits();
  uint32_t AlignBits = RecordAlignment.value() * 8;
  if (RecordAlignment > 8 || MemberBitSize > AlignBits)
    report_fatal_error("Unsupported field expression for llvm.bpf.preserve.field.info, "
                       "requiring too big alignment");

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
                                               Align RecordAlignment) {
  if (InfoKind == BPFCoreSharedInfo::FIELD_EXISTENCE)
      return 1;

  uint32_t Tag = CTy->getTag();
  if (InfoKind == BPFCoreSharedInfo::FIELD_BYTE_OFFSET) {
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
        GetStorageBitRange(MemberTy, RecordAlignment, SBitOffset,
                           NextSBitOffset);
        PatchImm += SBitOffset >> 3;
      }
    }
    return PatchImm;
  }

  if (InfoKind == BPFCoreSharedInfo::FIELD_BYTE_SIZE) {
    if (Tag == dwarf::DW_TAG_array_type) {
      auto *EltTy = stripQualifiers(CTy->getBaseType());
      return calcArraySize(CTy, 1) * (EltTy->getSizeInBits() >> 3);
    } else {
      auto *MemberTy = cast<DIDerivedType>(CTy->getElements()[AccessIndex]);
      uint32_t SizeInBits = MemberTy->getSizeInBits();
      if (!MemberTy->isBitField())
        return SizeInBits >> 3;

      unsigned SBitOffset, NextSBitOffset;
      GetStorageBitRange(MemberTy, RecordAlignment, SBitOffset, NextSBitOffset);
      SizeInBits = NextSBitOffset - SBitOffset;
      if (SizeInBits & (SizeInBits - 1))
        report_fatal_error("Unsupported field expression for llvm.bpf.preserve.field.info");
      return SizeInBits >> 3;
    }
  }

  if (InfoKind == BPFCoreSharedInfo::FIELD_SIGNEDNESS) {
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

  if (InfoKind == BPFCoreSharedInfo::FIELD_LSHIFT_U64) {
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
    GetStorageBitRange(MemberTy, RecordAlignment, SBitOffset, NextSBitOffset);
    if (NextSBitOffset - SBitOffset > 64)
      report_fatal_error("too big field size for llvm.bpf.preserve.field.info");

    unsigned OffsetInBits = MemberTy->getOffsetInBits();
    if (Triple.getArch() == Triple::bpfel)
      return SBitOffset + 64 - OffsetInBits - SizeInBits;
    else
      return OffsetInBits + 64 - NextSBitOffset;
  }

  if (InfoKind == BPFCoreSharedInfo::FIELD_RSHIFT_U64) {
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
    GetStorageBitRange(MemberTy, RecordAlignment, SBitOffset, NextSBitOffset);
    if (NextSBitOffset - SBitOffset > 64)
      report_fatal_error("too big field size for llvm.bpf.preserve.field.info");

    return 64 - SizeInBits;
  }

  llvm_unreachable("Unknown llvm.bpf.preserve.field.info info kind");
}

bool BPFAbstractMemberAccess::HasPreserveFieldInfoCall(CallInfoStack &CallStack) {
  // This is called in error return path, no need to maintain CallStack.
  while (CallStack.size()) {
    auto StackElem = CallStack.top();
    if (StackElem.second.Kind == BPFPreserveFieldInfoAI)
      return true;
    CallStack.pop();
  }
  return false;
}

/// Compute the base of the whole preserve_* intrinsics chains, i.e., the base
/// pointer of the first preserve_*_access_index call, and construct the access
/// string, which will be the name of a global variable.
Value *BPFAbstractMemberAccess::computeBaseAndAccessKey(CallInst *Call,
                                                        CallInfo &CInfo,
                                                        std::string &AccessKey,
                                                        MDNode *&TypeMeta) {
  Value *Base = nullptr;
  std::string TypeName;
  CallInfoStack CallStack;

  // Put the access chain into a stack with the top as the head of the chain.
  while (Call) {
    CallStack.push(std::make_pair(Call, CInfo));
    CInfo = AIChain[Call].second;
    Call = AIChain[Call].first;
  }

  // The access offset from the base of the head of chain is also
  // calculated here as all debuginfo types are available.

  // Get type name and calculate the first index.
  // We only want to get type name from typedef, structure or union.
  // If user wants a relocation like
  //    int *p; ... __builtin_preserve_access_index(&p[4]) ...
  // or
  //    int a[10][20]; ... __builtin_preserve_access_index(&a[2][3]) ...
  // we will skip them.
  uint32_t FirstIndex = 0;
  uint32_t PatchImm = 0; // AccessOffset or the requested field info
  uint32_t InfoKind = BPFCoreSharedInfo::FIELD_BYTE_OFFSET;
  while (CallStack.size()) {
    auto StackElem = CallStack.top();
    Call = StackElem.first;
    CInfo = StackElem.second;

    if (!Base)
      Base = CInfo.Base;

    DIType *PossibleTypeDef = stripQualifiers(cast<DIType>(CInfo.Metadata),
                                              false);
    DIType *Ty = stripQualifiers(PossibleTypeDef);
    if (CInfo.Kind == BPFPreserveUnionAI ||
        CInfo.Kind == BPFPreserveStructAI) {
      // struct or union type. If the typedef is in the metadata, always
      // use the typedef.
      TypeName = std::string(PossibleTypeDef->getName());
      TypeMeta = PossibleTypeDef;
      PatchImm += FirstIndex * (Ty->getSizeInBits() >> 3);
      break;
    }

    assert(CInfo.Kind == BPFPreserveArrayAI);

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
      if (!CTy) {
        if (HasPreserveFieldInfoCall(CallStack))
          report_fatal_error("Invalid field access for llvm.preserve.field.info intrinsic");
        return nullptr;
      }

      unsigned CTag = CTy->getTag();
      if (CTag == dwarf::DW_TAG_structure_type || CTag == dwarf::DW_TAG_union_type) {
        TypeName = std::string(CTy->getName());
      } else {
        if (HasPreserveFieldInfoCall(CallStack))
          report_fatal_error("Invalid field access for llvm.preserve.field.info intrinsic");
        return nullptr;
      }
      TypeMeta = CTy;
      PatchImm += FirstIndex * (CTy->getSizeInBits() >> 3);
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

    if (CInfo.Kind == BPFPreserveFieldInfoAI) {
      InfoKind = CInfo.AccessIndex;
      break;
    }

    // If the next Call (the top of the stack) is a BPFPreserveFieldInfoAI,
    // the action will be extracting field info.
    if (CallStack.size()) {
      auto StackElem2 = CallStack.top();
      CallInfo CInfo2 = StackElem2.second;
      if (CInfo2.Kind == BPFPreserveFieldInfoAI) {
        InfoKind = CInfo2.AccessIndex;
        assert(CallStack.size() == 1);
      }
    }

    // Access Index
    uint64_t AccessIndex = CInfo.AccessIndex;
    AccessKey += ":" + std::to_string(AccessIndex);

    MDNode *MDN = CInfo.Metadata;
    // At this stage, it cannot be pointer type.
    auto *CTy = cast<DICompositeType>(stripQualifiers(cast<DIType>(MDN)));
    PatchImm = GetFieldInfo(InfoKind, CTy, AccessIndex, PatchImm,
                            CInfo.RecordAlignment);
  }

  // Access key is the
  //   "llvm." + type name + ":" + reloc type + ":" + patched imm + "$" +
  //   access string,
  // uniquely identifying one relocation.
  // The prefix "llvm." indicates this is a temporary global, which should
  // not be emitted to ELF file.
  AccessKey = "llvm." + TypeName + ":" + std::to_string(InfoKind) + ":" +
              std::to_string(PatchImm) + "$" + AccessKey;

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

  BasicBlock *BB = Call->getParent();
  GlobalVariable *GV;

  if (GEPGlobals.find(AccessKey) == GEPGlobals.end()) {
    IntegerType *VarType;
    if (CInfo.Kind == BPFPreserveFieldInfoAI)
      VarType = Type::getInt32Ty(BB->getContext()); // 32bit return value
    else
      VarType = Type::getInt64Ty(BB->getContext()); // 64bit ptr arith

    GV = new GlobalVariable(M, VarType, false, GlobalVariable::ExternalLinkage,
                            NULL, AccessKey);
    GV->addAttribute(BPFCoreSharedInfo::AmaAttr);
    GV->setMetadata(LLVMContext::MD_preserve_access_index, TypeMeta);
    GEPGlobals[AccessKey] = GV;
  } else {
    GV = GEPGlobals[AccessKey];
  }

  if (CInfo.Kind == BPFPreserveFieldInfoAI) {
    // Load the global variable which represents the returned field info.
    auto *LDInst = new LoadInst(Type::getInt32Ty(BB->getContext()), GV, "",
                                Call);
    Call->replaceAllUsesWith(LDInst);
    Call->eraseFromParent();
    return true;
  }

  // For any original GEP Call and Base %2 like
  //   %4 = bitcast %struct.net_device** %dev1 to i64*
  // it is transformed to:
  //   %6 = load sk_buff:50:$0:0:0:2:0
  //   %7 = bitcast %struct.sk_buff* %2 to i8*
  //   %8 = getelementptr i8, i8* %7, %6
  //   %9 = bitcast i8* %8 to i64*
  //   using %9 instead of %4
  // The original Call inst is removed.

  // Load the global variable.
  auto *LDInst = new LoadInst(Type::getInt64Ty(BB->getContext()), GV, "", Call);

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
