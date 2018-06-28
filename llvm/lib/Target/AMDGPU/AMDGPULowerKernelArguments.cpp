//===-- AMDGPULowerKernelArguments.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This pass replaces accesses to kernel arguments with loads from
/// offsets from the kernarg base pointer.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "amdgpu-lower-kernel-arguments"

using namespace llvm;

namespace {

class AMDGPULowerKernelArguments : public FunctionPass{
public:
  static char ID;

  AMDGPULowerKernelArguments() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesAll();
 }
};

} // end anonymous namespace

bool AMDGPULowerKernelArguments::runOnFunction(Function &F) {
  CallingConv::ID CC = F.getCallingConv();
  if (CC != CallingConv::AMDGPU_KERNEL || F.arg_empty())
    return false;

  auto &TPC = getAnalysis<TargetPassConfig>();

  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  const SISubtarget &ST = TM.getSubtarget<SISubtarget>(F);
  LLVMContext &Ctx = F.getParent()->getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  BasicBlock &EntryBlock = *F.begin();
  IRBuilder<> Builder(&*EntryBlock.begin());

  SmallVector<Type *, 16> ArgTypes;
  for (Argument &Arg : F.args()) {
    Type *ArgTy = Arg.getType();
    unsigned Size = DL.getTypeStoreSizeInBits(ArgTy);
    bool IsExtArg = Size < 32 && (Arg.hasZExtAttr() || Arg.hasSExtAttr()) &&
                    !ST.isAmdHsaOS();

    // Clover seems to always pad i8/i16 to i32, but doesn't properly align
    // them?
    // Make sure the struct elements have correct size and alignment for ext
    // args. These seem to be padded up to 4-bytes but not correctly aligned.
    ArgTypes.push_back(
      IsExtArg ? ArrayType::get(ArgTy, 32 / Size) : Arg.getType());
  }

  StructType *ArgStructTy = StructType::create(Ctx, ArgTypes, F.getName());
  const StructLayout *Layout = DL.getStructLayout(ArgStructTy);

  // Minimum alignment for kern segment is 16.
  unsigned KernArgBaseAlign = std::max(16u, DL.getABITypeAlignment(ArgStructTy));
  const uint64_t BaseOffset = ST.getExplicitKernelArgOffset(F);

  // FIXME: Alignment is broken broken with explicit arg offset.;
  const uint64_t TotalKernArgSize = BaseOffset +
    ST.getKernArgSegmentSize(F, DL.getTypeAllocSize(ArgStructTy));
  if (TotalKernArgSize == 0)
    return false;

  CallInst *KernArgSegment =
    Builder.CreateIntrinsic(Intrinsic::amdgcn_kernarg_segment_ptr, nullptr,
                            F.getName() + ".kernarg.segment");

  KernArgSegment->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
  KernArgSegment->addAttribute(AttributeList::ReturnIndex,
    Attribute::getWithDereferenceableBytes(Ctx, TotalKernArgSize));
  KernArgSegment->addAttribute(AttributeList::ReturnIndex,
    Attribute::getWithAlignment(Ctx, KernArgBaseAlign));

  Value *KernArgBase = KernArgSegment;
  if (BaseOffset != 0) {
    KernArgBase = Builder.CreateConstInBoundsGEP1_64(KernArgBase, BaseOffset);
    KernArgBaseAlign = MinAlign(KernArgBaseAlign, BaseOffset);
  }

  unsigned AS = KernArgSegment->getType()->getPointerAddressSpace();
  Value *CastStruct = Builder.CreateBitCast(KernArgBase,
                                            ArgStructTy->getPointerTo(AS));
  for (Argument &Arg : F.args()) {
    if (Arg.use_empty())
      continue;

    Type *ArgTy = Arg.getType();
    if (PointerType *PT = dyn_cast<PointerType>(ArgTy)) {
      // FIXME: Hack. We rely on AssertZext to be able to fold DS addressing
      // modes on SI to know the high bits are 0 so pointer adds don't wrap. We
      // can't represent this with range metadata because it's only allowed for
      // integer types.
      if (PT->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS &&
          ST.getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS)
        continue;

      // FIXME: We can replace this with equivalent alias.scope/noalias
      // metadata, but this appears to be a lot of work.
      if (Arg.hasNoAliasAttr())
        continue;
    }

    VectorType *VT = dyn_cast<VectorType>(ArgTy);
    bool IsV3 = VT && VT->getNumElements() == 3;
    VectorType *V4Ty = nullptr;

    unsigned Size = DL.getTypeSizeInBits(ArgTy);
    bool IsExtArg = Size < 32 && (Arg.hasZExtAttr() || Arg.hasSExtAttr()) &&
                    !ST.isAmdHsaOS();
    int64_t EltOffset = Layout->getElementOffset(Arg.getArgNo());
    int64_t AlignDownOffset = alignDown(EltOffset, 4);
    int64_t OffsetDiff = EltOffset - AlignDownOffset;
    unsigned AdjustedAlign = MinAlign(KernArgBaseAlign, AlignDownOffset);

    Value *ArgPtr;
    if (Size < 32 && !ArgTy->isAggregateType()) { // FIXME: Handle aggregate types
      // Since we don't have sub-dword scalar loads, avoid doing an extload by
      // loading earlier than the argument address, and extracting the relevant
      // bits.
      //
      // Additionally widen any sub-dword load to i32 even if suitably aligned,
      // so that CSE between different argument loads works easily.

      ArgPtr = Builder.CreateConstGEP1_64(KernArgBase, AlignDownOffset);
      ArgPtr = Builder.CreateBitCast(
        ArgPtr,
        Builder.getInt32Ty()->getPointerTo(AS),
        Arg.getName() + ".kernarg.offset.align.down");
    } else {
      ArgPtr = Builder.CreateStructGEP(CastStruct, Arg.getArgNo(),
                                       Arg.getName() + ".kernarg.offset");
    }

    assert((!IsExtArg || !IsV3) && "incompatible situation");


    if (IsV3 && Size >= 32) {
      V4Ty = VectorType::get(VT->getVectorElementType(), 4);
      // Use the hack that clang uses to avoid SelectionDAG ruining v3 loads
      ArgPtr = Builder.CreateBitCast(ArgPtr, V4Ty->getPointerTo(AS));
    }

    LoadInst *Load = Builder.CreateAlignedLoad(ArgPtr, AdjustedAlign);
    Load->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(Ctx, {}));

    MDBuilder MDB(Ctx);

    if (isa<PointerType>(ArgTy)) {
      if (Arg.hasNonNullAttr())
        Load->setMetadata(LLVMContext::MD_nonnull, MDNode::get(Ctx, {}));

      uint64_t DerefBytes = Arg.getDereferenceableBytes();
      if (DerefBytes != 0) {
        Load->setMetadata(
          LLVMContext::MD_dereferenceable,
          MDNode::get(Ctx,
                      MDB.createConstant(
                        ConstantInt::get(Builder.getInt64Ty(), DerefBytes))));
      }

      uint64_t DerefOrNullBytes = Arg.getDereferenceableOrNullBytes();
      if (DerefOrNullBytes != 0) {
        Load->setMetadata(
          LLVMContext::MD_dereferenceable_or_null,
          MDNode::get(Ctx,
                      MDB.createConstant(ConstantInt::get(Builder.getInt64Ty(),
                                                          DerefOrNullBytes))));
      }

      unsigned ParamAlign = Arg.getParamAlignment();
      if (ParamAlign != 0) {
        Load->setMetadata(
          LLVMContext::MD_align,
          MDNode::get(Ctx,
                      MDB.createConstant(ConstantInt::get(Builder.getInt64Ty(),
                                                          ParamAlign))));
      }
    }

    // TODO: Convert noalias arg to !noalias

    if (Size < 32 && !ArgTy->isAggregateType()) {
      if (IsExtArg && OffsetDiff == 0) {
        Type *I32Ty = Builder.getInt32Ty();
        bool IsSext = Arg.hasSExtAttr();
        Metadata *LowAndHigh[] = {
          ConstantAsMetadata::get(
            ConstantInt::get(I32Ty, IsSext ? minIntN(Size) : 0)),
          ConstantAsMetadata::get(
            ConstantInt::get(I32Ty,
                             IsSext ? maxIntN(Size) + 1 : maxUIntN(Size) + 1))
        };

        Load->setMetadata(LLVMContext::MD_range, MDNode::get(Ctx, LowAndHigh));
      }

      Value *ExtractBits = OffsetDiff == 0 ?
        Load : Builder.CreateLShr(Load, OffsetDiff * 8);

      IntegerType *ArgIntTy = Builder.getIntNTy(Size);
      Value *Trunc = Builder.CreateTrunc(ExtractBits, ArgIntTy);
      Value *NewVal = Builder.CreateBitCast(Trunc, ArgTy,
                                            Arg.getName() + ".load");
      Arg.replaceAllUsesWith(NewVal);
    } else if (IsV3) {
      Value *Shuf = Builder.CreateShuffleVector(Load, UndefValue::get(V4Ty),
                                                {0, 1, 2},
                                                Arg.getName() + ".load");
      Arg.replaceAllUsesWith(Shuf);
    } else {
      Load->setName(Arg.getName() + ".load");
      Arg.replaceAllUsesWith(Load);
    }
  }

  return true;
}

INITIALIZE_PASS_BEGIN(AMDGPULowerKernelArguments, DEBUG_TYPE,
                      "AMDGPU Lower Kernel Arguments", false, false)
INITIALIZE_PASS_END(AMDGPULowerKernelArguments, DEBUG_TYPE, "AMDGPU Lower Kernel Arguments",
                    false, false)

char AMDGPULowerKernelArguments::ID = 0;

FunctionPass *llvm::createAMDGPULowerKernelArgumentsPass() {
  return new AMDGPULowerKernelArguments();
}
