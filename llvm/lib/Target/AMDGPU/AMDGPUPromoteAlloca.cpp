//===-- AMDGPUPromoteAlloca.cpp - Promote Allocas -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates allocas by either converting them into vectors or
// by migrating them to local address space.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "amdgpu-promote-alloca"

using namespace llvm;

namespace {

// FIXME: This can create globals so should be a module pass.
class AMDGPUPromoteAlloca : public FunctionPass,
                            public InstVisitor<AMDGPUPromoteAlloca> {
private:
  const TargetMachine *TM;
  Module *Mod;
  MDNode *MaxWorkGroupSizeRange;

  // FIXME: This should be per-kernel.
  int LocalMemAvailable;

  bool IsAMDGCN;
  bool IsAMDHSA;

  std::pair<Value *, Value *> getLocalSizeYZ(IRBuilder<> &Builder);
  Value *getWorkitemID(IRBuilder<> &Builder, unsigned N);

public:
  static char ID;

  AMDGPUPromoteAlloca(const TargetMachine *TM_ = nullptr) :
    FunctionPass(ID),
    TM(TM_),
    Mod(nullptr),
    MaxWorkGroupSizeRange(nullptr),
    LocalMemAvailable(0),
    IsAMDGCN(false),
    IsAMDHSA(false) { }

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  const char *getPassName() const override {
    return "AMDGPU Promote Alloca";
  }

  void visitAlloca(AllocaInst &I);
};

} // End anonymous namespace

char AMDGPUPromoteAlloca::ID = 0;

INITIALIZE_TM_PASS(AMDGPUPromoteAlloca, DEBUG_TYPE,
                   "AMDGPU promote alloca to vector or LDS", false, false)

char &llvm::AMDGPUPromoteAllocaID = AMDGPUPromoteAlloca::ID;


bool AMDGPUPromoteAlloca::doInitialization(Module &M) {
  if (!TM)
    return false;

  Mod = &M;

  // The maximum workitem id.
  //
  // FIXME: Should get as subtarget property. Usually runtime enforced max is
  // 256.
  MDBuilder MDB(Mod->getContext());
  MaxWorkGroupSizeRange = MDB.createRange(APInt(32, 0), APInt(32, 2048));

  const Triple &TT = TM->getTargetTriple();

  IsAMDGCN = TT.getArch() == Triple::amdgcn;
  IsAMDHSA = TT.getOS() == Triple::AMDHSA;

  return false;
}

bool AMDGPUPromoteAlloca::runOnFunction(Function &F) {
  if (!TM || F.hasFnAttribute(Attribute::OptimizeNone))
    return false;

  FunctionType *FTy = F.getFunctionType();

  // If the function has any arguments in the local address space, then it's
  // possible these arguments require the entire local memory space, so
  // we cannot use local memory in the pass.
  for (Type *ParamTy : FTy->params()) {
    PointerType *PtrTy = dyn_cast<PointerType>(ParamTy);
    if (PtrTy && PtrTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      LocalMemAvailable = 0;
      DEBUG(dbgs() << "Function has local memory argument.  Promoting to "
                      "local memory disabled.\n");
      return false;
    }
  }

  const AMDGPUSubtarget &ST = TM->getSubtarget<AMDGPUSubtarget>(F);
  LocalMemAvailable = ST.getLocalMemorySize();
  if (LocalMemAvailable == 0)
    return false;

  // Check how much local memory is being used by global objects
  for (GlobalVariable &GV : Mod->globals()) {
    if (GV.getType()->getAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
      continue;

    for (Use &U : GV.uses()) {
      Instruction *Use = dyn_cast<Instruction>(U);
      if (!Use)
        continue;

      if (Use->getParent()->getParent() == &F)
        LocalMemAvailable -=
          Mod->getDataLayout().getTypeAllocSize(GV.getValueType());
    }
  }

  LocalMemAvailable = std::max(0, LocalMemAvailable);
  DEBUG(dbgs() << LocalMemAvailable << " bytes free in local memory.\n");

  visit(F);

  return true;
}

std::pair<Value *, Value *>
AMDGPUPromoteAlloca::getLocalSizeYZ(IRBuilder<> &Builder) {
  if (!IsAMDHSA) {
    Function *LocalSizeYFn
      = Intrinsic::getDeclaration(Mod, Intrinsic::r600_read_local_size_y);
    Function *LocalSizeZFn
      = Intrinsic::getDeclaration(Mod, Intrinsic::r600_read_local_size_z);

    CallInst *LocalSizeY = Builder.CreateCall(LocalSizeYFn, {});
    CallInst *LocalSizeZ = Builder.CreateCall(LocalSizeZFn, {});

    LocalSizeY->setMetadata(LLVMContext::MD_range, MaxWorkGroupSizeRange);
    LocalSizeZ->setMetadata(LLVMContext::MD_range, MaxWorkGroupSizeRange);

    return std::make_pair(LocalSizeY, LocalSizeZ);
  }

  // We must read the size out of the dispatch pointer.
  assert(IsAMDGCN);

  // We are indexing into this struct, and want to extract the workgroup_size_*
  // fields.
  //
  //   typedef struct hsa_kernel_dispatch_packet_s {
  //     uint16_t header;
  //     uint16_t setup;
  //     uint16_t workgroup_size_x ;
  //     uint16_t workgroup_size_y;
  //     uint16_t workgroup_size_z;
  //     uint16_t reserved0;
  //     uint32_t grid_size_x ;
  //     uint32_t grid_size_y ;
  //     uint32_t grid_size_z;
  //
  //     uint32_t private_segment_size;
  //     uint32_t group_segment_size;
  //     uint64_t kernel_object;
  //
  // #ifdef HSA_LARGE_MODEL
  //     void *kernarg_address;
  // #elif defined HSA_LITTLE_ENDIAN
  //     void *kernarg_address;
  //     uint32_t reserved1;
  // #else
  //     uint32_t reserved1;
  //     void *kernarg_address;
  // #endif
  //     uint64_t reserved2;
  //     hsa_signal_t completion_signal; // uint64_t wrapper
  //   } hsa_kernel_dispatch_packet_t
  //
  Function *DispatchPtrFn
    = Intrinsic::getDeclaration(Mod, Intrinsic::amdgcn_dispatch_ptr);

  CallInst *DispatchPtr = Builder.CreateCall(DispatchPtrFn, {});
  DispatchPtr->addAttribute(AttributeSet::ReturnIndex, Attribute::NoAlias);
  DispatchPtr->addAttribute(AttributeSet::ReturnIndex, Attribute::NonNull);

  // Size of the dispatch packet struct.
  DispatchPtr->addDereferenceableAttr(AttributeSet::ReturnIndex, 64);

  Type *I32Ty = Type::getInt32Ty(Mod->getContext());
  Value *CastDispatchPtr = Builder.CreateBitCast(
    DispatchPtr, PointerType::get(I32Ty, AMDGPUAS::CONSTANT_ADDRESS));

  // We could do a single 64-bit load here, but it's likely that the basic
  // 32-bit and extract sequence is already present, and it is probably easier
  // to CSE this. The loads should be mergable later anyway.
  Value *GEPXY = Builder.CreateConstInBoundsGEP1_64(CastDispatchPtr, 1);
  LoadInst *LoadXY = Builder.CreateAlignedLoad(GEPXY, 4);

  Value *GEPZU = Builder.CreateConstInBoundsGEP1_64(CastDispatchPtr, 2);
  LoadInst *LoadZU = Builder.CreateAlignedLoad(GEPZU, 4);

  MDNode *MD = llvm::MDNode::get(Mod->getContext(), None);
  LoadXY->setMetadata(LLVMContext::MD_invariant_load, MD);
  LoadZU->setMetadata(LLVMContext::MD_invariant_load, MD);
  LoadZU->setMetadata(LLVMContext::MD_range, MaxWorkGroupSizeRange);

  // Extract y component. Upper half of LoadZU should be zero already.
  Value *Y = Builder.CreateLShr(LoadXY, 16);

  return std::make_pair(Y, LoadZU);
}

Value *AMDGPUPromoteAlloca::getWorkitemID(IRBuilder<> &Builder, unsigned N) {
  Intrinsic::ID IntrID = Intrinsic::ID::not_intrinsic;

  switch (N) {
  case 0:
    IntrID = IsAMDGCN ? Intrinsic::amdgcn_workitem_id_x
      : Intrinsic::r600_read_tidig_x;
    break;
  case 1:
    IntrID = IsAMDGCN ? Intrinsic::amdgcn_workitem_id_y
      : Intrinsic::r600_read_tidig_y;
    break;

  case 2:
    IntrID = IsAMDGCN ? Intrinsic::amdgcn_workitem_id_z
      : Intrinsic::r600_read_tidig_z;
    break;
  default:
    llvm_unreachable("invalid dimension");
  }

  Function *WorkitemIdFn = Intrinsic::getDeclaration(Mod, IntrID);
  CallInst *CI = Builder.CreateCall(WorkitemIdFn);
  CI->setMetadata(LLVMContext::MD_range, MaxWorkGroupSizeRange);

  return CI;
}

static VectorType *arrayTypeToVecType(Type *ArrayTy) {
  return VectorType::get(ArrayTy->getArrayElementType(),
                         ArrayTy->getArrayNumElements());
}

static Value *
calculateVectorIndex(Value *Ptr,
                     const std::map<GetElementPtrInst *, Value *> &GEPIdx) {
  if (isa<AllocaInst>(Ptr))
    return Constant::getNullValue(Type::getInt32Ty(Ptr->getContext()));

  GetElementPtrInst *GEP = cast<GetElementPtrInst>(Ptr);

  auto I = GEPIdx.find(GEP);
  return I == GEPIdx.end() ? nullptr : I->second;
}

static Value* GEPToVectorIndex(GetElementPtrInst *GEP) {
  // FIXME we only support simple cases
  if (GEP->getNumOperands() != 3)
    return NULL;

  ConstantInt *I0 = dyn_cast<ConstantInt>(GEP->getOperand(1));
  if (!I0 || !I0->isZero())
    return NULL;

  return GEP->getOperand(2);
}

// Not an instruction handled below to turn into a vector.
//
// TODO: Check isTriviallyVectorizable for calls and handle other
// instructions.
static bool canVectorizeInst(Instruction *Inst, User *User) {
  switch (Inst->getOpcode()) {
  case Instruction::Load:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
    return true;
  case Instruction::Store: {
    // Must be the stored pointer operand, not a stored value.
    StoreInst *SI = cast<StoreInst>(Inst);
    return SI->getPointerOperand() == User;
  }
  default:
    return false;
  }
}

static bool tryPromoteAllocaToVector(AllocaInst *Alloca) {
  ArrayType *AllocaTy = dyn_cast<ArrayType>(Alloca->getAllocatedType());

  DEBUG(dbgs() << "Alloca candidate for vectorization\n");

  // FIXME: There is no reason why we can't support larger arrays, we
  // are just being conservative for now.
  if (!AllocaTy ||
      AllocaTy->getElementType()->isVectorTy() ||
      AllocaTy->getNumElements() > 4) {
    DEBUG(dbgs() << "  Cannot convert type to vector\n");
    return false;
  }

  std::map<GetElementPtrInst*, Value*> GEPVectorIdx;
  std::vector<Value*> WorkList;
  for (User *AllocaUser : Alloca->users()) {
    GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(AllocaUser);
    if (!GEP) {
      if (!canVectorizeInst(cast<Instruction>(AllocaUser), Alloca))
        return false;

      WorkList.push_back(AllocaUser);
      continue;
    }

    Value *Index = GEPToVectorIndex(GEP);

    // If we can't compute a vector index from this GEP, then we can't
    // promote this alloca to vector.
    if (!Index) {
      DEBUG(dbgs() << "  Cannot compute vector index for GEP " << *GEP << '\n');
      return false;
    }

    GEPVectorIdx[GEP] = Index;
    for (User *GEPUser : AllocaUser->users()) {
      if (!canVectorizeInst(cast<Instruction>(GEPUser), AllocaUser))
        return false;

      WorkList.push_back(GEPUser);
    }
  }

  VectorType *VectorTy = arrayTypeToVecType(AllocaTy);

  DEBUG(dbgs() << "  Converting alloca to vector "
        << *AllocaTy << " -> " << *VectorTy << '\n');

  for (Value *V : WorkList) {
    Instruction *Inst = cast<Instruction>(V);
    IRBuilder<> Builder(Inst);
    switch (Inst->getOpcode()) {
    case Instruction::Load: {
      Value *Ptr = Inst->getOperand(0);
      Value *Index = calculateVectorIndex(Ptr, GEPVectorIdx);
      Value *BitCast = Builder.CreateBitCast(Alloca, VectorTy->getPointerTo(0));
      Value *VecValue = Builder.CreateLoad(BitCast);
      Value *ExtractElement = Builder.CreateExtractElement(VecValue, Index);
      Inst->replaceAllUsesWith(ExtractElement);
      Inst->eraseFromParent();
      break;
    }
    case Instruction::Store: {
      Value *Ptr = Inst->getOperand(1);
      Value *Index = calculateVectorIndex(Ptr, GEPVectorIdx);
      Value *BitCast = Builder.CreateBitCast(Alloca, VectorTy->getPointerTo(0));
      Value *VecValue = Builder.CreateLoad(BitCast);
      Value *NewVecValue = Builder.CreateInsertElement(VecValue,
                                                       Inst->getOperand(0),
                                                       Index);
      Builder.CreateStore(NewVecValue, BitCast);
      Inst->eraseFromParent();
      break;
    }
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast:
      break;

    default:
      Inst->dump();
      llvm_unreachable("Inconsistency in instructions promotable to vector");
    }
  }
  return true;
}

static bool isCallPromotable(CallInst *CI) {
  // TODO: We might be able to handle some cases where the callee is a
  // constantexpr bitcast of a function.
  if (!CI->getCalledFunction())
    return false;

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI);
  if (!II)
    return false;

  switch (II->getIntrinsicID()) {
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset:
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end:
  case Intrinsic::invariant_group_barrier:
  case Intrinsic::objectsize:
    return true;
  default:
    return false;
  }
}

static bool collectUsesWithPtrTypes(Value *Val, std::vector<Value*> &WorkList) {
  for (User *User : Val->users()) {
    if (std::find(WorkList.begin(), WorkList.end(), User) != WorkList.end())
      continue;

    if (CallInst *CI = dyn_cast<CallInst>(User)) {
      if (!isCallPromotable(CI))
        return false;

      WorkList.push_back(User);
      continue;
    }

    // FIXME: Correctly handle ptrtoint instructions.
    Instruction *UseInst = dyn_cast<Instruction>(User);
    if (UseInst && UseInst->getOpcode() == Instruction::PtrToInt)
      return false;

    if (StoreInst *SI = dyn_cast_or_null<StoreInst>(UseInst)) {
      // Reject if the stored value is not the pointer operand.
      if (SI->getPointerOperand() != Val)
        return false;
    }

    if (!User->getType()->isPointerTy())
      continue;

    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(UseInst)) {
      // Be conservative if an address could be computed outside the bounds of
      // the alloca.
      if (!GEP->isInBounds())
        return false;
    }

    WorkList.push_back(User);
    if (!collectUsesWithPtrTypes(User, WorkList))
      return false;
  }

  return true;
}

void AMDGPUPromoteAlloca::visitAlloca(AllocaInst &I) {
  if (!I.isStaticAlloca())
    return;

  IRBuilder<> Builder(&I);

  // First try to replace the alloca with a vector
  Type *AllocaTy = I.getAllocatedType();

  DEBUG(dbgs() << "Trying to promote " << I << '\n');

  if (tryPromoteAllocaToVector(&I))
    return;

  DEBUG(dbgs() << " alloca is not a candidate for vectorization.\n");

  // FIXME: This is the maximum work group size.  We should try to get
  // value from the reqd_work_group_size function attribute if it is
  // available.
  unsigned WorkGroupSize = 256;
  int AllocaSize =
      WorkGroupSize * Mod->getDataLayout().getTypeAllocSize(AllocaTy);

  if (AllocaSize > LocalMemAvailable) {
    DEBUG(dbgs() << " Not enough local memory to promote alloca.\n");
    return;
  }

  std::vector<Value*> WorkList;

  if (!collectUsesWithPtrTypes(&I, WorkList)) {
    DEBUG(dbgs() << " Do not know how to convert all uses\n");
    return;
  }

  DEBUG(dbgs() << "Promoting alloca to local memory\n");
  LocalMemAvailable -= AllocaSize;

  Function *F = I.getParent()->getParent();

  Type *GVTy = ArrayType::get(I.getAllocatedType(), 256);
  GlobalVariable *GV = new GlobalVariable(
      *Mod, GVTy, false, GlobalValue::InternalLinkage,
      UndefValue::get(GVTy),
      Twine(F->getName()) + Twine('.') + I.getName(),
      nullptr,
      GlobalVariable::NotThreadLocal,
      AMDGPUAS::LOCAL_ADDRESS);
  GV->setUnnamedAddr(true);
  GV->setAlignment(I.getAlignment());

  Value *TCntY, *TCntZ;

  std::tie(TCntY, TCntZ) = getLocalSizeYZ(Builder);
  Value *TIdX = getWorkitemID(Builder, 0);
  Value *TIdY = getWorkitemID(Builder, 1);
  Value *TIdZ = getWorkitemID(Builder, 2);

  Value *Tmp0 = Builder.CreateMul(TCntY, TCntZ, "", true, true);
  Tmp0 = Builder.CreateMul(Tmp0, TIdX);
  Value *Tmp1 = Builder.CreateMul(TIdY, TCntZ, "", true, true);
  Value *TID = Builder.CreateAdd(Tmp0, Tmp1);
  TID = Builder.CreateAdd(TID, TIdZ);

  Value *Indices[] = {
    Constant::getNullValue(Type::getInt32Ty(Mod->getContext())),
    TID
  };

  Value *Offset = Builder.CreateInBoundsGEP(GVTy, GV, Indices);
  I.mutateType(Offset->getType());
  I.replaceAllUsesWith(Offset);
  I.eraseFromParent();

  for (Value *V : WorkList) {
    CallInst *Call = dyn_cast<CallInst>(V);
    if (!Call) {
      Type *EltTy = V->getType()->getPointerElementType();
      PointerType *NewTy = PointerType::get(EltTy, AMDGPUAS::LOCAL_ADDRESS);

      // The operand's value should be corrected on its own.
      if (isa<AddrSpaceCastInst>(V))
        continue;

      // FIXME: It doesn't really make sense to try to do this for all
      // instructions.
      V->mutateType(NewTy);
      continue;
    }

    IntrinsicInst *Intr = dyn_cast<IntrinsicInst>(Call);
    if (!Intr) {
      // FIXME: What is this for? It doesn't make sense to promote arbitrary
      // function calls. If the call is to a defined function that can also be
      // promoted, we should be able to do this once that function is also
      // rewritten.

      std::vector<Type*> ArgTypes;
      for (unsigned ArgIdx = 0, ArgEnd = Call->getNumArgOperands();
                                ArgIdx != ArgEnd; ++ArgIdx) {
        ArgTypes.push_back(Call->getArgOperand(ArgIdx)->getType());
      }
      Function *F = Call->getCalledFunction();
      FunctionType *NewType = FunctionType::get(Call->getType(), ArgTypes,
                                                F->isVarArg());
      Constant *C = Mod->getOrInsertFunction((F->getName() + ".local").str(),
                                             NewType, F->getAttributes());
      Function *NewF = cast<Function>(C);
      Call->setCalledFunction(NewF);
      continue;
    }

    Builder.SetInsertPoint(Intr);
    switch (Intr->getIntrinsicID()) {
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      // These intrinsics are for address space 0 only
      Intr->eraseFromParent();
      continue;
    case Intrinsic::memcpy: {
      MemCpyInst *MemCpy = cast<MemCpyInst>(Intr);
      Builder.CreateMemCpy(MemCpy->getRawDest(), MemCpy->getRawSource(),
                           MemCpy->getLength(), MemCpy->getAlignment(),
                           MemCpy->isVolatile());
      Intr->eraseFromParent();
      continue;
    }
    case Intrinsic::memmove: {
      MemMoveInst *MemMove = cast<MemMoveInst>(Intr);
      Builder.CreateMemMove(MemMove->getRawDest(), MemMove->getRawSource(),
                            MemMove->getLength(), MemMove->getAlignment(),
                            MemMove->isVolatile());
      Intr->eraseFromParent();
      continue;
    }
    case Intrinsic::memset: {
      MemSetInst *MemSet = cast<MemSetInst>(Intr);
      Builder.CreateMemSet(MemSet->getRawDest(), MemSet->getValue(),
                           MemSet->getLength(), MemSet->getAlignment(),
                           MemSet->isVolatile());
      Intr->eraseFromParent();
      continue;
    }
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::invariant_group_barrier:
      Intr->eraseFromParent();
      // FIXME: I think the invariant marker should still theoretically apply,
      // but the intrinsics need to be changed to accept pointers with any
      // address space.
      continue;
    case Intrinsic::objectsize: {
      Value *Src = Intr->getOperand(0);
      Type *SrcTy = Src->getType()->getPointerElementType();
      Function *ObjectSize = Intrinsic::getDeclaration(Mod,
        Intrinsic::objectsize,
        { Intr->getType(), PointerType::get(SrcTy, AMDGPUAS::LOCAL_ADDRESS) }
      );

      CallInst *NewCall
        = Builder.CreateCall(ObjectSize, { Src, Intr->getOperand(1) });
      Intr->replaceAllUsesWith(NewCall);
      Intr->eraseFromParent();
      continue;
    }
    default:
      Intr->dump();
      llvm_unreachable("Don't know how to promote alloca intrinsic use.");
    }
  }
}

FunctionPass *llvm::createAMDGPUPromoteAlloca(const TargetMachine *TM) {
  return new AMDGPUPromoteAlloca(TM);
}
