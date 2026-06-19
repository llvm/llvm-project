//===- AIRFromGenericGPU.cpp - Lower generic GPU IR to AIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Make the AIR target a first-class consumer of GENERIC GPU LLVM IR -- i.e. IR
// produced by frontends other than Triton that lower to the OpenCL / SPIR-V
// builtin convention. The MLIR `gpu` dialect's `--convert-gpu-to-llvm-spv`
// lowering, SYCL/DPC++, and clang's OpenCL path all emit thread-index and
// barrier queries as calls to mangled OpenCL work-item builtins:
//
//     i64 @_Z12get_local_idj(i32 dim)     ; gpu.thread_id  / get_local_id
//     i64 @_Z12get_group_idj(i32 dim)     ; gpu.block_id   / get_group_id
//     i64 @_Z13get_global_idj(i32 dim)    ; global linear id
//     i64 @_Z14get_local_sizej(i32 dim)   ; threadgroup size
//     i64 @_Z14get_num_groupsj(i32 dim)   ; grid size in groups
//     void @_Z7barrierj(i32 flags)        ; gpu.barrier
//
// AIR expresses the same quantities through its own intrinsics, which return a
// `[3 x i32]` vector indexed by dimension:
//
//     [3 x i32] @air.thread_position_in_threadgroup()
//     [3 x i32] @air.threadgroup_position_in_grid()
//     [3 x i32] @air.threads_per_threadgroup()
//     [3 x i32] @air.threadgroups_per_grid()
//     void      @air.wg.barrier(i32)
//
// This pass rewrites each builtin call into the matching AIR intrinsic + an
// extractelement of the requested dimension (dim is a constant in practice).
// `get_global_id(d)` has no single AIR intrinsic; it is the standard
// `tid_in_tg + tgpos_in_grid * threads_per_tg`, so we synthesize it.
//
// It also marks every kernel (a function with the SPIR_KERNEL calling
// convention, or the `kernel`/`gpu.kernel` attribute) with the `"air-kernel"`
// function attribute that the AIR writer keys on, and strips the SPIR_KERNEL CC
// (AIR kernels use the C calling convention + the attribute). The AIR writer
// then synthesizes all air.* kernel/buffer metadata from the signature, so no
// metadata work is needed here.
//
// After this pass, generic-GPU IR is indistinguishable from Triton-produced IR
// to the rest of the AIR backend, so the existing pipeline (AIRPrepare, the
// AIRWriter, etc.) handles it unchanged. Run it FIRST, before the AIR passes.
//
//===----------------------------------------------------------------------===//

#include "AIR.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "air-from-generic-gpu"

namespace {

// One OpenCL work-item builtin -> the AIR intrinsic that supplies the same
// per-dimension [3 x i32] vector. Barrier and global-id are special-cased.
struct IndexBuiltin {
  StringRef OCLName;  // mangled OpenCL builtin
  StringRef AIRName;  // AIR intrinsic returning [3 x i32]
};

static const IndexBuiltin IndexBuiltins[] = {
    {"_Z12get_local_idj", "air.thread_position_in_threadgroup"},
    {"_Z12get_group_idj", "air.threadgroup_position_in_grid"},
    {"_Z14get_local_sizej", "air.threads_per_threadgroup"},
    {"_Z14get_num_groupsj", "air.threadgroups_per_grid"},
};

static constexpr StringRef kBarrierOCL = "_Z7barrierj";
static constexpr StringRef kGlobalIdOCL = "_Z13get_global_idj";

// Scalar sub-group (SIMD) builtins. Unlike the work-item builtins these take no
// dimension argument; each maps to an AIR intrinsic returning a single i32. The
// AIR names are exactly those Apple's `xcrun metal` emits as kernel-arg
// system-value attributes for the matching `[[*]]` qualifiers, so AIRSystemValues
// lowers them to implicit kernel params the GPU JIT accepts.
struct ScalarBuiltin {
  StringRef OCLName; // mangled OpenCL builtin
  StringRef AIRName; // AIR intrinsic returning i32
};

static const ScalarBuiltin ScalarBuiltins[] = {
    {"_Z18get_sub_group_sizev", "air.threads_per_simdgroup"},
    {"_Z18get_num_sub_groupsv", "air.simdgroups_per_threadgroup"},
    {"_Z16get_sub_group_idv", "air.simdgroup_index_in_threadgroup"},
    {"_Z22get_sub_group_local_idv", "air.thread_index_in_simdgroup"},
};

// Get (or declare) an AIR intrinsic of type `i32 ()`.
static FunctionCallee getAIRScalarIntrinsic(Module &M, StringRef Name) {
  LLVMContext &Ctx = M.getContext();
  FunctionType *FT = FunctionType::get(Type::getInt32Ty(Ctx), /*isVarArg=*/false);
  return M.getOrInsertFunction(Name, FT);
}

// Get (or declare) an AIR intrinsic of type `[3 x i32] ()`.
static FunctionCallee getAIRIndexIntrinsic(Module &M, StringRef Name) {
  LLVMContext &Ctx = M.getContext();
  Type *V3i32 = ArrayType::get(Type::getInt32Ty(Ctx), 3);
  FunctionType *FT = FunctionType::get(V3i32, /*isVarArg=*/false);
  return M.getOrInsertFunction(Name, FT);
}

// Lower one `[3 x i32] = air.intr(); extractelement [dim]` for a builtin call
// whose single operand is the dimension. The dim is a runtime i32; we
// extractvalue at a constant index when it is constant, else build a switch-free
// select chain (rare; the gpu dialect always passes a constant dim).
static Value *emitIndexForDim(IRBuilder<> &B, Module &M, StringRef AIRName,
                              Value *Dim) {
  Value *Vec = B.CreateCall(getAIRIndexIntrinsic(M, AIRName), {}, "air.idx");
  if (auto *CI = dyn_cast<ConstantInt>(Dim)) {
    unsigned D = CI->getZExtValue() & 0x3;
    return B.CreateExtractValue(Vec, {D}, "air.idx.d");
  }
  // Non-constant dim: pick lane via two selects (d==0 ? x : d==1 ? y : z).
  Value *X = B.CreateExtractValue(Vec, {0});
  Value *Y = B.CreateExtractValue(Vec, {1});
  Value *Z = B.CreateExtractValue(Vec, {2});
  Value *IsZero = B.CreateICmpEQ(Dim, B.getInt32(0));
  Value *IsOne = B.CreateICmpEQ(Dim, B.getInt32(1));
  return B.CreateSelect(IsZero, X, B.CreateSelect(IsOne, Y, Z));
}

class AIRFromGenericGPU : public ModulePass {
public:
  static char ID;
  AIRFromGenericGPU() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    bool Changed = false;
    Changed |= markKernels(M);
    Changed |= promoteWorkgroupAttributions(M);
    Changed |= lowerBuiltins(M);
    return Changed;
  }

private:
  // The MLIR gpu dialect passes a statically-sized `workgroup(...)` attribution
  // as an addrspace(3) kernel pointer PARAMETER (Apple's host-bound
  // `[[threadgroup(N)]]` shape). A consumer whose host has no API to bind
  // threadgroup memory at dispatch reads the buffer back as zero, so for those
  // we mirror Apple's frontend: replace each addrspace(3) pointer param with an
  // internal addrspace(3) global and drop it from the signature, letting the
  // existing threadgroup-global machinery (TGGlobalCoalesce/AIRPrepare) and
  // AIRSystemValues (which then sees no TG param) handle it unchanged.
  //
  // A frontend whose runtime DOES bind threadgroup memory (setThreadgroupMemory
  // Length:atIndex:) declares it via the "air.thread_group_bound" function
  // attribute; we then keep the param as a true dynamic threadgroup argument
  // (staticThreadgroupMemoryLength stays 0, freeing per-core occupancy).
  bool promoteWorkgroupAttributions(Module &M) {
    bool Changed = false;
    SmallVector<Function *, 2> Kernels;
    for (Function &F : M)
      if (!F.isDeclaration() && F.hasFnAttribute("air-kernel") &&
          !F.hasFnAttribute("air.thread_group_bound"))
        Kernels.push_back(&F);

    for (Function *F : Kernels) {
      SmallVector<unsigned, 2> TGParams;
      for (unsigned I = 0; I < F->arg_size(); ++I) {
        Type *T = F->getArg(I)->getType();
        if (T->isPointerTy() && T->getPointerAddressSpace() == 3)
          TGParams.push_back(I);
      }
      if (TGParams.empty())
        continue;

      for (unsigned I : TGParams) {
        Argument *Arg = F->getArg(I);
        auto [ElemTy, Count] = inferThreadgroupAlloc(Arg);
        auto *ArrTy = ArrayType::get(ElemTy, Count);
        auto *GV = new GlobalVariable(
            M, ArrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
            UndefValue::get(ArrTy),
            (F->getName() + ".wg." + Twine(I)).str(), nullptr,
            GlobalValue::NotThreadLocal, /*AddressSpace=*/3);
        GV->setAlignment(Align(ElemTy->getPrimitiveSizeInBits() >= 32 ? 4 : 2));
        Arg->replaceAllUsesWith(GV);
      }

      rebuildWithoutParams(F, TGParams);
      Changed = true;
    }
    return Changed;
  }

  // Element type and element count for an addrspace(3) attribution param.
  // Element type comes from the param's load/store/GEP uses; the count is the
  // largest constant bound the access index is compared against (the loop /
  // threadgroup range), defaulting to 1 when nothing constrains it.
  std::pair<Type *, uint64_t> inferThreadgroupAlloc(Argument *Arg) {
    Type *ElemTy = nullptr;
    uint64_t Count = 1;
    bool HaveAuthoritativeCount = false;
    {
      Attribute A = Arg->getParent()->getAttributes().getParamAttr(
          Arg->getArgNo(), "air.wg.num_elems");
      if (A.isValid()) {
        uint64_t N = 0;
        if (!A.getValueAsString().getAsInteger(10, N) && N > 0) {
          Count = N;
          HaveAuthoritativeCount = true;
        }
      }
    }
    SmallVector<Value *, 8> IdxValues;
    for (User *U : Arg->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        if (!ElemTy)
          ElemTy = GEP->getSourceElementType();
        for (auto It = GEP->idx_begin(); It != GEP->idx_end(); ++It)
          IdxValues.push_back(It->get());
      } else if (auto *LI = dyn_cast<LoadInst>(U)) {
        if (!ElemTy)
          ElemTy = LI->getType();
      } else if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (!ElemTy && SI->getPointerOperand() == Arg)
          ElemTy = SI->getValueOperand()->getType();
      }
    }
    if (!ElemTy)
      ElemTy = Type::getFloatTy(Arg->getContext());

    if (HaveAuthoritativeCount)
      return {ElemTy, Count};

    // Largest constant any index is icmp-compared against bounds the range.
    bool SawBound = false;
    for (Value *Idx : IdxValues)
      for (User *IU : Idx->users())
        if (auto *Cmp = dyn_cast<ICmpInst>(IU))
          for (Value *Op : Cmp->operands())
            if (auto *CI = dyn_cast<ConstantInt>(Op)) {
              Count = std::max(Count, CI->getZExtValue());
              SawBound = true;
            }
    if (!SawBound)
      report_fatal_error(
          Twine("AIRFromGenericGPU: cannot determine threadgroup allocation "
                "size for addrspace(3) parameter '") +
          Arg->getName() +
          "' (no air.wg.num_elems attribute and no bounding icmp). The "
          "threadgroup attribution size was lost; emit it via the AIR "
          "serializer or use an addrspace(3) global.");
    return {ElemTy, Count};
  }

  // Clone F without the parameters in Drop (sorted ascending), splicing the
  // body over and remapping surviving args, then RAUW the old function.
  void rebuildWithoutParams(Function *&F, ArrayRef<unsigned> Drop) {
    Module &M = *F->getParent();
    SmallDenseSet<unsigned, 4> DropSet(Drop.begin(), Drop.end());
    SmallVector<Type *, 8> NewParams;
    for (unsigned I = 0; I < F->arg_size(); ++I)
      if (!DropSet.count(I))
        NewParams.push_back(F->getArg(I)->getType());

    auto *NewFTy = FunctionType::get(F->getReturnType(), NewParams,
                                     F->isVarArg());
    auto *NewF = Function::Create(NewFTy, F->getLinkage(),
                                  F->getAddressSpace(), "", &M);
    NewF->copyAttributesFrom(F);
    NewF->setCallingConv(F->getCallingConv());
    NewF->takeName(F);
    NewF->splice(NewF->begin(), F);

    auto NewIt = NewF->arg_begin();
    for (unsigned I = 0; I < F->arg_size(); ++I) {
      if (DropSet.count(I))
        continue;
      Argument *Old = F->getArg(I);
      NewIt->setName(Old->getName());
      Old->replaceAllUsesWith(&*NewIt);
      ++NewIt;
    }
    F->eraseFromParent();
    F = NewF;
  }

  static bool isUncalledRoot(const Function &F) {
    for (const User *U : F.users())
      if (const auto *CB = dyn_cast<CallBase>(U))
        if (CB->getCalledFunction() == &F)
          return false;
    return true;
  }

  static bool usesGenericGPUBuiltins(const Function &F) {
    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB)
        if (const auto *CI = dyn_cast<CallInst>(&I))
          if (const Function *Callee = CI->getCalledFunction())
            if (isHandledBuiltin(Callee->getName()))
              return true;
    return false;
  }

  bool markKernels(Module &M) {
    bool Changed = false;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      bool IsKernel = F.getCallingConv() == CallingConv::SPIR_KERNEL ||
                      F.hasFnAttribute("gpu.kernel") ||
                      (isUncalledRoot(F) && usesGenericGPUBuiltins(F));
      if (!IsKernel || F.hasFnAttribute("air-kernel"))
        continue;
      F.addFnAttr("air-kernel");
      F.addFnAttr("air-from-generic-gpu");
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
        F.setCallingConv(CallingConv::C);
      Changed = true;
    }
    return Changed;
  }

  bool lowerBuiltins(Module &M) {
    LLVMContext &Ctx = M.getContext();
    Type *I64 = Type::getInt64Ty(Ctx);
    bool Changed = false;

    // Collect call sites first; we erase as we go.
    SmallVector<CallInst *, 16> Calls;
    for (Function &F : M)
      for (Instruction &I : instructions(F))
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Callee = CI->getCalledFunction())
            if (isHandledBuiltin(Callee->getName()))
              Calls.push_back(CI);

    for (CallInst *CI : Calls) {
      IRBuilder<> B(CI);
      StringRef Name = CI->getCalledFunction()->getName();
      Value *Repl = nullptr;

      if (Name == kBarrierOCL) {
        // void barrier(i32 flags) -> void air.wg.barrier(i32 mem_flags, i32
        // scope). air.wg.barrier is the name the AIR 4 JIT accepts (the legacy
        // air.threadgroup.barrier is what AIRBarrierRename renames; emit the
        // final name directly). The two operands are an AIR memory-fence-flags
        // mask and a sync scope; mirror what the Triton path emits
        // (mem_flags=2 = device|threadgroup fence, scope=1 = threadgroup), which
        // is the safe superset for an OpenCL barrier(CLK_LOCAL_MEM_FENCE).
        Type *I32 = Type::getInt32Ty(Ctx);
        FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), {I32, I32},
                                             /*isVarArg=*/false);
        FunctionCallee Bar = M.getOrInsertFunction("air.wg.barrier", FT);
        B.CreateCall(Bar, {B.getInt32(2), B.getInt32(1)});
        CI->eraseFromParent();
        Changed = true;
        continue;
      }

      bool IsScalar = false;
      for (const auto &E : ScalarBuiltins)
        if (E.OCLName == Name) {
          Repl = B.CreateCall(getAIRScalarIntrinsic(M, E.AIRName), {}, "air.sg");
          IsScalar = true;
          break;
        }
      if (IsScalar) {
        if (CI->getType() != Repl->getType())
          Repl = B.CreateZExtOrTrunc(Repl, CI->getType());
        CI->replaceAllUsesWith(Repl);
        CI->eraseFromParent();
        Changed = true;
        continue;
      }

      Value *Dim = CI->arg_size() ? CI->getArgOperand(0) : B.getInt32(0);
      if (Dim->getType() != Type::getInt32Ty(Ctx))
        Dim = B.CreateTrunc(Dim, Type::getInt32Ty(Ctx));

      if (Name == kGlobalIdOCL) {
        // get_global_id(d) is the GLOBAL thread position, which AIR exposes
        // directly as air.thread_position_in_grid -- no need to synthesize it
        // from tid_in_tg + group*size (that path needs air.threads_per_threadgroup,
        // which AIRSystemValues does not yet lower).
        Repl = emitIndexForDim(B, M, "air.thread_position_in_grid", Dim);
      } else {
        StringRef AIRName;
        for (const auto &E : IndexBuiltins)
          if (E.OCLName == Name)
            AIRName = E.AIRName;
        Repl = emitIndexForDim(B, M, AIRName, Dim);
      }

      // OpenCL builtins return size_t (i64 here); the AIR intrinsics give i32.
      if (CI->getType() != Repl->getType())
        Repl = B.CreateZExtOrTrunc(Repl, CI->getType());
      CI->replaceAllUsesWith(Repl);
      CI->eraseFromParent();
      Changed = true;
    }

    // Drop the now-unused OpenCL builtin declarations.
    SmallVector<Function *, 8> Dead;
    for (Function &F : M)
      if (F.isDeclaration() && isHandledBuiltin(F.getName()) && F.use_empty())
        Dead.push_back(&F);
    for (Function *F : Dead)
      F->eraseFromParent();

    (void)I64;
    return Changed;
  }

  static bool isHandledBuiltin(StringRef Name) {
    if (Name == kBarrierOCL || Name == kGlobalIdOCL)
      return true;
    for (const auto &E : IndexBuiltins)
      if (E.OCLName == Name)
        return true;
    for (const auto &E : ScalarBuiltins)
      if (E.OCLName == Name)
        return true;
    return false;
  }
};

} // namespace

char AIRFromGenericGPU::ID = 0;

INITIALIZE_PASS(AIRFromGenericGPU, DEBUG_TYPE,
                "Lower generic (OpenCL/SPIR-V) GPU IR to AIR", false, false)

ModulePass *llvm::createAIRFromGenericGPULegacyPass() {
  return new AIRFromGenericGPU();
}
