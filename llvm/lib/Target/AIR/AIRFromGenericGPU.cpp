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
#include "llvm/IR/Function.h"
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
    Changed |= lowerBuiltins(M);
    return Changed;
  }

private:
  // Mark SPIR_KERNEL / kernel-attributed functions as AIR kernels and normalize
  // their calling convention to C (the AIR writer keys on "air-kernel").
  bool markKernels(Module &M) {
    bool Changed = false;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      // Only the canonical generic-GPU kernel markers: the SPIR_KERNEL calling
      // convention (clang OpenCL / SYCL / SPIR-V) and the gpu dialect's
      // "gpu.kernel" attribute. We deliberately do NOT key on a bare "kernel"
      // attribute -- it is not an ecosystem-standard marker. Functions already
      // carrying "air-kernel" (Triton arrives pre-marked) are skipped, so this
      // is a no-op on AIR-native IR.
      bool IsKernel = F.getCallingConv() == CallingConv::SPIR_KERNEL ||
                      F.hasFnAttribute("gpu.kernel");
      if (!IsKernel || F.hasFnAttribute("air-kernel"))
        continue;
      F.addFnAttr("air-kernel");
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
