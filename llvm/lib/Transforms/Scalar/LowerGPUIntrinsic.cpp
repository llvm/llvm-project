//===- LowerGPUIntrinsic.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower the llvm.gpu intrinsics to target specific code sequences.
// Can be called from clang if building for a specific GPU or from the backend
// as part of a SPIRV lowering pipeline. Initial pass can lower to amdgcn or
// nvptx, adding further architectures means adding a column to the lookup table
// and further intrinsics adding a row.
//
// The idea is for the intrinsics to represent a thin abstraction over the
// different GPU architectures. In particular, code compiled to spirv-- without
// specifying a specific target can be specialised at JIT time, at which point
// this pass will rewrite those intrinsics to ones that the current backend
// knows.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LowerGPUIntrinsic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "lower-gpu-intrinsic"

using namespace llvm;

namespace {

// For each intrinsic, specify what function to call to lower it
typedef bool (*lowerFunction)(Module &M, IRBuilder<> &, Intrinsic::ID from,
                              CallBase *CI);

// Simple lowering, directly replace the intrinsic with a different one
// with the same type, and optionally refine range metadata on the return value
template <Intrinsic::ID To>
bool S(Module &M, IRBuilder<> &, Intrinsic::ID from, CallBase *CI) {

  static_assert(To != Intrinsic::not_intrinsic);
  Intrinsic::ID GenericID = from;
  Intrinsic::ID SpecificID = To;

  bool Changed = false;
  Function *Generic = Intrinsic::getDeclarationIfExists(&M, GenericID);
  auto *Specific = Intrinsic::getOrInsertDeclaration(&M, SpecificID);

  if ((Generic->getType() != Specific->getType()) ||
      (Generic->getReturnType() != Specific->getReturnType()))
    report_fatal_error("LowerGPUIntrinsic: Inconsistent types between "
                       "intrinsics in lookup table");

  CI->setCalledFunction(Specific);
  Changed = true;

  return Changed;
}

// Replace intrinsic call with a linear sequence of instructions
typedef Value *(*builder)(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                          CallBase *CI);

template <builder F>
bool B(Module &M, IRBuilder<> &Builder, Intrinsic::ID from, CallBase *CI) {
  bool Changed = false;

  Builder.SetInsertPoint(CI);

  Value *replacement = F(M, Builder, from, CI);
  if (replacement) {
    CI->replaceAllUsesWith(replacement);
    CI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

template <Intrinsic::ID Numerator, Intrinsic::ID Denominator>
Value *intrinsicRatio(Module &M, IRBuilder<> &Builder, Intrinsic::ID,
                      CallBase *) {
  Value *N = Builder.CreateIntrinsic(Numerator, {}, {});
  Value *D = Builder.CreateIntrinsic(Denominator, {}, {});
  return Builder.CreateUDiv(N, D);
}

namespace amdgpu {
Value *lane_mask(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                 CallBase *CI) {
  auto &Ctx = M.getContext();
  return Builder.CreateIntrinsic(
      Intrinsic::amdgcn_ballot, {Type::getInt64Ty(Ctx)},
      {ConstantInt::get(Type::getInt1Ty(Ctx), true)});
}

Value *lane_id(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
               CallBase *CI) {
  auto &Ctx = M.getContext();
  Constant *M1 = ConstantInt::get(Type::getInt32Ty(Ctx), -1);
  Constant *Z = ConstantInt::get(Type::getInt32Ty(Ctx), 0);

  CallInst *Lo =
      Builder.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_lo, {}, {M1, Z});
  return Builder.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_hi, {}, {M1, Lo});
}

Value *first_lane(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                  CallBase *CI) {
  auto &Ctx = M.getContext();
  return Builder.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane,
                                 {Type::getInt32Ty(Ctx)},
                                 {CI->getArgOperand(1)});
}

Value *shuffle_idx(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                   CallBase *CI) {
  auto &Ctx = M.getContext();

  Value *idx = CI->getArgOperand(1);
  Value *x = CI->getArgOperand(2);
  Value *width = CI->getArgOperand(3);

  Value *id = Builder.CreateIntrinsic(Intrinsic::gpu_lane_id, {}, {});

  Value *n = Builder.CreateSub(ConstantInt::get(Type::getInt32Ty(Ctx), 0),
                               width, "not");
  Value *a = Builder.CreateAnd(id, n, "and");
  Value *add = Builder.CreateAdd(a, idx, "add");
  Value *shl =
      Builder.CreateShl(add, ConstantInt::get(Type::getInt32Ty(Ctx), 2), "shl");
  return Builder.CreateIntrinsic(Intrinsic::amdgcn_ds_bpermute, {}, {shl, x});
}

Value *ballot(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
              CallBase *CI) {
  auto &Ctx = M.getContext();

  Value *C =
      Builder.CreateIntrinsic(Intrinsic::amdgcn_ballot, {Type::getInt64Ty(Ctx)},
                              {CI->getArgOperand(1)});

  return Builder.CreateAnd(C, CI->getArgOperand(0));
}

Value *sync_threads(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                    CallBase *CI) {
  auto &Ctx = M.getContext();
  Builder.CreateIntrinsic(Intrinsic::amdgcn_s_barrier, {}, {});

  Value *F = Builder.CreateFence(AtomicOrdering::SequentiallyConsistent,
                                 Ctx.getOrInsertSyncScopeID("workgroup"));

  return F;
}

Value *sync_lane(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                 CallBase *CI) {
  return Builder.CreateIntrinsic(Intrinsic::amdgcn_wave_barrier, {}, {});
}

Value *thread_suspend(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                      CallBase *CI) {

  auto &Ctx = M.getContext();
  return Builder.CreateIntrinsic(Intrinsic::amdgcn_s_sleep, {},
                                 {ConstantInt::get(Type::getInt32Ty(Ctx), 2)});
}

Value *dispatch_ptr(IRBuilder<> &Builder) {
  CallInst *Call =
      Builder.CreateIntrinsic(Intrinsic::amdgcn_dispatch_ptr, {}, {});
  Call->addRetAttr(
      Attribute::getWithDereferenceableBytes(Call->getContext(), 64));
  Call->addRetAttr(Attribute::getWithAlignment(Call->getContext(), Align(4)));
  return Call;
}

Value *implicit_arg_ptr(IRBuilder<> &Builder) {
  CallInst *Call =
      Builder.CreateIntrinsic(Intrinsic::amdgcn_implicitarg_ptr, {}, {});
  Call->addRetAttr(
      Attribute::getWithDereferenceableBytes(Call->getContext(), 256));
  Call->addRetAttr(Attribute::getWithAlignment(Call->getContext(), Align(8)));
  return Call;
}

template <unsigned Index>
Value *grid_size(Module &M, IRBuilder<> &Builder, Intrinsic::ID, CallBase *) {
  auto &Ctx = M.getContext();
  const unsigned XOffset = 12;
  auto *DP = dispatch_ptr(Builder);

  // Indexing the HSA kernel_dispatch_packet struct.
  auto *Offset = ConstantInt::get(Type::getInt32Ty(Ctx), XOffset + Index * 4);
  auto *GEP = Builder.CreateGEP(Type::getInt8Ty(Ctx), DP, Offset);
  auto *LD = Builder.CreateLoad(Type::getInt32Ty(Ctx), GEP);
  llvm::MDBuilder MDB(Ctx);
  // Known non-zero.
  LD->setMetadata(llvm::LLVMContext::MD_range,
                  MDB.createRange(APInt(32, 1), APInt::getZero(32)));
  LD->setMetadata(llvm::LLVMContext::MD_invariant_load,
                  llvm::MDNode::get(Ctx, {}));
  return LD;
}

template <int Index>
Value *WGSize(Module &M, IRBuilder<> &Builder, Intrinsic::ID ,
              CallBase *) {

  // Note: "__oclc_ABI_version" is supposed to be emitted and initialized by
  //       clang during compilation of user code.
  StringRef Name = "__oclc_ABI_version";
  auto *ABIVersionC = M.getNamedGlobal(Name);
  if (!ABIVersionC) {
    // In CGBuiltin, we'd have to create an extern variable to emit the load for
    // Here, we can leave the intrinsic in place and it'll get lowered later
    return nullptr;
  }
  auto &Ctx = M.getContext();

  Value *ABIVersion = Builder.CreateLoad(Type::getInt32Ty(Ctx), ABIVersionC);

  Value *IsCOV5 = Builder.CreateICmpSGE(
      ABIVersion,
      ConstantInt::get(Type::getInt32Ty(Ctx), CodeObjectVersionKind::COV_5));

  Value *ImplicitGEP = Builder.CreateConstGEP1_32(
      Type::getInt8Ty(Ctx), implicit_arg_ptr(Builder), 12 + Index * 2);

  // Indexing the HSA kernel_dispatch_packet struct.
  Value *DispatchGEP = Builder.CreateConstGEP1_32(
      Type::getInt8Ty(Ctx), dispatch_ptr(Builder), 4 + Index * 2);

  auto Result = Builder.CreateSelect(IsCOV5, ImplicitGEP, DispatchGEP);
  LoadInst *LD = Builder.CreateLoad(Type::getInt16Ty(Ctx), Result);

  // TODO: CGBuiltin digs MaxOpenCLWorkGroupSize out of targetinfo and limits
  // the range on the load based on that (MD_range)

  LD->setMetadata(llvm::LLVMContext::MD_noundef, llvm::MDNode::get(Ctx, {}));
  LD->setMetadata(llvm::LLVMContext::MD_invariant_load,
                  llvm::MDNode::get(Ctx, {}));

  // The workgroup size is a uint16_t but gpu_block_id returns a uint32_t
  return Builder.CreateZExt(LD, Type::getInt32Ty(Ctx));
}


template <int Index>
Value *NumBlocks(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
              CallBase *CI)
  {
    // This is __builtin_amdgcn_grid_size / gpu_num_threads
    // However we don't have a grid size intrinsic so can't expand to use that
    // Open code it directly instead as the equivalent to
    // Thus amdgpu::grid_size<Index> / amdgpu::WGSize<Index>
    Value *Numerator = grid_size<Index>(M, Builder, Intrinsic::not_intrinsic, nullptr);
    Value *Denominator = WGSize<Index>(M, Builder,  Intrinsic::not_intrinsic, nullptr);
    return Builder.CreateUDiv(Numerator, Denominator);
  }
  
} // namespace amdgpu

namespace nvptx {
Value *lane_mask(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                 CallBase *CI) {
  auto &Ctx = M.getContext();
  CallInst *C = Builder.CreateIntrinsic(Intrinsic::nvvm_activemask, {}, {});
  return Builder.CreateZExt(C, Type::getInt64Ty(Ctx), "conv");
}

Value *first_lane(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                  CallBase *CI) {
  auto &Ctx = M.getContext();
  Value *conv =
      Builder.CreateTrunc(CI->getArgOperand(0), Type::getInt32Ty(Ctx), "conv");
  Value *C = Builder.CreateIntrinsic(
      Intrinsic::cttz, {Type::getInt32Ty(Ctx)},
      {conv, ConstantInt::get(Type::getInt1Ty(Ctx), true)});
  Value *iszero = Builder.CreateICmpEQ(
      conv, ConstantInt::get(Type::getInt32Ty(Ctx), 0), "iszero");
  Value *sub = Builder.CreateSelect(
      iszero, ConstantInt::get(Type::getInt32Ty(Ctx), -1), C, "sub");

  return Builder.CreateIntrinsic(Intrinsic::nvvm_shfl_sync_idx_i32, {},
                                 {conv, CI->getArgOperand(1), sub,
                                  ConstantInt::get(Type::getInt32Ty(Ctx), 31)});
}

Value *shuffle_idx(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                   CallBase *CI) {
  auto &Ctx = M.getContext();

  Value *lane_mask = CI->getArgOperand(0);
  Value *idx = CI->getArgOperand(1);
  Value *x = CI->getArgOperand(2);
  Value *width = CI->getArgOperand(3);

  Value *Conv = Builder.CreateTrunc(lane_mask, Type::getInt32Ty(Ctx), "conv");

  Value *sh_prom = Builder.CreateZExt(idx, Type::getInt64Ty(Ctx), "sh_prom");
  Value *shl0 =
      Builder.CreateShl(width, ConstantInt::get(Type::getInt32Ty(Ctx), 8));
  Value *or0 = Builder.CreateSub(ConstantInt::get(Type::getInt32Ty(Ctx), 8223),
                                 shl0, "or");

  Value *core = Builder.CreateIntrinsic(Intrinsic::nvvm_shfl_sync_idx_i32, {},
                                        {Conv, x, idx, or0});

  Value *shl1 =
      Builder.CreateShl(ConstantInt::get(Type::getInt64Ty(Ctx), 1), sh_prom);
  Value *and0 = Builder.CreateAnd(shl1, lane_mask);
  Value *cmp =
      Builder.CreateICmpEQ(and0, ConstantInt::get(Type::getInt64Ty(Ctx), 0));
  Value *and4 = Builder.CreateSelect(
      cmp, ConstantInt::get(Type::getInt32Ty(Ctx), 0), core, "and4");

  return and4;
}

Value *ballot(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
              CallBase *CI) {
  auto &Ctx = M.getContext();
  Value *Conv =
      Builder.CreateTrunc(CI->getArgOperand(0), Type::getInt32Ty(Ctx), "conv");
  Value *C = Builder.CreateIntrinsic(Intrinsic::nvvm_vote_ballot_sync, {},
                                     {Conv, CI->getArgOperand(1)});

  return Builder.CreateZExt(C, Type::getInt64Ty(Ctx), "conv");
}

Value *sync_lane(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                 CallBase *CI) {

  auto &Ctx = M.getContext();
  Value *X = Builder.CreateTrunc(CI->getArgOperand(0), Type::getInt32Ty(Ctx));
  return Builder.CreateIntrinsic(Intrinsic::nvvm_bar_warp_sync, {}, {X});
}

Value *thread_suspend(Module &M, IRBuilder<> &Builder, Intrinsic::ID from,
                      CallBase *CI) {

  auto &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  Value *str = Builder.CreateGlobalString(
      "__CUDA_ARCH", "", DL.getDefaultGlobalsAddressSpace(), &M);

  Builder.SetInsertPoint(CI);
  Value *Reflect = Builder.CreateIntrinsic(Intrinsic::nvvm_reflect, {}, {str});
  Value *Cmp = Builder.CreateICmpUGT(
      Reflect, ConstantInt::get(Type::getInt32Ty(Ctx), 699));

  Builder.SetInsertPoint(SplitBlockAndInsertIfThen(Cmp, CI, false));

  Builder.CreateIntrinsic(Intrinsic::nvvm_nanosleep, {},
                          {ConstantInt::get(Type::getInt32Ty(Ctx), 64)});

  CI->eraseFromParent();
  return nullptr; // All done
}

} // namespace nvptx

struct IntrinsicMap {
  Intrinsic::ID Generic;
  lowerFunction AMDGPU;
  lowerFunction NVPTX;
};

using namespace Intrinsic;

static const IntrinsicMap ls[] = {
    // This table of intrinsic => what to do with it is walked in order.
    // A row can create calls to intrinsics that are expanded in subsequent rows
    // but that does mean that the order of rows is somewhat significant.
    // S<intrinsic> is a simple lowering to an existing intrinsic
    // B<function> involves building a short sequence of instructions

    {
        gpu_num_blocks_x,
        B<amdgpu::NumBlocks<0>>,
        S<nvvm_read_ptx_sreg_nctaid_x>,
    },
    {
        gpu_num_blocks_y,
        B<amdgpu::NumBlocks<1>>,
        S<nvvm_read_ptx_sreg_nctaid_y>,
    },
    {
        gpu_num_blocks_z,
        B<amdgpu::NumBlocks<2>>,
        S<nvvm_read_ptx_sreg_nctaid_z>,
    },

    // Note: Could canonicalise in favour of the target agnostic one without
    // breaking existing users of builtin or intrinsic:
    //  {amdgcn_workgroup_id_x, S<gpu_block_id_x>, nullptr},
    //  {gpu_block_id_x, nullptr, S<nvvm_read_ptx_sreg_ctaid_x>},
    // Using the target agnostic one throughout the rest of the backend would
    // work fine, and amdgpu-no-workgroup-id-x attribute and similar may be
    // applicable to other targets.
    // Map {block,thread}_id onto existing intrinsics for the time being.
    {gpu_block_id_x, S<amdgcn_workgroup_id_x>, S<nvvm_read_ptx_sreg_ctaid_x>},
    {gpu_block_id_y, S<amdgcn_workgroup_id_y>, S<nvvm_read_ptx_sreg_ctaid_y>},
    {gpu_block_id_z, S<amdgcn_workgroup_id_z>, S<nvvm_read_ptx_sreg_ctaid_z>},
    {gpu_thread_id_x, S<amdgcn_workitem_id_x>, S<nvvm_read_ptx_sreg_tid_x>},
    {gpu_thread_id_y, S<amdgcn_workitem_id_y>, S<nvvm_read_ptx_sreg_tid_y>},
    {gpu_thread_id_z, S<amdgcn_workitem_id_z>, S<nvvm_read_ptx_sreg_tid_z>},
    
    // CGBuiltin maps builtin_amdgcn_workgroup_size onto gpu_num_threads
    {gpu_num_threads_x, B<amdgpu::WGSize<0>>, S<nvvm_read_ptx_sreg_ntid_x>},
    {gpu_num_threads_y, B<amdgpu::WGSize<1>>, S<nvvm_read_ptx_sreg_ntid_y>},
    {gpu_num_threads_z, B<amdgpu::WGSize<2>>, S<nvvm_read_ptx_sreg_ntid_z>},

    // Some of the following intrinsics need minor impedance matching
    {gpu_num_lanes, S<amdgcn_wavefrontsize>, S<nvvm_read_ptx_sreg_warpsize>},
    {gpu_lane_mask, B<amdgpu::lane_mask>, B<nvptx::lane_mask>},

    {gpu_read_first_lane_u32, B<amdgpu::first_lane>, B<nvptx::first_lane>},
    {gpu_shuffle_idx_u32, B<amdgpu::shuffle_idx>, B<nvptx::shuffle_idx>},

    // shuffle sometimes emits call into lane_id so lower lane_id after shuffle
    {gpu_lane_id, B<amdgpu::lane_id>, S<nvvm_read_ptx_sreg_laneid>},

    {gpu_ballot, B<amdgpu::ballot>, B<nvptx::ballot>},

    {gpu_sync_threads, B<amdgpu::sync_threads>, S<nvvm_barrier0>},
    {gpu_sync_lane, B<amdgpu::sync_lane>, B<nvptx::sync_lane>},

    {gpu_thread_suspend, B<amdgpu::thread_suspend>, B<nvptx::thread_suspend>},
    {gpu_exit, S<amdgcn_endpgm>, S<nvvm_exit>},
};

class LowerGPUIntrinsic : public ModulePass {
public:
  static char ID;

  LowerGPUIntrinsic() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;
};

bool LowerGPUIntrinsic::runOnModule(Module &M) {
  bool Changed = false;

  Triple TT(M.getTargetTriple());

  if (!TT.isAMDGPU() && !TT.isNVPTX()) {
    return Changed;
  }

  auto &Ctx = M.getContext();
  IRBuilder<> Builder(Ctx);

  for (const IntrinsicMap &I : ls) {
    auto *Intr = Intrinsic::getDeclarationIfExists(&M, I.Generic);
    if (!Intr)
      continue;

    lowerFunction maybeLowering = TT.isAMDGPU() ? I.AMDGPU : I.NVPTX;
    if (maybeLowering == nullptr)
      continue;

    for (auto *U : make_early_inc_range(Intr->users())) {
      if (auto *CI = dyn_cast<CallBase>(U)) {
        assert (CI->getCalledFunction() == Intr);
        Changed |= maybeLowering(M, Builder, I.Generic, CI);
      }
    }
  }

  return Changed;
}

} // namespace

char LowerGPUIntrinsic::ID = 0;

INITIALIZE_PASS(LowerGPUIntrinsic, DEBUG_TYPE, "Lower GPU Intrinsic", false,
                false)

Pass *llvm::createLowerGPUIntrinsicPass() { return new LowerGPUIntrinsic(); }

PreservedAnalyses LowerGPUIntrinsicPass::run(Module &M,
                                             ModuleAnalysisManager &) {
  return LowerGPUIntrinsic().runOnModule(M) ? PreservedAnalyses::none()
                                            : PreservedAnalyses::all();
}
