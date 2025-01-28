//===- NVPTXIntrinsics.td - PTX Intrinsics Instructions -------*- tblgen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def immFloat0 : PatLeaf<(fpimm), [{
    float f = (float)N->getValueAPF().convertToFloat();
    return (f==0.0f);
}]>;

def immFloat1 : PatLeaf<(fpimm), [{
    float f = (float)N->getValueAPF().convertToFloat();
    return (f==1.0f);
}]>;

def immDouble0 : PatLeaf<(fpimm), [{
    double d = (double)N->getValueAPF().convertToDouble();
    return (d==0.0);
}]>;

def immDouble1 : PatLeaf<(fpimm), [{
    double d = (double)N->getValueAPF().convertToDouble();
    return (d==1.0);
}]>;

def AS_match {
  code generic = [{
   return ChkMemSDNodeAddressSpace(N, llvm::ADDRESS_SPACE_GENERIC);
  }];
  code shared = [{
   return ChkMemSDNodeAddressSpace(N, llvm::ADDRESS_SPACE_SHARED);
  }];
  code global = [{
   return ChkMemSDNodeAddressSpace(N, llvm::ADDRESS_SPACE_GLOBAL);
  }];
}

// A node that will be replaced with the current PTX version.
class PTX {
  SDNodeXForm PTXVerXform = SDNodeXForm<imm, [{
    return getI32Imm(Subtarget->getPTXVersion(), SDLoc(N));
  }]>;
  // (i32 0) will be XForm'ed to the currently used PTX version.
  dag version = (PTXVerXform (i32 0));
}
def ptx : PTX;

// Generates list of n sequential register names.
// E.g. RegNames<3,"r">.ret -> ["r0", "r1", "r2" ]
class RegSeq<int n, string prefix> {
  list<string> ret = !if(n, !listconcat(RegSeq<!sub(n, 1), prefix>.ret,
                                        [prefix # !sub(n, 1)]),
                            []);
}

class THREADMASK_INFO<bit sync> {
  list<bit> ret = !if(sync, [0, 1], [0]);
}

//-----------------------------------
// Synchronization and shuffle functions
//-----------------------------------
let isConvergent = true in {
def INT_BARRIER0 : NVPTXInst<(outs), (ins),
                  "bar.sync \t0;",
      [(int_nvvm_barrier0)]>;
def INT_BARRIERN : NVPTXInst<(outs), (ins Int32Regs:$src1),
                  "bar.sync \t$src1;",
      [(int_nvvm_barrier_n i32:$src1)]>;
def INT_BARRIER : NVPTXInst<(outs), (ins Int32Regs:$src1, Int32Regs:$src2),
                  "bar.sync \t$src1, $src2;",
      [(int_nvvm_barrier i32:$src1, i32:$src2)]>;
def INT_BARRIER0_POPC : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$pred),
  !strconcat("{{ \n\t",
             ".reg .pred \t%p1; \n\t",
             "setp.ne.u32 \t%p1, $pred, 0; \n\t",
             "bar.red.popc.u32 \t$dst, 0, %p1; \n\t",
             "}}"),
      [(set i32:$dst, (int_nvvm_barrier0_popc i32:$pred))]>;
def INT_BARRIER0_AND : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$pred),
  !strconcat("{{ \n\t",
             ".reg .pred \t%p1; \n\t",
             ".reg .pred \t%p2; \n\t",
             "setp.ne.u32 \t%p1, $pred, 0; \n\t",
             "bar.red.and.pred \t%p2, 0, %p1; \n\t",
             "selp.u32 \t$dst, 1, 0, %p2; \n\t",
             "}}"),
      [(set i32:$dst, (int_nvvm_barrier0_and i32:$pred))]>;
def INT_BARRIER0_OR : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$pred),
  !strconcat("{{ \n\t",
             ".reg .pred \t%p1; \n\t",
             ".reg .pred \t%p2; \n\t",
             "setp.ne.u32 \t%p1, $pred, 0; \n\t",
             "bar.red.or.pred \t%p2, 0, %p1; \n\t",
             "selp.u32 \t$dst, 1, 0, %p2; \n\t",
             "}}"),
      [(set i32:$dst, (int_nvvm_barrier0_or i32:$pred))]>;

def INT_BAR_SYNC : NVPTXInst<(outs), (ins i32imm:$i), "bar.sync \t$i;",
                             [(int_nvvm_bar_sync imm:$i)]>;

def INT_BAR_WARP_SYNC_I : NVPTXInst<(outs), (ins i32imm:$i), "bar.warp.sync \t$i;",
                             [(int_nvvm_bar_warp_sync imm:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BAR_WARP_SYNC_R : NVPTXInst<(outs), (ins Int32Regs:$i), "bar.warp.sync \t$i;",
                             [(int_nvvm_bar_warp_sync i32:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;

def INT_BARRIER_SYNC_I : NVPTXInst<(outs), (ins i32imm:$i), "barrier.sync \t$i;",
                                   [(int_nvvm_barrier_sync imm:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BARRIER_SYNC_R : NVPTXInst<(outs), (ins Int32Regs:$i), "barrier.sync \t$i;",
                                   [(int_nvvm_barrier_sync i32:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;

def INT_BARRIER_SYNC_CNT_RR : NVPTXInst<(outs), (ins Int32Regs:$id, Int32Regs:$cnt),
                 "barrier.sync \t$id, $cnt;",
                 [(int_nvvm_barrier_sync_cnt i32:$id, i32:$cnt)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BARRIER_SYNC_CNT_RI : NVPTXInst<(outs), (ins Int32Regs:$id, i32imm:$cnt),
                 "barrier.sync \t$id, $cnt;",
                 [(int_nvvm_barrier_sync_cnt i32:$id, imm:$cnt)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BARRIER_SYNC_CNT_IR : NVPTXInst<(outs), (ins i32imm:$id, Int32Regs:$cnt),
                 "barrier.sync \t$id, $cnt;",
                 [(int_nvvm_barrier_sync_cnt imm:$id, i32:$cnt)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BARRIER_SYNC_CNT_II : NVPTXInst<(outs), (ins i32imm:$id, i32imm:$cnt),
                 "barrier.sync \t$id, $cnt;",
                 [(int_nvvm_barrier_sync_cnt imm:$id, imm:$cnt)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;

class INT_BARRIER_CLUSTER<string variant, Intrinsic Intr,
                          list<Predicate> Preds = [hasPTX<78>, hasSM<90>]>:
        NVPTXInst<(outs), (ins), "barrier.cluster."# variant #";", [(Intr)]>,
        Requires<Preds>;

def barrier_cluster_arrive:
        INT_BARRIER_CLUSTER<"arrive", int_nvvm_barrier_cluster_arrive>;
def barrier_cluster_arrive_relaxed:
        INT_BARRIER_CLUSTER<"arrive.relaxed",
        int_nvvm_barrier_cluster_arrive_relaxed, [hasPTX<80>, hasSM<90>]>;
def barrier_cluster_wait:
        INT_BARRIER_CLUSTER<"wait", int_nvvm_barrier_cluster_wait>;

// 'aligned' versions of the cluster barrier intrinsics
def barrier_cluster_arrive_aligned:
        INT_BARRIER_CLUSTER<"arrive.aligned", int_nvvm_barrier_cluster_arrive_aligned>;
def barrier_cluster_arrive_relaxed_aligned:
        INT_BARRIER_CLUSTER<"arrive.relaxed.aligned",
        int_nvvm_barrier_cluster_arrive_relaxed_aligned, [hasPTX<80>, hasSM<90>]>;
def barrier_cluster_wait_aligned:
        INT_BARRIER_CLUSTER<"wait.aligned", int_nvvm_barrier_cluster_wait_aligned>;

class SHFL_INSTR<bit sync, string mode, string reg, bit return_pred,
                 bit offset_imm, bit mask_imm, bit threadmask_imm>
      : NVPTXInst<(outs), (ins), "?", []> {
  NVPTXRegClass rc = !cond(
    !eq(reg, "i32"): Int32Regs,
    !eq(reg, "f32"): Float32Regs);
  string IntrName = "int_nvvm_shfl_"
                    # !if(sync, "sync_", "")
                    # mode
                    # "_" # reg
                    # !if(return_pred, "p", "");
  Intrinsic Intr = !cast<Intrinsic>(IntrName);
  let InOperandList = !con(
    !if(sync,
        !dag(ins, !if(threadmask_imm, [i32imm], [Int32Regs]), ["threadmask"]),
        (ins)),
    (ins rc:$src),
    !dag(ins, !if(offset_imm, [i32imm], [Int32Regs]), ["offset"]),
    !dag(ins, !if(mask_imm, [i32imm], [Int32Regs]), ["mask"])
    );
  let OutOperandList = !if(return_pred, (outs rc:$dst, Int1Regs:$pred), (outs rc:$dst));
  let AsmString = "shfl."
     # !if(sync, "sync.", "")
     # mode # ".b32\t"
     # "$dst"
     # !if(return_pred, "|$pred", "") # ", "
     # "$src, $offset, $mask"
     # !if(sync, ", $threadmask", "")
     # ";"
     ;
  let Pattern = [!con(
      !foreach(tmp, OutOperandList,
             !subst(outs, set,
             !subst(i32imm, imm, tmp))),
      (set !foreach(tmp, InOperandList,
             !subst(ins, Intr,
             !subst(i32imm, imm, tmp))))
  )];
}

foreach sync = [false, true] in {
  foreach mode = ["up", "down", "bfly", "idx"] in {
    foreach regclass = ["i32", "f32"] in {
      foreach return_pred = [false, true] in {
        foreach offset_imm = [false, true] in {
          foreach mask_imm = [false, true] in {
            foreach threadmask_imm = THREADMASK_INFO<sync>.ret in {
              def : SHFL_INSTR<sync, mode, regclass, return_pred,
                               offset_imm, mask_imm, threadmask_imm>,
                    Requires<!if(sync, [hasSM<30>, hasPTX<60>], [hasSM<30>, hasSHFL])>;
            }
          }
        }
      }
    }
  }
}

// vote.{all,any,uni,ballot}
multiclass VOTE<NVPTXRegClass regclass, string mode, Intrinsic IntOp> {
  def : NVPTXInst<(outs regclass:$dest), (ins Int1Regs:$pred),
              "vote." # mode # " \t$dest, $pred;",
              [(set regclass:$dest, (IntOp i1:$pred))]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
}

defm VOTE_ALL : VOTE<Int1Regs, "all.pred", int_nvvm_vote_all>;
defm VOTE_ANY : VOTE<Int1Regs, "any.pred", int_nvvm_vote_any>;
defm VOTE_UNI : VOTE<Int1Regs, "uni.pred", int_nvvm_vote_uni>;
defm VOTE_BALLOT : VOTE<Int32Regs, "ballot.b32", int_nvvm_vote_ballot>;

// vote.sync.{all,any,uni,ballot}
multiclass VOTE_SYNC<NVPTXRegClass regclass, string mode, Intrinsic IntOp> {
  def i : NVPTXInst<(outs regclass:$dest), (ins i32imm:$mask, Int1Regs:$pred),
              "vote.sync." # mode # " \t$dest, $pred, $mask;",
              [(set regclass:$dest, (IntOp imm:$mask, i1:$pred))]>,
          Requires<[hasPTX<60>, hasSM<30>]>;
  def r : NVPTXInst<(outs regclass:$dest), (ins Int32Regs:$mask, Int1Regs:$pred),
              "vote.sync." # mode #" \t$dest, $pred, $mask;",
              [(set regclass:$dest, (IntOp i32:$mask, i1:$pred))]>,
          Requires<[hasPTX<60>, hasSM<30>]>;
}

defm VOTE_SYNC_ALL : VOTE_SYNC<Int1Regs, "all.pred", int_nvvm_vote_all_sync>;
defm VOTE_SYNC_ANY : VOTE_SYNC<Int1Regs, "any.pred", int_nvvm_vote_any_sync>;
defm VOTE_SYNC_UNI : VOTE_SYNC<Int1Regs, "uni.pred", int_nvvm_vote_uni_sync>;
defm VOTE_SYNC_BALLOT : VOTE_SYNC<Int32Regs, "ballot.b32", int_nvvm_vote_ballot_sync>;

// elect.sync
def INT_ELECT_SYNC_I : NVPTXInst<(outs Int32Regs:$dest, Int1Regs:$pred), (ins i32imm:$mask),
            "elect.sync \t$dest|$pred, $mask;",
            [(set i32:$dest, i1:$pred, (int_nvvm_elect_sync imm:$mask))]>,
            Requires<[hasPTX<80>, hasSM<90>]>;
def INT_ELECT_SYNC_R : NVPTXInst<(outs Int32Regs:$dest, Int1Regs:$pred), (ins Int32Regs:$mask),
            "elect.sync \t$dest|$pred, $mask;",
            [(set i32:$dest, i1:$pred, (int_nvvm_elect_sync i32:$mask))]>,
            Requires<[hasPTX<80>, hasSM<90>]>;

multiclass MATCH_ANY_SYNC<NVPTXRegClass regclass, string ptxtype, Intrinsic IntOp,
                          Operand ImmOp> {
  def ii : NVPTXInst<(outs Int32Regs:$dest), (ins i32imm:$mask, ImmOp:$value),
              "match.any.sync." # ptxtype # " \t$dest, $value, $mask;",
              [(set i32:$dest, (IntOp imm:$mask, imm:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def ir : NVPTXInst<(outs Int32Regs:$dest), (ins Int32Regs:$mask, ImmOp:$value),
              "match.any.sync." # ptxtype # " \t$dest, $value, $mask;",
              [(set i32:$dest, (IntOp i32:$mask, imm:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def ri : NVPTXInst<(outs Int32Regs:$dest), (ins i32imm:$mask, regclass:$value),
              "match.any.sync." # ptxtype # " \t$dest, $value, $mask;",
              [(set i32:$dest, (IntOp imm:$mask, regclass:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def rr : NVPTXInst<(outs Int32Regs:$dest), (ins Int32Regs:$mask, regclass:$value),
              "match.any.sync." # ptxtype # " \t$dest, $value, $mask;",
              [(set i32:$dest, (IntOp i32:$mask, regclass:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
}

// activemask.b32
def ACTIVEMASK : NVPTXInst<(outs Int32Regs:$dest), (ins),
                    "activemask.b32 \t$dest;",
                    [(set i32:$dest, (int_nvvm_activemask))]>,
                 Requires<[hasPTX<62>, hasSM<30>]>;

defm MATCH_ANY_SYNC_32 : MATCH_ANY_SYNC<Int32Regs, "b32", int_nvvm_match_any_sync_i32,
                                        i32imm>;
defm MATCH_ANY_SYNC_64 : MATCH_ANY_SYNC<Int64Regs, "b64", int_nvvm_match_any_sync_i64,
                                        i64imm>;

multiclass MATCH_ALLP_SYNC<NVPTXRegClass regclass, string ptxtype, Intrinsic IntOp,
                          Operand ImmOp> {
  def ii : NVPTXInst<(outs Int32Regs:$dest, Int1Regs:$pred),
                     (ins i32imm:$mask, ImmOp:$value),
              "match.all.sync." # ptxtype # " \t$dest|$pred, $value, $mask;",
              [(set i32:$dest, i1:$pred, (IntOp imm:$mask, imm:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def ir : NVPTXInst<(outs Int32Regs:$dest, Int1Regs:$pred),
                     (ins Int32Regs:$mask, ImmOp:$value),
              "match.all.sync." # ptxtype # " \t$dest|$pred, $value, $mask;",
              [(set i32:$dest, i1:$pred, (IntOp i32:$mask, imm:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def ri : NVPTXInst<(outs Int32Regs:$dest, Int1Regs:$pred),
                     (ins i32imm:$mask, regclass:$value),
              "match.all.sync." # ptxtype # " \t$dest|$pred, $value, $mask;",
              [(set i32:$dest, i1:$pred, (IntOp imm:$mask, regclass:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
  def rr : NVPTXInst<(outs Int32Regs:$dest, Int1Regs:$pred),
                     (ins Int32Regs:$mask, regclass:$value),
              "match.all.sync." # ptxtype # " \t$dest|$pred, $value, $mask;",
              [(set i32:$dest, i1:$pred, (IntOp i32:$mask, regclass:$value))]>,
           Requires<[hasPTX<60>, hasSM<70>]>;
}
defm MATCH_ALLP_SYNC_32 : MATCH_ALLP_SYNC<Int32Regs, "b32", int_nvvm_match_all_sync_i32p,
                                         i32imm>;
defm MATCH_ALLP_SYNC_64 : MATCH_ALLP_SYNC<Int64Regs, "b64", int_nvvm_match_all_sync_i64p,
                                         i64imm>;

multiclass REDUX_SYNC<string BinOp, string PTXType, Intrinsic Intrin> {
  def : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$src, Int32Regs:$mask),
          "redux.sync." # BinOp # "." # PTXType # " $dst, $src, $mask;",
          [(set i32:$dst, (Intrin i32:$src, Int32Regs:$mask))]>,
        Requires<[hasPTX<70>, hasSM<80>]>;
}

defm REDUX_SYNC_UMIN : REDUX_SYNC<"min", "u32", int_nvvm_redux_sync_umin>;
defm REDUX_SYNC_UMAX : REDUX_SYNC<"max", "u32", int_nvvm_redux_sync_umax>;
defm REDUX_SYNC_ADD : REDUX_SYNC<"add", "s32", int_nvvm_redux_sync_add>;
defm REDUX_SYNC_MIN : REDUX_SYNC<"min", "s32", int_nvvm_redux_sync_min>;
defm REDUX_SYNC_MAX : REDUX_SYNC<"max", "s32", int_nvvm_redux_sync_max>;
defm REDUX_SYNC_AND : REDUX_SYNC<"and", "b32", int_nvvm_redux_sync_and>;
defm REDUX_SYNC_XOR : REDUX_SYNC<"xor", "b32", int_nvvm_redux_sync_xor>;
defm REDUX_SYNC_OR : REDUX_SYNC<"or", "b32", int_nvvm_redux_sync_or>;

} // isConvergent = true

//-----------------------------------
// Explicit Memory Fence Functions
//-----------------------------------
class MEMBAR<string StrOp, Intrinsic IntOP> :
              NVPTXInst<(outs), (ins),
            StrOp, [(IntOP)]>;

def INT_MEMBAR_CTA : MEMBAR<"membar.cta;", int_nvvm_membar_cta>;
def INT_MEMBAR_GL  : MEMBAR<"membar.gl;",  int_nvvm_membar_gl>;
def INT_MEMBAR_SYS : MEMBAR<"membar.sys;", int_nvvm_membar_sys>;

def INT_FENCE_SC_CLUSTER:
       MEMBAR<"fence.sc.cluster;", int_nvvm_fence_sc_cluster>,
       Requires<[hasPTX<78>, hasSM<90>]>;

// Proxy fence (uni-directional)
// fence.proxy.tensormap.release variants

class FENCE_PROXY_TENSORMAP_GENERIC_RELEASE<string Scope, Intrinsic Intr> :
        NVPTXInst<(outs), (ins),
                  "fence.proxy.tensormap::generic.release." # Scope # ";", [(Intr)]>,
        Requires<[hasPTX<83>, hasSM<90>]>;

def INT_FENCE_PROXY_TENSORMAP_GENERIC_RELEASE_CTA:
      FENCE_PROXY_TENSORMAP_GENERIC_RELEASE<"cta",
        int_nvvm_fence_proxy_tensormap_generic_release_cta>;
def INT_FENCE_PROXY_TENSORMAP_GENERIC_RELEASE_CLUSTER:
      FENCE_PROXY_TENSORMAP_GENERIC_RELEASE<"cluster",
        int_nvvm_fence_proxy_tensormap_generic_release_cluster>;
def INT_FENCE_PROXY_TENSORMAP_GENERIC_RELEASE_GPU:
      FENCE_PROXY_TENSORMAP_GENERIC_RELEASE<"gpu",
        int_nvvm_fence_proxy_tensormap_generic_release_gpu>;
def INT_FENCE_PROXY_TENSORMAP_GENERIC_RELEASE_SYS:
      FENCE_PROXY_TENSORMAP_GENERIC_RELEASE<"sys",
        int_nvvm_fence_proxy_tensormap_generic_release_sys>;

// fence.proxy.tensormap.acquire variants

class FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE<string Scope, Intrinsic Intr> :
        NVPTXInst<(outs), (ins Int64Regs:$addr),
                  "fence.proxy.tensormap::generic.acquire." # Scope # " [$addr], 128;",
                  [(Intr i64:$addr, (i32 128))]>,
        Requires<[hasPTX<83>, hasSM<90>]>;

def INT_FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE_CTA :
      FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE<"cta",
        int_nvvm_fence_proxy_tensormap_generic_acquire_cta>;
def INT_FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE_CLUSTER :
      FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE<"cluster",
        int_nvvm_fence_proxy_tensormap_generic_acquire_cluster>;
def INT_FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE_GPU :
      FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE<"gpu",
        int_nvvm_fence_proxy_tensormap_generic_acquire_gpu>;
def INT_FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE_SYS :
      FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE<"sys",
        int_nvvm_fence_proxy_tensormap_generic_acquire_sys>;

//-----------------------------------
// Async Copy Functions
//-----------------------------------

multiclass CP_ASYNC_MBARRIER_ARRIVE<string NoInc, string AddrSpace, Intrinsic Intrin> {
  def _32 : NVPTXInst<(outs), (ins Int32Regs:$addr),
            !strconcat("cp.async.mbarrier.arrive", NoInc, AddrSpace, ".b64 [$addr];"),
            [(Intrin i32:$addr)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : NVPTXInst<(outs), (ins Int64Regs:$addr),
            !strconcat("cp.async.mbarrier.arrive", NoInc, AddrSpace, ".b64 [$addr];"),
            [(Intrin i64:$addr)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm CP_ASYNC_MBARRIER_ARRIVE :
  CP_ASYNC_MBARRIER_ARRIVE<"", "", int_nvvm_cp_async_mbarrier_arrive>;
defm CP_ASYNC_MBARRIER_ARRIVE_SHARED :
  CP_ASYNC_MBARRIER_ARRIVE<"", ".shared", int_nvvm_cp_async_mbarrier_arrive_shared>;
defm CP_ASYNC_MBARRIER_ARRIVE_NOINC :
  CP_ASYNC_MBARRIER_ARRIVE<".noinc", "", int_nvvm_cp_async_mbarrier_arrive_noinc>;
defm CP_ASYNC_MBARRIER_ARRIVE_NOINC_SHARED :
  CP_ASYNC_MBARRIER_ARRIVE<".noinc", ".shared", int_nvvm_cp_async_mbarrier_arrive_noinc_shared>;

multiclass CP_ASYNC_SHARED_GLOBAL_I<string cc, string cpsize, Intrinsic Intrin, Intrinsic IntrinS> {
  def _32 : NVPTXInst<(outs), (ins Int32Regs:$dst, Int32Regs:$src),
            !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ";"),
            [(Intrin i32:$dst, i32:$src)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : NVPTXInst<(outs), (ins Int64Regs:$dst, Int64Regs:$src),
            !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ";"),
            [(Intrin i64:$dst, i64:$src)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  // Variant with src_size parameter
  def _32s : NVPTXInst<(outs), (ins Int32Regs:$dst, Int32Regs:$src, Int32Regs:$src_size),
             !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ", $src_size;"),
             [(IntrinS i32:$dst, i32:$src, i32:$src_size)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _32si: NVPTXInst<(outs), (ins Int32Regs:$dst, Int32Regs:$src, i32imm:$src_size),
             !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ", $src_size;"),
             [(IntrinS i32:$dst, i32:$src, imm:$src_size)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64s : NVPTXInst<(outs), (ins Int64Regs:$dst, Int64Regs:$src, Int32Regs:$src_size),
             !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ", $src_size;"),
             [(IntrinS i64:$dst, i64:$src, i32:$src_size)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64si: NVPTXInst<(outs), (ins Int64Regs:$dst, Int64Regs:$src, i32imm:$src_size),
             !strconcat("cp.async.", cc, ".shared.global [$dst], [$src], ", cpsize, ", $src_size;"),
             [(IntrinS i64:$dst, i64:$src, imm:$src_size)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm CP_ASYNC_CA_SHARED_GLOBAL_4 :
  CP_ASYNC_SHARED_GLOBAL_I<"ca", "4", int_nvvm_cp_async_ca_shared_global_4,
                                      int_nvvm_cp_async_ca_shared_global_4_s>;

defm CP_ASYNC_CA_SHARED_GLOBAL_8 :
  CP_ASYNC_SHARED_GLOBAL_I<"ca", "8", int_nvvm_cp_async_ca_shared_global_8,
                                      int_nvvm_cp_async_ca_shared_global_8_s>;

defm CP_ASYNC_CA_SHARED_GLOBAL_16 :
  CP_ASYNC_SHARED_GLOBAL_I<"ca", "16", int_nvvm_cp_async_ca_shared_global_16,
                                       int_nvvm_cp_async_ca_shared_global_16_s>;

defm CP_ASYNC_CG_SHARED_GLOBAL_16 :
  CP_ASYNC_SHARED_GLOBAL_I<"cg", "16", int_nvvm_cp_async_cg_shared_global_16,
                                       int_nvvm_cp_async_cg_shared_global_16_s>;

def CP_ASYNC_COMMIT_GROUP :
  NVPTXInst<(outs), (ins), "cp.async.commit_group;", [(int_nvvm_cp_async_commit_group)]>,
  Requires<[hasPTX<70>, hasSM<80>]>;

def CP_ASYNC_WAIT_GROUP :
  NVPTXInst<(outs), (ins i32imm:$n), "cp.async.wait_group $n;",
  [(int_nvvm_cp_async_wait_group timm:$n)]>,
  Requires<[hasPTX<70>, hasSM<80>]>;

def CP_ASYNC_WAIT_ALL :
  NVPTXInst<(outs), (ins), "cp.async.wait_all;",
  [(int_nvvm_cp_async_wait_all)]>,
  Requires<[hasPTX<70>, hasSM<80>]>;

// cp.async.bulk variants of the commit/wait group
def CP_ASYNC_BULK_COMMIT_GROUP :
  NVPTXInst<(outs), (ins), "cp.async.bulk.commit_group;",
  [(int_nvvm_cp_async_bulk_commit_group)]>,
  Requires<[hasPTX<80>, hasSM<90>]>;

def CP_ASYNC_BULK_WAIT_GROUP :
  NVPTXInst<(outs), (ins i32imm:$n), "cp.async.bulk.wait_group $n;",
  [(int_nvvm_cp_async_bulk_wait_group timm:$n)]>,
  Requires<[hasPTX<80>, hasSM<90>]>;

def CP_ASYNC_BULK_WAIT_GROUP_READ :
  NVPTXInst<(outs), (ins i32imm:$n), "cp.async.bulk.wait_group.read $n;",
  [(int_nvvm_cp_async_bulk_wait_group_read timm:$n)]>,
  Requires<[hasPTX<80>, hasSM<90>]>;

//------------------------------
// TMA Async Bulk Copy Functions
//------------------------------

class CpAsyncBulkStr<bit mc, bit ch> {
  // Shared to Global memory
  string S2G = "cp.async.bulk.global.shared::cta.bulk_group"
               # !if(ch, ".L2::cache_hint", "");

  // Global to Shared cluster memory
  string G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
               # !if(mc, ".multicast::cluster", "")
               # !if(ch, ".L2::cache_hint", "");

  // Shared CTA to Cluster memory
  string C2C = "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes";
}

multiclass CP_ASYNC_BULK_S2G<NVPTXRegClass rc> {
  def NAME: NVPTXInst<(outs),
            (ins Int64Regs:$dst, rc:$src, Int32Regs:$size),
            !strconcat(CpAsyncBulkStr<0, 0>.S2G, " [$dst], [$src], $size;"), []>,
            Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _CH: NVPTXInst<(outs),
                  (ins Int64Regs:$dst, rc:$src, Int32Regs:$size, Int64Regs:$ch),
                  !strconcat(CpAsyncBulkStr<0, 1>.S2G, " [$dst], [$src], $size, $ch;"), []>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
}
defm CP_ASYNC_BULK_S2G : CP_ASYNC_BULK_S2G<Int64Regs>;
defm CP_ASYNC_BULK_S2G_SHARED32 : CP_ASYNC_BULK_S2G<Int32Regs>;

multiclass CP_ASYNC_BULK_G2S<NVPTXRegClass rc> {
  def NAME: NVPTXInst<(outs),
            (ins rc:$dst, rc:$mbar, Int64Regs:$src, Int32Regs:$size),
            !strconcat(CpAsyncBulkStr<0, 0>.G2S, " [$dst], [$src], $size, [$mbar];"), []>,
            Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _MC: NVPTXInst<(outs),
                  (ins rc:$dst, rc:$mbar, Int64Regs:$src, Int32Regs:$size, Int16Regs:$mc),
                  !strconcat(CpAsyncBulkStr<1, 0>.G2S, " [$dst], [$src], $size, [$mbar], $mc;"), []>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _CH: NVPTXInst<(outs),
                  (ins rc:$dst, rc:$mbar, Int64Regs:$src, Int32Regs:$size, Int64Regs:$ch),
                  !strconcat(CpAsyncBulkStr<0, 1>.G2S, " [$dst], [$src], $size, [$mbar], $ch;"), []>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _MC_CH: NVPTXInst<(outs),
                     (ins rc:$dst, rc:$mbar, Int64Regs:$src, Int32Regs:$size, Int16Regs:$mc, Int64Regs:$ch),
                     !strconcat(CpAsyncBulkStr<1, 1>.G2S, " [$dst], [$src], $size, [$mbar], $mc, $ch;"), []>,
                     Requires<[hasPTX<80>, hasSM<90>]>;
}
defm CP_ASYNC_BULK_G2S : CP_ASYNC_BULK_G2S<Int64Regs>;
defm CP_ASYNC_BULK_G2S_SHARED32 : CP_ASYNC_BULK_G2S<Int32Regs>;

multiclass CP_ASYNC_BULK_CTA_TO_CLUSTER<NVPTXRegClass rc> {
  def NAME: NVPTXInst<(outs),
            (ins rc:$dst, rc:$mbar, rc:$src, Int32Regs:$size),
            !strconcat(CpAsyncBulkStr<0, 0>.C2C, " [$dst], [$src], $size, [$mbar];"),
            [(int_nvvm_cp_async_bulk_shared_cta_to_cluster rc:$dst, rc:$mbar, rc:$src, Int32Regs:$size)]>,
            Requires<[hasPTX<80>, hasSM<90>]>;
}
defm CP_ASYNC_BULK_CTA_TO_CLUSTER : CP_ASYNC_BULK_CTA_TO_CLUSTER<Int64Regs>;
defm CP_ASYNC_BULK_CTA_TO_CLUSTER_SHARED32 : CP_ASYNC_BULK_CTA_TO_CLUSTER<Int32Regs>;

//------------------------------
// Bulk Copy Prefetch Functions
//------------------------------
def CP_ASYNC_BULK_PREFETCH : NVPTXInst<(outs),
                             (ins Int64Regs:$src, Int32Regs:$size),
                             "cp.async.bulk.prefetch.L2.global [$src], $size;", []>,
                             Requires<[hasPTX<80>, hasSM<90>]>;

def CP_ASYNC_BULK_PREFETCH_CH : NVPTXInst<(outs),
                                (ins Int64Regs:$src, Int32Regs:$size, Int64Regs:$ch),
                                "cp.async.bulk.prefetch.L2.global.L2::cache_hint [$src], $size, $ch;", []>,
                                Requires<[hasPTX<80>, hasSM<90>]>;
//-------------------------------------
// TMA Async Bulk Tensor Copy Functions
//-------------------------------------

// From Global to Shared memory (G2S)
class G2S_STRINGS<int dim, string mode, bit mc, bit ch, bit is_shared32 = 0> {
  string prefix = "cp.async.bulk.tensor";
  string dir = "shared::cluster.global";
  string completion = "mbarrier::complete_tx::bytes";
  string inst_name = prefix
                     # "." # dim # "d"
                     # "." # dir
                     # "." # mode
                     # "." # completion
                     # !if(mc, ".multicast::cluster", "")
                     # !if(ch, ".L2::cache_hint", "");
  string intr_name = "CP_ASYNC_BULK_TENSOR_G2S_"
                     # dim # "D"
                     # !if(is_shared32, "_SHARED32", "")
                     # !if(!eq(mode, "tile"), "_TILE", "_IM2COL");
}

multiclass CP_ASYNC_BULK_TENSOR_G2S_INTR<int dim, bit is_shared32, string mode> {
  defvar dims_dag = !dag(ins, !listsplat(Int32Regs, dim), !foreach(i, !range(dim), "d" # i));
  defvar dims_str = !interleave(!foreach(i, !range(dim), "$d" # i), ", ");
  defvar asm_str_default = " [$dst], [$tmap, {{" # dims_str # "}}], [$mbar]";
  defvar rc = !if(is_shared32, Int32Regs, Int64Regs);

  defvar num_im2col = !if(!ge(dim, 3), !add(dim, -2), 0);
  defvar im2col_dag = !if(!eq(mode, "im2col"),
    !dag(ins, !listsplat(Int16Regs, num_im2col), !foreach(i, !range(num_im2col), "im2col" # i)),
    (ins));
  defvar im2col_str = !interleave(!foreach(i, !range(num_im2col), "$im2col" # i), ", ");
  defvar im2col_asm_str = ", {{" # im2col_str # "}}";

  defvar asm_str = !if(!eq(mode, "im2col"),
    !strconcat(asm_str_default, im2col_asm_str), asm_str_default);

  def NAME: NVPTXInst<(outs),
            !con((ins rc:$dst, rc:$mbar, Int64Regs:$tmap), dims_dag, im2col_dag),
            !strconcat(G2S_STRINGS<dim, mode, 0, 0>.inst_name, asm_str, ";"), []>,
            Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _MC: NVPTXInst<(outs),
                  !con((ins rc:$dst, rc:$mbar, Int64Regs:$tmap), dims_dag, im2col_dag, (ins Int16Regs:$mc)),
                  !strconcat(G2S_STRINGS<dim, mode, 1, 0>.inst_name, asm_str, ", $mc;"), []>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _CH: NVPTXInst<(outs),
                  !con((ins rc:$dst, rc:$mbar, Int64Regs:$tmap), dims_dag, im2col_dag, (ins Int64Regs:$ch)),
                  !strconcat(G2S_STRINGS<dim, mode, 0, 1>.inst_name, asm_str, ", $ch;"), []>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _MC_CH: NVPTXInst<(outs),
                     !con((ins rc:$dst, rc:$mbar, Int64Regs:$tmap), dims_dag, im2col_dag, (ins Int16Regs:$mc, Int64Regs:$ch)),
                     !strconcat(G2S_STRINGS<dim, mode, 1, 1>.inst_name, asm_str, ", $mc, $ch;"), []>,
                     Requires<[hasPTX<80>, hasSM<90>]>;
}

foreach dim = [1, 2, 3, 4, 5] in {
  foreach shared32 = [true, false] in {
    foreach mode = !if(!ge(dim, 3), ["tile", "im2col"], ["tile"]) in {
      defm G2S_STRINGS<dim, mode, 0, 0, shared32>.intr_name :
        CP_ASYNC_BULK_TENSOR_G2S_INTR<dim, shared32, mode>;
    }
  }
}

// From Shared to Global memory (S2G)
class S2G_STRINGS<int dim, string mode, bit ch,
                  bit is_shared32 = 0, bit is_reduce = 0> {
  string dir = "global.shared::cta";
  string completion = "bulk_group";
  string inst_name = !if(is_reduce, "cp.reduce", "cp")
                     # ".async.bulk.tensor"
                     # "." # dim # "d"
                     # "." # dir
                     # "." # mode
                     # "." # completion
                     # !if(ch, ".L2::cache_hint", "");
  string intr_name = "CP_ASYNC_BULK_TENSOR_"
                     # !if(is_reduce, "RED_", "S2G_")
                     # dim # "D"
                     # !if(is_shared32, "_SHARED32", "")
                     # !if(!eq(mode, "tile"), "_TILE", "_IM2COL");
}

multiclass CP_ASYNC_BULK_TENSOR_S2G_INTR<int dim, bit shared32, string mode> {
  defvar dims_dag = !dag(ins, !listsplat(Int32Regs, dim), !foreach(i, !range(dim), "d" # i));
  defvar dims_str = !interleave(!foreach(i, !range(dim), "$d" # i), ", ");
  defvar asm_str = " [$tmap, {{" # dims_str # "}}], [$src]";
  defvar rc = !if(shared32, Int32Regs, Int64Regs);

  def NAME: NVPTXInst<(outs),
            !con((ins rc:$src, Int64Regs:$tmap), dims_dag),
            !strconcat(S2G_STRINGS<dim, mode, 0>.inst_name, asm_str, ";"), []>,
            Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _CH: NVPTXInst<(outs),
                  !con((ins rc:$src, Int64Regs:$tmap), dims_dag, (ins Int64Regs:$ch)),
                  !strconcat(S2G_STRINGS<dim, mode, 1>.inst_name, asm_str, ", $ch;"), []>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
}

def TMAReductionFlags : Operand<i32> {
  let PrintMethod = "printTmaReductionMode";
}

// TMA Copy from Shared to Global memory with Reduction
multiclass CP_ASYNC_BULK_TENSOR_REDUCE_INTR<int dim, bit shared32, string mode> {
  defvar dims_dag = !dag(ins, !listsplat(Int32Regs, dim), !foreach(i, !range(dim), "d" # i));
  defvar dims_str = !interleave(!foreach(i, !range(dim), "$d" # i), ", ");
  defvar asm_str = " [$tmap, {{" # dims_str # "}}], [$src]";
  defvar rc = !if(shared32, Int32Regs, Int64Regs);

  defvar prefix = "cp.reduce.async.bulk.tensor" # "." # dim # "d" # ".global.shared::cta";
  defvar suffix = "." # mode # ".bulk_group";

  def NAME: NVPTXInst<(outs),
            !con((ins rc:$src, Int64Regs:$tmap), dims_dag, (ins TMAReductionFlags:$red_op)),
            !strconcat(prefix, "${red_op}", suffix, asm_str, ";"), []>,
            Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _CH: NVPTXInst<(outs),
                  !con((ins rc:$src, Int64Regs:$tmap), dims_dag, (ins Int64Regs:$ch, TMAReductionFlags:$red_op)),
                  !strconcat(prefix, "${red_op}", suffix, ".L2::cache_hint", asm_str, ", $ch;"), []>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
}

foreach dim = [1, 2, 3, 4, 5] in {
  foreach shared32 = [true, false] in {
    foreach mode = !if(!ge(dim, 3), ["tile", "im2col_no_offs"], ["tile"]) in {
      defm S2G_STRINGS<dim, mode, 0, shared32>.intr_name :
        CP_ASYNC_BULK_TENSOR_S2G_INTR<dim, shared32, mode>;
      defm S2G_STRINGS<dim, mode, 0, shared32, 1>.intr_name :
        CP_ASYNC_BULK_TENSOR_REDUCE_INTR<dim, shared32, mode>;
    }
  }
}

// TMA Prefetch from Global memory to L2 cache
class PREFETCH_STRINGS<int dim, string mode, bit ch> {
  string prefix = "cp.async.bulk.prefetch.tensor";
  string dir = "L2.global";
  string inst_name = prefix
                     # "." # dim # "d"
                     # "." # dir
                     # "." # mode
                     # !if(ch, ".L2::cache_hint", "");
  string intr_name = "CP_ASYNC_BULK_TENSOR_PREFETCH_"
                     # dim # "D"
                     # !if(!eq(mode, "tile"), "_TILE", "_IM2COL");
}

multiclass CP_ASYNC_BULK_TENSOR_PREFETCH_INTR<int dim, string mode> {
  defvar dims_dag = !dag(ins, !listsplat(Int32Regs, dim), !foreach(i, !range(dim), "d" # i));
  defvar dims_str = !interleave(!foreach(i, !range(dim), "$d" # i), ", ");
  defvar asm_str_default = " [$tmap, {{" # dims_str # "}}]";

  defvar num_im2col = !if(!ge(dim, 3), !add(dim, -2), 0);
  defvar im2col_dag = !if(!eq(mode, "im2col"),
    !dag(ins, !listsplat(Int16Regs, num_im2col), !foreach(i, !range(num_im2col), "im2col" # i)),
    (ins));
  defvar im2col_str = !interleave(!foreach(i, !range(num_im2col), "$im2col" # i), ", ");
  defvar im2col_asm_str = ", {{" # im2col_str # "}}";

  defvar asm_str = !if(!eq(mode, "im2col"),
    !strconcat(asm_str_default, im2col_asm_str), asm_str_default);

  def NAME: NVPTXInst<(outs),
            !con((ins Int64Regs:$tmap), dims_dag, im2col_dag),
            !strconcat(PREFETCH_STRINGS<dim, mode, 0>.inst_name, asm_str, ";"), []>,
            Requires<[hasPTX<80>, hasSM<90>]>;
  def NAME # _CH: NVPTXInst<(outs),
                  !con((ins Int64Regs:$tmap), dims_dag, im2col_dag, (ins Int64Regs:$ch)),
                  !strconcat(PREFETCH_STRINGS<dim, mode, 1>.inst_name, asm_str, ", $ch;"), []>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
}

foreach dim = [1, 2, 3, 4, 5] in {
  foreach mode = !if(!ge(dim, 3), ["tile", "im2col"], ["tile"]) in {
    defm PREFETCH_STRINGS<dim, mode, 0>.intr_name :
      CP_ASYNC_BULK_TENSOR_PREFETCH_INTR<dim, mode>;
  }
}

//-----------------------------------
// MBarrier Functions
//-----------------------------------

multiclass MBARRIER_INIT<string AddrSpace, Intrinsic Intrin> {
  def _32 : NVPTXInst<(outs), (ins Int32Regs:$addr, Int32Regs:$count),
           !strconcat("mbarrier.init", AddrSpace, ".b64 [$addr], $count;"),
    [(Intrin i32:$addr, i32:$count)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : NVPTXInst<(outs), (ins Int64Regs:$addr, Int32Regs:$count),
           !strconcat("mbarrier.init", AddrSpace, ".b64 [$addr], $count;"),
    [(Intrin i64:$addr, i32:$count)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_INIT : MBARRIER_INIT<"", int_nvvm_mbarrier_init>;
defm MBARRIER_INIT_SHARED : MBARRIER_INIT<".shared",
                                          int_nvvm_mbarrier_init_shared>;

multiclass MBARRIER_INVAL<string AddrSpace, Intrinsic Intrin> {
  def _32 : NVPTXInst<(outs), (ins Int32Regs:$addr),
           !strconcat("mbarrier.inval", AddrSpace, ".b64 [$addr];"),
    [(Intrin i32:$addr)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : NVPTXInst<(outs), (ins Int64Regs:$addr),
           !strconcat("mbarrier.inval", AddrSpace, ".b64 [$addr];"),
    [(Intrin i64:$addr)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_INVAL : MBARRIER_INVAL<"", int_nvvm_mbarrier_inval>;
defm MBARRIER_INVAL_SHARED : MBARRIER_INVAL<".shared",
                                            int_nvvm_mbarrier_inval_shared>;

multiclass MBARRIER_ARRIVE<string AddrSpace, Intrinsic Intrin> {
  def _32 : NVPTXInst<(outs Int64Regs:$state), (ins Int32Regs:$addr),
           !strconcat("mbarrier.arrive", AddrSpace, ".b64 $state, [$addr];"),
    [(set i64:$state, (Intrin i32:$addr))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : NVPTXInst<(outs Int64Regs:$state), (ins Int64Regs:$addr),
           !strconcat("mbarrier.arrive", AddrSpace, ".b64 $state, [$addr];"),
    [(set i64:$state, (Intrin i64:$addr))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_ARRIVE : MBARRIER_ARRIVE<"", int_nvvm_mbarrier_arrive>;
defm MBARRIER_ARRIVE_SHARED :
  MBARRIER_ARRIVE<".shared", int_nvvm_mbarrier_arrive_shared>;

multiclass MBARRIER_ARRIVE_NOCOMPLETE<string AddrSpace, Intrinsic Intrin> {
  def _32 : NVPTXInst<(outs Int64Regs:$state),
           (ins Int32Regs:$addr, Int32Regs:$count),
           !strconcat("mbarrier.arrive.noComplete", AddrSpace,
                      ".b64 $state, [$addr], $count;"),
    [(set i64:$state, (Intrin i32:$addr, i32:$count))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : NVPTXInst<(outs Int64Regs:$state),
           (ins Int64Regs:$addr, Int32Regs:$count),
           !strconcat("mbarrier.arrive.noComplete", AddrSpace,
                      ".b64 $state, [$addr], $count;"),
    [(set i64:$state, (Intrin i64:$addr, i32:$count))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_ARRIVE_NOCOMPLETE :
  MBARRIER_ARRIVE_NOCOMPLETE<"", int_nvvm_mbarrier_arrive_noComplete>;
defm MBARRIER_ARRIVE_NOCOMPLETE_SHARED :
  MBARRIER_ARRIVE_NOCOMPLETE<".shared", int_nvvm_mbarrier_arrive_noComplete_shared>;

multiclass MBARRIER_ARRIVE_DROP<string AddrSpace, Intrinsic Intrin> {
  def _32 : NVPTXInst<(outs Int64Regs:$state), (ins Int32Regs:$addr),
           !strconcat("mbarrier.arrive_drop", AddrSpace,
                      ".b64 $state, [$addr];"),
           [(set i64:$state, (Intrin i32:$addr))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : NVPTXInst<(outs Int64Regs:$state), (ins Int64Regs:$addr),
           !strconcat("mbarrier.arrive_drop", AddrSpace,
                      ".b64 $state, [$addr];"),
           [(set i64:$state, (Intrin i64:$addr))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_ARRIVE_DROP :
  MBARRIER_ARRIVE_DROP<"", int_nvvm_mbarrier_arrive_drop>;
defm MBARRIER_ARRIVE_DROP_SHARED :
  MBARRIER_ARRIVE_DROP<".shared", int_nvvm_mbarrier_arrive_drop_shared>;

multiclass MBARRIER_ARRIVE_DROP_NOCOMPLETE<string AddrSpace, Intrinsic Intrin> {
  def _32 : NVPTXInst<(outs Int64Regs:$state),
           (ins Int32Regs:$addr, Int32Regs:$count),
           !strconcat("mbarrier.arrive_drop.noComplete", AddrSpace,
                      ".b64 $state, [$addr], $count;"),
           [(set i64:$state, (Intrin i32:$addr, i32:$count))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : NVPTXInst<(outs Int64Regs:$state),
           (ins Int64Regs:$addr, Int32Regs:$count),
           !strconcat("mbarrier.arrive_drop.noComplete", AddrSpace,
                      ".b64 $state, [$addr], $count;"),
           [(set i64:$state, (Intrin i64:$addr, i32:$count))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_ARRIVE_DROP_NOCOMPLETE :
  MBARRIER_ARRIVE_DROP_NOCOMPLETE<"", int_nvvm_mbarrier_arrive_drop_noComplete>;
defm MBARRIER_ARRIVE_DROP_NOCOMPLETE_SHARED :
  MBARRIER_ARRIVE_DROP_NOCOMPLETE<".shared",
                       int_nvvm_mbarrier_arrive_drop_noComplete_shared>;

multiclass MBARRIER_TEST_WAIT<string AddrSpace, Intrinsic Intrin> {
  def _32 : NVPTXInst<(outs Int1Regs:$res), (ins Int32Regs:$addr, Int64Regs:$state),
           !strconcat("mbarrier.test_wait", AddrSpace, ".b64 $res, [$addr], $state;"),
           [(set i1:$res, (Intrin i32:$addr, i64:$state))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _64 : NVPTXInst<(outs Int1Regs:$res), (ins Int64Regs:$addr, Int64Regs:$state),
           !strconcat("mbarrier.test_wait", AddrSpace, ".b64 $res, [$addr], $state;"),
           [(set i1:$res, (Intrin i64:$addr, i64:$state))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
}

defm MBARRIER_TEST_WAIT :
  MBARRIER_TEST_WAIT<"", int_nvvm_mbarrier_test_wait>;
defm MBARRIER_TEST_WAIT_SHARED :
  MBARRIER_TEST_WAIT<".shared", int_nvvm_mbarrier_test_wait_shared>;

class MBARRIER_PENDING_COUNT<Intrinsic Intrin> :
           NVPTXInst<(outs Int32Regs:$res), (ins Int64Regs:$state),
           "mbarrier.pending_count.b64 $res, $state;",
           [(set i32:$res, (Intrin i64:$state))]>,
    Requires<[hasPTX<70>, hasSM<80>]>;

def MBARRIER_PENDING_COUNT :
  MBARRIER_PENDING_COUNT<int_nvvm_mbarrier_pending_count>;

//-----------------------------------
// Math Functions
//-----------------------------------

// Map min(1.0, max(0.0, x)) to sat(x)
// Note that max(0.0, min(x, 1.0)) cannot be mapped to sat(x) because when x is
// NaN
// max(0.0, min(x, 1.0)) is 1.0 while sat(x) is 0.
// Same story for fmax, fmin.

def : Pat<(int_nvvm_fmin_f immFloat1,
            (int_nvvm_fmax_f immFloat0, f32:$a)),
          (CVT_f32_f32 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_f immFloat1,
            (int_nvvm_fmax_f f32:$a, immFloat0)),
          (CVT_f32_f32 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_f
            (int_nvvm_fmax_f immFloat0, f32:$a), immFloat1),
          (CVT_f32_f32 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_f
            (int_nvvm_fmax_f f32:$a, immFloat0), immFloat1),
          (CVT_f32_f32 $a, CvtSAT)>;

def : Pat<(int_nvvm_fmin_d immDouble1,
            (int_nvvm_fmax_d immDouble0, f64:$a)),
          (CVT_f64_f64 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_d immDouble1,
            (int_nvvm_fmax_d f64:$a, immDouble0)),
          (CVT_f64_f64 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_d
            (int_nvvm_fmax_d immDouble0, f64:$a), immDouble1),
          (CVT_f64_f64 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_d
            (int_nvvm_fmax_d f64:$a, immDouble0), immDouble1),
          (CVT_f64_f64 $a, CvtSAT)>;


// We need a full string for OpcStr here because we need to deal with case like
// INT_PTX_RECIP.
class F_MATH_1<string OpcStr, NVPTXRegClass target_regclass,
  NVPTXRegClass src_regclass, Intrinsic IntOP, list<Predicate> Preds = []>
            : NVPTXInst<(outs target_regclass:$dst), (ins src_regclass:$src0),
            OpcStr,
        [(set target_regclass:$dst, (IntOP src_regclass:$src0))]>,
        Requires<Preds>;

// We need a full string for OpcStr here because we need to deal with the case
// like INT_PTX_NATIVE_POWR_F.
class F_MATH_2<string OpcStr, NVPTXRegClass t_regclass,
  NVPTXRegClass s0_regclass, NVPTXRegClass s1_regclass, Intrinsic IntOP,
  list<Predicate> Preds = []>
            : NVPTXInst<(outs t_regclass:$dst),
              (ins s0_regclass:$src0, s1_regclass:$src1),
            OpcStr,
        [(set t_regclass:$dst, (IntOP s0_regclass:$src0, s1_regclass:$src1))]>,
        Requires<Preds>;

class F_MATH_3<string OpcStr, NVPTXRegClass t_regclass,
  NVPTXRegClass s0_regclass, NVPTXRegClass s1_regclass,
  NVPTXRegClass s2_regclass, Intrinsic IntOP, list<Predicate> Preds = []>
            : NVPTXInst<(outs t_regclass:$dst),
              (ins s0_regclass:$src0, s1_regclass:$src1, s2_regclass:$src2),
            OpcStr,
        [(set t_regclass:$dst,
          (IntOP s0_regclass:$src0, s1_regclass:$src1, s2_regclass:$src2))]>,
          Requires<Preds>;

//
// MISC
//

def INT_NVVM_PRMT : F_MATH_3<"prmt.b32 \t$dst, $src0, $src1, $src2;", Int32Regs,
  Int32Regs, Int32Regs, Int32Regs, int_nvvm_prmt>;

def INT_NVVM_NANOSLEEP_I : NVPTXInst<(outs), (ins i32imm:$i), "nanosleep.u32 \t$i;",
                             [(int_nvvm_nanosleep imm:$i)]>,
        Requires<[hasPTX<63>, hasSM<70>]>;
def INT_NVVM_NANOSLEEP_R : NVPTXInst<(outs), (ins Int32Regs:$i), "nanosleep.u32 \t$i;",
                             [(int_nvvm_nanosleep i32:$i)]>,
        Requires<[hasPTX<63>, hasSM<70>]>;
//
// Min Max
//

def INT_NVVM_FMIN_F : F_MATH_2<"min.f32 \t$dst, $src0, $src1;", Float32Regs,
  Float32Regs, Float32Regs, int_nvvm_fmin_f>;
def INT_NVVM_FMIN_FTZ_F : F_MATH_2<"min.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmin_ftz_f>;
def INT_NVVM_FMIN_NAN_F : F_MATH_2<"min.NaN.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmin_nan_f,
  [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_FMIN_FTZ_NAN_F : F_MATH_2<"min.ftz.NaN.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmin_ftz_nan_f,
  [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_FMIN_XORSIGN_ABS_F :
  F_MATH_2<"min.xorsign.abs.f32 \t$dst, $src0, $src1;",
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmin_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMIN_FTZ_XORSIGN_ABS_F :
  F_MATH_2<"min.ftz.xorsign.abs.f32 \t$dst, $src0, $src1;",
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmin_ftz_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMIN_NAN_XORSIGN_ABS_F :
  F_MATH_2<"min.NaN.xorsign.abs.f32 \t$dst, $src0, $src1;",
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmin_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMIN_FTZ_NAN_XORSIGN_ABS_F :
  F_MATH_2<"min.ftz.NaN.xorsign.abs.f32 \t$dst, $src0, $src1;",
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmin_ftz_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;

def INT_NVVM_FMAX_F : F_MATH_2<"max.f32 \t$dst, $src0, $src1;", Float32Regs,
  Float32Regs, Float32Regs, int_nvvm_fmax_f>;
def INT_NVVM_FMAX_FTZ_F : F_MATH_2<"max.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmax_ftz_f>;
def INT_NVVM_FMAX_NAN_F : F_MATH_2<"max.NaN.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmax_nan_f,
  [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_FMAX_FTZ_NAN_F : F_MATH_2<"max.ftz.NaN.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmax_ftz_nan_f,
  [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_FMAX_XORSIGN_ABS_F :
  F_MATH_2<"max.xorsign.abs.f32 \t$dst, $src0, $src1;",
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmax_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMAX_FTZ_XORSIGN_ABS_F :
  F_MATH_2<"max.ftz.xorsign.abs.f32 \t$dst, $src0, $src1;",
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmax_ftz_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMAX_NAN_XORSIGN_ABS_F :
  F_MATH_2<"max.NaN.xorsign.abs.f32 \t$dst, $src0, $src1;",
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmax_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMAX_FTZ_NAN_XORSIGN_ABS_F :
  F_MATH_2<"max.ftz.NaN.xorsign.abs.f32 \t$dst, $src0, $src1;",
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmax_ftz_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;

def INT_NVVM_FMIN_D : F_MATH_2<"min.f64 \t$dst, $src0, $src1;", Float64Regs,
  Float64Regs, Float64Regs, int_nvvm_fmin_d>;
def INT_NVVM_FMAX_D : F_MATH_2<"max.f64 \t$dst, $src0, $src1;", Float64Regs,
  Float64Regs, Float64Regs, int_nvvm_fmax_d>;

//
// Min Max f16, f16x2, bf16, bf16x2
//

class MIN_MAX_TUPLE<string V, Intrinsic I, NVPTXRegClass RC,
                    list<Predicate> Preds = [hasPTX<70>, hasSM<80>]> {
  string Variant = V;
  Intrinsic Intr = I;
  NVPTXRegClass RegClass = RC;
  list<Predicate> Predicates = Preds;
}

multiclass MIN_MAX<string IntName> {
  foreach P = [
    MIN_MAX_TUPLE<"_f16", !if(!eq(IntName, "min"), int_nvvm_fmin_f16,
      int_nvvm_fmax_f16), Int16Regs>,
    MIN_MAX_TUPLE<"_ftz_f16", !if(!eq(IntName, "min"), int_nvvm_fmin_ftz_f16,
      int_nvvm_fmax_ftz_f16), Int16Regs>,
    MIN_MAX_TUPLE<"_NaN_f16", !if(!eq(IntName, "min"), int_nvvm_fmin_nan_f16,
      int_nvvm_fmax_nan_f16), Int16Regs>,
    MIN_MAX_TUPLE<"_ftz_NaN_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_nan_f16, int_nvvm_fmax_ftz_nan_f16), Int16Regs>,
    MIN_MAX_TUPLE<"_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_xorsign_abs_f16, int_nvvm_fmax_xorsign_abs_f16),
      Int16Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_xorsign_abs_f16, int_nvvm_fmax_ftz_xorsign_abs_f16),
      Int16Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_xorsign_abs_f16, int_nvvm_fmax_nan_xorsign_abs_f16),
      Int16Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_NaN_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_nan_xorsign_abs_f16,
      int_nvvm_fmax_ftz_nan_xorsign_abs_f16), Int16Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_f16x2", !if(!eq(IntName, "min"), int_nvvm_fmin_f16x2,
      int_nvvm_fmax_f16x2), Int32Regs>,
    MIN_MAX_TUPLE<"_ftz_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_f16x2, int_nvvm_fmax_ftz_f16x2), Int32Regs>,
    MIN_MAX_TUPLE<"_NaN_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_f16x2, int_nvvm_fmax_nan_f16x2), Int32Regs>,
    MIN_MAX_TUPLE<"_ftz_NaN_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_nan_f16x2, int_nvvm_fmax_ftz_nan_f16x2), Int32Regs>,
    MIN_MAX_TUPLE<"_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_xorsign_abs_f16x2, int_nvvm_fmax_xorsign_abs_f16x2),
      Int32Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_xorsign_abs_f16x2, int_nvvm_fmax_ftz_xorsign_abs_f16x2),
      Int32Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_xorsign_abs_f16x2, int_nvvm_fmax_nan_xorsign_abs_f16x2),
      Int32Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_NaN_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_nan_xorsign_abs_f16x2,
      int_nvvm_fmax_ftz_nan_xorsign_abs_f16x2),
      Int32Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_bf16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_bf16, int_nvvm_fmax_bf16), Int16Regs>,
    MIN_MAX_TUPLE<"_NaN_bf16", !if(!eq(IntName, "min"), int_nvvm_fmin_nan_bf16,
      int_nvvm_fmax_nan_bf16), Int16Regs>,
    MIN_MAX_TUPLE<"_xorsign_abs_bf16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_xorsign_abs_bf16, int_nvvm_fmax_xorsign_abs_bf16),
      Int16Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_bf16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_xorsign_abs_bf16, int_nvvm_fmax_nan_xorsign_abs_bf16),
      Int16Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_bf16x2", !if(!eq(IntName, "min"), int_nvvm_fmin_bf16x2,
      int_nvvm_fmax_bf16x2), Int32Regs>,
    MIN_MAX_TUPLE<"_NaN_bf16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_bf16x2, int_nvvm_fmax_nan_bf16x2), Int32Regs>,
    MIN_MAX_TUPLE<"_xorsign_abs_bf16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_xorsign_abs_bf16x2, int_nvvm_fmax_xorsign_abs_bf16x2),
      Int32Regs, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_bf16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_xorsign_abs_bf16x2,
      int_nvvm_fmax_nan_xorsign_abs_bf16x2),
      Int32Regs, [hasPTX<72>, hasSM<86>]>] in {
        def P.Variant : F_MATH_2<!strconcat(
          IntName, !subst("_", ".", P.Variant), " \t$dst, $src0, $src1;"),
          P.RegClass, P.RegClass, P.RegClass, P.Intr, P.Predicates>;
  }
}

defm INT_NVVM_FMIN : MIN_MAX<"min">;
defm INT_NVVM_FMAN : MIN_MAX<"max">;

//
// Multiplication
//

def INT_NVVM_MULHI_S : F_MATH_2<"mul.hi.s16 \t$dst, $src0, $src1;", Int16Regs,
  Int16Regs, Int16Regs, int_nvvm_mulhi_s>;
def INT_NVVM_MULHI_US : F_MATH_2<"mul.hi.u16 \t$dst, $src0, $src1;", Int16Regs,
  Int16Regs, Int16Regs, int_nvvm_mulhi_us>;
def INT_NVVM_MULHI_I : F_MATH_2<"mul.hi.s32 \t$dst, $src0, $src1;", Int32Regs,
  Int32Regs, Int32Regs, int_nvvm_mulhi_i>;
def INT_NVVM_MULHI_UI : F_MATH_2<"mul.hi.u32 \t$dst, $src0, $src1;", Int32Regs,
  Int32Regs, Int32Regs, int_nvvm_mulhi_ui>;
def INT_NVVM_MULHI_LL : F_MATH_2<"mul.hi.s64 \t$dst, $src0, $src1;", Int64Regs,
  Int64Regs, Int64Regs, int_nvvm_mulhi_ll>;
def INT_NVVM_MULHI_ULL : F_MATH_2<"mul.hi.u64 \t$dst, $src0, $src1;", Int64Regs,
  Int64Regs, Int64Regs, int_nvvm_mulhi_ull>;

def INT_NVVM_MUL_RN_FTZ_F : F_MATH_2<"mul.rn.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rn_ftz_f>;
def INT_NVVM_MUL_RN_F : F_MATH_2<"mul.rn.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rn_f>;
def INT_NVVM_MUL_RZ_FTZ_F : F_MATH_2<"mul.rz.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rz_ftz_f>;
def INT_NVVM_MUL_RZ_F : F_MATH_2<"mul.rz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rz_f>;
def INT_NVVM_MUL_RM_FTZ_F : F_MATH_2<"mul.rm.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rm_ftz_f>;
def INT_NVVM_MUL_RM_F : F_MATH_2<"mul.rm.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rm_f>;
def INT_NVVM_MUL_RP_FTZ_F : F_MATH_2<"mul.rp.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rp_ftz_f>;
def INT_NVVM_MUL_RP_F : F_MATH_2<"mul.rp.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rp_f>;

def INT_NVVM_MUL_RN_D : F_MATH_2<"mul.rn.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_mul_rn_d>;
def INT_NVVM_MUL_RZ_D : F_MATH_2<"mul.rz.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_mul_rz_d>;
def INT_NVVM_MUL_RM_D : F_MATH_2<"mul.rm.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_mul_rm_d>;
def INT_NVVM_MUL_RP_D : F_MATH_2<"mul.rp.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_mul_rp_d>;

def INT_NVVM_MUL24_I : F_MATH_2<"mul24.lo.s32 \t$dst, $src0, $src1;",
  Int32Regs, Int32Regs, Int32Regs, int_nvvm_mul24_i>;
def INT_NVVM_MUL24_UI : F_MATH_2<"mul24.lo.u32 \t$dst, $src0, $src1;",
  Int32Regs, Int32Regs, Int32Regs, int_nvvm_mul24_ui>;

//
// Div
//

def INT_NVVM_DIV_APPROX_FTZ_F
  : F_MATH_2<"div.approx.ftz.f32 \t$dst, $src0, $src1;", Float32Regs,
    Float32Regs, Float32Regs, int_nvvm_div_approx_ftz_f>;
def INT_NVVM_DIV_APPROX_F : F_MATH_2<"div.approx.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_approx_f>;

def INT_NVVM_DIV_RN_FTZ_F : F_MATH_2<"div.rn.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rn_ftz_f>;
def INT_NVVM_DIV_RN_F     : F_MATH_2<"div.rn.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rn_f>;
def INT_NVVM_DIV_RZ_FTZ_F : F_MATH_2<"div.rz.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rz_ftz_f>;
def INT_NVVM_DIV_RZ_F     : F_MATH_2<"div.rz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rz_f>;
def INT_NVVM_DIV_RM_FTZ_F : F_MATH_2<"div.rm.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rm_ftz_f>;
def INT_NVVM_DIV_RM_F     : F_MATH_2<"div.rm.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rm_f>;
def INT_NVVM_DIV_RP_FTZ_F : F_MATH_2<"div.rp.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rp_ftz_f>;
def INT_NVVM_DIV_RP_F     : F_MATH_2<"div.rp.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rp_f>;

def INT_NVVM_DIV_RN_D : F_MATH_2<"div.rn.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_div_rn_d>;
def INT_NVVM_DIV_RZ_D : F_MATH_2<"div.rz.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_div_rz_d>;
def INT_NVVM_DIV_RM_D : F_MATH_2<"div.rm.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_div_rm_d>;
def INT_NVVM_DIV_RP_D : F_MATH_2<"div.rp.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_div_rp_d>;

def : Pat<(int_nvvm_div_full f32:$a, f32:$b),
          (FDIV32rr $a, $b)>;

def : Pat<(int_nvvm_div_full f32:$a, fpimm:$b),
          (FDIV32ri $a, f32imm:$b)>;

def : Pat<(int_nvvm_div_full_ftz f32:$a, f32:$b),
          (FDIV32rr_ftz $a, $b)>;

def : Pat<(int_nvvm_div_full_ftz f32:$a, fpimm:$b),
          (FDIV32ri_ftz $a, f32imm:$b)>;

//
// Sad
//

def INT_NVVM_SAD_S : F_MATH_3<"sad.s16 \t$dst, $src0, $src1, $src2;",
  Int16Regs, Int16Regs, Int16Regs, Int16Regs, int_nvvm_sad_s>;
def INT_NVVM_SAD_US : F_MATH_3<"sad.u16 \t$dst, $src0, $src1, $src2;",
  Int16Regs, Int16Regs, Int16Regs, Int16Regs, int_nvvm_sad_us>;
def INT_NVVM_SAD_I : F_MATH_3<"sad.s32 \t$dst, $src0, $src1, $src2;",
  Int32Regs, Int32Regs, Int32Regs, Int32Regs, int_nvvm_sad_i>;
def INT_NVVM_SAD_UI : F_MATH_3<"sad.u32 \t$dst, $src0, $src1, $src2;",
  Int32Regs, Int32Regs, Int32Regs, Int32Regs, int_nvvm_sad_ui>;
def INT_NVVM_SAD_LL : F_MATH_3<"sad.s64 \t$dst, $src0, $src1, $src2;",
  Int64Regs, Int64Regs, Int64Regs, Int64Regs, int_nvvm_sad_ll>;
def INT_NVVM_SAD_ULL : F_MATH_3<"sad.u64 \t$dst, $src0, $src1, $src2;",
  Int64Regs, Int64Regs, Int64Regs, Int64Regs, int_nvvm_sad_ull>;

//
// Floor  Ceil
//

def : Pat<(int_nvvm_floor_ftz_f f32:$a),
          (CVT_f32_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_floor_f f32:$a),
          (CVT_f32_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_floor_d f64:$a),
          (CVT_f64_f64 $a, CvtRMI)>;

def : Pat<(int_nvvm_ceil_ftz_f f32:$a),
          (CVT_f32_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_ceil_f f32:$a),
          (CVT_f32_f32 $a, CvtRPI)>;
def : Pat<(int_nvvm_ceil_d f64:$a),
          (CVT_f64_f64 $a, CvtRPI)>;

//
// Abs
//

def INT_NVVM_FABS_FTZ_F : F_MATH_1<"abs.ftz.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_fabs_ftz_f>;
def INT_NVVM_FABS_F : F_MATH_1<"abs.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_fabs_f>;

def INT_NVVM_FABS_D : F_MATH_1<"abs.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_fabs_d>;

//
// copysign
//

def fcopysign_nvptx : SDNode<"NVPTXISD::FCOPYSIGN", SDTFPBinOp>;

def COPYSIGN_F :
    NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$src0, Float32Regs:$src1),
              "copysign.f32 \t$dst, $src0, $src1;",
              [(set f32:$dst, (fcopysign_nvptx f32:$src1, f32:$src0))]>;

def COPYSIGN_D :
    NVPTXInst<(outs Float64Regs:$dst), (ins Float64Regs:$src0, Float64Regs:$src1),
              "copysign.f64 \t$dst, $src0, $src1;",
              [(set f64:$dst, (fcopysign_nvptx f64:$src1, f64:$src0))]>;

//
// Abs, Neg bf16, bf16x2
//

def INT_NVVM_ABS_BF16 : F_MATH_1<"abs.bf16 \t$dst, $src0;", Int16Regs,
  Int16Regs, int_nvvm_abs_bf16, [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_ABS_BF16X2 : F_MATH_1<"abs.bf16x2 \t$dst, $src0;", Int32Regs,
  Int32Regs, int_nvvm_abs_bf16x2, [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_NEG_BF16 : F_MATH_1<"neg.bf16 \t$dst, $src0;", Int16Regs,
  Int16Regs, int_nvvm_neg_bf16, [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_NEG_BF16X2 : F_MATH_1<"neg.bf16x2 \t$dst, $src0;", Int32Regs,
  Int32Regs, int_nvvm_neg_bf16x2, [hasPTX<70>, hasSM<80>]>;

//
// Round
//

def : Pat<(int_nvvm_round_ftz_f f32:$a),
          (CVT_f32_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_round_f f32:$a),
          (CVT_f32_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_round_d f64:$a),
          (CVT_f64_f64 $a, CvtRNI)>;

//
// Trunc
//

def : Pat<(int_nvvm_trunc_ftz_f f32:$a),
          (CVT_f32_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_trunc_f f32:$a),
          (CVT_f32_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_trunc_d f64:$a),
          (CVT_f64_f64 $a, CvtRZI)>;

//
// Saturate
//

def : Pat<(int_nvvm_saturate_ftz_f f32:$a),
          (CVT_f32_f32 $a, CvtSAT_FTZ)>;
def : Pat<(int_nvvm_saturate_f f32:$a),
          (CVT_f32_f32 $a, CvtSAT)>;
def : Pat<(int_nvvm_saturate_d f64:$a),
          (CVT_f64_f64 $a, CvtSAT)>;

//
// Exp2  Log2
//

def INT_NVVM_EX2_APPROX_FTZ_F : F_MATH_1<"ex2.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_ex2_approx_ftz_f>;
def INT_NVVM_EX2_APPROX_F : F_MATH_1<"ex2.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_ex2_approx_f>;
def INT_NVVM_EX2_APPROX_D : F_MATH_1<"ex2.approx.f64 \t$dst, $src0;",
  Float64Regs, Float64Regs, int_nvvm_ex2_approx_d>;

def INT_NVVM_EX2_APPROX_F16 : F_MATH_1<"ex2.approx.f16 \t$dst, $src0;",
  Int16Regs, Int16Regs, int_nvvm_ex2_approx_f16, [hasPTX<70>, hasSM<75>]>;
def INT_NVVM_EX2_APPROX_F16X2 : F_MATH_1<"ex2.approx.f16x2 \t$dst, $src0;",
  Int32Regs, Int32Regs, int_nvvm_ex2_approx_f16x2, [hasPTX<70>, hasSM<75>]>;

def : Pat<(fexp2 f32:$a),
          (INT_NVVM_EX2_APPROX_FTZ_F $a)>, Requires<[doF32FTZ]>;
def : Pat<(fexp2 f32:$a),
          (INT_NVVM_EX2_APPROX_F $a)>, Requires<[doNoF32FTZ]>;
def : Pat<(fexp2 f16:$a),
          (INT_NVVM_EX2_APPROX_F16 $a)>, Requires<[useFP16Math]>;
def : Pat<(fexp2 v2f16:$a),
          (INT_NVVM_EX2_APPROX_F16X2 $a)>, Requires<[useFP16Math]>;

def INT_NVVM_LG2_APPROX_FTZ_F : F_MATH_1<"lg2.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_lg2_approx_ftz_f>;
def INT_NVVM_LG2_APPROX_F : F_MATH_1<"lg2.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_lg2_approx_f>;
def INT_NVVM_LG2_APPROX_D : F_MATH_1<"lg2.approx.f64 \t$dst, $src0;",
  Float64Regs, Float64Regs, int_nvvm_lg2_approx_d>;

def : Pat<(flog2 f32:$a), (INT_NVVM_LG2_APPROX_FTZ_F $a)>,
          Requires<[doF32FTZ]>;
def : Pat<(flog2 f32:$a), (INT_NVVM_LG2_APPROX_F $a)>,
          Requires<[doNoF32FTZ]>;

//
// Sin  Cos
//

def INT_NVVM_SIN_APPROX_FTZ_F : F_MATH_1<"sin.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sin_approx_ftz_f>;
def INT_NVVM_SIN_APPROX_F : F_MATH_1<"sin.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sin_approx_f>;

def INT_NVVM_COS_APPROX_FTZ_F : F_MATH_1<"cos.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_cos_approx_ftz_f>;
def INT_NVVM_COS_APPROX_F : F_MATH_1<"cos.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_cos_approx_f>;

//
// Fma
//

class FMA_TUPLE<string V, Intrinsic I, NVPTXRegClass RC,
                list<Predicate> Preds = []> {
  string Variant = V;
  Intrinsic Intr = I;
  NVPTXRegClass RegClass = RC;
  list<Predicate> Predicates = Preds;
}

multiclass FMA_INST {
  foreach P = [
    FMA_TUPLE<"_rn_f64", int_nvvm_fma_rn_d, Float64Regs>,
    FMA_TUPLE<"_rz_f64", int_nvvm_fma_rz_d, Float64Regs>,
    FMA_TUPLE<"_rm_f64", int_nvvm_fma_rm_d, Float64Regs>,
    FMA_TUPLE<"_rp_f64", int_nvvm_fma_rp_d, Float64Regs>,

    FMA_TUPLE<"_rn_ftz_f32", int_nvvm_fma_rn_ftz_f, Float32Regs>,
    FMA_TUPLE<"_rn_f32", int_nvvm_fma_rn_f, Float32Regs>,
    FMA_TUPLE<"_rz_ftz_f32", int_nvvm_fma_rz_ftz_f, Float32Regs>,
    FMA_TUPLE<"_rz_f32", int_nvvm_fma_rz_f, Float32Regs>,
    FMA_TUPLE<"_rm_f32", int_nvvm_fma_rm_f, Float32Regs>,
    FMA_TUPLE<"_rm_ftz_f32", int_nvvm_fma_rm_ftz_f, Float32Regs>,
    FMA_TUPLE<"_rp_f32", int_nvvm_fma_rp_f, Float32Regs>,
    FMA_TUPLE<"_rp_ftz_f32", int_nvvm_fma_rp_ftz_f, Float32Regs>,

    FMA_TUPLE<"_rn_f16", int_nvvm_fma_rn_f16, Int16Regs, [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_f16", int_nvvm_fma_rn_ftz_f16, Int16Regs,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_sat_f16", int_nvvm_fma_rn_sat_f16, Int16Regs,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_sat_f16", int_nvvm_fma_rn_ftz_sat_f16, Int16Regs,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_relu_f16", int_nvvm_fma_rn_relu_f16, Int16Regs,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_relu_f16", int_nvvm_fma_rn_ftz_relu_f16, Int16Regs,
      [hasPTX<70>, hasSM<80>]>,

    FMA_TUPLE<"_rn_bf16", int_nvvm_fma_rn_bf16, Int16Regs, [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_bf16", int_nvvm_fma_rn_ftz_bf16, Int16Regs,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_sat_bf16", int_nvvm_fma_rn_sat_bf16, Int16Regs,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_sat_bf16", int_nvvm_fma_rn_ftz_sat_bf16, Int16Regs,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_relu_bf16", int_nvvm_fma_rn_relu_bf16, Int16Regs,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_relu_bf16", int_nvvm_fma_rn_ftz_relu_bf16, Int16Regs,
      [hasPTX<70>, hasSM<80>]>,

    FMA_TUPLE<"_rn_f16x2", int_nvvm_fma_rn_f16x2, Int32Regs,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_f16x2", int_nvvm_fma_rn_ftz_f16x2, Int32Regs,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_sat_f16x2", int_nvvm_fma_rn_sat_f16x2, Int32Regs,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_sat_f16x2", int_nvvm_fma_rn_ftz_sat_f16x2,
      Int32Regs, [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_relu_f16x2", int_nvvm_fma_rn_relu_f16x2, Int32Regs,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_relu_f16x2", int_nvvm_fma_rn_ftz_relu_f16x2,
      Int32Regs, [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_bf16x2", int_nvvm_fma_rn_bf16x2, Int32Regs,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_relu_bf16x2", int_nvvm_fma_rn_relu_bf16x2, Int32Regs,
      [hasPTX<70>, hasSM<80>]>
  ] in {
    def P.Variant :
      F_MATH_3<!strconcat("fma",
        !subst("_", ".", P.Variant), " \t$dst, $src0, $src1, $src2;"),
        P.RegClass, P.RegClass, P.RegClass, P.RegClass, P.Intr, P.Predicates>;
  }
}

defm INT_NVVM_FMA : FMA_INST;

//
// Rcp
//

def INT_NVVM_RCP_RN_FTZ_F : F_MATH_1<"rcp.rn.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rn_ftz_f>;
def INT_NVVM_RCP_RN_F : F_MATH_1<"rcp.rn.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rn_f>;
def INT_NVVM_RCP_RZ_FTZ_F : F_MATH_1<"rcp.rz.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rz_ftz_f>;
def INT_NVVM_RCP_RZ_F : F_MATH_1<"rcp.rz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rz_f>;
def INT_NVVM_RCP_RM_FTZ_F : F_MATH_1<"rcp.rm.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rm_ftz_f>;
def INT_NVVM_RCP_RM_F : F_MATH_1<"rcp.rm.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rm_f>;
def INT_NVVM_RCP_RP_FTZ_F : F_MATH_1<"rcp.rp.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rp_ftz_f>;
def INT_NVVM_RCP_RP_F : F_MATH_1<"rcp.rp.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rp_f>;

def INT_NVVM_RCP_RN_D : F_MATH_1<"rcp.rn.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_rcp_rn_d>;
def INT_NVVM_RCP_RZ_D : F_MATH_1<"rcp.rz.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_rcp_rz_d>;
def INT_NVVM_RCP_RM_D : F_MATH_1<"rcp.rm.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_rcp_rm_d>;
def INT_NVVM_RCP_RP_D : F_MATH_1<"rcp.rp.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_rcp_rp_d>;

def INT_NVVM_RCP_APPROX_FTZ_F : F_MATH_1<"rcp.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_approx_ftz_f>;
def INT_NVVM_RCP_APPROX_FTZ_D : F_MATH_1<"rcp.approx.ftz.f64 \t$dst, $src0;",
  Float64Regs, Float64Regs, int_nvvm_rcp_approx_ftz_d>;

//
// Sqrt
//

def INT_NVVM_SQRT_RN_FTZ_F : F_MATH_1<"sqrt.rn.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_rn_ftz_f>;
def INT_NVVM_SQRT_RN_F : F_MATH_1<"sqrt.rn.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_sqrt_rn_f>;
def INT_NVVM_SQRT_RZ_FTZ_F : F_MATH_1<"sqrt.rz.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_rz_ftz_f>;
def INT_NVVM_SQRT_RZ_F : F_MATH_1<"sqrt.rz.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_sqrt_rz_f>;
def INT_NVVM_SQRT_RM_FTZ_F : F_MATH_1<"sqrt.rm.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_rm_ftz_f>;
def INT_NVVM_SQRT_RM_F : F_MATH_1<"sqrt.rm.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_sqrt_rm_f>;
def INT_NVVM_SQRT_RP_FTZ_F : F_MATH_1<"sqrt.rp.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_rp_ftz_f>;
def INT_NVVM_SQRT_RP_F : F_MATH_1<"sqrt.rp.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_sqrt_rp_f>;
def INT_NVVM_SQRT_APPROX_FTZ_F : F_MATH_1<"sqrt.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_approx_ftz_f>;
def INT_NVVM_SQRT_APPROX_F : F_MATH_1<"sqrt.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_approx_f>;

def INT_NVVM_SQRT_RN_D : F_MATH_1<"sqrt.rn.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_sqrt_rn_d>;
def INT_NVVM_SQRT_RZ_D : F_MATH_1<"sqrt.rz.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_sqrt_rz_d>;
def INT_NVVM_SQRT_RM_D : F_MATH_1<"sqrt.rm.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_sqrt_rm_d>;
def INT_NVVM_SQRT_RP_D : F_MATH_1<"sqrt.rp.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_sqrt_rp_d>;

// nvvm_sqrt intrinsic
def : Pat<(int_nvvm_sqrt_f f32:$a),
          (INT_NVVM_SQRT_RN_FTZ_F $a)>, Requires<[doF32FTZ, do_SQRTF32_RN]>;
def : Pat<(int_nvvm_sqrt_f f32:$a),
          (INT_NVVM_SQRT_RN_F $a)>, Requires<[do_SQRTF32_RN]>;
def : Pat<(int_nvvm_sqrt_f f32:$a),
          (INT_NVVM_SQRT_APPROX_FTZ_F $a)>, Requires<[doF32FTZ]>;
def : Pat<(int_nvvm_sqrt_f f32:$a),
          (INT_NVVM_SQRT_APPROX_F $a)>;

//
// Rsqrt
//

def INT_NVVM_RSQRT_APPROX_FTZ_F
  : F_MATH_1<"rsqrt.approx.ftz.f32 \t$dst, $src0;", Float32Regs, Float32Regs,
    int_nvvm_rsqrt_approx_ftz_f>;
def INT_NVVM_RSQRT_APPROX_FTZ_D
  : F_MATH_1<"rsqrt.approx.ftz.f64 \t$dst, $src0;", Float64Regs, Float64Regs,
    int_nvvm_rsqrt_approx_ftz_d>;

def INT_NVVM_RSQRT_APPROX_F : F_MATH_1<"rsqrt.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rsqrt_approx_f>;
def INT_NVVM_RSQRT_APPROX_D : F_MATH_1<"rsqrt.approx.f64 \t$dst, $src0;",
  Float64Regs, Float64Regs, int_nvvm_rsqrt_approx_d>;

// 1.0f / sqrt_approx -> rsqrt_approx
def: Pat<(fdiv FloatConst1, (int_nvvm_sqrt_approx_f f32:$a)),
         (INT_NVVM_RSQRT_APPROX_F $a)>,
         Requires<[doRsqrtOpt]>;
def: Pat<(fdiv FloatConst1, (int_nvvm_sqrt_approx_ftz_f f32:$a)),
         (INT_NVVM_RSQRT_APPROX_FTZ_F $a)>,
         Requires<[doRsqrtOpt]>;
// same for int_nvvm_sqrt_f when non-precision sqrt is requested
def: Pat<(fdiv FloatConst1, (int_nvvm_sqrt_f f32:$a)),
         (INT_NVVM_RSQRT_APPROX_F $a)>,
         Requires<[doRsqrtOpt, do_SQRTF32_APPROX, doNoF32FTZ]>;
def: Pat<(fdiv FloatConst1, (int_nvvm_sqrt_f f32:$a)),
         (INT_NVVM_RSQRT_APPROX_FTZ_F $a)>,
         Requires<[doRsqrtOpt, do_SQRTF32_APPROX, doF32FTZ]>;

def: Pat<(fdiv FloatConst1, (fsqrt f32:$a)),
         (INT_NVVM_RSQRT_APPROX_F $a)>,
         Requires<[doRsqrtOpt, do_SQRTF32_APPROX, doNoF32FTZ]>;
def: Pat<(fdiv FloatConst1, (fsqrt f32:$a)),
         (INT_NVVM_RSQRT_APPROX_FTZ_F $a)>,
         Requires<[doRsqrtOpt, do_SQRTF32_APPROX, doF32FTZ]>;
//
// Add
//

def INT_NVVM_ADD_RN_FTZ_F : F_MATH_2<"add.rn.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rn_ftz_f>;
def INT_NVVM_ADD_RN_F : F_MATH_2<"add.rn.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rn_f>;
def INT_NVVM_ADD_RZ_FTZ_F : F_MATH_2<"add.rz.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rz_ftz_f>;
def INT_NVVM_ADD_RZ_F : F_MATH_2<"add.rz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rz_f>;
def INT_NVVM_ADD_RM_FTZ_F : F_MATH_2<"add.rm.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rm_ftz_f>;
def INT_NVVM_ADD_RM_F : F_MATH_2<"add.rm.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rm_f>;
def INT_NVVM_ADD_RP_FTZ_F : F_MATH_2<"add.rp.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rp_ftz_f>;
def INT_NVVM_ADD_RP_F : F_MATH_2<"add.rp.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rp_f>;

def INT_NVVM_ADD_RN_D : F_MATH_2<"add.rn.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_add_rn_d>;
def INT_NVVM_ADD_RZ_D : F_MATH_2<"add.rz.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_add_rz_d>;
def INT_NVVM_ADD_RM_D : F_MATH_2<"add.rm.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_add_rm_d>;
def INT_NVVM_ADD_RP_D : F_MATH_2<"add.rp.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_add_rp_d>;

//
// BFIND
//

foreach t = [I32RT, I64RT] in {
  foreach sign = ["s", "u"] in {
    defvar flo_intrin = !cast<Intrinsic>("int_nvvm_flo_" # sign);
    def BFIND_ # sign # t.Size
      : NVPTXInst<(outs Int32Regs:$dst), (ins t.RC:$src),
                  "bfind." # sign # t.Size # " \t$dst, $src;",
                  [(set i32:$dst, (flo_intrin t.Ty:$src, 0))]>;

    def BFIND_SHIFTAMT_ # sign # t.Size
      : NVPTXInst<(outs Int32Regs:$dst), (ins t.RC:$src),
                  "bfind.shiftamt." # sign # t.Size # " \t$dst, $src;",
                  [(set i32:$dst, (flo_intrin t.Ty:$src, -1))]>;
  }
}

//
// Convert
//

def : Pat<(int_nvvm_d2f_rn_ftz f64:$a),
          (CVT_f32_f64 $a, CvtRN_FTZ)>;
def : Pat<(int_nvvm_d2f_rn f64:$a),
          (CVT_f32_f64 $a, CvtRN)>;
def : Pat<(int_nvvm_d2f_rz_ftz f64:$a),
          (CVT_f32_f64 $a, CvtRZ_FTZ)>;
def : Pat<(int_nvvm_d2f_rz f64:$a),
          (CVT_f32_f64 $a, CvtRZ)>;
def : Pat<(int_nvvm_d2f_rm_ftz f64:$a),
          (CVT_f32_f64 $a, CvtRM_FTZ)>;
def : Pat<(int_nvvm_d2f_rm f64:$a),
          (CVT_f32_f64 $a, CvtRM)>;
def : Pat<(int_nvvm_d2f_rp_ftz f64:$a),
          (CVT_f32_f64 $a, CvtRP_FTZ)>;
def : Pat<(int_nvvm_d2f_rp f64:$a),
          (CVT_f32_f64 $a, CvtRP)>;

def : Pat<(int_nvvm_d2i_rn f64:$a),
          (CVT_s32_f64 $a, CvtRNI)>;
def : Pat<(int_nvvm_d2i_rz f64:$a),
          (CVT_s32_f64 $a, CvtRZI)>;
def : Pat<(int_nvvm_d2i_rm f64:$a),
          (CVT_s32_f64 $a, CvtRMI)>;
def : Pat<(int_nvvm_d2i_rp f64:$a),
          (CVT_s32_f64 $a, CvtRPI)>;

def : Pat<(int_nvvm_d2ui_rn f64:$a),
          (CVT_u32_f64 $a, CvtRNI)>;
def : Pat<(int_nvvm_d2ui_rz f64:$a),
          (CVT_u32_f64 $a, CvtRZI)>;
def : Pat<(int_nvvm_d2ui_rm f64:$a),
          (CVT_u32_f64 $a, CvtRMI)>;
def : Pat<(int_nvvm_d2ui_rp f64:$a),
          (CVT_u32_f64 $a, CvtRPI)>;

def : Pat<(int_nvvm_i2d_rn i32:$a),
          (CVT_f64_s32 $a, CvtRN)>;
def : Pat<(int_nvvm_i2d_rz i32:$a),
          (CVT_f64_s32 $a, CvtRZ)>;
def : Pat<(int_nvvm_i2d_rm i32:$a),
          (CVT_f64_s32 $a, CvtRM)>;
def : Pat<(int_nvvm_i2d_rp i32:$a),
          (CVT_f64_s32 $a, CvtRP)>;

def : Pat<(int_nvvm_ui2d_rn i32:$a),
          (CVT_f64_u32 $a, CvtRN)>;
def : Pat<(int_nvvm_ui2d_rz i32:$a),
          (CVT_f64_u32 $a, CvtRZ)>;
def : Pat<(int_nvvm_ui2d_rm i32:$a),
          (CVT_f64_u32 $a, CvtRM)>;
def : Pat<(int_nvvm_ui2d_rp i32:$a),
          (CVT_f64_u32 $a, CvtRP)>;

def : Pat<(int_nvvm_f2i_rn_ftz f32:$a),
          (CVT_s32_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2i_rn f32:$a),
          (CVT_s32_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_f2i_rz_ftz f32:$a),
          (CVT_s32_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2i_rz f32:$a),
          (CVT_s32_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_f2i_rm_ftz f32:$a),
          (CVT_s32_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2i_rm f32:$a),
          (CVT_s32_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_f2i_rp_ftz f32:$a),
          (CVT_s32_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2i_rp f32:$a),
          (CVT_s32_f32 $a, CvtRPI)>;

def : Pat<(int_nvvm_f2ui_rn_ftz f32:$a),
          (CVT_u32_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rn f32:$a),
          (CVT_u32_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_f2ui_rz_ftz f32:$a),
          (CVT_u32_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rz f32:$a),
          (CVT_u32_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_f2ui_rm_ftz f32:$a),
          (CVT_u32_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rm f32:$a),
          (CVT_u32_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_f2ui_rp_ftz f32:$a),
          (CVT_u32_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rp f32:$a),
          (CVT_u32_f32 $a, CvtRPI)>;

def : Pat<(int_nvvm_i2f_rn i32:$a),
          (CVT_f32_s32 $a, CvtRN)>;
def : Pat<(int_nvvm_i2f_rz i32:$a),
          (CVT_f32_s32 $a, CvtRZ)>;
def : Pat<(int_nvvm_i2f_rm i32:$a),
          (CVT_f32_s32 $a, CvtRM)>;
def : Pat<(int_nvvm_i2f_rp i32:$a),
          (CVT_f32_s32 $a, CvtRP)>;

def : Pat<(int_nvvm_ui2f_rn i32:$a),
          (CVT_f32_u32 $a, CvtRN)>;
def : Pat<(int_nvvm_ui2f_rz i32:$a),
          (CVT_f32_u32 $a, CvtRZ)>;
def : Pat<(int_nvvm_ui2f_rm i32:$a),
          (CVT_f32_u32 $a, CvtRM)>;
def : Pat<(int_nvvm_ui2f_rp i32:$a),
          (CVT_f32_u32 $a, CvtRP)>;

def : Pat<(int_nvvm_ff2bf16x2_rn f32:$a, f32:$b),
          (CVT_bf16x2_f32 $a, $b, CvtRN)>;
def : Pat<(int_nvvm_ff2bf16x2_rn_relu f32:$a, f32:$b),
          (CVT_bf16x2_f32 $a, $b, CvtRN_RELU)>;
def : Pat<(int_nvvm_ff2bf16x2_rz f32:$a, f32:$b),
          (CVT_bf16x2_f32 $a, $b, CvtRZ)>;
def : Pat<(int_nvvm_ff2bf16x2_rz_relu f32:$a, f32:$b),
          (CVT_bf16x2_f32 $a, $b, CvtRZ_RELU)>;

def : Pat<(int_nvvm_ff2f16x2_rn f32:$a, f32:$b),
          (CVT_f16x2_f32 $a, $b, CvtRN)>;
def : Pat<(int_nvvm_ff2f16x2_rn_relu f32:$a, f32:$b),
          (CVT_f16x2_f32 $a, $b, CvtRN_RELU)>;
def : Pat<(int_nvvm_ff2f16x2_rz f32:$a, f32:$b),
          (CVT_f16x2_f32 $a, $b, CvtRZ)>;
def : Pat<(int_nvvm_ff2f16x2_rz_relu f32:$a, f32:$b),
          (CVT_f16x2_f32 $a, $b, CvtRZ_RELU)>;

def : Pat<(int_nvvm_f2bf16_rn f32:$a),
          (CVT_bf16_f32 $a, CvtRN)>;
def : Pat<(int_nvvm_f2bf16_rn_relu f32:$a),
          (CVT_bf16_f32 $a, CvtRN_RELU)>;
def : Pat<(int_nvvm_f2bf16_rz f32:$a),
          (CVT_bf16_f32 $a, CvtRZ)>;
def : Pat<(int_nvvm_f2bf16_rz_relu f32:$a),
          (CVT_bf16_f32 $a, CvtRZ_RELU)>;

def INT_NVVM_LOHI_I2D : F_MATH_2<"mov.b64 \t$dst, {{$src0, $src1}};",
  Float64Regs, Int32Regs, Int32Regs, int_nvvm_lohi_i2d>;

def INT_NVVM_D2I_LO : F_MATH_1<
  !strconcat("{{\n\t",
             ".reg .b32 %temp; \n\t",
             "mov.b64 \t{$dst, %temp}, $src0;\n\t",
             "}}"),
  Int32Regs, Float64Regs, int_nvvm_d2i_lo>;
def INT_NVVM_D2I_HI : F_MATH_1<
  !strconcat("{{\n\t",
             ".reg .b32 %temp; \n\t",
             "mov.b64 \t{%temp, $dst}, $src0;\n\t",
             "}}"),
  Int32Regs, Float64Regs, int_nvvm_d2i_hi>;

def : Pat<(int_nvvm_f2ll_rn_ftz f32:$a),
          (CVT_s64_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rn f32:$a),
          (CVT_s64_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_f2ll_rz_ftz f32:$a),
          (CVT_s64_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rz f32:$a),
          (CVT_s64_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_f2ll_rm_ftz f32:$a),
          (CVT_s64_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rm f32:$a),
          (CVT_s64_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_f2ll_rp_ftz f32:$a),
          (CVT_s64_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rp f32:$a),
          (CVT_s64_f32 $a, CvtRPI)>;

def : Pat<(int_nvvm_f2ull_rn_ftz f32:$a),
          (CVT_u64_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rn f32:$a),
          (CVT_u64_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_f2ull_rz_ftz f32:$a),
          (CVT_u64_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rz f32:$a),
          (CVT_u64_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_f2ull_rm_ftz f32:$a),
          (CVT_u64_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rm f32:$a),
          (CVT_u64_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_f2ull_rp_ftz f32:$a),
          (CVT_u64_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rp f32:$a),
          (CVT_u64_f32 $a, CvtRPI)>;

def : Pat<(int_nvvm_d2ll_rn f64:$a),
          (CVT_s64_f64 $a, CvtRNI)>;
def : Pat<(int_nvvm_d2ll_rz f64:$a),
          (CVT_s64_f64 $a, CvtRZI)>;
def : Pat<(int_nvvm_d2ll_rm f64:$a),
          (CVT_s64_f64 $a, CvtRMI)>;
def : Pat<(int_nvvm_d2ll_rp f64:$a),
          (CVT_s64_f64 $a, CvtRPI)>;

def : Pat<(int_nvvm_d2ull_rn f64:$a),
          (CVT_u64_f64 $a, CvtRNI)>;
def : Pat<(int_nvvm_d2ull_rz f64:$a),
          (CVT_u64_f64 $a, CvtRZI)>;
def : Pat<(int_nvvm_d2ull_rm f64:$a),
          (CVT_u64_f64 $a, CvtRMI)>;
def : Pat<(int_nvvm_d2ull_rp f64:$a),
          (CVT_u64_f64 $a, CvtRPI)>;

def : Pat<(int_nvvm_ll2f_rn i64:$a),
          (CVT_f32_s64 $a, CvtRN)>;
def : Pat<(int_nvvm_ll2f_rz i64:$a),
          (CVT_f32_s64 $a, CvtRZ)>;
def : Pat<(int_nvvm_ll2f_rm i64:$a),
          (CVT_f32_s64 $a, CvtRM)>;
def : Pat<(int_nvvm_ll2f_rp i64:$a),
          (CVT_f32_s64 $a, CvtRP)>;

def : Pat<(int_nvvm_ull2f_rn i64:$a),
          (CVT_f32_u64 $a, CvtRN)>;
def : Pat<(int_nvvm_ull2f_rz i64:$a),
          (CVT_f32_u64 $a, CvtRZ)>;
def : Pat<(int_nvvm_ull2f_rm i64:$a),
          (CVT_f32_u64 $a, CvtRM)>;
def : Pat<(int_nvvm_ull2f_rp i64:$a),
          (CVT_f32_u64 $a, CvtRP)>;

def : Pat<(int_nvvm_ll2d_rn i64:$a),
          (CVT_f64_s64 $a, CvtRN)>;
def : Pat<(int_nvvm_ll2d_rz i64:$a),
          (CVT_f64_s64 $a, CvtRZ)>;
def : Pat<(int_nvvm_ll2d_rm i64:$a),
          (CVT_f64_s64 $a, CvtRM)>;
def : Pat<(int_nvvm_ll2d_rp i64:$a),
          (CVT_f64_s64 $a, CvtRP)>;

def : Pat<(int_nvvm_ull2d_rn i64:$a),
          (CVT_f64_u64 $a, CvtRN)>;
def : Pat<(int_nvvm_ull2d_rz i64:$a),
          (CVT_f64_u64 $a, CvtRZ)>;
def : Pat<(int_nvvm_ull2d_rm i64:$a),
          (CVT_f64_u64 $a, CvtRM)>;
def : Pat<(int_nvvm_ull2d_rp i64:$a),
          (CVT_f64_u64 $a, CvtRP)>;


def : Pat<(int_nvvm_f2h_rn_ftz f32:$a),
          (CVT_f16_f32 $a, CvtRN_FTZ)>;
def : Pat<(int_nvvm_f2h_rn f32:$a),
          (CVT_f16_f32 $a, CvtRN)>;

def : Pat<(int_nvvm_ff_to_e4m3x2_rn f32:$a, f32:$b),
          (CVT_e4m3x2_f32 $a, $b, CvtRN)>;
def : Pat<(int_nvvm_ff_to_e4m3x2_rn_relu f32:$a, f32:$b),
          (CVT_e4m3x2_f32 $a, $b, CvtRN_RELU)>;
def : Pat<(int_nvvm_ff_to_e5m2x2_rn f32:$a, f32:$b),
          (CVT_e5m2x2_f32 $a, $b, CvtRN)>;
def : Pat<(int_nvvm_ff_to_e5m2x2_rn_relu f32:$a, f32:$b),
          (CVT_e5m2x2_f32 $a, $b, CvtRN_RELU)>;

def : Pat<(int_nvvm_f16x2_to_e4m3x2_rn Int32Regs:$a),
          (CVT_e4m3x2_f16x2 $a, CvtRN)>;
def : Pat<(int_nvvm_f16x2_to_e4m3x2_rn_relu Int32Regs:$a),
          (CVT_e4m3x2_f16x2 $a, CvtRN_RELU)>;
def : Pat<(int_nvvm_f16x2_to_e5m2x2_rn Int32Regs:$a),
          (CVT_e5m2x2_f16x2 $a, CvtRN)>;
def : Pat<(int_nvvm_f16x2_to_e5m2x2_rn_relu Int32Regs:$a),
          (CVT_e5m2x2_f16x2 $a, CvtRN_RELU)>;

def : Pat<(int_nvvm_e4m3x2_to_f16x2_rn Int16Regs:$a),
          (CVT_f16x2_e4m3x2 $a, CvtRN)>;
def : Pat<(int_nvvm_e4m3x2_to_f16x2_rn_relu Int16Regs:$a),
          (CVT_f16x2_e4m3x2 $a, CvtRN_RELU)>;
def : Pat<(int_nvvm_e5m2x2_to_f16x2_rn Int16Regs:$a),
          (CVT_f16x2_e5m2x2 $a, CvtRN)>;
def : Pat<(int_nvvm_e5m2x2_to_f16x2_rn_relu Int16Regs:$a),
          (CVT_f16x2_e5m2x2 $a, CvtRN_RELU)>;

//
// FNS
//

class INT_FNS_MBO<dag ins, dag Operands>
  : NVPTXInst<(outs Int32Regs:$dst), ins,
               "fns.b32 \t$dst, $mask, $base, $offset;",
               [(set i32:$dst, Operands)]>,
    Requires<[hasPTX<60>, hasSM<30>]>;

def INT_FNS_rrr : INT_FNS_MBO<(ins Int32Regs:$mask, Int32Regs:$base, Int32Regs:$offset),
                     (int_nvvm_fns i32:$mask, i32:$base, i32:$offset)>;
def INT_FNS_rri : INT_FNS_MBO<(ins Int32Regs:$mask, Int32Regs:$base,    i32imm:$offset),
                     (int_nvvm_fns i32:$mask, i32:$base,       imm:$offset)>;
def INT_FNS_rir : INT_FNS_MBO<(ins Int32Regs:$mask,    i32imm:$base, Int32Regs:$offset),
                     (int_nvvm_fns i32:$mask,       imm:$base, i32:$offset)>;
def INT_FNS_rii : INT_FNS_MBO<(ins Int32Regs:$mask,    i32imm:$base,    i32imm:$offset),
                     (int_nvvm_fns i32:$mask,       imm:$base,       imm:$offset)>;
def INT_FNS_irr : INT_FNS_MBO<(ins    i32imm:$mask, Int32Regs:$base, Int32Regs:$offset),
                     (int_nvvm_fns       imm:$mask, i32:$base, i32:$offset)>;
def INT_FNS_iri : INT_FNS_MBO<(ins    i32imm:$mask, Int32Regs:$base,    i32imm:$offset),
                     (int_nvvm_fns       imm:$mask, i32:$base,       imm:$offset)>;
def INT_FNS_iir : INT_FNS_MBO<(ins    i32imm:$mask,    i32imm:$base, Int32Regs:$offset),
                     (int_nvvm_fns       imm:$mask,       imm:$base, i32:$offset)>;
def INT_FNS_iii : INT_FNS_MBO<(ins    i32imm:$mask,    i32imm:$base,    i32imm:$offset),
                     (int_nvvm_fns       imm:$mask,       imm:$base,       imm:$offset)>;

//-----------------------------------
// Atomic Functions
//-----------------------------------

class ATOMIC_GLOBAL_CHK <dag ops, dag frag>
 : PatFrag<ops, frag, AS_match.global>;
class ATOMIC_SHARED_CHK <dag ops, dag frag>
 : PatFrag<ops, frag, AS_match.shared>;
class ATOMIC_GENERIC_CHK <dag ops, dag frag>
 : PatFrag<ops, frag, AS_match.generic>;

multiclass F_ATOMIC_2_imp<ValueType ptrT, NVPTXRegClass ptrclass,
  ValueType regT, NVPTXRegClass regclass,
  string SpaceStr, string TypeStr, string OpcStr, PatFrag IntOp,
  Operand IMMType, SDNode IMM, list<Predicate> Pred> {
  let mayLoad = 1, mayStore = 1, hasSideEffects = 1 in {
    def reg : NVPTXInst<(outs regclass:$dst), (ins ptrclass:$addr, regclass:$b),
      !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b;"),
      [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), (regT regclass:$b)))]>,
    Requires<Pred>;
    def imm : NVPTXInst<(outs regclass:$dst), (ins ptrclass:$addr, IMMType:$b),
      !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b;", ""),
      [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), IMM:$b))]>,
    Requires<!if(!or(!eq(TypeStr, ".f16"), !eq(TypeStr, ".bf16")), [Predicate<"false">], Pred)>;
  }
}
multiclass F_ATOMIC_2<ValueType regT, NVPTXRegClass regclass, string SpaceStr, string TypeStr,
  string OpcStr, PatFrag IntOp, Operand IMMType, SDNode IMM,
  list<Predicate> Pred = []> {
  defm p32 : F_ATOMIC_2_imp<i32, Int32Regs, regT, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, IMM, Pred>;
  defm p64 : F_ATOMIC_2_imp<i64, Int64Regs, regT, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, IMM, Pred>;
}

// has 2 operands, neg the second one
multiclass F_ATOMIC_2_NEG_imp<ValueType ptrT, NVPTXRegClass ptrclass,
  ValueType regT, NVPTXRegClass regclass,
  string SpaceStr, string TypeStr, string OpcStr, PatFrag IntOp,
  list<Predicate> Pred> {
  let mayLoad = 1, mayStore = 1, hasSideEffects = 1 in {
    def reg : NVPTXInst<(outs regclass:$dst), (ins ptrclass:$addr, regclass:$b),
      !strconcat(
        "{{ \n\t",
        ".reg \t.s", TypeStr, " temp; \n\t",
        "neg.s", TypeStr, " \ttemp, $b; \n\t",
        "atom", SpaceStr, OpcStr, ".u", TypeStr, " \t$dst, [$addr], temp; \n\t",
        "}}"),
      [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), (regT regclass:$b)))]>,
    Requires<Pred>;
  }
}
multiclass F_ATOMIC_2_NEG<ValueType regT, NVPTXRegClass regclass, string SpaceStr,
  string TypeStr, string OpcStr, PatFrag IntOp, list<Predicate> Pred = []> {
 defm p32: F_ATOMIC_2_NEG_imp<i32, Int32Regs, regT, regclass, SpaceStr, TypeStr, OpcStr,
   IntOp, Pred> ;
 defm p64: F_ATOMIC_2_NEG_imp<i64, Int64Regs, regT, regclass, SpaceStr, TypeStr, OpcStr,
   IntOp, Pred> ;
}

// has 3 operands
multiclass F_ATOMIC_3_imp<ValueType ptrT, NVPTXRegClass ptrclass,
  ValueType regT, NVPTXRegClass regclass,
  string SpaceStr, string TypeStr, string OpcStr, PatFrag IntOp,
  Operand IMMType, list<Predicate> Pred> {
  let mayLoad = 1, mayStore = 1, hasSideEffects = 1 in {
    def reg : NVPTXInst<(outs regclass:$dst),
      (ins ptrclass:$addr, regclass:$b, regclass:$c),
      !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b, $c;"),
      [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), (regT regclass:$b), (regT regclass:$c)))]>,
    Requires<Pred>;

    def imm1 : NVPTXInst<(outs regclass:$dst),
      (ins ptrclass:$addr, IMMType:$b, regclass:$c),
      !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b, $c;"),
      [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), imm:$b, (regT regclass:$c)))]>,
    Requires<Pred>;

    def imm2 : NVPTXInst<(outs regclass:$dst),
      (ins ptrclass:$addr, regclass:$b, IMMType:$c),
      !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b, $c;", ""),
      [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), (regT regclass:$b), imm:$c))]>,
    Requires<Pred>;

    def imm3 : NVPTXInst<(outs regclass:$dst),
      (ins ptrclass:$addr, IMMType:$b, IMMType:$c),
      !strconcat("atom", SpaceStr, OpcStr, TypeStr, " \t$dst, [$addr], $b, $c;"),
      [(set (regT regclass:$dst), (IntOp (ptrT ptrclass:$addr), imm:$b, imm:$c))]>,
    Requires<Pred>;
  }
}
multiclass F_ATOMIC_3<ValueType regT, NVPTXRegClass regclass, string SpaceStr, string TypeStr,
  string OpcStr, PatFrag IntOp, Operand IMMType, list<Predicate> Pred = []> {
  defm p32 : F_ATOMIC_3_imp<i32, Int32Regs, regT, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, Pred>;
  defm p64 : F_ATOMIC_3_imp<i64, Int64Regs, regT, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, Pred>;
}

// atom_add

def atomic_load_add_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_add_i32 node:$a, node:$b)>;
def atomic_load_add_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_add_i32 node:$a, node:$b)>;
def atomic_load_add_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_add_i32 node:$a, node:$b)>;
def atomic_load_add_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_add_i64 node:$a, node:$b)>;
def atomic_load_add_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_add_i64 node:$a, node:$b)>;
def atomic_load_add_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_add_i64 node:$a, node:$b)>;
def atomic_load_add_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_fadd node:$a, node:$b)>;
def atomic_load_add_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_fadd node:$a, node:$b)>;
def atomic_load_add_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_fadd node:$a, node:$b)>;

defm INT_PTX_ATOM_ADD_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".u32", ".add",
  atomic_load_add_i32_g, i32imm, imm>;
defm INT_PTX_ATOM_ADD_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".u32", ".add",
  atomic_load_add_i32_s, i32imm, imm>;
defm INT_PTX_ATOM_ADD_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".u32", ".add",
  atomic_load_add_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_ADD_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global", ".u32",
  ".add", atomic_load_add_i32_gen, i32imm, imm>;

defm INT_PTX_ATOM_ADD_G_64 : F_ATOMIC_2<i64, Int64Regs, ".global", ".u64", ".add",
  atomic_load_add_i64_g, i64imm, imm>;
defm INT_PTX_ATOM_ADD_S_64 : F_ATOMIC_2<i64, Int64Regs, ".shared", ".u64", ".add",
  atomic_load_add_i64_s, i64imm, imm>;
defm INT_PTX_ATOM_ADD_GEN_64 : F_ATOMIC_2<i64, Int64Regs, "", ".u64", ".add",
  atomic_load_add_i64_gen, i64imm, imm>;
defm INT_PTX_ATOM_ADD_GEN_64_USE_G : F_ATOMIC_2<i64, Int64Regs, ".global", ".u64",
  ".add", atomic_load_add_i64_gen, i64imm, imm>;

defm INT_PTX_ATOM_ADD_G_F16 : F_ATOMIC_2<f16, Int16Regs, ".global", ".f16", ".add.noftz",
  atomic_load_add_g, f16imm, fpimm, [hasSM<70>, hasPTX<63>]>;
defm INT_PTX_ATOM_ADD_S_F16 : F_ATOMIC_2<f16, Int16Regs, ".shared", ".f16", ".add.noftz",
  atomic_load_add_s, f16imm, fpimm, [hasSM<70>, hasPTX<63>]>;
defm INT_PTX_ATOM_ADD_GEN_F16 : F_ATOMIC_2<f16, Int16Regs, "", ".f16", ".add.noftz",
  atomic_load_add_gen, f16imm, fpimm, [hasSM<70>, hasPTX<63>]>;

defm INT_PTX_ATOM_ADD_G_BF16 : F_ATOMIC_2<bf16, Int16Regs, ".global", ".bf16", ".add.noftz",
  atomic_load_add_g, bf16imm, fpimm, [hasSM<90>, hasPTX<78>]>;
defm INT_PTX_ATOM_ADD_S_BF16 : F_ATOMIC_2<bf16, Int16Regs, ".shared", ".bf16", ".add.noftz",
  atomic_load_add_s, bf16imm, fpimm, [hasSM<90>, hasPTX<78>]>;
defm INT_PTX_ATOM_ADD_GEN_BF16 : F_ATOMIC_2<bf16, Int16Regs, "", ".bf16", ".add.noftz",
  atomic_load_add_gen, bf16imm, fpimm, [hasSM<90>, hasPTX<78>]>;

defm INT_PTX_ATOM_ADD_G_F32 : F_ATOMIC_2<f32, Float32Regs, ".global", ".f32", ".add",
  atomic_load_add_g, f32imm, fpimm>;
defm INT_PTX_ATOM_ADD_S_F32 : F_ATOMIC_2<f32, Float32Regs, ".shared", ".f32", ".add",
  atomic_load_add_s, f32imm, fpimm>;
defm INT_PTX_ATOM_ADD_GEN_F32 : F_ATOMIC_2<f32, Float32Regs, "", ".f32", ".add",
  atomic_load_add_gen, f32imm, fpimm>;

defm INT_PTX_ATOM_ADD_G_F64 : F_ATOMIC_2<f64, Float64Regs, ".global", ".f64", ".add",
  atomic_load_add_g, f64imm, fpimm, [hasAtomAddF64]>;
defm INT_PTX_ATOM_ADD_S_F64 : F_ATOMIC_2<f64, Float64Regs, ".shared", ".f64", ".add",
  atomic_load_add_s, f64imm, fpimm, [hasAtomAddF64]>;
defm INT_PTX_ATOM_ADD_GEN_F64 : F_ATOMIC_2<f64, Float64Regs, "", ".f64", ".add",
  atomic_load_add_gen, f64imm, fpimm, [hasAtomAddF64]>;

// atom_sub

def atomic_load_sub_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_i32 node:$a, node:$b)>;
def atomic_load_sub_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_i32 node:$a, node:$b)>;
def atomic_load_sub_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_i32 node:$a, node:$b)>;
def atomic_load_sub_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_i64 node:$a, node:$b)>;
def atomic_load_sub_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_i64 node:$a, node:$b)>;
def atomic_load_sub_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_i64 node:$a, node:$b)>;

defm INT_PTX_ATOM_SUB_G_32 : F_ATOMIC_2_NEG<i32, Int32Regs, ".global", "32", ".add",
  atomic_load_sub_i32_g>;
defm INT_PTX_ATOM_SUB_G_64 : F_ATOMIC_2_NEG<i64, Int64Regs, ".global", "64", ".add",
  atomic_load_sub_i64_g>;
defm INT_PTX_ATOM_SUB_GEN_32 : F_ATOMIC_2_NEG<i32, Int32Regs, "", "32", ".add",
  atomic_load_sub_i32_gen>;
defm INT_PTX_ATOM_SUB_GEN_32_USE_G : F_ATOMIC_2_NEG<i32, Int32Regs, ".global", "32",
  ".add", atomic_load_sub_i32_gen>;
defm INT_PTX_ATOM_SUB_S_32 : F_ATOMIC_2_NEG<i32, Int32Regs, ".shared", "32", ".add",
  atomic_load_sub_i32_s>;
defm INT_PTX_ATOM_SUB_S_64 : F_ATOMIC_2_NEG<i64, Int64Regs, ".shared", "64", ".add",
  atomic_load_sub_i64_s>;
defm INT_PTX_ATOM_SUB_GEN_64 : F_ATOMIC_2_NEG<i64, Int64Regs, "", "64", ".add",
  atomic_load_sub_i64_gen>;
defm INT_PTX_ATOM_SUB_GEN_64_USE_G : F_ATOMIC_2_NEG<i64, Int64Regs, ".global", "64",
  ".add", atomic_load_sub_i64_gen>;

// atom_swap

def atomic_swap_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_swap_i32 node:$a, node:$b)>;
def atomic_swap_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_swap_i32 node:$a, node:$b)>;
def atomic_swap_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_swap_i32 node:$a, node:$b)>;
def atomic_swap_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_swap_i64 node:$a, node:$b)>;
def atomic_swap_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_swap_i64 node:$a, node:$b)>;
def atomic_swap_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_swap_i64 node:$a, node:$b)>;

defm INT_PTX_ATOM_SWAP_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".b32", ".exch",
  atomic_swap_i32_g, i32imm, imm>;
defm INT_PTX_ATOM_SWAP_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".b32", ".exch",
  atomic_swap_i32_s, i32imm, imm>;
defm INT_PTX_ATOM_SWAP_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".b32", ".exch",
  atomic_swap_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_SWAP_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global", ".b32",
  ".exch", atomic_swap_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_SWAP_G_64 : F_ATOMIC_2<i64, Int64Regs, ".global", ".b64", ".exch",
  atomic_swap_i64_g, i64imm, imm>;
defm INT_PTX_ATOM_SWAP_S_64 : F_ATOMIC_2<i64, Int64Regs, ".shared", ".b64", ".exch",
  atomic_swap_i64_s, i64imm, imm>;
defm INT_PTX_ATOM_SWAP_GEN_64 : F_ATOMIC_2<i64, Int64Regs, "", ".b64", ".exch",
  atomic_swap_i64_gen, i64imm, imm>;
defm INT_PTX_ATOM_SWAP_GEN_64_USE_G : F_ATOMIC_2<i64, Int64Regs, ".global", ".b64",
  ".exch", atomic_swap_i64_gen, i64imm, imm>;

// atom_max

def atomic_load_max_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b)
  , (atomic_load_max_i32 node:$a, node:$b)>;
def atomic_load_max_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_max_i32 node:$a, node:$b)>;
def atomic_load_max_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_max_i32 node:$a, node:$b)>;
def atomic_load_max_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b)
  , (atomic_load_max_i64 node:$a, node:$b)>;
def atomic_load_max_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_max_i64 node:$a, node:$b)>;
def atomic_load_max_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_max_i64 node:$a, node:$b)>;
def atomic_load_umax_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_i32 node:$a, node:$b)>;
def atomic_load_umax_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_i32 node:$a, node:$b)>;
def atomic_load_umax_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_i32 node:$a, node:$b)>;
def atomic_load_umax_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_i64 node:$a, node:$b)>;
def atomic_load_umax_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_i64 node:$a, node:$b)>;
def atomic_load_umax_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_i64 node:$a, node:$b)>;

defm INT_PTX_ATOM_LOAD_MAX_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".s32",
  ".max", atomic_load_max_i32_g, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MAX_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".s32",
  ".max", atomic_load_max_i32_s, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".s32", ".max",
  atomic_load_max_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global",
  ".s32", ".max", atomic_load_max_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MAX_G_64 : F_ATOMIC_2<i64, Int64Regs, ".global", ".s64",
  ".max", atomic_load_max_i64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MAX_S_64 : F_ATOMIC_2<i64, Int64Regs, ".shared", ".s64",
  ".max", atomic_load_max_i64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_64 : F_ATOMIC_2<i64, Int64Regs, "", ".s64", ".max",
  atomic_load_max_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_64_USE_G : F_ATOMIC_2<i64, Int64Regs, ".global",
  ".s64", ".max", atomic_load_max_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMAX_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".u32",
  ".max", atomic_load_umax_i32_g, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMAX_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".u32",
  ".max", atomic_load_umax_i32_s, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".u32", ".max",
  atomic_load_umax_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global",
  ".u32", ".max", atomic_load_umax_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMAX_G_64 : F_ATOMIC_2<i64, Int64Regs, ".global", ".u64",
  ".max", atomic_load_umax_i64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMAX_S_64 : F_ATOMIC_2<i64, Int64Regs, ".shared", ".u64",
  ".max", atomic_load_umax_i64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_64 : F_ATOMIC_2<i64, Int64Regs, "", ".u64", ".max",
  atomic_load_umax_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_64_USE_G : F_ATOMIC_2<i64, Int64Regs, ".global",
  ".u64", ".max", atomic_load_umax_i64_gen, i64imm, imm, [hasSM<32>]>;

// atom_min

def atomic_load_min_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_min_i32 node:$a, node:$b)>;
def atomic_load_min_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_min_i32 node:$a, node:$b)>;
def atomic_load_min_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_min_i32 node:$a, node:$b)>;
def atomic_load_min_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_min_i64 node:$a, node:$b)>;
def atomic_load_min_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_min_i64 node:$a, node:$b)>;
def atomic_load_min_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_min_i64 node:$a, node:$b)>;
def atomic_load_umin_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_i32 node:$a, node:$b)>;
def atomic_load_umin_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_i32 node:$a, node:$b)>;
def atomic_load_umin_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_i32 node:$a, node:$b)>;
def atomic_load_umin_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_i64 node:$a, node:$b)>;
def atomic_load_umin_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_i64 node:$a, node:$b)>;
def atomic_load_umin_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_i64 node:$a, node:$b)>;

defm INT_PTX_ATOM_LOAD_MIN_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".s32",
  ".min", atomic_load_min_i32_g, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MIN_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".s32",
  ".min", atomic_load_min_i32_s, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".s32", ".min",
  atomic_load_min_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global",
  ".s32", ".min", atomic_load_min_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_MIN_G_64 : F_ATOMIC_2<i64, Int64Regs, ".global", ".s64",
  ".min", atomic_load_min_i64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MIN_S_64 : F_ATOMIC_2<i64, Int64Regs, ".shared", ".s64",
  ".min", atomic_load_min_i64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_64 : F_ATOMIC_2<i64, Int64Regs, "", ".s64", ".min",
  atomic_load_min_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_64_USE_G : F_ATOMIC_2<i64, Int64Regs, ".global",
  ".s64", ".min", atomic_load_min_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMIN_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".u32",
  ".min", atomic_load_umin_i32_g, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMIN_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".u32",
  ".min", atomic_load_umin_i32_s, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".u32", ".min",
  atomic_load_umin_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global",
  ".u32", ".min", atomic_load_umin_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_LOAD_UMIN_G_64 : F_ATOMIC_2<i64, Int64Regs, ".global", ".u64",
  ".min", atomic_load_umin_i64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMIN_S_64 : F_ATOMIC_2<i64, Int64Regs, ".shared", ".u64",
  ".min", atomic_load_umin_i64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_64 : F_ATOMIC_2<i64, Int64Regs, "", ".u64", ".min",
  atomic_load_umin_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_64_USE_G : F_ATOMIC_2<i64, Int64Regs, ".global",
  ".u64", ".min", atomic_load_umin_i64_gen, i64imm, imm, [hasSM<32>]>;

// atom_inc  atom_dec

def atomic_load_inc_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_inc_32 node:$a, node:$b)>;
def atomic_load_inc_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_inc_32 node:$a, node:$b)>;
def atomic_load_inc_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_inc_32 node:$a, node:$b)>;
def atomic_load_dec_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_dec_32 node:$a, node:$b)>;
def atomic_load_dec_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_dec_32 node:$a, node:$b)>;
def atomic_load_dec_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_dec_32 node:$a, node:$b)>;

defm INT_PTX_ATOM_INC_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".u32", ".inc",
  atomic_load_inc_32_g, i32imm, imm>;
defm INT_PTX_ATOM_INC_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".u32", ".inc",
  atomic_load_inc_32_s, i32imm, imm>;
defm INT_PTX_ATOM_INC_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".u32", ".inc",
  atomic_load_inc_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_INC_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global", ".u32",
  ".inc", atomic_load_inc_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_DEC_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".u32", ".dec",
  atomic_load_dec_32_g, i32imm, imm>;
defm INT_PTX_ATOM_DEC_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".u32", ".dec",
  atomic_load_dec_32_s, i32imm, imm>;
defm INT_PTX_ATOM_DEC_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".u32", ".dec",
  atomic_load_dec_32_gen, i32imm, imm>;
defm INT_PTX_ATOM_DEC_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global", ".u32",
  ".dec", atomic_load_dec_32_gen, i32imm, imm>;

// atom_and

def atomic_load_and_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_and_i32 node:$a, node:$b)>;
def atomic_load_and_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_and_i32 node:$a, node:$b)>;
def atomic_load_and_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_and_i32 node:$a, node:$b)>;
def atomic_load_and_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_and_i64 node:$a, node:$b)>;
def atomic_load_and_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_and_i64 node:$a, node:$b)>;
def atomic_load_and_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_and_i64 node:$a, node:$b)>;

defm INT_PTX_ATOM_AND_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".b32", ".and",
  atomic_load_and_i32_g, i32imm, imm>;
defm INT_PTX_ATOM_AND_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".b32", ".and",
  atomic_load_and_i32_s, i32imm, imm>;
defm INT_PTX_ATOM_AND_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".b32", ".and",
  atomic_load_and_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_AND_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global", ".b32",
  ".and", atomic_load_and_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_AND_G_64 : F_ATOMIC_2<i64, Int64Regs, ".global", ".b64", ".and",
  atomic_load_and_i64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_AND_S_64 : F_ATOMIC_2<i64, Int64Regs, ".shared", ".b64", ".and",
  atomic_load_and_i64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_AND_GEN_64 : F_ATOMIC_2<i64, Int64Regs, "", ".b64", ".and",
  atomic_load_and_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_AND_GEN_64_USE_G : F_ATOMIC_2<i64, Int64Regs, ".global", ".b64",
  ".and", atomic_load_and_i64_gen, i64imm, imm, [hasSM<32>]>;

// atom_or

def atomic_load_or_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_or_i32 node:$a, node:$b)>;
def atomic_load_or_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_or_i32 node:$a, node:$b)>;
def atomic_load_or_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_or_i32 node:$a, node:$b)>;
def atomic_load_or_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_or_i64 node:$a, node:$b)>;
def atomic_load_or_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_or_i64 node:$a, node:$b)>;
def atomic_load_or_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_or_i64 node:$a, node:$b)>;

defm INT_PTX_ATOM_OR_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".b32", ".or",
  atomic_load_or_i32_g, i32imm, imm>;
defm INT_PTX_ATOM_OR_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".b32", ".or",
  atomic_load_or_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_OR_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global", ".b32",
  ".or", atomic_load_or_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_OR_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".b32", ".or",
  atomic_load_or_i32_s, i32imm, imm>;
defm INT_PTX_ATOM_OR_G_64 : F_ATOMIC_2<i64, Int64Regs, ".global", ".b64", ".or",
  atomic_load_or_i64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_OR_GEN_64 : F_ATOMIC_2<i64, Int64Regs, "", ".b64", ".or",
  atomic_load_or_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_OR_GEN_64_USE_G : F_ATOMIC_2<i64, Int64Regs, ".global", ".b64",
  ".or", atomic_load_or_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_OR_S_64 : F_ATOMIC_2<i64, Int64Regs, ".shared", ".b64", ".or",
  atomic_load_or_i64_s, i64imm, imm, [hasSM<32>]>;

// atom_xor

def atomic_load_xor_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_i32 node:$a, node:$b)>;
def atomic_load_xor_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_i32 node:$a, node:$b)>;
def atomic_load_xor_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_i32 node:$a, node:$b)>;
def atomic_load_xor_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_i64 node:$a, node:$b)>;
def atomic_load_xor_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_i64 node:$a, node:$b)>;
def atomic_load_xor_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_i64 node:$a, node:$b)>;

defm INT_PTX_ATOM_XOR_G_32 : F_ATOMIC_2<i32, Int32Regs, ".global", ".b32", ".xor",
  atomic_load_xor_i32_g, i32imm, imm>;
defm INT_PTX_ATOM_XOR_S_32 : F_ATOMIC_2<i32, Int32Regs, ".shared", ".b32", ".xor",
  atomic_load_xor_i32_s, i32imm, imm>;
defm INT_PTX_ATOM_XOR_GEN_32 : F_ATOMIC_2<i32, Int32Regs, "", ".b32", ".xor",
  atomic_load_xor_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_XOR_GEN_32_USE_G : F_ATOMIC_2<i32, Int32Regs, ".global", ".b32",
  ".xor", atomic_load_xor_i32_gen, i32imm, imm>;
defm INT_PTX_ATOM_XOR_G_64 : F_ATOMIC_2<i64, Int64Regs, ".global", ".b64", ".xor",
  atomic_load_xor_i64_g, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_XOR_S_64 : F_ATOMIC_2<i64, Int64Regs, ".shared", ".b64", ".xor",
  atomic_load_xor_i64_s, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_XOR_GEN_64 : F_ATOMIC_2<i64, Int64Regs, "", ".b64", ".xor",
  atomic_load_xor_i64_gen, i64imm, imm, [hasSM<32>]>;
defm INT_PTX_ATOM_XOR_GEN_64_USE_G : F_ATOMIC_2<i64, Int64Regs, ".global", ".b64",
  ".xor", atomic_load_xor_i64_gen, i64imm, imm, [hasSM<32>]>;

// atom_cas

def atomic_cmp_swap_i16_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_i16 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_i16_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_i16 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_i16_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_i16 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_i32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_i32 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_i32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_i32 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_i32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_i32 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_i64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_i64 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_i64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_i64 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_i64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_i64 node:$a, node:$b, node:$c)>;

defm INT_PTX_ATOM_CAS_G_16 : F_ATOMIC_3<i16, Int16Regs, ".global", ".b16", ".cas",
  atomic_cmp_swap_i16_g, i16imm, [hasSM<70>, hasPTX<63>]>;
defm INT_PTX_ATOM_CAS_S_16 : F_ATOMIC_3<i16, Int16Regs, ".shared", ".b16", ".cas",
  atomic_cmp_swap_i16_s, i16imm, [hasSM<70>, hasPTX<63>]>;
defm INT_PTX_ATOM_CAS_GEN_16 : F_ATOMIC_3<i16, Int16Regs, "", ".b16", ".cas",
  atomic_cmp_swap_i16_gen, i16imm, [hasSM<70>, hasPTX<63>]>;
defm INT_PTX_ATOM_CAS_GEN_16_USE_G : F_ATOMIC_3<i16, Int16Regs, ".global", ".b16", ".cas",
  atomic_cmp_swap_i16_gen, i16imm, [hasSM<70>, hasPTX<63>]>;
defm INT_PTX_ATOM_CAS_G_32 : F_ATOMIC_3<i32, Int32Regs, ".global", ".b32", ".cas",
  atomic_cmp_swap_i32_g, i32imm>;
defm INT_PTX_ATOM_CAS_S_32 : F_ATOMIC_3<i32, Int32Regs, ".shared", ".b32", ".cas",
  atomic_cmp_swap_i32_s, i32imm>;
defm INT_PTX_ATOM_CAS_GEN_32 : F_ATOMIC_3<i32, Int32Regs, "", ".b32", ".cas",
  atomic_cmp_swap_i32_gen, i32imm>;
defm INT_PTX_ATOM_CAS_GEN_32_USE_G : F_ATOMIC_3<i32, Int32Regs, ".global", ".b32",
  ".cas", atomic_cmp_swap_i32_gen, i32imm>;
defm INT_PTX_ATOM_CAS_G_64 : F_ATOMIC_3<i64, Int64Regs, ".global", ".b64", ".cas",
  atomic_cmp_swap_i64_g, i64imm>;
defm INT_PTX_ATOM_CAS_S_64 : F_ATOMIC_3<i64, Int64Regs, ".shared", ".b64", ".cas",
  atomic_cmp_swap_i64_s, i64imm>;
defm INT_PTX_ATOM_CAS_GEN_64 : F_ATOMIC_3<i64, Int64Regs, "", ".b64", ".cas",
  atomic_cmp_swap_i64_gen, i64imm>;
defm INT_PTX_ATOM_CAS_GEN_64_USE_G : F_ATOMIC_3<i64, Int64Regs, ".global", ".b64",
  ".cas", atomic_cmp_swap_i64_gen, i64imm>;

// Support for scoped atomic operations.  Matches
// int_nvvm_atomic_{op}_{space}_{type}_{scope}
// and converts it into the appropriate instruction.
// NOTE: not all possible combinations are implemented
//  'space' is limited to generic as it's the only one needed to support CUDA.
//  'scope' = 'gpu' is default and is handled by regular atomic instructions.
class ATOM23_impl<string AsmStr, ValueType regT, NVPTXRegClass regclass, list<Predicate> Preds,
                  dag ins, dag Operands>
      : NVPTXInst<(outs regclass:$result), ins,
                  AsmStr,
                  [(set regT:$result, Operands)]>,
        Requires<Preds>;

// Define instruction variants for all addressing modes.
multiclass ATOM2P_impl<string AsmStr,  Intrinsic Intr,
                       ValueType regT, NVPTXRegClass regclass, Operand ImmType,
                       SDNode Imm, ValueType ImmTy,
                       list<Predicate> Preds> {
  let AddedComplexity = 1 in {
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins Int16Regs:$src, regclass:$b),
                      (Intr i16:$src, regT:$b)>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins Int32Regs:$src, regclass:$b),
                      (Intr i32:$src, regT:$b)>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins Int64Regs:$src, regclass:$b),
                      (Intr i64:$src, regT:$b)>;
  }
  // tablegen can't infer argument types from Intrinsic (though it can
  // from Instruction) so we have to enforce specific type on
  // immediates via explicit cast to ImmTy.
  def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                    (ins Int16Regs:$src, ImmType:$b),
                    (Intr i16:$src, (ImmTy Imm:$b))>;
  def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                    (ins Int32Regs:$src, ImmType:$b),
                    (Intr i32:$src, (ImmTy Imm:$b))>;
  def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                    (ins Int64Regs:$src, ImmType:$b),
                    (Intr i64:$src, (ImmTy Imm:$b))>;
}

multiclass ATOM3P_impl<string AsmStr,  Intrinsic Intr,
                       ValueType regT, NVPTXRegClass regclass,
                       Operand ImmType, SDNode Imm, ValueType ImmTy,
                       list<Predicate> Preds> {
  // Variants for register/immediate permutations of $b and $c
  let AddedComplexity = 2 in {
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins Int32Regs:$src, regclass:$b, regclass:$c),
                      (Intr i32:$src, regT:$b, regT:$c)>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins Int64Regs:$src, regclass:$b, regclass:$c),
                      (Intr i64:$src, regT:$b, regT:$c)>;
  }
  let AddedComplexity = 1 in {
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins Int32Regs:$src, ImmType:$b, regclass:$c),
                      (Intr i32:$src, (ImmTy Imm:$b), regT:$c)>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins Int64Regs:$src, ImmType:$b, regclass:$c),
                      (Intr i64:$src, (ImmTy Imm:$b), regT:$c)>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins Int32Regs:$src, regclass:$b, ImmType:$c),
                      (Intr i32:$src, regT:$b, (ImmTy Imm:$c))>;
    def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                      (ins Int64Regs:$src, regclass:$b, ImmType:$c),
                      (Intr i64:$src, regT:$b, (ImmTy Imm:$c))>;
  }
  def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                    (ins Int32Regs:$src, ImmType:$b, ImmType:$c),
                    (Intr i32:$src, (ImmTy Imm:$b), (ImmTy Imm:$c))>;
  def : ATOM23_impl<AsmStr, regT, regclass, Preds,
                    (ins Int64Regs:$src, ImmType:$b, ImmType:$c),
                    (Intr i64:$src, (ImmTy Imm:$b), (ImmTy Imm:$c))>;
}

// Constructs intrinsic name and instruction asm strings.
multiclass ATOM2N_impl<string OpStr, string IntTypeStr, string TypeStr,
                       string ScopeStr, string SpaceStr,
                       ValueType regT, NVPTXRegClass regclass, Operand ImmType, SDNode Imm,
                       ValueType ImmTy, list<Predicate> Preds> {
  defm : ATOM2P_impl<"atom" # !if(!eq(SpaceStr, "gen"), "", "." # SpaceStr)
                            # !if(!eq(ScopeStr, "gpu"), "", "." # ScopeStr)
                            # "." # OpStr # "." # TypeStr
                            # " \t$result, [$src], $b;",
                     !cast<Intrinsic>(
                            "int_nvvm_atomic_" # OpStr
                            # "_" # SpaceStr # "_" # IntTypeStr
                            # !if(!empty(ScopeStr), "", "_" # ScopeStr)),
                     regT, regclass, ImmType, Imm, ImmTy, Preds>;
}
multiclass ATOM3N_impl<string OpStr, string IntTypeStr, string TypeStr,
                       string ScopeStr, string SpaceStr,
                       ValueType regT, NVPTXRegClass regclass, Operand ImmType, SDNode Imm,
                       ValueType ImmTy, list<Predicate> Preds> {
  defm : ATOM3P_impl<"atom" # !if(!eq(SpaceStr, "gen"), "", "." # SpaceStr)
                            # !if(!eq(ScopeStr, "gpu"), "", "." # ScopeStr)
                            # "." # OpStr # "." # TypeStr
                            # " \t$result, [$src], $b, $c;",
                     !cast<Intrinsic>(
                            "int_nvvm_atomic_" # OpStr
                            # "_" # SpaceStr # "_" # IntTypeStr
                            # !if(!empty(ScopeStr), "", "_" # ScopeStr)),
                     regT, regclass, ImmType, Imm, ImmTy, Preds>;
}

// Constructs variants for different address spaces.
// For now we only need variants for generic space pointers.
multiclass ATOM2A_impl<string OpStr, string IntTypeStr, string TypeStr,
                       string ScopeStr, ValueType regT, NVPTXRegClass regclass, Operand ImmType,
                       SDNode Imm, ValueType ImmTy, list<Predicate> Preds> {
   defm _gen_ : ATOM2N_impl<OpStr, IntTypeStr, TypeStr, ScopeStr, "gen",
                            regT, regclass, ImmType, Imm, ImmTy, Preds>;
}
multiclass ATOM3A_impl<string OpStr, string IntTypeStr, string TypeStr,
                       string ScopeStr, ValueType regT, NVPTXRegClass regclass, Operand ImmType,
                       SDNode Imm, ValueType ImmTy, list<Predicate> Preds> {
   defm _gen_ : ATOM3N_impl<OpStr, IntTypeStr, TypeStr, ScopeStr, "gen",
                            regT, regclass, ImmType, Imm, ImmTy, Preds>;
}

// Constructs variants for different scopes of atomic op.
multiclass ATOM2S_impl<string OpStr, string IntTypeStr, string TypeStr,
                       ValueType regT, NVPTXRegClass regclass, Operand ImmType, SDNode Imm,
                       ValueType ImmTy, list<Predicate> Preds> {
   // .gpu scope is default and is currently covered by existing
   // atomics w/o explicitly specified scope.
   defm _cta : ATOM2A_impl<OpStr, IntTypeStr, TypeStr, "cta",
                           regT, regclass, ImmType, Imm, ImmTy,
                           !listconcat(Preds,[hasAtomScope])>;
   defm _sys : ATOM2A_impl<OpStr, IntTypeStr, TypeStr, "sys",
                           regT, regclass, ImmType, Imm, ImmTy,
                           !listconcat(Preds,[hasAtomScope])>;
}
multiclass ATOM3S_impl<string OpStr, string IntTypeStr, string TypeStr,
           ValueType regT, NVPTXRegClass regclass, Operand ImmType, SDNode Imm, ValueType ImmTy,
           list<Predicate> Preds> {
   // No need to define ".gpu"-scoped atomics.  They do the same thing
   // as the regular, non-scoped atomics defined elsewhere.
   defm _cta : ATOM3A_impl<OpStr, IntTypeStr, TypeStr, "cta",
                           regT, regclass, ImmType, Imm, ImmTy,
                           !listconcat(Preds,[hasAtomScope])>;
   defm _sys : ATOM3A_impl<OpStr, IntTypeStr, TypeStr, "sys",
                           regT, regclass, ImmType, Imm, ImmTy,
                           !listconcat(Preds,[hasAtomScope])>;
}

// atom.add
multiclass ATOM2_add_impl<string OpStr> {
   defm _s32  : ATOM2S_impl<OpStr, "i", "s32", i32, Int32Regs, i32imm, imm, i32, []>;
   defm _u32  : ATOM2S_impl<OpStr, "i", "u32", i32, Int32Regs, i32imm, imm, i32, []>;
   defm _u64  : ATOM2S_impl<OpStr, "i", "u64", i64, Int64Regs, i64imm, imm, i64, []>;
   defm _bf16  : ATOM2S_impl<OpStr, "f", "bf16", bf16, Int16Regs, bf16imm, fpimm, bf16,
                            [hasSM<90>, hasPTX<78>]>;
   defm _f16  : ATOM2S_impl<OpStr, "f", "f16", f16, Int16Regs, f16imm, fpimm, f16,
                            [hasSM<70>, hasPTX<63>]>;
   defm _f32  : ATOM2S_impl<OpStr, "f", "f32", f32, Float32Regs, f32imm, fpimm, f32,
                            []>;
   defm _f64  : ATOM2S_impl<OpStr, "f", "f64", f64, Float64Regs, f64imm, fpimm, f64,
                            [hasAtomAddF64]>;
}

// atom.{and,or,xor}
multiclass ATOM2_bitwise_impl<string OpStr> {
   defm _b32  : ATOM2S_impl<OpStr, "i", "b32", i32, Int32Regs, i32imm, imm, i32, []>;
   defm _b64  : ATOM2S_impl<OpStr, "i", "b64", i64, Int64Regs, i64imm, imm, i64,
                            [hasAtomBitwise64]>;
}

// atom.exch
multiclass ATOM2_exch_impl<string OpStr> {
   defm _b32 : ATOM2S_impl<OpStr, "i", "b32", i32, Int32Regs, i32imm, imm, i32, []>;
   defm _b64 : ATOM2S_impl<OpStr, "i", "b64", i64, Int64Regs, i64imm, imm, i64, []>;
}

// atom.{min,max}
multiclass ATOM2_minmax_impl<string OpStr> {
   defm _s32  : ATOM2S_impl<OpStr, "i", "s32", i32, Int32Regs, i32imm, imm, i32, []>;
   defm _u32  : ATOM2S_impl<OpStr, "i", "u32", i32, Int32Regs, i32imm, imm, i32, []>;
   defm _s64  : ATOM2S_impl<OpStr, "i", "s64", i64, Int64Regs, i64imm, imm, i64,
                            [hasAtomMinMax64]>;
   defm _u64  : ATOM2S_impl<OpStr, "i", "u64", i64, Int64Regs, i64imm, imm, i64,
                            [hasAtomMinMax64]>;
}

// atom.{inc,dec}
multiclass ATOM2_incdec_impl<string OpStr> {
   defm _u32  : ATOM2S_impl<OpStr, "i", "u32", i32, Int32Regs, i32imm, imm, i32, []>;
}

// atom.cas
multiclass ATOM3_cas_impl<string OpStr> {
   defm _b16  : ATOM3S_impl<OpStr, "i", "b16", i16, Int16Regs, i16imm, imm, i16, []>;
   defm _b32  : ATOM3S_impl<OpStr, "i", "b32", i32, Int32Regs, i32imm, imm, i32, []>;
   defm _b64  : ATOM3S_impl<OpStr, "i", "b64", i64, Int64Regs, i64imm, imm, i64, []>;
}

defm INT_PTX_SATOM_ADD : ATOM2_add_impl<"add">;
defm INT_PTX_SATOM_AND : ATOM2_bitwise_impl<"and">;
defm INT_PTX_SATOM_CAS : ATOM3_cas_impl<"cas">;
defm INT_PTX_SATOM_DEC : ATOM2_incdec_impl<"dec">;
defm INT_PTX_SATOM_EXCH: ATOM2_exch_impl<"exch">;
defm INT_PTX_SATOM_INC : ATOM2_incdec_impl<"inc">;
defm INT_PTX_SATOM_MAX : ATOM2_minmax_impl<"max">;
defm INT_PTX_SATOM_MIN : ATOM2_minmax_impl<"min">;
defm INT_PTX_SATOM_OR  : ATOM2_bitwise_impl<"or">;
defm INT_PTX_SATOM_XOR : ATOM2_bitwise_impl<"xor">;

//-----------------------------------
// Support for ldu on sm_20 or later
//-----------------------------------

// Don't annotate ldu instructions as mayLoad, as they load from memory that is
// read-only in a kernel.

// Scalar

multiclass LDU_G<string TyStr, NVPTXRegClass regclass> {
  def areg: NVPTXInst<(outs regclass:$result), (ins Int32Regs:$src),
               !strconcat("ldu.global.", TyStr),
                      []>, Requires<[hasLDU]>;
  def areg64: NVPTXInst<(outs regclass:$result), (ins Int64Regs:$src),
               !strconcat("ldu.global.", TyStr),
                        []>, Requires<[hasLDU]>;
 def avar:  NVPTXInst<(outs regclass:$result), (ins imemAny:$src),
               !strconcat("ldu.global.", TyStr),
                      []>, Requires<[hasLDU]>;
 def ari :  NVPTXInst<(outs regclass:$result), (ins MEMri:$src),
               !strconcat("ldu.global.", TyStr),
                      []>, Requires<[hasLDU]>;
 def ari64 :  NVPTXInst<(outs regclass:$result), (ins MEMri64:$src),
               !strconcat("ldu.global.", TyStr),
                        []>, Requires<[hasLDU]>;
}

defm INT_PTX_LDU_GLOBAL_i8  : LDU_G<"u8 \t$result, [$src];", Int16Regs>;
defm INT_PTX_LDU_GLOBAL_i16 : LDU_G<"u16 \t$result, [$src];", Int16Regs>;
defm INT_PTX_LDU_GLOBAL_i32 : LDU_G<"u32 \t$result, [$src];", Int32Regs>;
defm INT_PTX_LDU_GLOBAL_i64 : LDU_G<"u64 \t$result, [$src];", Int64Regs>;
defm INT_PTX_LDU_GLOBAL_f32 : LDU_G<"f32 \t$result, [$src];", Float32Regs>;
defm INT_PTX_LDU_GLOBAL_f64 : LDU_G<"f64 \t$result, [$src];", Float64Regs>;

// vector

// Elementized vector ldu
multiclass VLDU_G_ELE_V2<string TyStr, NVPTXRegClass regclass> {
 def _areg32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins Int32Regs:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _areg64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins Int64Regs:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _ari32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _ari64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri64:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _avar: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins imemAny:$src),
                     !strconcat("ldu.global.", TyStr), []>;
}

multiclass VLDU_G_ELE_V4<string TyStr, NVPTXRegClass regclass> {
 def _areg32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins Int32Regs:$src),
               !strconcat("ldu.global.", TyStr), []>;
 def _areg64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins Int64Regs:$src),
               !strconcat("ldu.global.", TyStr), []>;
 def _ari32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins MEMri:$src),
               !strconcat("ldu.global.", TyStr), []>;
 def _ari64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins MEMri64:$src),
               !strconcat("ldu.global.", TyStr), []>;
 def _avar: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins imemAny:$src),
               !strconcat("ldu.global.", TyStr), []>;
}

defm INT_PTX_LDU_G_v2i8_ELE
  : VLDU_G_ELE_V2<"v2.u8 \t{{$dst1, $dst2}}, [$src];",  Int16Regs>;
defm INT_PTX_LDU_G_v2i16_ELE
  : VLDU_G_ELE_V2<"v2.u16 \t{{$dst1, $dst2}}, [$src];", Int16Regs>;
defm INT_PTX_LDU_G_v2i32_ELE
  : VLDU_G_ELE_V2<"v2.u32 \t{{$dst1, $dst2}}, [$src];", Int32Regs>;
defm INT_PTX_LDU_G_v2f32_ELE
  : VLDU_G_ELE_V2<"v2.f32 \t{{$dst1, $dst2}}, [$src];", Float32Regs>;
defm INT_PTX_LDU_G_v2i64_ELE
  : VLDU_G_ELE_V2<"v2.u64 \t{{$dst1, $dst2}}, [$src];", Int64Regs>;
defm INT_PTX_LDU_G_v2f64_ELE
  : VLDU_G_ELE_V2<"v2.f64 \t{{$dst1, $dst2}}, [$src];", Float64Regs>;
defm INT_PTX_LDU_G_v4i8_ELE
  : VLDU_G_ELE_V4<"v4.u8 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Int16Regs>;
defm INT_PTX_LDU_G_v4i16_ELE
  : VLDU_G_ELE_V4<"v4.u16 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];",
    Int16Regs>;
defm INT_PTX_LDU_G_v4i32_ELE
  : VLDU_G_ELE_V4<"v4.u32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];",
    Int32Regs>;
defm INT_PTX_LDU_G_v4f16_ELE
  : VLDU_G_ELE_V4<"v4.b16 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];",
    Int16Regs>;
defm INT_PTX_LDU_G_v4f16x2_ELE
  : VLDU_G_ELE_V4<"v4.b32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];",
    Int32Regs>;
defm INT_PTX_LDU_G_v4f32_ELE
  : VLDU_G_ELE_V4<"v4.f32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];",
    Float32Regs>;


//-----------------------------------
// Support for ldg on sm_35 or later
//-----------------------------------

// Don't annotate ld.global.nc as mayLoad, because these loads go through the
// non-coherent texture cache, and therefore the values read must be read-only
// during the lifetime of the kernel.

multiclass LDG_G<string TyStr, NVPTXRegClass regclass> {
  def areg: NVPTXInst<(outs regclass:$result), (ins Int32Regs:$src),
               !strconcat("ld.global.nc.", TyStr),
                      []>, Requires<[hasLDG]>;
  def areg64: NVPTXInst<(outs regclass:$result), (ins Int64Regs:$src),
               !strconcat("ld.global.nc.", TyStr),
                        []>, Requires<[hasLDG]>;
 def avar:  NVPTXInst<(outs regclass:$result), (ins imemAny:$src),
               !strconcat("ld.global.nc.", TyStr),
                      []>, Requires<[hasLDG]>;
 def ari :  NVPTXInst<(outs regclass:$result), (ins MEMri:$src),
               !strconcat("ld.global.nc.", TyStr),
                      []>, Requires<[hasLDG]>;
 def ari64 :  NVPTXInst<(outs regclass:$result), (ins MEMri64:$src),
               !strconcat("ld.global.nc.", TyStr),
                        []>, Requires<[hasLDG]>;
}

defm INT_PTX_LDG_GLOBAL_i8
  : LDG_G<"u8 \t$result, [$src];", Int16Regs>;
defm INT_PTX_LDG_GLOBAL_i16
  : LDG_G<"u16 \t$result, [$src];", Int16Regs>;
defm INT_PTX_LDG_GLOBAL_i32
  : LDG_G<"u32 \t$result, [$src];", Int32Regs>;
defm INT_PTX_LDG_GLOBAL_i64
  : LDG_G<"u64 \t$result, [$src];", Int64Regs>;
defm INT_PTX_LDG_GLOBAL_f32
  : LDG_G<"f32 \t$result, [$src];", Float32Regs>;
defm INT_PTX_LDG_GLOBAL_f64
  : LDG_G<"f64 \t$result, [$src];", Float64Regs>;

// vector

// Elementized vector ldg
multiclass VLDG_G_ELE_V2<string TyStr, NVPTXRegClass regclass> {
 def _areg32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins Int32Regs:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _areg64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins Int64Regs:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _ari32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _ari64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri64:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _avar: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins imemAny:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
}

multiclass VLDG_G_ELE_V4<string TyStr, NVPTXRegClass regclass> {
  def _areg32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                              regclass:$dst4), (ins Int32Regs:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
  def _areg64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                               regclass:$dst4), (ins Int64Regs:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
  def _ari32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                              regclass:$dst4), (ins MEMri:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
  def _ari64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                              regclass:$dst4), (ins MEMri64:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
  def _avar: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                             regclass:$dst4), (ins imemAny:$src),
               !strconcat("ld.global.nc.", TyStr), []>;
}

// FIXME: 8-bit LDG should be fixed once LDG/LDU nodes are made into proper loads.
defm INT_PTX_LDG_G_v2i8_ELE
  : VLDG_G_ELE_V2<"v2.u8 \t{{$dst1, $dst2}}, [$src];",  Int16Regs>;
defm INT_PTX_LDG_G_v2i16_ELE
  : VLDG_G_ELE_V2<"v2.u16 \t{{$dst1, $dst2}}, [$src];", Int16Regs>;
defm INT_PTX_LDG_G_v2i32_ELE
  : VLDG_G_ELE_V2<"v2.u32 \t{{$dst1, $dst2}}, [$src];", Int32Regs>;
defm INT_PTX_LDG_G_v2f32_ELE
  : VLDG_G_ELE_V2<"v2.f32 \t{{$dst1, $dst2}}, [$src];", Float32Regs>;
defm INT_PTX_LDG_G_v2i64_ELE
  : VLDG_G_ELE_V2<"v2.u64 \t{{$dst1, $dst2}}, [$src];", Int64Regs>;
defm INT_PTX_LDG_G_v2f64_ELE
  : VLDG_G_ELE_V2<"v2.f64 \t{{$dst1, $dst2}}, [$src];", Float64Regs>;
defm INT_PTX_LDG_G_v4i8_ELE
  : VLDG_G_ELE_V4<"v4.u8 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Int16Regs>;
defm INT_PTX_LDG_G_v4i16_ELE
  : VLDG_G_ELE_V4<"v4.u16 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Int16Regs>;
defm INT_PTX_LDG_G_v4i32_ELE
  : VLDG_G_ELE_V4<"v4.u32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Int32Regs>;
defm INT_PTX_LDG_G_v4f32_ELE
  : VLDG_G_ELE_V4<"v4.f32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Float32Regs>;


multiclass NG_TO_G<string Str> {
   def "" : NVPTXInst<(outs Int32Regs:$result), (ins Int32Regs:$src),
          "cvta." # Str # ".u32 \t$result, $src;", []>;
   def _64 : NVPTXInst<(outs Int64Regs:$result), (ins Int64Regs:$src),
          "cvta." # Str # ".u64 \t$result, $src;", []>;
}

multiclass G_TO_NG<string Str> {
   def "" : NVPTXInst<(outs Int32Regs:$result), (ins Int32Regs:$src),
          "cvta.to." # Str # ".u32 \t$result, $src;", []>;
   def _64 : NVPTXInst<(outs Int64Regs:$result), (ins Int64Regs:$src),
          "cvta.to." # Str # ".u64 \t$result, $src;", []>;
}

defm cvta_local  : NG_TO_G<"local">;
defm cvta_shared : NG_TO_G<"shared">;
defm cvta_global : NG_TO_G<"global">;
defm cvta_const  : NG_TO_G<"const">;

defm cvta_to_local  : G_TO_NG<"local">;
defm cvta_to_shared : G_TO_NG<"shared">;
defm cvta_to_global : G_TO_NG<"global">;
defm cvta_to_const  : G_TO_NG<"const">;

// nvvm.ptr.param.to.gen
defm cvta_param : NG_TO_G<"param">;

def : Pat<(int_nvvm_ptr_param_to_gen i32:$src),
          (cvta_param $src)>;

def : Pat<(int_nvvm_ptr_param_to_gen i64:$src),
          (cvta_param_64 $src)>;

// nvvm.ptr.gen.to.param
def : Pat<(int_nvvm_ptr_gen_to_param i32:$src),
          (i32 Int32Regs:$src)>;

def : Pat<(int_nvvm_ptr_gen_to_param i64:$src),
          (i64 Int64Regs:$src)>;

// nvvm.move intrinsicc
def nvvm_move_i16 : NVPTXInst<(outs Int16Regs:$r), (ins Int16Regs:$s),
                             "mov.b16 \t$r, $s;",
                             [(set i16:$r,
                               (int_nvvm_move_i16 i16:$s))]>;
def nvvm_move_i32 : NVPTXInst<(outs Int32Regs:$r), (ins Int32Regs:$s),
                             "mov.b32 \t$r, $s;",
                             [(set i32:$r,
                               (int_nvvm_move_i32 i32:$s))]>;
def nvvm_move_i64 : NVPTXInst<(outs Int64Regs:$r), (ins Int64Regs:$s),
                             "mov.b64 \t$r, $s;",
                             [(set i64:$r,
                               (int_nvvm_move_i64 i64:$s))]>;
def nvvm_move_float : NVPTXInst<(outs Float32Regs:$r), (ins Float32Regs:$s),
                             "mov.f32 \t$r, $s;",
                             [(set f32:$r,
                               (int_nvvm_move_float f32:$s))]>;
def nvvm_move_double : NVPTXInst<(outs Float64Regs:$r), (ins Float64Regs:$s),
                             "mov.f64 \t$r, $s;",
                             [(set f64:$r,
                               (int_nvvm_move_double f64:$s))]>;
def nvvm_move_ptr32 : NVPTXInst<(outs Int32Regs:$r), (ins Int32Regs:$s),
                             "mov.u32 \t$r, $s;",
                             [(set i32:$r,
                               (int_nvvm_move_ptr i32:$s))]>;
def nvvm_move_ptr64 : NVPTXInst<(outs Int64Regs:$r), (ins Int64Regs:$s),
                             "mov.u64 \t$r, $s;",
                             [(set i64:$r,
                               (int_nvvm_move_ptr i64:$s))]>;

// @TODO: Are these actually needed, or will we always just see symbols
// copied to registers first?
/*def nvvm_move_sym32 : NVPTXInst<(outs Int32Regs:$r), (ins imem:$s),
                             "mov.u32 \t$r, $s;",
                             [(set Int32Regs:$r,
                             (int_nvvm_move_ptr texternalsym:$s))]>;
def nvvm_move_sym64 : NVPTXInst<(outs Int64Regs:$r), (ins imem:$s),
                             "mov.u64 \t$r, $s;",
                             [(set Int64Regs:$r,
                             (int_nvvm_move_ptr texternalsym:$s))]>;*/

def texsurf_handles
  : NVPTXInst<(outs Int64Regs:$result), (ins imem:$src),
              "mov.u64 \t$result, $src;", []>;

//-----------------------------------
// Compiler Error Warn
// - Just ignore them in codegen
//-----------------------------------

def INT_NVVM_COMPILER_WARN_32 : NVPTXInst<(outs), (ins Int32Regs:$a),
                "// llvm.nvvm.compiler.warn()",
                [(int_nvvm_compiler_warn i32:$a)]>;
def INT_NVVM_COMPILER_WARN_64 : NVPTXInst<(outs), (ins Int64Regs:$a),
                "// llvm.nvvm.compiler.warn()",
                [(int_nvvm_compiler_warn i64:$a)]>;
def INT_NVVM_COMPILER_ERROR_32 : NVPTXInst<(outs), (ins Int32Regs:$a),
                "// llvm.nvvm.compiler.error()",
                [(int_nvvm_compiler_error i32:$a)]>;
def INT_NVVM_COMPILER_ERROR_64 : NVPTXInst<(outs), (ins Int64Regs:$a),
                "// llvm.nvvm.compiler.error()",
                [(int_nvvm_compiler_error i64:$a)]>;


// isspacep

multiclass ISSPACEP<string suffix, Intrinsic Intr, list<Predicate> Preds = []> {
  def _32: NVPTXInst<(outs Int1Regs:$d), (ins Int32Regs:$a),
              "isspacep." # suffix # "\t$d, $a;",
              [(set i1:$d, (Intr i32:$a))]>,
    Requires<Preds>;
  def _64: NVPTXInst<(outs Int1Regs:$d), (ins Int64Regs:$a),
              "isspacep." # suffix # "\t$d, $a;",
              [(set i1:$d, (Intr i64:$a))]>,
    Requires<Preds>;
}

defm isspace_const  : ISSPACEP<"const", int_nvvm_isspacep_const, [hasPTX<31>]>;
defm isspace_global : ISSPACEP<"global", int_nvvm_isspacep_global>;
defm isspace_local  : ISSPACEP<"local", int_nvvm_isspacep_local>;
defm isspace_shared : ISSPACEP<"shared", int_nvvm_isspacep_shared>;
defm isspace_shared_cluster : ISSPACEP<"shared::cluster",
                                       int_nvvm_isspacep_shared_cluster,
                                       [hasPTX<78>, hasSM<90>]>;

// Special register reads
def MOV_SPECIAL : NVPTXInst<(outs Int32Regs:$d),
                            (ins SpecialRegs:$r),
                            "mov.b32 \t$d, $r;", []>;

def : Pat<(int_nvvm_read_ptx_sreg_envreg0), (MOV_SPECIAL ENVREG0)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg1), (MOV_SPECIAL ENVREG1)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg2), (MOV_SPECIAL ENVREG2)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg3), (MOV_SPECIAL ENVREG3)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg4), (MOV_SPECIAL ENVREG4)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg5), (MOV_SPECIAL ENVREG5)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg6), (MOV_SPECIAL ENVREG6)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg7), (MOV_SPECIAL ENVREG7)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg8), (MOV_SPECIAL ENVREG8)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg9), (MOV_SPECIAL ENVREG9)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg10), (MOV_SPECIAL ENVREG10)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg11), (MOV_SPECIAL ENVREG11)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg12), (MOV_SPECIAL ENVREG12)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg13), (MOV_SPECIAL ENVREG13)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg14), (MOV_SPECIAL ENVREG14)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg15), (MOV_SPECIAL ENVREG15)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg16), (MOV_SPECIAL ENVREG16)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg17), (MOV_SPECIAL ENVREG17)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg18), (MOV_SPECIAL ENVREG18)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg19), (MOV_SPECIAL ENVREG19)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg20), (MOV_SPECIAL ENVREG20)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg21), (MOV_SPECIAL ENVREG21)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg22), (MOV_SPECIAL ENVREG22)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg23), (MOV_SPECIAL ENVREG23)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg24), (MOV_SPECIAL ENVREG24)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg25), (MOV_SPECIAL ENVREG25)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg26), (MOV_SPECIAL ENVREG26)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg27), (MOV_SPECIAL ENVREG27)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg28), (MOV_SPECIAL ENVREG28)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg29), (MOV_SPECIAL ENVREG29)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg30), (MOV_SPECIAL ENVREG30)>;
def : Pat<(int_nvvm_read_ptx_sreg_envreg31), (MOV_SPECIAL ENVREG31)>;


def : Pat<(int_nvvm_swap_lo_hi_b64 i64:$src),
          (V2I32toI64 (I64toI32H $src),
                      (I64toI32L $src))> ;

//-----------------------------------
// Texture Intrinsics
//-----------------------------------

// NOTE: For Fermi support, any new texture/surface/sampler intrinsics must be
// also defined in NVPTXReplaceImageHandles.cpp

// texmode_independent
let IsTex = true, IsTexModeUnified = false in {
// Texture fetch instructions using handles

class TEX_1D_base<string inst, NVPTXRegClass outtype,
                  NVPTXRegClass intype, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x\\}];",
                 pattern>;

multiclass TEX_1D<string inst, NVPTXRegClass outtype, NVPTXRegClass intype,
                  Intrinsic intr> {
  def _RR : TEX_1D_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x))]>;
  def _RI : TEX_1D_base<inst, outtype, intype,
                        (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_1D_base<inst, outtype, intype,
                        (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_1D_base<inst, outtype, intype,
                        (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_F32_S32 : TEX_1D<"tex.1d.v4.f32.s32", Float32Regs, Int32Regs,
                             int_nvvm_tex_1d_v4f32_s32>;
defm TEX_1D_F32_F32 : TEX_1D<"tex.1d.v4.f32.f32", Float32Regs, Float32Regs,
                             int_nvvm_tex_1d_v4f32_f32>;
defm TEX_1D_S32_S32 : TEX_1D<"tex.1d.v4.s32.s32", Int32Regs, Int32Regs,
                             int_nvvm_tex_1d_v4s32_s32>;
defm TEX_1D_S32_F32 : TEX_1D<"tex.1d.v4.s32.f32", Int32Regs, Float32Regs,
                             int_nvvm_tex_1d_v4s32_f32>;
defm TEX_1D_U32_S32 : TEX_1D<"tex.1d.v4.u32.s32", Int32Regs, Int32Regs,
                             int_nvvm_tex_1d_v4u32_s32>;
defm TEX_1D_U32_F32 : TEX_1D<"tex.1d.v4.u32.f32", Int32Regs, Float32Regs,
                             int_nvvm_tex_1d_v4u32_f32>;

class TEX_1D_LEVEL_base<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, dag texsamp,
                        list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x\\}], $lod;",
                 pattern>;

multiclass TEX_1D_LEVEL<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_1D_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$lod))]>;
  def _RI : TEX_1D_LEVEL_base<inst, outtype, intype,
                              (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_1D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_1D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_F32_F32_LEVEL :
  TEX_1D_LEVEL<"tex.level.1d.v4.f32.f32", Float32Regs, Float32Regs,
               int_nvvm_tex_1d_level_v4f32_f32>;
defm TEX_1D_S32_F32_LEVEL :
  TEX_1D_LEVEL<"tex.level.1d.v4.s32.f32", Int32Regs, Float32Regs,
               int_nvvm_tex_1d_level_v4s32_f32>;
defm TEX_1D_U32_F32_LEVEL :
  TEX_1D_LEVEL<"tex.level.1d.v4.u32.f32", Int32Regs, Float32Regs,
               int_nvvm_tex_1d_level_v4u32_f32>;

class TEX_1D_GRAD_base<string inst, NVPTXRegClass outtype,
                       NVPTXRegClass intype, dag texsamp,
                       list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$gradx, intype:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x\\}],"
                        " \\{$gradx\\}, \\{$grady\\};",
                 pattern>;

multiclass TEX_1D_GRAD<string inst, NVPTXRegClass outtype,
                       NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_1D_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$gradx, intype:$grady))]>;
  def _RI : TEX_1D_GRAD_base<inst, outtype, intype,
                             (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_1D_GRAD_base<inst, outtype, intype,
                             (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_1D_GRAD_base<inst, outtype, intype,
                             (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_F32_F32_GRAD
  : TEX_1D_GRAD<"tex.grad.1d.v4.f32.f32", Float32Regs, Float32Regs,
                int_nvvm_tex_1d_grad_v4f32_f32>;
defm TEX_1D_S32_F32_GRAD
  : TEX_1D_GRAD<"tex.grad.1d.v4.s32.f32", Int32Regs, Float32Regs,
                int_nvvm_tex_1d_grad_v4s32_f32>;
defm TEX_1D_U32_F32_GRAD
  : TEX_1D_GRAD<"tex.grad.1d.v4.u32.f32", Int32Regs, Float32Regs,
                int_nvvm_tex_1d_grad_v4u32_f32>;

class TEX_1D_ARRAY_base<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, dag texsamp,
                        list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins Int32Regs:$l, intype:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$l, $x\\}];",
                 pattern>;

multiclass TEX_1D_ARRAY<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_1D_ARRAY_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, i32:$l, intype:$x))]>;
  def _RI : TEX_1D_ARRAY_base<inst, outtype, intype,
                              (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_1D_ARRAY_base<inst, outtype, intype,
                              (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_1D_ARRAY_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_ARRAY_F32_F32
  : TEX_1D_ARRAY<"tex.a1d.v4.f32.f32", Float32Regs, Float32Regs,
                 int_nvvm_tex_1d_array_v4f32_f32>;
defm TEX_1D_ARRAY_F32_S32
  : TEX_1D_ARRAY<"tex.a1d.v4.f32.s32", Float32Regs, Int32Regs,
                 int_nvvm_tex_1d_array_v4f32_s32>;
defm TEX_1D_ARRAY_S32_S32
  : TEX_1D_ARRAY<"tex.a1d.v4.s32.s32", Int32Regs, Int32Regs,
                 int_nvvm_tex_1d_array_v4s32_s32>;
defm TEX_1D_ARRAY_S32_F32
  : TEX_1D_ARRAY<"tex.a1d.v4.s32.f32", Int32Regs, Float32Regs,
                 int_nvvm_tex_1d_array_v4s32_f32>;
defm TEX_1D_ARRAY_U32_S32
  : TEX_1D_ARRAY<"tex.a1d.v4.u32.s32", Int32Regs, Int32Regs,
                 int_nvvm_tex_1d_array_v4u32_s32>;
defm TEX_1D_ARRAY_U32_F32
  : TEX_1D_ARRAY<"tex.a1d.v4.u32.f32", Int32Regs, Float32Regs,
                 int_nvvm_tex_1d_array_v4u32_f32>;

class TEX_1D_ARRAY_LEVEL_base<string inst, NVPTXRegClass outtype,
                              NVPTXRegClass intype, dag texsamp,
                              list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins Int32Regs:$l, intype:$x, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x\\}], $lod;",
                 pattern>;

multiclass TEX_1D_ARRAY_LEVEL<string inst, NVPTXRegClass outtype,
                              NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_1D_ARRAY_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, i32:$l, intype:$x, intype:$lod))]>;
  def _RI : TEX_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                    (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                    (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                    (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_ARRAY_F32_F32_LEVEL
  : TEX_1D_ARRAY_LEVEL<"tex.level.a1d.v4.f32.f32", Float32Regs, Float32Regs,
                       int_nvvm_tex_1d_array_level_v4f32_f32>;
defm TEX_1D_ARRAY_S32_F32_LEVEL
  : TEX_1D_ARRAY_LEVEL<"tex.level.a1d.v4.s32.f32", Int32Regs, Float32Regs,
                       int_nvvm_tex_1d_array_level_v4s32_f32>;
defm TEX_1D_ARRAY_U32_F32_LEVEL
  : TEX_1D_ARRAY_LEVEL<"tex.level.a1d.v4.u32.f32", Int32Regs, Float32Regs,
                       int_nvvm_tex_1d_array_level_v4u32_f32>;

class TEX_1D_ARRAY_GRAD_base<string inst, NVPTXRegClass outtype,
                             NVPTXRegClass intype, dag texsamp,
                             list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins Int32Regs:$l, intype:$x,
                                    intype:$gradx, intype:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$l, $x\\}],"
                        " \\{$gradx\\}, \\{$grady\\};",
                 pattern>;

multiclass TEX_1D_ARRAY_GRAD<string inst, NVPTXRegClass outtype,
                             NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_1D_ARRAY_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, i32:$l, intype:$x,
                  intype:$gradx, intype:$grady))]>;
  def _RI : TEX_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                   (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                   (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                   (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_ARRAY_F32_F32_GRAD
  : TEX_1D_ARRAY_GRAD<"tex.grad.a1d.v4.f32.f32", Float32Regs, Float32Regs,
                      int_nvvm_tex_1d_array_grad_v4f32_f32>;
defm TEX_1D_ARRAY_S32_F32_GRAD
  : TEX_1D_ARRAY_GRAD<"tex.grad.a1d.v4.s32.f32", Int32Regs, Float32Regs,
                      int_nvvm_tex_1d_array_grad_v4s32_f32>;
defm TEX_1D_ARRAY_U32_F32_GRAD
  : TEX_1D_ARRAY_GRAD<"tex.grad.a1d.v4.u32.f32", Int32Regs, Float32Regs,
                      int_nvvm_tex_1d_array_grad_v4u32_f32>;

class TEX_2D_base<string inst, NVPTXRegClass outtype,
                  NVPTXRegClass intype, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x, $y\\}];",
                 pattern>;

multiclass TEX_2D<string inst, NVPTXRegClass outtype, NVPTXRegClass intype,
                  Intrinsic intr> {
  def _RR : TEX_2D_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$y))]>;
  def _RI : TEX_2D_base<inst, outtype, intype, (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_2D_base<inst, outtype, intype, (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_2D_base<inst, outtype, intype, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_F32_F32 : TEX_2D<"tex.2d.v4.f32.f32", Float32Regs, Float32Regs,
                             int_nvvm_tex_2d_v4f32_f32>;
defm TEX_2D_F32_S32 : TEX_2D<"tex.2d.v4.f32.s32", Float32Regs, Int32Regs,
                             int_nvvm_tex_2d_v4f32_s32>;
defm TEX_2D_S32_S32 : TEX_2D<"tex.2d.v4.s32.s32", Int32Regs, Int32Regs,
                             int_nvvm_tex_2d_v4s32_s32>;
defm TEX_2D_S32_F32 : TEX_2D<"tex.2d.v4.s32.f32", Int32Regs, Float32Regs,
                             int_nvvm_tex_2d_v4s32_f32>;
defm TEX_2D_U32_S32 : TEX_2D<"tex.2d.v4.u32.s32", Int32Regs, Int32Regs,
                             int_nvvm_tex_2d_v4u32_s32>;
defm TEX_2D_U32_F32 : TEX_2D<"tex.2d.v4.u32.f32", Int32Regs, Float32Regs,
                             int_nvvm_tex_2d_v4u32_f32>;

class TEX_2D_LEVEL_base<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, dag texsamp,
                        list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y\\}], $lod;",
                 pattern>;

multiclass TEX_2D_LEVEL<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_2D_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$y, intype:$lod))]>;
  def _RI : TEX_2D_LEVEL_base<inst, outtype, intype,
                              (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_2D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_2D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_F32_F32_LEVEL :
  TEX_2D_LEVEL<"tex.level.2d.v4.f32.f32", Float32Regs, Float32Regs,
               int_nvvm_tex_2d_level_v4f32_f32>;
defm TEX_2D_S32_F32_LEVEL :
  TEX_2D_LEVEL<"tex.level.2d.v4.s32.f32", Int32Regs, Float32Regs,
               int_nvvm_tex_2d_level_v4s32_f32>;
defm TEX_2D_U32_F32_LEVEL :
  TEX_2D_LEVEL<"tex.level.2d.v4.u32.f32", Int32Regs, Float32Regs,
               int_nvvm_tex_2d_level_v4u32_f32>;

class TEX_2D_GRAD_base<string inst, NVPTXRegClass outtype,
                       NVPTXRegClass intype, dag texsamp,
                       list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y,
                                    intype:$gradx0, intype:$gradx1,
                                    intype:$grady0, intype:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\};",
                 pattern>;

multiclass TEX_2D_GRAD<string inst, NVPTXRegClass outtype,
                       NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_2D_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$y,
                  intype:$gradx0, intype:$gradx1,
                  intype:$grady0, intype:$grady1))]>;
  def _RI : TEX_2D_GRAD_base<inst, outtype, intype,
                              (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_2D_GRAD_base<inst, outtype, intype,
                              (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_2D_GRAD_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_F32_F32_GRAD :
  TEX_2D_GRAD<"tex.grad.2d.v4.f32.f32", Float32Regs, Float32Regs,
              int_nvvm_tex_2d_grad_v4f32_f32>;
defm TEX_2D_S32_F32_GRAD :
  TEX_2D_GRAD<"tex.grad.2d.v4.s32.f32", Int32Regs, Float32Regs,
              int_nvvm_tex_2d_grad_v4s32_f32>;
defm TEX_2D_U32_F32_GRAD :
  TEX_2D_GRAD<"tex.grad.2d.v4.u32.f32", Int32Regs, Float32Regs,
              int_nvvm_tex_2d_grad_v4u32_f32>;

class TEX_2D_ARRAY_base<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, dag texsamp,
                        list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins Int32Regs:$l, intype:$x, intype:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $y\\}];",
                 pattern>;

multiclass TEX_2D_ARRAY<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_2D_ARRAY_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, i32:$l, intype:$x, intype:$y))]>;
  def _RI : TEX_2D_ARRAY_base<inst, outtype, intype,
                              (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_2D_ARRAY_base<inst, outtype, intype,
                              (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_2D_ARRAY_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_ARRAY_F32_F32
  : TEX_2D_ARRAY<"tex.a2d.v4.f32.f32", Float32Regs, Float32Regs,
                 int_nvvm_tex_2d_array_v4f32_f32>;
defm TEX_2D_ARRAY_F32_S32
  : TEX_2D_ARRAY<"tex.a2d.v4.f32.s32", Float32Regs, Int32Regs,
                 int_nvvm_tex_2d_array_v4f32_s32>;
defm TEX_2D_ARRAY_S32_S32
  : TEX_2D_ARRAY<"tex.a2d.v4.s32.s32", Int32Regs, Int32Regs,
                 int_nvvm_tex_2d_array_v4s32_s32>;
defm TEX_2D_ARRAY_S32_F32
  : TEX_2D_ARRAY<"tex.a2d.v4.s32.f32", Int32Regs, Float32Regs,
                 int_nvvm_tex_2d_array_v4s32_f32>;
defm TEX_2D_ARRAY_U32_S32
  : TEX_2D_ARRAY<"tex.a2d.v4.u32.s32", Int32Regs, Int32Regs,
                 int_nvvm_tex_2d_array_v4u32_s32>;
defm TEX_2D_ARRAY_U32_F32
  : TEX_2D_ARRAY<"tex.a2d.v4.u32.f32", Int32Regs, Float32Regs,
                 int_nvvm_tex_2d_array_v4u32_f32>;

class TEX_2D_ARRAY_LEVEL_base<string inst, NVPTXRegClass outtype,
                              NVPTXRegClass intype, dag texsamp,
                              list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins Int32Regs:$l, intype:$x, intype:$y,
                                    intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $y\\}], $lod;",
                 pattern>;

multiclass TEX_2D_ARRAY_LEVEL<string inst, NVPTXRegClass outtype,
                              NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_2D_ARRAY_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, i32:$l, intype:$x, intype:$y, intype:$lod))]>;
  def _RI : TEX_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                              (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_ARRAY_F32_F32_LEVEL
  : TEX_2D_ARRAY_LEVEL<"tex.level.a2d.v4.f32.f32", Float32Regs, Float32Regs,
                       int_nvvm_tex_2d_array_level_v4f32_f32>;
defm TEX_2D_ARRAY_S32_F32_LEVEL
  : TEX_2D_ARRAY_LEVEL<"tex.level.a2d.v4.s32.f32", Int32Regs, Float32Regs,
                       int_nvvm_tex_2d_array_level_v4s32_f32>;
defm TEX_2D_ARRAY_U32_F32_LEVEL
  : TEX_2D_ARRAY_LEVEL<"tex.level.a2d.v4.u32.f32", Int32Regs, Float32Regs,
                       int_nvvm_tex_2d_array_level_v4u32_f32>;

class TEX_2D_ARRAY_GRAD_base<string inst, NVPTXRegClass outtype,
                             NVPTXRegClass intype, dag texsamp,
                             list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins Int32Regs:$l, intype:$x, intype:$y,
                                    intype:$gradx0, intype:$gradx1,
                                    intype:$grady0, intype:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\};",
                 pattern>;

multiclass TEX_2D_ARRAY_GRAD<string inst, NVPTXRegClass outtype,
                             NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_2D_ARRAY_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, i32:$l, intype:$x, intype:$y,
                  intype:$gradx0, intype:$gradx1,
                  intype:$grady0, intype:$grady1))]>;
  def _RI : TEX_2D_ARRAY_GRAD_base<inst, outtype, intype,
                              (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_2D_ARRAY_GRAD_base<inst, outtype, intype,
                              (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_2D_ARRAY_GRAD_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_ARRAY_F32_F32_GRAD
  : TEX_2D_ARRAY_GRAD<"tex.grad.a2d.v4.f32.f32", Float32Regs, Float32Regs,
                      int_nvvm_tex_2d_array_grad_v4f32_f32>;
defm TEX_2D_ARRAY_S32_F32_GRAD
  : TEX_2D_ARRAY_GRAD<"tex.grad.a2d.v4.s32.f32", Int32Regs, Float32Regs,
                      int_nvvm_tex_2d_array_grad_v4s32_f32>;
defm TEX_2D_ARRAY_U32_F32_GRAD
  : TEX_2D_ARRAY_GRAD<"tex.grad.a2d.v4.u32.f32", Int32Regs, Float32Regs,
                      int_nvvm_tex_2d_array_grad_v4u32_f32>;

class TEX_3D_base<string inst, NVPTXRegClass outtype,
                  NVPTXRegClass intype, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}];",
                 pattern>;

multiclass TEX_3D<string inst, NVPTXRegClass outtype, NVPTXRegClass intype,
                  Intrinsic intr> {
  def _RR : TEX_3D_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$y, intype:$z))]>;
  def _RI : TEX_3D_base<inst, outtype, intype,
                              (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_3D_base<inst, outtype, intype,
                              (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_3D_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_3D_F32_F32 : TEX_3D<"tex.3d.v4.f32.f32", Float32Regs, Float32Regs,
                             int_nvvm_tex_3d_v4f32_f32>;
defm TEX_3D_F32_S32 : TEX_3D<"tex.3d.v4.f32.s32", Float32Regs, Int32Regs,
                             int_nvvm_tex_3d_v4f32_s32>;
defm TEX_3D_S32_S32 : TEX_3D<"tex.3d.v4.s32.s32", Int32Regs, Int32Regs,
                             int_nvvm_tex_3d_v4s32_s32>;
defm TEX_3D_S32_F32 : TEX_3D<"tex.3d.v4.s32.f32", Int32Regs, Float32Regs,
                             int_nvvm_tex_3d_v4s32_f32>;
defm TEX_3D_U32_S32 : TEX_3D<"tex.3d.v4.u32.s32", Int32Regs, Int32Regs,
                             int_nvvm_tex_3d_v4u32_s32>;
defm TEX_3D_U32_F32 : TEX_3D<"tex.3d.v4.u32.f32", Int32Regs, Float32Regs,
                             int_nvvm_tex_3d_v4u32_f32>;

class TEX_3D_LEVEL_base<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, dag texsamp,
                        list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z,
                                    intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}], $lod;",
                 pattern>;

multiclass TEX_3D_LEVEL<string inst, NVPTXRegClass outtype,
                        NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_3D_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$y, intype:$z,
                  intype:$lod))]>;
  def _RI : TEX_3D_LEVEL_base<inst, outtype, intype,
                              (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_3D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_3D_LEVEL_base<inst, outtype, intype,
                              (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_3D_F32_F32_LEVEL
  : TEX_3D_LEVEL<"tex.level.3d.v4.f32.f32", Float32Regs, Float32Regs,
                 int_nvvm_tex_3d_level_v4f32_f32>;
defm TEX_3D_S32_F32_LEVEL
  : TEX_3D_LEVEL<"tex.level.3d.v4.s32.f32", Int32Regs, Float32Regs,
                 int_nvvm_tex_3d_level_v4s32_f32>;
defm TEX_3D_U32_F32_LEVEL
  : TEX_3D_LEVEL<"tex.level.3d.v4.u32.f32", Int32Regs, Float32Regs,
                 int_nvvm_tex_3d_level_v4u32_f32>;

class TEX_3D_GRAD_base<string inst, NVPTXRegClass outtype,
                       NVPTXRegClass intype, dag texsamp,
                       list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z,
                                    intype :$gradx0, intype:$gradx1,
                                    intype:$gradx2, intype:$grady0,
                                    intype:$grady1, intype:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\};",
                 pattern>;

multiclass TEX_3D_GRAD<string inst, NVPTXRegClass outtype,
                       NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_3D_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$y, intype:$z,
                  intype:$gradx0, intype:$gradx1, intype:$gradx2,
                  intype:$grady0, intype:$grady1, intype:$grady2))]>;
  def _RI : TEX_3D_GRAD_base<inst, outtype, intype,
                             (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_3D_GRAD_base<inst, outtype, intype,
                             (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_3D_GRAD_base<inst, outtype, intype,
                             (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_3D_F32_F32_GRAD
  : TEX_3D_GRAD<"tex.grad.3d.v4.f32.f32", Float32Regs, Float32Regs,
                int_nvvm_tex_3d_grad_v4f32_f32>;
defm TEX_3D_S32_F32_GRAD
  : TEX_3D_GRAD<"tex.grad.3d.v4.s32.f32", Int32Regs, Float32Regs,
                int_nvvm_tex_3d_grad_v4s32_f32>;
defm TEX_3D_U32_F32_GRAD
  : TEX_3D_GRAD<"tex.grad.3d.v4.u32.f32", Int32Regs, Float32Regs,
                int_nvvm_tex_3d_grad_v4u32_f32>;

class TEX_CUBE_base<string inst, NVPTXRegClass outtype,
                    NVPTXRegClass intype, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}];",
                 pattern>;

multiclass TEX_CUBE<string inst, NVPTXRegClass outtype, NVPTXRegClass intype,
                    Intrinsic intr> {
  def _RR : TEX_CUBE_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$y, intype:$z))]>;
  def _RI : TEX_CUBE_base<inst, outtype, intype,
                          (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_base<inst, outtype, intype,
                          (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_CUBE_base<inst, outtype, intype,
                          (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_F32_F32
  : TEX_CUBE<"tex.cube.v4.f32.f32", Float32Regs, Float32Regs,
             int_nvvm_tex_cube_v4f32_f32>;
defm TEX_CUBE_S32_F32
  : TEX_CUBE<"tex.cube.v4.s32.f32", Int32Regs, Float32Regs,
             int_nvvm_tex_cube_v4s32_f32>;
defm TEX_CUBE_U32_F32
  : TEX_CUBE<"tex.cube.v4.u32.f32", Int32Regs, Float32Regs,
             int_nvvm_tex_cube_v4u32_f32>;

class TEX_CUBE_LEVEL_base<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, dag texsamp,
                          list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins intype:$x, intype:$y, intype:$z,
                                    intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}], $lod;",
                 pattern>;

multiclass TEX_CUBE_LEVEL<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_CUBE_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, intype:$x, intype:$y, intype:$z,
                  intype:$lod))]>;
  def _RI : TEX_CUBE_LEVEL_base<inst, outtype, intype,
                                (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_LEVEL_base<inst, outtype, intype,
                                (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_CUBE_LEVEL_base<inst, outtype, intype,
                                (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_F32_F32_LEVEL
  : TEX_CUBE_LEVEL<"tex.level.cube.v4.f32.f32", Float32Regs, Float32Regs,
                   int_nvvm_tex_cube_level_v4f32_f32>;
defm TEX_CUBE_S32_F32_LEVEL
  : TEX_CUBE_LEVEL<"tex.level.cube.v4.s32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_cube_level_v4s32_f32>;
defm TEX_CUBE_U32_F32_LEVEL
  : TEX_CUBE_LEVEL<"tex.level.cube.v4.u32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_cube_level_v4u32_f32>;

class TEX_CUBE_ARRAY_base<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, dag texsamp,
                          list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins Int32Regs:$l, intype:$x, intype:$y,
                                    intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $z\\}];",
                 pattern>;

multiclass TEX_CUBE_ARRAY<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_CUBE_ARRAY_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, i32:$l, intype:$x, intype:$y, intype:$z))]>;
  def _RI : TEX_CUBE_ARRAY_base<inst, outtype, intype,
                                (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_ARRAY_base<inst, outtype, intype,
                                (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_CUBE_ARRAY_base<inst, outtype, intype,
                                (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_ARRAY_F32_F32
  : TEX_CUBE_ARRAY<"tex.acube.v4.f32.f32", Float32Regs, Float32Regs,
                   int_nvvm_tex_cube_array_v4f32_f32>;
defm TEX_CUBE_ARRAY_S32_F32
  : TEX_CUBE_ARRAY<"tex.acube.v4.s32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_cube_array_v4s32_f32>;
defm TEX_CUBE_ARRAY_U32_F32
  : TEX_CUBE_ARRAY<"tex.acube.v4.u32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_cube_array_v4u32_f32>;

class TEX_CUBE_ARRAY_LEVEL_base<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, dag texsamp,
                                list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(texsamp, (ins Int32Regs:$l, intype:$x, intype:$y,
                                    intype:$z, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $z\\}], $lod;",
                 pattern>;

multiclass TEX_CUBE_ARRAY_LEVEL<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, Intrinsic intr> {
  def _RR : TEX_CUBE_ARRAY_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i64:$s, i32:$l, intype:$x, intype:$y, intype:$z,
                  intype:$lod))]>;
  def _RI : TEX_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                      (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                      (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TEX_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                      (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_ARRAY_F32_F32_LEVEL
  : TEX_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.f32.f32",
                         Float32Regs, Float32Regs,
                         int_nvvm_tex_cube_array_level_v4f32_f32>;
defm TEX_CUBE_ARRAY_S32_F32_LEVEL
  : TEX_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.s32.f32",
                         Int32Regs, Float32Regs,
                         int_nvvm_tex_cube_array_level_v4s32_f32>;
defm TEX_CUBE_ARRAY_U32_F32_LEVEL
  : TEX_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.u32.f32",
                         Int32Regs, Float32Regs,
                         int_nvvm_tex_cube_array_level_v4u32_f32>;

class TLD4_2D_base<string inst, NVPTXRegClass outtype,
                   NVPTXRegClass intype, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$v0, outtype:$v1,
                      outtype:$v2, outtype:$v3),
                 !con(texsamp, (ins intype:$x, intype:$y)),
                 inst # " \t\\{$v0, $v1, $v2, $v3\\}, [$t, $s, \\{$x, $y\\}];",
                 pattern>;

multiclass TLD4_2D<string inst, NVPTXRegClass outtype, NVPTXRegClass intype,
                   Intrinsic intr> {
  def _RR : TLD4_2D_base<
      inst, outtype, intype, (ins Int64Regs:$t, Int64Regs:$s),
      [(set outtype:$v0, outtype:$v1, outtype:$v2, outtype:$v3,
            (intr i64:$t, i64:$s, intype:$x, intype:$y))]>;
  def _RI : TLD4_2D_base<inst, outtype, intype,
                         (ins Int64Regs:$t, i64imm:$s)>;
  def _IR : TLD4_2D_base<inst, outtype, intype,
                         (ins i64imm:$t, Int64Regs:$s)>;
  def _II : TLD4_2D_base<inst, outtype, intype,
                         (ins i64imm:$t, i64imm:$s)>;
}

defm TLD4_R_2D_F32_F32
  : TLD4_2D<"tld4.r.2d.v4.f32.f32", Float32Regs, Float32Regs,
            int_nvvm_tld4_r_2d_v4f32_f32>;
defm TLD4_G_2D_F32_F32
  : TLD4_2D<"tld4.g.2d.v4.f32.f32", Float32Regs, Float32Regs,
            int_nvvm_tld4_g_2d_v4f32_f32>;
defm TLD4_B_2D_F32_F32
  : TLD4_2D<"tld4.b.2d.v4.f32.f32", Float32Regs, Float32Regs,
            int_nvvm_tld4_b_2d_v4f32_f32>;
defm TLD4_A_2D_F32_F32
  : TLD4_2D<"tld4.a.2d.v4.f32.f32", Float32Regs, Float32Regs,
            int_nvvm_tld4_a_2d_v4f32_f32>;

defm TLD4_R_2D_S32_F32
  : TLD4_2D<"tld4.r.2d.v4.s32.f32", Int32Regs, Float32Regs,
            int_nvvm_tld4_r_2d_v4s32_f32>;
defm TLD4_G_2D_S32_F32
  : TLD4_2D<"tld4.g.2d.v4.s32.f32", Int32Regs, Float32Regs,
            int_nvvm_tld4_g_2d_v4s32_f32>;
defm TLD4_B_2D_S32_F32
  : TLD4_2D<"tld4.b.2d.v4.s32.f32", Int32Regs, Float32Regs,
            int_nvvm_tld4_b_2d_v4s32_f32>;
defm TLD4_A_2D_S32_F32
  : TLD4_2D<"tld4.a.2d.v4.s32.f32", Int32Regs, Float32Regs,
            int_nvvm_tld4_a_2d_v4s32_f32>;

defm TLD4_R_2D_U32_F32
  : TLD4_2D<"tld4.r.2d.v4.u32.f32", Int32Regs, Float32Regs,
            int_nvvm_tld4_r_2d_v4u32_f32>;
defm TLD4_G_2D_U32_F32
  : TLD4_2D<"tld4.g.2d.v4.u32.f32", Int32Regs, Float32Regs,
            int_nvvm_tld4_g_2d_v4u32_f32>;
defm TLD4_B_2D_U32_F32
  : TLD4_2D<"tld4.b.2d.v4.u32.f32", Int32Regs, Float32Regs,
            int_nvvm_tld4_b_2d_v4u32_f32>;
defm TLD4_A_2D_U32_F32
  : TLD4_2D<"tld4.a.2d.v4.u32.f32", Int32Regs, Float32Regs,
            int_nvvm_tld4_a_2d_v4u32_f32>;

}


// texmode_unified
let IsTex = true, IsTexModeUnified = true in {
// Texture fetch instructions using handles

class TEX_UNIFIED_1D_base<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x\\}];",
                 pattern>;

multiclass TEX_UNIFIED_1D<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x))]>;
  def _I : TEX_UNIFIED_1D_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_F32_S32
  : TEX_UNIFIED_1D<"tex.1d.v4.f32.s32", Float32Regs, Int32Regs,
                   int_nvvm_tex_unified_1d_v4f32_s32>;
defm TEX_UNIFIED_1D_F32_F32
  : TEX_UNIFIED_1D<"tex.1d.v4.f32.f32", Float32Regs, Float32Regs,
                   int_nvvm_tex_unified_1d_v4f32_f32>;
defm TEX_UNIFIED_1D_S32_S32
  : TEX_UNIFIED_1D<"tex.1d.v4.s32.s32", Int32Regs, Int32Regs,
                   int_nvvm_tex_unified_1d_v4s32_s32>;
defm TEX_UNIFIED_1D_S32_F32
  : TEX_UNIFIED_1D<"tex.1d.v4.s32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_unified_1d_v4s32_f32>;
defm TEX_UNIFIED_1D_U32_S32
  : TEX_UNIFIED_1D<"tex.1d.v4.u32.s32", Int32Regs, Int32Regs,
                   int_nvvm_tex_unified_1d_v4u32_s32>;
defm TEX_UNIFIED_1D_U32_F32
  : TEX_UNIFIED_1D<"tex.1d.v4.u32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_unified_1d_v4u32_f32>;

class TEX_UNIFIED_1D_LEVEL_base<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, dag tex,
                                list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x\\}], $lod;",
                 pattern>;

multiclass TEX_UNIFIED_1D_LEVEL<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$lod))]>;
  def _I : TEX_UNIFIED_1D_LEVEL_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_F32_F32_LEVEL
  : TEX_UNIFIED_1D_LEVEL<"tex.level.1d.v4.f32.f32", Float32Regs, Float32Regs,
                         int_nvvm_tex_unified_1d_level_v4f32_f32>;
defm TEX_UNIFIED_1D_S32_F32_LEVEL
  : TEX_UNIFIED_1D_LEVEL<"tex.level.1d.v4.s32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_1d_level_v4s32_f32>;
defm TEX_UNIFIED_1D_U32_F32_LEVEL
  : TEX_UNIFIED_1D_LEVEL<"tex.level.1d.v4.u32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_1d_level_v4u32_f32>;

class TEX_UNIFIED_1D_GRAD_base<string inst, NVPTXRegClass outtype,
                               NVPTXRegClass intype, dag tex,
                               list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$gradx, intype:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$x\\}], \\{$gradx\\}, \\{$grady\\};",
                 pattern>;

multiclass TEX_UNIFIED_1D_GRAD<string inst, NVPTXRegClass outtype,
                               NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$gradx, intype:$grady))]>;
  def _I : TEX_UNIFIED_1D_GRAD_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_F32_F32_GRAD
  : TEX_UNIFIED_1D_GRAD<"tex.grad.1d.v4.f32.f32", Float32Regs, Float32Regs,
                        int_nvvm_tex_unified_1d_grad_v4f32_f32>;
defm TEX_UNIFIED_1D_S32_F32_GRAD
  : TEX_UNIFIED_1D_GRAD<"tex.grad.1d.v4.s32.f32", Int32Regs, Float32Regs,
                        int_nvvm_tex_unified_1d_grad_v4s32_f32>;
defm TEX_UNIFIED_1D_U32_F32_GRAD
  : TEX_UNIFIED_1D_GRAD<"tex.grad.1d.v4.u32.f32", Int32Regs, Float32Regs,
                        int_nvvm_tex_unified_1d_grad_v4u32_f32>;

class TEX_UNIFIED_1D_ARRAY_base<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, dag tex,
                                list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins Int32Regs:$l, intype:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x\\}];",
                 pattern>;

multiclass TEX_UNIFIED_1D_ARRAY<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_ARRAY_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i32:$l, intype:$x))]>;
  def _I : TEX_UNIFIED_1D_ARRAY_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_ARRAY_F32_S32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.f32.s32", Float32Regs, Int32Regs,
                         int_nvvm_tex_unified_1d_array_v4f32_s32>;
defm TEX_UNIFIED_1D_ARRAY_F32_F32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.f32.f32", Float32Regs, Float32Regs,
                         int_nvvm_tex_unified_1d_array_v4f32_f32>;
defm TEX_UNIFIED_1D_ARRAY_S32_S32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.s32.s32", Int32Regs, Int32Regs,
                         int_nvvm_tex_unified_1d_array_v4s32_s32>;
defm TEX_UNIFIED_1D_ARRAY_S32_F32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.s32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_1d_array_v4s32_f32>;
defm TEX_UNIFIED_1D_ARRAY_U32_S32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.u32.s32", Int32Regs, Int32Regs,
                         int_nvvm_tex_unified_1d_array_v4u32_s32>;
defm TEX_UNIFIED_1D_ARRAY_U32_F32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.u32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_1d_array_v4u32_f32>;

class TEX_UNIFIED_1D_ARRAY_LEVEL_base<string inst, NVPTXRegClass outtype,
                                      NVPTXRegClass intype, dag tex,
                                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins Int32Regs:$l, intype:$x, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x\\}], $lod;",
                 pattern>;

multiclass TEX_UNIFIED_1D_ARRAY_LEVEL<string inst, NVPTXRegClass outtype,
                                      NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_ARRAY_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i32:$l, intype:$x, intype:$lod))]>;
  def _I : TEX_UNIFIED_1D_ARRAY_LEVEL_base<inst, outtype, intype,
                                           (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_ARRAY_F32_F32_LEVEL
  : TEX_UNIFIED_1D_ARRAY_LEVEL<"tex.level.a1d.v4.f32.f32",
                               Float32Regs, Float32Regs,
                               int_nvvm_tex_unified_1d_array_level_v4f32_f32>;
defm TEX_UNIFIED_1D_ARRAY_S32_F32_LEVEL
  : TEX_UNIFIED_1D_ARRAY_LEVEL<"tex.level.a1d.v4.s32.f32",
                               Int32Regs, Float32Regs,
                               int_nvvm_tex_unified_1d_array_level_v4s32_f32>;
defm TEX_UNIFIED_1D_ARRAY_U32_F32_LEVEL
  : TEX_UNIFIED_1D_ARRAY_LEVEL<"tex.level.a1d.v4.u32.f32",
                               Int32Regs, Float32Regs,
                               int_nvvm_tex_unified_1d_array_level_v4u32_f32>;

class TEX_UNIFIED_1D_ARRAY_GRAD_base<string inst, NVPTXRegClass outtype,
                                     NVPTXRegClass intype, dag tex,
                                     list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins Int32Regs:$l, intype:$x,
                                intype:$gradx, intype:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        "  [$t, \\{$l, $x\\}], \\{$gradx\\}, \\{$grady\\};",
                 pattern>;

multiclass TEX_UNIFIED_1D_ARRAY_GRAD<string inst, NVPTXRegClass outtype,
                                     NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_ARRAY_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i32:$l, intype:$x, intype:$gradx, intype:$grady))]>;
  def _I : TEX_UNIFIED_1D_ARRAY_GRAD_base<inst, outtype, intype,
                                          (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_ARRAY_F32_F32_GRAD
  : TEX_UNIFIED_1D_ARRAY_GRAD<"tex.grad.a1d.v4.f32.f32",
                              Float32Regs, Float32Regs,
                              int_nvvm_tex_unified_1d_array_grad_v4f32_f32>;
defm TEX_UNIFIED_1D_ARRAY_S32_F32_GRAD
  : TEX_UNIFIED_1D_ARRAY_GRAD<"tex.grad.a1d.v4.s32.f32",
                              Int32Regs, Float32Regs,
                              int_nvvm_tex_unified_1d_array_grad_v4s32_f32>;
defm TEX_UNIFIED_1D_ARRAY_U32_F32_GRAD
  : TEX_UNIFIED_1D_ARRAY_GRAD<"tex.grad.a1d.v4.u32.f32",
                              Int32Regs, Float32Regs,
                              int_nvvm_tex_unified_1d_array_grad_v4u32_f32>;

class TEX_UNIFIED_2D_base<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y\\}];",
                 pattern>;

multiclass TEX_UNIFIED_2D<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$y))]>;
  def _I : TEX_UNIFIED_2D_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_F32_S32
  : TEX_UNIFIED_2D<"tex.2d.v4.f32.s32", Float32Regs, Int32Regs,
                   int_nvvm_tex_unified_2d_v4f32_s32>;
defm TEX_UNIFIED_2D_F32_F32
  : TEX_UNIFIED_2D<"tex.2d.v4.f32.f32", Float32Regs, Float32Regs,
                   int_nvvm_tex_unified_2d_v4f32_f32>;
defm TEX_UNIFIED_2D_S32_S32
  : TEX_UNIFIED_2D<"tex.2d.v4.s32.s32", Int32Regs, Int32Regs,
                   int_nvvm_tex_unified_2d_v4s32_s32>;
defm TEX_UNIFIED_2D_S32_F32
  : TEX_UNIFIED_2D<"tex.2d.v4.s32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_unified_2d_v4s32_f32>;
defm TEX_UNIFIED_2D_U32_S32
  : TEX_UNIFIED_2D<"tex.2d.v4.u32.s32", Int32Regs, Int32Regs,
                   int_nvvm_tex_unified_2d_v4u32_s32>;
defm TEX_UNIFIED_2D_U32_F32
  : TEX_UNIFIED_2D<"tex.2d.v4.u32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_unified_2d_v4u32_f32>;

class TEX_UNIFIED_2D_LEVEL_base<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, dag tex,
                                list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y\\}], $lod;",
                 pattern>;

multiclass TEX_UNIFIED_2D_LEVEL<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$y, intype:$lod))]>;
  def _I : TEX_UNIFIED_2D_LEVEL_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_F32_F32_LEVEL
  : TEX_UNIFIED_2D_LEVEL<"tex.level.2d.v4.f32.f32", Float32Regs, Float32Regs,
                         int_nvvm_tex_unified_2d_level_v4f32_f32>;
defm TEX_UNIFIED_2D_S32_F32_LEVEL
  : TEX_UNIFIED_2D_LEVEL<"tex.level.2d.v4.s32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_2d_level_v4s32_f32>;
defm TEX_UNIFIED_2D_U32_F32_LEVEL
  : TEX_UNIFIED_2D_LEVEL<"tex.level.2d.v4.u32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_2d_level_v4u32_f32>;

class TEX_UNIFIED_2D_GRAD_base<string inst, NVPTXRegClass outtype,
                               NVPTXRegClass intype, dag tex,
                               list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y,
                                intype:$gradx0, intype:$gradx1,
                                intype:$grady0, intype:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\};",
                 pattern>;
multiclass TEX_UNIFIED_2D_GRAD<string inst, NVPTXRegClass outtype,
                               NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$y,
                  intype:$gradx0, intype:$gradx1,
                  intype:$grady0, intype:$grady1))]>;
  def _I : TEX_UNIFIED_2D_GRAD_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_F32_F32_GRAD
  : TEX_UNIFIED_2D_GRAD<"tex.grad.2d.v4.f32.f32", Float32Regs, Float32Regs,
                        int_nvvm_tex_unified_2d_grad_v4f32_f32>;
defm TEX_UNIFIED_2D_S32_F32_GRAD
  : TEX_UNIFIED_2D_GRAD<"tex.grad.2d.v4.s32.f32", Int32Regs, Float32Regs,
                        int_nvvm_tex_unified_2d_grad_v4s32_f32>;
defm TEX_UNIFIED_2D_U32_F32_GRAD
  : TEX_UNIFIED_2D_GRAD<"tex.grad.2d.v4.u32.f32", Int32Regs, Float32Regs,
                        int_nvvm_tex_unified_2d_grad_v4u32_f32>;

class TEX_UNIFIED_2D_ARRAY_base<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, dag tex,
                                list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins Int32Regs:$l, intype:$x, intype:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $y\\}];",
                 pattern>;
multiclass TEX_UNIFIED_2D_ARRAY<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_ARRAY_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i32:$l, intype:$x, intype:$y))]>;
  def _I : TEX_UNIFIED_2D_ARRAY_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_ARRAY_F32_S32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.f32.s32", Float32Regs, Int32Regs,
                         int_nvvm_tex_unified_2d_array_v4f32_s32>;
defm TEX_UNIFIED_2D_ARRAY_F32_F32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.f32.f32", Float32Regs, Float32Regs,
                         int_nvvm_tex_unified_2d_array_v4f32_f32>;
defm TEX_UNIFIED_2D_ARRAY_S32_S32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.s32.s32", Int32Regs, Int32Regs,
                         int_nvvm_tex_unified_2d_array_v4s32_s32>;
defm TEX_UNIFIED_2D_ARRAY_S32_F32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.s32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_2d_array_v4s32_f32>;
defm TEX_UNIFIED_2D_ARRAY_U32_S32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.u32.s32", Int32Regs, Int32Regs,
                         int_nvvm_tex_unified_2d_array_v4u32_s32>;
defm TEX_UNIFIED_2D_ARRAY_U32_F32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.u32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_2d_array_v4u32_f32>;

class TEX_UNIFIED_2D_ARRAY_LEVEL_base<string inst, NVPTXRegClass outtype,
                                      NVPTXRegClass intype, dag tex,
                                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins Int32Regs:$l, intype:$x, intype:$y,
                                intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        "  [$t, \\{$l, $x, $y, $y\\}], $lod;",
                 pattern>;
multiclass TEX_UNIFIED_2D_ARRAY_LEVEL<string inst, NVPTXRegClass outtype,
                                      NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_ARRAY_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i32:$l, intype:$x, intype:$y, intype:$lod))]>;
  def _I : TEX_UNIFIED_2D_ARRAY_LEVEL_base<inst, outtype, intype,
                                           (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_ARRAY_F32_F32_LEVEL
  : TEX_UNIFIED_2D_ARRAY_LEVEL<"tex.level.a2d.v4.f32.f32",
                               Float32Regs, Float32Regs,
                               int_nvvm_tex_unified_2d_array_level_v4f32_f32>;
defm TEX_UNIFIED_2D_ARRAY_S32_F32_LEVEL
  : TEX_UNIFIED_2D_ARRAY_LEVEL<"tex.level.a2d.v4.s32.f32",
                               Int32Regs, Float32Regs,
                               int_nvvm_tex_unified_2d_array_level_v4s32_f32>;
defm TEX_UNIFIED_2D_ARRAY_U32_F32_LEVEL
  : TEX_UNIFIED_2D_ARRAY_LEVEL<"tex.level.a2d.v4.u32.f32",
                               Int32Regs, Float32Regs,
                               int_nvvm_tex_unified_2d_array_level_v4u32_f32>;

class TEX_UNIFIED_2D_ARRAY_GRAD_base<string inst, NVPTXRegClass outtype,
                                     NVPTXRegClass intype, dag tex,
                                     list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins Int32Regs:$l, intype:$x, intype:$y,
                                intype:$gradx0, intype:$gradx1,
                                intype:$grady0, intype:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\};",
                 pattern>;
multiclass TEX_UNIFIED_2D_ARRAY_GRAD<string inst, NVPTXRegClass outtype,
                                     NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_ARRAY_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i32:$l, intype:$x, intype:$y,
                  intype:$gradx0, intype:$gradx1,
                  intype:$grady0, intype:$grady1))]>;
  def _I : TEX_UNIFIED_2D_ARRAY_GRAD_base<inst, outtype, intype,
                                          (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_ARRAY_F32_F32_GRAD
  : TEX_UNIFIED_2D_ARRAY_GRAD<"tex.grad.a2d.v4.f32.f32",
                              Float32Regs, Float32Regs,
                              int_nvvm_tex_unified_2d_array_grad_v4f32_f32>;
defm TEX_UNIFIED_2D_ARRAY_S32_F32_GRAD
  : TEX_UNIFIED_2D_ARRAY_GRAD<"tex.grad.a2d.v4.s32.f32",
                              Int32Regs, Float32Regs,
                              int_nvvm_tex_unified_2d_array_grad_v4s32_f32>;
defm TEX_UNIFIED_2D_ARRAY_U32_F32_GRAD
  : TEX_UNIFIED_2D_ARRAY_GRAD<"tex.grad.a2d.v4.u32.f32",
                              Int32Regs, Float32Regs,
                              int_nvvm_tex_unified_2d_array_grad_v4u32_f32>;

class TEX_UNIFIED_3D_base<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}];",
                 pattern>;
multiclass TEX_UNIFIED_3D<string inst, NVPTXRegClass outtype,
                          NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_3D_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$y, intype:$z))]>;
  def _I : TEX_UNIFIED_3D_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_3D_F32_S32
  : TEX_UNIFIED_3D<"tex.3d.v4.f32.s32", Float32Regs, Int32Regs,
                   int_nvvm_tex_unified_3d_v4f32_s32>;
defm TEX_UNIFIED_3D_F32_F32
  : TEX_UNIFIED_3D<"tex.3d.v4.f32.f32", Float32Regs, Float32Regs,
                   int_nvvm_tex_unified_3d_v4f32_f32>;
defm TEX_UNIFIED_3D_S32_S32
  : TEX_UNIFIED_3D<"tex.3d.v4.s32.s32", Int32Regs, Int32Regs,
                   int_nvvm_tex_unified_3d_v4s32_s32>;
defm TEX_UNIFIED_3D_S32_F32
  : TEX_UNIFIED_3D<"tex.3d.v4.s32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_unified_3d_v4s32_f32>;
defm TEX_UNIFIED_3D_U32_S32
  : TEX_UNIFIED_3D<"tex.3d.v4.u32.s32", Int32Regs, Int32Regs,
                   int_nvvm_tex_unified_3d_v4u32_s32>;
defm TEX_UNIFIED_3D_U32_F32
  : TEX_UNIFIED_3D<"tex.3d.v4.u32.f32", Int32Regs, Float32Regs,
                   int_nvvm_tex_unified_3d_v4u32_f32>;

class TEX_UNIFIED_3D_LEVEL_base<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, dag tex,
                                list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$x, $y, $z, $z\\}], $lod;",
                 pattern>;
multiclass TEX_UNIFIED_3D_LEVEL<string inst, NVPTXRegClass outtype,
                                NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_3D_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$y, intype:$z, intype:$lod))]>;
  def _I : TEX_UNIFIED_3D_LEVEL_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_3D_F32_F32_LEVEL
  : TEX_UNIFIED_3D_LEVEL<"tex.level.3d.v4.f32.f32", Float32Regs, Float32Regs,
                         int_nvvm_tex_unified_3d_level_v4f32_f32>;
defm TEX_UNIFIED_3D_S32_F32_LEVEL
  : TEX_UNIFIED_3D_LEVEL<"tex.level.3d.v4.s32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_3d_level_v4s32_f32>;
defm TEX_UNIFIED_3D_U32_F32_LEVEL
  : TEX_UNIFIED_3D_LEVEL<"tex.level.3d.v4.u32.f32", Int32Regs, Float32Regs,
                         int_nvvm_tex_unified_3d_level_v4u32_f32>;

class TEX_UNIFIED_3D_GRAD_base<string inst, NVPTXRegClass outtype,
                               NVPTXRegClass intype, dag tex,
                               list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z,
                                intype:$gradx0, intype:$gradx1,
                                intype:$gradx2, intype:$grady0,
                                intype:$grady1, intype:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\};",
                 pattern>;
multiclass TEX_UNIFIED_3D_GRAD<string inst, NVPTXRegClass outtype,
                               NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_3D_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$y, intype:$z,
                  intype:$gradx0, intype:$gradx1, intype:$gradx2,
                  intype:$grady0, intype:$grady1, intype:$grady2))]>;
  def _I : TEX_UNIFIED_3D_GRAD_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_3D_F32_F32_GRAD
  : TEX_UNIFIED_3D_GRAD<"tex.grad.3d.v4.f32.f32", Float32Regs, Float32Regs,
                        int_nvvm_tex_unified_3d_grad_v4f32_f32>;
defm TEX_UNIFIED_3D_S32_F32_GRAD
  : TEX_UNIFIED_3D_GRAD<"tex.grad.3d.v4.s32.f32", Int32Regs, Float32Regs,
                        int_nvvm_tex_unified_3d_grad_v4s32_f32>;
defm TEX_UNIFIED_3D_U32_F32_GRAD
  : TEX_UNIFIED_3D_GRAD<"tex.grad.3d.v4.u32.f32", Int32Regs, Float32Regs,
                        int_nvvm_tex_unified_3d_grad_v4u32_f32>;

class TEX_UNIFIED_CUBE_base<string inst, NVPTXRegClass outtype,
                            NVPTXRegClass intype, dag tex,
                            list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}];",
                 pattern>;
multiclass TEX_UNIFIED_CUBE<string inst, NVPTXRegClass outtype,
                            NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$y, intype:$z))]>;
  def _I : TEX_UNIFIED_CUBE_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_F32_F32
  : TEX_UNIFIED_CUBE<"tex.cube.v4.f32.f32", Float32Regs, Float32Regs,
                     int_nvvm_tex_unified_cube_v4f32_f32>;
defm TEX_UNIFIED_CUBE_S32_F32
  : TEX_UNIFIED_CUBE<"tex.cube.v4.s32.f32", Int32Regs, Float32Regs,
                     int_nvvm_tex_unified_cube_v4s32_f32>;
defm TEX_UNIFIED_CUBE_U32_F32
  : TEX_UNIFIED_CUBE<"tex.cube.v4.u32.f32", Int32Regs, Float32Regs,
                     int_nvvm_tex_unified_cube_v4u32_f32>;

class TEX_UNIFIED_CUBE_LEVEL_base<string inst, NVPTXRegClass outtype,
                                  NVPTXRegClass intype, dag tex,
                                  list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z, intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$x, $y, $z, $z\\}], $lod;",
                 pattern>;
multiclass TEX_UNIFIED_CUBE_LEVEL<string inst, NVPTXRegClass outtype,
                                  NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$y, intype:$z, intype:$lod))]>;
  def _I : TEX_UNIFIED_CUBE_LEVEL_base<inst, outtype, intype,
                                       (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_F32_F32_LEVEL
  : TEX_UNIFIED_CUBE_LEVEL<"tex.level.cube.v4.f32.f32",
                           Float32Regs, Float32Regs,
                           int_nvvm_tex_unified_cube_level_v4f32_f32>;
defm TEX_UNIFIED_CUBE_S32_F32_LEVEL
  : TEX_UNIFIED_CUBE_LEVEL<"tex.level.cube.v4.s32.f32",
                           Int32Regs, Float32Regs,
                           int_nvvm_tex_unified_cube_level_v4s32_f32>;
defm TEX_UNIFIED_CUBE_U32_F32_LEVEL
  : TEX_UNIFIED_CUBE_LEVEL<"tex.level.cube.v4.u32.f32",
                           Int32Regs, Float32Regs,
                           int_nvvm_tex_unified_cube_level_v4u32_f32>;

class TEX_UNIFIED_CUBE_ARRAY_base<string inst, NVPTXRegClass outtype,
                                  NVPTXRegClass intype, dag tex,
                                  list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins Int32Regs:$l, intype:$x, intype:$y, intype:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $z\\}];",
                 pattern>;
multiclass TEX_UNIFIED_CUBE_ARRAY<string inst, NVPTXRegClass outtype,
                                  NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_ARRAY_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i32:$l, intype:$x, intype:$y, intype:$z))]>;
  def _I : TEX_UNIFIED_CUBE_ARRAY_base<inst, outtype, intype,
                                       (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_ARRAY_F32_F32
  : TEX_UNIFIED_CUBE_ARRAY<"tex.acube.v4.f32.f32", Float32Regs, Float32Regs,
                           int_nvvm_tex_unified_cube_array_v4f32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_S32_F32
  : TEX_UNIFIED_CUBE_ARRAY<"tex.acube.v4.s32.f32", Int32Regs, Float32Regs,
                           int_nvvm_tex_unified_cube_array_v4s32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_U32_F32
  : TEX_UNIFIED_CUBE_ARRAY<"tex.acube.v4.u32.f32", Int32Regs, Float32Regs,
                           int_nvvm_tex_unified_cube_array_v4u32_f32>;

class TEX_UNIFIED_CUBE_ARRAY_LEVEL_base<string inst, NVPTXRegClass outtype,
                                        NVPTXRegClass intype, dag tex,
                                        list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins Int32Regs:$l, intype:$x, intype:$y, intype:$z,
                                intype:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$l, $x, $y, $z\\}], $lod;",
                 pattern>;
multiclass TEX_UNIFIED_CUBE_ARRAY_LEVEL<string inst, NVPTXRegClass outtype,
                                        NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_ARRAY_LEVEL_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i32:$l, intype:$x, intype:$y, intype:$z, intype:$lod))]>;
  def _I : TEX_UNIFIED_CUBE_ARRAY_LEVEL_base<inst, outtype, intype,
                                             (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_ARRAY_F32_F32_LEVEL
  : TEX_UNIFIED_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.f32.f32",
                                 Float32Regs, Float32Regs,
                                 int_nvvm_tex_unified_cube_array_level_v4f32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_S32_F32_LEVEL
  : TEX_UNIFIED_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.s32.f32",
                                 Int32Regs, Float32Regs,
                                 int_nvvm_tex_unified_cube_array_level_v4s32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_U32_F32_LEVEL
  : TEX_UNIFIED_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.u32.f32",
                                 Int32Regs, Float32Regs,
                                 int_nvvm_tex_unified_cube_array_level_v4u32_f32>;

class TEX_UNIFIED_CUBE_GRAD_base<string inst, NVPTXRegClass outtype,
                                 NVPTXRegClass intype, dag tex,
                                 list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins intype:$x, intype:$y, intype:$z,
                                intype:$gradx0, intype:$gradx1,
                                intype:$gradx2, intype:$grady0,
                                intype:$grady1, intype:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\};",
                 pattern>;

multiclass TEX_UNIFIED_CUBE_GRAD<string inst, NVPTXRegClass outtype,
                                 NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, intype:$x, intype:$y, intype:$z,
                  intype:$gradx0, intype:$gradx1, intype:$gradx2,
                  intype:$grady0, intype:$grady1, intype:$grady2))]>;
  def _I : TEX_UNIFIED_CUBE_GRAD_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_F32_F32_GRAD
  : TEX_UNIFIED_CUBE_GRAD<"tex.grad.cube.v4.f32.f32", Float32Regs, Float32Regs,
                          int_nvvm_tex_unified_cube_grad_v4f32_f32>;
defm TEX_UNIFIED_CUBE_S32_F32_GRAD
  : TEX_UNIFIED_CUBE_GRAD<"tex.grad.cube.v4.s32.f32", Int32Regs, Float32Regs,
                          int_nvvm_tex_unified_cube_grad_v4s32_f32>;
defm TEX_UNIFIED_CUBE_U32_F32_GRAD
  : TEX_UNIFIED_CUBE_GRAD<"tex.grad.cube.v4.u32.f32", Int32Regs, Float32Regs,
                          int_nvvm_tex_unified_cube_grad_v4u32_f32>;

class TEX_UNIFIED_CUBE_ARRAY_GRAD_base<string inst, NVPTXRegClass outtype,
                                       NVPTXRegClass intype, dag tex,
                                       list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g,
                      outtype:$b, outtype:$a),
                 !con(tex, (ins Int32Regs:$l, intype:$x, intype:$y, intype:$z,
                                intype:$gradx0, intype:$gradx1,
                                intype:$gradx2, intype:$grady0,
                                intype:$grady1, intype:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\};",
                 pattern>;
multiclass TEX_UNIFIED_CUBE_ARRAY_GRAD<string inst, NVPTXRegClass outtype,
                                       NVPTXRegClass intype, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_ARRAY_GRAD_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
            (intr i64:$t, i32:$l, intype:$x, intype:$y, intype:$z,
                  intype:$gradx0, intype:$gradx1,
                  intype:$gradx2, intype:$grady0,
                  intype:$grady1, intype:$grady2))]>;
  def _I : TEX_UNIFIED_CUBE_ARRAY_GRAD_base<inst, outtype, intype,
                                            (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_ARRAY_F32_F32_GRAD
  : TEX_UNIFIED_CUBE_ARRAY_GRAD<"tex.grad.acube.v4.f32.f32",
                                Float32Regs, Float32Regs,
                                int_nvvm_tex_unified_cube_array_grad_v4f32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_S32_F32_GRAD
  : TEX_UNIFIED_CUBE_ARRAY_GRAD<"tex.grad.acube.v4.s32.f32",
                                Int32Regs, Float32Regs,
                                int_nvvm_tex_unified_cube_array_grad_v4s32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_U32_F32_GRAD
  : TEX_UNIFIED_CUBE_ARRAY_GRAD<"tex.grad.acube.v4.u32.f32",
                                Int32Regs, Float32Regs,
                                int_nvvm_tex_unified_cube_array_grad_v4u32_f32>;

class TLD4_UNIFIED_2D_base<string inst, NVPTXRegClass outtype,
                           NVPTXRegClass intype, dag tex,
                           list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$v0, outtype:$v1,
                      outtype:$v2, outtype:$v3),
                 !con(tex, (ins intype:$x, intype:$y)),
                 inst # " \t\\{$v0, $v1, $v2, $v3\\}, [$t, \\{$x, $y\\}];",
                 pattern>;
multiclass TLD4_UNIFIED_2D<string inst, NVPTXRegClass outtype,
                           NVPTXRegClass intype, Intrinsic intr> {
  def _R : TLD4_UNIFIED_2D_base<
      inst, outtype, intype, (ins Int64Regs:$t),
      [(set outtype:$v0, outtype:$v1, outtype:$v2, outtype:$v3,
            (intr i64:$t, intype:$x, intype:$y))]>;
  def _I : TLD4_UNIFIED_2D_base<inst, outtype, intype, (ins i64imm:$t)>;
}

defm TLD4_UNIFIED_R_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.r.2d.v4.f32.f32", Float32Regs, Float32Regs,
                    int_nvvm_tld4_unified_r_2d_v4f32_f32>;
defm TLD4_UNIFIED_G_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.g.2d.v4.f32.f32", Float32Regs, Float32Regs,
                    int_nvvm_tld4_unified_g_2d_v4f32_f32>;
defm TLD4_UNIFIED_B_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.b.2d.v4.f32.f32", Float32Regs, Float32Regs,
                    int_nvvm_tld4_unified_b_2d_v4f32_f32>;
defm TLD4_UNIFIED_A_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.a.2d.v4.f32.f32", Float32Regs, Float32Regs,
                    int_nvvm_tld4_unified_a_2d_v4f32_f32>;

defm TLD4_UNIFIED_R_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.r.2d.v4.s32.f32", Int32Regs, Float32Regs,
                    int_nvvm_tld4_unified_r_2d_v4s32_f32>;
defm TLD4_UNIFIED_G_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.g.2d.v4.s32.f32", Int32Regs, Float32Regs,
                    int_nvvm_tld4_unified_g_2d_v4s32_f32>;
defm TLD4_UNIFIED_B_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.b.2d.v4.s32.f32", Int32Regs, Float32Regs,
                    int_nvvm_tld4_unified_b_2d_v4s32_f32>;
defm TLD4_UNIFIED_A_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.a.2d.v4.s32.f32", Int32Regs, Float32Regs,
                    int_nvvm_tld4_unified_a_2d_v4s32_f32>;

defm TLD4_UNIFIED_R_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.r.2d.v4.u32.f32", Int32Regs, Float32Regs,
                    int_nvvm_tld4_unified_r_2d_v4u32_f32>;
defm TLD4_UNIFIED_G_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.g.2d.v4.u32.f32", Int32Regs, Float32Regs,
                    int_nvvm_tld4_unified_g_2d_v4u32_f32>;
defm TLD4_UNIFIED_B_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.b.2d.v4.u32.f32", Int32Regs, Float32Regs,
                    int_nvvm_tld4_unified_b_2d_v4u32_f32>;
defm TLD4_UNIFIED_A_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.a.2d.v4.u32.f32", Int32Regs, Float32Regs,
                    int_nvvm_tld4_unified_a_2d_v4u32_f32>;

}



//=== Surface load instructions

let IsSuld = true in {

class SULD_1D_base<string inst, NVPTXRegClass outtype, dag surf,
                   list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins Int32Regs:$x)),
                inst # " \\{$r\\}, [$s, \\{$x\\}];",
                pattern>;
multiclass SULD_1D<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_base<inst, outtype, (ins Int64Regs:$s),
                        [(set outtype:$r, (intr i64:$s, i32:$x))]>;
  def _I : SULD_1D_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_I8_CLAMP : SULD_1D<"suld.b.1d.b8.clamp", Int16Regs>;
defm SULD_1D_I16_CLAMP : SULD_1D<"suld.b.1d.b16.clamp", Int16Regs>;
defm SULD_1D_I32_CLAMP : SULD_1D<"suld.b.1d.b32.clamp", Int32Regs>;
defm SULD_1D_I64_CLAMP : SULD_1D<"suld.b.1d.b64.clamp", Int64Regs>;

defm SULD_1D_I8_TRAP : SULD_1D<"suld.b.1d.b8.trap", Int16Regs>;
defm SULD_1D_I16_TRAP : SULD_1D<"suld.b.1d.b16.trap", Int16Regs>;
defm SULD_1D_I32_TRAP : SULD_1D<"suld.b.1d.b32.trap", Int32Regs>;
defm SULD_1D_I64_TRAP : SULD_1D<"suld.b.1d.b64.trap", Int64Regs>;

defm SULD_1D_I8_ZERO : SULD_1D<"suld.b.1d.b8.zero", Int16Regs>;
defm SULD_1D_I16_ZERO : SULD_1D<"suld.b.1d.b16.zero", Int16Regs>;
defm SULD_1D_I32_ZERO : SULD_1D<"suld.b.1d.b32.zero", Int32Regs>;
defm SULD_1D_I64_ZERO : SULD_1D<"suld.b.1d.b64.zero", Int64Regs>;

class SULD_1D_ARRAY_base<string inst, NVPTXRegClass outtype, dag surf,
                         list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins Int32Regs:$l, Int32Regs:$x)),
                inst # " \\{$r\\}, [$s, \\{$l, $x\\}];",
                pattern>;
multiclass SULD_1D_ARRAY<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_ARRAY_base<inst, outtype, (ins Int64Regs:$s),
                              [(set outtype:$r,
                                    (intr i64:$s, i32:$l, i32:$x))]>;
  def _I : SULD_1D_ARRAY_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_ARRAY_I8_CLAMP
  : SULD_1D_ARRAY<"suld.b.a1d.b8.clamp", Int16Regs>;
defm SULD_1D_ARRAY_I16_CLAMP
  : SULD_1D_ARRAY<"suld.b.a1d.b16.clamp", Int16Regs>;
defm SULD_1D_ARRAY_I32_CLAMP
  : SULD_1D_ARRAY<"suld.b.a1d.b32.clamp", Int32Regs>;
defm SULD_1D_ARRAY_I64_CLAMP
  : SULD_1D_ARRAY<"suld.b.a1d.b64.clamp", Int64Regs>;

defm SULD_1D_ARRAY_I8_TRAP
  : SULD_1D_ARRAY<"suld.b.a1d.b8.trap", Int16Regs>;
defm SULD_1D_ARRAY_I16_TRAP
  : SULD_1D_ARRAY<"suld.b.a1d.b16.trap", Int16Regs>;
defm SULD_1D_ARRAY_I32_TRAP
  : SULD_1D_ARRAY<"suld.b.a1d.b32.trap", Int32Regs>;
defm SULD_1D_ARRAY_I64_TRAP
  : SULD_1D_ARRAY<"suld.b.a1d.b64.trap", Int64Regs>;

defm SULD_1D_ARRAY_I8_ZERO
  : SULD_1D_ARRAY<"suld.b.a1d.b8.zero", Int16Regs>;
defm SULD_1D_ARRAY_I16_ZERO
  : SULD_1D_ARRAY<"suld.b.a1d.b16.zero", Int16Regs>;
defm SULD_1D_ARRAY_I32_ZERO
  : SULD_1D_ARRAY<"suld.b.a1d.b32.zero", Int32Regs>;
defm SULD_1D_ARRAY_I64_ZERO
  : SULD_1D_ARRAY<"suld.b.a1d.b64.zero", Int64Regs>;

class SULD_2D_base<string inst, NVPTXRegClass outtype, dag surf,
                   list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y)),
                inst # " \\{$r\\}, [$s, \\{$x, $y\\}];",
                pattern>;
multiclass SULD_2D<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_base<inst, outtype, (ins Int64Regs:$s),
                        [(set outtype:$r, (intr i64:$s, i32:$x, i32:$y))]>;
  def _I : SULD_2D_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_I8_CLAMP : SULD_2D<"suld.b.2d.b8.clamp", Int16Regs>;
defm SULD_2D_I16_CLAMP : SULD_2D<"suld.b.2d.b16.clamp", Int16Regs>;
defm SULD_2D_I32_CLAMP : SULD_2D<"suld.b.2d.b32.clamp", Int32Regs>;
defm SULD_2D_I64_CLAMP : SULD_2D<"suld.b.2d.b64.clamp", Int64Regs>;

defm SULD_2D_I8_TRAP : SULD_2D<"suld.b.2d.b8.trap", Int16Regs>;
defm SULD_2D_I16_TRAP : SULD_2D<"suld.b.2d.b16.trap", Int16Regs>;
defm SULD_2D_I32_TRAP : SULD_2D<"suld.b.2d.b32.trap", Int32Regs>;
defm SULD_2D_I64_TRAP : SULD_2D<"suld.b.2d.b64.trap", Int64Regs>;

defm SULD_2D_I8_ZERO : SULD_2D<"suld.b.2d.b8.zero", Int16Regs>;
defm SULD_2D_I16_ZERO : SULD_2D<"suld.b.2d.b16.zero", Int16Regs>;
defm SULD_2D_I32_ZERO : SULD_2D<"suld.b.2d.b32.zero", Int32Regs>;
defm SULD_2D_I64_ZERO : SULD_2D<"suld.b.2d.b64.zero", Int64Regs>;

class SULD_2D_ARRAY_base<string inst, NVPTXRegClass outtype, dag surf,
                         list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins Int32Regs:$l, Int32Regs:$x, Int32Regs:$y)),
                inst # " \\{$r\\}, [$s, \\{$l, $x, $y, $y\\}];",
                pattern>;
multiclass SULD_2D_ARRAY<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_ARRAY_base<inst, outtype, (ins Int64Regs:$s),
                              [(set outtype:$r,
                                    (intr i64:$s, i32:$l, i32:$x, i32:$y))]>;
  def _I : SULD_2D_ARRAY_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_ARRAY_I8_CLAMP : SULD_2D_ARRAY<"suld.b.a2d.b8.clamp", Int16Regs>;
defm SULD_2D_ARRAY_I16_CLAMP : SULD_2D_ARRAY<"suld.b.a2d.b16.clamp", Int16Regs>;
defm SULD_2D_ARRAY_I32_CLAMP : SULD_2D_ARRAY<"suld.b.a2d.b32.clamp", Int32Regs>;
defm SULD_2D_ARRAY_I64_CLAMP : SULD_2D_ARRAY<"suld.b.a2d.b64.clamp", Int64Regs>;

defm SULD_2D_ARRAY_I8_TRAP : SULD_2D_ARRAY<"suld.b.a2d.b8.trap", Int16Regs>;
defm SULD_2D_ARRAY_I16_TRAP : SULD_2D_ARRAY<"suld.b.a2d.b16.trap", Int16Regs>;
defm SULD_2D_ARRAY_I32_TRAP : SULD_2D_ARRAY<"suld.b.a2d.b32.trap", Int32Regs>;
defm SULD_2D_ARRAY_I64_TRAP : SULD_2D_ARRAY<"suld.b.a2d.b64.trap", Int64Regs>;

defm SULD_2D_ARRAY_I8_ZERO : SULD_2D_ARRAY<"suld.b.a2d.b8.zero", Int16Regs>;
defm SULD_2D_ARRAY_I16_ZERO : SULD_2D_ARRAY<"suld.b.a2d.b16.zero", Int16Regs>;
defm SULD_2D_ARRAY_I32_ZERO : SULD_2D_ARRAY<"suld.b.a2d.b32.zero", Int32Regs>;
defm SULD_2D_ARRAY_I64_ZERO : SULD_2D_ARRAY<"suld.b.a2d.b64.zero", Int64Regs>;

class SULD_3D_base<string inst, NVPTXRegClass outtype, dag surf,
                   list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y, Int32Regs:$z)),
                inst # " \\{$r\\}, [$s, \\{$x, $y, $z, $z\\}];",
                pattern>;
multiclass SULD_3D<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_3D_base<inst, outtype, (ins Int64Regs:$s),
                        [(set outtype:$r,
                              (intr i64:$s, i32:$x, i32:$y, i32:$z))]>;
  def _I : SULD_3D_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_3D_I8_CLAMP : SULD_3D<"suld.b.3d.b8.clamp", Int16Regs>;
defm SULD_3D_I16_CLAMP : SULD_3D<"suld.b.3d.b16.clamp", Int16Regs>;
defm SULD_3D_I32_CLAMP : SULD_3D<"suld.b.3d.b32.clamp", Int32Regs>;
defm SULD_3D_I64_CLAMP : SULD_3D<"suld.b.3d.b64.clamp", Int64Regs>;

defm SULD_3D_I8_TRAP : SULD_3D<"suld.b.3d.b8.trap", Int16Regs>;
defm SULD_3D_I16_TRAP : SULD_3D<"suld.b.3d.b16.trap", Int16Regs>;
defm SULD_3D_I32_TRAP : SULD_3D<"suld.b.3d.b32.trap", Int32Regs>;
defm SULD_3D_I64_TRAP : SULD_3D<"suld.b.3d.b64.trap", Int64Regs>;

defm SULD_3D_I8_ZERO : SULD_3D<"suld.b.3d.b8.zero", Int16Regs>;
defm SULD_3D_I16_ZERO : SULD_3D<"suld.b.3d.b16.zero", Int16Regs>;
defm SULD_3D_I32_ZERO : SULD_3D<"suld.b.3d.b32.zero", Int32Regs>;
defm SULD_3D_I64_ZERO : SULD_3D<"suld.b.3d.b64.zero", Int64Regs>;
}

let IsSuld = 2 in {

class SULD_1D_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins Int32Regs:$x)),
                inst # " \\{$r, $g\\}, [$s, \\{$x\\}];",
                pattern>;
multiclass SULD_1D_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_V2_base<inst, outtype, (ins Int64Regs:$s),
                           [(set outtype:$r, outtype:$g,
                                 (intr i64:$s, i32:$x))]>;
  def _I : SULD_1D_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_V2I8_CLAMP : SULD_1D_V2<"suld.b.1d.v2.b8.clamp", Int16Regs>;
defm SULD_1D_V2I16_CLAMP : SULD_1D_V2<"suld.b.1d.v2.b16.clamp", Int16Regs>;
defm SULD_1D_V2I32_CLAMP : SULD_1D_V2<"suld.b.1d.v2.b32.clamp", Int32Regs>;
defm SULD_1D_V2I64_CLAMP : SULD_1D_V2<"suld.b.1d.v2.b64.clamp", Int64Regs>;

defm SULD_1D_V2I8_TRAP : SULD_1D_V2<"suld.b.1d.v2.b8.trap", Int16Regs>;
defm SULD_1D_V2I16_TRAP : SULD_1D_V2<"suld.b.1d.v2.b16.trap", Int16Regs>;
defm SULD_1D_V2I32_TRAP : SULD_1D_V2<"suld.b.1d.v2.b32.trap", Int32Regs>;
defm SULD_1D_V2I64_TRAP : SULD_1D_V2<"suld.b.1d.v2.b64.trap", Int64Regs>;

defm SULD_1D_V2I8_ZERO : SULD_1D_V2<"suld.b.1d.v2.b8.zero", Int16Regs>;
defm SULD_1D_V2I16_ZERO : SULD_1D_V2<"suld.b.1d.v2.b16.zero", Int16Regs>;
defm SULD_1D_V2I32_ZERO : SULD_1D_V2<"suld.b.1d.v2.b32.zero", Int32Regs>;
defm SULD_1D_V2I64_ZERO : SULD_1D_V2<"suld.b.1d.v2.b64.zero", Int64Regs>;

class SULD_1D_ARRAY_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                            list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins Int32Regs:$l, Int32Regs:$x)),
                inst # " \\{$r, $g\\}, [$s, \\{$l, $x\\}];",
                pattern>;
multiclass SULD_1D_ARRAY_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_ARRAY_V2_base<inst, outtype, (ins Int64Regs:$s),
                                 [(set outtype:$r, outtype:$g,
                                       (intr i64:$s, i32:$l, i32:$x))]>;
  def _I : SULD_1D_ARRAY_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_ARRAY_V2I8_CLAMP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b8.clamp", Int16Regs>;
defm SULD_1D_ARRAY_V2I16_CLAMP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b16.clamp", Int16Regs>;
defm SULD_1D_ARRAY_V2I32_CLAMP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b32.clamp", Int32Regs>;
defm SULD_1D_ARRAY_V2I64_CLAMP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b64.clamp", Int64Regs>;

defm SULD_1D_ARRAY_V2I8_TRAP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b8.trap", Int16Regs>;
defm SULD_1D_ARRAY_V2I16_TRAP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b16.trap", Int16Regs>;
defm SULD_1D_ARRAY_V2I32_TRAP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b32.trap", Int32Regs>;
defm SULD_1D_ARRAY_V2I64_TRAP
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b64.trap", Int64Regs>;

defm SULD_1D_ARRAY_V2I8_ZERO
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b8.zero", Int16Regs>;
defm SULD_1D_ARRAY_V2I16_ZERO
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b16.zero", Int16Regs>;
defm SULD_1D_ARRAY_V2I32_ZERO
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b32.zero", Int32Regs>;
defm SULD_1D_ARRAY_V2I64_ZERO
  : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b64.zero", Int64Regs>;

class SULD_2D_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y)),
                inst # " \\{$r, $g\\}, [$s, \\{$x, $y\\}];",
                pattern>;
multiclass SULD_2D_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_V2_base<inst, outtype, (ins Int64Regs:$s),
                           [(set outtype:$r, outtype:$g,
                                 (intr i64:$s, i32:$x, i32:$y))]>;
  def _I : SULD_2D_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_V2I8_CLAMP
  : SULD_2D_V2<"suld.b.2d.v2.b8.clamp", Int16Regs>;
defm SULD_2D_V2I16_CLAMP
  : SULD_2D_V2<"suld.b.2d.v2.b16.clamp", Int16Regs>;
defm SULD_2D_V2I32_CLAMP
  : SULD_2D_V2<"suld.b.2d.v2.b32.clamp", Int32Regs>;
defm SULD_2D_V2I64_CLAMP
  : SULD_2D_V2<"suld.b.2d.v2.b64.clamp", Int64Regs>;

defm SULD_2D_V2I8_TRAP
  : SULD_2D_V2<"suld.b.2d.v2.b8.trap", Int16Regs>;
defm SULD_2D_V2I16_TRAP
  : SULD_2D_V2<"suld.b.2d.v2.b16.trap", Int16Regs>;
defm SULD_2D_V2I32_TRAP
  : SULD_2D_V2<"suld.b.2d.v2.b32.trap", Int32Regs>;
defm SULD_2D_V2I64_TRAP
  : SULD_2D_V2<"suld.b.2d.v2.b64.trap", Int64Regs>;

defm SULD_2D_V2I8_ZERO
  : SULD_2D_V2<"suld.b.2d.v2.b8.zero", Int16Regs>;
defm SULD_2D_V2I16_ZERO
  : SULD_2D_V2<"suld.b.2d.v2.b16.zero", Int16Regs>;
defm SULD_2D_V2I32_ZERO
  : SULD_2D_V2<"suld.b.2d.v2.b32.zero", Int32Regs>;
defm SULD_2D_V2I64_ZERO
  : SULD_2D_V2<"suld.b.2d.v2.b64.zero", Int64Regs>;

class SULD_2D_ARRAY_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                            list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins Int32Regs:$l, Int32Regs:$x, Int32Regs:$y)),
                inst # " \\{$r, $g\\}, [$s, \\{$l, $x, $y, $y\\}];",
                pattern>;
multiclass SULD_2D_ARRAY_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_ARRAY_V2_base<inst, outtype, (ins Int64Regs:$s),
                                 [(set outtype:$r, outtype:$g,
                                       (intr i64:$s, i32:$l, i32:$x, i32:$y))]>;
  def _I : SULD_2D_ARRAY_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_ARRAY_V2I8_CLAMP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b8.clamp", Int16Regs>;
defm SULD_2D_ARRAY_V2I16_CLAMP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b16.clamp", Int16Regs>;
defm SULD_2D_ARRAY_V2I32_CLAMP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b32.clamp", Int32Regs>;
defm SULD_2D_ARRAY_V2I64_CLAMP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b64.clamp", Int64Regs>;

defm SULD_2D_ARRAY_V2I8_TRAP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b8.trap", Int16Regs>;
defm SULD_2D_ARRAY_V2I16_TRAP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b16.trap", Int16Regs>;
defm SULD_2D_ARRAY_V2I32_TRAP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b32.trap", Int32Regs>;
defm SULD_2D_ARRAY_V2I64_TRAP
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b64.trap", Int64Regs>;

defm SULD_2D_ARRAY_V2I8_ZERO
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b8.zero", Int16Regs>;
defm SULD_2D_ARRAY_V2I16_ZERO
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b16.zero", Int16Regs>;
defm SULD_2D_ARRAY_V2I32_ZERO
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b32.zero", Int32Regs>;
defm SULD_2D_ARRAY_V2I64_ZERO
  : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b64.zero", Int64Regs>;

class SULD_3D_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y, Int32Regs:$z)),
                inst # " \\{$r, $g\\}, [$s, \\{$x, $y, $z, $z\\}];",
                pattern>;
multiclass SULD_3D_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_3D_V2_base<inst, outtype, (ins Int64Regs:$s),
                           [(set outtype:$r, outtype:$g,
                                 (intr i64:$s, i32:$x, i32:$y, i32:$z))]>;
  def _I : SULD_3D_V2_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_3D_V2I8_CLAMP : SULD_3D_V2<"suld.b.3d.v2.b8.clamp", Int16Regs>;
defm SULD_3D_V2I16_CLAMP : SULD_3D_V2<"suld.b.3d.v2.b16.clamp", Int16Regs>;
defm SULD_3D_V2I32_CLAMP : SULD_3D_V2<"suld.b.3d.v2.b32.clamp", Int32Regs>;
defm SULD_3D_V2I64_CLAMP : SULD_3D_V2<"suld.b.3d.v2.b64.clamp", Int64Regs>;

defm SULD_3D_V2I8_TRAP : SULD_3D_V2<"suld.b.3d.v2.b8.trap", Int16Regs>;
defm SULD_3D_V2I16_TRAP : SULD_3D_V2<"suld.b.3d.v2.b16.trap", Int16Regs>;
defm SULD_3D_V2I32_TRAP : SULD_3D_V2<"suld.b.3d.v2.b32.trap", Int32Regs>;
defm SULD_3D_V2I64_TRAP : SULD_3D_V2<"suld.b.3d.v2.b64.trap", Int64Regs>;

defm SULD_3D_V2I8_ZERO : SULD_3D_V2<"suld.b.3d.v2.b8.zero", Int16Regs>;
defm SULD_3D_V2I16_ZERO : SULD_3D_V2<"suld.b.3d.v2.b16.zero", Int16Regs>;
defm SULD_3D_V2I32_ZERO : SULD_3D_V2<"suld.b.3d.v2.b32.zero", Int32Regs>;
defm SULD_3D_V2I64_ZERO : SULD_3D_V2<"suld.b.3d.v2.b64.zero", Int64Regs>;

}

let IsSuld = 3 in {

class SULD_1D_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins Int32Regs:$x)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$x\\}];",
                pattern>;
multiclass SULD_1D_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_V4_base<inst, outtype, (ins Int64Regs:$s),
                           [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
                                 (intr i64:$s, i32:$x))]>;
  def _I : SULD_1D_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_V4I8_CLAMP : SULD_1D_V4<"suld.b.1d.v4.b8.clamp", Int16Regs>;
defm SULD_1D_V4I16_CLAMP : SULD_1D_V4<"suld.b.1d.v4.b16.clamp", Int16Regs>;
defm SULD_1D_V4I32_CLAMP : SULD_1D_V4<"suld.b.1d.v4.b32.clamp", Int32Regs>;

defm SULD_1D_V4I8_TRAP : SULD_1D_V4<"suld.b.1d.v4.b8.trap", Int16Regs>;
defm SULD_1D_V4I16_TRAP : SULD_1D_V4<"suld.b.1d.v4.b16.trap", Int16Regs>;
defm SULD_1D_V4I32_TRAP : SULD_1D_V4<"suld.b.1d.v4.b32.trap", Int32Regs>;

defm SULD_1D_V4I8_ZERO : SULD_1D_V4<"suld.b.1d.v4.b8.zero", Int16Regs>;
defm SULD_1D_V4I16_ZERO : SULD_1D_V4<"suld.b.1d.v4.b16.zero", Int16Regs>;
defm SULD_1D_V4I32_ZERO : SULD_1D_V4<"suld.b.1d.v4.b32.zero", Int32Regs>;

class SULD_1D_ARRAY_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                            list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins Int32Regs:$l, Int32Regs:$x)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$l, $x\\}];",
                pattern>;
multiclass SULD_1D_ARRAY_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_ARRAY_V4_base<inst, outtype, (ins Int64Regs:$s),
                                 [(set outtype:$r, outtype:$g, outtype:$b,
                                       outtype:$a,
                                       (intr i64:$s, i32:$l, i32:$x))]>;
  def _I : SULD_1D_ARRAY_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_1D_ARRAY_V4I8_CLAMP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b8.clamp", Int16Regs>;
defm SULD_1D_ARRAY_V4I16_CLAMP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b16.clamp", Int16Regs>;
defm SULD_1D_ARRAY_V4I32_CLAMP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b32.clamp", Int32Regs>;

defm SULD_1D_ARRAY_V4I8_TRAP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b8.trap", Int16Regs>;
defm SULD_1D_ARRAY_V4I16_TRAP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b16.trap", Int16Regs>;
defm SULD_1D_ARRAY_V4I32_TRAP
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b32.trap", Int32Regs>;

defm SULD_1D_ARRAY_V4I8_ZERO
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b8.zero", Int16Regs>;
defm SULD_1D_ARRAY_V4I16_ZERO
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b16.zero", Int16Regs>;
defm SULD_1D_ARRAY_V4I32_ZERO
  : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b32.zero", Int32Regs>;

class SULD_2D_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$x, $y\\}];",
                pattern>;
multiclass SULD_2D_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_V4_base<inst, outtype, (ins Int64Regs:$s),
                           [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
                                 (intr i64:$s, i32:$x, i32:$y))]>;
  def _I : SULD_2D_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_V4I8_CLAMP : SULD_2D_V4<"suld.b.2d.v4.b8.clamp", Int16Regs>;
defm SULD_2D_V4I16_CLAMP : SULD_2D_V4<"suld.b.2d.v4.b16.clamp", Int16Regs>;
defm SULD_2D_V4I32_CLAMP : SULD_2D_V4<"suld.b.2d.v4.b32.clamp", Int32Regs>;

defm SULD_2D_V4I8_TRAP : SULD_2D_V4<"suld.b.2d.v4.b8.trap", Int16Regs>;
defm SULD_2D_V4I16_TRAP : SULD_2D_V4<"suld.b.2d.v4.b16.trap", Int16Regs>;
defm SULD_2D_V4I32_TRAP : SULD_2D_V4<"suld.b.2d.v4.b32.trap", Int32Regs>;

defm SULD_2D_V4I8_ZERO : SULD_2D_V4<"suld.b.2d.v4.b8.zero", Int16Regs>;
defm SULD_2D_V4I16_ZERO : SULD_2D_V4<"suld.b.2d.v4.b16.zero", Int16Regs>;
defm SULD_2D_V4I32_ZERO : SULD_2D_V4<"suld.b.2d.v4.b32.zero", Int32Regs>;

class SULD_2D_ARRAY_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                            list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins Int32Regs:$l, Int32Regs:$x, Int32Regs:$y)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$l, $x, $y, $y\\}];",
                pattern>;
multiclass SULD_2D_ARRAY_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_ARRAY_V4_base<inst, outtype, (ins Int64Regs:$s),
                                 [(set outtype:$r, outtype:$g, outtype:$b,
                                       outtype:$a,
                                       (intr i64:$s, i32:$l, i32:$x, i32:$y))]>;
  def _I : SULD_2D_ARRAY_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_2D_ARRAY_V4I8_CLAMP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b8.clamp", Int16Regs>;
defm SULD_2D_ARRAY_V4I16_CLAMP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b16.clamp", Int16Regs>;
defm SULD_2D_ARRAY_V4I32_CLAMP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b32.clamp", Int32Regs>;

defm SULD_2D_ARRAY_V4I8_TRAP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b8.trap", Int16Regs>;
defm SULD_2D_ARRAY_V4I16_TRAP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b16.trap", Int16Regs>;
defm SULD_2D_ARRAY_V4I32_TRAP
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b32.trap", Int32Regs>;

defm SULD_2D_ARRAY_V4I8_ZERO
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b8.zero", Int16Regs>;
defm SULD_2D_ARRAY_V4I16_ZERO
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b16.zero", Int16Regs>;
defm SULD_2D_ARRAY_V4I32_ZERO
  : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b32.zero", Int32Regs>;

class SULD_3D_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y, Int32Regs:$z)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$x, $y, $z, $z\\}];",
                pattern>;
multiclass SULD_3D_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_3D_V4_base<inst, outtype, (ins Int64Regs:$s),
                           [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
                                 (intr i64:$s, i32:$x, i32:$y, i32:$z))]>;
  def _I : SULD_3D_V4_base<inst, outtype, (ins i64imm:$s)>;
}

defm SULD_3D_V4I8_CLAMP : SULD_3D_V4<"suld.b.3d.v4.b8.clamp", Int16Regs>;
defm SULD_3D_V4I16_CLAMP : SULD_3D_V4<"suld.b.3d.v4.b16.clamp", Int16Regs>;
defm SULD_3D_V4I32_CLAMP : SULD_3D_V4<"suld.b.3d.v4.b32.clamp", Int32Regs>;

defm SULD_3D_V4I8_TRAP : SULD_3D_V4<"suld.b.3d.v4.b8.trap", Int16Regs>;
defm SULD_3D_V4I16_TRAP : SULD_3D_V4<"suld.b.3d.v4.b16.trap", Int16Regs>;
defm SULD_3D_V4I32_TRAP : SULD_3D_V4<"suld.b.3d.v4.b32.trap", Int32Regs>;

defm SULD_3D_V4I8_ZERO : SULD_3D_V4<"suld.b.3d.v4.b8.zero", Int16Regs>;
defm SULD_3D_V4I16_ZERO : SULD_3D_V4<"suld.b.3d.v4.b16.zero", Int16Regs>;
defm SULD_3D_V4I32_ZERO : SULD_3D_V4<"suld.b.3d.v4.b32.zero", Int32Regs>;

}

//-----------------------------------
// Texture Query Intrinsics
//-----------------------------------

let IsSurfTexQuery = true in {
def TXQ_CHANNEL_ORDER_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "txq.channel_order.b32 \t$d, [$a];",
              []>;
def TXQ_CHANNEL_ORDER_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "txq.channel_order.b32 \t$d, [$a];",
              []>;
def TXQ_CHANNEL_DATA_TYPE_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "txq.channel_data_type.b32 \t$d, [$a];",
              []>;
def TXQ_CHANNEL_DATA_TYPE_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "txq.channel_data_type.b32 \t$d, [$a];",
              []>;
def TXQ_WIDTH_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "txq.width.b32 \t$d, [$a];",
              []>;
def TXQ_WIDTH_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "txq.width.b32 \t$d, [$a];",
              []>;
def TXQ_HEIGHT_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "txq.height.b32 \t$d, [$a];",
              []>;
def TXQ_HEIGHT_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "txq.height.b32 \t$d, [$a];",
              []>;
def TXQ_DEPTH_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "txq.depth.b32 \t$d, [$a];",
              []>;
def TXQ_DEPTH_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "txq.depth.b32 \t$d, [$a];",
              []>;
def TXQ_ARRAY_SIZE_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "txq.array_size.b32 \t$d, [$a];",
              []>;
def TXQ_ARRAY_SIZE_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "txq.array_size.b32 \t$d, [$a];",
              []>;
def TXQ_NUM_SAMPLES_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "txq.num_samples.b32 \t$d, [$a];",
              []>;
def TXQ_NUM_SAMPLES_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "txq.num_samples.b32 \t$d, [$a];",
              []>;
def TXQ_NUM_MIPMAP_LEVELS_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "txq.num_mipmap_levels.b32 \t$d, [$a];",
              []>;
def TXQ_NUM_MIPMAP_LEVELS_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "txq.num_mipmap_levels.b32 \t$d, [$a];",
              []>;
}

def : Pat<(int_nvvm_txq_channel_order i64:$a),
          (TXQ_CHANNEL_ORDER_R $a)>;
def : Pat<(int_nvvm_txq_channel_data_type i64:$a),
          (TXQ_CHANNEL_DATA_TYPE_R $a)>;
def : Pat<(int_nvvm_txq_width i64:$a),
          (TXQ_WIDTH_R $a)>;
def : Pat<(int_nvvm_txq_height i64:$a),
          (TXQ_HEIGHT_R $a)>;
def : Pat<(int_nvvm_txq_depth i64:$a),
          (TXQ_DEPTH_R $a)>;
def : Pat<(int_nvvm_txq_array_size i64:$a),
          (TXQ_ARRAY_SIZE_R $a)>;
def : Pat<(int_nvvm_txq_num_samples i64:$a),
          (TXQ_NUM_SAMPLES_R $a)>;
def : Pat<(int_nvvm_txq_num_mipmap_levels i64:$a),
          (TXQ_NUM_MIPMAP_LEVELS_R $a)>;


//-----------------------------------
// Surface Query Intrinsics
//-----------------------------------

let IsSurfTexQuery = true in {
def SUQ_CHANNEL_ORDER_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "suq.channel_order.b32 \t$d, [$a];",
              []>;
def SUQ_CHANNEL_ORDER_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "suq.channel_order.b32 \t$d, [$a];",
              []>;
def SUQ_CHANNEL_DATA_TYPE_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "suq.channel_data_type.b32 \t$d, [$a];",
              []>;
def SUQ_CHANNEL_DATA_TYPE_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "suq.channel_data_type.b32 \t$d, [$a];",
              []>;
def SUQ_WIDTH_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "suq.width.b32 \t$d, [$a];",
              []>;
def SUQ_WIDTH_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "suq.width.b32 \t$d, [$a];",
              []>;
def SUQ_HEIGHT_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "suq.height.b32 \t$d, [$a];",
              []>;
def SUQ_HEIGHT_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "suq.height.b32 \t$d, [$a];",
              []>;
def SUQ_DEPTH_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "suq.depth.b32 \t$d, [$a];",
              []>;
def SUQ_DEPTH_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "suq.depth.b32 \t$d, [$a];",
              []>;
def SUQ_ARRAY_SIZE_R
  : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "suq.array_size.b32 \t$d, [$a];",
              []>;
def SUQ_ARRAY_SIZE_I
  : NVPTXInst<(outs Int32Regs:$d), (ins i64imm:$a),
              "suq.array_size.b32 \t$d, [$a];",
              []>;
}

def : Pat<(int_nvvm_suq_channel_order i64:$a),
          (SUQ_CHANNEL_ORDER_R $a)>;
def : Pat<(int_nvvm_suq_channel_data_type i64:$a),
          (SUQ_CHANNEL_DATA_TYPE_R $a)>;
def : Pat<(int_nvvm_suq_width i64:$a),
          (SUQ_WIDTH_R $a)>;
def : Pat<(int_nvvm_suq_height i64:$a),
          (SUQ_HEIGHT_R $a)>;
def : Pat<(int_nvvm_suq_depth i64:$a),
          (SUQ_DEPTH_R $a)>;
def : Pat<(int_nvvm_suq_array_size i64:$a),
          (SUQ_ARRAY_SIZE_R $a)>;


//===- Handle Query -------------------------------------------------------===//

// TODO: These intrinsics are not yet finalized, pending PTX ISA design work
def ISTYPEP_SAMPLER
  : NVPTXInst<(outs Int1Regs:$d), (ins Int64Regs:$a),
              "istypep.samplerref \t$d, $a;",
              [(set i1:$d, (int_nvvm_istypep_sampler i64:$a))]>;
def ISTYPEP_SURFACE
  : NVPTXInst<(outs Int1Regs:$d), (ins Int64Regs:$a),
              "istypep.surfref \t$d, $a;",
              [(set i1:$d, (int_nvvm_istypep_surface i64:$a))]>;
def ISTYPEP_TEXTURE
  : NVPTXInst<(outs Int1Regs:$d), (ins Int64Regs:$a),
              "istypep.texref \t$d, $a;",
              [(set i1:$d, (int_nvvm_istypep_texture i64:$a))]>;

//===- Surface Stores -----------------------------------------------------===//

let IsSust = true in {

class SUST_1D_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$x, intype:$r)),
                inst # " \t[$s, \\{$x\\}], \\{$r\\};",
                []>;
multiclass SUST_1D<string inst, NVPTXRegClass intype> {
  def _R : SUST_1D_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_1D_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_B8_CLAMP : SUST_1D<"sust.b.1d.b8.clamp", Int16Regs>;
defm SUST_B_1D_B16_CLAMP : SUST_1D<"sust.b.1d.b16.clamp", Int16Regs>;
defm SUST_B_1D_B32_CLAMP : SUST_1D<"sust.b.1d.b32.clamp", Int32Regs>;
defm SUST_B_1D_B64_CLAMP : SUST_1D<"sust.b.1d.b64.clamp", Int64Regs>;

defm SUST_B_1D_B8_TRAP : SUST_1D<"sust.b.1d.b8.trap", Int16Regs>;
defm SUST_B_1D_B16_TRAP : SUST_1D<"sust.b.1d.b16.trap", Int16Regs>;
defm SUST_B_1D_B32_TRAP : SUST_1D<"sust.b.1d.b32.trap", Int32Regs>;
defm SUST_B_1D_B64_TRAP : SUST_1D<"sust.b.1d.b64.trap", Int64Regs>;

defm SUST_B_1D_B8_ZERO : SUST_1D<"sust.b.1d.b8.zero", Int16Regs>;
defm SUST_B_1D_B16_ZERO : SUST_1D<"sust.b.1d.b16.zero", Int16Regs>;
defm SUST_B_1D_B32_ZERO : SUST_1D<"sust.b.1d.b32.zero", Int32Regs>;
defm SUST_B_1D_B64_ZERO : SUST_1D<"sust.b.1d.b64.zero", Int64Regs>;

defm SUST_P_1D_B8_TRAP : SUST_1D<"sust.p.1d.b8.trap", Int16Regs>;
defm SUST_P_1D_B16_TRAP : SUST_1D<"sust.p.1d.b16.trap", Int16Regs>;
defm SUST_P_1D_B32_TRAP : SUST_1D<"sust.p.1d.b32.trap", Int32Regs>;

class SUST_1D_V2_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$x, intype:$r, intype:$g)),
                inst # " \t[$s, \\{$x\\}], \\{$r, $g\\};",
                []>;
multiclass SUST_1D_V2<string inst, NVPTXRegClass intype> {
  def _R : SUST_1D_V2_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_1D_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_V2B8_CLAMP : SUST_1D_V2<"sust.b.1d.v2.b8.clamp", Int16Regs>;
defm SUST_B_1D_V2B16_CLAMP : SUST_1D_V2<"sust.b.1d.v2.b16.clamp", Int16Regs>;
defm SUST_B_1D_V2B32_CLAMP : SUST_1D_V2<"sust.b.1d.v2.b32.clamp", Int32Regs>;
defm SUST_B_1D_V2B64_CLAMP : SUST_1D_V2<"sust.b.1d.v2.b64.clamp", Int64Regs>;

defm SUST_B_1D_V2B8_TRAP : SUST_1D_V2<"sust.b.1d.v2.b8.trap", Int16Regs>;
defm SUST_B_1D_V2B16_TRAP : SUST_1D_V2<"sust.b.1d.v2.b16.trap", Int16Regs>;
defm SUST_B_1D_V2B32_TRAP : SUST_1D_V2<"sust.b.1d.v2.b32.trap", Int32Regs>;
defm SUST_B_1D_V2B64_TRAP : SUST_1D_V2<"sust.b.1d.v2.b64.trap", Int64Regs>;

defm SUST_B_1D_V2B8_ZERO : SUST_1D_V2<"sust.b.1d.v2.b8.zero", Int16Regs>;
defm SUST_B_1D_V2B16_ZERO : SUST_1D_V2<"sust.b.1d.v2.b16.zero", Int16Regs>;
defm SUST_B_1D_V2B32_ZERO : SUST_1D_V2<"sust.b.1d.v2.b32.zero", Int32Regs>;
defm SUST_B_1D_V2B64_ZERO : SUST_1D_V2<"sust.b.1d.v2.b64.zero", Int64Regs>;

defm SUST_P_1D_V2B8_TRAP : SUST_1D_V2<"sust.p.1d.v2.b8.trap", Int16Regs>;
defm SUST_P_1D_V2B16_TRAP : SUST_1D_V2<"sust.p.1d.v2.b16.trap", Int16Regs>;
defm SUST_P_1D_V2B32_TRAP : SUST_1D_V2<"sust.p.1d.v2.b32.trap", Int32Regs>;

class SUST_1D_V4_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$x, intype:$r, intype:$g,
                                intype:$b, intype:$a)),
                inst # " \t[$s, \\{$x\\}], \\{$r, $g, $b, $a\\};",
                []>;
multiclass SUST_1D_V4<string inst, NVPTXRegClass intype> {
  def _R : SUST_1D_V4_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_1D_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_V4B8_CLAMP : SUST_1D_V4<"sust.b.1d.v4.b8.clamp", Int16Regs>;
defm SUST_B_1D_V4B16_CLAMP : SUST_1D_V4<"sust.b.1d.v4.b16.clamp", Int16Regs>;
defm SUST_B_1D_V4B32_CLAMP : SUST_1D_V4<"sust.b.1d.v4.b32.clamp", Int32Regs>;

defm SUST_B_1D_V4B8_TRAP : SUST_1D_V4<"sust.b.1d.v4.b8.trap", Int16Regs>;
defm SUST_B_1D_V4B16_TRAP : SUST_1D_V4<"sust.b.1d.v4.b16.trap", Int16Regs>;
defm SUST_B_1D_V4B32_TRAP : SUST_1D_V4<"sust.b.1d.v4.b32.trap", Int32Regs>;

defm SUST_B_1D_V4B8_ZERO : SUST_1D_V4<"sust.b.1d.v4.b8.zero", Int16Regs>;
defm SUST_B_1D_V4B16_ZERO : SUST_1D_V4<"sust.b.1d.v4.b16.zero", Int16Regs>;
defm SUST_B_1D_V4B32_ZERO : SUST_1D_V4<"sust.b.1d.v4.b32.zero", Int32Regs>;

defm SUST_P_1D_V4B8_TRAP : SUST_1D_V4<"sust.p.1d.v4.b8.trap", Int16Regs>;
defm SUST_P_1D_V4B16_TRAP : SUST_1D_V4<"sust.p.1d.v4.b16.trap", Int16Regs>;
defm SUST_P_1D_V4B32_TRAP : SUST_1D_V4<"sust.p.1d.v4.b32.trap", Int32Regs>;

class SUST_1D_ARRAY_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$idx, Int32Regs:$x, intype:$r)),
                inst # " \t[$s, \\{$idx, $x\\}], \\{$r\\};",
                []>;
multiclass SUST_1D_ARRAY<string inst, NVPTXRegClass intype> {
  def _R : SUST_1D_ARRAY_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_1D_ARRAY_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_ARRAY_B8_CLAMP
  : SUST_1D_ARRAY<"sust.b.a1d.b8.clamp", Int16Regs>;
defm SUST_B_1D_ARRAY_B16_CLAMP
  : SUST_1D_ARRAY<"sust.b.a1d.b16.clamp", Int16Regs>;
defm SUST_B_1D_ARRAY_B32_CLAMP
  : SUST_1D_ARRAY<"sust.b.a1d.b32.clamp", Int32Regs>;
defm SUST_B_1D_ARRAY_B64_CLAMP
  : SUST_1D_ARRAY<"sust.b.a1d.b64.clamp", Int64Regs>;

defm SUST_B_1D_ARRAY_B8_TRAP
  : SUST_1D_ARRAY<"sust.b.a1d.b8.trap", Int16Regs>;
defm SUST_B_1D_ARRAY_B16_TRAP
  : SUST_1D_ARRAY<"sust.b.a1d.b16.trap", Int16Regs>;
defm SUST_B_1D_ARRAY_B32_TRAP
  : SUST_1D_ARRAY<"sust.b.a1d.b32.trap", Int32Regs>;
defm SUST_B_1D_ARRAY_B64_TRAP
  : SUST_1D_ARRAY<"sust.b.a1d.b64.trap", Int64Regs>;

defm SUST_B_1D_ARRAY_B8_ZERO
  : SUST_1D_ARRAY<"sust.b.a1d.b8.zero", Int16Regs>;
defm SUST_B_1D_ARRAY_B16_ZERO
  : SUST_1D_ARRAY<"sust.b.a1d.b16.zero", Int16Regs>;
defm SUST_B_1D_ARRAY_B32_ZERO
  : SUST_1D_ARRAY<"sust.b.a1d.b32.zero", Int32Regs>;
defm SUST_B_1D_ARRAY_B64_ZERO
  : SUST_1D_ARRAY<"sust.b.a1d.b64.zero", Int64Regs>;

defm SUST_P_1D_ARRAY_B8_TRAP
  : SUST_1D_ARRAY<"sust.p.a1d.b8.trap", Int16Regs>;
defm SUST_P_1D_ARRAY_B16_TRAP
  : SUST_1D_ARRAY<"sust.p.a1d.b16.trap", Int16Regs>;
defm SUST_P_1D_ARRAY_B32_TRAP
  : SUST_1D_ARRAY<"sust.p.a1d.b32.trap", Int32Regs>;

class SUST_1D_ARRAY_V2_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$idx, Int32Regs:$x,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$idx, $x\\}], \\{$r, $g\\};",
                []>;
multiclass SUST_1D_ARRAY_V2<string inst, NVPTXRegClass intype> {
  def _R : SUST_1D_ARRAY_V2_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_1D_ARRAY_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_ARRAY_V2B8_CLAMP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b8.clamp", Int16Regs>;
defm SUST_B_1D_ARRAY_V2B16_CLAMP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b16.clamp", Int16Regs>;
defm SUST_B_1D_ARRAY_V2B32_CLAMP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b32.clamp", Int32Regs>;
defm SUST_B_1D_ARRAY_V2B64_CLAMP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b64.clamp", Int64Regs>;

defm SUST_B_1D_ARRAY_V2B8_TRAP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b8.trap", Int16Regs>;
defm SUST_B_1D_ARRAY_V2B16_TRAP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b16.trap", Int16Regs>;
defm SUST_B_1D_ARRAY_V2B32_TRAP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b32.trap", Int32Regs>;
defm SUST_B_1D_ARRAY_V2B64_TRAP
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b64.trap", Int64Regs>;

defm SUST_B_1D_ARRAY_V2B8_ZERO
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b8.zero", Int16Regs>;
defm SUST_B_1D_ARRAY_V2B16_ZERO
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b16.zero", Int16Regs>;
defm SUST_B_1D_ARRAY_V2B32_ZERO
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b32.zero", Int32Regs>;
defm SUST_B_1D_ARRAY_V2B64_ZERO
  : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b64.zero", Int64Regs>;

defm SUST_P_1D_ARRAY_V2B8_TRAP
  : SUST_1D_ARRAY_V2<"sust.p.a1d.v2.b8.trap", Int16Regs>;
defm SUST_P_1D_ARRAY_V2B16_TRAP
  : SUST_1D_ARRAY_V2<"sust.p.a1d.v2.b16.trap", Int16Regs>;
defm SUST_P_1D_ARRAY_V2B32_TRAP
  : SUST_1D_ARRAY_V2<"sust.p.a1d.v2.b32.trap", Int32Regs>;

class SUST_1D_ARRAY_V4_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$idx, Int32Regs:$x,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$idx, $x\\}], \\{$r, $g, $b, $a\\};",
                []>;
multiclass SUST_1D_ARRAY_V4<string inst, NVPTXRegClass intype> {
  def _R : SUST_1D_ARRAY_V4_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_1D_ARRAY_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_1D_ARRAY_V4B8_CLAMP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b8.clamp", Int16Regs>;
defm SUST_B_1D_ARRAY_V4B16_CLAMP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b16.clamp", Int16Regs>;
defm SUST_B_1D_ARRAY_V4B32_CLAMP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b32.clamp", Int32Regs>;

defm SUST_B_1D_ARRAY_V4B8_TRAP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b8.trap", Int16Regs>;
defm SUST_B_1D_ARRAY_V4B16_TRAP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b16.trap", Int16Regs>;
defm SUST_B_1D_ARRAY_V4B32_TRAP
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b32.trap", Int32Regs>;

defm SUST_B_1D_ARRAY_V4B8_ZERO
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b8.zero", Int16Regs>;
defm SUST_B_1D_ARRAY_V4B16_ZERO
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b16.zero", Int16Regs>;
defm SUST_B_1D_ARRAY_V4B32_ZERO
  : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b32.zero", Int32Regs>;

defm SUST_P_1D_ARRAY_V4B8_TRAP
  : SUST_1D_ARRAY_V4<"sust.p.a1d.v4.b8.trap", Int16Regs>;
defm SUST_P_1D_ARRAY_V4B16_TRAP
  : SUST_1D_ARRAY_V4<"sust.p.a1d.v4.b16.trap", Int16Regs>;
defm SUST_P_1D_ARRAY_V4B32_TRAP
  : SUST_1D_ARRAY_V4<"sust.p.a1d.v4.b32.trap", Int32Regs>;

class SUST_2D_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y, intype:$r)),
                inst # " \t[$s, \\{$x, $y\\}], \\{$r\\};",
                []>;
multiclass SUST_2D<string inst, NVPTXRegClass intype> {
  def _R : SUST_2D_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_2D_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_B8_CLAMP : SUST_2D<"sust.b.2d.b8.clamp", Int16Regs>;
defm SUST_B_2D_B16_CLAMP : SUST_2D<"sust.b.2d.b16.clamp", Int16Regs>;
defm SUST_B_2D_B32_CLAMP : SUST_2D<"sust.b.2d.b32.clamp", Int32Regs>;
defm SUST_B_2D_B64_CLAMP : SUST_2D<"sust.b.2d.b64.clamp", Int64Regs>;

defm SUST_B_2D_B8_TRAP : SUST_2D<"sust.b.2d.b8.trap", Int16Regs>;
defm SUST_B_2D_B16_TRAP : SUST_2D<"sust.b.2d.b16.trap", Int16Regs>;
defm SUST_B_2D_B32_TRAP : SUST_2D<"sust.b.2d.b32.trap", Int32Regs>;
defm SUST_B_2D_B64_TRAP : SUST_2D<"sust.b.2d.b64.trap", Int64Regs>;

defm SUST_B_2D_B8_ZERO : SUST_2D<"sust.b.2d.b8.zero", Int16Regs>;
defm SUST_B_2D_B16_ZERO : SUST_2D<"sust.b.2d.b16.zero", Int16Regs>;
defm SUST_B_2D_B32_ZERO : SUST_2D<"sust.b.2d.b32.zero", Int32Regs>;
defm SUST_B_2D_B64_ZERO : SUST_2D<"sust.b.2d.b64.zero", Int64Regs>;

defm SUST_P_2D_B8_TRAP : SUST_2D<"sust.p.2d.b8.trap", Int16Regs>;
defm SUST_P_2D_B16_TRAP : SUST_2D<"sust.p.2d.b16.trap", Int16Regs>;
defm SUST_P_2D_B32_TRAP : SUST_2D<"sust.p.2d.b32.trap", Int32Regs>;

class SUST_2D_V2_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$x, $y\\}], \\{$r, $g\\};",
                []>;
multiclass SUST_2D_V2<string inst, NVPTXRegClass intype> {
  def _R : SUST_2D_V2_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_2D_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_V2B8_CLAMP : SUST_2D_V2<"sust.b.2d.v2.b8.clamp", Int16Regs>;
defm SUST_B_2D_V2B16_CLAMP : SUST_2D_V2<"sust.b.2d.v2.b16.clamp", Int16Regs>;
defm SUST_B_2D_V2B32_CLAMP : SUST_2D_V2<"sust.b.2d.v2.b32.clamp", Int32Regs>;
defm SUST_B_2D_V2B64_CLAMP : SUST_2D_V2<"sust.b.2d.v2.b64.clamp", Int64Regs>;

defm SUST_B_2D_V2B8_TRAP : SUST_2D_V2<"sust.b.2d.v2.b8.trap", Int16Regs>;
defm SUST_B_2D_V2B16_TRAP : SUST_2D_V2<"sust.b.2d.v2.b16.trap", Int16Regs>;
defm SUST_B_2D_V2B32_TRAP : SUST_2D_V2<"sust.b.2d.v2.b32.trap", Int32Regs>;
defm SUST_B_2D_V2B64_TRAP : SUST_2D_V2<"sust.b.2d.v2.b64.trap", Int64Regs>;

defm SUST_B_2D_V2B8_ZERO : SUST_2D_V2<"sust.b.2d.v2.b8.zero", Int16Regs>;
defm SUST_B_2D_V2B16_ZERO : SUST_2D_V2<"sust.b.2d.v2.b16.zero", Int16Regs>;
defm SUST_B_2D_V2B32_ZERO : SUST_2D_V2<"sust.b.2d.v2.b32.zero", Int32Regs>;
defm SUST_B_2D_V2B64_ZERO : SUST_2D_V2<"sust.b.2d.v2.b64.zero", Int64Regs>;

defm SUST_P_2D_V2B8_TRAP : SUST_2D_V2<"sust.p.2d.v2.b8.trap", Int16Regs>;
defm SUST_P_2D_V2B16_TRAP : SUST_2D_V2<"sust.p.2d.v2.b16.trap", Int16Regs>;
defm SUST_P_2D_V2B32_TRAP : SUST_2D_V2<"sust.p.2d.v2.b32.trap", Int32Regs>;

class SUST_2D_V4_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$x, $y\\}], \\{$r, $g, $b, $a\\};",
                []>;
multiclass SUST_2D_V4<string inst, NVPTXRegClass intype> {
  def _R : SUST_2D_V4_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_2D_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_V4B8_CLAMP : SUST_2D_V4<"sust.b.2d.v4.b8.clamp", Int16Regs>;
defm SUST_B_2D_V4B16_CLAMP : SUST_2D_V4<"sust.b.2d.v4.b16.clamp", Int16Regs>;
defm SUST_B_2D_V4B32_CLAMP : SUST_2D_V4<"sust.b.2d.v4.b32.clamp", Int32Regs>;

defm SUST_B_2D_V4B8_TRAP : SUST_2D_V4<"sust.b.2d.v4.b8.trap", Int16Regs>;
defm SUST_B_2D_V4B16_TRAP : SUST_2D_V4<"sust.b.2d.v4.b16.trap", Int16Regs>;
defm SUST_B_2D_V4B32_TRAP : SUST_2D_V4<"sust.b.2d.v4.b32.trap", Int32Regs>;

defm SUST_B_2D_V4B8_ZERO : SUST_2D_V4<"sust.b.2d.v4.b8.zero", Int16Regs>;
defm SUST_B_2D_V4B16_ZERO : SUST_2D_V4<"sust.b.2d.v4.b16.zero", Int16Regs>;
defm SUST_B_2D_V4B32_ZERO : SUST_2D_V4<"sust.b.2d.v4.b32.zero", Int32Regs>;

defm SUST_P_2D_V4B8_TRAP : SUST_2D_V4<"sust.p.2d.v4.b8.trap", Int16Regs>;
defm SUST_P_2D_V4B16_TRAP : SUST_2D_V4<"sust.p.2d.v4.b16.trap", Int16Regs>;
defm SUST_P_2D_V4B32_TRAP : SUST_2D_V4<"sust.p.2d.v4.b32.trap", Int32Regs>;

class SUST_2D_ARRAY_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$idx, Int32Regs:$x, Int32Regs:$y,
                                intype:$r)),
                inst # " \t[$s, \\{$idx, $x, $y, $y\\}], \\{$r\\};",
                []>;
multiclass SUST_2D_ARRAY<string inst, NVPTXRegClass intype> {
  def _R : SUST_2D_ARRAY_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_2D_ARRAY_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_ARRAY_B8_CLAMP
  : SUST_2D_ARRAY<"sust.b.a2d.b8.clamp", Int16Regs>;
defm SUST_B_2D_ARRAY_B16_CLAMP
  : SUST_2D_ARRAY<"sust.b.a2d.b16.clamp", Int16Regs>;
defm SUST_B_2D_ARRAY_B32_CLAMP
  : SUST_2D_ARRAY<"sust.b.a2d.b32.clamp", Int32Regs>;
defm SUST_B_2D_ARRAY_B64_CLAMP
  : SUST_2D_ARRAY<"sust.b.a2d.b64.clamp", Int64Regs>;

defm SUST_B_2D_ARRAY_B8_TRAP
  : SUST_2D_ARRAY<"sust.b.a2d.b8.trap", Int16Regs>;
defm SUST_B_2D_ARRAY_B16_TRAP
  : SUST_2D_ARRAY<"sust.b.a2d.b16.trap", Int16Regs>;
defm SUST_B_2D_ARRAY_B32_TRAP
  : SUST_2D_ARRAY<"sust.b.a2d.b32.trap", Int32Regs>;
defm SUST_B_2D_ARRAY_B64_TRAP
  : SUST_2D_ARRAY<"sust.b.a2d.b64.trap", Int64Regs>;

defm SUST_B_2D_ARRAY_B8_ZERO
  : SUST_2D_ARRAY<"sust.b.a2d.b8.zero", Int16Regs>;
defm SUST_B_2D_ARRAY_B16_ZERO
  : SUST_2D_ARRAY<"sust.b.a2d.b16.zero", Int16Regs>;
defm SUST_B_2D_ARRAY_B32_ZERO
  : SUST_2D_ARRAY<"sust.b.a2d.b32.zero", Int32Regs>;
defm SUST_B_2D_ARRAY_B64_ZERO
  : SUST_2D_ARRAY<"sust.b.a2d.b64.zero", Int64Regs>;

defm SUST_P_2D_ARRAY_B8_TRAP
  : SUST_2D_ARRAY<"sust.p.a2d.b8.trap", Int16Regs>;
defm SUST_P_2D_ARRAY_B16_TRAP
  : SUST_2D_ARRAY<"sust.p.a2d.b16.trap", Int16Regs>;
defm SUST_P_2D_ARRAY_B32_TRAP
  : SUST_2D_ARRAY<"sust.p.a2d.b32.trap", Int32Regs>;

class SUST_2D_ARRAY_V2_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$idx, Int32Regs:$x, Int32Regs:$y,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$idx, $x, $y, $y\\}], \\{$r, $g\\};",
                []>;
multiclass SUST_2D_ARRAY_V2<string inst, NVPTXRegClass intype> {
  def _R : SUST_2D_ARRAY_V2_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_2D_ARRAY_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_ARRAY_V2B8_CLAMP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b8.clamp", Int16Regs>;
defm SUST_B_2D_ARRAY_V2B16_CLAMP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b16.clamp", Int16Regs>;
defm SUST_B_2D_ARRAY_V2B32_CLAMP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b32.clamp", Int32Regs>;
defm SUST_B_2D_ARRAY_V2B64_CLAMP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b64.clamp", Int64Regs>;

defm SUST_B_2D_ARRAY_V2B8_TRAP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b8.trap", Int16Regs>;
defm SUST_B_2D_ARRAY_V2B16_TRAP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b16.trap", Int16Regs>;
defm SUST_B_2D_ARRAY_V2B32_TRAP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b32.trap", Int32Regs>;
defm SUST_B_2D_ARRAY_V2B64_TRAP
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b64.trap", Int64Regs>;

defm SUST_B_2D_ARRAY_V2B8_ZERO
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b8.zero", Int16Regs>;
defm SUST_B_2D_ARRAY_V2B16_ZERO
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b16.zero", Int16Regs>;
defm SUST_B_2D_ARRAY_V2B32_ZERO
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b32.zero", Int32Regs>;
defm SUST_B_2D_ARRAY_V2B64_ZERO
  : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b64.zero", Int64Regs>;

defm SUST_P_2D_ARRAY_V2B8_TRAP
  : SUST_2D_ARRAY_V2<"sust.p.a2d.v2.b8.trap", Int16Regs>;
defm SUST_P_2D_ARRAY_V2B16_TRAP
  : SUST_2D_ARRAY_V2<"sust.p.a2d.v2.b16.trap", Int16Regs>;
defm SUST_P_2D_ARRAY_V2B32_TRAP
  : SUST_2D_ARRAY_V2<"sust.p.a2d.v2.b32.trap", Int32Regs>;

class SUST_2D_ARRAY_V4_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$idx, Int32Regs:$x, Int32Regs:$y,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$idx, $x, $y, $y\\}], \\{$r, $g, $b, $a\\};",
                []>;
multiclass SUST_2D_ARRAY_V4<string inst, NVPTXRegClass intype> {
  def _R : SUST_2D_ARRAY_V4_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_2D_ARRAY_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_2D_ARRAY_V4B8_CLAMP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b8.clamp", Int16Regs>;
defm SUST_B_2D_ARRAY_V4B16_CLAMP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b16.clamp", Int16Regs>;
defm SUST_B_2D_ARRAY_V4B32_CLAMP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b32.clamp", Int32Regs>;

defm SUST_B_2D_ARRAY_V4B8_TRAP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b8.trap", Int16Regs>;
defm SUST_B_2D_ARRAY_V4B16_TRAP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b16.trap", Int16Regs>;
defm SUST_B_2D_ARRAY_V4B32_TRAP
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b32.trap", Int32Regs>;

defm SUST_B_2D_ARRAY_V4B8_ZERO
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b8.zero", Int16Regs>;
defm SUST_B_2D_ARRAY_V4B16_ZERO
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b16.zero", Int16Regs>;
defm SUST_B_2D_ARRAY_V4B32_ZERO
  : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b32.zero", Int32Regs>;

defm SUST_P_2D_ARRAY_V4B8_TRAP
  : SUST_2D_ARRAY_V4<"sust.p.a2d.v4.b8.trap", Int16Regs>;
defm SUST_P_2D_ARRAY_V4B16_TRAP
  : SUST_2D_ARRAY_V4<"sust.p.a2d.v4.b16.trap", Int16Regs>;
defm SUST_P_2D_ARRAY_V4B32_TRAP
  : SUST_2D_ARRAY_V4<"sust.p.a2d.v4.b32.trap", Int32Regs>;

class SUST_3D_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
                                intype:$r)),
                inst # " \t[$s, \\{$x, $y, $z, $z\\}], \\{$r\\};",
                []>;
multiclass SUST_3D<string inst, NVPTXRegClass intype> {
  def _R : SUST_3D_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_3D_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_3D_B8_CLAMP : SUST_3D<"sust.b.3d.b8.clamp", Int16Regs>;
defm SUST_B_3D_B16_CLAMP : SUST_3D<"sust.b.3d.b16.clamp", Int16Regs>;
defm SUST_B_3D_B32_CLAMP : SUST_3D<"sust.b.3d.b32.clamp", Int32Regs>;
defm SUST_B_3D_B64_CLAMP : SUST_3D<"sust.b.3d.b64.clamp", Int64Regs>;

defm SUST_B_3D_B8_TRAP : SUST_3D<"sust.b.3d.b8.trap", Int16Regs>;
defm SUST_B_3D_B16_TRAP : SUST_3D<"sust.b.3d.b16.trap", Int16Regs>;
defm SUST_B_3D_B32_TRAP : SUST_3D<"sust.b.3d.b32.trap", Int32Regs>;
defm SUST_B_3D_B64_TRAP : SUST_3D<"sust.b.3d.b64.trap", Int64Regs>;

defm SUST_B_3D_B8_ZERO : SUST_3D<"sust.b.3d.b8.zero", Int16Regs>;
defm SUST_B_3D_B16_ZERO : SUST_3D<"sust.b.3d.b16.zero", Int16Regs>;
defm SUST_B_3D_B32_ZERO : SUST_3D<"sust.b.3d.b32.zero", Int32Regs>;
defm SUST_B_3D_B64_ZERO : SUST_3D<"sust.b.3d.b64.zero", Int64Regs>;

defm SUST_P_3D_B8_TRAP : SUST_3D<"sust.p.3d.b8.trap", Int16Regs>;
defm SUST_P_3D_B16_TRAP : SUST_3D<"sust.p.3d.b16.trap", Int16Regs>;
defm SUST_P_3D_B32_TRAP : SUST_3D<"sust.p.3d.b32.trap", Int32Regs>;

class SUST_3D_V2_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$x, $y, $z, $z\\}], \\{$r, $g\\};",
                []>;
multiclass SUST_3D_V2<string inst, NVPTXRegClass intype> {
  def _R : SUST_3D_V2_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_3D_V2_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_3D_V2B8_CLAMP : SUST_3D_V2<"sust.b.3d.v2.b8.clamp", Int16Regs>;
defm SUST_B_3D_V2B16_CLAMP : SUST_3D_V2<"sust.b.3d.v2.b16.clamp", Int16Regs>;
defm SUST_B_3D_V2B32_CLAMP : SUST_3D_V2<"sust.b.3d.v2.b32.clamp", Int32Regs>;
defm SUST_B_3D_V2B64_CLAMP : SUST_3D_V2<"sust.b.3d.v2.b64.clamp", Int64Regs>;

defm SUST_B_3D_V2B8_TRAP : SUST_3D_V2<"sust.b.3d.v2.b8.trap", Int16Regs>;
defm SUST_B_3D_V2B16_TRAP : SUST_3D_V2<"sust.b.3d.v2.b16.trap", Int16Regs>;
defm SUST_B_3D_V2B32_TRAP : SUST_3D_V2<"sust.b.3d.v2.b32.trap", Int32Regs>;
defm SUST_B_3D_V2B64_TRAP : SUST_3D_V2<"sust.b.3d.v2.b64.trap", Int64Regs>;

defm SUST_B_3D_V2B8_ZERO : SUST_3D_V2<"sust.b.3d.v2.b8.zero", Int16Regs>;
defm SUST_B_3D_V2B16_ZERO : SUST_3D_V2<"sust.b.3d.v2.b16.zero", Int16Regs>;
defm SUST_B_3D_V2B32_ZERO : SUST_3D_V2<"sust.b.3d.v2.b32.zero", Int32Regs>;
defm SUST_B_3D_V2B64_ZERO : SUST_3D_V2<"sust.b.3d.v2.b64.zero", Int64Regs>;

defm SUST_P_3D_V2B8_TRAP : SUST_3D_V2<"sust.p.3d.v2.b8.trap", Int16Regs>;
defm SUST_P_3D_V2B16_TRAP : SUST_3D_V2<"sust.p.3d.v2.b16.trap", Int16Regs>;
defm SUST_P_3D_V2B32_TRAP : SUST_3D_V2<"sust.p.3d.v2.b32.trap", Int32Regs>;

class SUST_3D_V4_base<string inst, NVPTXRegClass intype, dag surf>
    : NVPTXInst<(outs),
                !con(surf, (ins Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$x, $y, $z, $z\\}], \\{$r, $g, $b, $a\\};",
                []>;
multiclass SUST_3D_V4<string inst, NVPTXRegClass intype> {
  def _R : SUST_3D_V4_base<inst, intype, (ins Int64Regs:$s)>;
  def _I : SUST_3D_V4_base<inst, intype, (ins i64imm:$s)>;
}

defm SUST_B_3D_V4B8_CLAMP : SUST_3D_V4<"sust.b.3d.v4.b8.clamp", Int16Regs>;
defm SUST_B_3D_V4B16_CLAMP : SUST_3D_V4<"sust.b.3d.v4.b16.clamp", Int16Regs>;
defm SUST_B_3D_V4B32_CLAMP : SUST_3D_V4<"sust.b.3d.v4.b32.clamp", Int32Regs>;

defm SUST_B_3D_V4B8_TRAP : SUST_3D_V4<"sust.b.3d.v4.b8.trap", Int16Regs>;
defm SUST_B_3D_V4B16_TRAP : SUST_3D_V4<"sust.b.3d.v4.b16.trap", Int16Regs>;
defm SUST_B_3D_V4B32_TRAP : SUST_3D_V4<"sust.b.3d.v4.b32.trap", Int32Regs>;

defm SUST_B_3D_V4B8_ZERO : SUST_3D_V4<"sust.b.3d.v4.b8.zero", Int16Regs>;
defm SUST_B_3D_V4B16_ZERO : SUST_3D_V4<"sust.b.3d.v4.b16.zero", Int16Regs>;
defm SUST_B_3D_V4B32_ZERO : SUST_3D_V4<"sust.b.3d.v4.b32.zero", Int32Regs>;

defm SUST_P_3D_V4B8_TRAP : SUST_3D_V4<"sust.p.3d.v4.b8.trap", Int16Regs>;
defm SUST_P_3D_V4B16_TRAP : SUST_3D_V4<"sust.p.3d.v4.b16.trap", Int16Regs>;
defm SUST_P_3D_V4B32_TRAP : SUST_3D_V4<"sust.p.3d.v4.b32.trap", Int32Regs>;

}

// Surface store instruction patterns
// I'm not sure why we can't just include these in the instruction definitions,
// but TableGen complains of type errors :(

// .clamp variant
def : Pat<(int_nvvm_sust_b_1d_i8_clamp
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_B8_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_i16_clamp
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_B16_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_i32_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$r),
          (SUST_B_1D_B32_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_i64_clamp
           Int64Regs:$s, Int32Regs:$x, Int64Regs:$r),
          (SUST_B_1D_B64_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_v2i8_clamp
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_V2B8_CLAMP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v2i16_clamp
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_V2B16_CLAMP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v2i32_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$r, Int32Regs:$g),
          (SUST_B_1D_V2B32_CLAMP_R Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v2i64_clamp
           Int64Regs:$s, Int32Regs:$x, Int64Regs:$r, Int64Regs:$g),
          (SUST_B_1D_V2B64_CLAMP_R Int64Regs:$s, Int32Regs:$x,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v4i8_clamp
           Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_V4B8_CLAMP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_v4i16_clamp
           Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_V4B16_CLAMP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_v4i32_clamp
           Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_1D_V4B32_CLAMP_R Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_1d_array_i8_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_ARRAY_B8_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_i16_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_ARRAY_B16_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_i32_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$r),
          (SUST_B_1D_ARRAY_B32_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_i64_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int64Regs:$r),
          (SUST_B_1D_ARRAY_B64_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i8_clamp
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_ARRAY_V2B8_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i16_clamp
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_ARRAY_V2B16_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i32_clamp
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$r, Int32Regs:$g),
          (SUST_B_1D_ARRAY_V2B32_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i64_clamp
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int64Regs:$r, Int64Regs:$g),
          (SUST_B_1D_ARRAY_V2B64_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v4i8_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_ARRAY_V4B8_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_array_v4i16_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_ARRAY_V4B16_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_array_v4i32_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_1D_ARRAY_V4B32_CLAMP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_2d_i8_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_B8_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_i16_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_B16_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_i32_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r),
          (SUST_B_2D_B32_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_i64_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r),
          (SUST_B_2D_B64_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_v2i8_clamp
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_V2B8_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v2i16_clamp
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_V2B16_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v2i32_clamp
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r, Int32Regs:$g),
          (SUST_B_2D_V2B32_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v2i64_clamp
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r, Int64Regs:$g),
          (SUST_B_2D_V2B64_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v4i8_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_V4B8_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_v4i16_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_V4B16_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_v4i32_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_2D_V4B32_CLAMP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_2d_array_i8_clamp
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_ARRAY_B8_CLAMP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_i16_clamp
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_ARRAY_B16_CLAMP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_i32_clamp
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r),
          (SUST_B_2D_ARRAY_B32_CLAMP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_i64_clamp
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r),
          (SUST_B_2D_ARRAY_B64_CLAMP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i8_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_ARRAY_V2B8_CLAMP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i16_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_ARRAY_V2B16_CLAMP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i32_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r,
           Int32Regs:$g),
          (SUST_B_2D_ARRAY_V2B32_CLAMP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i64_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r,
           Int64Regs:$g),
          (SUST_B_2D_ARRAY_V2B64_CLAMP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y, Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v4i8_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_ARRAY_V4B8_CLAMP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_array_v4i16_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_ARRAY_V4B16_CLAMP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_array_v4i32_clamp
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_2D_ARRAY_V4B32_CLAMP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_3d_i8_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r),
          (SUST_B_3D_B8_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_i16_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r),
          (SUST_B_3D_B16_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_i32_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r),
          (SUST_B_3D_B32_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_i64_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r),
          (SUST_B_3D_B64_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_v2i8_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_3D_V2B8_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v2i16_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_3D_V2B16_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v2i32_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g),
          (SUST_B_3D_V2B32_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v2i64_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r, Int64Regs:$g),
          (SUST_B_3D_V2B64_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v4i8_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_3D_V4B8_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_3d_v4i16_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_3D_V4B16_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_3d_v4i32_clamp
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_3D_V4B32_CLAMP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;


// .trap variant
def : Pat<(int_nvvm_sust_b_1d_i8_trap
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_B8_TRAP_R Int64Regs:$s, Int32Regs:$x, Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_i16_trap
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_B16_TRAP_R Int64Regs:$s, Int32Regs:$x, Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$r),
          (SUST_B_1D_B32_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_i64_trap
           Int64Regs:$s, Int32Regs:$x, Int64Regs:$r),
          (SUST_B_1D_B64_TRAP_R Int64Regs:$s, Int32Regs:$x, Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_v2i8_trap
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_V2B8_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v2i16_trap
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_V2B16_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v2i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$r, Int32Regs:$g),
          (SUST_B_1D_V2B32_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v2i64_trap
           Int64Regs:$s, Int32Regs:$x, Int64Regs:$r, Int64Regs:$g),
          (SUST_B_1D_V2B64_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v4i8_trap
           Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_V4B8_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_v4i16_trap
           Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_V4B16_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_v4i32_trap
           Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_1D_V4B32_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_1d_array_i8_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_ARRAY_B8_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_i16_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_ARRAY_B16_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_i32_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$r),
          (SUST_B_1D_ARRAY_B32_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_i64_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int64Regs:$r),
          (SUST_B_1D_ARRAY_B64_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i8_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_ARRAY_V2B8_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i16_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_ARRAY_V2B16_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i32_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$r, Int32Regs:$g),
          (SUST_B_1D_ARRAY_V2B32_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i64_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int64Regs:$r, Int64Regs:$g),
          (SUST_B_1D_ARRAY_V2B64_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v4i8_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_ARRAY_V4B8_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_array_v4i16_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_ARRAY_V4B16_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_array_v4i32_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_1D_ARRAY_V4B32_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_2d_i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_B8_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_B16_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r),
          (SUST_B_2D_B32_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_i64_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r),
          (SUST_B_2D_B64_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_v2i8_trap
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_V2B8_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v2i16_trap
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_V2B16_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v2i32_trap
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r, Int32Regs:$g),
          (SUST_B_2D_V2B32_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v2i64_trap
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r, Int64Regs:$g),
          (SUST_B_2D_V2B64_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v4i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_V4B8_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_v4i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_V4B16_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_v4i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_2D_V4B32_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_2d_array_i8_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_ARRAY_B8_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_i16_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_ARRAY_B16_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_i32_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r),
          (SUST_B_2D_ARRAY_B32_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_i64_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r),
          (SUST_B_2D_ARRAY_B64_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i8_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_ARRAY_V2B8_TRAP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i16_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_ARRAY_V2B16_TRAP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i32_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r,
           Int32Regs:$g),
          (SUST_B_2D_ARRAY_V2B32_TRAP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i64_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r,
           Int64Regs:$g),
          (SUST_B_2D_ARRAY_V2B64_TRAP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y, Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v4i8_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_ARRAY_V4B8_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_array_v4i16_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_ARRAY_V4B16_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_array_v4i32_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_2D_ARRAY_V4B32_TRAP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_3d_i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r),
          (SUST_B_3D_B8_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r),
          (SUST_B_3D_B16_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r),
          (SUST_B_3D_B32_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_i64_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r),
          (SUST_B_3D_B64_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_v2i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_3D_V2B8_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v2i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_3D_V2B16_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v2i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g),
          (SUST_B_3D_V2B32_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v2i64_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r, Int64Regs:$g),
          (SUST_B_3D_V2B64_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v4i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_3D_V4B8_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_3d_v4i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_3D_V4B16_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_3d_v4i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_3D_V4B32_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;


// .zero variant
def : Pat<(int_nvvm_sust_b_1d_i8_zero
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_B8_ZERO_R Int64Regs:$s, Int32Regs:$x, Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_i16_zero
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_B16_ZERO_R Int64Regs:$s, Int32Regs:$x, Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_i32_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$r),
          (SUST_B_1D_B32_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_i64_zero
           Int64Regs:$s, Int32Regs:$x, Int64Regs:$r),
          (SUST_B_1D_B64_ZERO_R Int64Regs:$s, Int32Regs:$x, Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_v2i8_zero
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_V2B8_ZERO_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v2i16_zero
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_V2B16_ZERO_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v2i32_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$r, Int32Regs:$g),
          (SUST_B_1D_V2B32_ZERO_R Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v2i64_zero
           Int64Regs:$s, Int32Regs:$x, Int64Regs:$r, Int64Regs:$g),
          (SUST_B_1D_V2B64_ZERO_R Int64Regs:$s, Int32Regs:$x,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_v4i8_zero
           Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_V4B8_ZERO_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_v4i16_zero
           Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_V4B16_ZERO_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_v4i32_zero
           Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_1D_V4B32_ZERO_R Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_1d_array_i8_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_ARRAY_B8_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_i16_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r),
          (SUST_B_1D_ARRAY_B16_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_i32_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$r),
          (SUST_B_1D_ARRAY_B32_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_i64_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int64Regs:$r),
          (SUST_B_1D_ARRAY_B64_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i8_zero
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_ARRAY_V2B8_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i16_zero
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_1D_ARRAY_V2B16_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i32_zero
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$r, Int32Regs:$g),
          (SUST_B_1D_ARRAY_V2B32_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v2i64_zero
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int64Regs:$r, Int64Regs:$g),
          (SUST_B_1D_ARRAY_V2B64_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_1d_array_v4i8_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_ARRAY_V4B8_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_array_v4i16_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_1D_ARRAY_V4B16_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_1d_array_v4i32_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_1D_ARRAY_V4B32_ZERO_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_2d_i8_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_B8_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_i16_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_B16_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_i32_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r),
          (SUST_B_2D_B32_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_i64_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r),
          (SUST_B_2D_B64_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_v2i8_zero
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_V2B8_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v2i16_zero
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_V2B16_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v2i32_zero
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r, Int32Regs:$g),
          (SUST_B_2D_V2B32_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v2i64_zero
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r, Int64Regs:$g),
          (SUST_B_2D_V2B64_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_v4i8_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_V4B8_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_v4i16_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_V4B16_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_v4i32_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_2D_V4B32_ZERO_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_2d_array_i8_zero
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_ARRAY_B8_ZERO_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_i16_zero
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_B_2D_ARRAY_B16_ZERO_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_i32_zero
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r),
          (SUST_B_2D_ARRAY_B32_ZERO_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_i64_zero
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r),
          (SUST_B_2D_ARRAY_B64_ZERO_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i8_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_ARRAY_V2B8_ZERO_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i16_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_2D_ARRAY_V2B16_ZERO_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i32_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r,
           Int32Regs:$g),
          (SUST_B_2D_ARRAY_V2B32_ZERO_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v2i64_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int64Regs:$r,
           Int64Regs:$g),
          (SUST_B_2D_ARRAY_V2B64_ZERO_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y, Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_2d_array_v4i8_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_ARRAY_V4B8_ZERO_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_array_v4i16_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_2D_ARRAY_V4B16_ZERO_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_2d_array_v4i32_zero
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_2D_ARRAY_V4B32_ZERO_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_b_3d_i8_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r),
          (SUST_B_3D_B8_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_i16_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r),
          (SUST_B_3D_B16_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_i32_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r),
          (SUST_B_3D_B32_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_i64_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r),
          (SUST_B_3D_B64_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r)>;

def : Pat<(int_nvvm_sust_b_3d_v2i8_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_3D_V2B8_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v2i16_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_B_3D_V2B16_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v2i32_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g),
          (SUST_B_3D_V2B32_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v2i64_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r, Int64Regs:$g),
          (SUST_B_3D_V2B64_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int64Regs:$r, Int64Regs:$g)>;

def : Pat<(int_nvvm_sust_b_3d_v4i8_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_3D_V4B8_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_3d_v4i16_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_B_3D_V4B16_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_b_3d_v4i32_zero
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_B_3D_V4B32_ZERO_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;




def : Pat<(int_nvvm_sust_p_1d_i8_trap
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r),
          (SUST_P_1D_B8_TRAP_R Int64Regs:$s, Int32Regs:$x, Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_1d_i16_trap
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r),
          (SUST_P_1D_B16_TRAP_R Int64Regs:$s, Int32Regs:$x, Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_1d_i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$r),
          (SUST_P_1D_B32_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_p_1d_v2i8_trap
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_P_1D_V2B8_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_1d_v2i16_trap
           Int64Regs:$s, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_P_1D_V2B16_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_1d_v2i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$r, Int32Regs:$g),
          (SUST_P_1D_V2B32_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_p_1d_v4i8_trap
           Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_1D_V4B8_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_1d_v4i16_trap
           Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_1D_V4B16_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_1d_v4i32_trap
           Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_P_1D_V4B32_TRAP_R Int64Regs:$s, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_p_1d_array_i8_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r),
          (SUST_P_1D_ARRAY_B8_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_1d_array_i16_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r),
          (SUST_P_1D_ARRAY_B16_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_1d_array_i32_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$r),
          (SUST_P_1D_ARRAY_B32_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_p_1d_array_v2i8_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_P_1D_ARRAY_V2B8_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_1d_array_v2i16_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int16Regs:$r, Int16Regs:$g),
          (SUST_P_1D_ARRAY_V2B16_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_1d_array_v2i32_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$r, Int32Regs:$g),
          (SUST_P_1D_ARRAY_V2B32_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_p_1d_array_v4i8_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_1D_ARRAY_V4B8_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_1d_array_v4i16_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_1D_ARRAY_V4B16_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_1d_array_v4i32_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_P_1D_ARRAY_V4B32_TRAP_R Int64Regs:$s, Int32Regs:$l, Int32Regs:$x,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_p_2d_i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_P_2D_B8_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_2d_i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_P_2D_B16_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_2d_i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r),
          (SUST_P_2D_B32_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_p_2d_v2i8_trap
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r, Int16Regs:$g),
          (SUST_P_2D_V2B8_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_2d_v2i16_trap
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r, Int16Regs:$g),
          (SUST_P_2D_V2B16_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_2d_v2i32_trap
          Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r, Int32Regs:$g),
          (SUST_P_2D_V2B32_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_p_2d_v4i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_2D_V4B8_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_2d_v4i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_2D_V4B16_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_2d_v4i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_P_2D_V4B32_TRAP_R Int64Regs:$s, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_p_2d_array_i8_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_P_2D_ARRAY_B8_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_2d_array_i16_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int16Regs:$r),
          (SUST_P_2D_ARRAY_B16_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_2d_array_i32_trap
          Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r),
          (SUST_P_2D_ARRAY_B32_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_p_2d_array_v2i8_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_P_2D_ARRAY_V2B8_TRAP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_2d_array_v2i16_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_P_2D_ARRAY_V2B16_TRAP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_2d_array_v2i32_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y, Int32Regs:$r,
           Int32Regs:$g),
          (SUST_P_2D_ARRAY_V2B32_TRAP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_p_2d_array_v4i8_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_2D_ARRAY_V4B8_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_2d_array_v4i16_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_2D_ARRAY_V4B16_TRAP_R Int64Regs:$s,
           Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_2d_array_v4i32_trap
           Int64Regs:$s, Int32Regs:$l, Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_P_2D_ARRAY_V4B32_TRAP_R Int64Regs:$s, Int32Regs:$l,
           Int32Regs:$x, Int32Regs:$y,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;



def : Pat<(int_nvvm_sust_p_3d_i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r),
          (SUST_P_3D_B8_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_3d_i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r),
          (SUST_P_3D_B16_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r)>;

def : Pat<(int_nvvm_sust_p_3d_i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r),
          (SUST_P_3D_B32_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r)>;

def : Pat<(int_nvvm_sust_p_3d_v2i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_P_3D_V2B8_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_3d_v2i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g),
          (SUST_P_3D_V2B16_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g)>;

def : Pat<(int_nvvm_sust_p_3d_v2i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g),
          (SUST_P_3D_V2B32_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g)>;

def : Pat<(int_nvvm_sust_p_3d_v4i8_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_3D_V4B8_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_3d_v4i16_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a),
          (SUST_P_3D_V4B16_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int16Regs:$r, Int16Regs:$g, Int16Regs:$b, Int16Regs:$a)>;

def : Pat<(int_nvvm_sust_p_3d_v4i32_trap
           Int64Regs:$s, Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a),
          (SUST_P_3D_V4B32_TRAP_R Int64Regs:$s,
           Int32Regs:$x, Int32Regs:$y, Int32Regs:$z,
           Int32Regs:$r, Int32Regs:$g, Int32Regs:$b, Int32Regs:$a)>;

//-----------------------------------
// Read Special Registers
//-----------------------------------

class PTX_READ_SREG_R64<string regname, Intrinsic intop, list<Predicate> Preds=[]>
  : NVPTXInst<(outs Int64Regs:$d), (ins),
              !strconcat("mov.u64 \t$d, %", regname, ";"),
              [(set i64:$d, (intop))]>,
    Requires<Preds>;

class PTX_READ_SREG_R32<string regname, Intrinsic intop, list<Predicate> Preds=[]>
  : NVPTXInst<(outs Int32Regs:$d), (ins),
              !strconcat("mov.u32 \t$d, %", regname, ";"),
              [(set i32:$d, (intop))]>,
    Requires<Preds>;

multiclass PTX_READ_SREG_R32V4<string regname, list<Predicate> Preds=[]> {
   foreach suffix = ["x", "y", "z", "w"] in {
      defvar reg = regname # "." # suffix;
      defvar intr = !cast<Intrinsic>("int_nvvm_read_ptx_sreg_" # regname # "_" # suffix);
      def "_"#suffix :  PTX_READ_SREG_R32<reg, intr, Preds>;
   }
}

// TODO Add read vector-version of special registers

defm INT_PTX_SREG_TID   : PTX_READ_SREG_R32V4<"tid">;
defm INT_PTX_SREG_NTID  : PTX_READ_SREG_R32V4<"ntid">;
defm INT_PTX_SREG_CTAID : PTX_READ_SREG_R32V4<"ctaid">;
defm INT_PTX_SREG_NCTAID: PTX_READ_SREG_R32V4<"nctaid">;

defm INT_PTX_SREG_CLUSTERID :
       PTX_READ_SREG_R32V4<"clusterid", [hasSM<90>, hasPTX<78>]>;
defm INT_PTX_SREG_NCLUSTERID :
       PTX_READ_SREG_R32V4<"nclusterid", [hasSM<90>, hasPTX<78>]>;
defm INT_PTX_SREG_CLUSTER_CTAID :
       PTX_READ_SREG_R32V4<"cluster_ctaid", [hasSM<90>, hasPTX<78>]>;
defm INT_PTX_SREG_CLUSTER_NCTAID:
       PTX_READ_SREG_R32V4<"cluster_nctaid", [hasSM<90>, hasPTX<78>]>;

def  INT_PTX_SREG_CLUSTER_CTARANK :
       PTX_READ_SREG_R32<"cluster_ctarank",
                         int_nvvm_read_ptx_sreg_cluster_ctarank,
                         [hasSM<90>, hasPTX<78>]>;
def  INT_PTX_SREG_CLUSTER_NCTARANK:
       PTX_READ_SREG_R32<"cluster_nctarank",
                         int_nvvm_read_ptx_sreg_cluster_nctarank,
                         [hasSM<90>, hasPTX<78>]>;


def INT_PTX_SREG_LANEID :
    PTX_READ_SREG_R32<"laneid", int_nvvm_read_ptx_sreg_laneid>;
def INT_PTX_SREG_WARPID :
    PTX_READ_SREG_R32<"warpid", int_nvvm_read_ptx_sreg_warpid>;
def INT_PTX_SREG_NWARPID :
    PTX_READ_SREG_R32<"nwarpid", int_nvvm_read_ptx_sreg_nwarpid>;
def INT_PTX_SREG_SMID :
    PTX_READ_SREG_R32<"smid", int_nvvm_read_ptx_sreg_smid>;
def INT_PTX_SREG_NSMID :
    PTX_READ_SREG_R32<"nsmid", int_nvvm_read_ptx_sreg_nsmid>;
def INT_PTX_SREG_GRIDID :
    PTX_READ_SREG_R32<"gridid", int_nvvm_read_ptx_sreg_gridid>;

def INT_PTX_SREG_LANEMASK_EQ :
    PTX_READ_SREG_R32<"lanemask_eq", int_nvvm_read_ptx_sreg_lanemask_eq>;
def INT_PTX_SREG_LANEMASK_LE :
    PTX_READ_SREG_R32<"lanemask_le", int_nvvm_read_ptx_sreg_lanemask_le>;
def INT_PTX_SREG_LANEMASK_LT :
    PTX_READ_SREG_R32<"lanemask_lt", int_nvvm_read_ptx_sreg_lanemask_lt>;
def INT_PTX_SREG_LANEMASK_GE :
    PTX_READ_SREG_R32<"lanemask_ge", int_nvvm_read_ptx_sreg_lanemask_ge>;
def INT_PTX_SREG_LANEMASK_GT :
    PTX_READ_SREG_R32<"lanemask_gt", int_nvvm_read_ptx_sreg_lanemask_gt>;

let hasSideEffects = 1 in {
def INT_PTX_SREG_CLOCK :
    PTX_READ_SREG_R32<"clock", int_nvvm_read_ptx_sreg_clock>;
def INT_PTX_SREG_CLOCK64 :
    PTX_READ_SREG_R64<"clock64", int_nvvm_read_ptx_sreg_clock64>;
def INT_PTX_SREG_GLOBALTIMER :
    PTX_READ_SREG_R64<"globaltimer", int_nvvm_read_ptx_sreg_globaltimer>;
}

def: Pat <(i64 (readcyclecounter)), (INT_PTX_SREG_CLOCK64)>;
def: Pat <(i64 (readsteadycounter)), (INT_PTX_SREG_GLOBALTIMER)>;

def INT_PTX_SREG_PM0 : PTX_READ_SREG_R32<"pm0", int_nvvm_read_ptx_sreg_pm0>;
def INT_PTX_SREG_PM1 : PTX_READ_SREG_R32<"pm1", int_nvvm_read_ptx_sreg_pm1>;
def INT_PTX_SREG_PM2 : PTX_READ_SREG_R32<"pm2", int_nvvm_read_ptx_sreg_pm2>;
def INT_PTX_SREG_PM3 : PTX_READ_SREG_R32<"pm3", int_nvvm_read_ptx_sreg_pm3>;

// TODO: It would be nice to use PTX_READ_SREG here, but it doesn't
// handle the constant.
def INT_PTX_SREG_WARPSIZE :
    NVPTXInst<(outs Int32Regs:$dst), (ins), "mov.u32 \t$dst, WARP_SZ;",
              [(set i32:$dst, (int_nvvm_read_ptx_sreg_warpsize))]>;

// Helper class that represents a 'fragment' of an NVPTX *MMA instruction.
// In addition to target-independent fields provided by WMMA_REGS, it adds
// the fields commonly used to implement specific PTX instruction -- register
// types and names, constraints, parts of assembly, etc.
class WMMA_REGINFO<WMMA_REGS r, string op>
      : WMMA_REGS<r.geom, r.frag, r.ptx_elt_type> {
  // NVPTX register types used to carry fragment data.
  NVPTXRegClass regclass = !cond(
    !eq(ptx_elt_type, "f16") : Int32Regs,
    !eq(ptx_elt_type, "f32") : Float32Regs,
    !eq(ptx_elt_type, "f64") : Float64Regs,
    !eq(ptx_elt_type, "bf16") : Int32Regs,
    !eq(ptx_elt_type, "tf32") : Int32Regs,
    !eq(ptx_elt_type, "s32") : Int32Regs,
    !eq(ptx_elt_type, "b16") : Int32Regs,
    !eq(ptx_elt_type, "b8") : Int32Regs,
    !eq(ptx_elt_type, "b8x16.b6x16_p32") : Int32Regs,
    !eq(ptx_elt_type, "b8x16.b4x16_p64") : Int32Regs,
    !eq(ptx_elt_type, "s8") : Int32Regs,
    !eq(ptx_elt_type, "u8") : Int32Regs,
    !eq(ptx_elt_type, "s4") : Int32Regs,
    !eq(ptx_elt_type, "u4") : Int32Regs,
    !eq(ptx_elt_type, "b1") : Int32Regs);

  // Instruction input/output arguments for the fragment.
  list<NVPTXRegClass> ptx_regs = !listsplat(regclass, !size(regs));

  // List of register names for the fragment -- ["ra0", "ra1",...]
  list<string> reg_names = RegSeq<!size(ptx_regs), "r"#frag>.ret;

  // Generates "{{$r0, $r1,.... $rN-1}}" for use in asm string construction.
  string regstring = "{{$" # !interleave(reg_names, ", $") # "}}";

  // Predicates for particular fragment variant. Technically those are
  // per-instruction predicates, but currently all fragments that can be used in
  // a given instruction are subject to the same constraints, so an instruction
  // can use predicates from any of its fragments. If/when this is no
  // longer the case, we can concat all per-fragment predicates to enforce that
  // all fragments of the instruction are viable.
  list<Predicate> Predicates = !cond(
    // fp16 -> fp16/fp32 @ m16n16k16
    !and(!eq(geom, "m16n16k16"),
         !or(!eq(ptx_elt_type, "f16"),
             !eq(ptx_elt_type, "f32"))) : [hasSM<70>, hasPTX<60>],

    !and(!eq(geom,"m8n8k4"),
         !eq(ptx_elt_type, "f64")) : [hasSM<80>, hasPTX<70>],

    // fp16 -> fp16/fp32 @ m8n32k16/m32n8k16
    !and(!or(!eq(geom, "m8n32k16"),
             !eq(geom, "m32n8k16")),
         !or(!eq(ptx_elt_type, "f16"),
             !eq(ptx_elt_type, "f32"))) : [hasSM<70>, hasPTX<61>],

    // u8/s8 -> s32 @ m16n16k16/m8n32k16/m32n8k16
    !and(!or(!eq(geom,"m16n16k16"),
             !eq(geom,"m8n32k16"),
             !eq(geom,"m32n8k16")),
         !or(!eq(ptx_elt_type, "u8"),
             !eq(ptx_elt_type, "s8"),
             !eq(ptx_elt_type, "s32"))) : [hasSM<72>, hasPTX<63>],

    !and(!or(!eq(geom,"m16n16k16"),
             !eq(geom,"m8n32k16"),
             !eq(geom,"m32n8k16")),
         !eq(ptx_elt_type, "bf16")) : [hasSM<80>, hasPTX<70>],

    !and(!eq(geom,"m16n16k8"),
         !eq(ptx_elt_type, "tf32")) : [hasSM<80>, hasPTX<70>],

    !and(!eq(geom,"m16n16k8"),
         !eq(ptx_elt_type, "f32")) : [hasSM<80>, hasPTX<70>],

    // b1 -> s32 @ m8n8k128(b1)
    !and(!ne(op,"mma"),
         !eq(geom,"m8n8k128")) : [hasSM<75>, hasPTX<63>],

    // u4/s4 -> s32 @ m8n8k32 (u4/s4)
    !and(!ne(op,"mma"),
         !eq(geom,"m8n8k32")) : [hasSM<75>, hasPTX<63>],

    !or(!eq(geom,"m16n8k8"),
        !eq(geom,"m8n8k16")) : [hasSM<75>, hasPTX<65>],

    !and(!ne(ptx_elt_type,"f64"),
         !eq(geom, "m8n8k4")) : [hasSM<70>, hasPTX<64>],

    // mma m8n8k32 requires higher PTX version
    !and(!eq(op,"mma"),
         !eq(geom,"m8n8k32")) : [hasSM<75>, hasPTX<65>],

    !and(!eq(ptx_elt_type,"f64"),
         !eq(geom, "m8n8k4")) : [hasSM<80>, hasPTX<70>],

    !and(!eq(op,"mma"),
         !or(!eq(geom, "m16n8k16"),
             !eq(geom, "m16n8k4"),
             !eq(geom, "m16n8k32"),
             !eq(geom, "m16n8k64"),
             !eq(geom, "m8n8k128"),
             !eq(geom, "m16n8k128"),
             !eq(geom, "m16n8k256"))) : [hasSM<80>, hasPTX<70>],

    !and(!eq(op,"ldmatrix"),
         !eq(ptx_elt_type,"b16"),
         !eq(geom, "m8n8")) : [hasSM<75>, hasPTX<65>],

    !and(!eq(op,"ldmatrix"),
         !eq(ptx_elt_type,"b8"),
         !eq(geom, "m16n16")) : [hasSM<100>, hasAAFeatures, hasPTX<86>],

    !and(!eq(op,"ldmatrix"),
         !eq(ptx_elt_type,"b8x16.b6x16_p32"),
         !eq(geom, "m16n16")) : [hasSM<100>, hasAAFeatures, hasPTX<86>],

    !and(!eq(op,"ldmatrix"),
         !eq(ptx_elt_type,"b8x16.b4x16_p64"),
         !eq(geom, "m16n16")) : [hasSM<100>, hasAAFeatures, hasPTX<86>],

    !and(!eq(op,"ldmatrix"),
         !eq(ptx_elt_type,"b8x16.b6x16_p32"),
         !eq(geom, "m8n16")) : [hasSM<100>, hasAAFeatures, hasPTX<86>],

    !and(!eq(op,"ldmatrix"),
         !eq(ptx_elt_type,"b8x16.b4x16_p64"),
         !eq(geom, "m8n16")) : [hasSM<100>, hasAAFeatures, hasPTX<86>]);

  // template DAGs for instruction inputs/output.
  dag Outs = !dag(outs, ptx_regs, reg_names);
  dag Ins = !dag(ins, ptx_regs, reg_names);
}

// Convert dag of arguments into a dag to match given intrinsic.
class BuildPatternI<Intrinsic Intr, dag Ins> {
  // Build a dag pattern that matches the intrinsic call.
  dag ret = !foreach(tmp, Ins,
                          !subst(imem, ADDRvar,
                          !subst(MEMri64, ADDRri64,
                          !subst(MEMri, ADDRri,
                          !subst(ins, Intr, tmp)))));
}

// Same as above, but uses PatFrag instead of an Intrinsic.
class BuildPatternPF<PatFrag Intr, dag Ins> {
  // Build a dag pattern that matches the intrinsic call.
  dag ret = !foreach(tmp, Ins,
                          !subst(imem, ADDRvar,
                          !subst(MEMri64, ADDRri64,
                          !subst(MEMri, ADDRri,
                          !subst(ins, Intr, tmp)))));
}

// Common WMMA-related fields used for building patterns for all MMA instructions.
class WMMA_INSTR<string _Intr, list<dag> _Args>
  : NVPTXInst<(outs), (ins), "?", []> {
  Intrinsic Intr = !cast<Intrinsic>(_Intr);
  // Concatenate all arguments into a single dag.
  dag Args = !foldl((ins), _Args, a, b, !con(a,b));
  // Pre-build the pattern to match (intrinsic arg0, arg1, ...).
  dag IntrinsicPattern = BuildPatternI<!cast<Intrinsic>(Intr), Args>.ret;
}

//
// wmma.load.[a|b|c].sync.[row|col].m16n16k16[|.global|.shared].[f16|f32]
//

class WMMA_LOAD<WMMA_REGINFO Frag, string Layout, string Space, bit WithStride,
                DAGOperand SrcOp>
  : WMMA_INSTR<WMMA_NAME_LDST<"load", Frag, Layout, WithStride>.record,
                              [!con((ins SrcOp:$src),
                                    !if(WithStride, (ins Int32Regs:$ldm), (ins)))]>,
    Requires<Frag.Predicates> {
  // Load/store intrinsics are overloaded on pointer's address space.
  // To match the right intrinsic, we need to build AS-constrained PatFrag.
  // Operands is a dag equivalent in shape to Args, but using (ops node:$name, .....).
  dag PFOperands = !if(WithStride, (ops node:$src, node:$ldm), (ops node:$src));
  dag PFOperandsIntr = !if(WithStride, (Intr node:$src, node:$ldm), (Intr node:$src));
  // Build PatFrag that only matches particular address space.
  PatFrag IntrFrag = PatFrag<PFOperands,
                             PFOperandsIntr,
                             !cond(!eq(Space, ".shared"): AS_match.shared,
                                   !eq(Space, ".global"): AS_match.global,
                                   true: AS_match.generic)>;
  // Build AS-constrained pattern.
  let IntrinsicPattern = BuildPatternPF<IntrFrag, Args>.ret;

  let OutOperandList = Frag.Outs;
  let InOperandList = !con(Args, (ins MmaCode:$ptx));
  let AsmString = "wmma.load."
                  # Frag.frag
                  # ".sync"
                  # "${ptx:aligned}"
                  # "." # Layout
                  # "." # Frag.geom
                  # Space
                  # "." # Frag.ptx_elt_type # " \t"
                  # Frag.regstring
                  # ", [$src]"
                  # !if(WithStride, ", $ldm", "")
                  # ";";
}

//
// wmma.store.d.sync.[row|col].m16n16k16[|.global|.shared].[f16|f32]
//
class WMMA_STORE_D<WMMA_REGINFO Frag, string Layout, string Space,
                   bit WithStride, DAGOperand DstOp>
  : WMMA_INSTR<WMMA_NAME_LDST<"store", Frag, Layout, WithStride>.record,
               [!con((ins DstOp:$dst),
                     Frag.Ins,
                     !if(WithStride, (ins Int32Regs:$ldm), (ins)))]>,
    Requires<Frag.Predicates> {

  // Load/store intrinsics are overloaded on pointer's address space.
  // To match the right intrinsic, we need to build AS-constrained PatFrag.
  // Operands is a dag equivalent in shape to Args, but using (ops node:$name, .....).
  dag PFOperands = !con((ops node:$dst),
                        !dag(ops, !listsplat(node, !size(Frag.regs)), Frag.reg_names),
                        !if(WithStride, (ops node:$ldm), (ops)));
  // Build PatFrag that only matches particular address space.
  PatFrag IntrFrag = PatFrag<PFOperands,
                             !foreach(tmp, PFOperands, !subst(ops, Intr, tmp)),
                             !cond(!eq(Space, ".shared"): AS_match.shared,
                                   !eq(Space, ".global"): AS_match.global,
                                   true: AS_match.generic)>;
  // Build AS-constrained pattern.
  let IntrinsicPattern = BuildPatternPF<IntrFrag, Args>.ret;

  let InOperandList  = !con(Args, (ins MmaCode:$ptx));
  let OutOperandList = (outs);
  let AsmString = "wmma.store.d.sync"
                  # "${ptx:aligned}"
                  # "." # Layout
                  # "." # Frag.geom
                  # Space
                  # "." # Frag.ptx_elt_type
                  # " \t[$dst],"
                  # Frag.regstring
                  # !if(WithStride, ", $ldm", "")
                  # ";";
}

// Create all load/store variants
defset list<WMMA_INSTR> MMA_LDSTs  = {
  foreach layout = ["row", "col"] in {
    foreach stride = [false, true] in {
      foreach space = [".global", ".shared", ""] in {
        foreach addr = [imem, Int32Regs, Int64Regs, MEMri, MEMri64] in {
          foreach frag = NVVM_MMA_OPS.all_ld_ops in
            if NVVM_WMMA_LDST_SUPPORTED<frag, layout>.ret then
              def : WMMA_LOAD<WMMA_REGINFO<frag, "load">, layout, space, stride, addr>;
          foreach frag = NVVM_MMA_OPS.all_st_ops in
            if NVVM_WMMA_LDST_SUPPORTED<frag, layout>.ret then
              def : WMMA_STORE_D<WMMA_REGINFO<frag, "store">, layout, space, stride, addr>;
        } // addr
      } // space
    } // stride
  } // layout
} // defset

// B1 instruction variants need extra constraints.
class MMA_OP_PREDICATES<WMMA_REGINFO FragA, string b1op> {
  string Op = b1op;
  WMMA_REGINFO Frag = FragA;
  list<Predicate> ret = !listconcat(
    FragA.Predicates,
    !if(!eq(b1op, ".and.popc"), [hasSM<80>,hasPTX<71>],[])
  );
}
// WMMA.MMA
class WMMA_MMA<WMMA_REGINFO FragA, WMMA_REGINFO FragB,
               WMMA_REGINFO FragC, WMMA_REGINFO FragD,
               string ALayout, string BLayout, int Satfinite, string rnd, string b1op>
  : WMMA_INSTR<WMMA_NAME<ALayout, BLayout, Satfinite, rnd, b1op, FragA, FragB, FragC, FragD>.record,
                         [FragA.Ins, FragB.Ins, FragC.Ins]>,
    // Requires does not seem to have effect on Instruction w/o Patterns.
    // We set it here anyways and propagate to the Pat<> we construct below.
    Requires<MMA_OP_PREDICATES<FragA, b1op>.ret> {
  let OutOperandList = FragD.Outs;
  let InOperandList  = !con(Args, (ins MmaCode:$ptx));
  string TypeList = !cond(
    !eq(FragA.ptx_elt_type, "f16") : "." # FragD.ptx_elt_type
                                     # "." # FragC.ptx_elt_type,
    1: "." # FragD.ptx_elt_type
       # "." # FragA.ptx_elt_type
       # "." # FragB.ptx_elt_type
       # "." # FragC.ptx_elt_type,
  );
  let AsmString = "wmma.mma"
                  # b1op
                  # ".sync"
                  # "${ptx:aligned}"
                  # "." # ALayout
                  # "." # BLayout
                  # "." # FragA.geom
                  # !if(!ne(rnd, ""), !strconcat(".", rnd), "")
                  # TypeList
                  # !if(Satfinite, ".satfinite", "") # "\n\t\t"
                  # FragD.regstring # ",\n\t\t"
                  # FragA.regstring # ",\n\t\t"
                  # FragB.regstring # ",\n\t\t"
                  # FragC.regstring # ";";
}

let isConvergent = true in {
defset list<WMMA_INSTR> WMMAs  = {
  foreach layout_a = ["row", "col"] in {
    foreach layout_b = ["row", "col"] in {
      foreach satf = [0, 1] in {
        foreach rnd = ["", "rn", "rz", "rm", "rp"] in {
          foreach op = NVVM_MMA_OPS.all_wmma_ops in {
            foreach b1op = NVVM_MMA_B1OPS<op>.ret in {
              if NVVM_WMMA_SUPPORTED<op, layout_a, layout_b, satf, rnd>.ret then {
                def : WMMA_MMA<WMMA_REGINFO<op[0], "wmma.mma">,
                              WMMA_REGINFO<op[1], "wmma.mma">,
                              WMMA_REGINFO<op[2], "wmma.mma">,
                              WMMA_REGINFO<op[3], "wmma.mma">,
                              layout_a, layout_b, satf, rnd, b1op>;
              }
            } // b1op
          } // op
        } // rnd
      } // satf
    } // layout_b
  } // layout_a
} // defset
}

// MMA
class MMA<WMMA_REGINFO FragA, WMMA_REGINFO FragB,
               WMMA_REGINFO FragC, WMMA_REGINFO FragD,
               string ALayout, string BLayout, int Satfinite, string b1op>
  : WMMA_INSTR<MMA_NAME<ALayout, BLayout, Satfinite, b1op, FragA, FragB, FragC, FragD>.record,
                        [FragA.Ins, FragB.Ins, FragC.Ins]>,
    // Requires does not seem to have effect on Instruction w/o Patterns.
    // We set it here anyways and propagate to the Pat<> we construct below.
  Requires<MMA_OP_PREDICATES<FragA, b1op>.ret> {
  let OutOperandList = FragD.Outs;
  let InOperandList  = !con(Args, (ins MmaCode:$ptx));
  string TypeList = "." # FragD.ptx_elt_type
                    # "." # FragA.ptx_elt_type
                    # "." # FragB.ptx_elt_type
                    # "." # FragC.ptx_elt_type;
  let AsmString = "mma.sync.aligned."
                  # FragA.geom
                  # "." # ALayout
                  # "." # BLayout
                  # !if(Satfinite, ".satfinite", "")
                  # TypeList
                  # b1op # "\n\t\t"
                  # FragD.regstring # ",\n\t\t"
                  # FragA.regstring # ",\n\t\t"
                  # FragB.regstring # ",\n\t\t"
                  # FragC.regstring # ";";
}

let isConvergent = true in {
defset list<WMMA_INSTR> MMAs  = {
  foreach layout_a = ["row", "col"] in {
    foreach layout_b = ["row", "col"] in {
      foreach satf = [0, 1] in {
        foreach op = NVVM_MMA_OPS.all_mma_ops in {
          foreach b1op = NVVM_MMA_B1OPS<op>.ret in {
            if NVVM_MMA_SUPPORTED<op, layout_a, layout_b, satf>.ret then {
              def : MMA<WMMA_REGINFO<op[0], "mma">,
                        WMMA_REGINFO<op[1], "mma">,
                        WMMA_REGINFO<op[2], "mma">,
                        WMMA_REGINFO<op[3], "mma">,
                        layout_a, layout_b, satf, b1op>;
            }
          } // b1op
        } // op
      } // satf
    } // layout_b
  } // layout_a
} // defset
}

//
// ldmatrix.sync.aligned.m8n8[|.trans][|.shared].b16
//
class LDMATRIX<WMMA_REGINFO Frag, bit Transposed, string Space,
               DAGOperand SrcOp>
  : WMMA_INSTR<LDMATRIX_NAME<Frag, Transposed>.record, [(ins SrcOp:$src)]>,
    Requires<Frag.Predicates> {
  // Build PatFrag that only matches particular address space.
  PatFrag IntrFrag = PatFrag<(ops node:$src), (Intr node:$src),
                             !cond(!eq(Space, ".shared"): AS_match.shared,
                                   true: AS_match.generic)>;
  // Build AS-constrained pattern.
  let IntrinsicPattern = BuildPatternPF<IntrFrag, Args>.ret;

  let OutOperandList = Frag.Outs;
  let InOperandList = !con(Args, (ins MmaCode:$ptx));
  let AsmString = "ldmatrix.sync.aligned."
                  # Frag.geom
                  # "." # Frag.frag
                  # !if(Transposed, ".trans", "")
                  # Space
                  # "." # Frag.ptx_elt_type
                  # " " # Frag.regstring # ", [$src];";
}

// Create all ldmatrix variants
defset list<WMMA_INSTR> LDMATRIXs  = {
  foreach transposed = [false, true] in {
    foreach space = [".shared", ""] in {
      foreach addr = [imem, Int32Regs, Int64Regs, MEMri, MEMri64] in {
        foreach frag = NVVM_MMA_OPS.all_ldmatrix_ops in
          if NVVM_LDMATRIX_SUPPORTED<frag, transposed>.ret then
            def : LDMATRIX<WMMA_REGINFO<frag, "ldmatrix">, transposed, space,
                            addr>;
      } // addr
    } // space
  } // transposed
} // defset

// Constructing non-flat DAGs is still a pain. I can't !subst a dag node with a
// dag, so the ptx.version must be appended *after* foreach replaces 'ins' with
// the instruction record.
class MMA_PAT<WMMA_INSTR wi>
      : Pat<wi.IntrinsicPattern,
            !con(!foreach(tmp, wi.Args, !subst(ins, wi, tmp)),
                 (wi ptx.version))>,
        Requires<wi.Predicates>;

// Build intrinsic->instruction patterns for all MMA instructions.
foreach mma = !listconcat(MMAs, WMMAs, MMA_LDSTs, LDMATRIXs) in
  def : MMA_PAT<mma>;

multiclass MAPA<string suffix, Intrinsic Intr> {
  def _32: NVPTXInst<(outs Int32Regs:$d), (ins Int32Regs:$a, Int32Regs:$b),
              "mapa" # suffix # ".u32\t$d, $a, $b;",
              [(set i32:$d, (Intr i32:$a, i32:$b))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
  def _32i: NVPTXInst<(outs Int32Regs:$d), (ins Int32Regs:$a, i32imm:$b),
              "mapa" # suffix # ".u32\t$d, $a, $b;",
              [(set i32:$d, (Intr i32:$a, imm:$b))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
  def _64: NVPTXInst<(outs Int64Regs:$d), (ins Int64Regs:$a, Int32Regs:$b),
              "mapa" # suffix # ".u64\t$d, $a, $b;",
              [(set i64:$d, (Intr i64:$a, i32:$b))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
  def _64i: NVPTXInst<(outs Int64Regs:$d), (ins Int64Regs:$a, i32imm:$b),
              "mapa" # suffix # ".u64\t$d, $a, $b;",
              [(set i64:$d, (Intr i64:$a, imm:$b))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
}

defm mapa  : MAPA<"", int_nvvm_mapa>;
defm mapa_shared_cluster  : MAPA<".shared::cluster", int_nvvm_mapa_shared_cluster>;


multiclass GETCTARANK<string suffix, Intrinsic Intr> {
  def _32: NVPTXInst<(outs Int32Regs:$d), (ins Int32Regs:$a),
              "getctarank" # suffix # ".u32\t$d, $a;",
              [(set i32:$d, (Intr i32:$a))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
  def _64: NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
              "getctarank" # suffix # ".u64\t$d, $a;",
              [(set i32:$d, (Intr i64:$a))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;
}

defm getctarank  : GETCTARANK<"", int_nvvm_getctarank>;
defm getctarank_shared_cluster  : GETCTARANK<".shared::cluster", int_nvvm_getctarank_shared_cluster>;

def is_explicit_cluster: NVPTXInst<(outs Int1Regs:$d), (ins),
              "mov.pred\t$d, %is_explicit_cluster;",
              [(set i1:$d, (int_nvvm_is_explicit_cluster))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;

// setmaxnreg inc/dec intrinsics
let isConvergent = true in {
multiclass SET_MAXNREG<string Action, Intrinsic Intr> {
  def : NVPTXInst<(outs), (ins i32imm:$reg_count),
          "setmaxnreg." # Action # ".sync.aligned.u32 $reg_count;",
          [(Intr timm:$reg_count)]>,
    Requires<[hasSM90a, hasPTX<80>]>;
}

defm INT_SET_MAXNREG_INC : SET_MAXNREG<"inc", int_nvvm_setmaxnreg_inc_sync_aligned_u32>;
defm INT_SET_MAXNREG_DEC : SET_MAXNREG<"dec", int_nvvm_setmaxnreg_dec_sync_aligned_u32>;

} // isConvergent

//
// WGMMA fence instructions
//
let isConvergent = true in {
def INT_NVVM_WGMMA_FENCE_SYNC_ALIGNED : NVPTXInst<(outs), (ins), "wgmma.fence.sync.aligned;",
                             [(int_nvvm_wgmma_fence_sync_aligned)]>, Requires<[hasSM90a, hasPTX<80>]>;

def INT_NVVM_WGMMA_COMMIT_GROUP_SYNC_ALIGNED : NVPTXInst<(outs), (ins), "wgmma.commit_group.sync.aligned;",
                             [(int_nvvm_wgmma_commit_group_sync_aligned)]>, Requires<[hasSM90a, hasPTX<80>]>;

def INT_NVVM_WGMMA_WAIT_GROUP_SYNC_ALIGNED : NVPTXInst<(outs), (ins i64imm:$n), "wgmma.wait_group.sync.aligned \t$n;",
                             [(int_nvvm_wgmma_wait_group_sync_aligned timm:$n)]>, Requires<[hasSM90a, hasPTX<80>]>;
} // isConvergent = true

def GRIDDEPCONTROL_LAUNCH_DEPENDENTS :
      NVPTXInst<(outs), (ins),
                "griddepcontrol.launch_dependents;",
                [(int_nvvm_griddepcontrol_launch_dependents)]>,
                Requires<[hasSM<90>, hasPTX<78>]>;

def GRIDDEPCONTROL_WAIT :
      NVPTXInst<(outs), (ins),
                "griddepcontrol.wait;",
                [(int_nvvm_griddepcontrol_wait)]>,
                Requires<[hasSM<90>, hasPTX<78>]>;

def INT_EXIT : NVPTXInst<(outs), (ins), "exit;", [(int_nvvm_exit)]>;
