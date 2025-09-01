//===- NVPTXIntrinsics.td - PTX Intrinsics Instructions -------*- tblgen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def AS_match {
  code generic = [{
   return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_GENERIC;
  }];
  code shared = [{
   return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_SHARED;
  }];
  code shared_cluster = [{
   return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_SHARED_CLUSTER;
  }];
  code global = [{
   return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_GLOBAL;
  }];
  code const = [{
   return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_CONST;
  }];
  code param = [{
   return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_PARAM;
  }];
}


//===----------------------------------------------------------------------===//
// NVPTX Scope Constants
// These map to the Scope enum in NVPTX.h
//===----------------------------------------------------------------------===//

def Scope_thread : PatLeaf<(i32 0)>;      // Thread = 0
def Scope_cta : PatLeaf<(i32 1)>;         // Block = 1
def Scope_cluster : PatLeaf<(i32 2)>;     // Cluster = 2
def Scope_device : PatLeaf<(i32 3)>;      // Device = 3
def Scope_sys : PatLeaf<(i32 4)>;         // System = 4

//===----------------------------------------------------------------------===//
// NVPTX Address Space Constants
// These map to the AddressSpace enum in NVPTX.h
//===----------------------------------------------------------------------===//

def AddrSpace_gen : PatLeaf<(i32 0)>;        // Generic = 0
def AddrSpace_global : PatLeaf<(i32 1)>;         // Global = 1
def AddrSpace_shared : PatLeaf<(i32 3)>;         // Shared = 3
def AddrSpace_const : PatLeaf<(i32 4)>;          // Const = 4
def AddrSpace_local : PatLeaf<(i32 5)>;          // Local = 5
def AddrSpace_shared_cluster : PatLeaf<(i32 7)>;  // SharedCluster = 7
def AddrSpace_param : PatLeaf<(i32 101)>;        // Param = 101

//===----------------------------------------------------------------------===//
// NVPTX Ordering Constants
// These map to the Ordering enum in NVPTX.h
//===----------------------------------------------------------------------===//

def Ordering_not_atomic : PatLeaf<(i32 0)>;           // NotAtomic = 0
def Ordering_relaxed : PatLeaf<(i32 2)>;             // Relaxed = 1
def Ordering_acquire : PatLeaf<(i32 4)>;             // Acquire = 4
def Ordering_release : PatLeaf<(i32 5)>;             // Release = 5
def Ordering_acquire_release : PatLeaf<(i32 6)>;      // AcquireRelease = 6
def Ordering_sequentially_consistent : PatLeaf<(i32 7)>; // SequentiallyConsistent = 7
def Ordering_volatile : PatLeaf<(i32 8)>;            // Volatile = 8
def Ordering_relaxed_mmio : PatLeaf<(i32 9)>;         // RelaxedMMIO = 9


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
// E.g. RegNames<3, "r">.ret -> ["r0", "r1", "r2" ]
class RegSeq<int n, string prefix> {
  list<string> ret = !if(n, !listconcat(RegSeq<!sub(n, 1), prefix>.ret,
                                        [prefix # !sub(n, 1)]),
                            []);
}

//-----------------------------------
// Synchronization and shuffle functions
//-----------------------------------
let isConvergent = true in {
def INT_BARRIER0_POPC : NVPTXInst<(outs B32:$dst), (ins B32:$pred),
  !strconcat("{{ \n\t",
             ".reg .pred \t%p1; \n\t",
             "setp.ne.u32 \t%p1, $pred, 0; \n\t",
             "bar.red.popc.u32 \t$dst, 0, %p1; \n\t",
             "}}"),
      [(set i32:$dst, (int_nvvm_barrier0_popc i32:$pred))]>;
def INT_BARRIER0_AND : NVPTXInst<(outs B32:$dst), (ins B32:$pred),
  !strconcat("{{ \n\t",
             ".reg .pred \t%p1; \n\t",
             ".reg .pred \t%p2; \n\t",
             "setp.ne.u32 \t%p1, $pred, 0; \n\t",
             "bar.red.and.pred \t%p2, 0, %p1; \n\t",
             "selp.u32 \t$dst, 1, 0, %p2; \n\t",
             "}}"),
      [(set i32:$dst, (int_nvvm_barrier0_and i32:$pred))]>;
def INT_BARRIER0_OR : NVPTXInst<(outs B32:$dst), (ins B32:$pred),
  !strconcat("{{ \n\t",
             ".reg .pred \t%p1; \n\t",
             ".reg .pred \t%p2; \n\t",
             "setp.ne.u32 \t%p1, $pred, 0; \n\t",
             "bar.red.or.pred \t%p2, 0, %p1; \n\t",
             "selp.u32 \t$dst, 1, 0, %p2; \n\t",
             "}}"),
      [(set i32:$dst, (int_nvvm_barrier0_or i32:$pred))]>;

def INT_BAR_WARP_SYNC_I : BasicNVPTXInst<(outs), (ins i32imm:$i), "bar.warp.sync",
                             [(int_nvvm_bar_warp_sync imm:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;
def INT_BAR_WARP_SYNC_R : BasicNVPTXInst<(outs), (ins B32:$i), "bar.warp.sync",
                             [(int_nvvm_bar_warp_sync i32:$i)]>,
        Requires<[hasPTX<60>, hasSM<30>]>;

multiclass BARRIER1<string asmstr, Intrinsic intrinsic, list<Predicate> requires = []> {
  def _i : BasicNVPTXInst<(outs), (ins i32imm:$i), asmstr,
                          [(intrinsic imm:$i)]>,
           Requires<requires>;

  def _r : BasicNVPTXInst<(outs), (ins B32:$i), asmstr,
                          [(intrinsic i32:$i)]>,
           Requires<requires>;
}

multiclass BARRIER2<string asmstr, Intrinsic intrinsic, list<Predicate> requires = []> {
  def _rr : BasicNVPTXInst<(outs), (ins B32:$i, B32:$j), asmstr,
                          [(intrinsic i32:$i, i32:$j)]>,
            Requires<requires>;

  def _ri : BasicNVPTXInst<(outs), (ins B32:$i, i32imm:$j), asmstr,
                          [(intrinsic i32:$i, imm:$j)]>,
            Requires<requires>;

  def _ir : BasicNVPTXInst<(outs), (ins i32imm:$i, B32:$j), asmstr,
                          [(intrinsic imm:$i, i32:$j)]>,
            Requires<requires>;

  def _ii : BasicNVPTXInst<(outs), (ins i32imm:$i, i32imm:$j), asmstr,
                          [(intrinsic imm:$i, imm:$j)]>,
            Requires<requires>;
}

// Note the "bar.sync" variants could be renamed to the equivalent corresponding
// "barrier.*.aligned" variants. We use the older syntax for compatibility with
// older versions of the PTX ISA.
defm BARRIER_CTA_SYNC_ALIGNED_ALL : BARRIER1<"bar.sync", int_nvvm_barrier_cta_sync_aligned_all>;
defm BARRIER_CTA_SYNC_ALIGNED : BARRIER2<"bar.sync", int_nvvm_barrier_cta_sync_aligned_count>;
defm BARRIER_CTA_ARRIVE_ALIGNED : BARRIER2<"bar.arrive", int_nvvm_barrier_cta_arrive_aligned_count>;

defm BARRIER_CTA_SYNC_ALL : BARRIER1<"barrier.sync", int_nvvm_barrier_cta_sync_all, [hasPTX<60>]>;
defm BARRIER_CTA_SYNC : BARRIER2<"barrier.sync", int_nvvm_barrier_cta_sync_count, [hasPTX<60>]>;
defm BARRIER_CTA_ARRIVE : BARRIER2<"barrier.arrive", int_nvvm_barrier_cta_arrive_count, [hasPTX<60>]>;

class INT_BARRIER_CLUSTER<string variant, Intrinsic Intr,
                          list<Predicate> Preds = [hasPTX<78>, hasSM<90>]>:
        BasicNVPTXInst<(outs), (ins), "barrier.cluster."# variant, [(Intr)]>,
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

foreach sync = [false, true] in {
  foreach mode = ["up", "down", "bfly", "idx"] in {
    foreach regclass = ["i32", "f32"] in {
      foreach return_pred = [false, true] in {
        foreach offset_imm = [false, true] in {
          foreach mask_imm = [false, true] in {
            foreach threadmask_imm = !if(sync, [0, 1], [0]) in {
              defvar Intr = !cast<Intrinsic>("int_nvvm_shfl_"
                                # !if(sync, "sync_", "")
                                # mode
                                # "_" # regclass
                                # !if(return_pred, "p", ""));
              defvar InOperandList = !con(
                (ins B32:$src),
                !dag(ins, !if(offset_imm, [i32imm], [B32]), ["offset"]),
                !dag(ins, !if(mask_imm, [i32imm], [B32]), ["mask"]),
                !if(sync,
                    !dag(ins, !if(threadmask_imm, [i32imm], [B32]), ["threadmask"]),
                    (ins)));
              defvar Pattern = !con(
                  (set B32:$dst),
                  !if(return_pred, (set B1:$pred), (set)),
                  (set !con(
                    !if(sync,
                      !dag(Intr, !if(threadmask_imm, [imm], [B32]), ["threadmask"]),
                      (Intr)),
                    (Intr B32:$src),
                    !dag(Intr, !if(offset_imm, [imm], [B32]), ["offset"]),
                    !dag(Intr, !if(mask_imm, [imm], [B32]), ["mask"]))));

              def : BasicNVPTXInst<
                      !if(return_pred, (outs B32:$dst, B1:$pred),
                                       (outs B32:$dst)),
                      InOperandList,
                      "shfl." # !if(sync, "sync.", "") # mode # ".b32",
                      [Pattern]>,
                    Requires<!if(sync, [hasSM<30>, hasPTX<60>], [hasSM<30>, hasSHFL])>;
            }
          }
        }
      }
    }
  }
}

// vote.{all,any,uni,ballot}
let Predicates = [hasPTX<60>, hasSM<30>] in {
  multiclass VOTE<string mode, RegTyInfo t, Intrinsic op> {
    def : BasicNVPTXInst<(outs t.RC:$dest), (ins B1:$pred),
                "vote." # mode # "." # t.PtxType,
                [(set t.Ty:$dest, (op i1:$pred))]>;
  }

  defm VOTE_ALL : VOTE<"all", I1RT, int_nvvm_vote_all>;
  defm VOTE_ANY : VOTE<"any", I1RT, int_nvvm_vote_any>;
  defm VOTE_UNI : VOTE<"uni", I1RT, int_nvvm_vote_uni>;
  defm VOTE_BALLOT : VOTE<"ballot", I32RT, int_nvvm_vote_ballot>;

  // vote.sync.{all,any,uni,ballot}
  multiclass VOTE_SYNC<string mode, RegTyInfo t, Intrinsic op> {
    def i : BasicNVPTXInst<(outs t.RC:$dest), (ins B1:$pred, i32imm:$mask),
                "vote.sync." # mode # "." # t.PtxType,
                [(set t.Ty:$dest, (op imm:$mask, i1:$pred))]>;
    def r : BasicNVPTXInst<(outs t.RC:$dest), (ins B1:$pred, B32:$mask),
                "vote.sync." # mode # "." # t.PtxType,
                [(set t.Ty:$dest, (op i32:$mask, i1:$pred))]>;
  }

  defm VOTE_SYNC_ALL : VOTE_SYNC<"all", I1RT, int_nvvm_vote_all_sync>;
  defm VOTE_SYNC_ANY : VOTE_SYNC<"any", I1RT, int_nvvm_vote_any_sync>;
  defm VOTE_SYNC_UNI : VOTE_SYNC<"uni", I1RT, int_nvvm_vote_uni_sync>;
  defm VOTE_SYNC_BALLOT : VOTE_SYNC<"ballot", I32RT, int_nvvm_vote_ballot_sync>;
}
// elect.sync
let Predicates = [hasPTX<80>, hasSM<90>] in {
def INT_ELECT_SYNC_I : BasicNVPTXInst<(outs B32:$dest, B1:$pred), (ins i32imm:$mask),
            "elect.sync",
            [(set i32:$dest, i1:$pred, (int_nvvm_elect_sync imm:$mask))]>;
def INT_ELECT_SYNC_R : BasicNVPTXInst<(outs B32:$dest, B1:$pred), (ins B32:$mask),
            "elect.sync",
            [(set i32:$dest, i1:$pred, (int_nvvm_elect_sync i32:$mask))]>;
}

let Predicates = [hasPTX<60>, hasSM<70>] in {
  multiclass MATCH_ANY_SYNC<Intrinsic op, RegTyInfo t> {
    def ii : BasicNVPTXInst<(outs B32:$dest), (ins t.Imm:$value, i32imm:$mask),
                "match.any.sync." # t.PtxType,
                [(set i32:$dest, (op imm:$mask, imm:$value))]>;
    def ir : BasicNVPTXInst<(outs B32:$dest), (ins t.Imm:$value, B32:$mask),
                "match.any.sync." # t.PtxType,
                [(set i32:$dest, (op i32:$mask, imm:$value))]>;
    def ri : BasicNVPTXInst<(outs B32:$dest), (ins t.RC:$value, i32imm:$mask),
                "match.any.sync." # t.PtxType,
                [(set i32:$dest, (op imm:$mask, t.Ty:$value))]>;
    def rr : BasicNVPTXInst<(outs B32:$dest), (ins t.RC:$value, B32:$mask),
                "match.any.sync." # t.PtxType,
                [(set i32:$dest, (op i32:$mask, t.Ty:$value))]>;
  }

  defm MATCH_ANY_SYNC_32 : MATCH_ANY_SYNC<int_nvvm_match_any_sync_i32, I32RT>;
  defm MATCH_ANY_SYNC_64 : MATCH_ANY_SYNC<int_nvvm_match_any_sync_i64, I64RT>;

  multiclass MATCH_ALLP_SYNC<RegTyInfo t, Intrinsic op> {
    def ii : BasicNVPTXInst<(outs B32:$dest, B1:$pred),
                      (ins t.Imm:$value, i32imm:$mask),
                "match.all.sync." # t.PtxType,
                [(set i32:$dest, i1:$pred, (op imm:$mask, imm:$value))]>;
    def ir : BasicNVPTXInst<(outs B32:$dest, B1:$pred),
                      (ins t.Imm:$value, B32:$mask),
                "match.all.sync." # t.PtxType,
                [(set i32:$dest, i1:$pred, (op i32:$mask, imm:$value))]>;
    def ri : BasicNVPTXInst<(outs B32:$dest, B1:$pred),
                      (ins t.RC:$value, i32imm:$mask),
                "match.all.sync." # t.PtxType,
                [(set i32:$dest, i1:$pred, (op imm:$mask, t.Ty:$value))]>;
    def rr : BasicNVPTXInst<(outs B32:$dest, B1:$pred),
                      (ins t.RC:$value, B32:$mask),
                "match.all.sync." # t.PtxType,
                [(set i32:$dest, i1:$pred, (op i32:$mask, t.Ty:$value))]>;
  }
  defm MATCH_ALLP_SYNC_32 : MATCH_ALLP_SYNC<I32RT, int_nvvm_match_all_sync_i32p>;
  defm MATCH_ALLP_SYNC_64 : MATCH_ALLP_SYNC<I64RT, int_nvvm_match_all_sync_i64p>;
}

// activemask.b32
def ACTIVEMASK : BasicNVPTXInst<(outs B32:$dest), (ins),
                    "activemask.b32",
                    [(set i32:$dest, (int_nvvm_activemask))]>,
                 Requires<[hasPTX<62>, hasSM<30>]>;

multiclass REDUX_SYNC<string BinOp, string PTXType, Intrinsic Intrin> {
  def : BasicNVPTXInst<(outs B32:$dst), (ins B32:$src, B32:$mask),
          "redux.sync." # BinOp # "." # PTXType,
          [(set i32:$dst, (Intrin i32:$src, B32:$mask))]>,
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

multiclass REDUX_SYNC_F<string BinOp, string abs, string NaN> {
  defvar intr_name = "int_nvvm_redux_sync_f" # BinOp # !subst(".", "_", abs) # !subst(".", "_", NaN);

  def : BasicNVPTXInst<(outs B32:$dst),
                  (ins B32:$src, B32:$mask),
                  "redux.sync." # BinOp # abs # NaN # ".f32",
                  [(set f32:$dst, (!cast<Intrinsic>(intr_name) f32:$src, B32:$mask))]>,
                  Requires<[hasPTX<86>, hasSM100a]>; 
}

defm REDUX_SYNC_FMIN : REDUX_SYNC_F<"min", "", "">;
defm REDUX_SYNC_FMIN_ABS : REDUX_SYNC_F<"min", ".abs", "">;
defm REDUX_SYNC_FMIN_NAN: REDUX_SYNC_F<"min", "", ".NaN">;
defm REDUX_SYNC_FMIN_ABS_NAN: REDUX_SYNC_F<"min", ".abs", ".NaN">;
defm REDUX_SYNC_FMAX : REDUX_SYNC_F<"max", "", "">;
defm REDUX_SYNC_FMAX_ABS : REDUX_SYNC_F<"max", ".abs", "">;
defm REDUX_SYNC_FMAX_NAN: REDUX_SYNC_F<"max", "", ".NaN">;
defm REDUX_SYNC_FMAX_ABS_NAN: REDUX_SYNC_F<"max", ".abs", ".NaN">;

} // isConvergent = true

//-----------------------------------
// Explicit Memory Fence Functions
//-----------------------------------
class NullaryInst<string StrOp, Intrinsic IntOP> :
              BasicNVPTXInst<(outs), (ins), StrOp, [(IntOP)]>;

def INT_MEMBAR_CTA : NullaryInst<"membar.cta", int_nvvm_membar_cta>;
def INT_MEMBAR_GL  : NullaryInst<"membar.gl",  int_nvvm_membar_gl>;
def INT_MEMBAR_SYS : NullaryInst<"membar.sys", int_nvvm_membar_sys>;

def INT_FENCE_SC_CLUSTER:
       NullaryInst<"fence.sc.cluster", int_nvvm_fence_sc_cluster>,
       Requires<[hasPTX<78>, hasSM<90>]>;

// Proxy fence (uni-directional)
class FENCE_PROXY_TENSORMAP_GENERIC_RELEASE<string Scope, Intrinsic Intr> :
        NullaryInst<"fence.proxy.tensormap::generic.release." # Scope, Intr>,
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
        NVPTXInst<(outs), (ins B64:$addr),
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
  def "" : BasicNVPTXInst<(outs), (ins ADDR:$addr),
            "cp.async.mbarrier.arrive" # NoInc # AddrSpace # ".b64",
            [(Intrin addr:$addr)]>,
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
  def "" : NVPTXInst<(outs), (ins ADDR:$dst, ADDR:$src),
            "cp.async." # cc # ".shared.global" # " [$dst], [$src], " # cpsize # ";",
            [(Intrin addr:$dst, addr:$src)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;

  // Variant with src_size parameter
  def _s : NVPTXInst<(outs), (ins ADDR:$dst, ADDR:$src, B32:$src_size),
             "cp.async." # cc # ".shared.global" # " [$dst], [$src], " # cpsize # ", $src_size;",
             [(IntrinS addr:$dst, addr:$src, i32:$src_size)]>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  def _si: NVPTXInst<(outs), (ins ADDR:$dst, ADDR:$src, i32imm:$src_size),
             "cp.async." # cc # ".shared.global" # " [$dst], [$src], " # cpsize # ", $src_size;",
             [(IntrinS addr:$dst, addr:$src, imm:$src_size)]>,
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

let Predicates = [hasPTX<70>, hasSM<80>] in {
  def CP_ASYNC_COMMIT_GROUP :
    NullaryInst<"cp.async.commit_group", int_nvvm_cp_async_commit_group>;

  def CP_ASYNC_WAIT_GROUP :
    BasicNVPTXInst<(outs), (ins i32imm:$n), "cp.async.wait_group",
    [(int_nvvm_cp_async_wait_group timm:$n)]>;

  def CP_ASYNC_WAIT_ALL :
    NullaryInst<"cp.async.wait_all", int_nvvm_cp_async_wait_all>;
}

let Predicates = [hasPTX<80>, hasSM<90>] in {
  // cp.async.bulk variants of the commit/wait group
  def CP_ASYNC_BULK_COMMIT_GROUP :
    NullaryInst<"cp.async.bulk.commit_group", int_nvvm_cp_async_bulk_commit_group>;

  def CP_ASYNC_BULK_WAIT_GROUP :
    BasicNVPTXInst<(outs), (ins i32imm:$n), "cp.async.bulk.wait_group",
    [(int_nvvm_cp_async_bulk_wait_group timm:$n)]>;

  def CP_ASYNC_BULK_WAIT_GROUP_READ :
    BasicNVPTXInst<(outs), (ins i32imm:$n), "cp.async.bulk.wait_group.read",
    [(int_nvvm_cp_async_bulk_wait_group_read timm:$n)]>;
}

//------------------------------
// TMA Async Bulk Copy Functions
//------------------------------

class CpAsyncBulkStr<bit mc, bit ch, bit mask = 0> {
  // Shared to Global memory
  string S2G = "cp.async.bulk.global.shared::cta.bulk_group"
               # !if(ch, ".L2::cache_hint", "")
               # !if(mask, ".cp_mask", "");

  // Global to Shared cluster memory
  string G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
               # !if(mc, ".multicast::cluster", "")
               # !if(ch, ".L2::cache_hint", "");

  // Shared CTA to Cluster memory
  string C2C = "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes";
}

multiclass CP_ASYNC_BULK_S2G_INTR<bit has_ch> {
  def "" : NVPTXInst<(outs), (ins ADDR:$dst, ADDR:$src, B32:$size, B64:$ch),
      !if(has_ch,
          CpAsyncBulkStr<0, 1>.S2G # " [$dst], [$src], $size, $ch;",
          CpAsyncBulkStr<0, 0>.S2G # " [$dst], [$src], $size;"),
      [(int_nvvm_cp_async_bulk_shared_cta_to_global addr:$dst, addr:$src, i32:$size, i64:$ch, !if(has_ch, -1, 0))]>,
      Requires<[hasPTX<80>, hasSM<90>]>;

  def _BM : NVPTXInst<(outs), (ins ADDR:$dst, ADDR:$src, B32:$size, B64:$ch, B16:$mask),
      !if(has_ch,
          CpAsyncBulkStr<0, 1, 1>.S2G # " [$dst], [$src], $size, $ch, $mask;",
          CpAsyncBulkStr<0, 0, 1>.S2G # " [$dst], [$src], $size, $mask;"),
      [(int_nvvm_cp_async_bulk_shared_cta_to_global_bytemask addr:$dst, addr:$src, i32:$size, i64:$ch, !if(has_ch, -1, 0), i16:$mask)]>,
      Requires<[hasPTX<86>, hasSM<100>]>;
}
defm CP_ASYNC_BULK_S2G    : CP_ASYNC_BULK_S2G_INTR<has_ch = 0>;
defm CP_ASYNC_BULK_S2G_CH : CP_ASYNC_BULK_S2G_INTR<has_ch = 1>;

multiclass CP_ASYNC_BULK_G2S_INTR<bit has_ch> {
  defvar Intr = int_nvvm_cp_async_bulk_global_to_shared_cluster;

  def "" : NVPTXInst<(outs),
      (ins ADDR:$dst, ADDR:$mbar, ADDR:$src,
           B32:$size, B16:$mask, B64:$ch),
      !if(has_ch,
          CpAsyncBulkStr<0, 1>.G2S # " [$dst], [$src], $size, [$mbar], $ch;",
          CpAsyncBulkStr<0, 0>.G2S # " [$dst], [$src], $size, [$mbar];"),
      [(Intr addr:$dst, addr:$mbar, addr:$src, i32:$size, i16:$mask, i64:$ch, 0, !if(has_ch, -1, 0))]>,
      Requires<[hasPTX<80>, hasSM<90>]>;

  def _MC : NVPTXInst<(outs),
      (ins ADDR:$dst, ADDR:$mbar, ADDR:$src,
           B32:$size, B16:$mask, B64:$ch),
      !if(has_ch,
          CpAsyncBulkStr<1, 1>.G2S # " [$dst], [$src], $size, [$mbar], $mask, $ch;",
          CpAsyncBulkStr<1, 0>.G2S # " [$dst], [$src], $size, [$mbar], $mask;"),
      [(Intr addr:$dst, addr:$mbar, addr:$src, i32:$size, i16:$mask, i64:$ch, -1, !if(has_ch, -1, 0))]>,
      Requires<[hasPTX<80>, hasSM<90>]>;
}
defm CP_ASYNC_BULK_G2S    : CP_ASYNC_BULK_G2S_INTR<has_ch = 0>;
defm CP_ASYNC_BULK_G2S_CH : CP_ASYNC_BULK_G2S_INTR<has_ch = 1>;

def CP_ASYNC_BULK_CTA_TO_CLUSTER : NVPTXInst<(outs),
  (ins ADDR:$dst, ADDR:$mbar, ADDR:$src, B32:$size),
  CpAsyncBulkStr<0, 0>.C2C # " [$dst], [$src], $size, [$mbar];",
  [(int_nvvm_cp_async_bulk_shared_cta_to_cluster addr:$dst, addr:$mbar, addr:$src, i32:$size)]>,
  Requires<[hasPTX<80>, hasSM<90>]>;

multiclass CP_ASYNC_BULK_PREFETCH_INTR<bit has_ch> {
  def "" : NVPTXInst<(outs), (ins ADDR:$src, B32:$size, B64:$ch),
      !if(has_ch,
          "cp.async.bulk.prefetch.L2.global.L2::cache_hint" # " [$src], $size, $ch;",
          "cp.async.bulk.prefetch.L2.global" # " [$src], $size;"),
      [(int_nvvm_cp_async_bulk_prefetch_L2 addr:$src, i32:$size, i64:$ch, !if(has_ch, -1, 0))]>,
      Requires<[hasPTX<80>, hasSM<90>]>;
}
defm CP_ASYNC_BULK_PREFETCH    : CP_ASYNC_BULK_PREFETCH_INTR<has_ch = 0>;
defm CP_ASYNC_BULK_PREFETCH_CH : CP_ASYNC_BULK_PREFETCH_INTR<has_ch = 1>;

//-------------------------------------
// TMA Async Bulk Tensor Copy Functions
//-------------------------------------

class TMA_DIMS_UTIL<int dim, string mode = ""> {
  // For example, when 'dim' is 3, this generates:
  // an ins_dag:    B32:$d0, B32:$d1, B32:$d2
  // with base_str: $d0, $d1, $d2
  dag ins_dag = !dag(ins, !listsplat(B32, dim), !foreach(i, !range(dim), "d" # i));
  string base_str = !interleave(!foreach(i, !range(dim), "$d" # i), ", ");

  // Tile::Gather4/scatter4 actually operate on a 2D tensor,
  // though they take 5 co-ordinates.
  //
  // The scatter-gather happens over 4 rows with a fixed
  // column-index. The first co-ordinate represents the
  // col-index followed by four row-indices.
  int num_dims = !cond(
                   !eq(mode, "tile_scatter4") : 2,
                   !eq(mode, "tile_gather4")  : 2,
                   true : dim); // for all other modes
}

class TMA_IM2COL_UTIL<int dim, string mode> {
  // For im2col_w/w_128 modes, number of offsets is always 2.
  // For im2col mode, offsets is (dim - 2).
  // For non-im2col modes (i.e. tile) there are no offsets.
  int offsets = !cond(
                  !eq(mode, "im2col") : !sub(dim, 2),
                  !eq(mode, "im2col_w") : 2,
                  !eq(mode, "im2col_w_128") : 2,
                  true : 0); // for all other modes

  dag ins_dag = !if(!gt(offsets, 0),
    !dag(ins, !listsplat(B16, offsets), !foreach(i, !range(offsets), "im2col" # i)),
    (ins));
  string base_str = !interleave(!foreach(i, !range(offsets), "$im2col" # i), ", ");
}

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

def CTAGroupFlags : Operand<i32> {
  let PrintMethod = "printCTAGroup";
}

multiclass CP_ASYNC_BULK_TENSOR_G2S_INTR<int dim, bit is_shared32, string mode> {
  defvar dims_dag = TMA_DIMS_UTIL<dim>.ins_dag;
  defvar dims_str = TMA_DIMS_UTIL<dim>.base_str;
  defvar asm_str_default = "$cg [$dst], [$tmap, {{" # dims_str # "}}], [$mbar]";
  defvar rc = !if(is_shared32, B32, B64);

  defvar num_im2col = !if(!ge(dim, 3), !add(dim, -2), 0);
  defvar im2col_dag = !if(!eq(mode, "im2col"),
    !dag(ins, !listsplat(B16, num_im2col), !foreach(i, !range(num_im2col), "im2col" # i)),
    (ins));
  defvar im2col_str = !interleave(!foreach(i, !range(num_im2col), "$im2col" # i), ", ");
  defvar im2col_asm_str = ", {{" # im2col_str # "}}";

  defvar asm_str = !if(!eq(mode, "im2col"),
    !strconcat(asm_str_default, im2col_asm_str), asm_str_default);

  def "" : NVPTXInst<(outs),
            !con((ins rc:$dst, rc:$mbar, B64:$tmap), dims_dag, im2col_dag, (ins CTAGroupFlags:$cg)),
            !strconcat(G2S_STRINGS<dim, mode, 0, 0>.inst_name, asm_str, ";")>,
            Requires<[hasPTX<80>, hasSM<90>]>;
  def _MC : NVPTXInst<(outs),
                  !con((ins rc:$dst, rc:$mbar, B64:$tmap), dims_dag, im2col_dag,
                       (ins B16:$mc, CTAGroupFlags:$cg)),
                  !strconcat(G2S_STRINGS<dim, mode, 1, 0>.inst_name, asm_str, ", $mc;")>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
  def _CH : NVPTXInst<(outs),
                  !con((ins rc:$dst, rc:$mbar, B64:$tmap), dims_dag, im2col_dag,
                       (ins B64:$ch, CTAGroupFlags:$cg)),
                  !strconcat(G2S_STRINGS<dim, mode, 0, 1>.inst_name, asm_str, ", $ch;")>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
  def _MC_CH : NVPTXInst<(outs),
                     !con((ins rc:$dst, rc:$mbar, B64:$tmap), dims_dag, im2col_dag,
                          (ins B16:$mc, B64:$ch, CTAGroupFlags:$cg)),
                     !strconcat(G2S_STRINGS<dim, mode, 1, 1>.inst_name, asm_str, ", $mc, $ch;")>,
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

multiclass TMA_TENSOR_G2S_INTR<int dim, string mode, list<Predicate> pred = []> {
  defvar dims_dag = TMA_DIMS_UTIL<dim>.ins_dag;
  defvar dims_str = TMA_DIMS_UTIL<dim>.base_str;
  defvar asm_str_base = "$cg [$dst], [$tmap, {{" # dims_str # "}}], [$mbar]";

  defvar im2col_dag = TMA_IM2COL_UTIL<dim, mode>.ins_dag;
  defvar im2col_str = TMA_IM2COL_UTIL<dim, mode>.base_str;
  defvar asm_str = !if(!empty(im2col_str),
                       asm_str_base,
                       asm_str_base # ", {{" # im2col_str # "}}");

  defvar dim_val = TMA_DIMS_UTIL<dim, mode>.num_dims;
  defvar inst_name = "cp.async.bulk.tensor"
                     # "." # dim_val # "d"
                     # "." # "shared::cluster.global"
                     # "." # !subst("_", "::", mode)
                     # "." # "mbarrier::complete_tx::bytes";
  defvar intr = !cast<Intrinsic>(
                  "int_nvvm_cp_async_bulk_tensor_g2s_" # mode # "_" # dim_val # "d");

  defvar ins_dag = !con(
                     (ins ADDR:$dst, ADDR:$mbar, B64:$tmap),
                     dims_dag, im2col_dag,
                     (ins B16:$mc, B64:$ch, CTAGroupFlags:$cg));

  defvar intr_dag_base = !con(
                         (intr addr:$dst, addr:$mbar, B64:$tmap),
                         !setdagop(dims_dag, intr),
                         !setdagop(im2col_dag, intr),
                         (intr B16:$mc, B64:$ch));
  defvar intr_dag_no_hints   = !con(intr_dag_base, (intr 0,  0,  timm:$cg));
  defvar intr_dag_with_mc    = !con(intr_dag_base, (intr -1, 0,  timm:$cg));
  defvar intr_dag_with_ch    = !con(intr_dag_base, (intr 0, -1,  timm:$cg));
  defvar intr_dag_with_mc_ch = !con(intr_dag_base, (intr -1, -1, timm:$cg));

  def "" : NVPTXInst<(outs), ins_dag,
             inst_name # asm_str # ";",
             [intr_dag_no_hints]>,
             Requires<pred>;
  def _MC : NVPTXInst<(outs), ins_dag,
              inst_name # ".multicast::cluster" # asm_str # ", $mc;",
              [intr_dag_with_mc]>,
              Requires<pred>;
  def _CH : NVPTXInst<(outs), ins_dag,
              inst_name # ".L2::cache_hint" # asm_str # ", $ch;",
              [intr_dag_with_ch]>,
              Requires<pred>;
  def _MC_CH : NVPTXInst<(outs), ins_dag,
                 inst_name # ".multicast::cluster.L2::cache_hint" # asm_str # ", $mc, $ch;",
                 [intr_dag_with_mc_ch]>,
                 Requires<pred>;
}
foreach dim = 3...5 in {
  foreach mode = ["im2col_w", "im2col_w_128"] in {
    defm TMA_G2S_ # !toupper(mode) # "_" # dim # "D"
      : TMA_TENSOR_G2S_INTR<dim, mode, [hasTMACTAGroupSupport]>;
  }
}
defm TMA_G2S_TILE_GATHER4_2D : TMA_TENSOR_G2S_INTR<5, "tile_gather4",
                               [hasTMACTAGroupSupport]>;

multiclass TMA_TENSOR_G2S_CTA_INTR<int dim, string mode, list<Predicate> pred = []> {
  defvar dims_dag = TMA_DIMS_UTIL<dim>.ins_dag;
  defvar dims_str = TMA_DIMS_UTIL<dim>.base_str;
  defvar asm_str_base = " [$dst], [$tmap, {{" # dims_str # "}}], [$mbar]";

  defvar im2col_dag = TMA_IM2COL_UTIL<dim, mode>.ins_dag;
  defvar im2col_str = TMA_IM2COL_UTIL<dim, mode>.base_str;
  defvar asm_str = !if(!empty(im2col_str),
                       asm_str_base,
                       asm_str_base # ", {{" # im2col_str # "}}");

  defvar ins_dag = !con(
                     (ins ADDR:$dst, ADDR:$mbar, B64:$tmap),
                     dims_dag, im2col_dag,
                     (ins B64:$ch));

  defvar dim_val = TMA_DIMS_UTIL<dim, mode>.num_dims;
  defvar intr = !cast<Intrinsic>(
                  "int_nvvm_cp_async_bulk_tensor_g2s_cta_" # mode # "_" # dim_val # "d");
  defvar intr_dag = !con(
                      (intr addr:$dst, addr:$mbar, B64:$tmap),
                      !setdagop(dims_dag, intr),
                      !setdagop(im2col_dag, intr),
                      (intr B64:$ch, 0));
  defvar intr_dag_with_ch = !con(
                              (intr addr:$dst, addr:$mbar, B64:$tmap),
                              !setdagop(dims_dag, intr),
                              !setdagop(im2col_dag, intr),
                              (intr B64:$ch, -1));
  defvar inst_name = "cp.async.bulk.tensor"
                     # "." # dim_val # "d"
                     # "." # "shared::cta.global"
                     # "." # !subst("_", "::", mode)
                     # "." # "mbarrier::complete_tx::bytes";

  def "" : NVPTXInst<(outs), ins_dag,
             inst_name # asm_str # ";",
             [intr_dag]>,
             Requires<pred>;
  def _CH : NVPTXInst<(outs), ins_dag,
              inst_name # ".L2::cache_hint" # asm_str # ", $ch;",
              [intr_dag_with_ch]>,
              Requires<pred>;
}
foreach dim = 1...5 in {
  defm TMA_G2S_CTA_TILE_ # dim # "D"
    : TMA_TENSOR_G2S_CTA_INTR<dim, "tile", [hasPTX<86>, hasSM<90>]>;
}
foreach dim = 3...5 in {
  defm TMA_G2S_CTA_IM2COL_ # dim # "D"
    : TMA_TENSOR_G2S_CTA_INTR<dim, "im2col", [hasPTX<86>, hasSM<90>]>;

  defm TMA_G2S_CTA_IM2COL_W_ # dim # "D"
    : TMA_TENSOR_G2S_CTA_INTR<dim, "im2col_w", [hasPTX<86>, hasSM<100>]>;

  defm TMA_G2S_CTA_IM2COL_W_128_ # dim # "D"
    : TMA_TENSOR_G2S_CTA_INTR<dim, "im2col_w_128", [hasTMACTAGroupSupport]>;
}
defm TMA_G2S_CTA_TILE_GATHER4_2D : TMA_TENSOR_G2S_CTA_INTR<5, "tile_gather4",
                                   [hasPTX<86>, hasSM<100>]>;

multiclass TMA_TENSOR_S2G_INTR<int dim, string mode,
                               list<Predicate> pred = [hasPTX<80>, hasSM<90>]> {
  defvar dims_dag = TMA_DIMS_UTIL<dim>.ins_dag;
  defvar dims_str = TMA_DIMS_UTIL<dim>.base_str;
  defvar asm_str = " [$tmap, {{" # dims_str # "}}], [$src]";

  defvar dim_val = TMA_DIMS_UTIL<dim, mode>.num_dims;
  defvar intr = !cast<Intrinsic>(
                  "int_nvvm_cp_async_bulk_tensor_s2g_" # mode # "_" # dim_val # "d");

  defvar intr_dag = !con((intr addr:$src, B64:$tmap),
                         !setdagop(dims_dag, intr),
                         (intr B64:$ch, 0));
  defvar intr_dag_with_ch = !con((intr addr:$src, B64:$tmap),
                                 !setdagop(dims_dag, intr),
                                 (intr B64:$ch, -1));

  // Fix-up the asm_str when it is im2col/scatter4.
  defvar mode_asm_str = !cond(
                          !eq(mode, "im2col") : "im2col_no_offs",
                          !eq(mode, "tile_scatter4") : "tile::scatter4",
                          true : mode);
  defvar prefix = "cp.async.bulk.tensor"
                  # "." # dim_val # "d"
                  # ".global.shared::cta"
                  # "." # mode_asm_str
                  # ".bulk_group";

  def "" : NVPTXInst<(outs),
             !con((ins ADDR:$src, B64:$tmap), dims_dag, (ins B64:$ch)),
             prefix # asm_str # ";",
             [intr_dag]>,
             Requires<pred>;
  def _CH : NVPTXInst<(outs),
              !con((ins ADDR:$src, B64:$tmap), dims_dag, (ins B64:$ch)),
              prefix # ".L2::cache_hint" # asm_str # ", $ch;",
              [intr_dag_with_ch]>,
              Requires<pred>;
}
foreach dim = 1...5 in {
  foreach mode = !if(!ge(dim, 3), ["tile", "im2col"], ["tile"]) in {
    defvar suffix = !toupper(mode) # "_" # dim # "D";
    defm TMA_TENSOR_S2G_ # suffix : TMA_TENSOR_S2G_INTR<dim, mode>;
  }
}
defm TMA_S2G_TILE_SCATTER4_2D : TMA_TENSOR_S2G_INTR<5, "tile_scatter4",
                                [hasTMACTAGroupSupport]>;

def TMAReductionFlags : Operand<i32> {
  let PrintMethod = "printTmaReductionMode";
}

// TMA Copy from Shared to Global memory with Reduction
multiclass CP_ASYNC_BULK_TENSOR_REDUCE_INTR<int dim, bit shared32, string mode> {
  defvar dims_dag = TMA_DIMS_UTIL<dim>.ins_dag;
  defvar dims_str = TMA_DIMS_UTIL<dim>.base_str;
  defvar asm_str = " [$tmap, {{" # dims_str # "}}], [$src]";
  defvar rc = !if(shared32, B32, B64);

  // For im2col mode, the actual asm_str is "im2col_no_offs"
  defvar mode_asm_str = !if(!eq(mode, "im2col"),
                            "im2col_no_offs", mode);
  defvar prefix = "cp.reduce.async.bulk.tensor" # "." # dim # "d" # ".global.shared::cta";
  defvar suffix = "." # mode_asm_str # ".bulk_group";

  def "" : NVPTXInst<(outs),
            !con((ins rc:$src, B64:$tmap), dims_dag, (ins TMAReductionFlags:$red_op)),
            !strconcat(prefix, "${red_op}", suffix, asm_str, ";")>,
            Requires<[hasPTX<80>, hasSM<90>]>;
  def _CH : NVPTXInst<(outs),
                  !con((ins rc:$src, B64:$tmap), dims_dag, (ins B64:$ch, TMAReductionFlags:$red_op)),
                  !strconcat(prefix, "${red_op}", suffix, ".L2::cache_hint", asm_str, ", $ch;")>,
                  Requires<[hasPTX<80>, hasSM<90>]>;
}

foreach dim = [1, 2, 3, 4, 5] in {
  foreach shared32 = [true, false] in {
    foreach mode = !if(!ge(dim, 3), ["tile", "im2col"], ["tile"]) in {
      defvar suffix = dim # "D"
                      # !if(shared32, "_SHARED32", "")
                      # "_" # !toupper(mode);
      defm CP_ASYNC_BULK_TENSOR_RED_ # suffix :
        CP_ASYNC_BULK_TENSOR_REDUCE_INTR<dim, shared32, mode>;
    }
  }
}

// TMA Prefetch from Global memory to L2 cache
multiclass TMA_TENSOR_PREFETCH_INTR<int dim, string mode,
                                    list<Predicate> pred = [hasPTX<80>, hasSM<90>]> {
  defvar dims_dag = TMA_DIMS_UTIL<dim>.ins_dag;
  defvar dims_str = TMA_DIMS_UTIL<dim>.base_str;
  defvar asm_str_base = " [$tmap, {{" # dims_str # "}}]";

  defvar im2col_dag = TMA_IM2COL_UTIL<dim, mode>.ins_dag;
  defvar im2col_str = TMA_IM2COL_UTIL<dim, mode>.base_str;
  defvar asm_str = !if(!empty(im2col_str),
                       asm_str_base,
                       asm_str_base # ", {{" # im2col_str # "}}");

  defvar dim_val = TMA_DIMS_UTIL<dim, mode>.num_dims;
  defvar inst_name = "cp.async.bulk.prefetch.tensor"
                     # "." # dim_val # "d"
                     # "." # "L2.global"
                     # "." # !subst("_", "::", mode);

  defvar intr = !cast<Intrinsic>(
                  "int_nvvm_cp_async_bulk_tensor_prefetch_" # mode # "_" # dim_val # "d");

  defvar ins_dag  = !con((ins  B64:$tmap),
                         dims_dag,
                         im2col_dag,
                         (ins B64:$ch));
  defvar intr_dag = !con((intr B64:$tmap),
                         !setdagop(dims_dag, intr),
                         !setdagop(im2col_dag, intr),
                         (intr B64:$ch, 0));
  defvar intr_dag_with_ch = !con((intr B64:$tmap),
                                 !setdagop(dims_dag, intr),
                                 !setdagop(im2col_dag, intr),
                                 (intr B64:$ch, -1));

  def "" : NVPTXInst<(outs), ins_dag,
             inst_name # asm_str # ";",
             [intr_dag]>,
             Requires<pred>;
  def _CH : NVPTXInst<(outs), ins_dag,
              inst_name # ".L2::cache_hint" # asm_str # ", $ch;",
              [intr_dag_with_ch]>,
              Requires<pred>;
}
foreach dim = 1...5 in {
  foreach mode = !if(!ge(dim, 3), ["tile", "im2col"], ["tile"]) in {
    defvar suffix = !toupper(mode) # "_" # dim # "D";
    defm TMA_TENSOR_PF_ # suffix : TMA_TENSOR_PREFETCH_INTR<dim, mode>;
  }
}
foreach dim = 3...5 in {
  foreach mode = ["im2col_w", "im2col_w_128"] in {
    defvar suffix = !toupper(mode) # "_" # dim # "D";
    defm TMA_TENSOR_PF_ # suffix : TMA_TENSOR_PREFETCH_INTR<dim, mode,
                                   [hasTMACTAGroupSupport]>;
  }
}
defm TMA_TENSOR_PF_TILE_GATHER4_2D : TMA_TENSOR_PREFETCH_INTR<5, "tile_gather4",
                                     [hasTMACTAGroupSupport]>;

//Prefetchu and Prefetch

defvar frag_pat = (int_nvvm_prefetch_tensormap node:$addr);

multiclass PREFETCH_TENSORMAP_PATFRAG<string suffix, code predicate> {
  def !tolower(suffix) : PatFrag<!setdagop(frag_pat, ops), frag_pat, predicate>;
}

defm prefetch_tensormap_ : PREFETCH_TENSORMAP_PATFRAG<"CONST", AS_match.const>;
defm prefetch_tensormap_ : PREFETCH_TENSORMAP_PATFRAG<"GENERIC", AS_match.generic>;
defm prefetch_tensormap_ : PREFETCH_TENSORMAP_PATFRAG<"PARAM", AS_match.param>;

multiclass PREFETCH_TENSORMAP_INST<string addrspace_name, PatFrag pattern_frag> {
  def "" : BasicNVPTXInst<(outs), (ins ADDR:$addr),
           "prefetch" # addrspace_name # ".tensormap",
           [(pattern_frag addr:$addr)]>,
           Requires<[hasPTX<80>, hasSM<90>]>;
}

defm PREFETCH_CONST_TENSORMAP   : PREFETCH_TENSORMAP_INST<".const", prefetch_tensormap_const>;
defm PREFETCH_GENERIC_TENSORMAP : PREFETCH_TENSORMAP_INST<"", prefetch_tensormap_generic>;
defm PREFETCH_PARAM_TENSORMAP   : PREFETCH_TENSORMAP_INST<".param", prefetch_tensormap_param>;
  
class PREFETCH_INTRS<string InstName, Intrinsic Intr> :
          BasicNVPTXInst<(outs), (ins ADDR:$addr),
          InstName,
          [(Intr addr:$addr)]>,
          Requires<[hasPTX<80>, hasSM<90>]>;

def PREFETCHU_L1 : PREFETCH_INTRS<"prefetchu.L1", int_nvvm_prefetchu_L1>;   
def PREFETCH_L1 : PREFETCH_INTRS<"prefetch.L1", int_nvvm_prefetch_L1>;
def PREFETCH_L2 : PREFETCH_INTRS<"prefetch.L2", int_nvvm_prefetch_L2>;
def PREFETCH_GLOBAL_L1 : PREFETCH_INTRS<"prefetch.global.L1", int_nvvm_prefetch_global_L1>;
def PREFETCH_LOCAL_L1 : PREFETCH_INTRS<"prefetch.local.L1", int_nvvm_prefetch_local_L1>;
def PREFETCH_GLOBAL_L2 : PREFETCH_INTRS<"prefetch.global.L2", int_nvvm_prefetch_global_L2>;
def PREFETCH_LOCAL_L2 : PREFETCH_INTRS<"prefetch.local.L2", int_nvvm_prefetch_local_L2>;
def PREFETCH_GLOBAL_L2_EVICT_NORMAL : PREFETCH_INTRS<"prefetch.global.L2::evict_normal", 
                                      int_nvvm_prefetch_global_L2_evict_normal>;
def PREFETCH_GLOBAL_L2_EVICT_LAST : PREFETCH_INTRS<"prefetch.global.L2::evict_last", 
                                    int_nvvm_prefetch_global_L2_evict_last>;

//Applypriority intrinsics
class APPLYPRIORITY_L2_INTRS<string addrspace> :
          BasicNVPTXInst<(outs), (ins ADDR:$addr, B64:$size),
          StrJoin<".", ["applypriority", addrspace , "L2::evict_normal"]>.ret,
          [(!cast<Intrinsic>(StrJoin<"_", ["int_nvvm_applypriority", addrspace , "L2_evict_normal"]>.ret)
          addr:$addr, i64:$size)]>,
          Requires<[hasPTX<74>, hasSM<80>]>;

def APPLYPRIORITY_L2_EVICT_NORMAL        : APPLYPRIORITY_L2_INTRS<"">;
def APPLYPRIORITY_GLOBAL_L2_EVICT_NORMAL : APPLYPRIORITY_L2_INTRS<"global">;

//Discard Intrinsics

def discard_size_imm : TImmLeaf<i64, [{ return Imm == 128; }]>;

class DISCARD_L2_INTRS<string addrspace> :
          BasicNVPTXInst<(outs), (ins ADDR:$addr, i64imm:$size),
          StrJoin<".", ["discard", addrspace , "L2"]>.ret,
          [(!cast<Intrinsic>(StrJoin<"_", ["int_nvvm_discard", addrspace , "L2"]>.ret)
          addr:$addr, discard_size_imm:$size)]>,
          Requires<[hasPTX<74>, hasSM<80>]>;

def DISCARD_L2        : DISCARD_L2_INTRS<"">;
def DISCARD_GLOBAL_L2 : DISCARD_L2_INTRS<"global">;

//-----------------------------------
// MBarrier Functions
//-----------------------------------

let Predicates = [hasPTX<70>, hasSM<80>] in {
  class MBARRIER_INIT<string AddrSpace, Intrinsic Intrin> :
            BasicNVPTXInst<(outs), (ins ADDR:$addr, B32:$count),
            "mbarrier.init" # AddrSpace # ".b64",
            [(Intrin addr:$addr, i32:$count)]>;

  def MBARRIER_INIT : MBARRIER_INIT<"", int_nvvm_mbarrier_init>;
  def MBARRIER_INIT_SHARED : MBARRIER_INIT<".shared",
                                            int_nvvm_mbarrier_init_shared>;

  class MBARRIER_INVAL<string AddrSpace, Intrinsic Intrin> :
            BasicNVPTXInst<(outs), (ins ADDR:$addr),
            "mbarrier.inval" # AddrSpace # ".b64",
            [(Intrin addr:$addr)]>;

  def MBARRIER_INVAL : MBARRIER_INVAL<"", int_nvvm_mbarrier_inval>;
  def MBARRIER_INVAL_SHARED : MBARRIER_INVAL<".shared",
                                              int_nvvm_mbarrier_inval_shared>;

  class MBARRIER_ARRIVE<string AddrSpace, Intrinsic Intrin> :
            BasicNVPTXInst<(outs B64:$state), (ins ADDR:$addr),
            "mbarrier.arrive" # AddrSpace # ".b64",
            [(set i64:$state, (Intrin addr:$addr))]>;

  def MBARRIER_ARRIVE : MBARRIER_ARRIVE<"", int_nvvm_mbarrier_arrive>;
  def MBARRIER_ARRIVE_SHARED :
    MBARRIER_ARRIVE<".shared", int_nvvm_mbarrier_arrive_shared>;

  class MBARRIER_ARRIVE_NOCOMPLETE<string AddrSpace, Intrinsic Intrin> :
            BasicNVPTXInst<(outs B64:$state),
            (ins ADDR:$addr, B32:$count),
            "mbarrier.arrive.noComplete" # AddrSpace # ".b64",
      [(set i64:$state, (Intrin addr:$addr, i32:$count))]>;

  def MBARRIER_ARRIVE_NOCOMPLETE :
    MBARRIER_ARRIVE_NOCOMPLETE<"", int_nvvm_mbarrier_arrive_noComplete>;
  def MBARRIER_ARRIVE_NOCOMPLETE_SHARED :
    MBARRIER_ARRIVE_NOCOMPLETE<".shared", int_nvvm_mbarrier_arrive_noComplete_shared>;

  class MBARRIER_ARRIVE_DROP<string AddrSpace, Intrinsic Intrin> :
            BasicNVPTXInst<(outs B64:$state), (ins ADDR:$addr),
            "mbarrier.arrive_drop" # AddrSpace # ".b64",
            [(set i64:$state, (Intrin addr:$addr))]>;

  def MBARRIER_ARRIVE_DROP :
    MBARRIER_ARRIVE_DROP<"", int_nvvm_mbarrier_arrive_drop>;
  def MBARRIER_ARRIVE_DROP_SHARED :
    MBARRIER_ARRIVE_DROP<".shared", int_nvvm_mbarrier_arrive_drop_shared>;

  class MBARRIER_ARRIVE_DROP_NOCOMPLETE<string AddrSpace, Intrinsic Intrin> :
            BasicNVPTXInst<(outs B64:$state),
            (ins ADDR:$addr, B32:$count),
            "mbarrier.arrive_drop.noComplete" # AddrSpace # ".b64",
            [(set i64:$state, (Intrin addr:$addr, i32:$count))]>;

  def MBARRIER_ARRIVE_DROP_NOCOMPLETE :
    MBARRIER_ARRIVE_DROP_NOCOMPLETE<"", int_nvvm_mbarrier_arrive_drop_noComplete>;
  def MBARRIER_ARRIVE_DROP_NOCOMPLETE_SHARED :
    MBARRIER_ARRIVE_DROP_NOCOMPLETE<".shared",
                        int_nvvm_mbarrier_arrive_drop_noComplete_shared>;

  class MBARRIER_TEST_WAIT<string AddrSpace, Intrinsic Intrin> :
            BasicNVPTXInst<(outs B1:$res), (ins ADDR:$addr, B64:$state),
            "mbarrier.test_wait" # AddrSpace # ".b64",
            [(set i1:$res, (Intrin addr:$addr, i64:$state))]>;

  def MBARRIER_TEST_WAIT :
    MBARRIER_TEST_WAIT<"", int_nvvm_mbarrier_test_wait>;
  def MBARRIER_TEST_WAIT_SHARED :
    MBARRIER_TEST_WAIT<".shared", int_nvvm_mbarrier_test_wait_shared>;

  def MBARRIER_PENDING_COUNT :
            BasicNVPTXInst<(outs B32:$res), (ins B64:$state),
            "mbarrier.pending_count.b64",
            [(set i32:$res, (int_nvvm_mbarrier_pending_count i64:$state))]>;
}
//-----------------------------------
// Math Functions
//-----------------------------------

// Map min(1.0, max(0.0, x)) to sat(x)
// Note that max(0.0, min(x, 1.0)) cannot be mapped to sat(x) because when x is
// NaN
// max(0.0, min(x, 1.0)) is 1.0 while sat(x) is 0.
// Same story for fmax, fmin.

def : Pat<(int_nvvm_fmin_f fpimm_1,
            (int_nvvm_fmax_f fpimm_0, f32:$a)),
          (CVT_f32_f32 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_f fpimm_1,
            (int_nvvm_fmax_f f32:$a, fpimm_0)),
          (CVT_f32_f32 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_f
            (int_nvvm_fmax_f fpimm_0, f32:$a), fpimm_1),
          (CVT_f32_f32 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_f
            (int_nvvm_fmax_f f32:$a, fpimm_0), fpimm_1),
          (CVT_f32_f32 $a, CvtSAT)>;

def : Pat<(int_nvvm_fmin_d fpimm_1,
            (int_nvvm_fmax_d fpimm_0, f64:$a)),
          (CVT_f64_f64 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_d fpimm_1,
            (int_nvvm_fmax_d f64:$a, fpimm_0)),
          (CVT_f64_f64 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_d
            (int_nvvm_fmax_d fpimm_0, f64:$a), fpimm_1),
          (CVT_f64_f64 $a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_d
            (int_nvvm_fmax_d f64:$a, fpimm_0), fpimm_1),
          (CVT_f64_f64 $a, CvtSAT)>;


// We need a full string for OpcStr here because we need to deal with case like
// INT_PTX_RECIP.
class F_MATH_1<string OpcStr, RegTyInfo dst, RegTyInfo src, Intrinsic IntOP,
               list<Predicate> Preds = []>
  : BasicNVPTXInst<(outs dst.RC:$dst),
              (ins src.RC:$src0),
              OpcStr,
              [(set dst.Ty:$dst, (IntOP src.Ty:$src0))]>,
    Requires<Preds>;

// We need a full string for OpcStr here because we need to deal with the case
// like INT_PTX_NATIVE_POWR_F.
class F_MATH_2<string OpcStr, NVPTXRegClass t_regclass,
  NVPTXRegClass s0_regclass, NVPTXRegClass s1_regclass, Intrinsic IntOP,
  list<Predicate> Preds = []>
            : BasicNVPTXInst<(outs t_regclass:$dst),
              (ins s0_regclass:$src0, s1_regclass:$src1),
            OpcStr,
        [(set t_regclass:$dst, (IntOP s0_regclass:$src0, s1_regclass:$src1))]>,
        Requires<Preds>;

class F_MATH_3<string OpcStr, NVPTXRegClass t_regclass,
  NVPTXRegClass s0_regclass, NVPTXRegClass s1_regclass,
  NVPTXRegClass s2_regclass, Intrinsic IntOP, list<Predicate> Preds = []>
            : BasicNVPTXInst<(outs t_regclass:$dst),
              (ins s0_regclass:$src0, s1_regclass:$src1, s2_regclass:$src2),
            OpcStr,
        [(set t_regclass:$dst,
          (IntOP s0_regclass:$src0, s1_regclass:$src1, s2_regclass:$src2))]>,
          Requires<Preds>;

//
// MISC
//

def INT_NVVM_NANOSLEEP_I : BasicNVPTXInst<(outs), (ins i32imm:$i), "nanosleep.u32",
                             [(int_nvvm_nanosleep imm:$i)]>,
        Requires<[hasPTX<63>, hasSM<70>]>;
def INT_NVVM_NANOSLEEP_R : BasicNVPTXInst<(outs), (ins B32:$i), "nanosleep.u32",
                             [(int_nvvm_nanosleep i32:$i)]>,
        Requires<[hasPTX<63>, hasSM<70>]>;

let hasSideEffects = 1 in {
// Performance Monitor events
def INT_PM_EVENT_MASK : BasicNVPTXInst<(outs),
                        (ins i16imm:$mask),
                        "pmevent.mask",
                        [(int_nvvm_pm_event_mask timm:$mask)]>,
                        Requires<[hasSM<20>, hasPTX<30>]>;
} // hasSideEffects

//
// Min Max
//

def : Pat<(int_nvvm_fmin_f f32:$a, f32:$b), (MIN_f32_rr $a, $b, NoFTZ)>;
def : Pat<(int_nvvm_fmin_ftz_f f32:$a, f32:$b), (MIN_f32_rr $a, $b, FTZ)>;

let Predicates = [hasPTX<70>, hasSM<80>] in {
  def : Pat<(int_nvvm_fmin_nan_f f32:$a, f32:$b), (MIN_NAN_f32_rr $a, $b, NoFTZ)>;
  def : Pat<(int_nvvm_fmin_ftz_nan_f f32:$a, f32:$b), (MIN_NAN_f32_rr $a, $b, FTZ)>;
}

def INT_NVVM_FMIN_XORSIGN_ABS_F :
 F_MATH_2<"min.xorsign.abs.f32", B32, B32, B32, int_nvvm_fmin_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMIN_FTZ_XORSIGN_ABS_F :
  F_MATH_2<"min.ftz.xorsign.abs.f32", B32, B32, B32, int_nvvm_fmin_ftz_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMIN_NAN_XORSIGN_ABS_F :
  F_MATH_2<"min.NaN.xorsign.abs.f32", B32, B32, B32, int_nvvm_fmin_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMIN_FTZ_NAN_XORSIGN_ABS_F :
  F_MATH_2<"min.ftz.NaN.xorsign.abs.f32", B32, B32, B32, int_nvvm_fmin_ftz_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;


def : Pat<(int_nvvm_fmax_f f32:$a, f32:$b), (MAX_f32_rr $a, $b, NoFTZ)>;
def : Pat<(int_nvvm_fmax_ftz_f f32:$a, f32:$b), (MAX_f32_rr $a, $b, FTZ)>;

let Predicates = [hasPTX<70>, hasSM<80>] in {
  def : Pat<(int_nvvm_fmax_nan_f f32:$a, f32:$b), (MAX_NAN_f32_rr $a, $b, NoFTZ)>;
  def : Pat<(int_nvvm_fmax_ftz_nan_f f32:$a, f32:$b), (MAX_NAN_f32_rr $a, $b, FTZ)>;
}

def INT_NVVM_FMAX_XORSIGN_ABS_F :
  F_MATH_2<"max.xorsign.abs.f32", B32, B32, B32, int_nvvm_fmax_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMAX_FTZ_XORSIGN_ABS_F :
  F_MATH_2<"max.ftz.xorsign.abs.f32", B32, B32, B32, int_nvvm_fmax_ftz_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMAX_NAN_XORSIGN_ABS_F :
  F_MATH_2<"max.NaN.xorsign.abs.f32", B32, B32, B32, int_nvvm_fmax_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;
def INT_NVVM_FMAX_FTZ_NAN_XORSIGN_ABS_F :
  F_MATH_2<"max.ftz.NaN.xorsign.abs.f32", B32, B32, B32, int_nvvm_fmax_ftz_nan_xorsign_abs_f,
    [hasPTX<72>, hasSM<86>]>;

def : Pat<(int_nvvm_fmin_d f64:$a, f64:$b), (MIN_f64_rr $a, $b)>;
def : Pat<(int_nvvm_fmax_d f64:$a, f64:$b), (MAX_f64_rr $a, $b)>;

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
      int_nvvm_fmax_f16), B16>,
    MIN_MAX_TUPLE<"_ftz_f16", !if(!eq(IntName, "min"), int_nvvm_fmin_ftz_f16,
      int_nvvm_fmax_ftz_f16), B16>,
    MIN_MAX_TUPLE<"_NaN_f16", !if(!eq(IntName, "min"), int_nvvm_fmin_nan_f16,
      int_nvvm_fmax_nan_f16), B16>,
    MIN_MAX_TUPLE<"_ftz_NaN_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_nan_f16, int_nvvm_fmax_ftz_nan_f16), B16>,
    MIN_MAX_TUPLE<"_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_xorsign_abs_f16, int_nvvm_fmax_xorsign_abs_f16),
      B16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_xorsign_abs_f16, int_nvvm_fmax_ftz_xorsign_abs_f16),
      B16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_xorsign_abs_f16, int_nvvm_fmax_nan_xorsign_abs_f16),
      B16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_NaN_xorsign_abs_f16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_nan_xorsign_abs_f16,
      int_nvvm_fmax_ftz_nan_xorsign_abs_f16), B16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_f16x2", !if(!eq(IntName, "min"), int_nvvm_fmin_f16x2,
      int_nvvm_fmax_f16x2), B32>,
    MIN_MAX_TUPLE<"_ftz_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_f16x2, int_nvvm_fmax_ftz_f16x2), B32>,
    MIN_MAX_TUPLE<"_NaN_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_f16x2, int_nvvm_fmax_nan_f16x2), B32>,
    MIN_MAX_TUPLE<"_ftz_NaN_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_nan_f16x2, int_nvvm_fmax_ftz_nan_f16x2), B32>,
    MIN_MAX_TUPLE<"_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_xorsign_abs_f16x2, int_nvvm_fmax_xorsign_abs_f16x2),
      B32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_xorsign_abs_f16x2, int_nvvm_fmax_ftz_xorsign_abs_f16x2),
      B32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_xorsign_abs_f16x2, int_nvvm_fmax_nan_xorsign_abs_f16x2),
      B32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_ftz_NaN_xorsign_abs_f16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_ftz_nan_xorsign_abs_f16x2,
      int_nvvm_fmax_ftz_nan_xorsign_abs_f16x2),
      B32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_bf16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_bf16, int_nvvm_fmax_bf16), B16>,
    MIN_MAX_TUPLE<"_NaN_bf16", !if(!eq(IntName, "min"), int_nvvm_fmin_nan_bf16,
      int_nvvm_fmax_nan_bf16), B16>,
    MIN_MAX_TUPLE<"_xorsign_abs_bf16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_xorsign_abs_bf16, int_nvvm_fmax_xorsign_abs_bf16),
      B16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_bf16", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_xorsign_abs_bf16, int_nvvm_fmax_nan_xorsign_abs_bf16),
      B16, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_bf16x2", !if(!eq(IntName, "min"), int_nvvm_fmin_bf16x2,
      int_nvvm_fmax_bf16x2), B32>,
    MIN_MAX_TUPLE<"_NaN_bf16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_bf16x2, int_nvvm_fmax_nan_bf16x2), B32>,
    MIN_MAX_TUPLE<"_xorsign_abs_bf16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_xorsign_abs_bf16x2, int_nvvm_fmax_xorsign_abs_bf16x2),
      B32, [hasPTX<72>, hasSM<86>]>,
    MIN_MAX_TUPLE<"_NaN_xorsign_abs_bf16x2", !if(!eq(IntName, "min"),
      int_nvvm_fmin_nan_xorsign_abs_bf16x2,
      int_nvvm_fmax_nan_xorsign_abs_bf16x2),
      B32, [hasPTX<72>, hasSM<86>]>] in {
        def P.Variant : F_MATH_2<!strconcat(
          IntName, !subst("_", ".", P.Variant)),
          P.RegClass, P.RegClass, P.RegClass, P.Intr, P.Predicates>;
  }
}

defm INT_NVVM_FMIN : MIN_MAX<"min">;
defm INT_NVVM_FMAN : MIN_MAX<"max">;

//
// Multiplication
//

def : Pat<(int_nvvm_mulhi_s i16:$a, i16:$b), (MUL_HI_S16rr $a, $b)>;
def : Pat<(int_nvvm_mulhi_us i16:$a, i16:$b), (MUL_HI_U16rr $a, $b)>;
def : Pat<(int_nvvm_mulhi_i i32:$a, i32:$b), (MUL_HI_S32rr $a, $b)>;
def : Pat<(int_nvvm_mulhi_ui i32:$a, i32:$b), (MUL_HI_U32rr $a, $b)>;
def : Pat<(int_nvvm_mulhi_ll i64:$a, i64:$b), (MUL_HI_S64rr $a, $b)>;
def : Pat<(int_nvvm_mulhi_ull i64:$a, i64:$b), (MUL_HI_U64rr $a, $b)>;

def INT_NVVM_MUL_RN_FTZ_F : F_MATH_2<"mul.rn.ftz.f32", B32, B32, B32, int_nvvm_mul_rn_ftz_f>;
def INT_NVVM_MUL_RN_F : F_MATH_2<"mul.rn.f32", B32, B32, B32, int_nvvm_mul_rn_f>;
def INT_NVVM_MUL_RZ_FTZ_F : F_MATH_2<"mul.rz.ftz.f32", B32, B32, B32, int_nvvm_mul_rz_ftz_f>;
def INT_NVVM_MUL_RZ_F : F_MATH_2<"mul.rz.f32", B32, B32, B32, int_nvvm_mul_rz_f>;
def INT_NVVM_MUL_RM_FTZ_F : F_MATH_2<"mul.rm.ftz.f32", B32, B32, B32, int_nvvm_mul_rm_ftz_f>;
def INT_NVVM_MUL_RM_F : F_MATH_2<"mul.rm.f32", B32, B32, B32, int_nvvm_mul_rm_f>;
def INT_NVVM_MUL_RP_FTZ_F : F_MATH_2<"mul.rp.ftz.f32", B32, B32, B32, int_nvvm_mul_rp_ftz_f>;
def INT_NVVM_MUL_RP_F : F_MATH_2<"mul.rp.f32", B32, B32, B32, int_nvvm_mul_rp_f>;

def INT_NVVM_MUL_RN_D : F_MATH_2<"mul.rn.f64", B64, B64, B64, int_nvvm_mul_rn_d>;
def INT_NVVM_MUL_RZ_D : F_MATH_2<"mul.rz.f64", B64, B64, B64, int_nvvm_mul_rz_d>;
def INT_NVVM_MUL_RM_D : F_MATH_2<"mul.rm.f64", B64, B64, B64, int_nvvm_mul_rm_d>;
def INT_NVVM_MUL_RP_D : F_MATH_2<"mul.rp.f64", B64, B64, B64, int_nvvm_mul_rp_d>;

def INT_NVVM_MUL24_I : F_MATH_2<"mul24.lo.s32", B32, B32, B32, int_nvvm_mul24_i>;
def INT_NVVM_MUL24_UI : F_MATH_2<"mul24.lo.u32", B32, B32, B32, int_nvvm_mul24_ui>;

//
// Div
//

def : Pat<(int_nvvm_div_approx_ftz_f f32:$a, f32:$b), (DIV_APPROX_F32_rr $a, $b, FTZ)>;
def : Pat<(int_nvvm_div_approx_f f32:$a, f32:$b), (DIV_APPROX_F32_rr $a, $b, NoFTZ)>;

def INT_NVVM_DIV_RN_FTZ_F : F_MATH_2<"div.rn.ftz.f32", B32, B32, B32, int_nvvm_div_rn_ftz_f>;
def INT_NVVM_DIV_RN_F     : F_MATH_2<"div.rn.f32", B32, B32, B32, int_nvvm_div_rn_f>;
def INT_NVVM_DIV_RZ_FTZ_F : F_MATH_2<"div.rz.ftz.f32", B32, B32, B32, int_nvvm_div_rz_ftz_f>;
def INT_NVVM_DIV_RZ_F     : F_MATH_2<"div.rz.f32", B32, B32, B32, int_nvvm_div_rz_f>;
def INT_NVVM_DIV_RM_FTZ_F : F_MATH_2<"div.rm.ftz.f32", B32, B32, B32, int_nvvm_div_rm_ftz_f>;
def INT_NVVM_DIV_RM_F     : F_MATH_2<"div.rm.f32", B32, B32, B32, int_nvvm_div_rm_f>;
def INT_NVVM_DIV_RP_FTZ_F : F_MATH_2<"div.rp.ftz.f32", B32, B32, B32, int_nvvm_div_rp_ftz_f>;
def INT_NVVM_DIV_RP_F     : F_MATH_2<"div.rp.f32", B32, B32, B32, int_nvvm_div_rp_f>;

def INT_NVVM_DIV_RN_D : F_MATH_2<"div.rn.f64", B64, B64, B64, int_nvvm_div_rn_d>;
def INT_NVVM_DIV_RZ_D : F_MATH_2<"div.rz.f64", B64, B64, B64, int_nvvm_div_rz_d>;
def INT_NVVM_DIV_RM_D : F_MATH_2<"div.rm.f64", B64, B64, B64, int_nvvm_div_rm_d>;
def INT_NVVM_DIV_RP_D : F_MATH_2<"div.rp.f64", B64, B64, B64, int_nvvm_div_rp_d>;

def : Pat<(int_nvvm_div_full f32:$a, f32:$b), (FDIV32rr $a, $b, NoFTZ)>;
def : Pat<(int_nvvm_div_full f32:$a, fpimm:$b), (FDIV32ri $a, f32imm:$b, NoFTZ)>;
def : Pat<(int_nvvm_div_full_ftz f32:$a, f32:$b), (FDIV32rr $a, $b, FTZ)>;
def : Pat<(int_nvvm_div_full_ftz f32:$a, fpimm:$b), (FDIV32ri $a, f32imm:$b, FTZ)>;

//
// Sad
//

def INT_NVVM_SAD_S : F_MATH_3<"sad.s16", B16, B16, B16, B16, int_nvvm_sad_s>;
def INT_NVVM_SAD_US : F_MATH_3<"sad.u16", B16, B16, B16, B16, int_nvvm_sad_us>;
def INT_NVVM_SAD_I : F_MATH_3<"sad.s32", B32, B32, B32, B32, int_nvvm_sad_i>;
def INT_NVVM_SAD_UI : F_MATH_3<"sad.u32", B32, B32, B32, B32, int_nvvm_sad_ui>;
def INT_NVVM_SAD_LL : F_MATH_3<"sad.s64", B64, B64, B64, B64, int_nvvm_sad_ll>;
def INT_NVVM_SAD_ULL : F_MATH_3<"sad.u64", B64, B64, B64, B64, int_nvvm_sad_ull>;

//
// Floor  Ceil
//

def : Pat<(int_nvvm_floor_ftz_f f32:$a), (CVT_f32_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_floor_f f32:$a),     (CVT_f32_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_floor_d f64:$a),     (CVT_f64_f64 $a, CvtRMI)>;

def : Pat<(int_nvvm_ceil_ftz_f f32:$a), (CVT_f32_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_ceil_f f32:$a),     (CVT_f32_f32 $a, CvtRPI)>;
def : Pat<(int_nvvm_ceil_d f64:$a),     (CVT_f64_f64 $a, CvtRPI)>;

//
// Abs
//

multiclass F_ABS<string suffix, RegTyInfo RT, bit support_ftz, list<Predicate> preds = []> {
  def "" : F_MATH_1<"abs." # suffix, RT, RT, int_nvvm_fabs, preds>;
  if support_ftz then
    def _FTZ : F_MATH_1<"abs.ftz." # suffix, RT, RT, int_nvvm_fabs_ftz, preds>;
}

defm ABS_F16 : F_ABS<"f16", F16RT, support_ftz = true, preds = [hasPTX<65>, hasSM<53>]>;
defm ABS_F16X2 : F_ABS<"f16x2", F16X2RT, support_ftz = true, preds = [hasPTX<65>, hasSM<53>]>;

defm ABS_BF16 : F_ABS<"bf16", BF16RT, support_ftz = false, preds = [hasPTX<70>, hasSM<80>]>;
defm ABS_BF16X2 : F_ABS<"bf16x2", BF16X2RT, support_ftz = false, preds = [hasPTX<70>, hasSM<80>]>;

defm ABS_F32 : F_ABS<"f32", F32RT, support_ftz = true>;
defm ABS_F64 : F_ABS<"f64", F64RT, support_ftz = false>;

//
// copysign
//

def fcopysign_nvptx : SDNode<"NVPTXISD::FCOPYSIGN", SDTFPBinOp>;

foreach t = [F32RT, F64RT] in
  def COPYSIGN_ # t :
      BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$src0, t.RC:$src1),
                "copysign." # t.PtxType,
              [(set t.Ty:$dst, (fcopysign_nvptx t.Ty:$src1, t.Ty:$src0))]>;

//
// Neg bf16, bf16x2
//

def INT_NVVM_NEG_BF16 : F_MATH_1<"neg.bf16", BF16RT,
  BF16RT, int_nvvm_neg_bf16, [hasPTX<70>, hasSM<80>]>;
def INT_NVVM_NEG_BF16X2 : F_MATH_1<"neg.bf16x2", BF16X2RT,
  BF16X2RT, int_nvvm_neg_bf16x2, [hasPTX<70>, hasSM<80>]>;

//
// Round
//

def : Pat<(int_nvvm_round_ftz_f f32:$a), (CVT_f32_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_round_f f32:$a),     (CVT_f32_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_round_d f64:$a),     (CVT_f64_f64 $a, CvtRNI)>;

//
// Trunc
//

def : Pat<(int_nvvm_trunc_ftz_f f32:$a), (CVT_f32_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_trunc_f f32:$a),     (CVT_f32_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_trunc_d f64:$a),     (CVT_f64_f64 $a, CvtRZI)>;

//
// Saturate
//

def : Pat<(int_nvvm_saturate_ftz_f f32:$a), (CVT_f32_f32 $a, CvtSAT_FTZ)>;
def : Pat<(int_nvvm_saturate_f f32:$a),     (CVT_f32_f32 $a, CvtSAT)>;
def : Pat<(int_nvvm_saturate_d f64:$a),     (CVT_f64_f64 $a, CvtSAT)>;

//
// Exp2  Log2
//

def : Pat<(int_nvvm_ex2_approx_ftz_f f32:$a), (EX2_APPROX_f32 $a, FTZ)>;
def : Pat<(int_nvvm_ex2_approx_f f32:$a), (EX2_APPROX_f32 $a, NoFTZ)>;

let Predicates = [hasPTX<70>, hasSM<75>] in {
  def : Pat<(int_nvvm_ex2_approx_f16 f16:$a), (EX2_APPROX_f16 $a)>;
  def : Pat<(int_nvvm_ex2_approx_f16x2 v2f16:$a), (EX2_APPROX_f16x2 $a)>;
}

def LG2_APPROX_f32 :
  BasicFlagsNVPTXInst<(outs B32:$dst), (ins B32:$src), (ins FTZFlag:$ftz),
    "lg2.approx$ftz.f32",
    [(set f32:$dst, (flog2 f32:$src))]>;

def LG2_APPROX_f64 :
  BasicNVPTXInst<(outs B64:$dst), (ins B64:$src),
    "lg2.approx.f64",
    [(set f64:$dst, (flog2 f64:$src))]>;

def : Pat<(int_nvvm_lg2_approx_ftz_f f32:$a), (LG2_APPROX_f32 $a, FTZ)>;
def : Pat<(int_nvvm_lg2_approx_f f32:$a), (LG2_APPROX_f32 $a, NoFTZ)>;
def : Pat<(int_nvvm_lg2_approx_d f64:$a), (LG2_APPROX_f64 $a)>;

//
// Sin  Cos
//

def : Pat<(int_nvvm_sin_approx_ftz_f f32:$a), (SIN_APPROX_f32 $a, FTZ)>;
def : Pat<(int_nvvm_sin_approx_f f32:$a), (SIN_APPROX_f32 $a, NoFTZ)>;
def : Pat<(int_nvvm_cos_approx_ftz_f f32:$a), (COS_APPROX_f32 $a, FTZ)>;
def : Pat<(int_nvvm_cos_approx_f f32:$a), (COS_APPROX_f32 $a, NoFTZ)>;

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
    FMA_TUPLE<"_rn_f64", int_nvvm_fma_rn_d, B64>,
    FMA_TUPLE<"_rz_f64", int_nvvm_fma_rz_d, B64>,
    FMA_TUPLE<"_rm_f64", int_nvvm_fma_rm_d, B64>,
    FMA_TUPLE<"_rp_f64", int_nvvm_fma_rp_d, B64>,

    FMA_TUPLE<"_rn_ftz_f32", int_nvvm_fma_rn_ftz_f, B32>,
    FMA_TUPLE<"_rn_f32", int_nvvm_fma_rn_f, B32>,
    FMA_TUPLE<"_rz_ftz_f32", int_nvvm_fma_rz_ftz_f, B32>,
    FMA_TUPLE<"_rz_f32", int_nvvm_fma_rz_f, B32>,
    FMA_TUPLE<"_rm_f32", int_nvvm_fma_rm_f, B32>,
    FMA_TUPLE<"_rm_ftz_f32", int_nvvm_fma_rm_ftz_f, B32>,
    FMA_TUPLE<"_rp_f32", int_nvvm_fma_rp_f, B32>,
    FMA_TUPLE<"_rp_ftz_f32", int_nvvm_fma_rp_ftz_f, B32>,

    FMA_TUPLE<"_rn_f16", int_nvvm_fma_rn_f16, B16, [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_f16", int_nvvm_fma_rn_ftz_f16, B16,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_sat_f16", int_nvvm_fma_rn_sat_f16, B16,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_sat_f16", int_nvvm_fma_rn_ftz_sat_f16, B16,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_relu_f16", int_nvvm_fma_rn_relu_f16, B16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_relu_f16", int_nvvm_fma_rn_ftz_relu_f16, B16,
      [hasPTX<70>, hasSM<80>]>,

    FMA_TUPLE<"_rn_bf16", int_nvvm_fma_rn_bf16, B16, [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_bf16", int_nvvm_fma_rn_ftz_bf16, B16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_sat_bf16", int_nvvm_fma_rn_sat_bf16, B16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_sat_bf16", int_nvvm_fma_rn_ftz_sat_bf16, B16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_relu_bf16", int_nvvm_fma_rn_relu_bf16, B16,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_relu_bf16", int_nvvm_fma_rn_ftz_relu_bf16, B16,
      [hasPTX<70>, hasSM<80>]>,

    FMA_TUPLE<"_rn_f16x2", int_nvvm_fma_rn_f16x2, B32,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_f16x2", int_nvvm_fma_rn_ftz_f16x2, B32,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_sat_f16x2", int_nvvm_fma_rn_sat_f16x2, B32,
      [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_ftz_sat_f16x2", int_nvvm_fma_rn_ftz_sat_f16x2,
      B32, [hasPTX<42>, hasSM<53>]>,
    FMA_TUPLE<"_rn_relu_f16x2", int_nvvm_fma_rn_relu_f16x2, B32,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_ftz_relu_f16x2", int_nvvm_fma_rn_ftz_relu_f16x2,
      B32, [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_bf16x2", int_nvvm_fma_rn_bf16x2, B32,
      [hasPTX<70>, hasSM<80>]>,
    FMA_TUPLE<"_rn_relu_bf16x2", int_nvvm_fma_rn_relu_bf16x2, B32,
      [hasPTX<70>, hasSM<80>]>
  ] in {
    def P.Variant :
      F_MATH_3<!strconcat("fma", !subst("_", ".", P.Variant)),
        P.RegClass, P.RegClass, P.RegClass, P.RegClass, P.Intr, P.Predicates>;
  }
}

defm INT_NVVM_FMA : FMA_INST;

//
// Rcp
//

def INT_NVVM_RCP_RN_FTZ_F : F_MATH_1<"rcp.rn.ftz.f32", F32RT, F32RT, int_nvvm_rcp_rn_ftz_f>;
def INT_NVVM_RCP_RN_F : F_MATH_1<"rcp.rn.f32", F32RT, F32RT, int_nvvm_rcp_rn_f>;
def INT_NVVM_RCP_RZ_FTZ_F : F_MATH_1<"rcp.rz.ftz.f32", F32RT, F32RT, int_nvvm_rcp_rz_ftz_f>;
def INT_NVVM_RCP_RZ_F : F_MATH_1<"rcp.rz.f32", F32RT, F32RT, int_nvvm_rcp_rz_f>;
def INT_NVVM_RCP_RM_FTZ_F : F_MATH_1<"rcp.rm.ftz.f32", F32RT, F32RT, int_nvvm_rcp_rm_ftz_f>;
def INT_NVVM_RCP_RM_F : F_MATH_1<"rcp.rm.f32", F32RT, F32RT, int_nvvm_rcp_rm_f>;
def INT_NVVM_RCP_RP_FTZ_F : F_MATH_1<"rcp.rp.ftz.f32", F32RT, F32RT, int_nvvm_rcp_rp_ftz_f>;
def INT_NVVM_RCP_RP_F : F_MATH_1<"rcp.rp.f32", F32RT, F32RT, int_nvvm_rcp_rp_f>;

def INT_NVVM_RCP_RN_D : F_MATH_1<"rcp.rn.f64", F64RT, F64RT, int_nvvm_rcp_rn_d>;
def INT_NVVM_RCP_RZ_D : F_MATH_1<"rcp.rz.f64", F64RT, F64RT, int_nvvm_rcp_rz_d>;
def INT_NVVM_RCP_RM_D : F_MATH_1<"rcp.rm.f64", F64RT, F64RT, int_nvvm_rcp_rm_d>;
def INT_NVVM_RCP_RP_D : F_MATH_1<"rcp.rp.f64", F64RT, F64RT, int_nvvm_rcp_rp_d>;

def INT_NVVM_RCP_APPROX_FTZ_F : F_MATH_1<"rcp.approx.ftz.f32",
  F32RT, F32RT, int_nvvm_rcp_approx_ftz_f>;
def INT_NVVM_RCP_APPROX_FTZ_D : F_MATH_1<"rcp.approx.ftz.f64",
  F64RT, F64RT, int_nvvm_rcp_approx_ftz_d>;

//
// Sqrt
//

def INT_NVVM_SQRT_RN_FTZ_F : F_MATH_1<"sqrt.rn.ftz.f32",
  F32RT, F32RT, int_nvvm_sqrt_rn_ftz_f>;
def INT_NVVM_SQRT_RN_F : F_MATH_1<"sqrt.rn.f32", F32RT,
  F32RT, int_nvvm_sqrt_rn_f>;
def INT_NVVM_SQRT_RZ_FTZ_F : F_MATH_1<"sqrt.rz.ftz.f32",
  F32RT, F32RT, int_nvvm_sqrt_rz_ftz_f>;
def INT_NVVM_SQRT_RZ_F : F_MATH_1<"sqrt.rz.f32", F32RT,
  F32RT, int_nvvm_sqrt_rz_f>;
def INT_NVVM_SQRT_RM_FTZ_F : F_MATH_1<"sqrt.rm.ftz.f32",
  F32RT, F32RT, int_nvvm_sqrt_rm_ftz_f>;
def INT_NVVM_SQRT_RM_F : F_MATH_1<"sqrt.rm.f32", F32RT,
  F32RT, int_nvvm_sqrt_rm_f>;
def INT_NVVM_SQRT_RP_FTZ_F : F_MATH_1<"sqrt.rp.ftz.f32",
  F32RT, F32RT, int_nvvm_sqrt_rp_ftz_f>;
def INT_NVVM_SQRT_RP_F : F_MATH_1<"sqrt.rp.f32", F32RT,
  F32RT, int_nvvm_sqrt_rp_f>;
def INT_NVVM_SQRT_APPROX_FTZ_F : F_MATH_1<"sqrt.approx.ftz.f32",
  F32RT, F32RT, int_nvvm_sqrt_approx_ftz_f>;
def INT_NVVM_SQRT_APPROX_F : F_MATH_1<"sqrt.approx.f32",
  F32RT, F32RT, int_nvvm_sqrt_approx_f>;

def INT_NVVM_SQRT_RN_D : F_MATH_1<"sqrt.rn.f64", F64RT, F64RT, int_nvvm_sqrt_rn_d>;
def INT_NVVM_SQRT_RZ_D : F_MATH_1<"sqrt.rz.f64", F64RT, F64RT, int_nvvm_sqrt_rz_d>;
def INT_NVVM_SQRT_RM_D : F_MATH_1<"sqrt.rm.f64", F64RT, F64RT, int_nvvm_sqrt_rm_d>;
def INT_NVVM_SQRT_RP_D : F_MATH_1<"sqrt.rp.f64", F64RT, F64RT, int_nvvm_sqrt_rp_d>;

def fsqrt_approx : PatFrags<(ops node:$a),
                            [(fsqrt node:$a),
                             (int_nvvm_sqrt_f node:$a)], [{
  return !usePrecSqrtF32(N);
}]>;

// nvvm_sqrt intrinsic
def : Pat<(int_nvvm_sqrt_f f32:$a), (INT_NVVM_SQRT_RN_FTZ_F $a)>, Requires<[doF32FTZ]>;
def : Pat<(int_nvvm_sqrt_f f32:$a), (INT_NVVM_SQRT_RN_F $a)>;

def : Pat<(fsqrt_approx f32:$a), (INT_NVVM_SQRT_APPROX_FTZ_F $a)>, Requires<[doF32FTZ]>;
def : Pat<(fsqrt_approx f32:$a), (INT_NVVM_SQRT_APPROX_F $a)>;

//
// Rsqrt
//

foreach t = [F32RT, F64RT] in {
  def RSQRT_APPROX_ # t.Ty :
    BasicFlagsNVPTXInst<(outs t.RC:$dst),
                        (ins t.RC:$a), (ins FTZFlag:$ftz),
                        "rsqrt.approx$ftz.f" # t.Size>;
}

def : Pat<(int_nvvm_rsqrt_approx_ftz_f f32:$a), (RSQRT_APPROX_f32 $a, FTZ)>;
def : Pat<(int_nvvm_rsqrt_approx_ftz_d f64:$a), (RSQRT_APPROX_f64 $a, FTZ)>;
def : Pat<(int_nvvm_rsqrt_approx_f f32:$a), (RSQRT_APPROX_f32 $a, NoFTZ)>;
def : Pat<(int_nvvm_rsqrt_approx_d f64:$a), (RSQRT_APPROX_f64 $a, NoFTZ)>;


// 1.0f / sqrt_approx -> rsqrt_approx
let Predicates = [doRsqrtOpt] in {
  def : Pat<(fdiv fpimm_1, (int_nvvm_sqrt_approx_f f32:$a)),
          (RSQRT_APPROX_f32 $a, NoFTZ)>;
  def : Pat<(fdiv fpimm_1, (int_nvvm_sqrt_approx_ftz_f f32:$a)),
          (RSQRT_APPROX_f32 $a, FTZ)>;

  // same for int_nvvm_sqrt_f when non-precision sqrt is requested
  def : Pat<(fdiv fpimm_1, (fsqrt_approx f32:$a)),
          (RSQRT_APPROX_f32 $a)>;
}
//
// Add
//

def INT_NVVM_ADD_RN_FTZ_F : F_MATH_2<"add.rn.ftz.f32", B32, B32, B32, int_nvvm_add_rn_ftz_f>;
def INT_NVVM_ADD_RN_F : F_MATH_2<"add.rn.f32", B32, B32, B32, int_nvvm_add_rn_f>;
def INT_NVVM_ADD_RZ_FTZ_F : F_MATH_2<"add.rz.ftz.f32", B32, B32, B32, int_nvvm_add_rz_ftz_f>;
def INT_NVVM_ADD_RZ_F : F_MATH_2<"add.rz.f32", B32, B32, B32, int_nvvm_add_rz_f>;
def INT_NVVM_ADD_RM_FTZ_F : F_MATH_2<"add.rm.ftz.f32", B32, B32, B32, int_nvvm_add_rm_ftz_f>;
def INT_NVVM_ADD_RM_F : F_MATH_2<"add.rm.f32", B32, B32, B32, int_nvvm_add_rm_f>;
def INT_NVVM_ADD_RP_FTZ_F : F_MATH_2<"add.rp.ftz.f32", B32, B32, B32, int_nvvm_add_rp_ftz_f>;
def INT_NVVM_ADD_RP_F : F_MATH_2<"add.rp.f32", B32, B32, B32, int_nvvm_add_rp_f>;

def INT_NVVM_ADD_RN_D : F_MATH_2<"add.rn.f64", B64, B64, B64, int_nvvm_add_rn_d>;
def INT_NVVM_ADD_RZ_D : F_MATH_2<"add.rz.f64", B64, B64, B64, int_nvvm_add_rz_d>;
def INT_NVVM_ADD_RM_D : F_MATH_2<"add.rm.f64", B64, B64, B64, int_nvvm_add_rm_d>;
def INT_NVVM_ADD_RP_D : F_MATH_2<"add.rp.f64", B64, B64, B64, int_nvvm_add_rp_d>;

//
// BFIND
//

foreach t = [I32RT, I64RT] in {
  foreach sign = ["s", "u"] in {
    defvar flo_intrin = !cast<Intrinsic>("int_nvvm_flo_" # sign);
    def BFIND_ # sign # t.Size
      : BasicNVPTXInst<(outs B32:$dst), (ins t.RC:$src),
                  "bfind." # sign # t.Size,
                  [(set i32:$dst, (flo_intrin t.Ty:$src, 0))]>;

    def BFIND_SHIFTAMT_ # sign # t.Size
      : BasicNVPTXInst<(outs B32:$dst), (ins t.RC:$src),
                  "bfind.shiftamt." # sign # t.Size,
                  [(set i32:$dst, (flo_intrin t.Ty:$src, -1))]>;
  }
}

//
// szext
//

foreach sign = ["s", "u"] in {
  foreach mode = ["wrap", "clamp"] in {
    defvar ext = !if(!eq(sign, "s"), "sext", "zext");
    defvar intrin = !cast<Intrinsic>("int_nvvm_" # ext # "_" # mode);
    defm SZEXT_ # sign # _ # mode
      : I3Inst<"szext." # mode # "." # sign # "32",
               intrin, I32RT, commutative = false,
               requires = [hasSM<70>, hasPTX<76>]>;
  }
}

//
// BMSK
//

foreach mode = ["wrap", "clamp"] in {
  defvar intrin = !cast<Intrinsic>("int_nvvm_bmsk_" # mode);
  defm BMSK_ # mode
    : I3Inst<"bmsk." # mode # ".b32",
             intrin, I32RT, commutative = false,
             requires = [hasSM<70>, hasPTX<76>]>;
}

//
// Convert
//

def : Pat<(int_nvvm_d2f_rn_ftz f64:$a), (CVT_f32_f64 $a, CvtRN_FTZ)>;
def : Pat<(int_nvvm_d2f_rn f64:$a),     (CVT_f32_f64 $a, CvtRN)>;
def : Pat<(int_nvvm_d2f_rz_ftz f64:$a), (CVT_f32_f64 $a, CvtRZ_FTZ)>;
def : Pat<(int_nvvm_d2f_rz f64:$a),     (CVT_f32_f64 $a, CvtRZ)>;
def : Pat<(int_nvvm_d2f_rm_ftz f64:$a), (CVT_f32_f64 $a, CvtRM_FTZ)>;
def : Pat<(int_nvvm_d2f_rm f64:$a),     (CVT_f32_f64 $a, CvtRM)>;
def : Pat<(int_nvvm_d2f_rp_ftz f64:$a), (CVT_f32_f64 $a, CvtRP_FTZ)>;
def : Pat<(int_nvvm_d2f_rp f64:$a),     (CVT_f32_f64 $a, CvtRP)>;

def : Pat<(int_nvvm_d2i_rn f64:$a), (CVT_s32_f64 $a, CvtRNI)>;
def : Pat<(int_nvvm_d2i_rz f64:$a), (CVT_s32_f64 $a, CvtRZI)>;
def : Pat<(int_nvvm_d2i_rm f64:$a), (CVT_s32_f64 $a, CvtRMI)>;
def : Pat<(int_nvvm_d2i_rp f64:$a), (CVT_s32_f64 $a, CvtRPI)>;

def : Pat<(int_nvvm_d2ui_rn f64:$a), (CVT_u32_f64 $a, CvtRNI)>;
def : Pat<(int_nvvm_d2ui_rz f64:$a), (CVT_u32_f64 $a, CvtRZI)>;
def : Pat<(int_nvvm_d2ui_rm f64:$a), (CVT_u32_f64 $a, CvtRMI)>;
def : Pat<(int_nvvm_d2ui_rp f64:$a), (CVT_u32_f64 $a, CvtRPI)>;

def : Pat<(int_nvvm_i2d_rn i32:$a), (CVT_f64_s32 $a, CvtRN)>;
def : Pat<(int_nvvm_i2d_rz i32:$a), (CVT_f64_s32 $a, CvtRZ)>;
def : Pat<(int_nvvm_i2d_rm i32:$a), (CVT_f64_s32 $a, CvtRM)>;
def : Pat<(int_nvvm_i2d_rp i32:$a), (CVT_f64_s32 $a, CvtRP)>;

def : Pat<(int_nvvm_ui2d_rn i32:$a), (CVT_f64_u32 $a, CvtRN)>;
def : Pat<(int_nvvm_ui2d_rz i32:$a), (CVT_f64_u32 $a, CvtRZ)>;
def : Pat<(int_nvvm_ui2d_rm i32:$a), (CVT_f64_u32 $a, CvtRM)>;
def : Pat<(int_nvvm_ui2d_rp i32:$a), (CVT_f64_u32 $a, CvtRP)>;

def : Pat<(int_nvvm_f2i_rn_ftz f32:$a), (CVT_s32_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2i_rn f32:$a),     (CVT_s32_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_f2i_rz_ftz f32:$a), (CVT_s32_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2i_rz f32:$a),     (CVT_s32_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_f2i_rm_ftz f32:$a), (CVT_s32_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2i_rm f32:$a),     (CVT_s32_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_f2i_rp_ftz f32:$a), (CVT_s32_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2i_rp f32:$a),     (CVT_s32_f32 $a, CvtRPI)>;

def : Pat<(int_nvvm_f2ui_rn_ftz f32:$a), (CVT_u32_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rn f32:$a),     (CVT_u32_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_f2ui_rz_ftz f32:$a), (CVT_u32_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rz f32:$a),     (CVT_u32_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_f2ui_rm_ftz f32:$a), (CVT_u32_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rm f32:$a),     (CVT_u32_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_f2ui_rp_ftz f32:$a), (CVT_u32_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rp f32:$a),     (CVT_u32_f32 $a, CvtRPI)>;

def : Pat<(int_nvvm_i2f_rn i32:$a), (CVT_f32_s32 $a, CvtRN)>;
def : Pat<(int_nvvm_i2f_rz i32:$a), (CVT_f32_s32 $a, CvtRZ)>;
def : Pat<(int_nvvm_i2f_rm i32:$a), (CVT_f32_s32 $a, CvtRM)>;
def : Pat<(int_nvvm_i2f_rp i32:$a), (CVT_f32_s32 $a, CvtRP)>;

def : Pat<(int_nvvm_ui2f_rn i32:$a), (CVT_f32_u32 $a, CvtRN)>;
def : Pat<(int_nvvm_ui2f_rz i32:$a), (CVT_f32_u32 $a, CvtRZ)>;
def : Pat<(int_nvvm_ui2f_rm i32:$a), (CVT_f32_u32 $a, CvtRM)>;
def : Pat<(int_nvvm_ui2f_rp i32:$a), (CVT_f32_u32 $a, CvtRP)>;

def : Pat<(int_nvvm_ff2bf16x2_rn f32:$a, f32:$b),      (CVT_bf16x2_f32 $a, $b, CvtRN)>;
def : Pat<(int_nvvm_ff2bf16x2_rn_relu f32:$a, f32:$b), (CVT_bf16x2_f32 $a, $b, CvtRN_RELU)>;
def : Pat<(int_nvvm_ff2bf16x2_rz f32:$a, f32:$b),      (CVT_bf16x2_f32 $a, $b, CvtRZ)>;
def : Pat<(int_nvvm_ff2bf16x2_rz_relu f32:$a, f32:$b), (CVT_bf16x2_f32 $a, $b, CvtRZ_RELU)>;

def : Pat<(int_nvvm_ff2f16x2_rn f32:$a, f32:$b),      (CVT_f16x2_f32 $a, $b, CvtRN)>;
def : Pat<(int_nvvm_ff2f16x2_rn_relu f32:$a, f32:$b), (CVT_f16x2_f32 $a, $b, CvtRN_RELU)>;
def : Pat<(int_nvvm_ff2f16x2_rz f32:$a, f32:$b),      (CVT_f16x2_f32 $a, $b, CvtRZ)>;
def : Pat<(int_nvvm_ff2f16x2_rz_relu f32:$a, f32:$b), (CVT_f16x2_f32 $a, $b, CvtRZ_RELU)>;

def : Pat<(int_nvvm_f2bf16_rn f32:$a),      (CVT_bf16_f32 $a, CvtRN)>;
def : Pat<(int_nvvm_f2bf16_rn_relu f32:$a), (CVT_bf16_f32 $a, CvtRN_RELU)>;
def : Pat<(int_nvvm_f2bf16_rz f32:$a),      (CVT_bf16_f32 $a, CvtRZ)>;
def : Pat<(int_nvvm_f2bf16_rz_relu f32:$a), (CVT_bf16_f32 $a, CvtRZ_RELU)>;

def : Pat<(int_nvvm_lohi_i2d i32:$a, i32:$b), (V2I32toI64 $a, $b)>;
def : Pat<(int_nvvm_d2i_lo f64:$a), (I64toI32L $a)>;
def : Pat<(int_nvvm_d2i_hi f64:$a), (I64toI32H $a)>;

def : Pat<(int_nvvm_d2i_lo f64:$a), (I64toI32L_Sink $a)>, Requires<[hasPTX<71>]>;
def : Pat<(int_nvvm_d2i_hi f64:$a), (I64toI32H_Sink $a)>, Requires<[hasPTX<71>]>;

def : Pat<(int_nvvm_f2ll_rn_ftz f32:$a), (CVT_s64_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rn f32:$a),     (CVT_s64_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_f2ll_rz_ftz f32:$a), (CVT_s64_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rz f32:$a),     (CVT_s64_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_f2ll_rm_ftz f32:$a), (CVT_s64_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rm f32:$a),     (CVT_s64_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_f2ll_rp_ftz f32:$a), (CVT_s64_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rp f32:$a),     (CVT_s64_f32 $a, CvtRPI)>;

def : Pat<(int_nvvm_f2ull_rn_ftz f32:$a), (CVT_u64_f32 $a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rn f32:$a),     (CVT_u64_f32 $a, CvtRNI)>;
def : Pat<(int_nvvm_f2ull_rz_ftz f32:$a), (CVT_u64_f32 $a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rz f32:$a),     (CVT_u64_f32 $a, CvtRZI)>;
def : Pat<(int_nvvm_f2ull_rm_ftz f32:$a), (CVT_u64_f32 $a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rm f32:$a),     (CVT_u64_f32 $a, CvtRMI)>;
def : Pat<(int_nvvm_f2ull_rp_ftz f32:$a), (CVT_u64_f32 $a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rp f32:$a),     (CVT_u64_f32 $a, CvtRPI)>;

def : Pat<(int_nvvm_d2ll_rn f64:$a), (CVT_s64_f64 $a, CvtRNI)>;
def : Pat<(int_nvvm_d2ll_rz f64:$a), (CVT_s64_f64 $a, CvtRZI)>;
def : Pat<(int_nvvm_d2ll_rm f64:$a), (CVT_s64_f64 $a, CvtRMI)>;
def : Pat<(int_nvvm_d2ll_rp f64:$a), (CVT_s64_f64 $a, CvtRPI)>;

def : Pat<(int_nvvm_d2ull_rn f64:$a), (CVT_u64_f64 $a, CvtRNI)>;
def : Pat<(int_nvvm_d2ull_rz f64:$a), (CVT_u64_f64 $a, CvtRZI)>;
def : Pat<(int_nvvm_d2ull_rm f64:$a), (CVT_u64_f64 $a, CvtRMI)>;
def : Pat<(int_nvvm_d2ull_rp f64:$a), (CVT_u64_f64 $a, CvtRPI)>;

def : Pat<(int_nvvm_ll2f_rn i64:$a), (CVT_f32_s64 $a, CvtRN)>;
def : Pat<(int_nvvm_ll2f_rz i64:$a), (CVT_f32_s64 $a, CvtRZ)>;
def : Pat<(int_nvvm_ll2f_rm i64:$a), (CVT_f32_s64 $a, CvtRM)>;
def : Pat<(int_nvvm_ll2f_rp i64:$a), (CVT_f32_s64 $a, CvtRP)>;

def : Pat<(int_nvvm_ull2f_rn i64:$a), (CVT_f32_u64 $a, CvtRN)>;
def : Pat<(int_nvvm_ull2f_rz i64:$a), (CVT_f32_u64 $a, CvtRZ)>;
def : Pat<(int_nvvm_ull2f_rm i64:$a), (CVT_f32_u64 $a, CvtRM)>;
def : Pat<(int_nvvm_ull2f_rp i64:$a), (CVT_f32_u64 $a, CvtRP)>;

def : Pat<(int_nvvm_ll2d_rn i64:$a), (CVT_f64_s64 $a, CvtRN)>;
def : Pat<(int_nvvm_ll2d_rz i64:$a), (CVT_f64_s64 $a, CvtRZ)>;
def : Pat<(int_nvvm_ll2d_rm i64:$a), (CVT_f64_s64 $a, CvtRM)>;
def : Pat<(int_nvvm_ll2d_rp i64:$a), (CVT_f64_s64 $a, CvtRP)>;

def : Pat<(int_nvvm_ull2d_rn i64:$a), (CVT_f64_u64 $a, CvtRN)>;
def : Pat<(int_nvvm_ull2d_rz i64:$a), (CVT_f64_u64 $a, CvtRZ)>;
def : Pat<(int_nvvm_ull2d_rm i64:$a), (CVT_f64_u64 $a, CvtRM)>;
def : Pat<(int_nvvm_ull2d_rp i64:$a), (CVT_f64_u64 $a, CvtRP)>;


def : Pat<(int_nvvm_f2h_rn_ftz f32:$a), (CVT_f16_f32 $a, CvtRN_FTZ)>;
def : Pat<(int_nvvm_f2h_rn f32:$a), (CVT_f16_f32 $a, CvtRN)>;

def : Pat<(int_nvvm_ff_to_e4m3x2_rn f32:$a, f32:$b),
          (CVT_e4m3x2_f32 $a, $b, CvtRN)>;
def : Pat<(int_nvvm_ff_to_e4m3x2_rn_relu f32:$a, f32:$b),
          (CVT_e4m3x2_f32 $a, $b, CvtRN_RELU)>;
def : Pat<(int_nvvm_ff_to_e5m2x2_rn f32:$a, f32:$b),
          (CVT_e5m2x2_f32 $a, $b, CvtRN)>;
def : Pat<(int_nvvm_ff_to_e5m2x2_rn_relu f32:$a, f32:$b),
          (CVT_e5m2x2_f32 $a, $b, CvtRN_RELU)>;

def : Pat<(int_nvvm_f16x2_to_e4m3x2_rn v2f16:$a),
          (CVT_e4m3x2_f16x2 $a, CvtRN)>;
def : Pat<(int_nvvm_f16x2_to_e4m3x2_rn_relu v2f16:$a),
          (CVT_e4m3x2_f16x2 $a, CvtRN_RELU)>;
def : Pat<(int_nvvm_f16x2_to_e5m2x2_rn v2f16:$a),
          (CVT_e5m2x2_f16x2 $a, CvtRN)>;
def : Pat<(int_nvvm_f16x2_to_e5m2x2_rn_relu v2f16:$a),
          (CVT_e5m2x2_f16x2 $a, CvtRN_RELU)>;

def : Pat<(int_nvvm_e4m3x2_to_f16x2_rn i16:$a),
          (CVT_f16x2_e4m3x2 $a, CvtRN)>;
def : Pat<(int_nvvm_e4m3x2_to_f16x2_rn_relu i16:$a),
          (CVT_f16x2_e4m3x2 $a, CvtRN_RELU)>;
def : Pat<(int_nvvm_e5m2x2_to_f16x2_rn i16:$a),
          (CVT_f16x2_e5m2x2 $a, CvtRN)>;
def : Pat<(int_nvvm_e5m2x2_to_f16x2_rn_relu i16:$a),
          (CVT_f16x2_e5m2x2 $a, CvtRN_RELU)>;

let Predicates = [hasPTX<86>, hasSM<100>, hasArchAccelFeatures] in {
  def : Pat<(int_nvvm_ff_to_e2m3x2_rn_satfinite f32:$a, f32:$b),
            (CVT_e2m3x2_f32_sf $a, $b, CvtRN)>;
  def : Pat<(int_nvvm_ff_to_e2m3x2_rn_relu_satfinite f32:$a, f32:$b),
            (CVT_e2m3x2_f32_sf $a, $b, CvtRN_RELU)>;
  def : Pat<(int_nvvm_ff_to_e3m2x2_rn_satfinite f32:$a, f32:$b),
            (CVT_e3m2x2_f32_sf $a, $b, CvtRN)>;
  def : Pat<(int_nvvm_ff_to_e3m2x2_rn_relu_satfinite f32:$a, f32:$b),
            (CVT_e3m2x2_f32_sf $a, $b, CvtRN_RELU)>;

  def : Pat<(int_nvvm_e2m3x2_to_f16x2_rn i16:$a),
            (CVT_f16x2_e2m3x2 $a, CvtRN)>;
  def : Pat<(int_nvvm_e2m3x2_to_f16x2_rn_relu i16:$a),
            (CVT_f16x2_e2m3x2 $a, CvtRN_RELU)>;
  def : Pat<(int_nvvm_e3m2x2_to_f16x2_rn i16:$a),
            (CVT_f16x2_e3m2x2 $a, CvtRN)>;
  def : Pat<(int_nvvm_e3m2x2_to_f16x2_rn_relu i16:$a),
            (CVT_f16x2_e3m2x2 $a, CvtRN_RELU)>;

  def : Pat<(int_nvvm_ff_to_e2m1x2_rn_satfinite f32:$a, f32:$b),
            (CVT_e2m1x2_f32_sf $a, $b, CvtRN)>;
  def : Pat<(int_nvvm_ff_to_e2m1x2_rn_relu_satfinite f32:$a, f32:$b),
            (CVT_e2m1x2_f32_sf $a, $b, CvtRN_RELU)>;

  def : Pat<(int_nvvm_e2m1x2_to_f16x2_rn i16:$a),
            (CVT_f16x2_e2m1x2 $a, CvtRN)>;
  def : Pat<(int_nvvm_e2m1x2_to_f16x2_rn_relu i16:$a),
            (CVT_f16x2_e2m1x2 $a, CvtRN_RELU)>;

  def : Pat<(int_nvvm_ff_to_ue8m0x2_rz f32:$a, f32:$b),
            (CVT_ue8m0x2_f32 $a, $b, CvtRZ)>;
  def : Pat<(int_nvvm_ff_to_ue8m0x2_rz_satfinite f32:$a, f32:$b),
            (CVT_ue8m0x2_f32_sf $a, $b, CvtRZ)>;
  def : Pat<(int_nvvm_ff_to_ue8m0x2_rp f32:$a, f32:$b),
            (CVT_ue8m0x2_f32 $a, $b, CvtRP)>;
  def : Pat<(int_nvvm_ff_to_ue8m0x2_rp_satfinite f32:$a, f32:$b),
            (CVT_ue8m0x2_f32_sf $a, $b, CvtRP)>;

  def : Pat<(int_nvvm_bf16x2_to_ue8m0x2_rz v2bf16:$a),
            (CVT_ue8m0x2_bf16x2 $a, CvtRZ)>;
  def : Pat<(int_nvvm_bf16x2_to_ue8m0x2_rz_satfinite v2bf16:$a),
            (CVT_ue8m0x2_bf16x2_sf $a, CvtRZ)>;
  def : Pat<(int_nvvm_bf16x2_to_ue8m0x2_rp v2bf16:$a),
            (CVT_ue8m0x2_bf16x2 $a, CvtRP)>;
  def : Pat<(int_nvvm_bf16x2_to_ue8m0x2_rp_satfinite v2bf16:$a),
            (CVT_ue8m0x2_bf16x2_sf $a, CvtRP)>;

  def : Pat<(int_nvvm_ue8m0x2_to_bf16x2 i16:$a),
            (CVT_bf16x2_ue8m0x2 $a)>;
}

//
// FNS
//

class INT_FNS_MBO<dag ins, dag Operands>
  : BasicNVPTXInst<(outs B32:$dst), ins,
               "fns.b32",
               [(set i32:$dst, Operands)]>,
    Requires<[hasPTX<60>, hasSM<30>]>;

def INT_FNS_rrr : INT_FNS_MBO<(ins B32:$mask, B32:$base, B32:$offset),
                     (int_nvvm_fns i32:$mask, i32:$base, i32:$offset)>;
def INT_FNS_rri : INT_FNS_MBO<(ins B32:$mask, B32:$base,    i32imm:$offset),
                     (int_nvvm_fns i32:$mask, i32:$base,       imm:$offset)>;
def INT_FNS_rir : INT_FNS_MBO<(ins B32:$mask,    i32imm:$base, B32:$offset),
                     (int_nvvm_fns i32:$mask,       imm:$base, i32:$offset)>;
def INT_FNS_rii : INT_FNS_MBO<(ins B32:$mask,    i32imm:$base,    i32imm:$offset),
                     (int_nvvm_fns i32:$mask,       imm:$base,       imm:$offset)>;
def INT_FNS_irr : INT_FNS_MBO<(ins    i32imm:$mask, B32:$base, B32:$offset),
                     (int_nvvm_fns       imm:$mask, i32:$base, i32:$offset)>;
def INT_FNS_iri : INT_FNS_MBO<(ins    i32imm:$mask, B32:$base,    i32imm:$offset),
                     (int_nvvm_fns       imm:$mask, i32:$base,       imm:$offset)>;
def INT_FNS_iir : INT_FNS_MBO<(ins    i32imm:$mask,    i32imm:$base, B32:$offset),
                     (int_nvvm_fns       imm:$mask,       imm:$base, i32:$offset)>;
def INT_FNS_iii : INT_FNS_MBO<(ins    i32imm:$mask,    i32imm:$base,    i32imm:$offset),
                     (int_nvvm_fns       imm:$mask,       imm:$base,       imm:$offset)>;

//-----------------------------------
// Atomic Functions
//-----------------------------------

class ATOMIC_GLOBAL_CHK <dag frag>
 : PatFrag<!setdagop(frag, ops), frag, AS_match.global>;
class ATOMIC_SHARED_CHK <dag frag>
 : PatFrag<!setdagop(frag, ops), frag, AS_match.shared>;
class ATOMIC_SHARED_CLUSTER_CHK <dag frag>
 : PatFrag<!setdagop(frag, ops), frag, AS_match.shared_cluster>;
class ATOMIC_GENERIC_CHK <dag frag>
 : PatFrag<!setdagop(frag, ops), frag, AS_match.generic>;

multiclass F_ATOMIC_2<RegTyInfo t, string sem_str, string as_str, string op_str,
                      SDPatternOperator op, list<Predicate> preds> {
  defvar asm_str = "atom" # sem_str # as_str # "." # op_str;
  let mayLoad = 1, mayStore = 1, hasSideEffects = 1 in {
    def r : BasicNVPTXInst<(outs t.RC:$dst), (ins ADDR:$addr, t.RC:$b),
      asm_str,
      [(set t.Ty:$dst, (op addr:$addr, t.Ty:$b))]>,
    Requires<preds>;
    if t.SupportsImm then
      def i : BasicNVPTXInst<(outs t.RC:$dst), (ins ADDR:$addr, t.Imm:$b),
        asm_str,
        [(set t.Ty:$dst, (op addr:$addr, (t.Ty t.ImmNode:$b)))]>,
      Requires<preds>;
  }
}

multiclass F_ATOMIC_3<RegTyInfo t, string op_str, SDPatternOperator op, SDNode atomic> {
  defvar asm_str = "atom${sem:sem}${scope:scope}${addsp:addsp}" # op_str;

  let mayLoad = 1, mayStore = 1, hasSideEffects = 1 in {
    def _rr : BasicFlagsNVPTXInst<(outs t.RC:$dst),
      (ins ADDR:$addr, t.RC:$b, t.RC:$c),
      (ins AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp),
      asm_str>;

    def _ir : BasicFlagsNVPTXInst<(outs t.RC:$dst),
      (ins ADDR:$addr, t.Imm:$b, t.RC:$c),
      (ins AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp),
      asm_str>;

    def _ri : BasicFlagsNVPTXInst<(outs t.RC:$dst),
      (ins ADDR:$addr, t.RC:$b, t.Imm:$c),
      (ins AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp),
      asm_str>;

    def _ii : BasicFlagsNVPTXInst<(outs t.RC:$dst),
      (ins ADDR:$addr, t.Imm:$b, t.Imm:$c),
      (ins AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp),
      asm_str>;
  }

  defvar GetSem = SDNodeXForm<atomic, [{
    return getI32Imm(getMemOrder(cast<MemSDNode>(N)), SDLoc(N));
  }]>;

  defvar GetScope = SDNodeXForm<atomic, [{
    return getI32Imm(getAtomicScope(cast<MemSDNode>(N)), SDLoc(N));
  }]>;

  defvar GetAddSp = SDNodeXForm<atomic, [{
    return getI32Imm(getAddrSpace(cast<MemSDNode>(N)), SDLoc(N));
  }]>;

  def : Pat<(op:$this addr:$addr, t.Ty:$b, t.Ty:$c),
        (!cast<Instruction>(NAME # _rr) ADDR:$addr, t.Ty:$b, t.Ty:$c, (GetSem $this), (GetScope $this), (GetAddSp $this))>;

  def : Pat<(op:$this addr:$addr, (t.Ty t.ImmNode:$b), t.Ty:$c),
        (!cast<Instruction>(NAME # _ir) ADDR:$addr, (t.Ty t.ImmNode:$b), t.Ty:$c, (GetSem $this), (GetScope $this), (GetAddSp $this))>;

  def : Pat<(op:$this addr:$addr, t.Ty:$b, (t.Ty t.ImmNode:$c)),
        (!cast<Instruction>(NAME # _ri) ADDR:$addr, t.Ty:$b, (t.Ty t.ImmNode:$c), (GetSem $this), (GetScope $this), (GetAddSp $this))>;

  def : Pat<(op:$this addr:$addr, (t.Ty t.ImmNode:$b), (t.Ty t.ImmNode:$c)),
        (!cast<Instruction>(NAME # _ii) ADDR:$addr, (t.Ty t.ImmNode:$b), (t.Ty t.ImmNode:$c), (GetSem $this), (GetScope $this), (GetAddSp $this))>;
}

multiclass F_ATOMIC_2_AS<RegTyInfo t, SDPatternOperator frag, string op_str, list<Predicate> preds = []> {
  defvar frag_pat = (frag node:$a, node:$b);
  defm _G : F_ATOMIC_2<t, "", ".global", op_str, ATOMIC_GLOBAL_CHK<frag_pat>, preds>;
  defm _S : F_ATOMIC_2<t, "", ".shared", op_str, ATOMIC_SHARED_CHK<frag_pat>, preds>;
  defm _S_C : F_ATOMIC_2<t, "", ".shared::cluster", op_str, ATOMIC_SHARED_CLUSTER_CHK<frag_pat>, !listconcat([hasClusters], preds)>;
  defm _GEN : F_ATOMIC_2<t, "", "", op_str, ATOMIC_GENERIC_CHK<frag_pat>, preds>;
}

// atom_add
defm INT_PTX_ATOM_ADD_32 : F_ATOMIC_2_AS<I32RT, atomic_load_add_i32, "add.u32">;
defm INT_PTX_ATOM_ADD_64 : F_ATOMIC_2_AS<I64RT, atomic_load_add_i64, "add.u64">;

defm INT_PTX_ATOM_ADD_F16  : F_ATOMIC_2_AS<F16RT, atomic_load_fadd, "add.noftz.f16", [hasSM<70>, hasPTX<63>]>;
defm INT_PTX_ATOM_ADD_BF16 : F_ATOMIC_2_AS<BF16RT, atomic_load_fadd, "add.noftz.bf16", [hasSM<90>, hasPTX<78>]>;
defm INT_PTX_ATOM_ADD_F32  : F_ATOMIC_2_AS<F32RT, atomic_load_fadd, "add.f32">;
defm INT_PTX_ATOM_ADD_F64  : F_ATOMIC_2_AS<F64RT, atomic_load_fadd, "add.f64", [hasAtomAddF64]>;

// atom_swap
defm INT_PTX_ATOM_SWAP_32 : F_ATOMIC_2_AS<I32RT, atomic_swap_i32, "exch.b32">;
defm INT_PTX_ATOM_SWAP_64 : F_ATOMIC_2_AS<I64RT, atomic_swap_i64, "exch.b64">;

// atom_max
defm INT_PTX_ATOMIC_MAX_32 : F_ATOMIC_2_AS<I32RT, atomic_load_max_i32, "max.s32">;
defm INT_PTX_ATOMIC_MAX_64 : F_ATOMIC_2_AS<I64RT, atomic_load_max_i64, "max.s64", [hasSM<32>]>;
defm INT_PTX_ATOMIC_UMAX_32 : F_ATOMIC_2_AS<I32RT, atomic_load_umax_i32, "max.u32">;
defm INT_PTX_ATOMIC_UMAX_64 : F_ATOMIC_2_AS<I64RT, atomic_load_umax_i64, "max.u64", [hasSM<32>]>;

// atom_min
defm INT_PTX_ATOMIC_MIN_32 : F_ATOMIC_2_AS<I32RT, atomic_load_min_i32, "min.s32">;
defm INT_PTX_ATOMIC_MIN_64 : F_ATOMIC_2_AS<I64RT, atomic_load_min_i64, "min.s64", [hasSM<32>]>;
defm INT_PTX_ATOMIC_UMIN_32 : F_ATOMIC_2_AS<I32RT, atomic_load_umin_i32, "min.u32">;
defm INT_PTX_ATOMIC_UMIN_64 : F_ATOMIC_2_AS<I64RT, atomic_load_umin_i64, "min.u64", [hasSM<32>]>;

// atom_inc  atom_dec
defm INT_PTX_ATOM_INC_32 : F_ATOMIC_2_AS<I32RT, atomic_load_uinc_wrap_i32, "inc.u32">;
defm INT_PTX_ATOM_DEC_32 : F_ATOMIC_2_AS<I32RT, atomic_load_udec_wrap_i32, "dec.u32">;

// atom_and
defm INT_PTX_ATOM_AND_32 : F_ATOMIC_2_AS<I32RT, atomic_load_and_i32, "and.b32">;
defm INT_PTX_ATOM_AND_64 : F_ATOMIC_2_AS<I64RT, atomic_load_and_i64, "and.b64", [hasSM<32>]>;

// atom_or
defm INT_PTX_ATOM_OR_32 : F_ATOMIC_2_AS<I32RT, atomic_load_or_i32, "or.b32">;
defm INT_PTX_ATOM_OR_64 : F_ATOMIC_2_AS<I64RT, atomic_load_or_i64, "or.b64", [hasSM<32>]>;

// atom_xor
defm INT_PTX_ATOM_XOR_32 : F_ATOMIC_2_AS<I32RT, atomic_load_xor_i32, "xor.b32">;
defm INT_PTX_ATOM_XOR_64 : F_ATOMIC_2_AS<I64RT, atomic_load_xor_i64, "xor.b64", [hasSM<32>]>;


// Define atom.cas for all combinations of size x addrspace x memory order
// supported in PTX *and* on the hardware.
foreach t = [I16RT, I32RT, I64RT] in {
    defvar atomic_cmp_swap_pat = !cast<PatFrag>("atomic_cmp_swap_i"#t.Size);
    defm INT_PTX_ATOM_CAS_#t.Size
     : F_ATOMIC_3<t, ".cas.b"#t.Size, atomic_cmp_swap_pat, atomic_cmp_swap>;
}

// Support for scoped atomic operations.  Matches
// int_nvvm_atomic_{op}_{space}_{type}_{scope}
// and converts it into the appropriate instruction.
// NOTE: not all possible combinations are implemented
//  'space' is limited to generic as it's the only one needed to support CUDA.
//  'scope' = 'gpu' is default and is handled by regular atomic instructions.

// Define instruction variants for all addressing modes.

// Constructs intrinsic name and instruction asm strings.
multiclass ATOM2N_impl<string OpStr, string IntTypeStr, string TypeStr,
                       string ScopeStr, string SpaceStr,
                       RegTyInfo t, list<Predicate> Preds> {
  defm "" : F_ATOMIC_2<t,
                       as_str = !if(!eq(ScopeStr, "gpu"), "", "." # ScopeStr),
                       sem_str = !if(!eq(SpaceStr, "gen"), "", "." # SpaceStr),
                       op_str = OpStr # "." # TypeStr,
                       op = !cast<Intrinsic>(
                              "int_nvvm_atomic_" # OpStr
                              # "_" # SpaceStr # "_" # IntTypeStr
                              # !if(!empty(ScopeStr), "", "_" # ScopeStr)),
                       preds = Preds>;
}

// Constructs variants for different scopes of atomic op.
multiclass ATOM2S_impl<string OpStr, string IntTypeStr, string TypeStr,
                       RegTyInfo t, list<Predicate> Preds> {
   // .gpu scope is default and is currently covered by existing
   // atomics w/o explicitly specified scope.
  foreach scope = ["cta", "sys"] in {
    // For now we only need variants for generic space pointers.
    foreach space = ["gen"] in {
      defm _#scope#space : ATOM2N_impl<OpStr, IntTypeStr, TypeStr, scope, space,
                         t, !listconcat(Preds, [hasAtomScope])>;
    }
  }
}

multiclass F_ATOMIC_3_INTRINSIC_PATTERN<RegTyInfo t, string OpStr, string InstructionName> {
  foreach scope = ["cta", "sys"] in {
    foreach space = ["gen"] in {
      defvar intrinsic = !cast<SDPatternOperator>("int_nvvm_atomic_" # OpStr # "_" # space # "_i_" # scope);
      def : Pat<(t.Ty (intrinsic addr:$addr, t.Ty:$b, t.Ty:$c)),
            (!cast<Instruction>(InstructionName # "_rr") ADDR:$addr, t.Ty:$b, t.Ty:$c, Ordering_not_atomic, !cast<PatLeaf>("Scope_" # scope), !cast<PatLeaf>("AddrSpace_" # space))>;

      def : Pat<(t.Ty (intrinsic addr:$addr, (t.Ty t.ImmNode:$b), t.Ty:$c)),
            (!cast<Instruction>(InstructionName # "_ir") ADDR:$addr, (t.Ty t.ImmNode:$b), t.Ty:$c, Ordering_not_atomic, !cast<PatLeaf>("Scope_" # scope), !cast<PatLeaf>("AddrSpace_" # space))>;

      def : Pat<(t.Ty (intrinsic addr:$addr, t.Ty:$b, (t.Ty t.ImmNode:$c))),
            (!cast<Instruction>(InstructionName # "_ri") ADDR:$addr, t.Ty:$b, (t.Ty t.ImmNode:$c), Ordering_not_atomic, !cast<PatLeaf>("Scope_" # scope), !cast<PatLeaf>("AddrSpace_" # space))>;

      def : Pat<(t.Ty (intrinsic addr:$addr, (t.Ty t.ImmNode:$b), (t.Ty t.ImmNode:$c))),
            (!cast<Instruction>(InstructionName # "_ii") ADDR:$addr, (t.Ty t.ImmNode:$b), (t.Ty t.ImmNode:$c), Ordering_not_atomic, !cast<PatLeaf>("Scope_" # scope), !cast<PatLeaf>("AddrSpace_" # space))>;
    }
  }
}

// atom.add
multiclass ATOM2_add_impl<string OpStr> {
  defm _s32  : ATOM2S_impl<OpStr, "i", "s32", I32RT, []>;
  defm _u32  : ATOM2S_impl<OpStr, "i", "u32", I32RT, []>;
  defm _u64  : ATOM2S_impl<OpStr, "i", "u64", I64RT, []>;
  defm _bf16 : ATOM2S_impl<OpStr, "f", "bf16", BF16RT, [hasSM<90>, hasPTX<78>]>;
  defm _f16  : ATOM2S_impl<OpStr, "f", "f16", F16RT, []>;
  defm _f32  : ATOM2S_impl<OpStr, "f", "f32", F32RT, []>;
  defm _f64  : ATOM2S_impl<OpStr, "f", "f64", F64RT, []>;
}

// atom.{and,or,xor}
multiclass ATOM2_bitwise_impl<string OpStr> {
  defm _b32 : ATOM2S_impl<OpStr, "i", "b32", I32RT, []>;
  defm _b64 : ATOM2S_impl<OpStr, "i", "b64", I64RT, [hasAtomBitwise64]>;
}

// atom.exch
multiclass ATOM2_exch_impl<string OpStr> {
  defm _b32 : ATOM2S_impl<OpStr, "i", "b32", I32RT, []>;
  defm _b64 : ATOM2S_impl<OpStr, "i", "b64", I64RT, []>;
}

// atom.{min,max}
multiclass ATOM2_minmax_impl<string OpStr> {
  defm _s32 : ATOM2S_impl<OpStr, "i", "s32", I32RT, []>;
  defm _u32 : ATOM2S_impl<OpStr, "i", "u32", I32RT, []>;
  defm _s64 : ATOM2S_impl<OpStr, "i", "s64", I64RT, [hasAtomMinMax64]>;
  defm _u64 : ATOM2S_impl<OpStr, "i", "u64", I64RT, [hasAtomMinMax64]>;
}

// atom.{inc,dec}
multiclass ATOM2_incdec_impl<string OpStr> {
  defm _u32 : ATOM2S_impl<OpStr, "i", "u32", I32RT, []>;
}

// atom.cas
multiclass ATOM3_cas_impl<string OpStr> {
  defm _b16 : F_ATOMIC_3_INTRINSIC_PATTERN<I16RT, OpStr, "INT_PTX_ATOM_CAS_16">;
  defm _b32 : F_ATOMIC_3_INTRINSIC_PATTERN<I32RT, OpStr, "INT_PTX_ATOM_CAS_32">;
  defm _b64 : F_ATOMIC_3_INTRINSIC_PATTERN<I64RT, OpStr, "INT_PTX_ATOM_CAS_64">;
}

defm INT_PTX_SATOM_ADD  : ATOM2_add_impl<"add">;
defm INT_PTX_SATOM_AND  : ATOM2_bitwise_impl<"and">;
defm INT_PTX_SATOM_CAS  : ATOM3_cas_impl<"cas">;
defm INT_PTX_SATOM_DEC  : ATOM2_incdec_impl<"dec">;
defm INT_PTX_SATOM_EXCH : ATOM2_exch_impl<"exch">;
defm INT_PTX_SATOM_INC  : ATOM2_incdec_impl<"inc">;
defm INT_PTX_SATOM_MAX  : ATOM2_minmax_impl<"max">;
defm INT_PTX_SATOM_MIN  : ATOM2_minmax_impl<"min">;
defm INT_PTX_SATOM_OR   : ATOM2_bitwise_impl<"or">;
defm INT_PTX_SATOM_XOR  : ATOM2_bitwise_impl<"xor">;

// atom.*.b128

let mayLoad = true, mayStore = true, hasSideEffects = true,
    Predicates = [hasAtomSwap128] in {
  def ATOM_CAS_B128 :
    NVPTXInst<
        (outs B64:$dst0, B64:$dst1),
        (ins ADDR:$addr, B64:$cmp0, B64:$cmp1, B64:$swap0, B64:$swap1,
             AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp),
        "{{\n\t"
        ".reg .b128 cmp, swap, dst;\n\t"
        "mov.b128 cmp, {$cmp0, $cmp1};\n\t"
        "mov.b128 swap, {$swap0, $swap1};\n\t"
        "atom${sem:sem}${scope:scope}${addsp:addsp}.cas.b128 dst, [$addr], cmp, swap;\n\t"
        "mov.b128 {$dst0, $dst1}, dst;\n\t"
        "}}">;

  def ATOM_EXCH_B128 :
    NVPTXInst<
        (outs B64:$dst0, B64:$dst1),
        (ins ADDR:$addr, B64:$amt0, B64:$amt1,
             AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp),
        "{{\n\t"
        ".reg .b128 amt, dst;\n\t"
        "mov.b128 amt, {$amt0, $amt1};\n\t"
        "atom${sem:sem}${scope:scope}${addsp:addsp}.exch.b128 dst, [$addr], amt;\n\t"
        "mov.b128 {$dst0, $dst1}, dst;\n\t"
        "}}">;
}


//-----------------------------------
// Support for ldu on sm_20 or later
//-----------------------------------

// Don't annotate ldu instructions as mayLoad, as they load from memory that is
// read-only in a kernel.

// Scalar

class LDU_G<NVPTXRegClass regclass>
  :  NVPTXInst<(outs regclass:$result), (ins i32imm:$fromWidth, ADDR:$src),
               "ldu.global.b$fromWidth \t$result, [$src];">;

def LDU_GLOBAL_i16 : LDU_G<B16>;
def LDU_GLOBAL_i32 : LDU_G<B32>;
def LDU_GLOBAL_i64 : LDU_G<B64>;

// vector

// Elementized vector ldu
class VLDU_G_ELE_V2<NVPTXRegClass regclass>
  : NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
              (ins i32imm:$fromWidth, ADDR:$src),
              "ldu.global.v2.b$fromWidth \t{{$dst1, $dst2}}, [$src];">;


class VLDU_G_ELE_V4<NVPTXRegClass regclass>
  : NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3, regclass:$dst4),
              (ins i32imm:$fromWidth, ADDR:$src),
               "ldu.global.v4.b$fromWidth \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];">;


def LDU_GLOBAL_v2i16 : VLDU_G_ELE_V2<B16>;
def LDU_GLOBAL_v2i32 : VLDU_G_ELE_V2<B32>;
def LDU_GLOBAL_v2i64 : VLDU_G_ELE_V2<B64>;

def LDU_GLOBAL_v4i16 : VLDU_G_ELE_V4<B16>;
def LDU_GLOBAL_v4i32 : VLDU_G_ELE_V4<B32>;


//-----------------------------------
// Support for ldg on sm_35 or later
//-----------------------------------

// Don't annotate ld.global.nc as mayLoad, because these loads go through the
// non-coherent texture cache, and therefore the values read must be read-only
// during the lifetime of the kernel.

class LDG_G<NVPTXRegClass regclass>
  : NVPTXInst<(outs regclass:$result), (ins AtomicCode:$Sign, i32imm:$fromWidth, ADDR:$src),
               "ld.global.nc.${Sign:sign}$fromWidth \t$result, [$src];">;

def LD_GLOBAL_NC_i16 : LDG_G<B16>;
def LD_GLOBAL_NC_i32 : LDG_G<B32>;
def LD_GLOBAL_NC_i64 : LDG_G<B64>;

// vector

// Elementized vector ldg
class VLDG_G_ELE_V2<NVPTXRegClass regclass> :
  NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
            (ins AtomicCode:$Sign, i32imm:$fromWidth, ADDR:$src),
            "ld.global.nc.v2.${Sign:sign}$fromWidth \t{{$dst1, $dst2}}, [$src];">;


class VLDG_G_ELE_V4<NVPTXRegClass regclass> :
  NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3, regclass:$dst4), 
            (ins AtomicCode:$Sign, i32imm:$fromWidth, ADDR:$src),
            "ld.global.nc.v4.${Sign:sign}$fromWidth \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];">;

class VLDG_G_ELE_V8<NVPTXRegClass regclass> :
  NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3, regclass:$dst4,
                  regclass:$dst5, regclass:$dst6, regclass:$dst7, regclass:$dst8),
             (ins AtomicCode:$Sign, i32imm:$fromWidth, ADDR:$src),
             "ld.global.nc.v8.${Sign:sign}$fromWidth \t{{$dst1, $dst2, $dst3, $dst4, $dst5, $dst6, $dst7, $dst8}}, [$src];">;

// FIXME: 8-bit LDG should be fixed once LDG/LDU nodes are made into proper loads.
def LD_GLOBAL_NC_v2i16 : VLDG_G_ELE_V2<B16>;
def LD_GLOBAL_NC_v2i32 : VLDG_G_ELE_V2<B32>;
def LD_GLOBAL_NC_v2i64 : VLDG_G_ELE_V2<B64>;

def LD_GLOBAL_NC_v4i16 : VLDG_G_ELE_V4<B16>;
def LD_GLOBAL_NC_v4i32 : VLDG_G_ELE_V4<B32>;

def LD_GLOBAL_NC_v4i64 : VLDG_G_ELE_V4<B64>;
def LD_GLOBAL_NC_v8i32 : VLDG_G_ELE_V8<B32>;

multiclass NG_TO_G<string Str, bit Supports32 = 1, list<Predicate> Preds = []> {
  if Supports32 then
    def "" : BasicNVPTXInst<(outs B32:$result), (ins B32:$src),
             "cvta." # Str # ".u32">, Requires<Preds>;
  
  def _64 : BasicNVPTXInst<(outs B64:$result), (ins B64:$src),
              "cvta." # Str # ".u64">, Requires<Preds>;
}

multiclass G_TO_NG<string Str, bit Supports32 = 1, list<Predicate> Preds = []> {
  if Supports32 then
    def "" : BasicNVPTXInst<(outs B32:$result), (ins B32:$src),
            "cvta.to." # Str # ".u32">, Requires<Preds>;
  
  def _64 : BasicNVPTXInst<(outs B64:$result), (ins B64:$src),
            "cvta.to." # Str # ".u64">, Requires<Preds>;
}

foreach space = ["local", "shared", "global", "const", "param"] in {
  defm cvta_#space : NG_TO_G<space>;
  defm cvta_to_#space : G_TO_NG<space>;
}

defm cvta_shared_cluster : NG_TO_G<"shared::cluster", false, [hasClusters]>;
defm cvta_to_shared_cluster : G_TO_NG<"shared::cluster", false, [hasClusters]>;


// nvvm.move intrinsicc
def nvvm_move_i16 : BasicNVPTXInst<(outs B16:$r), (ins B16:$s),
                             "mov.b16",
                             [(set i16:$r,
                               (int_nvvm_move_i16 i16:$s))]>;
def nvvm_move_i32 : BasicNVPTXInst<(outs B32:$r), (ins B32:$s),
                             "mov.b32",
                             [(set i32:$r,
                               (int_nvvm_move_i32 i32:$s))]>;
def nvvm_move_i64 : BasicNVPTXInst<(outs B64:$r), (ins B64:$s),
                             "mov.b64",
                             [(set i64:$r,
                               (int_nvvm_move_i64 i64:$s))]>;
def nvvm_move_float : BasicNVPTXInst<(outs B32:$r), (ins B32:$s),
                             "mov.f32",
                             [(set f32:$r,
                               (int_nvvm_move_float f32:$s))]>;
def nvvm_move_double : BasicNVPTXInst<(outs B64:$r), (ins B64:$s),
                             "mov.f64",
                             [(set f64:$r,
                               (int_nvvm_move_double f64:$s))]>;
def nvvm_move_ptr32 : BasicNVPTXInst<(outs B32:$r), (ins B32:$s),
                             "mov.u32",
                             [(set i32:$r,
                               (int_nvvm_move_ptr i32:$s))]>;
def nvvm_move_ptr64 : BasicNVPTXInst<(outs B64:$r), (ins B64:$s),
                             "mov.u64",
                             [(set i64:$r,
                               (int_nvvm_move_ptr i64:$s))]>;

// @TODO: Are these actually needed, or will we always just see symbols
// copied to registers first?
/*def nvvm_move_sym32 : NVPTXInst<(outs B32:$r), (ins ADDR_base:$s),
                             "mov.u32 \t$r, $s;",
                             [(set B32:$r,
                             (int_nvvm_move_ptr texternalsym:$s))]>;
def nvvm_move_sym64 : NVPTXInst<(outs B64:$r), (ins ADDR_base:$s),
                             "mov.u64 \t$r, $s;",
                             [(set B64:$r,
                             (int_nvvm_move_ptr texternalsym:$s))]>;*/

def texsurf_handles
  : BasicNVPTXInst<(outs B64:$result), (ins ADDR_base:$src), "mov.u64">;
def : Pat<(int_nvvm_texsurf_handle_internal globaladdr:$src),
          (texsurf_handles (to_tglobaladdr $src))>;

//-----------------------------------
// Compiler Error Warn
// - Just ignore them in codegen
//-----------------------------------

def INT_NVVM_COMPILER_WARN_32 : NVPTXInst<(outs), (ins B32:$a),
                "// llvm.nvvm.compiler.warn()",
                [(int_nvvm_compiler_warn i32:$a)]>;
def INT_NVVM_COMPILER_WARN_64 : NVPTXInst<(outs), (ins B64:$a),
                "// llvm.nvvm.compiler.warn()",
                [(int_nvvm_compiler_warn i64:$a)]>;
def INT_NVVM_COMPILER_ERROR_32 : NVPTXInst<(outs), (ins B32:$a),
                "// llvm.nvvm.compiler.error()",
                [(int_nvvm_compiler_error i32:$a)]>;
def INT_NVVM_COMPILER_ERROR_64 : NVPTXInst<(outs), (ins B64:$a),
                "// llvm.nvvm.compiler.error()",
                [(int_nvvm_compiler_error i64:$a)]>;


// isspacep

multiclass ISSPACEP<string suffix, Intrinsic Intr, list<Predicate> Preds = []> {
  def _32: BasicNVPTXInst<(outs B1:$d), (ins B32:$a),
              "isspacep." # suffix,
              [(set i1:$d, (Intr i32:$a))]>,
    Requires<Preds>;
  def _64: BasicNVPTXInst<(outs B1:$d), (ins B64:$a),
              "isspacep." # suffix,
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
def MOV_SPECIAL : BasicNVPTXInst<(outs B32:$d),
                            (ins SpecialRegs:$r),
                            "mov.b32", []>;

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

//-----------------------------------
// Texture Intrinsics
//-----------------------------------

// NOTE: For Fermi support, any new texture/surface/sampler intrinsics must be
// also defined in NVPTXReplaceImageHandles.cpp

// texmode_independent
let IsTex = true, IsTexModeUnified = false in {
// Texture fetch instructions using handles

class TEX_1D_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x\\}];",
                 pattern>;

multiclass TEX_1D<string inst, Intrinsic intr> {
  def _RR : TEX_1D_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x))]>;
  def _RI : TEX_1D_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_1D_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_1D_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_F32_S32 : TEX_1D<"tex.1d.v4.f32.s32", int_nvvm_tex_1d_v4f32_s32>;
defm TEX_1D_F32_F32 : TEX_1D<"tex.1d.v4.f32.f32", int_nvvm_tex_1d_v4f32_f32>;
defm TEX_1D_S32_S32 : TEX_1D<"tex.1d.v4.s32.s32", int_nvvm_tex_1d_v4s32_s32>;
defm TEX_1D_S32_F32 : TEX_1D<"tex.1d.v4.s32.f32", int_nvvm_tex_1d_v4s32_f32>;
defm TEX_1D_U32_S32 : TEX_1D<"tex.1d.v4.u32.s32", int_nvvm_tex_1d_v4u32_s32>;
defm TEX_1D_U32_F32 : TEX_1D<"tex.1d.v4.u32.f32", int_nvvm_tex_1d_v4u32_f32>;

class TEX_1D_LEVEL_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x\\}], $lod;",
                 pattern>;

multiclass TEX_1D_LEVEL<string inst, Intrinsic intr> {
  def _RR : TEX_1D_LEVEL_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$lod))]>;
  def _RI : TEX_1D_LEVEL_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_1D_LEVEL_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_1D_LEVEL_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_F32_F32_LEVEL :
  TEX_1D_LEVEL<"tex.level.1d.v4.f32.f32", int_nvvm_tex_1d_level_v4f32_f32>;
defm TEX_1D_S32_F32_LEVEL :
  TEX_1D_LEVEL<"tex.level.1d.v4.s32.f32", int_nvvm_tex_1d_level_v4s32_f32>;
defm TEX_1D_U32_F32_LEVEL :
  TEX_1D_LEVEL<"tex.level.1d.v4.u32.f32", int_nvvm_tex_1d_level_v4u32_f32>;

class TEX_1D_GRAD_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$gradx, B32:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x\\}],"
                        " \\{$gradx\\}, \\{$grady\\};",
                 pattern>;

multiclass TEX_1D_GRAD<string inst, Intrinsic intr> {
  def _RR : TEX_1D_GRAD_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$gradx, B32:$grady))]>;
  def _RI : TEX_1D_GRAD_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_1D_GRAD_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_1D_GRAD_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_F32_F32_GRAD
  : TEX_1D_GRAD<"tex.grad.1d.v4.f32.f32", int_nvvm_tex_1d_grad_v4f32_f32>;
defm TEX_1D_S32_F32_GRAD
  : TEX_1D_GRAD<"tex.grad.1d.v4.s32.f32", int_nvvm_tex_1d_grad_v4s32_f32>;
defm TEX_1D_U32_F32_GRAD
  : TEX_1D_GRAD<"tex.grad.1d.v4.u32.f32", int_nvvm_tex_1d_grad_v4u32_f32>;

class TEX_1D_ARRAY_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$l, B32:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$l, $x\\}];",
                 pattern>;

multiclass TEX_1D_ARRAY<string inst, Intrinsic intr> {
  def _RR : TEX_1D_ARRAY_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$l, B32:$x))]>;
  def _RI : TEX_1D_ARRAY_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_1D_ARRAY_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_1D_ARRAY_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_ARRAY_F32_F32
  : TEX_1D_ARRAY<"tex.a1d.v4.f32.f32", int_nvvm_tex_1d_array_v4f32_f32>;
defm TEX_1D_ARRAY_F32_S32
  : TEX_1D_ARRAY<"tex.a1d.v4.f32.s32", int_nvvm_tex_1d_array_v4f32_s32>;
defm TEX_1D_ARRAY_S32_S32
  : TEX_1D_ARRAY<"tex.a1d.v4.s32.s32", int_nvvm_tex_1d_array_v4s32_s32>;
defm TEX_1D_ARRAY_S32_F32
  : TEX_1D_ARRAY<"tex.a1d.v4.s32.f32", int_nvvm_tex_1d_array_v4s32_f32>;
defm TEX_1D_ARRAY_U32_S32
  : TEX_1D_ARRAY<"tex.a1d.v4.u32.s32", int_nvvm_tex_1d_array_v4u32_s32>;
defm TEX_1D_ARRAY_U32_F32
  : TEX_1D_ARRAY<"tex.a1d.v4.u32.f32", int_nvvm_tex_1d_array_v4u32_f32>;

class TEX_1D_ARRAY_LEVEL_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$l, B32:$x, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x\\}], $lod;",
                 pattern>;

multiclass TEX_1D_ARRAY_LEVEL<string inst, Intrinsic intr> {
  def _RR : TEX_1D_ARRAY_LEVEL_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$l, B32:$x, B32:$lod))]>;
  def _RI : TEX_1D_ARRAY_LEVEL_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_1D_ARRAY_LEVEL_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_1D_ARRAY_LEVEL_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_ARRAY_F32_F32_LEVEL
  : TEX_1D_ARRAY_LEVEL<"tex.level.a1d.v4.f32.f32", int_nvvm_tex_1d_array_level_v4f32_f32>;
defm TEX_1D_ARRAY_S32_F32_LEVEL
  : TEX_1D_ARRAY_LEVEL<"tex.level.a1d.v4.s32.f32", int_nvvm_tex_1d_array_level_v4s32_f32>;
defm TEX_1D_ARRAY_U32_F32_LEVEL
  : TEX_1D_ARRAY_LEVEL<"tex.level.a1d.v4.u32.f32", int_nvvm_tex_1d_array_level_v4u32_f32>;

class TEX_1D_ARRAY_GRAD_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$l, B32:$x, B32:$gradx, B32:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$l, $x\\}],"
                        " \\{$gradx\\}, \\{$grady\\};",
                 pattern>;

multiclass TEX_1D_ARRAY_GRAD<string inst, Intrinsic intr> {
  def _RR : TEX_1D_ARRAY_GRAD_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$l, B32:$x,
                  B32:$gradx, B32:$grady))]>;
  def _RI : TEX_1D_ARRAY_GRAD_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_1D_ARRAY_GRAD_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_1D_ARRAY_GRAD_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_1D_ARRAY_F32_F32_GRAD
  : TEX_1D_ARRAY_GRAD<"tex.grad.a1d.v4.f32.f32", int_nvvm_tex_1d_array_grad_v4f32_f32>;
defm TEX_1D_ARRAY_S32_F32_GRAD
  : TEX_1D_ARRAY_GRAD<"tex.grad.a1d.v4.s32.f32", int_nvvm_tex_1d_array_grad_v4s32_f32>;
defm TEX_1D_ARRAY_U32_F32_GRAD
  : TEX_1D_ARRAY_GRAD<"tex.grad.a1d.v4.u32.f32", int_nvvm_tex_1d_array_grad_v4u32_f32>;

class TEX_2D_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x, $y\\}];",
                 pattern>;

multiclass TEX_2D<string inst, Intrinsic intr> {
  def _RR : TEX_2D_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$y))]>;
  def _RI : TEX_2D_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_2D_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_2D_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_F32_F32 : TEX_2D<"tex.2d.v4.f32.f32", int_nvvm_tex_2d_v4f32_f32>;
defm TEX_2D_F32_S32 : TEX_2D<"tex.2d.v4.f32.s32", int_nvvm_tex_2d_v4f32_s32>;
defm TEX_2D_S32_S32 : TEX_2D<"tex.2d.v4.s32.s32", int_nvvm_tex_2d_v4s32_s32>;
defm TEX_2D_S32_F32 : TEX_2D<"tex.2d.v4.s32.f32", int_nvvm_tex_2d_v4s32_f32>;
defm TEX_2D_U32_S32 : TEX_2D<"tex.2d.v4.u32.s32", int_nvvm_tex_2d_v4u32_s32>;
defm TEX_2D_U32_F32 : TEX_2D<"tex.2d.v4.u32.f32", int_nvvm_tex_2d_v4u32_f32>;

class TEX_2D_LEVEL_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$y, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y\\}], $lod;",
                 pattern>;

multiclass TEX_2D_LEVEL<string inst, Intrinsic intr> {
  def _RR : TEX_2D_LEVEL_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$y, B32:$lod))]>;
  def _RI : TEX_2D_LEVEL_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_2D_LEVEL_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_2D_LEVEL_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_F32_F32_LEVEL :
  TEX_2D_LEVEL<"tex.level.2d.v4.f32.f32", int_nvvm_tex_2d_level_v4f32_f32>;
defm TEX_2D_S32_F32_LEVEL :
  TEX_2D_LEVEL<"tex.level.2d.v4.s32.f32", int_nvvm_tex_2d_level_v4s32_f32>;
defm TEX_2D_U32_F32_LEVEL :
  TEX_2D_LEVEL<"tex.level.2d.v4.u32.f32", int_nvvm_tex_2d_level_v4u32_f32>;

class TEX_2D_GRAD_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$y,
                                    B32:$gradx0, B32:$gradx1,
                                    B32:$grady0, B32:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, $s, \\{$x, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\};",
                 pattern>;

multiclass TEX_2D_GRAD<string inst, Intrinsic intr> {
  def _RR : TEX_2D_GRAD_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$y,
                  B32:$gradx0, B32:$gradx1,
                  B32:$grady0, B32:$grady1))]>;
  def _RI : TEX_2D_GRAD_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_2D_GRAD_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_2D_GRAD_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_F32_F32_GRAD :
  TEX_2D_GRAD<"tex.grad.2d.v4.f32.f32", int_nvvm_tex_2d_grad_v4f32_f32>;
defm TEX_2D_S32_F32_GRAD :
  TEX_2D_GRAD<"tex.grad.2d.v4.s32.f32", int_nvvm_tex_2d_grad_v4s32_f32>;
defm TEX_2D_U32_F32_GRAD :
  TEX_2D_GRAD<"tex.grad.2d.v4.u32.f32", int_nvvm_tex_2d_grad_v4u32_f32>;

class TEX_2D_ARRAY_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$l, B32:$x, B32:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $y\\}];",
                 pattern>;

multiclass TEX_2D_ARRAY<string inst, Intrinsic intr> {
  def _RR : TEX_2D_ARRAY_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$l, B32:$x, B32:$y))]>;
  def _RI : TEX_2D_ARRAY_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_2D_ARRAY_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_2D_ARRAY_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_ARRAY_F32_F32
  : TEX_2D_ARRAY<"tex.a2d.v4.f32.f32", int_nvvm_tex_2d_array_v4f32_f32>;
defm TEX_2D_ARRAY_F32_S32
  : TEX_2D_ARRAY<"tex.a2d.v4.f32.s32", int_nvvm_tex_2d_array_v4f32_s32>;
defm TEX_2D_ARRAY_S32_S32
  : TEX_2D_ARRAY<"tex.a2d.v4.s32.s32", int_nvvm_tex_2d_array_v4s32_s32>;
defm TEX_2D_ARRAY_S32_F32
  : TEX_2D_ARRAY<"tex.a2d.v4.s32.f32", int_nvvm_tex_2d_array_v4s32_f32>;
defm TEX_2D_ARRAY_U32_S32
  : TEX_2D_ARRAY<"tex.a2d.v4.u32.s32", int_nvvm_tex_2d_array_v4u32_s32>;
defm TEX_2D_ARRAY_U32_F32
  : TEX_2D_ARRAY<"tex.a2d.v4.u32.f32", int_nvvm_tex_2d_array_v4u32_f32>;

class TEX_2D_ARRAY_LEVEL_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$l, B32:$x, B32:$y, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $y\\}], $lod;",
                 pattern>;

multiclass TEX_2D_ARRAY_LEVEL<string inst, Intrinsic intr> {
  def _RR : TEX_2D_ARRAY_LEVEL_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$l, B32:$x, B32:$y, B32:$lod))]>;
  def _RI : TEX_2D_ARRAY_LEVEL_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_2D_ARRAY_LEVEL_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_2D_ARRAY_LEVEL_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_ARRAY_F32_F32_LEVEL
  : TEX_2D_ARRAY_LEVEL<"tex.level.a2d.v4.f32.f32", int_nvvm_tex_2d_array_level_v4f32_f32>;
defm TEX_2D_ARRAY_S32_F32_LEVEL
  : TEX_2D_ARRAY_LEVEL<"tex.level.a2d.v4.s32.f32", int_nvvm_tex_2d_array_level_v4s32_f32>;
defm TEX_2D_ARRAY_U32_F32_LEVEL
  : TEX_2D_ARRAY_LEVEL<"tex.level.a2d.v4.u32.f32", int_nvvm_tex_2d_array_level_v4u32_f32>;

class TEX_2D_ARRAY_GRAD_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$l, B32:$x, B32:$y,
                                    B32:$gradx0, B32:$gradx1,
                                    B32:$grady0, B32:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\};",
                 pattern>;

multiclass TEX_2D_ARRAY_GRAD<string inst, Intrinsic intr> {
  def _RR : TEX_2D_ARRAY_GRAD_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$l, B32:$x, B32:$y,
                  B32:$gradx0, B32:$gradx1,
                  B32:$grady0, B32:$grady1))]>;
  def _RI : TEX_2D_ARRAY_GRAD_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_2D_ARRAY_GRAD_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_2D_ARRAY_GRAD_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_2D_ARRAY_F32_F32_GRAD
  : TEX_2D_ARRAY_GRAD<"tex.grad.a2d.v4.f32.f32", int_nvvm_tex_2d_array_grad_v4f32_f32>;
defm TEX_2D_ARRAY_S32_F32_GRAD
  : TEX_2D_ARRAY_GRAD<"tex.grad.a2d.v4.s32.f32", int_nvvm_tex_2d_array_grad_v4s32_f32>;
defm TEX_2D_ARRAY_U32_F32_GRAD
  : TEX_2D_ARRAY_GRAD<"tex.grad.a2d.v4.u32.f32", int_nvvm_tex_2d_array_grad_v4u32_f32>;

class TEX_3D_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$y, B32:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}];",
                 pattern>;

multiclass TEX_3D<string inst, Intrinsic intr> {
  def _RR : TEX_3D_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$y, B32:$z))]>;
  def _RI : TEX_3D_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_3D_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_3D_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_3D_F32_F32 : TEX_3D<"tex.3d.v4.f32.f32", int_nvvm_tex_3d_v4f32_f32>;
defm TEX_3D_F32_S32 : TEX_3D<"tex.3d.v4.f32.s32", int_nvvm_tex_3d_v4f32_s32>;
defm TEX_3D_S32_S32 : TEX_3D<"tex.3d.v4.s32.s32", int_nvvm_tex_3d_v4s32_s32>;
defm TEX_3D_S32_F32 : TEX_3D<"tex.3d.v4.s32.f32", int_nvvm_tex_3d_v4s32_f32>;
defm TEX_3D_U32_S32 : TEX_3D<"tex.3d.v4.u32.s32", int_nvvm_tex_3d_v4u32_s32>;
defm TEX_3D_U32_F32 : TEX_3D<"tex.3d.v4.u32.f32", int_nvvm_tex_3d_v4u32_f32>;

class TEX_3D_LEVEL_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$y, B32:$z, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}], $lod;",
                 pattern>;

multiclass TEX_3D_LEVEL<string inst, Intrinsic intr> {
  def _RR : TEX_3D_LEVEL_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$y, B32:$z, B32:$lod))]>;
  def _RI : TEX_3D_LEVEL_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_3D_LEVEL_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_3D_LEVEL_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_3D_F32_F32_LEVEL
  : TEX_3D_LEVEL<"tex.level.3d.v4.f32.f32", int_nvvm_tex_3d_level_v4f32_f32>;
defm TEX_3D_S32_F32_LEVEL
  : TEX_3D_LEVEL<"tex.level.3d.v4.s32.f32", int_nvvm_tex_3d_level_v4s32_f32>;
defm TEX_3D_U32_F32_LEVEL
  : TEX_3D_LEVEL<"tex.level.3d.v4.u32.f32", int_nvvm_tex_3d_level_v4u32_f32>;

class TEX_3D_GRAD_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$y, B32:$z,
                                    B32:$gradx0, B32:$gradx1,
                                    B32:$gradx2, B32:$grady0,
                                    B32:$grady1, B32:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\};",
                 pattern>;

multiclass TEX_3D_GRAD<string inst, Intrinsic intr> {
  def _RR : TEX_3D_GRAD_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$y, B32:$z,
                  B32:$gradx0, B32:$gradx1, B32:$gradx2,
                  B32:$grady0, B32:$grady1, B32:$grady2))]>;
  def _RI : TEX_3D_GRAD_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_3D_GRAD_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_3D_GRAD_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_3D_F32_F32_GRAD
  : TEX_3D_GRAD<"tex.grad.3d.v4.f32.f32", int_nvvm_tex_3d_grad_v4f32_f32>;
defm TEX_3D_S32_F32_GRAD
  : TEX_3D_GRAD<"tex.grad.3d.v4.s32.f32", int_nvvm_tex_3d_grad_v4s32_f32>;
defm TEX_3D_U32_F32_GRAD
  : TEX_3D_GRAD<"tex.grad.3d.v4.u32.f32", int_nvvm_tex_3d_grad_v4u32_f32>;

class TEX_CUBE_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$y, B32:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}];",
                 pattern>;

multiclass TEX_CUBE<string inst, Intrinsic intr> {
  def _RR : TEX_CUBE_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$y, B32:$z))]>;
  def _RI : TEX_CUBE_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_CUBE_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_F32_F32
  : TEX_CUBE<"tex.cube.v4.f32.f32", int_nvvm_tex_cube_v4f32_f32>;
defm TEX_CUBE_S32_F32
  : TEX_CUBE<"tex.cube.v4.s32.f32", int_nvvm_tex_cube_v4s32_f32>;
defm TEX_CUBE_U32_F32
  : TEX_CUBE<"tex.cube.v4.u32.f32", int_nvvm_tex_cube_v4u32_f32>;

class TEX_CUBE_LEVEL_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$x, B32:$y, B32:$z, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$x, $y, $z, $z\\}], $lod;",
                 pattern>;

multiclass TEX_CUBE_LEVEL<string inst, Intrinsic intr> {
  def _RR : TEX_CUBE_LEVEL_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$x, B32:$y, B32:$z,
                  B32:$lod))]>;
  def _RI : TEX_CUBE_LEVEL_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_LEVEL_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_CUBE_LEVEL_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_F32_F32_LEVEL
  : TEX_CUBE_LEVEL<"tex.level.cube.v4.f32.f32", int_nvvm_tex_cube_level_v4f32_f32>;
defm TEX_CUBE_S32_F32_LEVEL
  : TEX_CUBE_LEVEL<"tex.level.cube.v4.s32.f32", int_nvvm_tex_cube_level_v4s32_f32>;
defm TEX_CUBE_U32_F32_LEVEL
  : TEX_CUBE_LEVEL<"tex.level.cube.v4.u32.f32", int_nvvm_tex_cube_level_v4u32_f32>;

class TEX_CUBE_ARRAY_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$l, B32:$x, B32:$y, B32:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $z\\}];",
                 pattern>;

multiclass TEX_CUBE_ARRAY<string inst, Intrinsic intr> {
  def _RR : TEX_CUBE_ARRAY_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$l, B32:$x, B32:$y, B32:$z))]>;
  def _RI : TEX_CUBE_ARRAY_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_ARRAY_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_CUBE_ARRAY_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_ARRAY_F32_F32
  : TEX_CUBE_ARRAY<"tex.acube.v4.f32.f32", int_nvvm_tex_cube_array_v4f32_f32>;
defm TEX_CUBE_ARRAY_S32_F32
  : TEX_CUBE_ARRAY<"tex.acube.v4.s32.f32", int_nvvm_tex_cube_array_v4s32_f32>;
defm TEX_CUBE_ARRAY_U32_F32
  : TEX_CUBE_ARRAY<"tex.acube.v4.u32.f32", int_nvvm_tex_cube_array_v4u32_f32>;

class TEX_CUBE_ARRAY_LEVEL_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(texsamp, (ins B32:$l, B32:$x, B32:$y, B32:$z, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, $s, \\{$l, $x, $y, $z\\}], $lod;",
                 pattern>;

multiclass TEX_CUBE_ARRAY_LEVEL<string inst, Intrinsic intr> {
  def _RR : TEX_CUBE_ARRAY_LEVEL_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i64:$s, B32:$l, B32:$x, B32:$y, B32:$z,
                  B32:$lod))]>;
  def _RI : TEX_CUBE_ARRAY_LEVEL_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TEX_CUBE_ARRAY_LEVEL_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TEX_CUBE_ARRAY_LEVEL_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TEX_CUBE_ARRAY_F32_F32_LEVEL
  : TEX_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.f32.f32",
                         int_nvvm_tex_cube_array_level_v4f32_f32>;
defm TEX_CUBE_ARRAY_S32_F32_LEVEL
  : TEX_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.s32.f32",
                         int_nvvm_tex_cube_array_level_v4s32_f32>;
defm TEX_CUBE_ARRAY_U32_F32_LEVEL
  : TEX_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.u32.f32",
                         int_nvvm_tex_cube_array_level_v4u32_f32>;

class TLD4_2D_base<string inst, dag texsamp, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$v0, B32:$v1, B32:$v2, B32:$v3),
                 !con(texsamp, (ins B32:$x, B32:$y)),
                 inst # " \t\\{$v0, $v1, $v2, $v3\\}, [$t, $s, \\{$x, $y\\}];",
                 pattern>;

multiclass TLD4_2D<string inst, Intrinsic intr> {
  def _RR : TLD4_2D_base<inst, (ins B64:$t, B64:$s),
      [(set B32:$v0, B32:$v1, B32:$v2, B32:$v3,
            (intr i64:$t, i64:$s, B32:$x, B32:$y))]>;
  def _RI : TLD4_2D_base<inst, (ins B64:$t, i64imm:$s)>;
  def _IR : TLD4_2D_base<inst, (ins i64imm:$t, B64:$s)>;
  def _II : TLD4_2D_base<inst, (ins i64imm:$t, i64imm:$s)>;
}

defm TLD4_R_2D_F32_F32
  : TLD4_2D<"tld4.r.2d.v4.f32.f32", int_nvvm_tld4_r_2d_v4f32_f32>;
defm TLD4_G_2D_F32_F32
  : TLD4_2D<"tld4.g.2d.v4.f32.f32", int_nvvm_tld4_g_2d_v4f32_f32>;
defm TLD4_B_2D_F32_F32
  : TLD4_2D<"tld4.b.2d.v4.f32.f32", int_nvvm_tld4_b_2d_v4f32_f32>;
defm TLD4_A_2D_F32_F32
  : TLD4_2D<"tld4.a.2d.v4.f32.f32", int_nvvm_tld4_a_2d_v4f32_f32>;

defm TLD4_R_2D_S32_F32
  : TLD4_2D<"tld4.r.2d.v4.s32.f32", int_nvvm_tld4_r_2d_v4s32_f32>;
defm TLD4_G_2D_S32_F32
  : TLD4_2D<"tld4.g.2d.v4.s32.f32", int_nvvm_tld4_g_2d_v4s32_f32>;
defm TLD4_B_2D_S32_F32
  : TLD4_2D<"tld4.b.2d.v4.s32.f32", int_nvvm_tld4_b_2d_v4s32_f32>;
defm TLD4_A_2D_S32_F32
  : TLD4_2D<"tld4.a.2d.v4.s32.f32", int_nvvm_tld4_a_2d_v4s32_f32>;

defm TLD4_R_2D_U32_F32
  : TLD4_2D<"tld4.r.2d.v4.u32.f32", int_nvvm_tld4_r_2d_v4u32_f32>;
defm TLD4_G_2D_U32_F32
  : TLD4_2D<"tld4.g.2d.v4.u32.f32", int_nvvm_tld4_g_2d_v4u32_f32>;
defm TLD4_B_2D_U32_F32
  : TLD4_2D<"tld4.b.2d.v4.u32.f32", int_nvvm_tld4_b_2d_v4u32_f32>;
defm TLD4_A_2D_U32_F32
  : TLD4_2D<"tld4.a.2d.v4.u32.f32", int_nvvm_tld4_a_2d_v4u32_f32>;

}


// texmode_unified
let IsTex = true, IsTexModeUnified = true in {
// Texture fetch instructions using handles

class TEX_UNIFIED_1D_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x\\}];",
                 pattern>;

multiclass TEX_UNIFIED_1D<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a, (intr i64:$t, B32:$x))]>;
  def _I : TEX_UNIFIED_1D_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_F32_S32
  : TEX_UNIFIED_1D<"tex.1d.v4.f32.s32", int_nvvm_tex_unified_1d_v4f32_s32>;
defm TEX_UNIFIED_1D_F32_F32
  : TEX_UNIFIED_1D<"tex.1d.v4.f32.f32", int_nvvm_tex_unified_1d_v4f32_f32>;
defm TEX_UNIFIED_1D_S32_S32
  : TEX_UNIFIED_1D<"tex.1d.v4.s32.s32", int_nvvm_tex_unified_1d_v4s32_s32>;
defm TEX_UNIFIED_1D_S32_F32
  : TEX_UNIFIED_1D<"tex.1d.v4.s32.f32", int_nvvm_tex_unified_1d_v4s32_f32>;
defm TEX_UNIFIED_1D_U32_S32
  : TEX_UNIFIED_1D<"tex.1d.v4.u32.s32", int_nvvm_tex_unified_1d_v4u32_s32>;
defm TEX_UNIFIED_1D_U32_F32
  : TEX_UNIFIED_1D<"tex.1d.v4.u32.f32", int_nvvm_tex_unified_1d_v4u32_f32>;

class TEX_UNIFIED_1D_LEVEL_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x\\}], $lod;",
                 pattern>;

multiclass TEX_UNIFIED_1D_LEVEL<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_LEVEL_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$lod))]>;
  def _I : TEX_UNIFIED_1D_LEVEL_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_F32_F32_LEVEL
  : TEX_UNIFIED_1D_LEVEL<"tex.level.1d.v4.f32.f32", int_nvvm_tex_unified_1d_level_v4f32_f32>;
defm TEX_UNIFIED_1D_S32_F32_LEVEL
  : TEX_UNIFIED_1D_LEVEL<"tex.level.1d.v4.s32.f32", int_nvvm_tex_unified_1d_level_v4s32_f32>;
defm TEX_UNIFIED_1D_U32_F32_LEVEL
  : TEX_UNIFIED_1D_LEVEL<"tex.level.1d.v4.u32.f32", int_nvvm_tex_unified_1d_level_v4u32_f32>;

class TEX_UNIFIED_1D_GRAD_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$gradx, B32:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$x\\}], \\{$gradx\\}, \\{$grady\\};",
                 pattern>;

multiclass TEX_UNIFIED_1D_GRAD<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_GRAD_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$gradx, B32:$grady))]>;
  def _I : TEX_UNIFIED_1D_GRAD_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_F32_F32_GRAD
  : TEX_UNIFIED_1D_GRAD<"tex.grad.1d.v4.f32.f32", int_nvvm_tex_unified_1d_grad_v4f32_f32>;
defm TEX_UNIFIED_1D_S32_F32_GRAD
  : TEX_UNIFIED_1D_GRAD<"tex.grad.1d.v4.s32.f32", int_nvvm_tex_unified_1d_grad_v4s32_f32>;
defm TEX_UNIFIED_1D_U32_F32_GRAD
  : TEX_UNIFIED_1D_GRAD<"tex.grad.1d.v4.u32.f32", int_nvvm_tex_unified_1d_grad_v4u32_f32>;

class TEX_UNIFIED_1D_ARRAY_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$l, B32:$x)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x\\}];",
                 pattern>;

multiclass TEX_UNIFIED_1D_ARRAY<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_ARRAY_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a, (intr i64:$t, B32:$l, B32:$x))]>;
  def _I : TEX_UNIFIED_1D_ARRAY_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_ARRAY_F32_S32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.f32.s32", int_nvvm_tex_unified_1d_array_v4f32_s32>;
defm TEX_UNIFIED_1D_ARRAY_F32_F32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.f32.f32", int_nvvm_tex_unified_1d_array_v4f32_f32>;
defm TEX_UNIFIED_1D_ARRAY_S32_S32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.s32.s32", int_nvvm_tex_unified_1d_array_v4s32_s32>;
defm TEX_UNIFIED_1D_ARRAY_S32_F32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.s32.f32", int_nvvm_tex_unified_1d_array_v4s32_f32>;
defm TEX_UNIFIED_1D_ARRAY_U32_S32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.u32.s32", int_nvvm_tex_unified_1d_array_v4u32_s32>;
defm TEX_UNIFIED_1D_ARRAY_U32_F32
  : TEX_UNIFIED_1D_ARRAY<"tex.a1d.v4.u32.f32", int_nvvm_tex_unified_1d_array_v4u32_f32>;

class TEX_UNIFIED_1D_ARRAY_LEVEL_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$l, B32:$x, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x\\}], $lod;",
                 pattern>;

multiclass TEX_UNIFIED_1D_ARRAY_LEVEL<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_ARRAY_LEVEL_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$l, B32:$x, B32:$lod))]>;
  def _I : TEX_UNIFIED_1D_ARRAY_LEVEL_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_ARRAY_F32_F32_LEVEL
  : TEX_UNIFIED_1D_ARRAY_LEVEL<"tex.level.a1d.v4.f32.f32",
                               int_nvvm_tex_unified_1d_array_level_v4f32_f32>;
defm TEX_UNIFIED_1D_ARRAY_S32_F32_LEVEL
  : TEX_UNIFIED_1D_ARRAY_LEVEL<"tex.level.a1d.v4.s32.f32",
                               int_nvvm_tex_unified_1d_array_level_v4s32_f32>;
defm TEX_UNIFIED_1D_ARRAY_U32_F32_LEVEL
  : TEX_UNIFIED_1D_ARRAY_LEVEL<"tex.level.a1d.v4.u32.f32",
                               int_nvvm_tex_unified_1d_array_level_v4u32_f32>;

class TEX_UNIFIED_1D_ARRAY_GRAD_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$l, B32:$x, B32:$gradx, B32:$grady)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        "  [$t, \\{$l, $x\\}], \\{$gradx\\}, \\{$grady\\};",
                 pattern>;

multiclass TEX_UNIFIED_1D_ARRAY_GRAD<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_1D_ARRAY_GRAD_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$l, B32:$x, B32:$gradx, B32:$grady))]>;
  def _I : TEX_UNIFIED_1D_ARRAY_GRAD_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_1D_ARRAY_F32_F32_GRAD
  : TEX_UNIFIED_1D_ARRAY_GRAD<"tex.grad.a1d.v4.f32.f32",
                              int_nvvm_tex_unified_1d_array_grad_v4f32_f32>;
defm TEX_UNIFIED_1D_ARRAY_S32_F32_GRAD
  : TEX_UNIFIED_1D_ARRAY_GRAD<"tex.grad.a1d.v4.s32.f32",
                              int_nvvm_tex_unified_1d_array_grad_v4s32_f32>;
defm TEX_UNIFIED_1D_ARRAY_U32_F32_GRAD
  : TEX_UNIFIED_1D_ARRAY_GRAD<"tex.grad.a1d.v4.u32.f32",
                              int_nvvm_tex_unified_1d_array_grad_v4u32_f32>;

class TEX_UNIFIED_2D_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y\\}];",
                 pattern>;

multiclass TEX_UNIFIED_2D<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$y))]>;
  def _I : TEX_UNIFIED_2D_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_F32_S32
  : TEX_UNIFIED_2D<"tex.2d.v4.f32.s32", int_nvvm_tex_unified_2d_v4f32_s32>;
defm TEX_UNIFIED_2D_F32_F32
  : TEX_UNIFIED_2D<"tex.2d.v4.f32.f32", int_nvvm_tex_unified_2d_v4f32_f32>;
defm TEX_UNIFIED_2D_S32_S32
  : TEX_UNIFIED_2D<"tex.2d.v4.s32.s32", int_nvvm_tex_unified_2d_v4s32_s32>;
defm TEX_UNIFIED_2D_S32_F32
  : TEX_UNIFIED_2D<"tex.2d.v4.s32.f32", int_nvvm_tex_unified_2d_v4s32_f32>;
defm TEX_UNIFIED_2D_U32_S32
  : TEX_UNIFIED_2D<"tex.2d.v4.u32.s32", int_nvvm_tex_unified_2d_v4u32_s32>;
defm TEX_UNIFIED_2D_U32_F32
  : TEX_UNIFIED_2D<"tex.2d.v4.u32.f32", int_nvvm_tex_unified_2d_v4u32_f32>;

class TEX_UNIFIED_2D_LEVEL_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$y, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y\\}], $lod;",
                 pattern>;

multiclass TEX_UNIFIED_2D_LEVEL<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_LEVEL_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$y, B32:$lod))]>;
  def _I : TEX_UNIFIED_2D_LEVEL_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_F32_F32_LEVEL
  : TEX_UNIFIED_2D_LEVEL<"tex.level.2d.v4.f32.f32", int_nvvm_tex_unified_2d_level_v4f32_f32>;
defm TEX_UNIFIED_2D_S32_F32_LEVEL
  : TEX_UNIFIED_2D_LEVEL<"tex.level.2d.v4.s32.f32", int_nvvm_tex_unified_2d_level_v4s32_f32>;
defm TEX_UNIFIED_2D_U32_F32_LEVEL
  : TEX_UNIFIED_2D_LEVEL<"tex.level.2d.v4.u32.f32", int_nvvm_tex_unified_2d_level_v4u32_f32>;

class TEX_UNIFIED_2D_GRAD_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$y,
                                B32:$gradx0, B32:$gradx1,
                                B32:$grady0, B32:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\};",
                 pattern>;
multiclass TEX_UNIFIED_2D_GRAD<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_GRAD_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$y,
                  B32:$gradx0, B32:$gradx1,
                  B32:$grady0, B32:$grady1))]>;
  def _I : TEX_UNIFIED_2D_GRAD_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_F32_F32_GRAD
  : TEX_UNIFIED_2D_GRAD<"tex.grad.2d.v4.f32.f32", int_nvvm_tex_unified_2d_grad_v4f32_f32>;
defm TEX_UNIFIED_2D_S32_F32_GRAD
  : TEX_UNIFIED_2D_GRAD<"tex.grad.2d.v4.s32.f32", int_nvvm_tex_unified_2d_grad_v4s32_f32>;
defm TEX_UNIFIED_2D_U32_F32_GRAD
  : TEX_UNIFIED_2D_GRAD<"tex.grad.2d.v4.u32.f32", int_nvvm_tex_unified_2d_grad_v4u32_f32>;

class TEX_UNIFIED_2D_ARRAY_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$l, B32:$x, B32:$y)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $y\\}];",
                 pattern>;
multiclass TEX_UNIFIED_2D_ARRAY<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_ARRAY_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$l, B32:$x, B32:$y))]>;
  def _I : TEX_UNIFIED_2D_ARRAY_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_ARRAY_F32_S32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.f32.s32", int_nvvm_tex_unified_2d_array_v4f32_s32>;
defm TEX_UNIFIED_2D_ARRAY_F32_F32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.f32.f32", int_nvvm_tex_unified_2d_array_v4f32_f32>;
defm TEX_UNIFIED_2D_ARRAY_S32_S32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.s32.s32", int_nvvm_tex_unified_2d_array_v4s32_s32>;
defm TEX_UNIFIED_2D_ARRAY_S32_F32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.s32.f32", int_nvvm_tex_unified_2d_array_v4s32_f32>;
defm TEX_UNIFIED_2D_ARRAY_U32_S32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.u32.s32", int_nvvm_tex_unified_2d_array_v4u32_s32>;
defm TEX_UNIFIED_2D_ARRAY_U32_F32
  : TEX_UNIFIED_2D_ARRAY<"tex.a2d.v4.u32.f32", int_nvvm_tex_unified_2d_array_v4u32_f32>;

class TEX_UNIFIED_2D_ARRAY_LEVEL_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$l, B32:$x, B32:$y, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        "  [$t, \\{$l, $x, $y, $y\\}], $lod;",
                 pattern>;
multiclass TEX_UNIFIED_2D_ARRAY_LEVEL<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_ARRAY_LEVEL_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$l, B32:$x, B32:$y, B32:$lod))]>;
  def _I : TEX_UNIFIED_2D_ARRAY_LEVEL_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_ARRAY_F32_F32_LEVEL
  : TEX_UNIFIED_2D_ARRAY_LEVEL<"tex.level.a2d.v4.f32.f32",
                               int_nvvm_tex_unified_2d_array_level_v4f32_f32>;
defm TEX_UNIFIED_2D_ARRAY_S32_F32_LEVEL
  : TEX_UNIFIED_2D_ARRAY_LEVEL<"tex.level.a2d.v4.s32.f32",
                               int_nvvm_tex_unified_2d_array_level_v4s32_f32>;
defm TEX_UNIFIED_2D_ARRAY_U32_F32_LEVEL
  : TEX_UNIFIED_2D_ARRAY_LEVEL<"tex.level.a2d.v4.u32.f32",
                               int_nvvm_tex_unified_2d_array_level_v4u32_f32>;

class TEX_UNIFIED_2D_ARRAY_GRAD_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$l, B32:$x, B32:$y,
                                B32:$gradx0, B32:$gradx1,
                                B32:$grady0, B32:$grady1)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $y\\}],"
                        " \\{$gradx0, $gradx1\\}, \\{$grady0, $grady1\\};",
                 pattern>;
multiclass TEX_UNIFIED_2D_ARRAY_GRAD<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_2D_ARRAY_GRAD_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$l, B32:$x, B32:$y,
                  B32:$gradx0, B32:$gradx1,
                  B32:$grady0, B32:$grady1))]>;
  def _I : TEX_UNIFIED_2D_ARRAY_GRAD_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_2D_ARRAY_F32_F32_GRAD
  : TEX_UNIFIED_2D_ARRAY_GRAD<"tex.grad.a2d.v4.f32.f32",
                              int_nvvm_tex_unified_2d_array_grad_v4f32_f32>;
defm TEX_UNIFIED_2D_ARRAY_S32_F32_GRAD
  : TEX_UNIFIED_2D_ARRAY_GRAD<"tex.grad.a2d.v4.s32.f32",
                              int_nvvm_tex_unified_2d_array_grad_v4s32_f32>;
defm TEX_UNIFIED_2D_ARRAY_U32_F32_GRAD
  : TEX_UNIFIED_2D_ARRAY_GRAD<"tex.grad.a2d.v4.u32.f32",
                              int_nvvm_tex_unified_2d_array_grad_v4u32_f32>;

class TEX_UNIFIED_3D_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$y, B32:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}];",
                 pattern>;
multiclass TEX_UNIFIED_3D<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_3D_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$y, B32:$z))]>;
  def _I : TEX_UNIFIED_3D_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_3D_F32_S32
  : TEX_UNIFIED_3D<"tex.3d.v4.f32.s32", int_nvvm_tex_unified_3d_v4f32_s32>;
defm TEX_UNIFIED_3D_F32_F32
  : TEX_UNIFIED_3D<"tex.3d.v4.f32.f32", int_nvvm_tex_unified_3d_v4f32_f32>;
defm TEX_UNIFIED_3D_S32_S32
  : TEX_UNIFIED_3D<"tex.3d.v4.s32.s32", int_nvvm_tex_unified_3d_v4s32_s32>;
defm TEX_UNIFIED_3D_S32_F32
  : TEX_UNIFIED_3D<"tex.3d.v4.s32.f32", int_nvvm_tex_unified_3d_v4s32_f32>;
defm TEX_UNIFIED_3D_U32_S32
  : TEX_UNIFIED_3D<"tex.3d.v4.u32.s32", int_nvvm_tex_unified_3d_v4u32_s32>;
defm TEX_UNIFIED_3D_U32_F32
  : TEX_UNIFIED_3D<"tex.3d.v4.u32.f32", int_nvvm_tex_unified_3d_v4u32_f32>;

class TEX_UNIFIED_3D_LEVEL_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$y, B32:$z, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$x, $y, $z, $z\\}], $lod;",
                 pattern>;
multiclass TEX_UNIFIED_3D_LEVEL<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_3D_LEVEL_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$y, B32:$z, B32:$lod))]>;
  def _I : TEX_UNIFIED_3D_LEVEL_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_3D_F32_F32_LEVEL
  : TEX_UNIFIED_3D_LEVEL<"tex.level.3d.v4.f32.f32", int_nvvm_tex_unified_3d_level_v4f32_f32>;
defm TEX_UNIFIED_3D_S32_F32_LEVEL
  : TEX_UNIFIED_3D_LEVEL<"tex.level.3d.v4.s32.f32", int_nvvm_tex_unified_3d_level_v4s32_f32>;
defm TEX_UNIFIED_3D_U32_F32_LEVEL
  : TEX_UNIFIED_3D_LEVEL<"tex.level.3d.v4.u32.f32", int_nvvm_tex_unified_3d_level_v4u32_f32>;

class TEX_UNIFIED_3D_GRAD_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$y, B32:$z,
                                B32:$gradx0, B32:$gradx1,
                                B32:$gradx2, B32:$grady0,
                                B32:$grady1, B32:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\};",
                 pattern>;
multiclass TEX_UNIFIED_3D_GRAD<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_3D_GRAD_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$y, B32:$z,
                  B32:$gradx0, B32:$gradx1, B32:$gradx2,
                  B32:$grady0, B32:$grady1, B32:$grady2))]>;
  def _I : TEX_UNIFIED_3D_GRAD_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_3D_F32_F32_GRAD
  : TEX_UNIFIED_3D_GRAD<"tex.grad.3d.v4.f32.f32", int_nvvm_tex_unified_3d_grad_v4f32_f32>;
defm TEX_UNIFIED_3D_S32_F32_GRAD
  : TEX_UNIFIED_3D_GRAD<"tex.grad.3d.v4.s32.f32", int_nvvm_tex_unified_3d_grad_v4s32_f32>;
defm TEX_UNIFIED_3D_U32_F32_GRAD
  : TEX_UNIFIED_3D_GRAD<"tex.grad.3d.v4.u32.f32", int_nvvm_tex_unified_3d_grad_v4u32_f32>;

class TEX_UNIFIED_CUBE_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$y, B32:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}];",
                 pattern>;
multiclass TEX_UNIFIED_CUBE<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$y, B32:$z))]>;
  def _I : TEX_UNIFIED_CUBE_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_F32_F32
  : TEX_UNIFIED_CUBE<"tex.cube.v4.f32.f32", int_nvvm_tex_unified_cube_v4f32_f32>;
defm TEX_UNIFIED_CUBE_S32_F32
  : TEX_UNIFIED_CUBE<"tex.cube.v4.s32.f32", int_nvvm_tex_unified_cube_v4s32_f32>;
defm TEX_UNIFIED_CUBE_U32_F32
  : TEX_UNIFIED_CUBE<"tex.cube.v4.u32.f32", int_nvvm_tex_unified_cube_v4u32_f32>;

class TEX_UNIFIED_CUBE_LEVEL_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$y, B32:$z, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$x, $y, $z, $z\\}], $lod;",
                 pattern>;
multiclass TEX_UNIFIED_CUBE_LEVEL<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_LEVEL_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$y, B32:$z, B32:$lod))]>;
  def _I : TEX_UNIFIED_CUBE_LEVEL_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_F32_F32_LEVEL
  : TEX_UNIFIED_CUBE_LEVEL<"tex.level.cube.v4.f32.f32",
                           int_nvvm_tex_unified_cube_level_v4f32_f32>;
defm TEX_UNIFIED_CUBE_S32_F32_LEVEL
  : TEX_UNIFIED_CUBE_LEVEL<"tex.level.cube.v4.s32.f32",
                           int_nvvm_tex_unified_cube_level_v4s32_f32>;
defm TEX_UNIFIED_CUBE_U32_F32_LEVEL
  : TEX_UNIFIED_CUBE_LEVEL<"tex.level.cube.v4.u32.f32",
                           int_nvvm_tex_unified_cube_level_v4u32_f32>;

class TEX_UNIFIED_CUBE_ARRAY_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$l, B32:$x, B32:$y, B32:$z)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $z\\}];",
                 pattern>;
multiclass TEX_UNIFIED_CUBE_ARRAY<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_ARRAY_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i32:$l, B32:$x, B32:$y, B32:$z))]>;
  def _I : TEX_UNIFIED_CUBE_ARRAY_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_ARRAY_F32_F32
  : TEX_UNIFIED_CUBE_ARRAY<"tex.acube.v4.f32.f32", int_nvvm_tex_unified_cube_array_v4f32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_S32_F32
  : TEX_UNIFIED_CUBE_ARRAY<"tex.acube.v4.s32.f32", int_nvvm_tex_unified_cube_array_v4s32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_U32_F32
  : TEX_UNIFIED_CUBE_ARRAY<"tex.acube.v4.u32.f32", int_nvvm_tex_unified_cube_array_v4u32_f32>;

class TEX_UNIFIED_CUBE_ARRAY_LEVEL_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$l, B32:$x, B32:$y, B32:$z, B32:$lod)),
                 inst # " \t\\{$r, $g, $b, $a\\},"
                        " [$t, \\{$l, $x, $y, $z\\}], $lod;",
                 pattern>;
multiclass TEX_UNIFIED_CUBE_ARRAY_LEVEL<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_ARRAY_LEVEL_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i32:$l, B32:$x, B32:$y, B32:$z, B32:$lod))]>;
  def _I : TEX_UNIFIED_CUBE_ARRAY_LEVEL_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_ARRAY_F32_F32_LEVEL
  : TEX_UNIFIED_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.f32.f32",
                                 int_nvvm_tex_unified_cube_array_level_v4f32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_S32_F32_LEVEL
  : TEX_UNIFIED_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.s32.f32",
                                 int_nvvm_tex_unified_cube_array_level_v4s32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_U32_F32_LEVEL
  : TEX_UNIFIED_CUBE_ARRAY_LEVEL<"tex.level.acube.v4.u32.f32",
                                 int_nvvm_tex_unified_cube_array_level_v4u32_f32>;

class TEX_UNIFIED_CUBE_GRAD_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$x, B32:$y, B32:$z,
                                B32:$gradx0, B32:$gradx1,
                                B32:$gradx2, B32:$grady0,
                                B32:$grady1, B32:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$x, $y, $z, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\};",
                 pattern>;

multiclass TEX_UNIFIED_CUBE_GRAD<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_GRAD_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, B32:$x, B32:$y, B32:$z,
                  B32:$gradx0, B32:$gradx1, B32:$gradx2,
                  B32:$grady0, B32:$grady1, B32:$grady2))]>;
  def _I : TEX_UNIFIED_CUBE_GRAD_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_F32_F32_GRAD
  : TEX_UNIFIED_CUBE_GRAD<"tex.grad.cube.v4.f32.f32", int_nvvm_tex_unified_cube_grad_v4f32_f32>;
defm TEX_UNIFIED_CUBE_S32_F32_GRAD
  : TEX_UNIFIED_CUBE_GRAD<"tex.grad.cube.v4.s32.f32", int_nvvm_tex_unified_cube_grad_v4s32_f32>;
defm TEX_UNIFIED_CUBE_U32_F32_GRAD
  : TEX_UNIFIED_CUBE_GRAD<"tex.grad.cube.v4.u32.f32", int_nvvm_tex_unified_cube_grad_v4u32_f32>;

class TEX_UNIFIED_CUBE_ARRAY_GRAD_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$r, B32:$g, B32:$b, B32:$a),
                 !con(tex, (ins B32:$l, B32:$x, B32:$y, B32:$z,
                                B32:$gradx0, B32:$gradx1,
                                B32:$gradx2, B32:$grady0,
                                B32:$grady1, B32:$grady2)),
                 inst # " \t\\{$r, $g, $b, $a\\}, [$t, \\{$l, $x, $y, $z\\}],"
                        " \\{$gradx0, $gradx1, $gradx2, $gradx2\\},"
                        " \\{$grady0, $grady1, $grady2, $grady2\\};",
                 pattern>;
multiclass TEX_UNIFIED_CUBE_ARRAY_GRAD<string inst, Intrinsic intr> {
  def _R : TEX_UNIFIED_CUBE_ARRAY_GRAD_base<inst, (ins B64:$t),
      [(set B32:$r, B32:$g, B32:$b, B32:$a,
            (intr i64:$t, i32:$l, B32:$x, B32:$y, B32:$z,
                  B32:$gradx0, B32:$gradx1, B32:$gradx2,
                  B32:$grady0, B32:$grady1, B32:$grady2))]>;
  def _I : TEX_UNIFIED_CUBE_ARRAY_GRAD_base<inst, (ins i64imm:$t)>;
}

defm TEX_UNIFIED_CUBE_ARRAY_F32_F32_GRAD
  : TEX_UNIFIED_CUBE_ARRAY_GRAD<"tex.grad.acube.v4.f32.f32",
                                int_nvvm_tex_unified_cube_array_grad_v4f32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_S32_F32_GRAD
  : TEX_UNIFIED_CUBE_ARRAY_GRAD<"tex.grad.acube.v4.s32.f32",
                                int_nvvm_tex_unified_cube_array_grad_v4s32_f32>;
defm TEX_UNIFIED_CUBE_ARRAY_U32_F32_GRAD
  : TEX_UNIFIED_CUBE_ARRAY_GRAD<"tex.grad.acube.v4.u32.f32",
                                int_nvvm_tex_unified_cube_array_grad_v4u32_f32>;

class TLD4_UNIFIED_2D_base<string inst, dag tex, list<dag> pattern = []>
    : NVPTXInst<(outs B32:$v0, B32:$v1, B32:$v2, B32:$v3),
                 !con(tex, (ins B32:$x, B32:$y)),
                 inst # " \t\\{$v0, $v1, $v2, $v3\\}, [$t, \\{$x, $y\\}];",
                 pattern>;
multiclass TLD4_UNIFIED_2D<string inst, Intrinsic intr> {
  def _R : TLD4_UNIFIED_2D_base<inst, (ins B64:$t),
      [(set B32:$v0, B32:$v1, B32:$v2, B32:$v3,
            (intr i64:$t, B32:$x, B32:$y))]>;
  def _I : TLD4_UNIFIED_2D_base<inst, (ins i64imm:$t)>;
}

defm TLD4_UNIFIED_R_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.r.2d.v4.f32.f32", int_nvvm_tld4_unified_r_2d_v4f32_f32>;
defm TLD4_UNIFIED_G_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.g.2d.v4.f32.f32", int_nvvm_tld4_unified_g_2d_v4f32_f32>;
defm TLD4_UNIFIED_B_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.b.2d.v4.f32.f32", int_nvvm_tld4_unified_b_2d_v4f32_f32>;
defm TLD4_UNIFIED_A_2D_F32_F32
  : TLD4_UNIFIED_2D<"tld4.a.2d.v4.f32.f32", int_nvvm_tld4_unified_a_2d_v4f32_f32>;

defm TLD4_UNIFIED_R_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.r.2d.v4.s32.f32", int_nvvm_tld4_unified_r_2d_v4s32_f32>;
defm TLD4_UNIFIED_G_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.g.2d.v4.s32.f32", int_nvvm_tld4_unified_g_2d_v4s32_f32>;
defm TLD4_UNIFIED_B_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.b.2d.v4.s32.f32", int_nvvm_tld4_unified_b_2d_v4s32_f32>;
defm TLD4_UNIFIED_A_2D_S32_F32
  : TLD4_UNIFIED_2D<"tld4.a.2d.v4.s32.f32", int_nvvm_tld4_unified_a_2d_v4s32_f32>;

defm TLD4_UNIFIED_R_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.r.2d.v4.u32.f32", int_nvvm_tld4_unified_r_2d_v4u32_f32>;
defm TLD4_UNIFIED_G_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.g.2d.v4.u32.f32", int_nvvm_tld4_unified_g_2d_v4u32_f32>;
defm TLD4_UNIFIED_B_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.b.2d.v4.u32.f32", int_nvvm_tld4_unified_b_2d_v4u32_f32>;
defm TLD4_UNIFIED_A_2D_U32_F32
  : TLD4_UNIFIED_2D<"tld4.a.2d.v4.u32.f32", int_nvvm_tld4_unified_a_2d_v4u32_f32>;

}


//=== Surface load instructions

let IsSuld = true in {

class SULD_1D_base<string inst, NVPTXRegClass outtype, dag surf,
                   list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins B32:$x)),
                inst # " \\{$r\\}, [$s, \\{$x\\}];",
                pattern>;
multiclass SULD_1D<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_base<inst, outtype, (ins B64:$s),
                        [(set outtype:$r, (intr i64:$s, i32:$x))]>;
  def _I : SULD_1D_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_1D_I8_ # op_upper : SULD_1D<"suld.b.1d.b8." # op, B16>;
  defm SULD_1D_I16_ # op_upper : SULD_1D<"suld.b.1d.b16." # op, B16>;
  defm SULD_1D_I32_ # op_upper : SULD_1D<"suld.b.1d.b32." # op, B32>;
  defm SULD_1D_I64_ # op_upper : SULD_1D<"suld.b.1d.b64." # op, B64>;
}

class SULD_1D_ARRAY_base<string inst, NVPTXRegClass outtype, dag surf,
                         list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins B32:$l, B32:$x)),
                inst # " \\{$r\\}, [$s, \\{$l, $x\\}];",
                pattern>;
multiclass SULD_1D_ARRAY<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_ARRAY_base<inst, outtype, (ins B64:$s),
                              [(set outtype:$r,
                                    (intr i64:$s, i32:$l, i32:$x))]>;
  def _I : SULD_1D_ARRAY_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_1D_ARRAY_I8_ # op_upper : SULD_1D_ARRAY<"suld.b.a1d.b8." # op, B16>;
  defm SULD_1D_ARRAY_I16_ # op_upper : SULD_1D_ARRAY<"suld.b.a1d.b16." # op, B16>;
  defm SULD_1D_ARRAY_I32_ # op_upper : SULD_1D_ARRAY<"suld.b.a1d.b32." # op, B32>;
  defm SULD_1D_ARRAY_I64_ # op_upper : SULD_1D_ARRAY<"suld.b.a1d.b64." # op, B64>;
}

class SULD_2D_base<string inst, NVPTXRegClass outtype, dag surf,
                   list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins B32:$x, B32:$y)),
                inst # " \\{$r\\}, [$s, \\{$x, $y\\}];",
                pattern>;
multiclass SULD_2D<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_base<inst, outtype, (ins B64:$s),
                        [(set outtype:$r, (intr i64:$s, i32:$x, i32:$y))]>;
  def _I : SULD_2D_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_2D_I8_ # op_upper : SULD_2D<"suld.b.2d.b8." # op, B16>;
  defm SULD_2D_I16_ # op_upper : SULD_2D<"suld.b.2d.b16." # op, B16>;
  defm SULD_2D_I32_ # op_upper : SULD_2D<"suld.b.2d.b32." # op, B32>;
  defm SULD_2D_I64_ # op_upper : SULD_2D<"suld.b.2d.b64." # op, B64>;
}

class SULD_2D_ARRAY_base<string inst, NVPTXRegClass outtype, dag surf,
                         list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins B32:$l, B32:$x, B32:$y)),
                inst # " \\{$r\\}, [$s, \\{$l, $x, $y, $y\\}];",
                pattern>;
multiclass SULD_2D_ARRAY<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_ARRAY_base<inst, outtype, (ins B64:$s),
                              [(set outtype:$r,
                                    (intr i64:$s, i32:$l, i32:$x, i32:$y))]>;
  def _I : SULD_2D_ARRAY_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_2D_ARRAY_I8_ # op_upper : SULD_2D_ARRAY<"suld.b.a2d.b8." # op, B16>;
  defm SULD_2D_ARRAY_I16_ # op_upper : SULD_2D_ARRAY<"suld.b.a2d.b16." # op, B16>;
  defm SULD_2D_ARRAY_I32_ # op_upper : SULD_2D_ARRAY<"suld.b.a2d.b32." # op, B32>;
  defm SULD_2D_ARRAY_I64_ # op_upper : SULD_2D_ARRAY<"suld.b.a2d.b64." # op, B64>;
}

class SULD_3D_base<string inst, NVPTXRegClass outtype, dag surf,
                   list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r),
                !con(surf, (ins B32:$x, B32:$y, B32:$z)),
                inst # " \\{$r\\}, [$s, \\{$x, $y, $z, $z\\}];",
                pattern>;
multiclass SULD_3D<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_3D_base<inst, outtype, (ins B64:$s),
                        [(set outtype:$r,
                              (intr i64:$s, i32:$x, i32:$y, i32:$z))]>;
  def _I : SULD_3D_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_3D_I8_ # op_upper : SULD_3D<"suld.b.3d.b8." # op, B16>;
  defm SULD_3D_I16_ # op_upper : SULD_3D<"suld.b.3d.b16." # op, B16>;
  defm SULD_3D_I32_ # op_upper : SULD_3D<"suld.b.3d.b32." # op, B32>;
  defm SULD_3D_I64_ # op_upper : SULD_3D<"suld.b.3d.b64." # op, B64>;
}
}

let IsSuld = 2 in {

class SULD_1D_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins B32:$x)),
                inst # " \\{$r, $g\\}, [$s, \\{$x\\}];",
                pattern>;
multiclass SULD_1D_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_V2_base<inst, outtype, (ins B64:$s),
                           [(set outtype:$r, outtype:$g,
                                 (intr i64:$s, i32:$x))]>;
  def _I : SULD_1D_V2_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_1D_V2I8_ # op_upper : SULD_1D_V2<"suld.b.1d.v2.b8." # op, B16>;
  defm SULD_1D_V2I16_ # op_upper : SULD_1D_V2<"suld.b.1d.v2.b16." # op, B16>;
  defm SULD_1D_V2I32_ # op_upper : SULD_1D_V2<"suld.b.1d.v2.b32." # op, B32>;
  defm SULD_1D_V2I64_ # op_upper : SULD_1D_V2<"suld.b.1d.v2.b64." # op, B64>;
}

class SULD_1D_ARRAY_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                            list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins B32:$l, B32:$x)),
                inst # " \\{$r, $g\\}, [$s, \\{$l, $x\\}];",
                pattern>;
multiclass SULD_1D_ARRAY_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_ARRAY_V2_base<inst, outtype, (ins B64:$s),
                                 [(set outtype:$r, outtype:$g,
                                       (intr i64:$s, i32:$l, i32:$x))]>;
  def _I : SULD_1D_ARRAY_V2_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_1D_ARRAY_V2I8_ # op_upper : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b8." # op, B16>;
  defm SULD_1D_ARRAY_V2I16_ # op_upper : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b16." # op, B16>;
  defm SULD_1D_ARRAY_V2I32_ # op_upper : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b32." # op, B32>;
  defm SULD_1D_ARRAY_V2I64_ # op_upper : SULD_1D_ARRAY_V2<"suld.b.a1d.v2.b64." # op, B64>;
}

class SULD_2D_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins B32:$x, B32:$y)),
                inst # " \\{$r, $g\\}, [$s, \\{$x, $y\\}];",
                pattern>;
multiclass SULD_2D_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_V2_base<inst, outtype, (ins B64:$s),
                           [(set outtype:$r, outtype:$g,
                                 (intr i64:$s, i32:$x, i32:$y))]>;
  def _I : SULD_2D_V2_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_2D_V2I8_ # op_upper : SULD_2D_V2<"suld.b.2d.v2.b8." # op, B16>;
  defm SULD_2D_V2I16_ # op_upper : SULD_2D_V2<"suld.b.2d.v2.b16." # op, B16>;
  defm SULD_2D_V2I32_ # op_upper : SULD_2D_V2<"suld.b.2d.v2.b32." # op, B32>;
  defm SULD_2D_V2I64_ # op_upper : SULD_2D_V2<"suld.b.2d.v2.b64." # op, B64>;
}

class SULD_2D_ARRAY_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                            list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins B32:$l, B32:$x, B32:$y)),
                inst # " \\{$r, $g\\}, [$s, \\{$l, $x, $y, $y\\}];",
                pattern>;
multiclass SULD_2D_ARRAY_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_ARRAY_V2_base<inst, outtype, (ins B64:$s),
                                 [(set outtype:$r, outtype:$g,
                                       (intr i64:$s, i32:$l, i32:$x, i32:$y))]>;
  def _I : SULD_2D_ARRAY_V2_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_2D_ARRAY_V2I8_ # op_upper : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b8." # op, B16>;
  defm SULD_2D_ARRAY_V2I16_ # op_upper : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b16." # op, B16>;
  defm SULD_2D_ARRAY_V2I32_ # op_upper : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b32." # op, B32>;
  defm SULD_2D_ARRAY_V2I64_ # op_upper : SULD_2D_ARRAY_V2<"suld.b.a2d.v2.b64." # op, B64>;
}

class SULD_3D_V2_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g),
                !con(surf, (ins B32:$x, B32:$y, B32:$z)),
                inst # " \\{$r, $g\\}, [$s, \\{$x, $y, $z, $z\\}];",
                pattern>;
multiclass SULD_3D_V2<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_3D_V2_base<inst, outtype, (ins B64:$s),
                           [(set outtype:$r, outtype:$g,
                                 (intr i64:$s, i32:$x, i32:$y, i32:$z))]>;
  def _I : SULD_3D_V2_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_3D_V2I8_ # op_upper : SULD_3D_V2<"suld.b.3d.v2.b8." # op, B16>;
  defm SULD_3D_V2I16_ # op_upper : SULD_3D_V2<"suld.b.3d.v2.b16." # op, B16>;
  defm SULD_3D_V2I32_ # op_upper : SULD_3D_V2<"suld.b.3d.v2.b32." # op, B32>;
  defm SULD_3D_V2I64_ # op_upper : SULD_3D_V2<"suld.b.3d.v2.b64." # op, B64>;
}

}

let IsSuld = 3 in {

class SULD_1D_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins B32:$x)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$x\\}];",
                pattern>;
multiclass SULD_1D_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_V4_base<inst, outtype, (ins B64:$s),
                           [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
                                 (intr i64:$s, i32:$x))]>;
  def _I : SULD_1D_V4_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_1D_V4I8_ # op_upper : SULD_1D_V4<"suld.b.1d.v4.b8." # op, B16>;
  defm SULD_1D_V4I16_ # op_upper : SULD_1D_V4<"suld.b.1d.v4.b16." # op, B16>;
  defm SULD_1D_V4I32_ # op_upper : SULD_1D_V4<"suld.b.1d.v4.b32." # op, B32>;
}

class SULD_1D_ARRAY_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                            list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins B32:$l, B32:$x)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$l, $x\\}];",
                pattern>;
multiclass SULD_1D_ARRAY_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_1D_ARRAY_V4_base<inst, outtype, (ins B64:$s),
                                 [(set outtype:$r, outtype:$g, outtype:$b,
                                       outtype:$a,
                                       (intr i64:$s, i32:$l, i32:$x))]>;
  def _I : SULD_1D_ARRAY_V4_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_1D_ARRAY_V4I8_ # op_upper : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b8." # op, B16>;
  defm SULD_1D_ARRAY_V4I16_ # op_upper : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b16." # op, B16>;
  defm SULD_1D_ARRAY_V4I32_ # op_upper : SULD_1D_ARRAY_V4<"suld.b.a1d.v4.b32." # op, B32>;
}

class SULD_2D_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins B32:$x, B32:$y)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$x, $y\\}];",
                pattern>;
multiclass SULD_2D_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_V4_base<inst, outtype, (ins B64:$s),
                           [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
                                 (intr i64:$s, i32:$x, i32:$y))]>;
  def _I : SULD_2D_V4_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_2D_V4I8_ # op_upper : SULD_2D_V4<"suld.b.2d.v4.b8." # op, B16>;
  defm SULD_2D_V4I16_ # op_upper : SULD_2D_V4<"suld.b.2d.v4.b16." # op, B16>;
  defm SULD_2D_V4I32_ # op_upper : SULD_2D_V4<"suld.b.2d.v4.b32." # op, B32>;
}

class SULD_2D_ARRAY_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                            list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins B32:$l, B32:$x, B32:$y)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$l, $x, $y, $y\\}];",
                pattern>;
multiclass SULD_2D_ARRAY_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_2D_ARRAY_V4_base<inst, outtype, (ins B64:$s),
                                 [(set outtype:$r, outtype:$g, outtype:$b,
                                       outtype:$a,
                                       (intr i64:$s, i32:$l, i32:$x, i32:$y))]>;
  def _I : SULD_2D_ARRAY_V4_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_2D_ARRAY_V4I8_ # op_upper : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b8." # op, B16>;
  defm SULD_2D_ARRAY_V4I16_ # op_upper : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b16." # op, B16>;
  defm SULD_2D_ARRAY_V4I32_ # op_upper : SULD_2D_ARRAY_V4<"suld.b.a2d.v4.b32." # op, B32>;
}

class SULD_3D_V4_base<string inst, NVPTXRegClass outtype, dag surf,
                      list<dag> pattern = []>
    : NVPTXInst<(outs outtype:$r, outtype:$g, outtype:$b, outtype:$a),
                !con(surf, (ins B32:$x, B32:$y, B32:$z)),
                inst # " \\{$r, $g, $b, $a\\}, [$s, \\{$x, $y, $z, $z\\}];",
                pattern>;
multiclass SULD_3D_V4<string inst, NVPTXRegClass outtype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SULD_3D_V4_base<inst, outtype, (ins B64:$s),
                           [(set outtype:$r, outtype:$g, outtype:$b, outtype:$a,
                                 (intr i64:$s, i32:$x, i32:$y, i32:$z))]>;
  def _I : SULD_3D_V4_base<inst, outtype, (ins i64imm:$s)>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SULD_3D_V4I8_ # op_upper : SULD_3D_V4<"suld.b.3d.v4.b8." # op, B16>;
  defm SULD_3D_V4I16_ # op_upper : SULD_3D_V4<"suld.b.3d.v4.b16." # op, B16>;
  defm SULD_3D_V4I32_ # op_upper : SULD_3D_V4<"suld.b.3d.v4.b32." # op, B32>;
}

}

//-----------------------------------
// Texture Query Intrinsics
//-----------------------------------

let IsSurfTexQuery = true in {
  foreach query = ["channel_order", "channel_data_type", "width", "height", 
                   "depth", "array_size", "num_samples", "num_mipmap_levels"] in {
    def TXQ_ # !toupper(query) # _R
      : NVPTXInst<(outs B32:$d), (ins B64:$a),
                  "txq." # query # ".b32 \t$d, [$a];",
                  [(set i32:$d, (!cast<Intrinsic>("int_nvvm_txq_" # query) i64:$a))]>;
    def TXQ_ # !toupper(query) # _I
      : NVPTXInst<(outs B32:$d), (ins i64imm:$a),
                  "txq." # query # ".b32 \t$d, [$a];",
                  []>;
  }
}

//-----------------------------------
// Surface Query Intrinsics
//-----------------------------------

let IsSurfTexQuery = true in {
  foreach query = ["channel_order", "channel_data_type", "width", "height", "depth", "array_size"] in {
    def SUQ_ # !toupper(query) # _R
      : NVPTXInst<(outs B32:$d), (ins B64:$a),
                  "suq." # query # ".b32 \t$d, [$a];",
                  [(set i32:$d, (!cast<Intrinsic>("int_nvvm_suq_" # query) i64:$a))]>;
    def SUQ_ # !toupper(query) # _I
      : NVPTXInst<(outs B32:$d), (ins i64imm:$a),
                  "suq." # query # ".b32 \t$d, [$a];",
                  []>;
  }
}

//===- Handle Query -------------------------------------------------------===//

// TODO: These intrinsics are not yet finalized, pending PTX ISA design work
def ISTYPEP_SAMPLER
  : BasicNVPTXInst<(outs B1:$d), (ins B64:$a),
              "istypep.samplerref",
              [(set i1:$d, (int_nvvm_istypep_sampler i64:$a))]>;
def ISTYPEP_SURFACE
  : BasicNVPTXInst<(outs B1:$d), (ins B64:$a),
              "istypep.surfref",
              [(set i1:$d, (int_nvvm_istypep_surface i64:$a))]>;
def ISTYPEP_TEXTURE
  : BasicNVPTXInst<(outs B1:$d), (ins B64:$a),
              "istypep.texref",
              [(set i1:$d, (int_nvvm_istypep_texture i64:$a))]>;

//===- Surface Stores -----------------------------------------------------===//

let IsSust = true in {

class SUST_1D_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$x, intype:$r)),
                inst # " \t[$s, \\{$x\\}], \\{$r\\};", pat>;
multiclass SUST_1D<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));

  def _R : SUST_1D_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$x, intype:$r)]>;
  def _I : SUST_1D_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_1D_I8_ # op_upper : SUST_1D<"sust.b.1d.b8." # op, B16>;
  defm SUST_B_1D_I16_ # op_upper : SUST_1D<"sust.b.1d.b16." # op, B16>;
  defm SUST_B_1D_I32_ # op_upper : SUST_1D<"sust.b.1d.b32." # op, B32>;
  defm SUST_B_1D_I64_ # op_upper : SUST_1D<"sust.b.1d.b64." # op, B64>;
}

defm SUST_P_1D_I8_TRAP : SUST_1D<"sust.p.1d.b8.trap", B16>;
defm SUST_P_1D_I16_TRAP : SUST_1D<"sust.p.1d.b16.trap", B16>;
defm SUST_P_1D_I32_TRAP : SUST_1D<"sust.p.1d.b32.trap", B32>;

class SUST_1D_V2_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$x, intype:$r, intype:$g)),
                inst # " \t[$s, \\{$x\\}], \\{$r, $g\\};",
                pat>;
multiclass SUST_1D_V2<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_1D_V2_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$x, intype:$r, intype:$g)]>;
  def _I : SUST_1D_V2_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_1D_V2I8_ # op_upper : SUST_1D_V2<"sust.b.1d.v2.b8." # op, B16>;
  defm SUST_B_1D_V2I16_ # op_upper : SUST_1D_V2<"sust.b.1d.v2.b16." # op, B16>;
  defm SUST_B_1D_V2I32_ # op_upper : SUST_1D_V2<"sust.b.1d.v2.b32." # op, B32>;
  defm SUST_B_1D_V2I64_ # op_upper : SUST_1D_V2<"sust.b.1d.v2.b64." # op, B64>;
}
defm SUST_P_1D_V2I8_TRAP : SUST_1D_V2<"sust.p.1d.v2.b8.trap", B16>;
defm SUST_P_1D_V2I16_TRAP : SUST_1D_V2<"sust.p.1d.v2.b16.trap", B16>;
defm SUST_P_1D_V2I32_TRAP : SUST_1D_V2<"sust.p.1d.v2.b32.trap", B32>;

class SUST_1D_V4_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$x, intype:$r, intype:$g,
                                intype:$b, intype:$a)),
                inst # " \t[$s, \\{$x\\}], \\{$r, $g, $b, $a\\};",
                pat>;
multiclass SUST_1D_V4<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_1D_V4_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$x, intype:$r, intype:$g,
                intype:$b, intype:$a)]>;
  def _I : SUST_1D_V4_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_1D_V4I8_ # op_upper : SUST_1D_V4<"sust.b.1d.v4.b8." # op, B16>;
  defm SUST_B_1D_V4I16_ # op_upper : SUST_1D_V4<"sust.b.1d.v4.b16." # op, B16>;
  defm SUST_B_1D_V4I32_ # op_upper : SUST_1D_V4<"sust.b.1d.v4.b32." # op, B32>;
}

defm SUST_P_1D_V4I8_TRAP : SUST_1D_V4<"sust.p.1d.v4.b8.trap", B16>;
defm SUST_P_1D_V4I16_TRAP : SUST_1D_V4<"sust.p.1d.v4.b16.trap", B16>;
defm SUST_P_1D_V4I32_TRAP : SUST_1D_V4<"sust.p.1d.v4.b32.trap", B32>;

class SUST_1D_ARRAY_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$idx, B32:$x, intype:$r)),
                inst # " \t[$s, \\{$idx, $x\\}], \\{$r\\};",
                pat>;
multiclass SUST_1D_ARRAY<string inst, NVPTXRegClass intype> {
    defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_1D_ARRAY_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$idx, B32:$x, intype:$r)]>;
  def _I : SUST_1D_ARRAY_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_1D_ARRAY_I8_ # op_upper : SUST_1D_ARRAY<"sust.b.a1d.b8." # op, B16>;
  defm SUST_B_1D_ARRAY_I16_ # op_upper : SUST_1D_ARRAY<"sust.b.a1d.b16." # op, B16>;
  defm SUST_B_1D_ARRAY_I32_ # op_upper : SUST_1D_ARRAY<"sust.b.a1d.b32." # op, B32>;
  defm SUST_B_1D_ARRAY_I64_ # op_upper : SUST_1D_ARRAY<"sust.b.a1d.b64." # op, B64>;
}

defm SUST_P_1D_ARRAY_I8_TRAP : SUST_1D_ARRAY<"sust.p.a1d.b8.trap", B16>;
defm SUST_P_1D_ARRAY_I16_TRAP : SUST_1D_ARRAY<"sust.p.a1d.b16.trap", B16>;
defm SUST_P_1D_ARRAY_I32_TRAP : SUST_1D_ARRAY<"sust.p.a1d.b32.trap", B32>;

class SUST_1D_ARRAY_V2_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$idx, B32:$x,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$idx, $x\\}], \\{$r, $g\\};",
                pat>;
multiclass SUST_1D_ARRAY_V2<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_1D_ARRAY_V2_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$idx, B32:$x,
                intype:$r, intype:$g)]>;
  def _I : SUST_1D_ARRAY_V2_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_1D_ARRAY_V2I8_ # op_upper : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b8." # op, B16>;
  defm SUST_B_1D_ARRAY_V2I16_ # op_upper : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b16." # op, B16>;
  defm SUST_B_1D_ARRAY_V2I32_ # op_upper : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b32." # op, B32>;
  defm SUST_B_1D_ARRAY_V2I64_ # op_upper : SUST_1D_ARRAY_V2<"sust.b.a1d.v2.b64." # op, B64>;
}

defm SUST_P_1D_ARRAY_V2I8_TRAP : SUST_1D_ARRAY_V2<"sust.p.a1d.v2.b8.trap", B16>;
defm SUST_P_1D_ARRAY_V2I16_TRAP : SUST_1D_ARRAY_V2<"sust.p.a1d.v2.b16.trap", B16>;
defm SUST_P_1D_ARRAY_V2I32_TRAP : SUST_1D_ARRAY_V2<"sust.p.a1d.v2.b32.trap", B32>;

class SUST_1D_ARRAY_V4_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$idx, B32:$x,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$idx, $x\\}], \\{$r, $g, $b, $a\\};",
                pat>;
multiclass SUST_1D_ARRAY_V4<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_1D_ARRAY_V4_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$idx, B32:$x,
                intype:$r, intype:$g, intype:$b, intype:$a)]>;
  def _I : SUST_1D_ARRAY_V4_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_1D_ARRAY_V4I8_ # op_upper : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b8." # op, B16>;
  defm SUST_B_1D_ARRAY_V4I16_ # op_upper : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b16." # op, B16>;
  defm SUST_B_1D_ARRAY_V4I32_ # op_upper : SUST_1D_ARRAY_V4<"sust.b.a1d.v4.b32." # op, B32>;
}

defm SUST_P_1D_ARRAY_V4I8_TRAP : SUST_1D_ARRAY_V4<"sust.p.a1d.v4.b8.trap", B16>;
defm SUST_P_1D_ARRAY_V4I16_TRAP : SUST_1D_ARRAY_V4<"sust.p.a1d.v4.b16.trap", B16>;
defm SUST_P_1D_ARRAY_V4I32_TRAP : SUST_1D_ARRAY_V4<"sust.p.a1d.v4.b32.trap", B32>;

class SUST_2D_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$x, B32:$y, intype:$r)),
                inst # " \t[$s, \\{$x, $y\\}], \\{$r\\};",
                pat>;
multiclass SUST_2D<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_2D_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$x, B32:$y, intype:$r)]>;
  def _I : SUST_2D_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_2D_I8_ # op_upper : SUST_2D<"sust.b.2d.b8." # op, B16>;
  defm SUST_B_2D_I16_ # op_upper : SUST_2D<"sust.b.2d.b16." # op, B16>;
  defm SUST_B_2D_I32_ # op_upper : SUST_2D<"sust.b.2d.b32." # op, B32>;
  defm SUST_B_2D_I64_ # op_upper : SUST_2D<"sust.b.2d.b64." # op, B64>;
}

defm SUST_P_2D_I8_TRAP : SUST_2D<"sust.p.2d.b8.trap", B16>;
defm SUST_P_2D_I16_TRAP : SUST_2D<"sust.p.2d.b16.trap", B16>;
defm SUST_P_2D_I32_TRAP : SUST_2D<"sust.p.2d.b32.trap", B32>;

class SUST_2D_V2_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$x, B32:$y,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$x, $y\\}], \\{$r, $g\\};",
                pat>;
multiclass SUST_2D_V2<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_2D_V2_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$x, B32:$y,
                intype:$r, intype:$g)]>;
  def _I : SUST_2D_V2_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_2D_V2I8_ # op_upper : SUST_2D_V2<"sust.b.2d.v2.b8." # op, B16>;
  defm SUST_B_2D_V2I16_ # op_upper : SUST_2D_V2<"sust.b.2d.v2.b16." # op, B16>;
  defm SUST_B_2D_V2I32_ # op_upper : SUST_2D_V2<"sust.b.2d.v2.b32." # op, B32>;
  defm SUST_B_2D_V2I64_ # op_upper : SUST_2D_V2<"sust.b.2d.v2.b64." # op, B64>;
}

defm SUST_P_2D_V2I8_TRAP : SUST_2D_V2<"sust.p.2d.v2.b8.trap", B16>;
defm SUST_P_2D_V2I16_TRAP : SUST_2D_V2<"sust.p.2d.v2.b16.trap", B16>;
defm SUST_P_2D_V2I32_TRAP : SUST_2D_V2<"sust.p.2d.v2.b32.trap", B32>;

class SUST_2D_V4_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$x, B32:$y,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$x, $y\\}], \\{$r, $g, $b, $a\\};",
                pat>;
multiclass SUST_2D_V4<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_2D_V4_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$x, B32:$y,
                intype:$r, intype:$g, intype:$b, intype:$a)]>;
  def _I : SUST_2D_V4_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_2D_V4I8_ # op_upper : SUST_2D_V4<"sust.b.2d.v4.b8." # op, B16>;
  defm SUST_B_2D_V4I16_ # op_upper : SUST_2D_V4<"sust.b.2d.v4.b16." # op, B16>;
  defm SUST_B_2D_V4I32_ # op_upper : SUST_2D_V4<"sust.b.2d.v4.b32." # op, B32>;
}

defm SUST_P_2D_V4I8_TRAP : SUST_2D_V4<"sust.p.2d.v4.b8.trap", B16>;
defm SUST_P_2D_V4I16_TRAP : SUST_2D_V4<"sust.p.2d.v4.b16.trap", B16>;
defm SUST_P_2D_V4I32_TRAP : SUST_2D_V4<"sust.p.2d.v4.b32.trap", B32>;

class SUST_2D_ARRAY_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$idx, B32:$x, B32:$y,
                                intype:$r)),
                inst # " \t[$s, \\{$idx, $x, $y, $y\\}], \\{$r\\};",
                pat>;
multiclass SUST_2D_ARRAY<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_2D_ARRAY_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$idx, B32:$x, B32:$y,
                intype:$r)]>;
  def _I : SUST_2D_ARRAY_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_2D_ARRAY_I8_ # op_upper : SUST_2D_ARRAY<"sust.b.a2d.b8." # op, B16>;
  defm SUST_B_2D_ARRAY_I16_ # op_upper : SUST_2D_ARRAY<"sust.b.a2d.b16." # op, B16>;
  defm SUST_B_2D_ARRAY_I32_ # op_upper : SUST_2D_ARRAY<"sust.b.a2d.b32." # op, B32>;
  defm SUST_B_2D_ARRAY_I64_ # op_upper : SUST_2D_ARRAY<"sust.b.a2d.b64." # op, B64>;
}

defm SUST_P_2D_ARRAY_I8_TRAP : SUST_2D_ARRAY<"sust.p.a2d.b8.trap", B16>;
defm SUST_P_2D_ARRAY_I16_TRAP : SUST_2D_ARRAY<"sust.p.a2d.b16.trap", B16>;
defm SUST_P_2D_ARRAY_I32_TRAP : SUST_2D_ARRAY<"sust.p.a2d.b32.trap", B32>;

class SUST_2D_ARRAY_V2_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$idx, B32:$x, B32:$y,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$idx, $x, $y, $y\\}], \\{$r, $g\\};",
                pat>;
multiclass SUST_2D_ARRAY_V2<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_2D_ARRAY_V2_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$idx, B32:$x, B32:$y,
                intype:$r, intype:$g)]>;
  def _I : SUST_2D_ARRAY_V2_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_2D_ARRAY_V2I8_ # op_upper : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b8." # op, B16>;
  defm SUST_B_2D_ARRAY_V2I16_ # op_upper : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b16." # op, B16>;
  defm SUST_B_2D_ARRAY_V2I32_ # op_upper : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b32." # op, B32>;
  defm SUST_B_2D_ARRAY_V2I64_ # op_upper : SUST_2D_ARRAY_V2<"sust.b.a2d.v2.b64." # op, B64>;
}

defm SUST_P_2D_ARRAY_V2I8_TRAP : SUST_2D_ARRAY_V2<"sust.p.a2d.v2.b8.trap", B16>;
defm SUST_P_2D_ARRAY_V2I16_TRAP : SUST_2D_ARRAY_V2<"sust.p.a2d.v2.b16.trap", B16>;
defm SUST_P_2D_ARRAY_V2I32_TRAP : SUST_2D_ARRAY_V2<"sust.p.a2d.v2.b32.trap", B32>;

class SUST_2D_ARRAY_V4_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$idx, B32:$x, B32:$y,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$idx, $x, $y, $y\\}], \\{$r, $g, $b, $a\\};",
                pat>;
multiclass SUST_2D_ARRAY_V4<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_2D_ARRAY_V4_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$idx, B32:$x, B32:$y,
                intype:$r, intype:$g, intype:$b, intype:$a)]>;
  def _I : SUST_2D_ARRAY_V4_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_2D_ARRAY_V4I8_ # op_upper : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b8." # op, B16>;
  defm SUST_B_2D_ARRAY_V4I16_ # op_upper : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b16." # op, B16>;
  defm SUST_B_2D_ARRAY_V4I32_ # op_upper : SUST_2D_ARRAY_V4<"sust.b.a2d.v4.b32." # op, B32>;
}

defm SUST_P_2D_ARRAY_V4I8_TRAP : SUST_2D_ARRAY_V4<"sust.p.a2d.v4.b8.trap", B16>;
defm SUST_P_2D_ARRAY_V4I16_TRAP : SUST_2D_ARRAY_V4<"sust.p.a2d.v4.b16.trap", B16>;
defm SUST_P_2D_ARRAY_V4I32_TRAP : SUST_2D_ARRAY_V4<"sust.p.a2d.v4.b32.trap", B32>;

class SUST_3D_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$x, B32:$y, B32:$z,
                                intype:$r)),
                inst # " \t[$s, \\{$x, $y, $z, $z\\}], \\{$r\\};",
                pat>;
multiclass SUST_3D<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_3D_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$x, B32:$y, B32:$z,
                intype:$r)]>;
  def _I : SUST_3D_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_3D_I8_ # op_upper : SUST_3D<"sust.b.3d.b8." # op, B16>;
  defm SUST_B_3D_I16_ # op_upper : SUST_3D<"sust.b.3d.b16." # op, B16>;
  defm SUST_B_3D_I32_ # op_upper : SUST_3D<"sust.b.3d.b32." # op, B32>;
  defm SUST_B_3D_I64_ # op_upper : SUST_3D<"sust.b.3d.b64." # op, B64>;
}
defm SUST_P_3D_I8_TRAP : SUST_3D<"sust.p.3d.b8.trap", B16>;
defm SUST_P_3D_I16_TRAP : SUST_3D<"sust.p.3d.b16.trap", B16>;
defm SUST_P_3D_I32_TRAP : SUST_3D<"sust.p.3d.b32.trap", B32>;

class SUST_3D_V2_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$x, B32:$y, B32:$z,
                                intype:$r, intype:$g)),
                inst # " \t[$s, \\{$x, $y, $z, $z\\}], \\{$r, $g\\};",
                pat>;
multiclass SUST_3D_V2<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_3D_V2_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$x, B32:$y, B32:$z,
                intype:$r, intype:$g)]>;
  def _I : SUST_3D_V2_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_3D_V2I8_ # op_upper : SUST_3D_V2<"sust.b.3d.v2.b8." # op, B16>;
  defm SUST_B_3D_V2I16_ # op_upper : SUST_3D_V2<"sust.b.3d.v2.b16." # op, B16>;
  defm SUST_B_3D_V2I32_ # op_upper : SUST_3D_V2<"sust.b.3d.v2.b32." # op, B32>;
  defm SUST_B_3D_V2I64_ # op_upper : SUST_3D_V2<"sust.b.3d.v2.b64." # op, B64>;
}
defm SUST_P_3D_V2I8_TRAP : SUST_3D_V2<"sust.p.3d.v2.b8.trap", B16>;
defm SUST_P_3D_V2I16_TRAP : SUST_3D_V2<"sust.p.3d.v2.b16.trap", B16>;
defm SUST_P_3D_V2I32_TRAP : SUST_3D_V2<"sust.p.3d.v2.b32.trap", B32>;

class SUST_3D_V4_base<string inst, NVPTXRegClass intype, dag surf, list<dag> pat>
    : NVPTXInst<(outs),
                !con(surf, (ins B32:$x, B32:$y, B32:$z,
                                intype:$r, intype:$g, intype:$b, intype:$a)),
                inst # " \t[$s, \\{$x, $y, $z, $z\\}], \\{$r, $g, $b, $a\\};",
                pat>;
multiclass SUST_3D_V4<string inst, NVPTXRegClass intype> {
  defvar intr = !cast<Intrinsic>("int_nvvm_" # !tolower(NAME));
  def _R : SUST_3D_V4_base<inst, intype, (ins B64:$s),
              [(intr B64:$s, B32:$x, B32:$y, B32:$z,
                intype:$r, intype:$g, intype:$b, intype:$a)]>;
  def _I : SUST_3D_V4_base<inst, intype, (ins i64imm:$s), []>;
}

foreach op = ["clamp", "trap", "zero"] in {
  defvar op_upper = !toupper(op);
  defm SUST_B_3D_V4I8_ # op_upper : SUST_3D_V4<"sust.b.3d.v4.b8." # op, B16>;
  defm SUST_B_3D_V4I16_ # op_upper : SUST_3D_V4<"sust.b.3d.v4.b16." # op, B16>;
  defm SUST_B_3D_V4I32_ # op_upper : SUST_3D_V4<"sust.b.3d.v4.b32." # op, B32>;
}

defm SUST_P_3D_V4I8_TRAP : SUST_3D_V4<"sust.p.3d.v4.b8.trap", B16>;
defm SUST_P_3D_V4I16_TRAP : SUST_3D_V4<"sust.p.3d.v4.b16.trap", B16>;
defm SUST_P_3D_V4I32_TRAP : SUST_3D_V4<"sust.p.3d.v4.b32.trap", B32>;

}


//-----------------------------------
// Read Special Registers
//-----------------------------------

class PTX_READ_SREG_R64<string regname, Intrinsic intop, list<Predicate> Preds=[]>
  : NVPTXInst<(outs B64:$d), (ins),
              "mov.u64 \t$d, %" # regname # ";",
              [(set i64:$d, (intop))]>,
    Requires<Preds>;

class PTX_READ_SREG_R32<string regname, Intrinsic intop, list<Predicate> Preds=[]>
  : NVPTXInst<(outs B32:$d), (ins),
              "mov.u32 \t$d, %" # regname # ";",
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


def SREG_LANEID : PTX_READ_SREG_R32<"laneid", int_nvvm_read_ptx_sreg_laneid>;
def SREG_WARPID : PTX_READ_SREG_R32<"warpid", int_nvvm_read_ptx_sreg_warpid>;
def SREG_NWARPID : PTX_READ_SREG_R32<"nwarpid", int_nvvm_read_ptx_sreg_nwarpid>;
def SREG_SMID : PTX_READ_SREG_R32<"smid", int_nvvm_read_ptx_sreg_smid>;
def SREG_NSMID : PTX_READ_SREG_R32<"nsmid", int_nvvm_read_ptx_sreg_nsmid>;
def SREG_GRIDID : PTX_READ_SREG_R32<"gridid", int_nvvm_read_ptx_sreg_gridid>;

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
  def SREG_CLOCK : PTX_READ_SREG_R32<"clock", int_nvvm_read_ptx_sreg_clock>;
  def SREG_CLOCK64 : PTX_READ_SREG_R64<"clock64", int_nvvm_read_ptx_sreg_clock64>;
  def SREG_GLOBALTIMER : PTX_READ_SREG_R64<"globaltimer", int_nvvm_read_ptx_sreg_globaltimer>;
  def SREG_GLOBALTIMER_LO : PTX_READ_SREG_R32<"globaltimer_lo", int_nvvm_read_ptx_sreg_globaltimer_lo>;
}

def: Pat <(i64 (readcyclecounter)), (SREG_CLOCK64)>;
def: Pat <(i64 (readsteadycounter)), (SREG_GLOBALTIMER)>;
def: Pat <(i32 (readsteadycounter)), (SREG_GLOBALTIMER_LO)>;

def INT_PTX_SREG_PM0 : PTX_READ_SREG_R32<"pm0", int_nvvm_read_ptx_sreg_pm0>;
def INT_PTX_SREG_PM1 : PTX_READ_SREG_R32<"pm1", int_nvvm_read_ptx_sreg_pm1>;
def INT_PTX_SREG_PM2 : PTX_READ_SREG_R32<"pm2", int_nvvm_read_ptx_sreg_pm2>;
def INT_PTX_SREG_PM3 : PTX_READ_SREG_R32<"pm3", int_nvvm_read_ptx_sreg_pm3>;

// TODO: It would be nice to use PTX_READ_SREG here, but it doesn't
// handle the constant.
def INT_PTX_SREG_WARPSIZE :
    NVPTXInst<(outs B32:$dst), (ins), "mov.u32 \t$dst, WARP_SZ;",
              [(set i32:$dst, (int_nvvm_read_ptx_sreg_warpsize))]>;

// Helper class that represents a 'fragment' of an NVPTX *MMA instruction.
// In addition to target-independent fields provided by WMMA_REGS, it adds
// the fields commonly used to implement specific PTX instruction -- register
// types and names, constraints, parts of assembly, etc.
class WMMA_REGINFO<WMMA_REGS r, string op, string metadata = "", string kind = "">
      : WMMA_REGS<r.geom, r.frag, r.ptx_elt_type, !eq(op, "mma.sp")> {
  // NVPTX register types used to carry fragment data.
  NVPTXRegClass regclass = !cond(
    !eq(ptx_elt_type, "e4m3") : B32,
    !eq(ptx_elt_type, "e5m2") : B32,
    !eq(ptx_elt_type, "e3m2") : B32,
    !eq(ptx_elt_type, "e2m3") : B32,
    !eq(ptx_elt_type, "e2m1") : B32,
    !eq(ptx_elt_type, "f16") : B32,
    !eq(ptx_elt_type, "f32") : B32,
    !eq(ptx_elt_type, "f64") : B64,
    !eq(ptx_elt_type, "bf16") : B32,
    !eq(ptx_elt_type, "tf32") : B32,
    !eq(ptx_elt_type, "s32") : B32,
    !eq(ptx_elt_type, "b16") : B32,
    !eq(ptx_elt_type, "b8") : B32,
    !eq(ptx_elt_type, "b8x16.b6x16_p32") : B32,
    !eq(ptx_elt_type, "b8x16.b4x16_p64") : B32,
    !eq(ptx_elt_type, "s8") : B32,
    !eq(ptx_elt_type, "u8") : B32,
    !eq(ptx_elt_type, "s4") : B32,
    !eq(ptx_elt_type, "u4") : B32,
    !eq(ptx_elt_type, "b1") : B32);

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
    !or(!eq(ptx_elt_type, "e3m2"),
        !eq(ptx_elt_type, "e2m3"),
        !eq(ptx_elt_type, "e2m1"),
        !ne(kind, "")) : [hasSM120a, hasPTX<87>],

    !or(!eq(ptx_elt_type, "e4m3"),
        !eq(ptx_elt_type, "e5m2")) : [hasSM<89>, hasPTX<84>],

    !and(!eq(op, "mma.sp"),
         !ne(metadata, "sp")) : [hasSM<80>, hasPTX<85>],
    !eq(op, "mma.sp") : [hasSM<80>, hasPTX<71>],

    // fp16 -> fp16/fp32 @ m16n16k16
    !and(!eq(geom, "m16n16k16"),
         !or(!eq(ptx_elt_type, "f16"),
             !eq(ptx_elt_type, "f32"))) : [hasSM<70>, hasPTX<60>],

    !and(!eq(geom, "m8n8k4"),
         !eq(ptx_elt_type, "f64")) : [hasSM<80>, hasPTX<70>],

    // fp16 -> fp16/fp32 @ m8n32k16/m32n8k16
    !and(!or(!eq(geom, "m8n32k16"),
             !eq(geom, "m32n8k16")),
         !or(!eq(ptx_elt_type, "f16"),
             !eq(ptx_elt_type, "f32"))) : [hasSM<70>, hasPTX<61>],

    // u8/s8 -> s32 @ m16n16k16/m8n32k16/m32n8k16
    !and(!or(!eq(geom, "m16n16k16"),
             !eq(geom, "m8n32k16"),
             !eq(geom, "m32n8k16")),
         !or(!eq(ptx_elt_type, "u8"),
             !eq(ptx_elt_type, "s8"),
             !eq(ptx_elt_type, "s32"))) : [hasSM<72>, hasPTX<63>],

    !and(!or(!eq(geom, "m16n16k16"),
             !eq(geom, "m8n32k16"),
             !eq(geom, "m32n8k16")),
         !eq(ptx_elt_type, "bf16")) : [hasSM<80>, hasPTX<70>],

    !and(!eq(geom, "m16n16k8"),
         !eq(ptx_elt_type, "tf32")) : [hasSM<80>, hasPTX<70>],

    !and(!eq(geom, "m16n16k8"),
         !eq(ptx_elt_type, "f32")) : [hasSM<80>, hasPTX<70>],

    // b1 -> s32 @ m8n8k128(b1)
    !and(!ne(op, "mma"),
         !eq(geom, "m8n8k128")) : [hasSM<75>, hasPTX<63>],

    // u4/s4 -> s32 @ m8n8k32 (u4/s4)
    !and(!ne(op, "mma"),
         !eq(geom, "m8n8k32")) : [hasSM<75>, hasPTX<63>],

    !or(!eq(geom, "m16n8k8"),
        !eq(geom, "m8n8k16")) : [hasSM<75>, hasPTX<65>],

    !and(!ne(ptx_elt_type, "f64"),
         !eq(geom, "m8n8k4")) : [hasSM<70>, hasPTX<64>],

    // mma m8n8k32 requires higher PTX version
    !and(!eq(op, "mma"),
         !eq(geom, "m8n8k32")) : [hasSM<75>, hasPTX<65>],

    !and(!eq(ptx_elt_type, "f64"),
         !eq(geom, "m8n8k4")) : [hasSM<80>, hasPTX<70>],

    !and(!eq(op, "mma"),
         !or(!eq(geom, "m16n8k16"),
             !eq(geom, "m16n8k4"),
             !eq(geom, "m16n8k32"),
             !eq(geom, "m16n8k64"),
             !eq(geom, "m8n8k128"),
             !eq(geom, "m16n8k128"),
             !eq(geom, "m16n8k256"))) : [hasSM<80>, hasPTX<70>],

    !and(!eq(op, "ldmatrix"),
         !eq(ptx_elt_type, "b16"),
         !eq(geom, "m8n8")) : [hasSM<75>, hasPTX<65>],

    !and(!eq(op, "ldmatrix"),
         !eq(ptx_elt_type, "b8"),
         !eq(geom, "m16n16")) : [hasSM<100>, hasArchAccelFeatures, hasPTX<86>],

    !and(!eq(op, "ldmatrix"),
         !eq(ptx_elt_type, "b8x16.b6x16_p32"),
         !eq(geom, "m16n16")) : [hasSM<100>, hasArchAccelFeatures, hasPTX<86>],

    !and(!eq(op, "ldmatrix"),
         !eq(ptx_elt_type, "b8x16.b4x16_p64"),
         !eq(geom, "m16n16")) : [hasSM<100>, hasArchAccelFeatures, hasPTX<86>],

    !and(!eq(op, "ldmatrix"),
         !eq(ptx_elt_type, "b8x16.b6x16_p32"),
         !eq(geom, "m8n16")) : [hasSM<100>, hasArchAccelFeatures, hasPTX<86>],

    !and(!eq(op, "ldmatrix"),
         !eq(ptx_elt_type, "b8x16.b4x16_p64"),
         !eq(geom, "m8n16")) : [hasSM<100>, hasArchAccelFeatures, hasPTX<86>],

    !and(!eq(op, "stmatrix"),!eq(ptx_elt_type, "b16"),
         !eq(geom, "m8n8")) : [hasSM<90>, hasPTX<78>],

    !and(!eq(op, "stmatrix"),
         !eq(ptx_elt_type, "b8"),
         !eq(geom, "m16n8")) : [hasSM<100>, hasArchAccelFeatures, hasPTX<86>]);

  // template DAGs for instruction inputs/output.
  dag Outs = !dag(outs, ptx_regs, reg_names);
  dag Ins = !dag(ins, ptx_regs, reg_names);
}

// Convert dag of arguments into a dag to match given intrinsic.
class BuildPatternI<Intrinsic Intr, dag Ins> {
  // Build a dag pattern that matches the intrinsic call.
  dag ret = !foreach(tmp, Ins,
                          !subst(ADDR, addr,
                          !subst(ins, Intr,
                          !subst(i32imm, timm, tmp))));
}

// Same as above, but uses PatFrag instead of an Intrinsic.
class BuildPatternPF<PatFrag Intr, dag Ins> {
  // Build a dag pattern that matches the intrinsic call.
  dag ret = !foreach(tmp, Ins,
                          !subst(ADDR, addr,
                          !subst(ins, Intr, tmp)));
}

// Common WMMA-related fields used for building patterns for all MMA instructions.
class WMMA_INSTR<string _Intr, list<dag> _Args>
  : NVPTXInst<(outs), (ins), "?", []> {
  Intrinsic Intr = !cast<Intrinsic>(_Intr);
  // Concatenate all arguments into a single dag.
  dag Args = !foldl((ins), _Args, a, b, !con(a, b));
  // Pre-build the pattern to match (intrinsic arg0, arg1, ...).
  dag IntrinsicPattern = BuildPatternI<!cast<Intrinsic>(Intr), Args>.ret;
}

//
// wmma.load.[a|b|c].sync.[row|col].m16n16k16[|.global|.shared].[f16|f32]
//

class WMMA_LOAD<WMMA_REGINFO Frag, string Layout, string Space, bit WithStride>
  : WMMA_INSTR<WMMA_NAME_LDST<"load", Frag, Layout, WithStride>.record,
                              [!con((ins ADDR:$src),
                                    !if(WithStride, (ins B32:$ldm), (ins)))]>,
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
                   bit WithStride>
  : WMMA_INSTR<WMMA_NAME_LDST<"store", Frag, Layout, WithStride>.record,
               [!con((ins ADDR:$dst),
                     Frag.Ins,
                     !if(WithStride, (ins B32:$ldm), (ins)))]>,
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
        foreach frag = NVVM_MMA_OPS.all_ld_ops in
          if NVVM_WMMA_LDST_SUPPORTED<frag, layout>.ret then
            def : WMMA_LOAD<WMMA_REGINFO<frag, "load">, layout, space, stride>;
        foreach frag = NVVM_MMA_OPS.all_st_ops in
          if NVVM_WMMA_LDST_SUPPORTED<frag, layout>.ret then
            def : WMMA_STORE_D<WMMA_REGINFO<frag, "store">, layout, space, stride>;
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
    !if(!eq(b1op, ".and.popc"), [hasSM<80>, hasPTX<71>], [])
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

// MMA SP
class MMA_SP<WMMA_REGINFO FragA, WMMA_REGINFO FragB,
             WMMA_REGINFO FragC, WMMA_REGINFO FragD,
             string Metadata, string Kind, int Satfinite>
  : WMMA_INSTR<MMA_SP_NAME<Metadata, Kind, Satfinite,
                           FragA, FragB, FragC, FragD>.record,
               [FragA.Ins, FragB.Ins, FragC.Ins,
                (ins B32:$metadata, i32imm:$selector)]>,
    // Requires does not seem to have effect on Instruction w/o Patterns.
    // We set it here anyways and propagate to the Pat<> we construct below.
    Requires<!listconcat(FragA.Predicates,
                         FragB.Predicates,
                         FragC.Predicates,
                         FragD.Predicates)> {
  let OutOperandList = FragD.Outs;
  let InOperandList = !con(Args, (ins MmaCode:$ptx));
  string TypeList = "." # FragD.ptx_elt_type
                    # "." # FragA.ptx_elt_type
                    # "." # FragB.ptx_elt_type
                    # "." # FragC.ptx_elt_type;
  let AsmString = "mma"
                  # "." # Metadata
                  # ".sync.aligned."
                  # FragA.geom
                  # ".row.col"
                  # !if(!ne(Kind, ""), "." # Kind, "")
                  # !if(Satfinite, ".satfinite", "")
                  # TypeList # "\n\t\t"
                  # FragD.regstring # ",\n\t\t"
                  # FragA.regstring # ",\n\t\t"
                  # FragB.regstring # ",\n\t\t"
                  # FragC.regstring # ",\n\t\t"
                  # "$metadata" # ",\n\t\t"
                  # "$selector" # ";";
}

let isConvergent = true in {
defset list<WMMA_INSTR> MMA_SPs = {
  foreach metadata = ["sp", "sp::ordered_metadata"] in {
    foreach kind = ["", "kind::f8f6f4"] in {
      foreach satf = [0, 1] in {
        foreach op = NVVM_MMA_OPS.all_mma_sp_ops in {
          if NVVM_MMA_SP_SUPPORTED<op, metadata, kind, satf>.ret then {
            def : MMA_SP<WMMA_REGINFO<op[0], "mma.sp", metadata, kind>,
                         WMMA_REGINFO<op[1], "mma.sp", metadata, kind>,
                         WMMA_REGINFO<op[2], "mma.sp", metadata, kind>,
                         WMMA_REGINFO<op[3], "mma.sp", metadata, kind>,
                         metadata, kind, satf>;
          }
        } // op
      } // satf
    } // kind
  } // metadata
} // defset
}

//
// ldmatrix.sync.aligned.m8n8[|.trans][|.shared].b16
//
class LDMATRIX<WMMA_REGINFO Frag, bit Transposed, string Space>
  : WMMA_INSTR<LDMATRIX_NAME<Frag, Transposed>.record, [(ins ADDR:$src)]>,
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
      foreach frag = NVVM_MMA_OPS.all_ldmatrix_ops in
        if NVVM_LDMATRIX_SUPPORTED<frag, transposed>.ret then
          def : LDMATRIX<WMMA_REGINFO<frag, "ldmatrix">, transposed, space>;
    } // space
  } // transposed
} // defset

//
// stmatrix.sync.aligned.m8n8[|.trans][|.shared].b16
//
class STMATRIX<WMMA_REGINFO Frag, bit Transposed, string Space>
  : WMMA_INSTR<STMATRIX_NAME<Frag, Transposed>.record, [!con((ins ADDR:$dst), Frag.Ins)]>,
    Requires<Frag.Predicates> {
  // Build PatFrag that only matches particular address space.
  dag PFOperands = !con((ops node:$dst),
                        !dag(ops, !listsplat(node, !size(Frag.regs)), Frag.reg_names));
  PatFrag IntrFrag = PatFrag<PFOperands,
                             !foreach(tmp, PFOperands, !subst(ops, Intr, tmp)),
                             !cond(!eq(Space, ".shared"): AS_match.shared,
                                   true: AS_match.generic)>;
  // Build AS-constrained pattern.
  let IntrinsicPattern = BuildPatternPF<IntrFrag, Args>.ret;
  let OutOperandList = (outs);
  let InOperandList = !con(Args, (ins MmaCode:$ptx));
  let AsmString = "stmatrix.sync.aligned."
                  # Frag.geom
                  # "." # Frag.frag
                  # !if(Transposed, ".trans", "")
                  # Space
                  # "." # Frag.ptx_elt_type
                  # " [$dst], " # Frag.regstring # ";";
}

// Create all stmatrix variants
defset list<WMMA_INSTR> STMATRIXs = {
  foreach transposed = [false, true] in {foreach space = [".shared", ""] in {
      foreach frag = NVVM_MMA_OPS.all_stmatrix_ops in
        if NVVM_STMATRIX_SUPPORTED<frag, transposed>.ret then
          def : STMATRIX<WMMA_REGINFO<frag, "stmatrix">, transposed, space>;
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
foreach mma = !listconcat(MMAs, WMMAs, MMA_LDSTs, LDMATRIXs, STMATRIXs, MMA_SPs) in
  def : MMA_PAT<mma>;

multiclass MAPA<string suffix, Intrinsic Intr> {
  let Predicates = [hasSM<90>, hasPTX<78>] in {
    def _32: BasicNVPTXInst<(outs B32:$d), (ins B32:$a, B32:$b),
                "mapa" # suffix # ".u32",
                [(set i32:$d, (Intr i32:$a, i32:$b))]>;
    def _32i: BasicNVPTXInst<(outs B32:$d), (ins B32:$a, i32imm:$b),
                "mapa" # suffix # ".u32",
                [(set i32:$d, (Intr i32:$a, imm:$b))]>;
    def _64: BasicNVPTXInst<(outs B64:$d), (ins B64:$a, B32:$b),
                "mapa" # suffix # ".u64",
                [(set i64:$d, (Intr i64:$a, i32:$b))]>;
    def _64i: BasicNVPTXInst<(outs B64:$d), (ins B64:$a, i32imm:$b),
                "mapa" # suffix # ".u64",
                [(set i64:$d, (Intr i64:$a, imm:$b))]>;
  }
}


defm mapa  : MAPA<"", int_nvvm_mapa>;
defm mapa_shared_cluster  : MAPA<".shared::cluster", int_nvvm_mapa_shared_cluster>;


multiclass GETCTARANK<string suffix, Intrinsic Intr> {
  let Predicates = [hasSM<90>, hasPTX<78>] in {
    def _32: BasicNVPTXInst<(outs B32:$d), (ins B32:$a),
                "getctarank" # suffix # ".u32",
                [(set i32:$d, (Intr i32:$a))]>;
    def _64: BasicNVPTXInst<(outs B32:$d), (ins B64:$a),
                "getctarank" # suffix # ".u64",
                [(set i32:$d, (Intr i64:$a))]>;
  }
}

defm getctarank  : GETCTARANK<"", int_nvvm_getctarank>;
defm getctarank_shared_cluster  : GETCTARANK<".shared::cluster", int_nvvm_getctarank_shared_cluster>;

def is_explicit_cluster: NVPTXInst<(outs B1:$d), (ins),
              "mov.pred\t$d, %is_explicit_cluster;",
              [(set i1:$d, (int_nvvm_is_explicit_cluster))]>,
    Requires<[hasSM<90>, hasPTX<78>]>;

// setmaxnreg inc/dec intrinsics
let isConvergent = true in {
multiclass SET_MAXNREG<string Action, Intrinsic Intr> {
  def : BasicNVPTXInst<(outs), (ins i32imm:$reg_count),
          "setmaxnreg." # Action # ".sync.aligned.u32",
          [(Intr timm:$reg_count)]>,
    Requires<[hasArchAccelFeatures, hasSM<90>, hasPTX<80>]>;
}

defm INT_SET_MAXNREG_INC : SET_MAXNREG<"inc", int_nvvm_setmaxnreg_inc_sync_aligned_u32>;
defm INT_SET_MAXNREG_DEC : SET_MAXNREG<"dec", int_nvvm_setmaxnreg_dec_sync_aligned_u32>;

} // isConvergent

//
// WGMMA fence instructions
//
let isConvergent = true, Predicates = [hasSM90a, hasPTX<80>] in {
  def WGMMA_FENCE_SYNC_ALIGNED : NullaryInst<"wgmma.fence.sync.aligned", int_nvvm_wgmma_fence_sync_aligned>;

  def WGMMA_COMMIT_GROUP_SYNC_ALIGNED : NullaryInst<"wgmma.commit_group.sync.aligned", int_nvvm_wgmma_commit_group_sync_aligned>;

  def WGMMA_WAIT_GROUP_SYNC_ALIGNED : BasicNVPTXInst<(outs), (ins i64imm:$n), "wgmma.wait_group.sync.aligned",
                              [(int_nvvm_wgmma_wait_group_sync_aligned timm:$n)]>;
}

let Predicates = [hasSM<90>, hasPTX<78>] in {
  def GRIDDEPCONTROL_LAUNCH_DEPENDENTS :
        NullaryInst<"griddepcontrol.launch_dependents", int_nvvm_griddepcontrol_launch_dependents>;
  def GRIDDEPCONTROL_WAIT :
        NullaryInst<"griddepcontrol.wait", int_nvvm_griddepcontrol_wait>;
}

def EXIT : NullaryInst<"exit", int_nvvm_exit>;

// Tcgen05 intrinsics
let isConvergent = true, Predicates = [hasTcgen05Instructions] in {

multiclass TCGEN05_ALLOC_INTR<string AS, string num, Intrinsic Intr> {
  def "" : BasicNVPTXInst<(outs),
             (ins ADDR:$dst, B32:$ncols),
             "tcgen05.alloc.cta_group::" # num # ".sync.aligned" # AS # ".b32",
             [(Intr addr:$dst, B32:$ncols)]>;
}

defm TCGEN05_ALLOC_CG1 : TCGEN05_ALLOC_INTR<"", "1", int_nvvm_tcgen05_alloc_cg1>;
defm TCGEN05_ALLOC_CG2 : TCGEN05_ALLOC_INTR<"", "2", int_nvvm_tcgen05_alloc_cg2>;

defm TCGEN05_ALLOC_S64_CG1 : TCGEN05_ALLOC_INTR<".shared::cta", "1", int_nvvm_tcgen05_alloc_shared_cg1>;
defm TCGEN05_ALLOC_S64_CG2 : TCGEN05_ALLOC_INTR<".shared::cta", "2", int_nvvm_tcgen05_alloc_shared_cg2>;

multiclass TCGEN05_DEALLOC_INTR<string num, Intrinsic Intr> {
  def "" : BasicNVPTXInst<(outs),
             (ins B32:$tmem_addr, B32:$ncols),
             "tcgen05.dealloc.cta_group::" # num # ".sync.aligned.b32",
             [(Intr B32:$tmem_addr, B32:$ncols)]>;
}
defm TCGEN05_DEALLOC_CG1: TCGEN05_DEALLOC_INTR<"1", int_nvvm_tcgen05_dealloc_cg1>;
defm TCGEN05_DEALLOC_CG2: TCGEN05_DEALLOC_INTR<"2", int_nvvm_tcgen05_dealloc_cg2>;

multiclass TCGEN05_RELINQ_PERMIT_INTR<string num, Intrinsic Intr> {
  def "" : NullaryInst<"tcgen05.relinquish_alloc_permit.cta_group::" # num # ".sync.aligned", Intr>;
}
defm TCGEN05_RELINQ_CG1: TCGEN05_RELINQ_PERMIT_INTR<"1", int_nvvm_tcgen05_relinq_alloc_permit_cg1>;
defm TCGEN05_RELINQ_CG2: TCGEN05_RELINQ_PERMIT_INTR<"2", int_nvvm_tcgen05_relinq_alloc_permit_cg2>;

def tcgen05_wait_ld: NullaryInst<"tcgen05.wait::ld.sync.aligned", int_nvvm_tcgen05_wait_ld>;
def tcgen05_wait_st: NullaryInst<"tcgen05.wait::st.sync.aligned", int_nvvm_tcgen05_wait_st>;

multiclass TCGEN05_COMMIT_INTR<string AS, string num> {
  defvar prefix = "tcgen05.commit.cta_group::" # num #".mbarrier::arrive::one.shared::cluster";

  defvar intr_suffix = !if(!eq(AS, "shared"), "_shared", "") # "_cg" # num;
  defvar Intr = !cast<Intrinsic>("int_nvvm_tcgen05_commit" # intr_suffix);
  defvar IntrMC = !cast<Intrinsic>("int_nvvm_tcgen05_commit_mc" # intr_suffix);

  def "" : BasicNVPTXInst<(outs), (ins ADDR:$mbar),
             prefix # ".b64",
             [(Intr addr:$mbar)]>;
  def _MC : BasicNVPTXInst<(outs), (ins ADDR:$mbar, B16:$mc),
                   prefix # ".multicast::cluster.b64",
                   [(IntrMC addr:$mbar, B16:$mc)]>;
}

defm TCGEN05_COMMIT_CG1 : TCGEN05_COMMIT_INTR<"", "1">;
defm TCGEN05_COMMIT_CG2 : TCGEN05_COMMIT_INTR<"", "2">;
defm TCGEN05_COMMIT_S64_CG1 : TCGEN05_COMMIT_INTR<"shared", "1">;
defm TCGEN05_COMMIT_S64_CG2 : TCGEN05_COMMIT_INTR<"shared", "2">;

multiclass TCGEN05_SHIFT_INTR<string num, Intrinsic Intr> {
  def "" : BasicNVPTXInst<(outs),
             (ins ADDR:$tmem_addr),
             "tcgen05.shift.cta_group::" # num # ".down",
             [(Intr addr:$tmem_addr)]>;
}
defm TCGEN05_SHIFT_CG1: TCGEN05_SHIFT_INTR<"1", int_nvvm_tcgen05_shift_down_cg1>;
defm TCGEN05_SHIFT_CG2: TCGEN05_SHIFT_INTR<"2", int_nvvm_tcgen05_shift_down_cg2>;

multiclass TCGEN05_CP_INTR<string shape, string src_fmt, string mc = ""> {
  defvar dst_fmt = !if(!eq(src_fmt, ""), "", ".b8x16");
  defvar fmt_asm = StrJoin<".", [dst_fmt, src_fmt]>.ret;
  defvar fmt_intr = StrJoin<"_", [src_fmt]>.ret;

  defvar shape_mc_asm = StrJoin<".", [shape, mc]>.ret;
  defvar shape_mc_intr = !subst("::", "_", !subst(".", "_", shape_mc_asm));

  defvar intr_prefix = StrJoin<"_", ["int_nvvm_tcgen05_cp", shape_mc_intr, fmt_intr]>.ret;
  defvar IntrCG1 = !cast<Intrinsic>(intr_prefix # "_cg1");
  defvar IntrCG2 = !cast<Intrinsic>(intr_prefix # "_cg2");

  def _cg1 : BasicNVPTXInst<(outs),
                    (ins ADDR:$tmem_addr, B64:$sdesc),
                    "tcgen05.cp.cta_group::1." # shape_mc_asm # fmt_asm,
                    [(IntrCG1 addr:$tmem_addr, B64:$sdesc)]>;
  def _cg2 : BasicNVPTXInst<(outs),
                    (ins ADDR:$tmem_addr, B64:$sdesc),
                    "tcgen05.cp.cta_group::2." # shape_mc_asm # fmt_asm,
                    [(IntrCG2 addr:$tmem_addr, B64:$sdesc)]>;
}

foreach src_fmt = ["", "b6x16_p32", "b4x16_p64"] in {
  defm TCGEN05_CP_128x256b # src_fmt : TCGEN05_CP_INTR<"128x256b", src_fmt>;
  defm TCGEN05_CP_4x256b # src_fmt   : TCGEN05_CP_INTR<"4x256b", src_fmt>;
  defm TCGEN05_CP_128x128b # src_fmt : TCGEN05_CP_INTR<"128x128b", src_fmt>;
  defm TCGEN05_CP_64x128_1 # src_fmt : TCGEN05_CP_INTR<"64x128b", src_fmt, "warpx2::02_13">;
  defm TCGEN05_CP_64x128_2 # src_fmt : TCGEN05_CP_INTR<"64x128b", src_fmt, "warpx2::01_23">;
  defm TCGEN05_CP_32x128 # src_fmt   : TCGEN05_CP_INTR<"32x128b", src_fmt, "warpx4">;
}
} // isConvergent

let hasSideEffects = 1, Predicates = [hasTcgen05Instructions] in {

  def tcgen05_fence_before_thread_sync: NullaryInst<
    "tcgen05.fence::before_thread_sync", int_nvvm_tcgen05_fence_before_thread_sync>;

  def tcgen05_fence_after_thread_sync: NullaryInst<
    "tcgen05.fence::after_thread_sync", int_nvvm_tcgen05_fence_after_thread_sync>;

} // hasSideEffects

// name class for tcgen05.{ld, st}
class TCGEN05_LDST_INST_NAME<string Op, string shape, int lg2Count, bit packOrUnpack> {
  string name = "TCGEN05_" # Op
                # "_" # shape
                # "_x" # !shl(1, lg2Count)
                # !if(!eq(packOrUnpack, 1), !if(!eq(Op, "LD"), "_PACK", "_UNPACK"), "");
}

// reginfo class tcgen05.{ld, st}
class TCGEN05_LDST_REGINFO<int Veclen> {
  // create a list of types for load/store operands
  list<NVPTXRegClass> regs = !listsplat(B32, Veclen);
  // generate list of regnames for load/store operands
  list<string> reg_names = !foreach(x, !range(0, Veclen), "r" # x);
  string regstring = "{{" # !interleave(!foreach(n, !range(0, Veclen), "$r" # n), ", ") # "}}";
  dag Ins = !dag(ins, regs, reg_names);
  dag Outs = !dag(outs, regs, reg_names);
}

//
// tcgen05.ld.sync.aligned.shape.x[1, 2, 4, 8, 16, 32, 64, 128][|.pack::16b].[b32]
//

class TCGEN05_LD_INST<string Shape, int Num, bit Pack> :
        NVPTXInst<(outs), (ins), "?", []>,
        Requires<[hasTcgen05Instructions]> {

  TCGEN05_LDST_REGINFO Info = TCGEN05_LDST_REGINFO<
                                NVVM_TCGEN05_LDST_ACCESS_SIZE<Shape, Num>.veclen>;

  let InOperandList = !con((ins B32:$taddr),
                           !if(!eq(Shape, "16x32bx2"), (ins i64imm:$offset), (ins)));
  let OutOperandList = Info.Outs;
  let AsmString = "tcgen05.ld.sync.aligned"
                  # "." # Shape
                  # ".x" # !shl(1, Num)
                  # !if(!eq(Pack, 1), ".pack::16b", "")
                  # ".b32 "
                  # Info.regstring # ", "
                  # "[$taddr]"
                  # !if(!eq(Shape, "16x32bx2"), ", $offset", "")
                  # ";";
}

//
// tcgen05.st.sync.aligned.shape.x[1, 2, 4, 8, 16, 32, 64, 128][|.unpack::16b].[b32]
//

class TCGEN05_ST_INST<string Shape, int Num, bit Unpack> :
        NVPTXInst<(outs), (ins), "?", []>,
        Requires<[hasTcgen05Instructions]> {

  TCGEN05_LDST_REGINFO Info = TCGEN05_LDST_REGINFO<
                                NVVM_TCGEN05_LDST_ACCESS_SIZE<Shape, Num>.veclen>;

  let InOperandList = !con((ins B32:$taddr),
                           !if(!eq(Shape, "16x32bx2"), (ins i64imm:$offset), (ins)),
                           Info.Ins);
  let OutOperandList = (outs);
  let AsmString = "tcgen05.st.sync.aligned"
                  # "." # Shape
                  # ".x" # !shl(1, Num)
                  # !if(!eq(Unpack, 1), ".unpack::16b", "")
                  # ".b32 [$taddr]"
                  # !if(!eq(Shape, "16x32bx2"), ", $offset", "")
                  # ", " # Info.regstring
                  # ";";
}

let isConvergent = true in {

foreach shape = ["16x64b", "16x128b", "16x256b", "32x32b", "16x32bx2"] in {
  foreach num = !range(0, 8) in {
    foreach packOrUnpack = [false, true] in {
      if NVVM_TCGEN05_LDST_ACCESS_SIZE<shape, num>.valid then {
        def TCGEN05_LDST_INST_NAME<"LD", shape, num, packOrUnpack>.name :
              TCGEN05_LD_INST<shape, num, packOrUnpack>;
        def TCGEN05_LDST_INST_NAME<"ST", shape, num, packOrUnpack>.name :
              TCGEN05_ST_INST<shape, num, packOrUnpack>;
      }
    }
  }
}

} // isConvergent

// Bulk store instructions
def st_bulk_imm : TImmLeaf<i64, [{ return Imm == 0; }]>;

let Predicates = [hasSM<100>, hasPTX<86>] in {
  def INT_NVVM_ST_BULK_GENERIC :
    BasicNVPTXInst<(outs), (ins ADDR:$dest_addr, B64:$size, i64imm:$value),
              "st.bulk",
              [(int_nvvm_st_bulk addr:$dest_addr, i64:$size, st_bulk_imm:$value)]>;

  def INT_NVVM_ST_BULK_SHARED_CTA:
    BasicNVPTXInst<(outs), (ins ADDR:$dest_addr, B64:$size, i64imm:$value),
              "st.bulk.shared::cta",
              [(int_nvvm_st_bulk_shared_cta addr:$dest_addr, i64:$size, st_bulk_imm:$value)]>;
}

//
// clusterlaunchcontorl Instructions
//

def CLUSTERLAUNCHCONTRL_TRY_CANCEL:
      BasicNVPTXInst<(outs), (ins ADDR:$addr, ADDR:$mbar),
                "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
                [(int_nvvm_clusterlaunchcontrol_try_cancel_async_shared addr:$addr, addr:$mbar)]>,
      Requires<[hasSM<100>, hasPTX<86>]>;

def CLUSTERLAUNCHCONTRL_TRY_CANCEL_MULTICAST:
      BasicNVPTXInst<(outs), (ins ADDR:$addr, ADDR:$mbar),
                "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes" #
                ".multicast::cluster::all.b128",
                [(int_nvvm_clusterlaunchcontrol_try_cancel_async_multicast_shared addr:$addr, addr:$mbar)]>,
      Requires<[hasSM<100>, hasArchAccelFeatures, hasPTX<86>]>;

def SDTClusterLaunchControlQueryCancelIsCanceled: SDTypeProfile<1, 2, []>;
def clusterlaunchcontrol_query_cancel_is_canceled:
      SDNode<"NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_IS_CANCELED",
             SDTClusterLaunchControlQueryCancelIsCanceled, []>;

def CLUSTERLAUNCHCONTROL_QUERY_CANCEL_IS_CANCELED:
  NVPTXInst<(outs B1:$pred), (ins B64:$try_cancel_response0, B64:$try_cancel_response1),
            "{{\n\t" #
               ".reg .b128 %clc_handle;\n\t" #
               "mov.b128 %clc_handle, {$try_cancel_response0, $try_cancel_response1};\n\t" #
               "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 $pred, %clc_handle;\n\t" #
            "}}", [(set i1:$pred,
                        (clusterlaunchcontrol_query_cancel_is_canceled i64:$try_cancel_response0, i64:$try_cancel_response1))]>,
            Requires<[hasSM<100>, hasPTX<86>]>;

class CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID<string Dim>:
  NVPTXInst<(outs B32:$reg), (ins B64:$try_cancel_response0, B64:$try_cancel_response1),
            "{{\n\t" #
               ".reg .b128 %clc_handle;\n\t" #
               "mov.b128 %clc_handle, {$try_cancel_response0, $try_cancel_response1};\n\t" #
               "clusterlaunchcontrol.query_cancel.get_first_ctaid::" # Dim # ".b32.b128 $reg, %clc_handle;\n\t" #
            "}}", [(set i32:$reg,
                        (!cast<SDNode>("clusterlaunchcontrol_query_cancel_first_cta_id_" # Dim)
                          i64:$try_cancel_response0, i64:$try_cancel_response1))]>,
            Requires<[hasSM<100>, hasPTX<86>]>;

foreach dim = ["x", "y", "z"] in {
  def SDTClusterLaunchControlQueryCancelGetFirstCtaId # dim: SDTypeProfile<1, 2, []>;

  def clusterlaunchcontrol_query_cancel_first_cta_id_ # dim :
        SDNode<"NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID_" # !toupper(dim),
               !cast<SDTypeProfile>("SDTClusterLaunchControlQueryCancelGetFirstCtaId" # dim), []>;

  def CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID_ # dim:
        CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID<dim>;
}
