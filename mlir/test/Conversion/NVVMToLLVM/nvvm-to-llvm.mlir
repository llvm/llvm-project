// RUN: mlir-opt --convert-nvvm-to-llvm --convert-arith-to-llvm --split-input-file %s | FileCheck %s

// Same below, but using the `ConvertToLLVMPatternInterface` entry point
// and the generic `convert-to-llvm` pass.
// RUN: mlir-opt --convert-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL: @init_mbarrier
llvm.func @init_mbarrier(%barrier_gen : !llvm.ptr, %barrier : !llvm.ptr<3>, %count : i32, %pred : i1) {
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$2 mbarrier.init.shared.b64 [$0], $1;", "r,r,b" 
  nvvm.mbarrier.init.shared %barrier, %count, predicate = %pred : !llvm.ptr<3>, i32, i1 
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$2 mbarrier.init.b64 [$0], $1;", "l,r,b" 
  nvvm.mbarrier.init %barrier_gen, %count, predicate = %pred : !llvm.ptr, i32, i1
  llvm.return
}

// CHECK-LABEL: @init_mbarrier_arrive_expect_tx
llvm.func @init_mbarrier_arrive_expect_tx(%barrier : !llvm.ptr<3>, %txcount : i32, %pred : i1) {
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.shared.b64 _, [$0], $1;", "r,r"
  nvvm.mbarrier.arrive.expect_tx.shared %barrier, %txcount : !llvm.ptr<3>, i32
  //CHECK:  llvm.inline_asm has_side_effects asm_dialect = att "@$2 mbarrier.arrive.expect_tx.shared.b64 _, [$0], $1;", "r,r,b"
  nvvm.mbarrier.arrive.expect_tx.shared %barrier, %txcount, predicate = %pred : !llvm.ptr<3>, i32, i1 
  llvm.return
}

// CHECK-LABEL: @init_mbarrier_arrive_expect_tx_generic
llvm.func @init_mbarrier_arrive_expect_tx_generic(%barrier : !llvm.ptr, %txcount : i32, %pred : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.b64 _, [$0], $1;", "l,r" 
  nvvm.mbarrier.arrive.expect_tx %barrier, %txcount : !llvm.ptr, i32
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$2 mbarrier.arrive.expect_tx.b64 _, [$0], $1;", "l,r,b"
  nvvm.mbarrier.arrive.expect_tx %barrier, %txcount, predicate = %pred : !llvm.ptr, i32, i1 
  llvm.return
}

// CHECK-LABEL: @init_mbarrier_try_wait_shared
llvm.func @init_mbarrier_try_wait_shared(%barrier : !llvm.ptr<3>, %ticks : i32, %phase : i32) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "{
  // CHECK-SAME: .reg .pred       P1;
  // CHECK-SAME: LAB_WAIT: 
  // CHECK-SAME: mbarrier.try_wait.parity.shared.b64 P1, [$0], $1, $2;
  // CHECK-SAME: @P1 bra.uni DONE;
  // CHECK-SAME: bra.uni     LAB_WAIT;
  // CHECK-SAME: DONE:
  // CHECK-SAME: }",
  // CHECK-SAME: "r,r,r"
   nvvm.mbarrier.try_wait.parity.shared %barrier, %phase, %ticks : !llvm.ptr<3>, i32, i32
  llvm.return
}

// CHECK-LABEL: @init_mbarrier_try_wait
llvm.func @init_mbarrier_try_wait(%barrier : !llvm.ptr, %ticks : i32, %phase : i32){
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
  // CHECK-SAME: "{
  // CHECK-SAME: .reg .pred       P1;
  // CHECK-SAME: LAB_WAIT: 
  // CHECK-SAME: mbarrier.try_wait.parity.b64 P1, [$0], $1, $2;
  // CHECK-SAME: @P1 bra.uni DONE;
  // CHECK-SAME: bra.uni     LAB_WAIT;
  // CHECK-SAME: DONE:
  // CHECK-SAME: }",
  // CHECK-SAME: "l,r,r"
  nvvm.mbarrier.try_wait.parity %barrier, %phase, %ticks : !llvm.ptr, i32, i32
  llvm.return
}

// CHECK-LABEL: @async_cp
func.func @async_cp(%dst: !llvm.ptr<3>, %src: !llvm.ptr<1>) {
  // CHECK: nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, 16, cache =  ca : !llvm.ptr<3>, !llvm.ptr<1>
  nvvm.cp.async.shared.global %dst, %src, 16, cache =  ca : !llvm.ptr<3>, !llvm.ptr<1>
  // CHECK: nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, 16, cache =  cg : !llvm.ptr<3>, !llvm.ptr<1>
  nvvm.cp.async.shared.global %dst, %src, 16, cache =  cg : !llvm.ptr<3>, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @async_cp_zfill
func.func @async_cp_zfill(%dst: !llvm.ptr<3>, %src: !llvm.ptr<1>, %cpSize: i32) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "cp.async.cg.shared.global [$0], [$1], $2, $3;\0A", 
  // CHECK-SAME: "r,l,n,r" %{{.*}}, %{{.*}}, %{{.*}} : (!llvm.ptr<3>, !llvm.ptr<1>, i32, i32) -> ()
  nvvm.cp.async.shared.global %dst, %src, 16, cache =  cg, %cpSize : !llvm.ptr<3>, !llvm.ptr<1>, i32
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "cp.async.ca.shared.global [$0], [$1], $2, $3;\0A", 
  // CHECK-SAME: "r,l,n,r" %{{.*}}, %{{.*}}, %{{.*}} : (!llvm.ptr<3>, !llvm.ptr<1>, i32, i32) -> ()
  nvvm.cp.async.shared.global %dst, %src, 4, cache =  ca, %cpSize : !llvm.ptr<3>, !llvm.ptr<1>, i32
  return
}

// CHECK-LABEL: @cp_async_mbarrier_arrive
func.func @cp_async_mbarrier_arrive(%bar_shared: !llvm.ptr<3>, %bar_gen: !llvm.ptr) {
  // CHECK: nvvm.cp.async.mbarrier.arrive %{{.*}}
  nvvm.cp.async.mbarrier.arrive %bar_gen : !llvm.ptr
  // CHECK: nvvm.cp.async.mbarrier.arrive %{{.*}} {noinc = true}
  nvvm.cp.async.mbarrier.arrive %bar_gen {noinc = true} : !llvm.ptr
  // CHECK: nvvm.cp.async.mbarrier.arrive.shared %{{.*}}
  nvvm.cp.async.mbarrier.arrive.shared %bar_shared : !llvm.ptr<3>
  // CHECK: nvvm.cp.async.mbarrier.arrive.shared %{{.*}} {noinc = true}
  nvvm.cp.async.mbarrier.arrive.shared %bar_shared {noinc = true} : !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: @tma_load_3d_all
func.func @tma_load_3d_all(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %off0: i16, %off1: i16, %ctamask : i16, %cacheHint : i64, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.im2col.multicast::cluster.L2::cache_hint [$0], [$1, {$2,$3,$4} ], [$5],{$6}, $7, $8;", "r,l,r,r,r,r,h,h,l"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2] im2col[%off0] multicast_mask = %ctamask l2_cache_hint = %cacheHint : !llvm.ptr<3>, !llvm.ptr  
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$9 cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.im2col.multicast::cluster.L2::cache_hint [$0], [$1, {$2,$3,$4} ], [$5],{$6}, $7, $8;", "r,l,r,r,r,r,h,h,l,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2] im2col[%off0] multicast_mask = %ctamask l2_cache_hint = %cacheHint predicate = %p : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_4d_all
func.func @tma_load_4d_all(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %off0: i16, %off1: i16, %ctamask : i16, %cacheHint : i64, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.im2col.multicast::cluster.L2::cache_hint [$0], [$1, {$2,$3,$4,$5} ], [$6],{$7,$8}, $9, $10;", "r,l,r,r,r,r,r,h,h,h,l"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3] im2col[%off0,%off1] multicast_mask = %ctamask l2_cache_hint = %cacheHint : !llvm.ptr<3>, !llvm.ptr  
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$11 cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.im2col.multicast::cluster.L2::cache_hint [$0], [$1, {$2,$3,$4,$5} ], [$6],{$7,$8}, $9, $10;", "r,l,r,r,r,r,r,h,h,h,l,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3] im2col[%off0,%off1] multicast_mask = %ctamask l2_cache_hint = %cacheHint predicate = %p : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_5d_all
func.func @tma_load_5d_all(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %off0: i16, %off1: i16, %off2: i16, %ctamask : i16, %cacheHint : i64, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.im2col.multicast::cluster.L2::cache_hint [$0], [$1, {$2,$3,$4,$5,$6} ], [$7],{$8,$9,$10}, $11, $12;", "r,l,r,r,r,r,r,r,h,h,h,h,l"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3,%crd4] im2col[%off0,%off1,%off2] multicast_mask = %ctamask l2_cache_hint = %cacheHint : !llvm.ptr<3>, !llvm.ptr  
  // CHECK: lvm.inline_asm has_side_effects asm_dialect = att "@$13 cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.im2col.multicast::cluster.L2::cache_hint [$0], [$1, {$2,$3,$4,$5,$6} ], [$7],{$8,$9,$10}, $11, $12;", "r,l,r,r,r,r,r,r,h,h,h,h,l,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3,%crd4] im2col[%off0,%off1,%off2] multicast_mask = %ctamask l2_cache_hint = %cacheHint predicate = %p : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_1d
func.func @tma_load_1d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2} ], [$3];", "r,l,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor, %barrier, box[%crd0] : !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$4 cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2} ], [$3];", "r,l,r,r,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0] predicate=%p : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_2d
func.func @tma_load_2d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3} ], [$4];", "r,l,r,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor, %barrier, box[%crd0,%crd1] : !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$5 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3} ], [$4];", "r,l,r,r,r,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor, %barrier, box[%crd0,%crd1] predicate=%p  : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_3d
func.func @tma_load_3d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3,$4} ], [$5];", "r,l,r,r,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2] : !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$6 cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3,$4} ], [$5];", "r,l,r,r,r,r,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2] predicate=%p  : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_4d
func.func @tma_load_4d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3,$4,$5} ], [$6];", "r,l,r,r,r,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3] : !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$7 cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3,$4,$5} ], [$6];", "r,l,r,r,r,r,r,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3] predicate=%p  : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_5d
func.func @tma_load_5d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3,$4,$5,$6} ], [$7];", "r,l,r,r,r,r,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3,%crd4] : !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$8 cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3,$4,$5,$6} ], [$7];", "r,l,r,r,r,r,r,r,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3,%crd4] predicate=%p  : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_multicast1d
func.func @tma_load_multicast1d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %multicastMask : i16, %crd0: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2} ], [$3], $4;", "r,l,r,r,h"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0] multicast_mask = %multicastMask : !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$5 cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2} ], [$3], $4;", "r,l,r,r,h,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0] multicast_mask = %multicastMask predicate=%p : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_multicast2d
func.func @tma_load_multicast2d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %multicastMask : i16, %crd0: i32, %crd1: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2,$3} ], [$4], $5;", "r,l,r,r,r,h"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0,%crd1] multicast_mask = %multicastMask : !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$6 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2,$3} ], [$4], $5;", "r,l,r,r,r,h,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0,%crd1] multicast_mask = %multicastMask  predicate=%p  : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_multicast3d
func.func @tma_load_multicast3d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %multicastMask : i16, %crd0: i32, %crd1: i32, %crd2: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2,$3,$4} ], [$5], $6;", "r,l,r,r,r,r,h"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0,%crd1,%crd2] multicast_mask = %multicastMask : !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$7 cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2,$3,$4} ], [$5], $6;", "r,l,r,r,r,r,h,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0,%crd1,%crd2] multicast_mask = %multicastMask  predicate=%p  : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_multicast4d
func.func @tma_load_multicast4d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %multicastMask : i16, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2,$3,$4,$5} ], [$6], $7;", "r,l,r,r,r,r,r,h"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0,%crd1,%crd2,%crd3] multicast_mask = %multicastMask: !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$8 cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2,$3,$4,$5} ], [$6], $7;", "r,l,r,r,r,r,r,h,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0,%crd1,%crd2,%crd3] multicast_mask = %multicastMask predicate=%p  : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_load_multicast5d
func.func @tma_load_multicast5d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %multicastMask : i16, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2,$3,$4,$5,$6} ], [$7], $8;", "r,l,r,r,r,r,r,r,h"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0,%crd1,%crd2,%crd3,%crd4] multicast_mask = %multicastMask : !llvm.ptr<3>, !llvm.ptr
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$9 cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2,$3,$4,$5,$6} ], [$7], $8;", "r,l,r,r,r,r,r,r,h,b"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box [%crd0,%crd1,%crd2,%crd3,%crd4] multicast_mask = %multicastMask predicate=%p  : !llvm.ptr<3>, !llvm.ptr
  return
}

// CHECK-LABEL: @tma_store_1d
func.func @tma_store_1d(%tmaDescriptor: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [$0, {$2} ], [$1];", "l,r,r"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0] : !llvm.ptr, !llvm.ptr<3>, i32
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$3 cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [$0, {$2} ], [$1];", "l,r,r,b"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0], predicate=%p : !llvm.ptr, !llvm.ptr<3>, i32, i1
  return
}

// CHECK-LABEL: @tma_store_2d
func.func @tma_store_2d(%tmaDescriptor: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$0, {$2, $3} ], [$1];", "l,r,r,r"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0,%crd1] : !llvm.ptr, !llvm.ptr<3>, i32, i32
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$4 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$0, {$2, $3} ], [$1];", "l,r,r,r,b"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0,%crd1], predicate=%p : !llvm.ptr, !llvm.ptr<3>, i32, i32, i1
  return
}

// CHECK-LABEL: @tma_store_3d
func.func @tma_store_3d(%tmaDescriptor: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [$0, {$2, $3, $4} ], [$1];", "l,r,r,r,r"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0,%crd1,%crd2] : !llvm.ptr, !llvm.ptr<3>, i32, i32, i32
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$5 cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [$0, {$2, $3, $4} ], [$1];", "l,r,r,r,r,b"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0,%crd1,%crd2], predicate=%p : !llvm.ptr, !llvm.ptr<3>, i32, i32, i32, i1
  return
}

// CHECK-LABEL: @tma_store_4d
func.func @tma_store_4d(%tmaDescriptor: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %p : i1) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [$0, {$2, $3, $4, $5} ], [$1];", "l,r,r,r,r,r"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0,%crd1,%crd2,%crd3] : !llvm.ptr, !llvm.ptr<3>, i32, i32, i32, i32
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$6 cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [$0, {$2, $3, $4, $5} ], [$1];", "l,r,r,r,r,r,b"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0,%crd1,%crd2,%crd3], predicate=%p : !llvm.ptr, !llvm.ptr<3>, i32, i32, i32, i32, i1
  return
}

// CHECK-LABEL: @tma_store_5d
func.func @tma_store_5d(%tmaDescriptor: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %p : i1) {
  // CHECK-NEXT: llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [$0, {$2, $3, $4, $5, $6} ], [$1];", "l,r,r,r,r,r,r"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4] : !llvm.ptr, !llvm.ptr<3>, i32, i32, i32, i32, i32

  // CHECK-NEXT: llvm.inline_asm has_side_effects asm_dialect = att "@$7 cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [$0, {$2, $3, $4, $5, $6} ], [$1];", "l,r,r,r,r,r,r,b"
  nvvm.cp.async.bulk.tensor.global.shared.cta %tmaDescriptor, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4], predicate=%p : !llvm.ptr, !llvm.ptr<3>, i32, i32, i32, i32, i32, i1
  return
}

// CHECK-LABEL: @wgmma_execute
func.func @wgmma_execute() {  
  nvvm.wgmma.fence.aligned
  nvvm.wgmma.commit.group.sync.aligned
  nvvm.wgmma.wait.group.sync.aligned 0
  // CHECK: nvvm.wgmma.fence.aligned
  // CHECK: nvvm.wgmma.commit.group.sync.aligned
  // CHECK: nvvm.wgmma.wait.group.sync.aligned 0
  

  nvvm.wgmma.fence.aligned
  nvvm.wgmma.commit.group.sync.aligned
  nvvm.wgmma.wait.group.sync.aligned 5
  // CHECK: nvvm.wgmma.fence.aligned
  // CHECK: nvvm.wgmma.commit.group.sync.aligned
  // CHECK: nvvm.wgmma.wait.group.sync.aligned 5
  return
}


// -----

!mat64f32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32)>

// CHECK-LABEL: @wgmma_f32_f16_f16(
// CHECK-SAME: %[[ARG0:.+]]: i64, %[[ARG1:.+]]: i64
func.func @wgmma_f32_f16_f16(%descA : i64, %descB : i64) -> !mat64f32{  
  // CHECK: %[[RES:.*]] = llvm.mlir.undef : !llvm.struct
  // CHECK: %[[A1:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[A2:.*]] = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: %[[A3:.*]] = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: %[[A4:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[A5:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[V0:.*]] = llvm.extractvalue %[[RES]][0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
  // CHECK: %[[V4:.*]] = llvm.extractvalue %[[RES]][4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
  // CHECK: %[[V11:.*]] = llvm.extractvalue %[[RES]][11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>  
  // CHECK: %[[V13:.*]] = llvm.extractvalue %[[RES]][13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
  // CHECK: %[[RES1:.+]] = llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "{
  // CHECK-SAME: reg .pred p;
  // CHECK-SAME: setp.ne.b32 p, $34, 0;
  // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 
  // CHECK-SAME: {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, $32, $33, p, $35,  $36, $37,  $38;\0A}\0A", 
  // CHECK-SAME: "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,l,l,n,n,n,n,n" 
  // CHECK-SAME: %[[V0]], %{{.*}}, %{{.*}}, %{{.*}}, %[[V4]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[V11]], %{{.*}}, %[[V13]], %{{.*}}, %{{.*}}, %[[ARG0]], %[[ARG1]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]] 
  // CHECK-SAME: : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, i64, i64, i32, i32, i32, i32, i32) 
  // CHECK-SAME: -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  // CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[DESCa:.+]] = llvm.add %[[ARG0]], %[[C2]] : i64
  // CHECK: %[[DESCb:.+]] = llvm.add %[[ARG1]], %[[C2]] : i64
  // CHECK: %[[V0_2:.*]] = llvm.extractvalue %[[RES1]][0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
  // CHECK: %[[V4_2:.*]] = llvm.extractvalue %[[RES1]][4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
  // CHECK: %[[V11_2:.*]] = llvm.extractvalue %[[RES1]][11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>  
  // CHECK: %[[V13_2:.*]] = llvm.extractvalue %[[RES1]][13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
  // CHECK: %[[RES_2:.+]] = llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "{
    // CHECK-SAME: .reg .pred p;
    // CHECK-SAME: setp.ne.b32 p, $34, 0;
    // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 
    // CHECK-SAME: {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, $32, $33, p, $35,  $36, $37,  $38;\0A}\0A", 
    // CHECK-SAME: "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,l,l,n,n,n,n,n" 
    // CHECK-SAME: %[[V0_2]], %{{.*}}, %{{.*}}, %{{.*}}, %[[V4_2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[V11_2]], %{{.*}}, %[[V13_2]], %{{.*}}, %{{.*}}, %[[DESCa]], %[[DESCb]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} 
  %result = llvm.mlir.undef : !mat64f32
  %result1 = nvvm.wgmma.mma_async 
      %descA, %descB, %result,
      #nvvm.shape<m = 64, n = 32, k = 16>, 
      D [<f32>, #nvvm.wgmma_scale_out<zero>],
      A [<f16>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<f16>, #nvvm.wgmma_scale_in<neg>, <col>]
      :!mat64f32 -> !mat64f32
  %c2 = arith.constant 2 : i64
  %descAnext = arith.addi %descA, %c2 : i64
  %descBnext = arith.addi %descB, %c2 : i64
  %result2 = nvvm.wgmma.mma_async 
      %descAnext, %descBnext, %result1,
      #nvvm.shape<m = 64, n = 32, k = 16>, 
      D [<f32>, #nvvm.wgmma_scale_out<zero>],
      A [<f16>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<f16>, #nvvm.wgmma_scale_in<neg>, <col>]
      : !mat64f32 -> !mat64f32
  return %result2 : !mat64f32
}

// -----

!mat16i32 = !llvm.struct<(i32, i32, i32, i32)>

// CHECK-LABEL: @wgmma_s32_s8_s8_satfinite(
// CHECK-SAME: %[[ARG0:.+]]: i64, %[[ARG1:.+]]: i64
func.func @wgmma_s32_s8_s8_satfinite(%descA : i64, %descB : i64) -> !mat16i32{  
  %result = llvm.mlir.undef : !mat16i32
// CHECK: %[[RES:.*]] = llvm.mlir.undef : !llvm.struct
// CHECK: %[[A1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[V0:.*]] = llvm.extractvalue %[[RES]][0]
// CHECK: %[[V1:.*]] = llvm.extractvalue %[[RES]][1]
// CHECK: %[[V2:.*]] = llvm.extractvalue %[[RES]][2]
// CHECK: %[[V3:.*]] = llvm.extractvalue %[[RES]][3]
// CHECK: %[[RES_2:.*]] =  llvm.inline_asm has_side_effects asm_dialect = att 
// CHECK-SAME: "{
// CHECK-SAME: .reg .pred p;
// CHECK-SAME: setp.ne.b32 p, $10, 0;
// CHECK-SAME: wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite 
// CHECK-SAME: {$0, $1, $2, $3}, $8, $9, p;\0A}\0A", "=r,=r,=r,=r,0,1,2,3,l,l,n" 
// CHECK-SAME: %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[ARG0]], %[[ARG1]], %[[A1]] : 
// CHECK-SAME: (i32, i32, i32, i32, i64, i64, i32) -> !llvm.struct<(i32, i32, i32, i32)>
// CHECK: %[[V0_2:.*]] = llvm.extractvalue %[[RES_2]][0]
// CHECK: %[[V1_2:.*]] = llvm.extractvalue %[[RES_2]][1]
// CHECK: %[[V2_2:.*]] = llvm.extractvalue %[[RES_2]][2]
// CHECK: %[[V3_2:.*]] = llvm.extractvalue %[[RES_2]][3]
// CHECK: %[[RES_3:.*]] = llvm.inline_asm has_side_effects asm_dialect = att 
// CHECK-SAME: "{
// CHECK-SAME: .reg .pred p;
// CHECK-SAME: setp.ne.b32 p, $10, 0;
// CHECK-SAME: wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite 
// CHECK-SAME: {$0, $1, $2, $3}, $8, $9, p;\0A}\0A", 
// CHECK-SAME: "=r,=r,=r,=r,0,1,2,3,l,l,n" 
// CHECK-SAME: %[[V0_2]], %[[V1_2]], %[[V2_2]], %[[V3_2]], %[[ARG0]], %[[ARG1]], %{{.*}}
// CHECK: %[[V0_3:.*]] = llvm.extractvalue %[[RES_3]][0]
// CHECK: %[[V1_3:.*]] = llvm.extractvalue %[[RES_3]][1]
// CHECK: %[[V2_3:.*]] = llvm.extractvalue %[[RES_3]][2]
// CHECK: %[[V3_3:.*]] = llvm.extractvalue %[[RES_3]][3]
// CHECK: %[[RES1:.*]] = llvm.inline_asm has_side_effects asm_dialect = att 
// CHECK-SAME:"{
// CHECK-SAME:.reg .pred p;
// CHECK-SAME: setp.ne.b32 p, $10, 0;
// CHECK-SAME: wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite
// CHECK-SAME: {$0, $1, $2, $3}, $8, $9, p;\0A}\0A", "=r,=r,=r,=r,0,1,2,3,l,l,n" 
// CHECK-SAME: %[[V0_3]], %[[V1_3]], %[[V2_3]], %[[V3_3]], %[[ARG0]], %[[ARG1]], %{{.*}} 
  %result1 = nvvm.wgmma.mma_async %descA, %descB, %result, 
      #nvvm.shape<m = 64, n = 8, k = 32>, 
      D [<s32>, #nvvm.wgmma_scale_out<one>, <satfinite>],
      A [<s8>, #nvvm.wgmma_scale_in<one>, <row>], 
      B [<s8>, #nvvm.wgmma_scale_in<one>, <col>]
      : !mat16i32 -> !mat16i32
  %result2 = nvvm.wgmma.mma_async %descA, %descB, %result1, 
      #nvvm.shape<m = 64, n = 8, k = 32>, 
      D [<s32>, #nvvm.wgmma_scale_out<one>, <satfinite>],
      A [<s8>, #nvvm.wgmma_scale_in<one>, <row>], 
      B [<s8>, #nvvm.wgmma_scale_in<one>, <col>]
      : !mat16i32 -> !mat16i32
  %result3 = nvvm.wgmma.mma_async %descA, %descB, %result2, 
      #nvvm.shape<m = 64, n = 8, k = 32>, 
      D [<s32>, #nvvm.wgmma_scale_out<one>, <satfinite>],
      A [<s8>, #nvvm.wgmma_scale_in<one>, <row>], 
      B [<s8>, #nvvm.wgmma_scale_in<one>, <col>]
      : !mat16i32 -> !mat16i32
  return %result3 : !mat16i32
}

// CHECK-LABEL: @wgmma_s32_u8_u8(
  // CHECK-SAME: %[[ARG0:.+]]: i64, %[[ARG1:.+]]: i64
func.func @wgmma_s32_u8_u8(%descA : i64, %descB : i64) -> !mat16i32 {  
// CHECK: %[[RES:.*]] = llvm.mlir.undef : !llvm.struct
// CHECK: %[[A1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[V0:.*]] = llvm.extractvalue %[[RES]][0]
// CHECK: %[[V1:.*]] = llvm.extractvalue %[[RES]][1]
// CHECK: %[[V2:.*]] = llvm.extractvalue %[[RES]][2]
// CHECK: %[[V3:.*]] = llvm.extractvalue %[[RES]][3]
// CHECK: %[[RES_2:.*]] =  llvm.inline_asm has_side_effects asm_dialect = att 
// CHECK-SAME: "{
// CHECK-SAME: .reg .pred p;
// CHECK-SAME: setp.ne.b32 p, $10, 0;
// CHECK-SAME: wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 {$0, $1, $2, $3}, $8, $9, p;
// CHECK-SAME: }\0A",
// CHECK-SAME: "=r,=r,=r,=r,0,1,2,3,l,l,n" %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[ARG0]], %[[ARG1]], %[[A1]] : 
// CHECK-SAME:(i32, i32, i32, i32, i64, i64, i32) -> !llvm.struct<(i32, i32, i32, i32)>
// CHECK: %[[V0_2:.*]] = llvm.extractvalue %[[RES_2]][0]
// CHECK: %[[V1_2:.*]] = llvm.extractvalue %[[RES_2]][1]
// CHECK: %[[V2_2:.*]] = llvm.extractvalue %[[RES_2]][2]
// CHECK: %[[V3_2:.*]] = llvm.extractvalue %[[RES_2]][3]
// CHECK: %[[RES_3:.*]] = llvm.inline_asm has_side_effects asm_dialect = att 
// CHECK-SAME:"{
// CHECK-SAME: .reg .pred p;
// CHECK-SAME: setp.ne.b32 p, $10, 0;
// CHECK-SAME: wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 {$0, $1, $2, $3}, $8, $9, p;
// CHECK-SAME: }\0A",
// CHECK-SAME: "=r,=r,=r,=r,0,1,2,3,l,l,n" %[[V0_2]], %[[V1_2]], %[[V2_2]], %[[V3_2]], %[[ARG0]], %[[ARG1]], %{{.*}}
// CHECK: %[[V0_3:.*]] = llvm.extractvalue %[[RES_3]][0]
// CHECK: %[[V1_3:.*]] = llvm.extractvalue %[[RES_3]][1]
// CHECK: %[[V2_3:.*]] = llvm.extractvalue %[[RES_3]][2]
// CHECK: %[[V3_3:.*]] = llvm.extractvalue %[[RES_3]][3]
// CHECK: %[[RES1:.*]] = llvm.inline_asm has_side_effects asm_dialect = att 
// CHECK-SAME:"{
// CHECK-SAME: .reg .pred p;
// CHECK-SAME: setp.ne.b32 p, $10, 0;
// CHECK-SAME: wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 {$0, $1, $2, $3}, $8, $9, p;
// CHECK-SAME:}\0A", 
// CHECK-SAME:"=r,=r,=r,=r,0,1,2,3,l,l,n" %[[V0_3]], %[[V1_3]], %[[V2_3]], %[[V3_3]], %[[ARG0]], %[[ARG1]], %{{.*}} 
  %result = llvm.mlir.undef : !mat16i32
  %result1 = nvvm.wgmma.mma_async %descA, %descB, %result,
      #nvvm.shape<m = 64, n = 8, k = 32>, 
      D [<s32>, #nvvm.wgmma_scale_out<one>],
      A [<u8>, #nvvm.wgmma_scale_in<one>, <row>], 
      B [<u8>, #nvvm.wgmma_scale_in<one>, <col>]
      : !mat16i32 -> !mat16i32
  %result2 = nvvm.wgmma.mma_async %descA, %descB, %result1,
      #nvvm.shape<m = 64, n = 8, k = 32>, 
      D [<s32>, #nvvm.wgmma_scale_out<one>],
      A [<u8>, #nvvm.wgmma_scale_in<one>, <row>], 
      B [<u8>, #nvvm.wgmma_scale_in<one>, <col>]
      : !mat16i32 -> !mat16i32
  %result3 = nvvm.wgmma.mma_async %descA, %descB, %result2,
      #nvvm.shape<m = 64, n = 8, k = 32>, 
      D [<s32>, #nvvm.wgmma_scale_out<one>],
      A [<u8>, #nvvm.wgmma_scale_in<one>, <row>], 
      B [<u8>, #nvvm.wgmma_scale_in<one>, <col>]
      : !mat16i32 -> !mat16i32
  return %result3 : !mat16i32
}

// -----

!mat32f32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32)>

// CHECK-LABEL: @wgmma_f32_tf32_tf32
func.func @wgmma_f32_tf32_tf32(%descA : i64, %descB : i64) -> !mat32f32 {  
  // CHECK: %[[RES:.+]] = llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME:"{
  // CHECK-SAME: .reg .pred p;
  // CHECK-SAME: setp.ne.b32 p, $66, 0;
  // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $64, $65, p, $67,  $68;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,n,n,n"
  // CHECK: %[[RES_2:.+]] = llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "{
  // CHECK-SAME: .reg .pred p;
  // CHECK-SAME: setp.ne.b32 p, $66, 0;
  // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $64, $65, p, $67,  $68;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,n,n,n"
  %result = llvm.mlir.undef : !mat32f32
  %result1 = nvvm.wgmma.mma_async %descA, %descB, %result,
      #nvvm.shape<m = 64, n = 64, k = 8>, 
      D [#nvvm.wgmma_type<f32>, #nvvm.wgmma_scale_out<one>],
      A [#nvvm.wgmma_type<tf32>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<row>], 
      B [#nvvm.wgmma_type<tf32>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<col>]
       : !mat32f32 -> !mat32f32
  %result2 = nvvm.wgmma.mma_async %descA, %descB, %result1,
      #nvvm.shape<m = 64, n = 64, k = 8>, 
      D [#nvvm.wgmma_type<f32>, #nvvm.wgmma_scale_out<one>],
      A [#nvvm.wgmma_type<tf32>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<row>], 
      B [#nvvm.wgmma_type<tf32>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<col>]
      : !mat32f32 -> !mat32f32
  return %result2 : !mat32f32
}


// -----

!mat32f32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32)>

// CHECK-LABEL: @wgmma_f32_e4m3_e4m3
func.func @wgmma_f32_e4m3_e4m3(%descA : i64, %descB : i64) -> !mat32f32 {  
  // CHECK: %[[RES:.+]] = llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "{\0A.reg .pred p;\0Asetp.ne.b32 p, $66, 0;
  // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $64, $65, p, $67,  $68;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,n,n,n"
  // CHECK: %[[RES_2:.+]] = llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "{\0A.reg .pred p;\0Asetp.ne.b32 p, $66, 0;
  // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $64, $65, p, $67,  $68;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,n,n,n"
  %result = llvm.mlir.undef : !mat32f32
  %result1 = nvvm.wgmma.mma_async %descA, %descB, %result,
      #nvvm.shape<m = 64, n = 64, k = 32>, 
      D [#nvvm.wgmma_type<f32>, #nvvm.wgmma_scale_out<one>],
      A [#nvvm.wgmma_type<e4m3>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<row>], 
      B [#nvvm.wgmma_type<e4m3>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<col>]
       : !mat32f32 -> !mat32f32
  %result2 = nvvm.wgmma.mma_async %descA, %descB, %result1,
      #nvvm.shape<m = 64, n = 64, k = 32>, 
      D [#nvvm.wgmma_type<f32>, #nvvm.wgmma_scale_out<one>],
      A [#nvvm.wgmma_type<e4m3>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<row>], 
      B [#nvvm.wgmma_type<e4m3>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<col>]
      : !mat32f32 -> !mat32f32
  return %result2 : !mat32f32
}

// -----

!mat32f32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32)>

// CHECK-LABEL: @wgmma_f32_e5m2_e4m3
func.func @wgmma_f32_e5m2_e4m3(%descA : i64, %descB : i64) -> !mat32f32 {  
  // CHECK: %[[RES:.+]] = llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "{\0A.reg .pred p;\0Asetp.ne.b32 p, $66, 0;
  // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e4m3 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $64, $65, p, $67,  $68;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,n,n,n"
  // CHECK: %[[RES_2:.+]] = llvm.inline_asm has_side_effects asm_dialect = att 
  // CHECK-SAME: "{\0A.reg .pred p;\0Asetp.ne.b32 p, $66, 0;
  // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e4m3 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $64, $65, p, $67,  $68;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,n,n,n"
  %result = llvm.mlir.undef : !mat32f32
  %result1 = nvvm.wgmma.mma_async %descA, %descB, %result,
      #nvvm.shape<m = 64, n = 64, k = 32>, 
      D [#nvvm.wgmma_type<f32>, #nvvm.wgmma_scale_out<one>],
      A [#nvvm.wgmma_type<e5m2>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<row>], 
      B [#nvvm.wgmma_type<e4m3>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<col>]
       : !mat32f32 -> !mat32f32
  %result2 = nvvm.wgmma.mma_async %descA, %descB, %result1,
      #nvvm.shape<m = 64, n = 64, k = 32>, 
      D [#nvvm.wgmma_type<f32>, #nvvm.wgmma_scale_out<one>],
      A [#nvvm.wgmma_type<e5m2>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<row>], 
      B [#nvvm.wgmma_type<e4m3>, #nvvm.wgmma_scale_in<one>, #nvvm.mma_layout<col>]
      : !mat32f32 -> !mat32f32
  return %result2 : !mat32f32
}

// -----

func.func @elect_one_leader_sync() {  
  // CHECK: %[[RES:.*]] = nvvm.elect.sync -> i1
  %cnd = nvvm.elect.sync -> i1 
  return 
}

// -----

// CHECK-LABEL: @stmatrix(
// CHECK-SAME: %[[arg0:[a-zA-Z0-9_]+]]: !llvm.ptr<3>, 
// CHECK-SAME: %[[arg1:[a-zA-Z0-9_]+]]: i32,
// CHECK-SAME: %[[arg2:[a-zA-Z0-9_]+]]: i32,
// CHECK-SAME: %[[arg3:[a-zA-Z0-9_]+]]: i32,
// CHECK-SAME: %[[arg4:[a-zA-Z0-9_]+]]: i32)
llvm.func @stmatrix(%arg0 : !llvm.ptr<3>, %m1 : i32, %m2 : i32, %m3 : i32, %m4 : i32) {
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "stmatrix.sync.aligned.x1.m8n8.shared.b16 [$0], {$1};", "r,r" %[[arg0]], %[[arg1]] : (!llvm.ptr<3>, i32) -> ()
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "stmatrix.sync.aligned.x2.m8n8.shared.b16 [$0], {$1, $2};", "r,r,r" %[[arg0]], %[[arg1]], %[[arg2]] : (!llvm.ptr<3>, i32, i32) -> ()
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "stmatrix.sync.aligned.x4.m8n8.shared.b16 [$0], {$1, $2, $3, $4};", "r,r,r,r,r" %[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]], %[[arg4]] : (!llvm.ptr<3>, i32, i32, i32, i32) -> ()
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [$0], {$1};", "r,r" %[[arg0]], %[[arg1]] : (!llvm.ptr<3>, i32) -> ()
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [$0], {$1, $2};", "r,r,r" %[[arg0]], %[[arg1]], %[[arg2]] : (!llvm.ptr<3>, i32, i32) -> ()
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [$0], {$1, $2, $3, $4};", "r,r,r,r,r" %[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]], %[[arg4]] : (!llvm.ptr<3>, i32, i32, i32, i32) -> ()
  nvvm.stmatrix %arg0, %m1 {layout = #nvvm.mma_layout<row>} : !llvm.ptr<3>, i32
  nvvm.stmatrix %arg0, %m1, %m2 {layout = #nvvm.mma_layout<row>} : !llvm.ptr<3>, i32, i32
  nvvm.stmatrix %arg0, %m1, %m2, %m3, %m4 {layout = #nvvm.mma_layout<row>} : !llvm.ptr<3>, i32, i32, i32, i32
  nvvm.stmatrix %arg0, %m1 {layout = #nvvm.mma_layout<col>} : !llvm.ptr<3>, i32
  nvvm.stmatrix %arg0, %m1, %m2 {layout = #nvvm.mma_layout<col>} : !llvm.ptr<3>, i32, i32
  nvvm.stmatrix %arg0, %m1, %m2, %m3, %m4 {layout = #nvvm.mma_layout<col>} : !llvm.ptr<3>, i32, i32, i32, i32
  llvm.return 
}

// -----

// CHECK-LABEL: @init_mbarrier_arrive_expect_tx
llvm.func @init_mbarrier_arrive_expect_tx(%desc : !llvm.ptr, %pred : i1) {
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "prefetch.tensormap [$0];", "l"
  nvvm.prefetch.tensormap %desc : !llvm.ptr
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "@$1 prefetch.tensormap [$0];", "l,b"
  nvvm.prefetch.tensormap %desc, predicate = %pred : !llvm.ptr, i1
  llvm.return
}

// -----

func.func @set_max_register() {
  // CHECK: nvvm.setmaxregister increase 232
  nvvm.setmaxregister increase 232

  // CHECK: nvvm.setmaxregister decrease 40
  nvvm.setmaxregister decrease 40
  func.return
}

// -----

func.func @cp_async_bulk_commit() {
  // CHECK: nvvm.cp.async.bulk.commit.group
  nvvm.cp.async.bulk.commit.group
  func.return
}

// -----

func.func @cp_async_bulk_wait_group() {
  // CHECK: nvvm.cp.async.bulk.wait_group 1
  // CHECK: nvvm.cp.async.bulk.wait_group 0
  // CHECK: nvvm.cp.async.bulk.wait_group 5 {read}
  // CHECK: nvvm.cp.async.bulk.wait_group 0 {read}
  nvvm.cp.async.bulk.wait_group 1
  nvvm.cp.async.bulk.wait_group 0
  nvvm.cp.async.bulk.wait_group 5 {read}
  nvvm.cp.async.bulk.wait_group 0 {read}
  func.return
}

// -----

func.func @fence_mbarrier_init() {
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "fence.mbarrier_init.release.cluster;"
  nvvm.fence.mbarrier.init
  func.return 
}
// -----

func.func @fence_proxy() {
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "fence.proxy.alias;", ""  : () -> ()
  nvvm.fence.proxy { kind = #nvvm.proxy_kind<alias>}
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "fence.proxy.async;", ""  : () -> ()
  nvvm.fence.proxy { kind = #nvvm.proxy_kind<async>}
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "fence.proxy.async.global;", ""  : () -> ()
  nvvm.fence.proxy { kind = #nvvm.proxy_kind<async.global>}
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "fence.proxy.async.shared::cta;", ""  : () -> ()
  nvvm.fence.proxy { kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
  //CHECK: llvm.inline_asm has_side_effects asm_dialect = att "fence.proxy.async.shared::cluster;", ""  : () -> ()
  nvvm.fence.proxy { kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
  func.return
}

// -----

// CHECK-LABEL: @llvm_nvvm_barrier_arrive
// CHECK-SAME: (%[[barId:.*]]: i32, %[[numberOfThreads:.*]]: i32)
llvm.func @llvm_nvvm_barrier_arrive(%barID : i32, %numberOfThreads : i32) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "bar.arrive 0, $0;", "r" %[[numberOfThreads]] : (i32) -> ()
  nvvm.barrier.arrive number_of_threads = %numberOfThreads
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "bar.arrive $0, $1;", "r,r" %[[barId]], %[[numberOfThreads]] : (i32, i32) -> ()
  nvvm.barrier.arrive id = %barID number_of_threads = %numberOfThreads
  llvm.return
}
