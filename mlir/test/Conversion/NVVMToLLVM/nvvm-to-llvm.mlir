// RUN: mlir-opt --convert-nvvm-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL : @init_mbarrier_arrive_expect_tx
llvm.func @init_mbarrier_arrive_expect_tx(%barrier : !llvm.ptr<3>, %txcount : i32) {
  //CHECK : llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.shared.b64 _, [$0], $1;", "r,r" 
  nvvm.mbarrier.arrive.expect_tx.shared %barrier, %txcount : !llvm.ptr<3>, i32
  llvm.return
}

// CHECK-LABEL : @init_mbarrier_arrive_expect_tx_generic
llvm.func @init_mbarrier_arrive_expect_tx_generic(%barrier : !llvm.ptr, %txcount : i32) {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.b64 _, [$0], $1;", "l,r" 
  nvvm.mbarrier.arrive.expect_tx %barrier, %txcount : !llvm.ptr, i32
  llvm.return
}

// CHECK-LABEL : @init_mbarrier_try_wait.parity.shared
llvm.func @init_mbarrier_try_wait_shared(%barrier : !llvm.ptr<3>, %ticks : i32, %phase : i32) {
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "{\0A\09.reg .pred       P1; \0A\09LAB_WAIT: \0A\09mbarrier.try_wait.parity.shared.b64 P1, [$0], $1, $2; \0A\09@P1 bra.uni DONE; \0A\09bra.uni     LAB_WAIT; \0A\09DONE: \0A\09}", "r,r,r"
   nvvm.mbarrier.try_wait.parity.shared %barrier, %phase, %ticks : !llvm.ptr<3>, i32, i32
  llvm.return
}

// CHECK-LABEL : @init_mbarrier_try_wait.parity
llvm.func @init_mbarrier_try_wait(%barrier : !llvm.ptr, %ticks : i32, %phase : i32){
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "{\0A\09.reg .pred       P1; \0A\09LAB_WAIT: \0A\09mbarrier.try_wait.parity.b64 P1, [$0], $1, $2; \0A\09@P1 bra.uni DONE; \0A\09bra.uni     LAB_WAIT; \0A\09DONE: \0A\09}", "r,r,r"
  nvvm.mbarrier.try_wait.parity %barrier, %phase, %ticks : !llvm.ptr, i32, i32
  llvm.return
}

// CHECK-LABEL : @async_cp
func.func @async_cp(%dst: !llvm.ptr<3>, %src: !llvm.ptr<1>) {
  // CHECK : nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, 16, cache =  ca : !llvm.ptr<3>, !llvm.ptr<1>
  nvvm.cp.async.shared.global %dst, %src, 16, cache =  ca : !llvm.ptr<3>, !llvm.ptr<1>
  // CHECK : nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, 16, cache =  cg : !llvm.ptr<3>, !llvm.ptr<1>
  nvvm.cp.async.shared.global %dst, %src, 16, cache =  cg : !llvm.ptr<3>, !llvm.ptr<1>
  return
}

// CHECK-LABEL : @async_cp_zfill
func.func @async_cp_zfill(%dst: !llvm.ptr<3>, %src: !llvm.ptr<1>, %cpSize: i32) {
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "cp.async.cg.shared.global [$0], [$1], $2, $3;\0A", "r,l,r" %{{.*}}, %{{.*}}, %{{.*}} : (!llvm.ptr<3>, !llvm.ptr<1>, i32) -> !llvm.void
  nvvm.cp.async.shared.global %dst, %src, 16, cache =  cg, %cpSize : !llvm.ptr<3>, !llvm.ptr<1>, i32
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "cp.async.ca.shared.global [$0], [$1], $2, $3;\0A", "r,l,r" %{{.*}}, %{{.*}}, %{{.*}} : (!llvm.ptr<3>, !llvm.ptr<1>, i32) -> !llvm.void
  nvvm.cp.async.shared.global %dst, %src, 4, cache =  ca, %cpSize : !llvm.ptr<3>, !llvm.ptr<1>, i32
  return
}

// CHECK-LABEL : @tma_load_1d
func.func @tma_load_1d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32) {
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3}], [$2];", "l,r,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0] : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32
  return
}

// CHECK-LABEL : @tma_load_2d
func.func @tma_load_2d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32) {
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4}], [$2];", "l,r,r,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1] : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32, i32
  return
}

// CHECK-LABEL : @tma_load_3d
func.func @tma_load_3d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32) {
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5}], [$2];", "l,r,r,r,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2] : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32, i32, i32
  return
}

// CHECK-LABEL : @tma_load_4d
func.func @tma_load_4d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32) {
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5, $6}], [$2];", "l,r,r,r,r,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3] : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32, i32, i32, i32
  return
}

// CHECK-LABEL : @tma_load_5d
func.func @tma_load_5d(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32) {
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5, $6, $7}], [$2];", "l,r,r,r,r,r,r,r"
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3,%crd4] : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32, i32, i32, i32, i32
  return
}


// CHECK-LABEL : @wgmma_execute
func.func @wgmma_execute() {  
  nvvm.wgmma.fence.aligned
  nvvm.wgmma.commit.group.sync.aligned
  nvvm.wgmma.wait.group.sync.aligned 0
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "wgmma.fence.sync.aligned;", ""
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "wgmma.commit_group.sync.aligned;", ""
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "wgmma.wait_group.sync.aligned %0;", "n" %{{.*}} : (i32)
  

  nvvm.wgmma.fence.aligned
  nvvm.wgmma.commit.group.sync.aligned
  nvvm.wgmma.wait.group.sync.aligned 1
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "wgmma.fence.sync.aligned;", ""
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "wgmma.commit_group.sync.aligned;", ""
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "wgmma.wait_group.sync.aligned %0;", "n" %{{.*}} : (i32)
  return
}
