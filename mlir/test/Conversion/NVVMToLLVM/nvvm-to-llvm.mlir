// RUN: mlir-opt --convert-nvvm-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL : @init_mbarrier_arrive_expect_tx
llvm.func @init_mbarrier_arrive_expect_tx(%barrier : !llvm.ptr<3>, %txcount : i32) -> i64 {
  //CHECK : llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.shared.b64 $0, [$1], $2;", "=l,r,r" %{{.*}}, %{{.*}} : (!llvm.ptr<3>, i32) -> i64            
  %res = nvvm.mbarrier.arrive.expect_tx.shared %barrier, %txcount : !llvm.ptr<3>, i32 -> i64
  llvm.return %res : i64
}

// CHECK-LABEL : @init_mbarrier_arrive_expect_tx_generic
llvm.func @init_mbarrier_arrive_expect_tx_generic(%barrier : !llvm.ptr, %txcount : i32)-> i64 {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.b64 $0, [$1], $2;", "=l,l,r" %{{.*}}, %{{.*}} : (!llvm.ptr, i32) -> i64
  %res = nvvm.mbarrier.arrive.expect_tx %barrier, %txcount : !llvm.ptr, i32 -> i64
  llvm.return %res : i64
}

// CHECK-LABEL : @init_mbarrier_try_wait.parity.shared
llvm.func @init_mbarrier_try_wait_shared(%barrier : !llvm.ptr<3>, %token : i32) -> i32 {
  // CHECK : llvm.inline_asm has_side_effects asm_dialect = att "{\0A\09.reg .pred P1; \0A\09mbarrier.try_wait.parity.shared.b64 P1, [$1], $2; \0A\09selp.b32 $0, 1, 0, P1; \0A\09}", "=r,r,r" %{{.*}}, %{{.*}} : (!llvm.ptr<3>, i32) -> i32
  %res = nvvm.mbarrier.try_wait.parity.shared %barrier, %token : !llvm.ptr<3>, i32 -> i32
  llvm.return %res : i32
}

// CHECK-LABEL : @init_mbarrier_try_wait.parity
llvm.func @init_mbarrier_try_wait(%barrier : !llvm.ptr, %token : i32) -> i32{
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "{\0A\09.reg .pred P1; \0A\09mbarrier.try_wait.parity.b64 P1, [$1], $2; \0A\09selp.b32 $0, 1, 0, P1; \0A\09}", "=r,l,r" %{{.*}}, %{{.*}} : (!llvm.ptr, i32) -> i32
  %res = nvvm.mbarrier.try_wait.parity %barrier, %token : !llvm.ptr, i32 -> i32
  llvm.return %res : i32
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
