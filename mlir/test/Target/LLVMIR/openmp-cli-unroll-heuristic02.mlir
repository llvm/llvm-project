// Test lowering of the omp.unroll_heuristic
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s


// CHECK-LABEL: define void @unroll_heuristic_nested_loop(
// CHECK-SAME:    ptr %[[ptr:.+]], i32 %[[outer_tc:.+]], i32 %[[inner_tc:.+]]) {
// CHECK-NEXT:   br label %omp_omp.loop.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.preheader:
// CHECK-NEXT:   br label %omp_omp.loop.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.header:
// CHECK-NEXT:   %omp_omp.loop.iv = phi i32 [ 0, %omp_omp.loop.preheader ], [ %omp_omp.loop.next, %omp_omp.loop.inc ]
// CHECK-NEXT:   br label %omp_omp.loop.cond
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.cond:
// CHECK-NEXT:   %omp_omp.loop.cmp = icmp ult i32 %omp_omp.loop.iv, %[[outer_tc]]
// CHECK-NEXT:   br i1 %omp_omp.loop.cmp, label %omp_omp.loop.body, label %omp_omp.loop.exit
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.body:
// CHECK-NEXT:   br label %omp.loop.region
// CHECK-EMPTY:
// CHECK-NEXT: omp.loop.region:
// CHECK-NEXT:   br label %omp_omp.loop.preheader1
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.preheader1:
// CHECK-NEXT:   br label %omp_omp.loop.header2
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.header2:
// CHECK-NEXT:   %omp_omp.loop.iv8 = phi i32 [ 0, %omp_omp.loop.preheader1 ], [ %omp_omp.loop.next10, %omp_omp.loop.inc5 ]
// CHECK-NEXT:   br label %omp_omp.loop.cond3
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.cond3:
// CHECK-NEXT:   %omp_omp.loop.cmp9 = icmp ult i32 %omp_omp.loop.iv8, %[[inner_tc]]
// CHECK-NEXT:   br i1 %omp_omp.loop.cmp9, label %omp_omp.loop.body4, label %omp_omp.loop.exit6
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.body4:
// CHECK-NEXT:   br label %omp.loop.region12
// CHECK-EMPTY:
// CHECK-NEXT: omp.loop.region12:
// CHECK-NEXT:   store float 4.200000e+01, ptr %[[ptr]], align 4
// CHECK-NEXT:   br label %omp.region.cont11
// CHECK-EMPTY:
// CHECK-NEXT: omp.region.cont11:
// CHECK-NEXT:   br label %omp_omp.loop.inc5
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.inc5:
// CHECK-NEXT:   %omp_omp.loop.next10 = add nuw i32 %omp_omp.loop.iv8, 1
// CHECK-NEXT:   br label %omp_omp.loop.header2, !llvm.loop ![[$MD1:[0-9]+]]
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.exit6:
// CHECK-NEXT:   br label %omp_omp.loop.after7
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.after7:
// CHECK-NEXT:   br label %omp.region.cont
// CHECK-EMPTY:
// CHECK-NEXT: omp.region.cont:
// CHECK-NEXT:   br label %omp_omp.loop.inc
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.inc:
// CHECK-NEXT:   %omp_omp.loop.next = add nuw i32 %omp_omp.loop.iv, 1
// CHECK-NEXT:   br label %omp_omp.loop.header, !llvm.loop ![[$MD3:[0-9]+]]
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.exit:
// CHECK-NEXT:   br label %omp_omp.loop.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.after:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @unroll_heuristic_nested_loop(%ptr: !llvm.ptr, %outer_tc: i32, %inner_tc: i32) -> () {
  %outer_cli = omp.new_cli
  %inner_cli = omp.new_cli
  omp.canonical_loop(%outer_cli) %outer_iv : i32 in range(%outer_tc) {
    omp.canonical_loop(%inner_cli) %inner_iv : i32 in range(%inner_tc) {
      %val = llvm.mlir.constant(42.0 : f32) : f32
      llvm.store %val, %ptr : f32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  omp.unroll_heuristic(%outer_cli)
  omp.unroll_heuristic(%inner_cli)
  llvm.return
}


// Start of metadata
// CHECK-LABEL: !llvm.module.flags

// CHECK: ![[$MD1]] = distinct !{![[$MD1]], ![[$MD2:[0-9]+]]}
// CHECK: ![[$MD2]] = !{!"llvm.loop.unroll.enable"}
// CHECK: ![[$MD3]] = distinct !{![[$MD3]], ![[$MD2]]}

