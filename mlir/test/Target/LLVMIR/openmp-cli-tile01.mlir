// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s



llvm.func @tile_trivial_loop(%baseptr: !llvm.ptr, %tc: i32, %ts: i32) -> () {
  %literal_cli = omp.new_cli
  omp.canonical_loop(%literal_cli) %iv : i32 in range(%tc) {
    %ptr = llvm.getelementptr inbounds %baseptr[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %val = llvm.mlir.constant(42.0 : f32) : f32
    llvm.store %val, %ptr : f32, !llvm.ptr
    omp.terminator
  }
  omp.tile <- (%literal_cli) sizes(%ts : i32)
  llvm.return
}


// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECK-NEXT: source_filename = "LLVMDialectModule"
// CHECK-EMPTY:
// CHECK-NEXT: define void @tile_trivial_loop(ptr %0, i32 %1, i32 %2) {
// CHECK-NEXT:   br label %omp_omp.loop.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.preheader:                           ; preds = %3
// CHECK-NEXT:   %4 = udiv i32 %1, %2
// CHECK-NEXT:   %5 = urem i32 %1, %2
// CHECK-NEXT:   %6 = icmp ne i32 %5, 0
// CHECK-NEXT:   %7 = zext i1 %6 to i32
// CHECK-NEXT:   %omp_floor0.tripcount = add nuw i32 %4, %7
// CHECK-NEXT:   br label %omp_floor0.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.preheader:                             ; preds = %omp_omp.loop.preheader
// CHECK-NEXT:   br label %omp_floor0.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.header:                                ; preds = %omp_floor0.inc, %omp_floor0.preheader
// CHECK-NEXT:   %omp_floor0.iv = phi i32 [ 0, %omp_floor0.preheader ], [ %omp_floor0.next, %omp_floor0.inc ]
// CHECK-NEXT:   br label %omp_floor0.cond
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.cond:                                  ; preds = %omp_floor0.header
// CHECK-NEXT:   %omp_floor0.cmp = icmp ult i32 %omp_floor0.iv, %omp_floor0.tripcount
// CHECK-NEXT:   br i1 %omp_floor0.cmp, label %omp_floor0.body, label %omp_floor0.exit
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.body:                                  ; preds = %omp_floor0.cond
// CHECK-NEXT:   %8 = icmp eq i32 %omp_floor0.iv, %4
// CHECK-NEXT:   %9 = select i1 %8, i32 %5, i32 %2
// CHECK-NEXT:   br label %omp_tile0.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.preheader:                              ; preds = %omp_floor0.body
// CHECK-NEXT:   br label %omp_tile0.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.header:                                 ; preds = %omp_tile0.inc, %omp_tile0.preheader
// CHECK-NEXT:   %omp_tile0.iv = phi i32 [ 0, %omp_tile0.preheader ], [ %omp_tile0.next, %omp_tile0.inc ]
// CHECK-NEXT:   br label %omp_tile0.cond
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.cond:                                   ; preds = %omp_tile0.header
// CHECK-NEXT:   %omp_tile0.cmp = icmp ult i32 %omp_tile0.iv, %9
// CHECK-NEXT:   br i1 %omp_tile0.cmp, label %omp_tile0.body, label %omp_tile0.exit
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.body:                                   ; preds = %omp_tile0.cond
// CHECK-NEXT:   %10 = mul nuw i32 %2, %omp_floor0.iv
// CHECK-NEXT:   %11 = add nuw i32 %10, %omp_tile0.iv
// CHECK-NEXT:   br label %omp_omp.loop.body
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.body:                                ; preds = %omp_tile0.body
// CHECK-NEXT:   br label %omp.loop.region
// CHECK-EMPTY:
// CHECK-NEXT: omp.loop.region:                                  ; preds = %omp_omp.loop.body
// CHECK-NEXT:   %12 = getelementptr inbounds float, ptr %0, i32 %11
// CHECK-NEXT:   store float 4.200000e+01, ptr %12, align 4
// CHECK-NEXT:   br label %omp.region.cont
// CHECK-EMPTY:
// CHECK-NEXT: omp.region.cont:                                  ; preds = %omp.loop.region
// CHECK-NEXT:   br label %omp_tile0.inc
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.inc:                                    ; preds = %omp.region.cont
// CHECK-NEXT:   %omp_tile0.next = add nuw i32 %omp_tile0.iv, 1
// CHECK-NEXT:   br label %omp_tile0.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.exit:                                   ; preds = %omp_tile0.cond
// CHECK-NEXT:   br label %omp_tile0.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.after:                                  ; preds = %omp_tile0.exit
// CHECK-NEXT:   br label %omp_floor0.inc
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.inc:                                   ; preds = %omp_tile0.after
// CHECK-NEXT:   %omp_floor0.next = add nuw i32 %omp_floor0.iv, 1
// CHECK-NEXT:   br label %omp_floor0.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.exit:                                  ; preds = %omp_floor0.cond
// CHECK-NEXT:   br label %omp_floor0.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.after:                                 ; preds = %omp_floor0.exit
// CHECK-NEXT:   br label %omp_omp.loop.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.after:                               ; preds = %omp_floor0.after
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: !llvm.module.flags = !{!0}
// CHECK-EMPTY:
// CHECK-NEXT: !0 = !{i32 2, !"Debug Info Version", i32 3}
