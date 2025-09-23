// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s


llvm.func @tile_2d_loop(%baseptr: !llvm.ptr, %tc1: i32, %tc2: i32, %ts1: i32, %ts2: i32) -> () {
  %literal_outer = omp.new_cli
  %literal_inner = omp.new_cli
  omp.canonical_loop(%literal_outer) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%literal_inner) %iv2 : i32 in range(%tc2) {
      %idx = llvm.add %iv1, %iv2 : i32
      %ptr = llvm.getelementptr inbounds %baseptr[%idx] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %val = llvm.mlir.constant(42.0 : f32) : f32
      llvm.store %val, %ptr : f32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  omp.tile <- (%literal_outer, %literal_inner) sizes(%ts1, %ts2 : i32,i32)
  llvm.return
}


// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECK-NEXT: source_filename = "LLVMDialectModule"
// CHECK-EMPTY:
// CHECK-NEXT: define void @tile_2d_loop(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4) {
// CHECK-NEXT:   br label %omp_omp.loop.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.preheader:                           ; preds = %5
// CHECK-NEXT:   %6 = udiv i32 %1, %3
// CHECK-NEXT:   %7 = urem i32 %1, %3
// CHECK-NEXT:   %8 = icmp ne i32 %7, 0
// CHECK-NEXT:   %9 = zext i1 %8 to i32
// CHECK-NEXT:   %omp_floor0.tripcount = add nuw i32 %6, %9
// CHECK-NEXT:   %10 = udiv i32 %2, %4
// CHECK-NEXT:   %11 = urem i32 %2, %4
// CHECK-NEXT:   %12 = icmp ne i32 %11, 0
// CHECK-NEXT:   %13 = zext i1 %12 to i32
// CHECK-NEXT:   %omp_floor1.tripcount = add nuw i32 %10, %13
// CHECK-NEXT:   br label %omp_floor0.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.header:                              ; preds = %omp_omp.loop.inc
// CHECK-NEXT:   %omp_omp.loop.iv = phi i32 [ %omp_omp.loop.next, %omp_omp.loop.inc ]
// CHECK-NEXT:   br label %omp_omp.loop.cond
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.cond:                                ; preds = %omp_omp.loop.header
// CHECK-NEXT:   %omp_omp.loop.cmp = icmp ult i32 %19, %1
// CHECK-NEXT:   br i1 %omp_omp.loop.cmp, label %omp_omp.loop.body, label %omp_omp.loop.exit
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.body:                                ; preds = %omp_tile1.body, %omp_omp.loop.cond
// CHECK-NEXT:   br label %omp.loop.region
// CHECK-EMPTY:
// CHECK-NEXT: omp.loop.region:                                  ; preds = %omp_omp.loop.body
// CHECK-NEXT:   br label %omp_omp.loop.preheader1
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.preheader1:                          ; preds = %omp.loop.region
// CHECK-NEXT:   br label %omp_omp.loop.body4
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
// CHECK-NEXT:   br label %omp_floor1.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor1.preheader:                             ; preds = %omp_floor0.body
// CHECK-NEXT:   br label %omp_floor1.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor1.header:                                ; preds = %omp_floor1.inc, %omp_floor1.preheader
// CHECK-NEXT:   %omp_floor1.iv = phi i32 [ 0, %omp_floor1.preheader ], [ %omp_floor1.next, %omp_floor1.inc ]
// CHECK-NEXT:   br label %omp_floor1.cond
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor1.cond:                                  ; preds = %omp_floor1.header
// CHECK-NEXT:   %omp_floor1.cmp = icmp ult i32 %omp_floor1.iv, %omp_floor1.tripcount
// CHECK-NEXT:   br i1 %omp_floor1.cmp, label %omp_floor1.body, label %omp_floor1.exit
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor1.body:                                  ; preds = %omp_floor1.cond
// CHECK-NEXT:   %14 = icmp eq i32 %omp_floor0.iv, %6
// CHECK-NEXT:   %15 = select i1 %14, i32 %7, i32 %3
// CHECK-NEXT:   %16 = icmp eq i32 %omp_floor1.iv, %10
// CHECK-NEXT:   %17 = select i1 %16, i32 %11, i32 %4
// CHECK-NEXT:   br label %omp_tile0.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.preheader:                              ; preds = %omp_floor1.body
// CHECK-NEXT:   br label %omp_tile0.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.header:                                 ; preds = %omp_tile0.inc, %omp_tile0.preheader
// CHECK-NEXT:   %omp_tile0.iv = phi i32 [ 0, %omp_tile0.preheader ], [ %omp_tile0.next, %omp_tile0.inc ]
// CHECK-NEXT:   br label %omp_tile0.cond
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.cond:                                   ; preds = %omp_tile0.header
// CHECK-NEXT:   %omp_tile0.cmp = icmp ult i32 %omp_tile0.iv, %15
// CHECK-NEXT:   br i1 %omp_tile0.cmp, label %omp_tile0.body, label %omp_tile0.exit
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.body:                                   ; preds = %omp_tile0.cond
// CHECK-NEXT:   br label %omp_tile1.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile1.preheader:                              ; preds = %omp_tile0.body
// CHECK-NEXT:   br label %omp_tile1.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile1.header:                                 ; preds = %omp_tile1.inc, %omp_tile1.preheader
// CHECK-NEXT:   %omp_tile1.iv = phi i32 [ 0, %omp_tile1.preheader ], [ %omp_tile1.next, %omp_tile1.inc ]
// CHECK-NEXT:   br label %omp_tile1.cond
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile1.cond:                                   ; preds = %omp_tile1.header
// CHECK-NEXT:   %omp_tile1.cmp = icmp ult i32 %omp_tile1.iv, %17
// CHECK-NEXT:   br i1 %omp_tile1.cmp, label %omp_tile1.body, label %omp_tile1.exit
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile1.body:                                   ; preds = %omp_tile1.cond
// CHECK-NEXT:   %18 = mul nuw i32 %3, %omp_floor0.iv
// CHECK-NEXT:   %19 = add nuw i32 %18, %omp_tile0.iv
// CHECK-NEXT:   %20 = mul nuw i32 %4, %omp_floor1.iv
// CHECK-NEXT:   %21 = add nuw i32 %20, %omp_tile1.iv
// CHECK-NEXT:   br label %omp_omp.loop.body
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.body4:                               ; preds = %omp_omp.loop.preheader1
// CHECK-NEXT:   br label %omp.loop.region12
// CHECK-EMPTY:
// CHECK-NEXT: omp.loop.region12:                                ; preds = %omp_omp.loop.body4
// CHECK-NEXT:   %22 = add i32 %19, %21
// CHECK-NEXT:   %23 = getelementptr inbounds float, ptr %0, i32 %22
// CHECK-NEXT:   store float 4.200000e+01, ptr %23, align 4
// CHECK-NEXT:   br label %omp.region.cont11
// CHECK-EMPTY:
// CHECK-NEXT: omp.region.cont11:                                ; preds = %omp.loop.region12
// CHECK-NEXT:   br label %omp_tile1.inc
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile1.inc:                                    ; preds = %omp.region.cont11
// CHECK-NEXT:   %omp_tile1.next = add nuw i32 %omp_tile1.iv, 1
// CHECK-NEXT:   br label %omp_tile1.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile1.exit:                                   ; preds = %omp_tile1.cond
// CHECK-NEXT:   br label %omp_tile1.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile1.after:                                  ; preds = %omp_tile1.exit
// CHECK-NEXT:   br label %omp_tile0.inc
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.inc:                                    ; preds = %omp_tile1.after
// CHECK-NEXT:   %omp_tile0.next = add nuw i32 %omp_tile0.iv, 1
// CHECK-NEXT:   br label %omp_tile0.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.exit:                                   ; preds = %omp_tile0.cond
// CHECK-NEXT:   br label %omp_tile0.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_tile0.after:                                  ; preds = %omp_tile0.exit
// CHECK-NEXT:   br label %omp_floor1.inc
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor1.inc:                                   ; preds = %omp_tile0.after
// CHECK-NEXT:   %omp_floor1.next = add nuw i32 %omp_floor1.iv, 1
// CHECK-NEXT:   br label %omp_floor1.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor1.exit:                                  ; preds = %omp_floor1.cond
// CHECK-NEXT:   br label %omp_floor1.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor1.after:                                 ; preds = %omp_floor1.exit
// CHECK-NEXT:   br label %omp_floor0.inc
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.inc:                                   ; preds = %omp_floor1.after
// CHECK-NEXT:   %omp_floor0.next = add nuw i32 %omp_floor0.iv, 1
// CHECK-NEXT:   br label %omp_floor0.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.exit:                                  ; preds = %omp_floor0.cond
// CHECK-NEXT:   br label %omp_floor0.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_floor0.after:                                 ; preds = %omp_floor0.exit
// CHECK-NEXT:   br label %omp_omp.loop.after
// CHECK-EMPTY:
// CHECK-NEXT: omp.region.cont:                                  ; No predecessors!
// CHECK-NEXT:   br label %omp_omp.loop.inc
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.inc:                                 ; preds = %omp.region.cont
// CHECK-NEXT:   %omp_omp.loop.next = add nuw i32 %19, 1
// CHECK-NEXT:   br label %omp_omp.loop.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.exit:                                ; preds = %omp_omp.loop.cond
// CHECK-NEXT:   br label %omp_omp.loop.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_omp.loop.after:                               ; preds = %omp_floor0.after, %omp_omp.loop.exit
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: !llvm.module.flags = !{!0}
// CHECK-EMPTY:
// CHECK-NEXT: !0 = !{i32 2, !"Debug Info Version", i32 3}
