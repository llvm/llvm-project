; RUN: opt < %s -aa-pipeline=basic-aa -passes=dse -S | FileCheck %s

define <vscale x 4 x float> @dead_scalable_store(ptr %0) {
; CHECK-LABEL: define <vscale x 4 x float> @dead_scalable_store(
; CHECK: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.16, ptr nonnull %gep.arr.16, i32 1, <vscale x 4 x i1> %mask)
; CHECK-NOT: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.32, ptr nonnull %gep.arr.32, i32 1, <vscale x 4 x i1> %mask)
; CHECK: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.48, ptr nonnull %gep.arr.48, i32 1, <vscale x 4 x i1> %mask)
;
  %arr = alloca [64 x i32], align 4
  %mask = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 0, i32 4)

  %gep.0.16 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %gep.0.32 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %gep.0.48 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %gep.arr.16 = getelementptr inbounds nuw i8, ptr %arr, i64 16
  %gep.arr.32 = getelementptr inbounds nuw i8, ptr %arr, i64 32
  %gep.arr.48 = getelementptr inbounds nuw i8, ptr %arr, i64 48

  %load.0.16 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.0.16, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.16, ptr nonnull %gep.arr.16, i32 1, <vscale x 4 x i1> %mask)

  %load.0.32 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.0.32, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.32, ptr nonnull %gep.arr.32, i32 1, <vscale x 4 x i1> %mask)

  %load.0.48 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.0.48, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.48, ptr nonnull %gep.arr.48, i32 1, <vscale x 4 x i1> %mask)

  %faddop0 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.arr.16, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  %faddop1 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.arr.48, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  %fadd = fadd <vscale x 4 x float> %faddop0, %faddop1

  ret <vscale x 4 x float> %fadd
}

define <4 x float> @dead_scalable_store_fixed(ptr %0) {
; CHECK-LABEL: define <4 x float> @dead_scalable_store_fixed(
; CHECK: call void @llvm.masked.store.v4f32.p0(<4 x float> %load.0.16, ptr nonnull %gep.arr.16, i32 1, <4 x i1> %mask)
; CHECK-NOT: call void @llvm.masked.store.v4f32.p0(<4 x float> %load.0.32, ptr nonnull %gep.arr.36, i32 1, <4 x i1> %mask2)
; CHECK: call void @llvm.masked.store.v4f32.p0(<4 x float> %load.0.48, ptr nonnull %gep.arr.48, i32 1, <4 x i1> %mask)
;
  %arr = alloca [64 x i32], align 4
  %mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 0, i32 4)
  %mask2 = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 0, i32 3)

  %gep.0.16 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %gep.0.36 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %gep.0.48 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %gep.arr.16 = getelementptr inbounds nuw i8, ptr %arr, i64 16
  %gep.arr.36 = getelementptr inbounds nuw i8, ptr %arr, i64 36
  %gep.arr.48 = getelementptr inbounds nuw i8, ptr %arr, i64 48

  %load.0.16 = call <4 x float> @llvm.masked.load.v4f32.p0(ptr nonnull %gep.0.16, i32 1, <4 x i1> %mask, <4 x float> zeroinitializer)
  call void @llvm.masked.store.v4f32.p0(<4 x float> %load.0.16, ptr nonnull %gep.arr.16, i32 1, <4 x i1> %mask)

  %load.0.36 = call <4 x float> @llvm.masked.load.v4f32.p0(ptr nonnull %gep.0.36, i32 1, <4 x i1> %mask2, <4 x float> zeroinitializer)
  call void @llvm.masked.store.v4f32.p0(<4 x float> %load.0.36, ptr nonnull %gep.arr.36, i32 1, <4 x i1> %mask2)

  %load.0.48 = call <4 x float> @llvm.masked.load.v4f32.p0(ptr nonnull %gep.0.48, i32 1, <4 x i1> %mask, <4 x float> zeroinitializer)
  call void @llvm.masked.store.v4f32.p0(<4 x float> %load.0.48, ptr nonnull %gep.arr.48, i32 1, <4 x i1> %mask)

  %faddop0 = call <4 x float> @llvm.masked.load.v4f32.p0(ptr nonnull %gep.arr.16, i32 1, <4 x i1> %mask, <4 x float> zeroinitializer)
  %faddop1 = call <4 x float> @llvm.masked.load.v4f32.p0(ptr nonnull %gep.arr.48, i32 1, <4 x i1> %mask, <4 x float> zeroinitializer)
  %fadd = fadd <4 x float> %faddop0, %faddop1

  ret <4 x float> %fadd
}

define <vscale x 4 x float> @scalable_store_partial_overwrite(ptr %0) {
; CHECK-LABEL: define <vscale x 4 x float> @scalable_store_partial_overwrite(
; CHECK: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.16, ptr nonnull %gep.arr.16, i32 1, <vscale x 4 x i1> %mask)
; CHECK: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.30, ptr nonnull %gep.arr.30, i32 1, <vscale x 4 x i1> %mask)
; CHECK: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.48, ptr nonnull %gep.arr.48, i32 1, <vscale x 4 x i1> %mask)
;
  %arr = alloca [64 x i32], align 4
  %mask = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 0, i32 4)

  %gep.0.16 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %gep.0.30 = getelementptr inbounds nuw i8, ptr %0, i64 30
  %gep.0.48 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %gep.arr.16 = getelementptr inbounds nuw i8, ptr %arr, i64 16
  %gep.arr.30 = getelementptr inbounds nuw i8, ptr %arr, i64 30
  %gep.arr.48 = getelementptr inbounds nuw i8, ptr %arr, i64 48

  %load.0.16 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.0.16, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.16, ptr nonnull %gep.arr.16, i32 1, <vscale x 4 x i1> %mask)

  %load.0.30 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.0.30, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.30, ptr nonnull %gep.arr.30, i32 1, <vscale x 4 x i1> %mask)

  %load.0.48 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.0.48, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.48, ptr nonnull %gep.arr.48, i32 1, <vscale x 4 x i1> %mask)

  %faddop0 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.arr.16, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  %faddop1 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.arr.48, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  %fadd = fadd <vscale x 4 x float> %faddop0, %faddop1

  ret <vscale x 4 x float> %fadd
}

define <vscale x 4 x float> @dead_scalable_store_small_mask(ptr %0) {
; CHECK-LABEL: define <vscale x 4 x float> @dead_scalable_store_small_mask(
; CHECK: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.16, ptr nonnull %gep.arr.16, i32 1, <vscale x 4 x i1> %mask)
; CHECK-NOT: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.30, ptr nonnull %gep.arr.30, i32 1, <vscale x 4 x i1> %mask)
; CHECK: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.46, ptr nonnull %gep.arr.46, i32 1, <vscale x 4 x i1> %mask)
  %arr = alloca [64 x i32], align 4
  %mask = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 0, i32 4)

  %gep.0.16 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %gep.0.30 = getelementptr inbounds nuw i8, ptr %0, i64 30
  %gep.0.46 = getelementptr inbounds nuw i8, ptr %0, i64 46
  %gep.arr.16 = getelementptr inbounds nuw i8, ptr %arr, i64 16
  %gep.arr.30 = getelementptr inbounds nuw i8, ptr %arr, i64 30
  %gep.arr.46 = getelementptr inbounds nuw i8, ptr %arr, i64 46

  %load.0.16 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.0.16, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.16, ptr nonnull %gep.arr.16, i32 1, <vscale x 4 x i1> %mask)

  %load.0.30 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.0.30, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.30, ptr nonnull %gep.arr.30, i32 1, <vscale x 4 x i1> %mask)

  %load.0.46 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.0.46, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0.46, ptr nonnull %gep.arr.46, i32 1, <vscale x 4 x i1> %mask)

  %smallmask = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.32(i32 0, i32 2)
  %faddop0 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.arr.16, i32 1, <vscale x 4 x i1> %smallmask, <vscale x 4 x float> zeroinitializer)
  %faddop1 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %gep.arr.46, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  %fadd = fadd <vscale x 4 x float> %faddop0, %faddop1

  ret <vscale x 4 x float> %fadd
}

define <vscale x 4 x float> @dead_scalar_store(ptr noalias %0, ptr %1) {
; CHECK-LABEL: define <vscale x 4 x float> @dead_scalar_store(
; CHECK-NOT: store i32 20, ptr %gep.1.12
;
  %mask = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i128(i128 0, i128 4)
  %gep.1.12 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 20, ptr %gep.1.12

  %load.0 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %0, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0, ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask)
  %retval = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  ret <vscale x 4 x float> %retval
}


; CHECK-LABEL: define <4 x float> @dead_scalable_store_fixed_large_mask(
; CHECK-NOT: store i32 20, ptr %1
; CHECK: store i32 50, ptr %gep.5
define <4 x float> @dead_scalable_store_fixed_large_mask(ptr noalias %0, ptr %1) {
  %mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 0, i32 7)
  store i32 20, ptr %1

  %gep.5 = getelementptr inbounds nuw i32, ptr %1, i64 5
  store i32 50, ptr %gep.5

  %load.0 = call <4 x float> @llvm.masked.load.v4f32.p0(ptr nonnull %0, i32 1, <4 x i1> %mask, <4 x float> zeroinitializer)
  call void @llvm.masked.store.v4f32.p0(<4 x float> %load.0, ptr nonnull %1, i32 1, <4 x i1> %mask)
  %retval = call <4 x float> @llvm.masked.load.v4f32.p0(ptr nonnull %1, i32 1, <4 x i1> %mask, <4 x float> zeroinitializer)
  ret <4 x float> %retval
}

; We don't know if the scalar store is dead as we can't determine vscale.
; This get active lane mask may cover 4 or 8 integers
define <vscale x 4 x float> @mask_gt_minimum_num_elts(ptr noalias %0, ptr %1) {
; CHECK-LABEL: define <vscale x 4 x float> @mask_gt_minimum_num_elts(
; CHECK: store i32 10, ptr %gep.1.12
; CHECK: store i32 20, ptr %gep.1.28
;
  %mask = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 0, i32 8)
  %gep.1.12 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 10, ptr %gep.1.12
  %gep.1.28 = getelementptr inbounds nuw i8, ptr %1, i64 28
  store i32 20, ptr %gep.1.28

  %load.0 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %0, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0, ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask)
  %retval = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  ret <vscale x 4 x float> %retval
}

; Don't do anything if the mask's Op1 < Op0
define <vscale x 4 x float> @active_lane_mask_lt(ptr noalias %0, ptr %1) {
; CHECK-LABEL: define <vscale x 4 x float> @active_lane_mask_lt(
; CHECK: store i32 20, ptr %1
;
  %mask = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 4, i32 2)
  store i32 20, ptr %1

  %load.0 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %0, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0, ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask)
  %retval = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  ret <vscale x 4 x float> %retval
}

; Don't do anything if the mask's Op1 == Op0
define <vscale x 4 x float> @active_lane_mask_eq(ptr noalias %0, ptr %1) {
; CHECK-LABEL: define <vscale x 4 x float> @active_lane_mask_eq(
; CHECK: store i32 20, ptr %1
;
  %mask = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 2, i32 2)
  store i32 20, ptr %1

  %load.0 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %0, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0, ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask)
  %retval = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  ret <vscale x 4 x float> %retval
}

define <vscale x 16 x i8> @scalar_stores_small_mask(ptr noalias %0, ptr %1) {
; CHECK-LABEL: define <vscale x 16 x i8> @scalar_stores_small_mask(
; CHECK-NOT: store i8 60, ptr %gep.1.6
; CHECK: store i8 120, ptr %gep.1.8
;
  %mask = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i8.i8(i8 0, i8 7)
  %gep.1.6 = getelementptr inbounds nuw i8, ptr %1, i64 6
  store i8 60, ptr %gep.1.6
  %gep.1.8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i8 120, ptr %gep.1.8

  %load.0 = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr nonnull %0, i32 1, <vscale x 16 x i1> %mask, <vscale x 16 x i8> zeroinitializer)
  call void @llvm.masked.store.nxv16i8.p0(<vscale x 16 x i8> %load.0, ptr %1, i32 1, <vscale x 16 x i1> %mask)
  %retval = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr %1, i32 1, <vscale x 16 x i1> %mask, <vscale x 16 x i8> zeroinitializer)
  ret <vscale x 16 x i8> %retval
}

define <vscale x 4 x float> @dead_scalar_store_offset(ptr noalias %0, ptr %1) {
; CHECK-LABEL: define <vscale x 4 x float> @dead_scalar_store_offset(
; CHECK-NOT: store i32 10, ptr %gep.1.0
; CHECK-NOT: store i32 20, ptr %gep.1.4
; CHECK-NOT: store i32 30, ptr %gep.1.8
; CHECK: store i32 40, ptr %gep.1.12
;
  %mask = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 1, i32 4)
  %gep.1.0 = getelementptr inbounds nuw i8, ptr %1, i64 0
  store i32 10, ptr %gep.1.0
  %gep.1.4 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store i32 20, ptr %gep.1.4
  %gep.1.8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 30, ptr %gep.1.8
  %gep.1.12 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 40, ptr %gep.1.12

  %load.0 = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %0, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %load.0, ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask)
  %retval = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr nonnull %1, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x float> zeroinitializer)
  ret <vscale x 4 x float> %retval
}
