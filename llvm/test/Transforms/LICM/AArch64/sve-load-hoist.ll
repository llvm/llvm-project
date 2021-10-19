; RUN: opt -licm -mtriple aarch64-linux-gnu -mattr=+sve -S < %s | FileCheck %s

define void @no_hoist_load1_nxv2i64(<vscale x 2 x i64>* %out, i8* %in8, i32 %n) {
; CHECK-LABEL: @no_hoist_load1_nxv2i64(
; CHECK: entry:
; CHECK-NOT: load
; CHECK: for.body:
; CHECK: load
entry:
  %cmp0 = icmp ugt i32 %n, 0
  %invst = call {}* @llvm.invariant.start.p0i8(i64 16, i8* %in8)
  %in = bitcast i8* %in8 to <vscale x 2 x i64>*
  br i1 %cmp0, label %for.body, label %for.end

for.body:
  %i = phi i32 [0, %entry], [%inc, %for.body]
  %i2 = zext i32 %i to i64
  %ptr = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %out, i64 %i2
  %val = load <vscale x 2 x i64>, <vscale x 2 x i64>* %in, align 16
  store <vscale x 2 x i64> %val, <vscale x 2 x i64>* %ptr, align 16
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define void @no_hoist_gather(<vscale x 2 x i32>* %out_ptr, <vscale x 2 x i32>* %in_ptr, <vscale x 2 x i64> %ptr_vec, i64 %n, <vscale x 2 x i1> %pred) {
; CHECK-LABEL: @no_hoist_gather(
; CHECK: entry:
; CHECK-NOT: llvm.aarch64.sve.ld1.gather.scalar.offset
; CHECK: for.body:
; CHECK: llvm.aarch64.sve.ld1.gather.scalar.offset
entry:
  br label %for.body

for.body:
  %i = phi i64 [0, %entry], [%inc, %for.body]
  %gather = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1> %pred, <vscale x 2 x i64> %ptr_vec, i64 0)
  %in_ptr_gep = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %in_ptr, i64 %i
  %in_ptr_load = load <vscale x 2 x i32>, <vscale x 2 x i32>* %in_ptr_gep, align 8
  %sum = add <vscale x 2 x i32> %gather, %in_ptr_load
  %out_ptr_gep = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %out_ptr, i64 %i
  store  <vscale x 2 x i32> %sum, <vscale x 2 x i32>* %out_ptr_gep, align 8
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define void @no_hoist_scatter(<vscale x 2 x i32>* %out_ptr, <vscale x 2 x i32>* %in_ptr, <vscale x 2 x i64> %ptr_vec, i64 %n, <vscale x 2 x i1> %pred) {
; CHECK-LABEL: @no_hoist_scatter(
; CHECK: entry:
; CHECK-NOT: load
; CHECK: for.body:
; CHECK: load
entry:
  br label %for.body

for.body:
  %i = phi i64 [0, %entry], [%inc, %for.body]
  %in_ptr_load = load <vscale x 2 x i32>, <vscale x 2 x i32>* %in_ptr, align 8
  call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32> %in_ptr_load, <vscale x 2 x i1> %pred, <vscale x 2 x i64> %ptr_vec, i64 %i)
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) nounwind readonly

declare void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i32>, <vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)
