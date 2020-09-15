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

declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) nounwind readonly

