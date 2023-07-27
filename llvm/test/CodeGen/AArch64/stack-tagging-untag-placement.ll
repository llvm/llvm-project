;; RUN:  opt -S -aarch64-stack-tagging %s -o - | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

define void @f() local_unnamed_addr #0  {
S0:
; CHECK-LABEL: S0:
; CHECK: %basetag = call ptr @llvm.aarch64.irg.sp(i64 0)
  %v = alloca i8, i32 48, align 8
; CHECK: %v.tag = call ptr @llvm.aarch64.tagp.p0(ptr %v, ptr %basetag, i64 0)
  %w = alloca i8, i32 48, align 16
; CHECK: %w.tag = call ptr @llvm.aarch64.tagp.p0(ptr %w, ptr %basetag, i64 1)

  %t0 = call i32 @g0() #1
  %b0 = icmp eq i32 %t0, 0
  br i1 %b0, label %S1, label %exit3

S1:
; CHECK-LABEL: S1:
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %v) #1
; CHECK: call void @llvm.aarch64.settag(ptr %v.tag, i64 48)
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %w) #1
; CHECK: call void @llvm.aarch64.settag(ptr %w.tag, i64 48)
  %t1 = call i32 @g1(ptr nonnull %v, ptr nonnull %w) #1
; CHECK: call i32 @g1
; CHECK-NOT: settag{{.*}}%v
; CHECK: call void @llvm.aarch64.settag(ptr %w, i64 48)
; CHECK-NOT: settag{{.*}}%v
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %w) #1
; CHECK: call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %w.tag)
  %b1 = icmp eq i32 %t1, 0
  br i1 %b1, label %S2, label %S3
; CHECK-NOT: settag

S2:
; CHECK-LABEL: S2:
  call void @z0() #1
  br label %exit1
; CHECK-NOT: settag

S3:
; CHECK-LABEL: S3:
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %v) #1
  tail call void @z1() #1
  br label %exit2
; CHECK-NOT: settag

exit1:
; CHECK-LABEL: exit1:
; CHECK: call void @llvm.aarch64.settag(ptr %v, i64 48)
  ret void

exit2:
; CHECK-LABEL: exit2:
; CHECK: call void @llvm.aarch64.settag(ptr %v, i64 48)
  ret void

exit3:
; CHECK-LABEL: exit3:
  call void @z2() #1
; CHECK-NOT: settag
  ret void
; CHECK:  ret void
}

declare i32 @g0() #0

declare i32 @g1(ptr, ptr) #0

declare void @z0() #0

declare void @z1() #0

declare void @z2() #0

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { sanitize_memtag "correctly-rounded-divide-sqrt-fp-math"="false" "denormal-fp-math"="preserve-sign" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+mte,+neon,+v8.5a" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

