; RUN: opt -domtree -instcombine -loops -S < %s | FileCheck %s
; Note: The -loops above can be anything that requires the domtree, and is
; necessary to work around a pass-manager bug.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.s = type { double* }

; Function Attrs: nounwind uwtable
define void @_Z3fooR1s(%struct.s* nocapture readonly dereferenceable(8) %x) #0 {

; CHECK-LABEL: @_Z3fooR1s
; CHECK: call void @llvm.assume
; CHECK-NOT: call void @llvm.assume

entry:
  %a = getelementptr inbounds %struct.s, %struct.s* %x, i64 0, i32 0
  %0 = load double*, double** %a, align 8
  %ptrint = ptrtoint double* %0 to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.1, %for.body ]
  tail call void @llvm.assume(i1 %maskcond)
  %arrayidx = getelementptr inbounds double, double* %0, i64 %indvars.iv
  %1 = load double, double* %arrayidx, align 16
  %add = fadd double %1, 1.000000e+00
  tail call void @llvm.assume(i1 %maskcond)
  %mul = fmul double %add, 2.000000e+00
  store double %mul, double* %arrayidx, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  tail call void @llvm.assume(i1 %maskcond)
  %arrayidx.1 = getelementptr inbounds double, double* %0, i64 %indvars.iv.next
  %2 = load double, double* %arrayidx.1, align 8
  %add.1 = fadd double %2, 1.000000e+00
  tail call void @llvm.assume(i1 %maskcond)
  %mul.1 = fmul double %add.1, 2.000000e+00
  store double %mul.1, double* %arrayidx.1, align 8
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv.next, 1
  %exitcond.1 = icmp eq i64 %indvars.iv.next, 1599
  br i1 %exitcond.1, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare align 8 i8* @get()

; Check that redundant align assume is removed
; CHECK-LABEL: @test
; CHECK-NOT: call void @llvm.assume
define void @test1() {
  %p = call align 8 i8* @get()
  %ptrint = ptrtoint i8* %p to i64
  %maskedptr = and i64 %ptrint, 7
  %maskcond = icmp eq i64 %maskedptr, 0
  call void @llvm.assume(i1 %maskcond)
  ret void
}

; Check that redundant align assume is removed
; CHECK-LABEL: @test
; CHECK-NOT: call void @llvm.assume
define void @test3() {
  %p = alloca i8, align 8
  %ptrint = ptrtoint i8* %p to i64
  %maskedptr = and i64 %ptrint, 7
  %maskcond = icmp eq i64 %maskedptr, 0
  call void @llvm.assume(i1 %maskcond)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.assume(i1) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

