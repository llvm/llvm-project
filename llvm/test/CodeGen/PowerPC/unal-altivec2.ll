; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(ptr noalias nocapture %x, ptr noalias nocapture readonly %y) #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
; CHECK-LABEL: @foo
; CHECK: lvsl
; CHECK: blr
  %index = phi i64 [ 0, %entry ], [ %index.next.15, %vector.body ]
  %0 = getelementptr inbounds float, ptr %y, i64 %index
  %wide.load = load <4 x float>, ptr %0, align 4
  %1 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load)
  %2 = getelementptr inbounds float, ptr %x, i64 %index
  store <4 x float> %1, ptr %2, align 4
  %index.next = add i64 %index, 4
  %3 = getelementptr inbounds float, ptr %y, i64 %index.next
  %wide.load.1 = load <4 x float>, ptr %3, align 4
  %4 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.1)
  %5 = getelementptr inbounds float, ptr %x, i64 %index.next
  store <4 x float> %4, ptr %5, align 4
  %index.next.1 = add i64 %index.next, 4
  %6 = getelementptr inbounds float, ptr %y, i64 %index.next.1
  %wide.load.2 = load <4 x float>, ptr %6, align 4
  %7 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.2)
  %8 = getelementptr inbounds float, ptr %x, i64 %index.next.1
  store <4 x float> %7, ptr %8, align 4
  %index.next.2 = add i64 %index.next.1, 4
  %9 = getelementptr inbounds float, ptr %y, i64 %index.next.2
  %wide.load.3 = load <4 x float>, ptr %9, align 4
  %10 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.3)
  %11 = getelementptr inbounds float, ptr %x, i64 %index.next.2
  store <4 x float> %10, ptr %11, align 4
  %index.next.3 = add i64 %index.next.2, 4
  %12 = getelementptr inbounds float, ptr %y, i64 %index.next.3
  %wide.load.4 = load <4 x float>, ptr %12, align 4
  %13 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.4)
  %14 = getelementptr inbounds float, ptr %x, i64 %index.next.3
  store <4 x float> %13, ptr %14, align 4
  %index.next.4 = add i64 %index.next.3, 4
  %15 = getelementptr inbounds float, ptr %y, i64 %index.next.4
  %wide.load.5 = load <4 x float>, ptr %15, align 4
  %16 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.5)
  %17 = getelementptr inbounds float, ptr %x, i64 %index.next.4
  store <4 x float> %16, ptr %17, align 4
  %index.next.5 = add i64 %index.next.4, 4
  %18 = getelementptr inbounds float, ptr %y, i64 %index.next.5
  %wide.load.6 = load <4 x float>, ptr %18, align 4
  %19 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.6)
  %20 = getelementptr inbounds float, ptr %x, i64 %index.next.5
  store <4 x float> %19, ptr %20, align 4
  %index.next.6 = add i64 %index.next.5, 4
  %21 = getelementptr inbounds float, ptr %y, i64 %index.next.6
  %wide.load.7 = load <4 x float>, ptr %21, align 4
  %22 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.7)
  %23 = getelementptr inbounds float, ptr %x, i64 %index.next.6
  store <4 x float> %22, ptr %23, align 4
  %index.next.7 = add i64 %index.next.6, 4
  %24 = getelementptr inbounds float, ptr %y, i64 %index.next.7
  %wide.load.8 = load <4 x float>, ptr %24, align 4
  %25 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.8)
  %26 = getelementptr inbounds float, ptr %x, i64 %index.next.7
  store <4 x float> %25, ptr %26, align 4
  %index.next.8 = add i64 %index.next.7, 4
  %27 = getelementptr inbounds float, ptr %y, i64 %index.next.8
  %wide.load.9 = load <4 x float>, ptr %27, align 4
  %28 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.9)
  %29 = getelementptr inbounds float, ptr %x, i64 %index.next.8
  store <4 x float> %28, ptr %29, align 4
  %index.next.9 = add i64 %index.next.8, 4
  %30 = getelementptr inbounds float, ptr %y, i64 %index.next.9
  %wide.load.10 = load <4 x float>, ptr %30, align 4
  %31 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.10)
  %32 = getelementptr inbounds float, ptr %x, i64 %index.next.9
  store <4 x float> %31, ptr %32, align 4
  %index.next.10 = add i64 %index.next.9, 4
  %33 = getelementptr inbounds float, ptr %y, i64 %index.next.10
  %wide.load.11 = load <4 x float>, ptr %33, align 4
  %34 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.11)
  %35 = getelementptr inbounds float, ptr %x, i64 %index.next.10
  store <4 x float> %34, ptr %35, align 4
  %index.next.11 = add i64 %index.next.10, 4
  %36 = getelementptr inbounds float, ptr %y, i64 %index.next.11
  %wide.load.12 = load <4 x float>, ptr %36, align 4
  %37 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.12)
  %38 = getelementptr inbounds float, ptr %x, i64 %index.next.11
  store <4 x float> %37, ptr %38, align 4
  %index.next.12 = add i64 %index.next.11, 4
  %39 = getelementptr inbounds float, ptr %y, i64 %index.next.12
  %wide.load.13 = load <4 x float>, ptr %39, align 4
  %40 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.13)
  %41 = getelementptr inbounds float, ptr %x, i64 %index.next.12
  store <4 x float> %40, ptr %41, align 4
  %index.next.13 = add i64 %index.next.12, 4
  %42 = getelementptr inbounds float, ptr %y, i64 %index.next.13
  %wide.load.14 = load <4 x float>, ptr %42, align 4
  %43 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.14)
  %44 = getelementptr inbounds float, ptr %x, i64 %index.next.13
  store <4 x float> %43, ptr %44, align 4
  %index.next.14 = add i64 %index.next.13, 4
  %45 = getelementptr inbounds float, ptr %y, i64 %index.next.14
  %wide.load.15 = load <4 x float>, ptr %45, align 4
  %46 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.15)
  %47 = getelementptr inbounds float, ptr %x, i64 %index.next.14
  store <4 x float> %46, ptr %47, align 4
  %index.next.15 = add i64 %index.next.14, 4
  %48 = icmp eq i64 %index.next.15, 2048
  br i1 %48, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm_cos_v4f32(<4 x float>) #1

define <2 x double> @bar(ptr %x) {
entry:
  %r = load <2 x double>, ptr %x, align 8

; CHECK-LABEL: @bar
; CHECK-NOT: lvsl
; CHECK: blr

  ret <2 x double> %r
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
