; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(float* noalias nocapture %x, float* noalias nocapture readonly %y) #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
; CHECK-LABEL: @foo
; CHECK: lvsl
; CHECK: blr
  %index = phi i64 [ 0, %entry ], [ %index.next.15, %vector.body ]
  %0 = getelementptr inbounds float, float* %y, i64 %index
  %1 = bitcast float* %0 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %1, align 4
  %2 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load)
  %3 = getelementptr inbounds float, float* %x, i64 %index
  %4 = bitcast float* %3 to <4 x float>*
  store <4 x float> %2, <4 x float>* %4, align 4
  %index.next = add i64 %index, 4
  %5 = getelementptr inbounds float, float* %y, i64 %index.next
  %6 = bitcast float* %5 to <4 x float>*
  %wide.load.1 = load <4 x float>, <4 x float>* %6, align 4
  %7 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.1)
  %8 = getelementptr inbounds float, float* %x, i64 %index.next
  %9 = bitcast float* %8 to <4 x float>*
  store <4 x float> %7, <4 x float>* %9, align 4
  %index.next.1 = add i64 %index.next, 4
  %10 = getelementptr inbounds float, float* %y, i64 %index.next.1
  %11 = bitcast float* %10 to <4 x float>*
  %wide.load.2 = load <4 x float>, <4 x float>* %11, align 4
  %12 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.2)
  %13 = getelementptr inbounds float, float* %x, i64 %index.next.1
  %14 = bitcast float* %13 to <4 x float>*
  store <4 x float> %12, <4 x float>* %14, align 4
  %index.next.2 = add i64 %index.next.1, 4
  %15 = getelementptr inbounds float, float* %y, i64 %index.next.2
  %16 = bitcast float* %15 to <4 x float>*
  %wide.load.3 = load <4 x float>, <4 x float>* %16, align 4
  %17 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.3)
  %18 = getelementptr inbounds float, float* %x, i64 %index.next.2
  %19 = bitcast float* %18 to <4 x float>*
  store <4 x float> %17, <4 x float>* %19, align 4
  %index.next.3 = add i64 %index.next.2, 4
  %20 = getelementptr inbounds float, float* %y, i64 %index.next.3
  %21 = bitcast float* %20 to <4 x float>*
  %wide.load.4 = load <4 x float>, <4 x float>* %21, align 4
  %22 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.4)
  %23 = getelementptr inbounds float, float* %x, i64 %index.next.3
  %24 = bitcast float* %23 to <4 x float>*
  store <4 x float> %22, <4 x float>* %24, align 4
  %index.next.4 = add i64 %index.next.3, 4
  %25 = getelementptr inbounds float, float* %y, i64 %index.next.4
  %26 = bitcast float* %25 to <4 x float>*
  %wide.load.5 = load <4 x float>, <4 x float>* %26, align 4
  %27 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.5)
  %28 = getelementptr inbounds float, float* %x, i64 %index.next.4
  %29 = bitcast float* %28 to <4 x float>*
  store <4 x float> %27, <4 x float>* %29, align 4
  %index.next.5 = add i64 %index.next.4, 4
  %30 = getelementptr inbounds float, float* %y, i64 %index.next.5
  %31 = bitcast float* %30 to <4 x float>*
  %wide.load.6 = load <4 x float>, <4 x float>* %31, align 4
  %32 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.6)
  %33 = getelementptr inbounds float, float* %x, i64 %index.next.5
  %34 = bitcast float* %33 to <4 x float>*
  store <4 x float> %32, <4 x float>* %34, align 4
  %index.next.6 = add i64 %index.next.5, 4
  %35 = getelementptr inbounds float, float* %y, i64 %index.next.6
  %36 = bitcast float* %35 to <4 x float>*
  %wide.load.7 = load <4 x float>, <4 x float>* %36, align 4
  %37 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.7)
  %38 = getelementptr inbounds float, float* %x, i64 %index.next.6
  %39 = bitcast float* %38 to <4 x float>*
  store <4 x float> %37, <4 x float>* %39, align 4
  %index.next.7 = add i64 %index.next.6, 4
  %40 = getelementptr inbounds float, float* %y, i64 %index.next.7
  %41 = bitcast float* %40 to <4 x float>*
  %wide.load.8 = load <4 x float>, <4 x float>* %41, align 4
  %42 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.8)
  %43 = getelementptr inbounds float, float* %x, i64 %index.next.7
  %44 = bitcast float* %43 to <4 x float>*
  store <4 x float> %42, <4 x float>* %44, align 4
  %index.next.8 = add i64 %index.next.7, 4
  %45 = getelementptr inbounds float, float* %y, i64 %index.next.8
  %46 = bitcast float* %45 to <4 x float>*
  %wide.load.9 = load <4 x float>, <4 x float>* %46, align 4
  %47 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.9)
  %48 = getelementptr inbounds float, float* %x, i64 %index.next.8
  %49 = bitcast float* %48 to <4 x float>*
  store <4 x float> %47, <4 x float>* %49, align 4
  %index.next.9 = add i64 %index.next.8, 4
  %50 = getelementptr inbounds float, float* %y, i64 %index.next.9
  %51 = bitcast float* %50 to <4 x float>*
  %wide.load.10 = load <4 x float>, <4 x float>* %51, align 4
  %52 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.10)
  %53 = getelementptr inbounds float, float* %x, i64 %index.next.9
  %54 = bitcast float* %53 to <4 x float>*
  store <4 x float> %52, <4 x float>* %54, align 4
  %index.next.10 = add i64 %index.next.9, 4
  %55 = getelementptr inbounds float, float* %y, i64 %index.next.10
  %56 = bitcast float* %55 to <4 x float>*
  %wide.load.11 = load <4 x float>, <4 x float>* %56, align 4
  %57 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.11)
  %58 = getelementptr inbounds float, float* %x, i64 %index.next.10
  %59 = bitcast float* %58 to <4 x float>*
  store <4 x float> %57, <4 x float>* %59, align 4
  %index.next.11 = add i64 %index.next.10, 4
  %60 = getelementptr inbounds float, float* %y, i64 %index.next.11
  %61 = bitcast float* %60 to <4 x float>*
  %wide.load.12 = load <4 x float>, <4 x float>* %61, align 4
  %62 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.12)
  %63 = getelementptr inbounds float, float* %x, i64 %index.next.11
  %64 = bitcast float* %63 to <4 x float>*
  store <4 x float> %62, <4 x float>* %64, align 4
  %index.next.12 = add i64 %index.next.11, 4
  %65 = getelementptr inbounds float, float* %y, i64 %index.next.12
  %66 = bitcast float* %65 to <4 x float>*
  %wide.load.13 = load <4 x float>, <4 x float>* %66, align 4
  %67 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.13)
  %68 = getelementptr inbounds float, float* %x, i64 %index.next.12
  %69 = bitcast float* %68 to <4 x float>*
  store <4 x float> %67, <4 x float>* %69, align 4
  %index.next.13 = add i64 %index.next.12, 4
  %70 = getelementptr inbounds float, float* %y, i64 %index.next.13
  %71 = bitcast float* %70 to <4 x float>*
  %wide.load.14 = load <4 x float>, <4 x float>* %71, align 4
  %72 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.14)
  %73 = getelementptr inbounds float, float* %x, i64 %index.next.13
  %74 = bitcast float* %73 to <4 x float>*
  store <4 x float> %72, <4 x float>* %74, align 4
  %index.next.14 = add i64 %index.next.13, 4
  %75 = getelementptr inbounds float, float* %y, i64 %index.next.14
  %76 = bitcast float* %75 to <4 x float>*
  %wide.load.15 = load <4 x float>, <4 x float>* %76, align 4
  %77 = call <4 x float> @llvm_cos_v4f32(<4 x float> %wide.load.15)
  %78 = getelementptr inbounds float, float* %x, i64 %index.next.14
  %79 = bitcast float* %78 to <4 x float>*
  store <4 x float> %77, <4 x float>* %79, align 4
  %index.next.15 = add i64 %index.next.14, 4
  %80 = icmp eq i64 %index.next.15, 2048
  br i1 %80, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm_cos_v4f32(<4 x float>) #1

define <2 x double> @bar(double* %x) {
entry:
  %p = bitcast double* %x to <2 x double>*
  %r = load <2 x double>, <2 x double>* %p, align 8

; CHECK-LABEL: @bar
; CHECK-NOT: lvsl
; CHECK: blr

  ret <2 x double> %r
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
