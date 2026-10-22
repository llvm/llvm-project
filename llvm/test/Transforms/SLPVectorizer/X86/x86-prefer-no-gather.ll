; Verify that the TuningPreferNoGather feature takes effect when enabled.
; Gather instructions are generated on CPUs that support them, but not
; when the tuning is on.

; RUN: opt -passes=slp-vectorizer -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 -S %s | FileCheck %s --check-prefix=GATHER
; RUN: opt -passes=slp-vectorizer -mtriple=x86_64-unknown-linux-gnu -mcpu=c86-4g-m7 -S %s | FileCheck %s --check-prefix=NO-GATHER
; RUN: opt -passes=slp-vectorizer -mtriple=x86_64-unknown-linux-gnu -mcpu=c86-4g-m8 -S %s | FileCheck %s --check-prefix=NO-GATHER

target triple = "x86_64-unknown-linux-gnu"

; GATHER-LABEL: @gather_test(
; GATHER: call <{{.*}} x float> @llvm.masked.gather

; NO-GATHER-LABEL: @gather_test(
; NO-GATHER-NOT: @llvm.masked.gather

define float @gather_test(ptr noalias %p, ptr noalias %addr) {
entry:
  %0 = load i32, ptr %addr, align 4
  %1 = getelementptr inbounds float, ptr %p, i32 %0
  %2 = load float, ptr %1, align 4

  %3 = getelementptr inbounds i32, ptr %addr, i32 1
  %4 = load i32, ptr %3, align 4
  %5 = getelementptr inbounds float, ptr %p, i32 %4
  %6 = load float, ptr %5, align 4

  %7 = getelementptr inbounds i32, ptr %addr, i32 2
  %8 = load i32, ptr %7, align 4
  %9 = getelementptr inbounds float, ptr %p, i32 %8
  %10 = load float, ptr %9, align 4

  %11 = getelementptr inbounds i32, ptr %addr, i32 3
  %12 = load i32, ptr %11, align 4
  %13 = getelementptr inbounds float, ptr %p, i32 %12
  %14 = load float, ptr %13, align 4

  %15 = getelementptr inbounds i32, ptr %addr, i32 4
  %16 = load i32, ptr %15, align 4
  %17 = getelementptr inbounds float, ptr %p, i32 %16
  %18 = load float, ptr %17, align 4

  %19 = getelementptr inbounds i32, ptr %addr, i32 5
  %20 = load i32, ptr %19, align 4
  %21 = getelementptr inbounds float, ptr %p, i32 %20
  %22 = load float, ptr %21, align 4

  %23 = getelementptr inbounds i32, ptr %addr, i32 6
  %24 = load i32, ptr %23, align 4
  %25 = getelementptr inbounds float, ptr %p, i32 %24
  %26 = load float, ptr %25, align 4

  %27 = getelementptr inbounds i32, ptr %addr, i32 7
  %28 = load i32, ptr %27, align 4
  %29 = getelementptr inbounds float, ptr %p, i32 %28
  %30 = load float, ptr %29, align 4

  %31 = fadd float %2, %6
  %32 = fadd float %31, %10
  %33 = fadd float %32, %14
  %34 = fadd float %33, %18
  %35 = fadd float %34, %22
  %36 = fadd float %35, %26
  %37 = fadd float %36, %30

  ret float %37
}
