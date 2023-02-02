; LoongArch does not support mcjit.
; UNSUPPORTED: target=loongarch{{.*}}

; LLI.exe used to crash on Windows\X86 when certain single precession
; floating point intrinsics (defined as macros) are used.
; This unit test guards against the failure.
;
; RUN: %lli -jit-kind=mcjit %s | FileCheck %s
; RUN: %lli %s | FileCheck %s

@flt = internal global float 12.0e+0
@str = internal constant [18 x i8] c"Double value: %f\0A\00"

declare i32 @printf(ptr nocapture, ...) nounwind
declare i32 @fflush(ptr) nounwind

define i32 @main() {
  %flt = load float, ptr @flt
  %float2 = frem float %flt, 5.0
  %double1 = fpext float %float2 to double
  call i32 (ptr, ...) @printf(ptr @str, double %double1)
  call i32 @fflush(ptr null)
  ret i32 0
}

; CHECK: Double value: 2.0
