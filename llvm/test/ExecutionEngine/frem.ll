; LLI.exe used to crash on Windows\X86 when certain single precession
; floating point intrinsics (defined as macros) are used.
; This unit test guards against the failure.
;
; RUN: %lli -jit-kind=mcjit %s | FileCheck %s
; RUN: %lli %s | FileCheck %s

@flt = internal global float 12.0e+0
@str = internal constant [18 x i8] c"Double value: %f\0A\00"

declare i32 @printf(i8* nocapture, ...) nounwind
declare i32 @fflush(i8*) nounwind

define i32 @main() {
  %flt = load float, float* @flt
  %float2 = frem float %flt, 5.0
  %double1 = fpext float %float2 to double
  call i32 (i8*, ...) @printf(i8* getelementptr ([18 x i8], [18 x i8]* @str, i32 0, i64 0), double %double1)
  call i32 @fflush(i8* null)
  ret i32 0
}

; CHECK: Double value: 2.0
