; RUN: %lli -jit-kind=mcjit %s | FileCheck %s
; RUN: %lli %s | FileCheck %s
; REQUIRES: fma3
; CHECK: 12.000000

@msg_double = internal global [4 x i8] c"%f\0A\00"

declare i32 @printf(ptr, ...)

define i32 @main() {
  %fma = tail call double @llvm.fma.f64(double 3.0, double 3.0, double 3.0) nounwind readnone

  call i32 (ptr,...) @printf(ptr @msg_double, double %fma)

  ret i32 0
}

declare double @llvm.fma.f64(double, double, double) nounwind readnone
