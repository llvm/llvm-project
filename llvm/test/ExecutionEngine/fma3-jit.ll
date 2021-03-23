; RUN: %lli %s | FileCheck %s
; REQUIRES: fma3
; CHECK: 12.000000

@msg_double = internal global [4 x i8] c"%f\0A\00"

declare i32 @printf(i8*, ...)

define i32 @main() {
  %fma = tail call double @llvm.fma.f64(double 3.0, double 3.0, double 3.0) nounwind readnone

  %ptr1 = getelementptr [4 x i8], [4 x i8]* @msg_double, i32 0, i32 0
  call i32 (i8*,...) @printf(i8* %ptr1, double %fma)

  ret i32 0
}

declare double @llvm.fma.f64(double, double, double) nounwind readnone
