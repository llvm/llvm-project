; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; The offset argument must be a scalar integer.
; CHECK: intrinsic argument 1 type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare ptr @llvm.load.relative.v4i32(ptr, <4 x i32>)
declare ptr @llvm.load.relative.v4i32(ptr, <4 x i32>)
