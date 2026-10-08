; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; The return type must be a scalar integer.
; CHECK: intrinsic return type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.objectsize.v4i32.p0(ptr, i1, i1, i1)
declare <4 x i32> @llvm.objectsize.v4i32.p0(ptr, i1, i1, i1)
