; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; The allocation size must be a scalar integer.
; CHECK: intrinsic argument 0 type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare token @llvm.coro.alloca.alloc.v4i32(<4 x i32>, i32)
declare token @llvm.coro.alloca.alloc.v4i32(<4 x i32>, i32)
