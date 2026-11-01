; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; The length argument of the memcpy/memmove/memset intrinsics must be a scalar integer.

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memcpy.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)
declare void @llvm.memcpy.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memcpy.inline.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)
declare void @llvm.memcpy.inline.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memmove.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)
declare void @llvm.memmove.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)

; CHECK: intrinsic argument 2 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memset.p0.v4i32(ptr, i8, <4 x i32>, i1)
declare void @llvm.memset.p0.v4i32(ptr, i8, <4 x i32>, i1)

; CHECK: intrinsic argument 2 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memset.inline.p0.v4i32(ptr, i8, <4 x i32>, i1)
declare void @llvm.memset.inline.p0.v4i32(ptr, i8, <4 x i32>, i1)

; The count argument of llvm.experimental.memset.pattern must be a scalar integer.

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.memset.pattern.p0.i32.v4i32(ptr, i32, <4 x i32>, i1)
declare void @llvm.experimental.memset.pattern.p0.i32.v4i32(ptr, i32, <4 x i32>, i1)
