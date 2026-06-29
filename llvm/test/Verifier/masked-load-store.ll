; RUN: not opt -S -passes=verify -disable-output 2>&1 < %s | FileCheck %s

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with vscale x 4 elements (overload type 0 is <vscale x 4 x i32>), but got <4 x i1>
; CHECK-NEXT: declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr, <4 x i1>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr, <4 x i1>, <vscale x 4 x i32>)

; CHECK: intrinsic return type (overload type 0) expected any vector type, but got i32
; CHECK-NEXT: declare i32 @llvm.masked.load.i32.p0(ptr, i1, i32)
declare i32 @llvm.masked.load.i32.p0(ptr, i1, i32)

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector (overload type 0 is <4 x i32>), but got i1
; CHECK-NEXT: declare <4 x i32> @llvm.masked.load.v4i32.p0(ptr, i1, <4 x i32>)
declare <4 x i32> @llvm.masked.load.v4i32.p0(ptr, i1, <4 x i32>)

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with 8 elements (overload type 0 is <8 x i32>), but got <5 x i1>
; CHECK-NEXT: declare <8 x i32> @llvm.masked.load.v8i32.p0(ptr, <5 x i1>, <8 x i32>)
declare <8 x i32> @llvm.masked.load.v8i32.p0(ptr, <5 x i1>, <8 x i32>)

; CHECK: intrinsic argument 2 type (matching overload type 0) expected <7 x i32>, but got <6 x i32>
; CHECK-NEXT: declare <7 x i32> @llvm.masked.load.v7i32.p0(ptr, <7 x i1>, <6 x i32>)
declare <7 x i32> @llvm.masked.load.v7i32.p0(ptr, <7 x i1>, <6 x i32>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any vector type, but got i32
; CHECK-NEXT: declare void @llvm.masked.store.i32.p0(i32, ptr, i1)
declare void @llvm.masked.store.i32.p0(i32, ptr, i1)

; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector (overload type 0 is <4 x i32>), but got i1
; CHECK-NEXT: declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr, i1)
declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr, i1)

; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector with 5 elements (overload type 0 is <5 x i32>), but got <4 x i1>
; CHECK-NEXT: declare void @llvm.masked.store.v5i32.p0(<5 x i32>, ptr, <4 x i1>)
declare void @llvm.masked.store.v5i32.p0(<5 x i32>, ptr, <4 x i1>)
