; RUN: not opt -S -passes=verify 2>&1 < %s | FileCheck %s

declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr, <4 x i1>, <vscale x 4 x i32>)
define <vscale x 4 x i32> @masked_load(ptr %addr, <4 x i1> %mask, <vscale x 4 x i32> %dst) {
; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with vscale x 4 elements (overload type 0 is <vscale x 4 x i32>), but got <4 x i1>
; CHECK-NEXT: ptr @llvm.masked.load.nxv4i32.p0
  %res = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr %addr, <4 x i1> %mask, <vscale x 4 x i32> %dst)
  ret <vscale x 4 x i32> %res
}

declare i32 @llvm.masked.load.i32.p0(ptr, i1, i32)
define void @t1() {
; CHECK: intrinsic return type (overload type 0) expected any vector type, but got i32
; CHECK-NEXT: ptr @llvm.masked.load.i32.p0
  call i32 @llvm.masked.load.i32.p0(ptr poison, i1 0, i32 0)
  ret void
}

declare <4 x i32> @llvm.masked.load.v4i32.p0(ptr, i1, <4 x i32>)
define void @t2() {
; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector (overload type 0 is <4 x i32>), but got i1
; CHECK-NEXT: ptr @llvm.masked.load.v4i32.p0
  call <4 x i32> @llvm.masked.load.v4i32.p0(ptr poison, i1 0, <4 x i32> poison)
  ret void
}

declare <8 x i32> @llvm.masked.load.v8i32.p0(ptr, <5 x i1>, <8 x i32>)
define void @t3() {
; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with 8 elements (overload type 0 is <8 x i32>), but got <5 x i1>
; CHECK-NEXT: ptr @llvm.masked.load.v8i32.p0
  call <8 x i32> @llvm.masked.load.v8i32.p0(ptr poison, <5 x i1> poison, <8 x i32> poison)
  ret void
}

declare <7 x i32> @llvm.masked.load.v7i32.p0(ptr, <7 x i1>, <6 x i32>)
define void @t4() {
; CHECK: intrinsic argument 2 type (matching overload type 0) expected <7 x i32>, but got <6 x i32>
; CHECK-NEXT: ptr @llvm.masked.load.v7i32.p0
  call <7 x i32> @llvm.masked.load.v7i32.p0(ptr poison, <7 x i1> poison, <6 x i32> poison)
  ret void
}

declare void @llvm.masked.store.i32.p0(i32, ptr, i1)
define void @t5() {
; CHECK: intrinsic argument 0 type (overload type 0) expected any vector type, but got i32
; CHECK-NEXT: ptr @llvm.masked.store.i32.p0
  call void @llvm.masked.store.i32.p0(i32 0, ptr poison, i1 0)
  ret void
}

declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr, i1)
define void @t6() {
; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector (overload type 0 is <4 x i32>), but got i1
; CHECK-NEXT: ptr @llvm.masked.store.v4i32.p0
  call void @llvm.masked.store.v4i32.p0(<4 x i32> poison, ptr poison, i1 0)
  ret void
}

declare void @llvm.masked.store.v5i32.p0(<5 x i32>, ptr, <4 x i1>)
define void @t7() {
; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector with 5 elements (overload type 0 is <5 x i32>), but got <4 x i1>
; CHECK-NEXT: ptr @llvm.masked.store.v5i32.p0
  call void @llvm.masked.store.v5i32.p0(<5 x i32> poison, ptr poison, <4 x i1> poison)
  ret void
}
