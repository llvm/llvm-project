; RUN: opt -S -instcombine %s -o - | FileCheck %s

; Regression test of PR31990. A memcpy of one byte, copying 0xff, was
; replaced with a single store of an i4 0xf.

@g = constant i8 -1

define void @foo() {
entry:
  %0 = alloca i8
  %1 = bitcast i8* %0 to i4*
  call void @bar(i4* %1)
  %2 = bitcast i4* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %2, i8* @g, i32 1, i1 false)
  call void @gaz(i8* %2)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1)
declare void @bar(i4*)
declare void @gaz(i8*)

; The mempcy should be simplified to a single store of an i8, not i4
; CHECK: store i8 -1
; CHECK-NOT: store i4 -1
