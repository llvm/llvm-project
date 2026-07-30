; RUN: opt -S -passes=instcombine %s -o - | FileCheck %s

; Regression test of PR31990. A memcpy of one byte, copying 0xff, was
; replaced with a single store of an i4 0xf.

@g = constant i8 -1

define void @foo() {
entry:
  %0 = alloca i8
  call void @bar(ptr %0)
  call void @llvm.memcpy.p0.p0.i32(ptr %0, ptr @g, i32 1, i1 false)
  call void @gaz(ptr %0)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1)
declare void @bar(ptr)
declare void @gaz(ptr)

; The mempcy should be simplified to a single store of an i8, not i4
; CHECK: store i8 -1
; CHECK-NOT: store i4 -1
