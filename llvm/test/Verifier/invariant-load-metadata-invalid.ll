; RUN: not opt -passes=verify -S < %s 2>&1 | FileCheck %s

declare i32 @writing_call(ptr)

define void @invalid(ptr %dst, ptr %src) {
; CHECK: invariant.load metadata is only for loads and readonly calls
  call i32 @writing_call(ptr %src), !invariant.load !0

; CHECK: invariant.load metadata is only for loads and readonly calls
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 8, i1 false), !invariant.load !0

  ret void
}

!0 = !{}
