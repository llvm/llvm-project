; RUN: not opt -passes=verify -S < %s 2>&1 | FileCheck %s

declare i32 @readonly_call(ptr) memory(read)

define void @invalid(ptr %dst, ptr %src) {
; CHECK: invariant.load metadata is only for loads and readonly intrinsic calls
  call i32 @readonly_call(ptr %src), !invariant.load !0

; CHECK: invariant.load metadata is only for loads and readonly intrinsic calls
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 8, i1 false), !invariant.load !0

  ret void
}

!0 = !{}
