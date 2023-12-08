; RUN: opt < %s -O3 -S | FileCheck %s

; Address space intrinsics were erroneously marked NoCapture, leading to bad
; optimizations (such as the store below being eliminated as dead code). This
; test makes sure we don't regress.

declare void @foo(ptr addrspace(1))

declare ptr addrspace(1) @llvm.nvvm.ptr.gen.to.global.p1.p0(ptr)

; CHECK: @bar
define void @bar() {
  %t1 = alloca i32
; CHECK: call ptr addrspace(1) @llvm.nvvm.ptr.gen.to.global.p1.p0(ptr nonnull %t1)
; CHECK-NEXT: store i32 10, ptr %t1
  %t2 = call ptr addrspace(1) @llvm.nvvm.ptr.gen.to.global.p1.p0(ptr %t1)
  store i32 10, ptr %t1
  call void @foo(ptr addrspace(1) %t2)
  ret void
}

