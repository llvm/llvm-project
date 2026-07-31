; The PISA address-space query intrinsics take a generic (addrspace 0) pointer
; and return an i32 predicate. Verify they round-trip through the IR verifier
; and that the pointer operand type is preserved.

; RUN: opt -S < %s | FileCheck %s

define void @test(ptr %p) {
; CHECK-LABEL: define void @test(ptr %p)
  %g = call i32 @llvm.pisa.isaddr.global(ptr %p)
; CHECK: call i32 @llvm.pisa.isaddr.global(ptr %p)
  %s = call i32 @llvm.pisa.isaddr.shared(ptr %p)
; CHECK: call i32 @llvm.pisa.isaddr.shared(ptr %p)
  %pr = call i32 @llvm.pisa.isaddr.private(ptr %p)
; CHECK: call i32 @llvm.pisa.isaddr.private(ptr %p)
  ret void
}

; CHECK: declare i32 @llvm.pisa.isaddr.global(ptr)
; CHECK: declare i32 @llvm.pisa.isaddr.private(ptr)
; CHECK: declare i32 @llvm.pisa.isaddr.shared(ptr)
