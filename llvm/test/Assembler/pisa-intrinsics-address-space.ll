; REQUIRES: pisa-registered-target
; RUN: llvm-as %s -o /dev/null

; Verify that the PISA address-space query intrinsics accept a generic
; (address space 0) pointer and return an i32 predicate.

define void @test(ptr %p) {
  %g = call i32 @llvm.pisa.isaddr.global(ptr %p)
  %s = call i32 @llvm.pisa.isaddr.shared(ptr %p)
  %pr = call i32 @llvm.pisa.isaddr.private(ptr %p)
  ret void
}
