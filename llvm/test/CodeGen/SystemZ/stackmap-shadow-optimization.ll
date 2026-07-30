; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check that the stackmap shadow optimization is only outputting a 2-byte
; nop here. 8-bytes are requested, but 6 are covered by the code for the call to
; bar.  However, the frame teardown and the return do not count towards the
; stackmap shadow as the call return counts as a branch target so must flush
; the shadow.
; Note that in order for a thread to not return in to the patched space
; the call must be at the end of the shadow, so the required nop must be
; before the call, not after.
define void @shadow_optimization_test() {
entry:
; CHECK-LABEL:  shadow_optimization_test:
; CHECK:        brasl %r14, bar@PLT
; CHECK-NEXT:   .Ltmp
; CHECK-NEXT:   bcr 0, %r0
; CHECK-NEXT:   brasl %r14, bar@PLT
; CHECK-NEXT:   brasl %r14, bar@PLT
  call void @bar()
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 8)
  call void @bar()
  call void @bar()
  ret void
}
declare void @bar()

declare void @llvm.experimental.stackmap(i64, i32, ...)
