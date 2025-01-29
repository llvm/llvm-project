; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=special-globals --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck --implicit-check-not=define --check-prefix=CHECK-FINAL %s < %t.0

; Check that we don't hit "input module no longer interesting after
; counting chunks" The special-globals reduction was not checking
; shouldKeep before unconditionally erasing it.

; CHECK-INTERESTINGNESS: llvm.used
; CHECK-FINAL: llvm.used
; CHECK-FINAL: define void @kept_used
; CHECK-FINAL: define void @other
@llvm.used = appending global [2 x ptr] [ptr @kept_used, ptr @other ]

define void @kept_used() {
  ret void
}

define void @other() {
  ret void
}
