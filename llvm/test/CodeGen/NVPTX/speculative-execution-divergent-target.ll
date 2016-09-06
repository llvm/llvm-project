; Checks that speculative-execution only runs on divergent targets, if you pass
; -spec-exec-only-if-divergent-target.

; RUN: opt < %s -S -mtriple=nvptx-nvidia-cuda -speculative-execution | \
; RUN:   FileCheck --check-prefix=ON %s
; RUN: opt < %s -S -mtriple=nvptx-nvidia-cuda -speculative-execution \
; RUN:   -spec-exec-only-if-divergent-target | \
; RUN:   FileCheck --check-prefix=ON %s
; RUN: opt < %s -S -speculative-execution -spec-exec-only-if-divergent-target | \
; RUN:   FileCheck --check-prefix=OFF %s

; Hoist in if-then pattern.
define void @f() {
; ON: %x = add i32 2, 3
; ON: br i1 true
; OFF: br i1 true
; OFF: %x = add i32 2, 3
  br i1 true, label %a, label %b
a:
  %x = add i32 2, 3
  br label %b
b:
  ret void
}
