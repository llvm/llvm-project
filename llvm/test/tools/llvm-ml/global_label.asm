; 'identifier ::' defines a label and gives it global (public) binding, which is
; equivalent to a plain label plus a PUBLIC directive.
; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.code

foo::
  ret

; CHECK: foo:
; CHECK: .globl foo
