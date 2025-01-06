; RUN: opt -disable-output --passes=sandbox-vectorizer \
; RUN:    -sbvec-passes='regions-from-metadata<print-instruction-count>' %s | FileCheck %s

define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1, !sandboxvec !0
  %t1 = add i8 %t0, %v1, !sandboxvec !1
  %t2 = add i8 %t1, %v1, !sandboxvec !1
  ret i8 %t2
}

!0 = distinct !{!"sandboxregion"}
!1 = distinct !{!"sandboxregion"}

; CHECK: InstructionCount: 1
; CHECK: InstructionCount: 2
