; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s
; RUN: opt -passes=ejit-aot-module,lower-expect -S %s -o /dev/null
; RUN: opt -passes='default<O2>' -S %s -o /dev/null

; PHI-incoming-block regression coverage (companion to
; ejit-wrapper-gen-phi-entry.ll). The original crash was found via
; short-circuit && inside __builtin_expect, but the same dangling-PHI hazard
; applies to ANY loop whose header PHI lists the entry block as an incoming
; predecessor — the single most common real-world shape. PASS3 erases the
; original entry block, so the loop header's [ init, %entry ] incoming edge
; must be rewritten to point at jit_fallback (where the entry body is spliced).

define i32 @loop_entry(i32 %n) !ejit.metadata !0 {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %acc.next = add i32 %acc, %i
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %done

done:
  ret i32 %acc
}

!0 = distinct !{!{!"ejit_entry"}}

; CHECK-LABEL: define i32 @loop_entry(i32 %n)
; CHECK: jit_entry:
; CHECK: call ptr @ejit_compile_or_get(i64 {{.*}}, ptr null)
; The loop header PHIs' entry-incoming edge must now name jit_fallback, the
; block the original entry body was spliced into — never the erased entry.
; CHECK: loop:
; CHECK: %i = phi i32 [ 0, %jit_fallback ], [ %i.next, %loop ]
; CHECK: %acc = phi i32 [ 0, %jit_fallback ], [ %acc.next, %loop ]
; CHECK-NOT: %entry
; CHECK: jit_fallback:
; CHECK: jit_dispatch:
; CHECK: call i32 {{.*}}(i32 %n)
