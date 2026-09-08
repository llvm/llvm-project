; RUN: llc -O2 -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

; Herbception (throws): the NZCV.C discriminant of a throws call is
; materialized by a HERB_CSET glued to the call. A peephole folds
;   %disc = herb_cset; cbz/tbz %disc
; into a direct B.cc on live NZCV. That fold is only valid when nothing
; between the HERB_CSET and the branch modifies NZCV: any intervening
; flag-setting instruction (here the cmp of the block result) would make the
; rewritten branch test the wrong flags and corrupt the error check. The
; HERB_CSET/branch pair must be kept and the discriminant tested from the
; register.

declare dso_local { { ptr, i64 }, i1 } @callee(ptr, ptr, ptr) #1

define dso_local { { ptr, i64 }, i1 } @caller(ptr %out, ptr %scat, i64 %n) #0 {
entry:
  br label %loop

loop:                                              ; preds = %body, %entry
  %i = phi ptr [ %scat, %entry ], [ %next, %body ]
  %cont = icmp ne ptr %i, %scat
  br i1 %cont, label %body, label %exit

body:                                              ; preds = %loop
  %base = load ptr, ptr %i, align 8
  %lptr = getelementptr inbounds nuw i8, ptr %i, i64 8
  %len = load i64, ptr %lptr, align 8
  %last = getelementptr inbounds nuw i8, ptr %base, i64 %len
  %r = call { { ptr, i64 }, i1 } @callee(ptr %out, ptr noundef %base, ptr noundef %last) #1
  %u = extractvalue { { ptr, i64 }, i1 } %r, 0
  %d = extractvalue { { ptr, i64 }, i1 } %r, 1
  %v1 = extractvalue { ptr, i64 } %u, 1
  %w = icmp eq i64 %v1, 0
  %wz = zext i1 %w to i64
  %next = getelementptr inbounds nuw i8, ptr %i, i64 16
  br i1 %d, label %exit, label %loop

exit:                                              ; preds = %loop, %body
  %q = phi i64 [ undef, %loop ], [ %wz, %body ]
  %s0 = insertvalue { { ptr, i64 }, i1 } poison, ptr %i, 0, 0
  %s1 = insertvalue { { ptr, i64 }, i1 } %s0, i64 %q, 0, 1
  %s2 = insertvalue { { ptr, i64 }, i1 } %s1, i1 %cont, 1
  ret { { ptr, i64 }, i1 } %s2
}

attributes #0 = { minsize optsize }
attributes #1 = { throws }

; The cset must be kept and the tbz must test the cset register; the fold
; into a b.lo across the intervening cmp must not happen.
; CHECK-LABEL: {{^}}caller:
; CHECK: bl callee
; CHECK: cset [[DISC:w[0-9]+]], hs
; CHECK: cmp {{x[0-9]+}}, #0
; CHECK: tbz [[DISC]], #0
; CHECK-NOT: {{^}}	b.lo
