; RUN: llc -O2 -o - %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-grtev4-linux-gnu"

; Intended layout:
; The code for tail-duplication during layout will produce the layout:
; test1
; test2
; body1 (with copy of test2)
; body2
; exit

;CHECK-LABEL: tail_dup_break_cfg:
;CHECK: mr [[TAGREG:[0-9]+]], 3
;CHECK: andi. {{[0-9]+}}, [[TAGREG]], 1
;CHECK-NEXT: bc 12, 1, [[BODY1LABEL:[._0-9A-Za-z]+]]
;CHECK-NEXT: # %test2
;CHECK-NEXT: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 30, 30
;CHECK-NEXT: bne 0, [[BODY2LABEL:[._0-9A-Za-z]+]]
;CHECK: [[EXITLABEL:[._0-9A-Za-z]+]]: # %exit
;CHECK: blr
;CHECK-NEXT: [[BODY1LABEL]]
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 30, 30
;CHECK-NEXT: beq 0, [[EXITLABEL]]
;CHECK-NEXT: [[BODY2LABEL:[._0-9A-Za-z]+]]:
;CHECK: b [[EXITLABEL]]
define void @tail_dup_break_cfg(i32 %tag) {
entry:
  br label %test1
test1:
  %tagbit1 = and i32 %tag, 1
  %tagbit1eq0 = icmp eq i32 %tagbit1, 0
  br i1 %tagbit1eq0, label %test2, label %body1, !prof !1 ; %test2 more likely
body1:
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  br label %test2
test2:
  %tagbit2 = and i32 %tag, 2
  %tagbit2eq0 = icmp eq i32 %tagbit2, 0
  br i1 %tagbit2eq0, label %exit, label %body2, !prof !1 ; %exit more likely
body2:
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  br label %exit
exit:
  ret void
}

; The branch weights here hint that we shouldn't tail duplicate in this case.
;CHECK-LABEL: tail_dup_dont_break_cfg:
;CHECK: mr [[TAGREG:[0-9]+]], 3
;CHECK: andi. {{[0-9]+}}, [[TAGREG]], 1
;CHECK-NEXT: bc 4, 1, [[TEST2LABEL:[._0-9A-Za-z]+]]
;CHECK-NEXT: # %body1
;CHECK: [[TEST2LABEL]]: # %test2
;CHECK-NEXT: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 30, 30
;CHECK-NEXT: beq 0, [[EXITLABEL:[._0-9A-Za-z]+]]
;CHECK-NEXT: # %body2
;CHECK: [[EXITLABEL:[._0-9A-Za-z]+]]: # %exit
;CHECK: blr
define void @tail_dup_dont_break_cfg(i32 %tag) {
entry:
  br label %test1
test1:
  %tagbit1 = and i32 %tag, 1
  %tagbit1eq0 = icmp eq i32 %tagbit1, 0
  br i1 %tagbit1eq0, label %test2, label %body1, !prof !1 ; %test2 more likely
body1:
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  br label %test2
test2:
  %tagbit2 = and i32 %tag, 2
  %tagbit2eq0 = icmp ne i32 %tagbit2, 0
  br i1 %tagbit2eq0, label %body2, label %exit, !prof !3 ; %body2 more likely
body2:
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  br label %exit
exit:
  ret void
}
declare void @a()
declare void @b()
declare void @c()
declare void @d()

; This function arranges for the successors of %succ to have already been laid
; out. When we consider whether to lay out succ after bb and to tail-duplicate
; it, v and ret have already been placed, so we tail-duplicate as it removes a
; branch and strictly increases fallthrough
; CHECK-LABEL: tail_dup_no_succ
; CHECK: # %entry
; CHECK: # %v
; CHECK: # %ret
; CHECK: # %bb
; CHECK: # %succ
; CHECK: # %c
; CHECK: bl c
; CHECK: rlwinm. {{[0-9]+}}, {{[0-9]+}}, 0, 29, 29
; CHECK: beq
; CHECK: b
define void @tail_dup_no_succ(i32 %tag) {
entry:
  %tagbit1 = and i32 %tag, 1
  %tagbit1eq0 = icmp eq i32 %tagbit1, 0
  br i1 %tagbit1eq0, label %v, label %bb, !prof !2 ; %v very much more likely
bb:
  %tagbit2 = and i32 %tag, 2
  %tagbit2eq0 = icmp eq i32 %tagbit2, 0
  br i1 %tagbit2eq0, label %succ, label %c, !prof !3 ; %succ more likely
c:
  call void @c()
  call void @c()
  br label %succ
succ:
  %tagbit3 = and i32 %tag, 4
  %tagbit3eq0 = icmp eq i32 %tagbit3, 0
  br i1 %tagbit3eq0, label %ret, label %v, !prof !1 ; %u more likely
v:
  call void @d()
  call void @d()
  br label %ret
ret:
  ret void
}


!1 = !{!"branch_weights", i32 5, i32 3}
!2 = !{!"branch_weights", i32 95, i32 5}
!3 = !{!"branch_weights", i32 8, i32 3}
