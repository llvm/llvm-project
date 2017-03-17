; RUN: llc -O2 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-grtev4-linux-gnu"

; Intended layout:
; The chain-based outlining produces the layout
; test1
; test2
; test3
; test4
; optional1
; optional2
; optional3
; optional4
; exit
; Tail duplication puts test n+1 at the end of optional n
; so optional1 includes a copy of test2 at the end, and branches
; to test3 (at the top) or falls through to optional 2.
; The CHECK statements check for the whole string of tests
; and then check that the correct test has been duplicated into the end of
; the optional blocks and that the optional blocks are in the correct order.
;CHECK-LABEL: straight_test:
; test1 may have been merged with entry
;CHECK: mr [[TAGREG:[0-9]+]], 3
;CHECK: andi. {{[0-9]+}}, [[TAGREG]], 1
;CHECK-NEXT: bc 12, 1, .[[OPT1LABEL:[_0-9A-Za-z]+]]
;CHECK-NEXT: # %test2
;CHECK-NEXT: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 30, 30
;CHECK-NEXT: bne 0, .[[OPT2LABEL:[_0-9A-Za-z]+]]
;CHECK-NEXT: .[[TEST3LABEL:[_0-9A-Za-z]+]]: # %test3
;CHECK-NEXT: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 29, 29
;CHECK-NEXT: bne 0, .[[OPT3LABEL:[_0-9A-Za-z]+]]
;CHECK-NEXT: .[[TEST4LABEL:[_0-9A-Za-z]+]]: # %test4
;CHECK-NEXT: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 28, 28
;CHECK-NEXT: bne 0, .[[OPT4LABEL:[_0-9A-Za-z]+]]
;CHECK-NEXT: .[[EXITLABEL:[_0-9A-Za-z]+]]: # %exit
;CHECK: blr
;CHECK-NEXT: .[[OPT1LABEL]]:
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 30, 30
;CHECK-NEXT: beq 0, .[[TEST3LABEL]]
;CHECK-NEXT: .[[OPT2LABEL]]:
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 29, 29
;CHECK-NEXT: beq 0, .[[TEST4LABEL]]
;CHECK-NEXT: .[[OPT3LABEL]]:
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 28, 28
;CHECK-NEXT: beq 0, .[[EXITLABEL]]
;CHECK-NEXT: .[[OPT4LABEL]]:
;CHECK: b .[[EXITLABEL]]

define void @straight_test(i32 %tag) {
entry:
  br label %test1
test1:
  %tagbit1 = and i32 %tag, 1
  %tagbit1eq0 = icmp eq i32 %tagbit1, 0
  br i1 %tagbit1eq0, label %test2, label %optional1, !prof !1
optional1:
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  br label %test2
test2:
  %tagbit2 = and i32 %tag, 2
  %tagbit2eq0 = icmp eq i32 %tagbit2, 0
  br i1 %tagbit2eq0, label %test3, label %optional2, !prof !1
optional2:
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  br label %test3
test3:
  %tagbit3 = and i32 %tag, 4
  %tagbit3eq0 = icmp eq i32 %tagbit3, 0
  br i1 %tagbit3eq0, label %test4, label %optional3, !prof !1
optional3:
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  br label %test4
test4:
  %tagbit4 = and i32 %tag, 8
  %tagbit4eq0 = icmp eq i32 %tagbit4, 0
  br i1 %tagbit4eq0, label %exit, label %optional4, !prof !1
optional4:
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  br label %exit
exit:
  ret void
}

; Intended layout:
; The chain-of-triangles based duplicating produces the layout
; test1
; test2
; test3
; test4
; optional1
; optional2
; optional3
; optional4
; exit
; even for 50/50 branches.
; Tail duplication puts test n+1 at the end of optional n
; so optional1 includes a copy of test2 at the end, and branches
; to test3 (at the top) or falls through to optional 2.
; The CHECK statements check for the whole string of tests
; and then check that the correct test has been duplicated into the end of
; the optional blocks and that the optional blocks are in the correct order.
;CHECK-LABEL: straight_test_50:
; test1 may have been merged with entry
;CHECK: mr [[TAGREG:[0-9]+]], 3
;CHECK: andi. {{[0-9]+}}, [[TAGREG]], 1
;CHECK-NEXT: bc 12, 1, .[[OPT1LABEL:[_0-9A-Za-z]+]]
;CHECK-NEXT: # %test2
;CHECK-NEXT: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 30, 30
;CHECK-NEXT: bne 0, .[[OPT2LABEL:[_0-9A-Za-z]+]]
;CHECK-NEXT: .[[TEST3LABEL:[_0-9A-Za-z]+]]: # %test3
;CHECK-NEXT: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 29, 29
;CHECK-NEXT: bne 0, .[[OPT3LABEL:[_0-9A-Za-z]+]]
;CHECK-NEXT: .[[EXITLABEL:[_0-9A-Za-z]+]]: # %exit
;CHECK: blr
;CHECK-NEXT: .[[OPT1LABEL]]:
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 30, 30
;CHECK-NEXT: beq 0, .[[TEST3LABEL]]
;CHECK-NEXT: .[[OPT2LABEL]]:
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 29, 29
;CHECK-NEXT: beq 0, .[[EXITLABEL]]
;CHECK-NEXT: .[[OPT3LABEL]]:
;CHECK: b .[[EXITLABEL]]

define void @straight_test_50(i32 %tag) {
entry:
  br label %test1
test1:
  %tagbit1 = and i32 %tag, 1
  %tagbit1eq0 = icmp eq i32 %tagbit1, 0
  br i1 %tagbit1eq0, label %test2, label %optional1, !prof !2
optional1:
  call void @a()
  br label %test2
test2:
  %tagbit2 = and i32 %tag, 2
  %tagbit2eq0 = icmp eq i32 %tagbit2, 0
  br i1 %tagbit2eq0, label %test3, label %optional2, !prof !2
optional2:
  call void @b()
  br label %test3
test3:
  %tagbit3 = and i32 %tag, 4
  %tagbit3eq0 = icmp eq i32 %tagbit3, 0
  br i1 %tagbit3eq0, label %exit, label %optional3, !prof !1
optional3:
  call void @c()
  br label %exit
exit:
  ret void
}

; Intended layout:
; The chain-based outlining produces the layout
; entry
; --- Begin loop ---
; for.latch
; for.check
; test1
; test2
; test3
; test4
; optional1
; optional2
; optional3
; optional4
; --- End loop ---
; exit
; The CHECK statements check for the whole string of tests and exit block,
; and then check that the correct test has been duplicated into the end of
; the optional blocks and that the optional blocks are in the correct order.
;CHECK-LABEL: loop_test:
;CHECK: add [[TAGPTRREG:[0-9]+]], 3, 4
;CHECK: .[[LATCHLABEL:[._0-9A-Za-z]+]]: # %for.latch
;CHECK: addi
;CHECK: .[[CHECKLABEL:[._0-9A-Za-z]+]]: # %for.check
;CHECK: lwz [[TAGREG:[0-9]+]], 0([[TAGPTRREG]])
;CHECK: # %test1
;CHECK: andi. {{[0-9]+}}, [[TAGREG]], 1
;CHECK-NEXT: bc 12, 1, .[[OPT1LABEL:[._0-9A-Za-z]+]]
;CHECK-NEXT: # %test2
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 30, 30
;CHECK-NEXT: bne 0, .[[OPT2LABEL:[._0-9A-Za-z]+]]
;CHECK-NEXT: .[[TEST3LABEL:[._0-9A-Za-z]+]]: # %test3
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 29, 29
;CHECK-NEXT: bne 0, .[[OPT3LABEL:[._0-9A-Za-z]+]]
;CHECK-NEXT: .[[TEST4LABEL:[._0-9A-Za-z]+]]: # %{{(test4|optional3)}}
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 28, 28
;CHECK-NEXT: beq 0, .[[LATCHLABEL]]
;CHECK-NEXT: b .[[OPT4LABEL:[._0-9A-Za-z]+]]
;CHECK: [[OPT1LABEL]]
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 30, 30
;CHECK-NEXT: beq 0, .[[TEST3LABEL]]
;CHECK-NEXT: .[[OPT2LABEL]]
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 29, 29
;CHECK-NEXT: beq 0, .[[TEST4LABEL]]
;CHECK-NEXT: .[[OPT3LABEL]]
;CHECK: rlwinm. {{[0-9]+}}, [[TAGREG]], 0, 28, 28
;CHECK-NEXT: beq 0, .[[LATCHLABEL]]
;CHECK: [[OPT4LABEL]]:
;CHECK: b .[[LATCHLABEL]]
define void @loop_test(i32* %tags, i32 %count) {
entry:
  br label %for.check
for.check:
  %count.loop = phi i32 [%count, %entry], [%count.sub, %for.latch]
  %done.count = icmp ugt i32 %count.loop, 0
  %tag_ptr = getelementptr inbounds i32, i32* %tags, i32 %count
  %tag = load i32, i32* %tag_ptr
  %done.tag = icmp eq i32 %tag, 0
  %done = and i1 %done.count, %done.tag
  br i1 %done, label %test1, label %exit, !prof !1
test1:
  %tagbit1 = and i32 %tag, 1
  %tagbit1eq0 = icmp eq i32 %tagbit1, 0
  br i1 %tagbit1eq0, label %test2, label %optional1, !prof !1
optional1:
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  br label %test2
test2:
  %tagbit2 = and i32 %tag, 2
  %tagbit2eq0 = icmp eq i32 %tagbit2, 0
  br i1 %tagbit2eq0, label %test3, label %optional2, !prof !1
optional2:
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  br label %test3
test3:
  %tagbit3 = and i32 %tag, 4
  %tagbit3eq0 = icmp eq i32 %tagbit3, 0
  br i1 %tagbit3eq0, label %test4, label %optional3, !prof !1
optional3:
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  br label %test4
test4:
  %tagbit4 = and i32 %tag, 8
  %tagbit4eq0 = icmp eq i32 %tagbit4, 0
  br i1 %tagbit4eq0, label %for.latch, label %optional4, !prof !1
optional4:
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  br label %for.latch
for.latch:
  %count.sub = sub i32 %count.loop, 1
  br label %for.check
exit:
  ret void
}

; The block then2 is not unavoidable, meaning it does not dominate the exit.
; But since it can be tail-duplicated, it should be placed as a fallthrough from
; test2 and copied. The purpose here is to make sure that the tail-duplication
; code is independent of the outlining code, which works by choosing the
; "unavoidable" blocks.
; CHECK-LABEL: avoidable_test:
; CHECK: # %entry
; CHECK: andi.
; CHECK: # %test2
; Make sure then2 falls through from test2
; CHECK-NOT: # %{{[-_a-zA-Z0-9]+}}
; CHECK: # %then2
; CHECK: rlwinm. {{[0-9]+}}, {{[0-9]+}}, 0, 29, 29
; CHECK: # %else1
; CHECK: bl a
; CHECK: bl a
; Make sure then2 was copied into else1
; CHECK: rlwinm. {{[0-9]+}}, {{[0-9]+}}, 0, 29, 29
; CHECK: # %end1
; CHECK: bl d
; CHECK: # %else2
; CHECK: bl c
; CHECK: # %end2
define void @avoidable_test(i32 %tag) {
entry:
  br label %test1
test1:
  %tagbit1 = and i32 %tag, 1
  %tagbit1eq0 = icmp eq i32 %tagbit1, 0
  br i1 %tagbit1eq0, label %test2, label %else1, !prof !1 ; %test2 more likely
else1:
  call void @a()
  call void @a()
  br label %then2
test2:
  %tagbit2 = and i32 %tag, 2
  %tagbit2eq0 = icmp eq i32 %tagbit2, 0
  br i1 %tagbit2eq0, label %then2, label %else2, !prof !1 ; %then2 more likely
then2:
  %tagbit3 = and i32 %tag, 4
  %tagbit3eq0 = icmp eq i32 %tagbit3, 0
  br i1 %tagbit3eq0, label %end2, label %end1, !prof !1 ; %end2 more likely
else2:
  call void @c()
  br label %end2
end2:
  ret void
end1:
  call void @d()
  ret void
}

; CHECK-LABEL: trellis_test
; The number in the block labels is the expected block frequency given the
; probabilities annotated. There is a conflict in the b;c->d;e trellis that
; should be resolved as c->e;b->d.
; The d;e->f;g trellis should be resolved as e->g;d->f.
; The f;g->h;i trellis should be resolved as f->i;g->h.
; The h;i->j;ret trellis contains a triangle edge, and should be resolved as
; h->j->ret
; CHECK: # %entry
; CHECK: # %c10
; CHECK: # %e9
; CHECK: # %g10
; CHECK: # %h10
; CHECK: # %j8
; CHECK: # %ret
; CHECK: # %b6
; CHECK: # %d7
; CHECK: # %f6
; CHECK: # %i6
define void @trellis_test(i32 %tag) {
entry:
  br label %a16
a16:
  call void @a()
  call void @a()
  %tagbits.a = and i32 %tag, 3
  %tagbits.a.eq0 = icmp eq i32 %tagbits.a, 0
  br i1 %tagbits.a.eq0, label %c10, label %b6, !prof !1 ; 10 to 6
c10:
  call void @c()
  call void @c()
  %tagbits.c = and i32 %tag, 12
  %tagbits.c.eq0 = icmp eq i32 %tagbits.c, 0
  ; Both of these edges should be hotter than the other incoming edge
  ; for e9 or d7
  br i1 %tagbits.c.eq0, label %e9, label %d7, !prof !3 ; 6 to 4
e9:
  call void @e()
  call void @e()
  %tagbits.e = and i32 %tag, 48
  %tagbits.e.eq0 = icmp eq i32 %tagbits.e, 0
  br i1 %tagbits.e.eq0, label %g10, label %f6, !prof !4 ; 7 to 2
g10:
  call void @g()
  call void @g()
  %tagbits.g = and i32 %tag, 192
  %tagbits.g.eq0 = icmp eq i32 %tagbits.g, 0
  br i1 %tagbits.g.eq0, label %i6, label %h10, !prof !5 ; 2 to 8
i6:
  call void @i()
  call void @i()
  %tagbits.i = and i32 %tag, 768
  %tagbits.i.eq0 = icmp eq i32 %tagbits.i, 0
  br i1 %tagbits.i.eq0, label %ret, label %j8, !prof !2 ; balanced (3 to 3)
b6:
  call void @b()
  call void @b()
  %tagbits.b = and i32 %tag, 12
  %tagbits.b.eq1 = icmp eq i32 %tagbits.b, 8
  br i1 %tagbits.b.eq1, label %e9, label %d7, !prof !2 ; balanced (3 to 3)
d7:
  call void @d()
  call void @d()
  %tagbits.d = and i32 %tag, 48
  %tagbits.d.eq1 = icmp eq i32 %tagbits.d, 32
  br i1 %tagbits.d.eq1, label %g10, label %f6, !prof !6 ; 3 to 4
f6:
  call void @f()
  call void @f()
  %tagbits.f = and i32 %tag, 192
  %tagbits.f.eq1 = icmp eq i32 %tagbits.f, 128
  br i1 %tagbits.f.eq1, label %i6, label %h10, !prof !7 ; 4 to 2
h10:
  call void @h()
  call void @h()
  %tagbits.h = and i32 %tag, 768
  %tagbits.h.eq1 = icmp eq i32 %tagbits.h, 512
  br i1 %tagbits.h.eq1, label %ret, label %j8, !prof !2 ; balanced (5 to 5)
j8:
  call void @j()
  call void @j()
  br label %ret
ret:
  ret void
}

; Verify that we still consider tail-duplication opportunities if we find a
; triangle trellis. Here D->F->G is the triangle, and D;E are both predecessors
; of both F and G. The basic trellis algorithm picks the F->G edge, but after
; checking, it's profitable to duplicate G into F. The weights here are not
; really important. They are there to help make the test stable.
; CHECK-LABEL: trellis_then_dup_test
; CHECK: # %entry
; CHECK: # %b
; CHECK: # %d
; CHECK: # %g
; CHECK: # %ret1
; CHECK: # %c
; CHECK: # %e
; CHECK: # %f
; CHECK: # %ret2
; CHECK: # %ret
define void @trellis_then_dup_test(i32 %tag) {
entry:
  br label %a
a:
  call void @a()
  call void @a()
  %tagbits.a = and i32 %tag, 3
  %tagbits.a.eq0 = icmp eq i32 %tagbits.a, 0
  br i1 %tagbits.a.eq0, label %b, label %c, !prof !1 ; 5 to 3
b:
  call void @b()
  call void @b()
  %tagbits.b = and i32 %tag, 12
  %tagbits.b.eq1 = icmp eq i32 %tagbits.b, 8
  br i1 %tagbits.b.eq1, label %d, label %e, !prof !1 ; 5 to 3
d:
  call void @d()
  call void @d()
  %tagbits.d = and i32 %tag, 48
  %tagbits.d.eq1 = icmp eq i32 %tagbits.d, 32
  br i1 %tagbits.d.eq1, label %g, label %f, !prof !1 ; 5 to 3
f:
  call void @f()
  call void @f()
  br label %g
g:
  %tagbits.g = and i32 %tag, 192
  %tagbits.g.eq0 = icmp eq i32 %tagbits.g, 0
  br i1 %tagbits.g.eq0, label %ret1, label %ret2, !prof !2 ; balanced
c:
  call void @c()
  call void @c()
  %tagbits.c = and i32 %tag, 12
  %tagbits.c.eq0 = icmp eq i32 %tagbits.c, 0
  br i1 %tagbits.c.eq0, label %d, label %e, !prof !1 ; 5 to 3
e:
  call void @e()
  call void @e()
  %tagbits.e = and i32 %tag, 48
  %tagbits.e.eq0 = icmp eq i32 %tagbits.e, 0
  br i1 %tagbits.e.eq0, label %g, label %f, !prof !1 ; 5 to 3
ret1:
  call void @a()
  br label %ret
ret2:
  call void @b()
  br label %ret
ret:
  ret void
}

declare void @a()
declare void @b()
declare void @c()
declare void @d()
declare void @e()
declare void @f()
declare void @g()
declare void @h()
declare void @i()
declare void @j()

!1 = !{!"branch_weights", i32 5, i32 3}
!2 = !{!"branch_weights", i32 50, i32 50}
!3 = !{!"branch_weights", i32 6, i32 4}
!4 = !{!"branch_weights", i32 7, i32 2}
!5 = !{!"branch_weights", i32 2, i32 8}
!6 = !{!"branch_weights", i32 3, i32 4}
!7 = !{!"branch_weights", i32 4, i32 2}
