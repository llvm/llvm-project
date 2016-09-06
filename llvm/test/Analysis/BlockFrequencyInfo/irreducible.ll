; RUN: opt < %s -analyze -block-freq | FileCheck %s
; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

; A loop with multiple exits isn't irreducible.  It should be handled
; correctly.
;
; CHECK-LABEL: Printing analysis {{.*}} for function 'multiexit':
; CHECK-NEXT: block-frequency-info: multiexit
define void @multiexit(i1 %x) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %loop.1

; CHECK-NEXT: loop.1: float = 2.0,
loop.1:
  br i1 %x, label %exit.1, label %loop.2, !prof !0

; CHECK-NEXT: loop.2: float = 1.75,
loop.2:
  br i1 %x, label %exit.2, label %loop.1, !prof !1

; CHECK-NEXT: exit.1: float = 0.25,
exit.1:
  br label %return

; CHECK-NEXT: exit.2: float = 0.75,
exit.2:
  br label %return

; CHECK-NEXT: return: float = 1.0, int = [[ENTRY]]
return:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 7}
!1 = !{!"branch_weights", i32 3, i32 4}

; Irreducible control flow
; ========================
;
; LoopInfo defines a loop as a non-trivial SCC dominated by a single block,
; called the header.  A given loop, L, can have sub-loops, which are loops
; within the subgraph of L that excludes the header.
;
; In addition to loops, -block-freq has limited support for irreducible SCCs,
; which are SCCs with multiple entry blocks.  Irreducible SCCs are discovered
; on they fly, and modelled as loops with multiple headers.
;
; The headers of irreducible sub-SCCs consist of its entry blocks and all nodes
; that are targets of a backedge within it (excluding backedges within true
; sub-loops).
;
; -block-freq is currently designed to act like a block is inserted that
; intercepts all the edges to the headers.  All backedges and entries point to
; this block.  Its successors are the headers, which split the frequency
; evenly.
;
; There are a number of testcases below.  Only the first two have detailed
; explanations.
;
; Testcase #1
; ===========
;
; In this case c1 and c2 should have frequencies of 15/7 and 13/7,
; respectively.  To calculate this, consider assigning 1.0 to entry, and
; distributing frequency iteratively (to infinity).  At the first iteration,
; entry gives 3/4 to c1 and 1/4 to c2.  At every step after, c1 and c2 give 3/4
; of what they have to each other.  Somehow, all of it comes out to exit.
;
;       c1 = 3/4 + 1/4*3/4 + 3/4*3^2/4^2 + 1/4*3^3/4^3 + 3/4*3^3/4^3 + ...
;       c2 = 1/4 + 3/4*3/4 + 1/4*3^2/4^2 + 3/4*3^3/4^3 + 1/4*3^3/4^3 + ...
;
; Simplify by splitting up the odd and even terms of the series and taking out
; factors so that the infite series matches:
;
;       c1 =  3/4 *(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;          +  3/16*(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;       c2 =  1/4 *(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;          +  9/16*(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;
;       c1 = 15/16*(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;       c2 = 13/16*(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;
; Since this geometric series sums to 16/7:
;
;       c1 = 15/7
;       c2 = 13/7
;
; If we treat c1 and c2 as members of the same loop, the exit frequency of the
; loop as a whole is 1/4, so the loop scale should be 4.  Summing c1 and c2
; gives 28/7, or 4.0, which is nice confirmation of the math above.
;
; -block-freq currently treats the two nodes as equals.
define void @multientry(i1 %x) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'multientry':
; CHECK-NEXT: block-frequency-info: multientry
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  br i1 %x, label %c1, label %c2, !prof !2

c1:
; CHECK-NEXT: c1: float = 2.0,
; The "correct" answer is: float = 2.142857{{[0-9]*}},
  br i1 %x, label %c2, label %exit, !prof !2

c2:
; CHECK-NEXT: c2: float = 2.0,
; The "correct" answer is: float = 1.857142{{[0-9]*}},
  br i1 %x, label %c1, label %exit, !prof !2

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}

!2 = !{!"branch_weights", i32 3, i32 1}

; Testcase #2
; ===========
;
; In this case c1 and c2 should be treated as equals in a single loop.  The
; exit frequency is 1/3, so the scaling factor for the loop should be 3.0.  The
; loop is entered 2/3 of the time, and c1 and c2 split the total loop frequency
; evenly (1/2), so they should each have frequencies of 1.0 (3.0*2/3*1/2).
; Another way of computing this result is by assigning 1.0 to entry and showing
; that c1 and c2 should accumulate frequencies of:
;
;       1/3   +   2/9   +   4/27  +   8/81  + ...
;     2^0/3^1 + 2^1/3^2 + 2^2/3^3 + 2^3/3^4 + ...
;
; At the first step, c1 and c2 each get 1/3 of the entry.  At each subsequent
; step, c1 and c2 each get 1/3 of what's left in c1 and c2 combined.  This
; infinite series sums to 1.
define void @crossloops(i2 %x) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'crossloops':
; CHECK-NEXT: block-frequency-info: crossloops
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  switch i2 %x, label %exit [ i2 1, label %c1
                              i2 2, label %c2 ], !prof !3

c1:
; CHECK-NEXT: c1: float = 1.0,
  switch i2 %x, label %exit [ i2 1, label %c1
                              i2 2, label %c2 ], !prof !3

c2:
; CHECK-NEXT: c2: float = 1.0,
  switch i2 %x, label %exit [ i2 1, label %c1
                              i2 2, label %c2 ], !prof !3

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}

!3 = !{!"branch_weights", i32 2, i32 2, i32 2}

; A true loop with irreducible control flow inside.
define void @loop_around_irreducible(i1 %x) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'loop_around_irreducible':
; CHECK-NEXT: block-frequency-info: loop_around_irreducible
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  br label %loop

loop:
; CHECK-NEXT: loop: float = 4.0, int = [[HEAD:[0-9]+]]
  br i1 %x, label %left, label %right, !prof !4

left:
; CHECK-NEXT: left: float = 8.0,
  br i1 %x, label %right, label %loop.end, !prof !5

right:
; CHECK-NEXT: right: float = 8.0,
  br i1 %x, label %left, label %loop.end, !prof !5

loop.end:
; CHECK-NEXT: loop.end: float = 4.0, int = [[HEAD]]
  br i1 %x, label %loop, label %exit, !prof !5

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}
!4 = !{!"branch_weights", i32 1, i32 1}
!5 = !{!"branch_weights", i32 3, i32 1}

; Two unrelated irreducible SCCs.
define void @two_sccs(i1 %x) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'two_sccs':
; CHECK-NEXT: block-frequency-info: two_sccs
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  br i1 %x, label %a, label %b, !prof !6

a:
; CHECK-NEXT: a: float = 0.75,
  br i1 %x, label %a.left, label %a.right, !prof !7

a.left:
; CHECK-NEXT: a.left: float = 1.5,
  br i1 %x, label %a.right, label %exit, !prof !6

a.right:
; CHECK-NEXT: a.right: float = 1.5,
  br i1 %x, label %a.left, label %exit, !prof !6

b:
; CHECK-NEXT: b: float = 0.25,
  br i1 %x, label %b.left, label %b.right, !prof !7

b.left:
; CHECK-NEXT: b.left: float = 0.625,
  br i1 %x, label %b.right, label %exit, !prof !8

b.right:
; CHECK-NEXT: b.right: float = 0.625,
  br i1 %x, label %b.left, label %exit, !prof !8

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}
!6 = !{!"branch_weights", i32 3, i32 1}
!7 = !{!"branch_weights", i32 1, i32 1}
!8 = !{!"branch_weights", i32 4, i32 1}

; A true loop inside irreducible control flow.
define void @loop_inside_irreducible(i1 %x) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'loop_inside_irreducible':
; CHECK-NEXT: block-frequency-info: loop_inside_irreducible
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  br i1 %x, label %left, label %right, !prof !9

left:
; CHECK-NEXT: left: float = 2.0,
  br i1 %x, label %right, label %exit, !prof !10

right:
; CHECK-NEXT: right: float = 2.0, int = [[RIGHT:[0-9]+]]
  br label %loop

loop:
; CHECK-NEXT: loop: float = 6.0,
  br i1 %x, label %loop, label %right.end, !prof !11

right.end:
; CHECK-NEXT: right.end: float = 2.0, int = [[RIGHT]]
  br i1 %x, label %left, label %exit, !prof !10

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}
!9 = !{!"branch_weights", i32 1, i32 1}
!10 = !{!"branch_weights", i32 3, i32 1}
!11 = !{!"branch_weights", i32 2, i32 1}

; Irreducible control flow in a branch that's in a true loop.
define void @loop_around_branch_with_irreducible(i1 %x) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'loop_around_branch_with_irreducible':
; CHECK-NEXT: block-frequency-info: loop_around_branch_with_irreducible
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  br label %loop

loop:
; CHECK-NEXT: loop: float = 2.0, int = [[LOOP:[0-9]+]]
  br i1 %x, label %normal, label %irreducible.entry, !prof !12

normal:
; CHECK-NEXT: normal: float = 1.5,
  br label %loop.end

irreducible.entry:
; CHECK-NEXT: irreducible.entry: float = 0.5, int = [[IRREDUCIBLE:[0-9]+]]
  br i1 %x, label %left, label %right, !prof !13

left:
; CHECK-NEXT: left: float = 1.0,
  br i1 %x, label %right, label %irreducible.exit, !prof !12

right:
; CHECK-NEXT: right: float = 1.0,
  br i1 %x, label %left, label %irreducible.exit, !prof !12

irreducible.exit:
; CHECK-NEXT: irreducible.exit: float = 0.5, int = [[IRREDUCIBLE]]
  br label %loop.end

loop.end:
; CHECK-NEXT: loop.end: float = 2.0, int = [[LOOP]]
  br i1 %x, label %loop, label %exit, !prof !13

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}
!12 = !{!"branch_weights", i32 3, i32 1}
!13 = !{!"branch_weights", i32 1, i32 1}

; Irreducible control flow between two true loops.
define void @loop_around_branch_with_irreducible_around_loop(i1 %x) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'loop_around_branch_with_irreducible_around_loop':
; CHECK-NEXT: block-frequency-info: loop_around_branch_with_irreducible_around_loop
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  br label %loop

loop:
; CHECK-NEXT: loop: float = 3.0, int = [[LOOP:[0-9]+]]
  br i1 %x, label %normal, label %irreducible, !prof !14

normal:
; CHECK-NEXT: normal: float = 2.0,
  br label %loop.end

irreducible:
; CHECK-NEXT: irreducible: float = 1.0,
  br i1 %x, label %left, label %right, !prof !15

left:
; CHECK-NEXT: left: float = 2.0,
  br i1 %x, label %right, label %loop.end, !prof !16

right:
; CHECK-NEXT: right: float = 2.0, int = [[RIGHT:[0-9]+]]
  br label %right.loop

right.loop:
; CHECK-NEXT: right.loop: float = 10.0,
  br i1 %x, label %right.loop, label %right.end, !prof !17

right.end:
; CHECK-NEXT: right.end: float = 2.0, int = [[RIGHT]]
  br i1 %x, label %left, label %loop.end, !prof !16

loop.end:
; CHECK-NEXT: loop.end: float = 3.0, int = [[LOOP]]
  br i1 %x, label %loop, label %exit, !prof !14

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}
!14 = !{!"branch_weights", i32 2, i32 1}
!15 = !{!"branch_weights", i32 1, i32 1}
!16 = !{!"branch_weights", i32 3, i32 1}
!17 = !{!"branch_weights", i32 4, i32 1}

; An irreducible SCC with a non-header.
define void @nonheader(i1 %x) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'nonheader':
; CHECK-NEXT: block-frequency-info: nonheader
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  br i1 %x, label %left, label %right, !prof !18

left:
; CHECK-NEXT: left: float = 1.0,
  br i1 %x, label %bottom, label %exit, !prof !19

right:
; CHECK-NEXT: right: float = 1.0,
  br i1 %x, label %bottom, label %exit, !prof !20

bottom:
; CHECK-NEXT: bottom: float = 1.0,
  br i1 %x, label %left, label %right, !prof !18

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}
!18 = !{!"branch_weights", i32 1, i32 1}
!19 = !{!"branch_weights", i32 1, i32 3}
!20 = !{!"branch_weights", i32 3, i32 1}

; An irreducible SCC with an irreducible sub-SCC.  In the current version of
; -block-freq, this means an extra header.
;
; This testcases uses non-trivial branch weights.  The CHECK statements here
; will start to fail if we change -block-freq to be more accurate.  Currently,
; loop headers are affected by the weight of their corresponding back edges.
define void @nonentry_header(i1 %x, i2 %y) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'nonentry_header':
; CHECK-NEXT: block-frequency-info: nonentry_header
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  br i1 %x, label %left, label %right, !prof !21

left:
; CHECK-NEXT: left: float = 0.14
  br i1 %x, label %top, label %bottom, !prof !22

right:
; CHECK-NEXT: right: float = 0.42
  br i1 %x, label %top, label %bottom, !prof !22

top:
; CHECK-NEXT: top: float = 8.43
  switch i2 %y, label %exit [ i2 0, label %left
                              i2 1, label %right
                              i2 2, label %bottom ], !prof !23

bottom:
; CHECK-NEXT: bottom: float = 4.5,
  br label %top

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}
!21 = !{!"branch_weights", i32 2, i32 1}
!22 = !{!"branch_weights", i32 1, i32 1}
!23 = !{!"branch_weights", i32 8, i32 1, i32 3, i32 12}
