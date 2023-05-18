; RUN: opt < %s -disable-output -passes='print<cycles>' 2>&1 | FileCheck %s -check-prefix=CHECK

define void @empty() {
; CHECK-LABEL: CycleInfo for function: empty
; CHECK-NOT:       depth

  ret void
}

define void @simple() {
; CHECK-LABEL: CycleInfo for function: simple
; CHECK:           depth=1: entries(loop)
entry:
  br label %loop

loop:
  br i1 undef, label %loop, label %exit

exit:
  ret void
}

define void @two_latches() {
; CHECK-LABEL: CycleInfo for function: two_latches
; CHECK:           depth=1: entries(loop) loop_next
entry:
  br label %loop

loop:
  br i1 undef, label %loop, label %loop_next

loop_next:
  br i1 undef, label %exit, label %loop

exit:
  ret void
}

define void @nested_simple() {
; CHECK-LABEL: CycleInfo for function: nested_simple
; CHECK:           depth=1: entries(outer_header) outer_latch inner
; CHECK:               depth=2: entries(inner)
entry:
  br label %outer_header

outer_header:
  br label %inner

inner:
  br i1 undef, label %inner, label %outer_latch

outer_latch:
  br i1 undef, label %outer_header, label %exit

exit:
  ret void
}

define void @nested_outer_latch_in_inner_loop() {
; CHECK-LABEL: CycleInfo for function: nested_outer_latch_in_inner_loop
; CHECK:           depth=1: entries(outer_header) inner_header inner_latch
; CHECK:               depth=2: entries(inner_header) inner_latch
entry:
  br label %outer_header

outer_header:
  br label %inner_header

inner_header:
  br i1 undef, label %inner_latch, label %outer_header

inner_latch:
  br i1 undef, label %exit, label %inner_header

exit:
  ret void
}

define void @sibling_loops() {
; CHECK-LABEL: CycleInfo for function: sibling_loops
; CHECK-DAG:       depth=1: entries(left)
; CHECK-DAG:       depth=1: entries(right)
entry:
  br i1 undef, label %left, label %right

left:
  br i1 undef, label %left, label %exit

right:
  br i1 undef, label %right, label %exit

exit:
  ret void
}

define void @serial_loops() {
; CHECK-LABEL: CycleInfo for function: serial_loops
; CHECK-DAG:       depth=1: entries(second)
; CHECK-DAG:       depth=1: entries(first)
entry:
  br label %first

first:
  br i1 undef, label %first, label %second

second:
  br i1 undef, label %second, label %exit

exit:
  ret void
}

define void @nested_sibling_loops() {
; CHECK-LABEL: CycleInfo for function: nested_sibling_loops
; CHECK:           depth=1: entries(outer_header) left right
; CHECK-DAG:           depth=2: entries(right)
; CHECK-DAG:           depth=2: entries(left)
entry:
  br label %outer_header

outer_header:
  br i1 undef, label %left, label %right

left:
  switch i32 undef, label %exit [ i32 0, label %left
                                  i32 1, label %outer_header ]

right:
  switch i32 undef, label %outer_header [ i32 0, label %exit
                                          i32 1, label %right ]

exit:
  ret void
}

define void @deeper_nest() {
; CHECK-LABEL: CycleInfo for function: deeper_nest
; CHECK:           depth=1: entries(outer_header) outer_latch middle_header inner_header inner_latch
; CHECK:               depth=2: entries(middle_header) inner_header inner_latch
; CHECK:                   depth=3: entries(inner_header) inner_latch
entry:
  br label %outer_header

outer_header:
  br label %middle_header

middle_header:
  br label %inner_header

inner_header:
  br i1 undef, label %middle_header, label %inner_latch

inner_latch:
  br i1 undef, label %inner_header, label %outer_latch

outer_latch:
  br i1 undef, label %outer_header, label %exit

exit:
  ret void
}

define void @irreducible_basic() {
; CHECK-LABEL: CycleInfo for function: irreducible_basic
; CHECK:           depth=1: entries(right left)
entry:
  br i1 undef, label %left, label %right

left:
  br i1 undef, label %right, label %exit

right:
  br i1 undef, label %left, label %exit

exit:
  ret void
}

define void @irreducible_mess() {
; CHECK-LABEL: CycleInfo for function: irreducible_mess
; CHECK:           depth=1: entries(B A) D C
; CHECK:               depth=2: entries(D C A)
; CHECK:                   depth=3: entries(C A)
entry:
  br i1 undef, label %A, label %B

A:
  br i1 undef, label %C, label %D

B:
  br i1 undef, label %C, label %D

C:
  switch i32 undef, label %A [ i32 0, label %D
                               i32 1, label %exit ]

D:
  switch i32 undef, label %B [ i32 0, label %C
                               i32 1, label %exit ]

exit:
  ret void
}

define void @irreducible_into_simple_cycle() {
; CHECK-LABEL: CycleInfo for function: irreducible_into_simple_cycle
; CHECK:           depth=1: entries(F C A) E D B
entry:
  switch i32 undef, label %A [ i32 0, label %C
                               i32 1, label %F ]

A:
  br label %B

B:
  br label %C

C:
  br label %D

D:
  br i1 undef, label %E, label %exit

E:
  br label %F

F:
  br i1 undef, label %A, label %exit

exit:
  ret void
}

define void @irreducible_mountain_bug() {
; CHECK-LABEL: CycleInfo for function: irreducible_mountain_bug
; CHECK:           depth=1: entries(while.cond)
; CHECK:               depth=2: entries(cond.end61 cond.true49) while.body63 while.cond47
; CHECK:                   depth=3: entries(while.body63 cond.true49) while.cond47
entry:
  br i1 undef, label %if.end, label %if.then

if.end:
  br i1 undef, label %if.then7, label %if.else

if.then7:
  br label %if.end16

if.else:
  br label %if.end16

if.end16:
  br i1 undef, label %while.cond.preheader, label %if.then39

while.cond.preheader:
  br label %while.cond

while.cond:
  br i1 undef, label %cond.true49, label %lor.rhs

cond.true49:
  br i1 undef, label %if.then69, label %while.body63

while.body63:
  br i1 undef, label %exit, label %while.cond47

while.cond47:
  br i1 undef, label %cond.true49, label %cond.end61

cond.end61:
  br i1 undef, label %while.body63, label %while.cond

if.then69:
  br i1 undef, label %exit, label %while.cond

lor.rhs:
  br i1 undef, label %cond.end61, label %while.end76

while.end76:
  br label %exit

if.then39:
  br i1 undef, label %exit, label %if.end.i145

if.end.i145:
  br i1 undef, label %exit, label %if.end8.i149

if.end8.i149:
  br label %exit

if.then:
  br i1 undef, label %exit, label %if.end.i

if.end.i:
  br i1 undef, label %exit, label %if.end8.i

if.end8.i:
  br label %exit

exit:
  ret void
}
