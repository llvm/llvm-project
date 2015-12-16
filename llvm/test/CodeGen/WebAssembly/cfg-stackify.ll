; RUN: llc < %s -asm-verbose=false -disable-block-placement -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs | FileCheck -check-prefix=OPT %s

; Test the CFG stackifier pass.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @something()

; Test that loops are made contiguous, even in the presence of split backedges.

; CHECK-LABEL: test0:
; CHECK: loop
; CHECK-NOT: br
; CHECK: i32.add
; CHECK-NEXT: i32.ge_s
; CHECK-NEXT: br_if
; CHECK-NOT: br
; CHECK: call
; CHECK: br BB0_1{{$}}
; CHECK: return{{$}}
; OPT-LABEL: test0:
; OPT: loop
; OPT-NOT: br
; OPT: i32.add
; OPT-NEXT: i32.ge_s
; OPT-NEXT: br_if
; OPT-NOT: br
; OPT: call
; OPT: br BB0_1{{$}}
; OPT: return{{$}}
define void @test0(i32 %n) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %back ]
  %i.next = add i32 %i, 1

  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %back, label %exit

exit:
  ret void

back:
  call void @something()
  br label %header
}

; Same as test0, but the branch condition is reversed.

; CHECK-LABEL: test1:
; CHECK: loop
; CHECK-NOT: br
; CHECK: i32.add
; CHECK-NEXT: i32.ge_s
; CHECK-NEXT: br_if
; CHECK-NOT: br
; CHECK: call
; CHECK: br BB1_1{{$}}
; CHECK: return{{$}}
; OPT-LABEL: test1:
; OPT: loop
; OPT-NOT: br
; OPT: i32.add
; OPT-NEXT: i32.ge_s
; OPT-NEXT: br_if
; OPT-NOT: br
; OPT: call
; OPT: br BB1_1{{$}}
; OPT: return{{$}}
define void @test1(i32 %n) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %back ]
  %i.next = add i32 %i, 1

  %c = icmp sge i32 %i.next, %n
  br i1 %c, label %exit, label %back

exit:
  ret void

back:
  call void @something()
  br label %header
}

; Test that a simple loop is handled as expected.

; CHECK-LABEL: test2:
; CHECK: block BB2_2{{$}}
; CHECK: br_if {{[^,]*}}, BB2_2{{$}}
; CHECK: BB2_1:
; CHECK: br_if ${{[0-9]+}}, BB2_1{{$}}
; CHECK: BB2_2:
; CHECK: return{{$}}
; OPT-LABEL: test2:
; OPT: block BB2_2{{$}}
; OPT: br_if {{[^,]*}}, BB2_2{{$}}
; OPT: BB2_1:
; OPT: br_if ${{[0-9]+}}, BB2_1{{$}}
; OPT: BB2_2:
; OPT: return{{$}}
define void @test2(double* nocapture %p, i32 %n) {
entry:
  %cmp.4 = icmp sgt i32 %n, 0
  br i1 %cmp.4, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds double, double* %p, i32 %i.05
  %0 = load double, double* %arrayidx, align 8
  %mul = fmul double %0, 3.200000e+00
  store double %mul, double* %arrayidx, align 8
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; CHECK-LABEL: doublediamond:
; CHECK: block BB3_5{{$}}
; CHECK: block BB3_2{{$}}
; CHECK: br_if $0, BB3_2{{$}}
; CHECK: block BB3_4{{$}}
; CHECK: br_if $1, BB3_4{{$}}
; CHECK: br BB3_5{{$}}
; CHECK: BB3_4:
; CHECK: BB3_5:
; CHECK: return ${{[0-9]+}}{{$}}
; OPT-LABEL: doublediamond:
; OPT: block BB3_5{{$}}
; OPT: block BB3_4{{$}}
; OPT: br_if {{[^,]*}}, BB3_4{{$}}
; OPT: block BB3_3{{$}}
; OPT: br_if {{[^,]*}}, BB3_3{{$}}
; OPT: br BB3_5{{$}}
; OPT: BB3_4:
; OPT: BB3_5:
; OPT: return ${{[0-9]+}}{{$}}
define i32 @doublediamond(i32 %a, i32 %b, i32* %p) {
entry:
  %c = icmp eq i32 %a, 0
  %d = icmp eq i32 %b, 0
  store volatile i32 0, i32* %p
  br i1 %c, label %true, label %false
true:
  store volatile i32 1, i32* %p
  br label %exit
false:
  store volatile i32 2, i32* %p
  br i1 %d, label %ft, label %ff
ft:
  store volatile i32 3, i32* %p
  br label %exit
ff:
  store volatile i32 4, i32* %p
  br label %exit
exit:
  store volatile i32 5, i32* %p
  ret i32 0
}

; CHECK-LABEL: triangle:
; CHECK: block BB4_2{{$}}
; CHECK: br_if $1, BB4_2{{$}}
; CHECK: BB4_2:
; CHECK: return ${{[0-9]+}}{{$}}
; OPT-LABEL: triangle:
; OPT: block BB4_2{{$}}
; OPT: br_if $1, BB4_2{{$}}
; OPT: BB4_2:
; OPT: return ${{[0-9]+}}{{$}}
define i32 @triangle(i32* %p, i32 %a) {
entry:
  %c = icmp eq i32 %a, 0
  store volatile i32 0, i32* %p
  br i1 %c, label %true, label %exit
true:
  store volatile i32 1, i32* %p
  br label %exit
exit:
  store volatile i32 2, i32* %p
  ret i32 0
}

; CHECK-LABEL: diamond:
; CHECK: block BB5_3{{$}}
; CHECK: block BB5_2{{$}}
; CHECK: br_if $1, BB5_2{{$}}
; CHECK: br BB5_3{{$}}
; CHECK: BB5_2:
; CHECK: BB5_3:
; CHECK: return ${{[0-9]+}}{{$}}
; OPT-LABEL: diamond:
; OPT: block BB5_3{{$}}
; OPT: block BB5_2{{$}}
; OPT: br_if {{[^,]*}}, BB5_2{{$}}
; OPT: br BB5_3{{$}}
; OPT: BB5_2:
; OPT: BB5_3:
; OPT: return ${{[0-9]+}}{{$}}
define i32 @diamond(i32* %p, i32 %a) {
entry:
  %c = icmp eq i32 %a, 0
  store volatile i32 0, i32* %p
  br i1 %c, label %true, label %false
true:
  store volatile i32 1, i32* %p
  br label %exit
false:
  store volatile i32 2, i32* %p
  br label %exit
exit:
  store volatile i32 3, i32* %p
  ret i32 0
}

; CHECK-LABEL: single_block:
; CHECK-NOT: br
; CHECK: return $pop{{[0-9]+}}{{$}}
; OPT-LABEL: single_block:
; OPT-NOT: br
; OPT: return $pop{{[0-9]+}}{{$}}
define i32 @single_block(i32* %p) {
entry:
  store volatile i32 0, i32* %p
  ret i32 0
}

; CHECK-LABEL: minimal_loop:
; CHECK-NOT: br
; CHECK: BB7_1:
; CHECK: i32.store $discard=, 0($0), $pop{{[0-9]+}}{{$}}
; CHECK: br BB7_1{{$}}
; CHECK: BB7_2:
; OPT-LABEL: minimal_loop:
; OPT-NOT: br
; OPT: BB7_1:
; OPT: i32.store $discard=, 0($0), $pop{{[0-9]+}}{{$}}
; OPT: br BB7_1{{$}}
; OPT: BB7_2:
define i32 @minimal_loop(i32* %p) {
entry:
  store volatile i32 0, i32* %p
  br label %loop
loop:
  store volatile i32 1, i32* %p
  br label %loop
}

; CHECK-LABEL: simple_loop:
; CHECK-NOT: br
; CHECK: BB8_1:
; CHECK: loop BB8_2{{$}}
; CHECK: br_if $pop{{[0-9]+}}, BB8_1{{$}}
; CHECK: BB8_2:
; CHECK: return ${{[0-9]+}}{{$}}
; OPT-LABEL: simple_loop:
; OPT-NOT: br
; OPT: BB8_1:
; OPT: loop BB8_2{{$}}
; OPT: br_if {{[^,]*}}, BB8_1{{$}}
; OPT: BB8_2:
; OPT: return ${{[0-9]+}}{{$}}
define i32 @simple_loop(i32* %p, i32 %a) {
entry:
  %c = icmp eq i32 %a, 0
  store volatile i32 0, i32* %p
  br label %loop
loop:
  store volatile i32 1, i32* %p
  br i1 %c, label %loop, label %exit
exit:
  store volatile i32 2, i32* %p
  ret i32 0
}

; CHECK-LABEL: doubletriangle:
; CHECK: block BB9_4{{$}}
; CHECK: br_if $0, BB9_4{{$}}
; CHECK: block BB9_3{{$}}
; CHECK: br_if $1, BB9_3{{$}}
; CHECK: BB9_3:
; CHECK: BB9_4:
; CHECK: return ${{[0-9]+}}{{$}}
; OPT-LABEL: doubletriangle:
; OPT: block BB9_4{{$}}
; OPT: br_if $0, BB9_4{{$}}
; OPT: block BB9_3{{$}}
; OPT: br_if $1, BB9_3{{$}}
; OPT: BB9_3:
; OPT: BB9_4:
; OPT: return ${{[0-9]+}}{{$}}
define i32 @doubletriangle(i32 %a, i32 %b, i32* %p) {
entry:
  %c = icmp eq i32 %a, 0
  %d = icmp eq i32 %b, 0
  store volatile i32 0, i32* %p
  br i1 %c, label %true, label %exit
true:
  store volatile i32 2, i32* %p
  br i1 %d, label %tt, label %tf
tt:
  store volatile i32 3, i32* %p
  br label %tf
tf:
  store volatile i32 4, i32* %p
  br label %exit
exit:
  store volatile i32 5, i32* %p
  ret i32 0
}

; CHECK-LABEL: ifelse_earlyexits:
; CHECK: block BB10_4{{$}}
; CHECK: block BB10_2{{$}}
; CHECK: br_if $0, BB10_2{{$}}
; CHECK: br BB10_4{{$}}
; CHECK: BB10_2:
; CHECK: br_if $1, BB10_4{{$}}
; CHECK: BB10_4:
; CHECK: return ${{[0-9]+}}{{$}}
; OPT-LABEL: ifelse_earlyexits:
; OPT: block BB10_4{{$}}
; OPT: block BB10_3{{$}}
; OPT: br_if {{[^,]*}}, BB10_3{{$}}
; OPT: br_if $1, BB10_4{{$}}
; OPT: br BB10_4{{$}}
; OPT: BB10_3:
; OPT: BB10_4:
; OPT: return ${{[0-9]+}}{{$}}
define i32 @ifelse_earlyexits(i32 %a, i32 %b, i32* %p) {
entry:
  %c = icmp eq i32 %a, 0
  %d = icmp eq i32 %b, 0
  store volatile i32 0, i32* %p
  br i1 %c, label %true, label %false
true:
  store volatile i32 1, i32* %p
  br label %exit
false:
  store volatile i32 2, i32* %p
  br i1 %d, label %ft, label %exit
ft:
  store volatile i32 3, i32* %p
  br label %exit
exit:
  store volatile i32 4, i32* %p
  ret i32 0
}

; CHECK-LABEL: doublediamond_in_a_loop:
; CHECK: BB11_1:
; CHECK: loop            BB11_7{{$}}
; CHECK: block           BB11_6{{$}}
; CHECK: block           BB11_3{{$}}
; CHECK: br_if           $0, BB11_3{{$}}
; CHECK: br              BB11_6{{$}}
; CHECK: BB11_3:
; CHECK: block           BB11_5{{$}}
; CHECK: br_if           $1, BB11_5{{$}}
; CHECK: br              BB11_6{{$}}
; CHECK: BB11_5:
; CHECK: BB11_6:
; CHECK: br              BB11_1{{$}}
; CHECK: BB11_7:
; OPT-LABEL: doublediamond_in_a_loop:
; OPT: BB11_1:
; OPT: loop            BB11_7{{$}}
; OPT: block           BB11_6{{$}}
; OPT: block           BB11_5{{$}}
; OPT: br_if           {{[^,]*}}, BB11_5{{$}}
; OPT: block           BB11_4{{$}}
; OPT: br_if           {{[^,]*}}, BB11_4{{$}}
; OPT: br              BB11_6{{$}}
; OPT: BB11_4:
; OPT: br              BB11_6{{$}}
; OPT: BB11_5:
; OPT: BB11_6:
; OPT: br              BB11_1{{$}}
; OPT: BB11_7:
define i32 @doublediamond_in_a_loop(i32 %a, i32 %b, i32* %p) {
entry:
  br label %header
header:
  %c = icmp eq i32 %a, 0
  %d = icmp eq i32 %b, 0
  store volatile i32 0, i32* %p
  br i1 %c, label %true, label %false
true:
  store volatile i32 1, i32* %p
  br label %exit
false:
  store volatile i32 2, i32* %p
  br i1 %d, label %ft, label %ff
ft:
  store volatile i32 3, i32* %p
  br label %exit
ff:
  store volatile i32 4, i32* %p
  br label %exit
exit:
  store volatile i32 5, i32* %p
  br label %header
}

; Test that nested loops are handled.

; CHECK-LABEL: test3:
; CHECK: loop
; CHECK-NEXT: br_if
; CHECK-NEXT: BB{{[0-9]+}}_{{[0-9]+}}:
; CHECK-NEXT: loop
; OPT-LABEL: test3:
; OPT: loop
; OPT-NEXT: br_if
; OPT-NEXT: BB{{[0-9]+}}_{{[0-9]+}}:
; OPT-NEXT: loop
declare void @bar()
define void @test3(i32 %w)  {
entry:
  br i1 undef, label %outer.ph, label %exit

outer.ph:
  br label %outer

outer:
  %tobool = icmp eq i32 undef, 0
  br i1 %tobool, label %inner, label %unreachable

unreachable:
  unreachable

inner:
  %c = icmp eq i32 undef, %w
  br i1 %c, label %if.end, label %inner

exit:
  ret void

if.end:
  call void @bar()
  br label %outer
}

; Test switch lowering and block placement.

; CHECK-LABEL: test4:
; CHECK-NEXT: .param      i32{{$}}
; CHECK:      block       BB13_8{{$}}
; CHECK-NEXT: block       BB13_7{{$}}
; CHECK-NEXT: block       BB13_4{{$}}
; CHECK-NEXT: br_if       $pop{{[0-9]*}}, BB13_4{{$}}
; CHECK-NEXT: block       BB13_3{{$}}
; CHECK:      br_if       $pop{{[0-9]*}}, BB13_3{{$}}
; CHECK:      br_if       $pop{{[0-9]*}}, BB13_7{{$}}
; CHECK-NEXT: BB13_3:
; CHECK-NEXT: return{{$}}
; CHECK-NEXT: BB13_4:
; CHECK:      br_if       $pop{{[0-9]*}}, BB13_8{{$}}
; CHECK:      br_if       $pop{{[0-9]*}}, BB13_7{{$}}
; CHECK-NEXT: return{{$}}
; CHECK-NEXT: BB13_7:
; CHECK-NEXT: return{{$}}
; CHECK-NEXT: BB13_8:
; CHECK-NEXT: return{{$}}
; OPT-LABEL: test4:
; OPT-NEXT: .param      i32{{$}}
; OPT:      block       BB13_8{{$}}
; OPT-NEXT: block       BB13_7{{$}}
; OPT-NEXT: block       BB13_4{{$}}
; OPT-NEXT: br_if       $pop{{[0-9]*}}, BB13_4{{$}}
; OPT-NEXT: block       BB13_3{{$}}
; OPT:      br_if       $pop{{[0-9]*}}, BB13_3{{$}}
; OPT:      br_if       $pop{{[0-9]*}}, BB13_7{{$}}
; OPT-NEXT: BB13_3:
; OPT-NEXT: return{{$}}
; OPT-NEXT: BB13_4:
; OPT:      br_if       $pop{{[0-9]*}}, BB13_8{{$}}
; OPT:      br_if       $pop{{[0-9]*}}, BB13_7{{$}}
; OPT-NEXT: return{{$}}
; OPT-NEXT: BB13_7:
; OPT-NEXT: return{{$}}
; OPT-NEXT: BB13_8:
; OPT-NEXT: return{{$}}
define void @test4(i32 %t) {
entry:
  switch i32 %t, label %default [
    i32 0, label %bb2
    i32 2, label %bb2
    i32 4, label %bb1
    i32 622, label %bb0
  ]

bb0:
  ret void

bb1:
  ret void

bb2:
  ret void

default:
  ret void
}

; Test a case where the BLOCK needs to be placed before the LOOP in the
; same basic block.

; CHECK-LABEL: test5:
; CHECK:       BB14_1:
; CHECK-NEXT:  block BB14_4{{$}}
; CHECK-NEXT:  loop BB14_3{{$}}
; CHECK:       br_if {{[^,]*}}, BB14_4{{$}}
; CHECK:       br_if {{[^,]*}}, BB14_1{{$}}
; CHECK-NEXT:  BB14_3:
; CHECK:       return{{$}}
; CHECK-NEXT:  BB14_4:
; CHECK:       return{{$}}
; OPT-LABEL: test5:
; OPT:       BB14_1:
; OPT-NEXT:  block BB14_4{{$}}
; OPT-NEXT:  loop BB14_3{{$}}
; OPT:       br_if {{[^,]*}}, BB14_4{{$}}
; OPT:       br_if {{[^,]*}}, BB14_1{{$}}
; OPT-NEXT:  BB14_3:
; OPT:       return{{$}}
; OPT-NEXT:  BB14_4:
; OPT:       return{{$}}
define void @test5(i1 %p, i1 %q) {
entry:
  br label %header

header:
  store volatile i32 0, i32* null
  br i1 %p, label %more, label %alt

more:
  store volatile i32 1, i32* null
  br i1 %q, label %header, label %return

alt:
  store volatile i32 2, i32* null
  ret void

return:
  store volatile i32 3, i32* null
  ret void
}

; Test an interesting case of a loop with multiple exits, which
; aren't to layout successors of the loop, and one of which is to a successors
; which has another predecessor.

; CHECK-LABEL: test6:
; CHECK:       BB15_1:
; CHECK-NEXT:  block BB15_6{{$}}
; CHECK-NEXT:  block BB15_5{{$}}
; CHECK-NEXT:  loop  BB15_4{{$}}
; CHECK-NOT:   block
; CHECK:       br_if {{[^,]*}}, BB15_6{{$}}
; CHECK-NOT:   block
; CHECK:       br_if {{[^,]*}}, BB15_5{{$}}
; CHECK-NOT:   block
; CHECK:       br_if {{[^,]*}}, BB15_1{{$}}
; CHECK-NEXT:  BB15_4:
; CHECK-NOT:   block
; CHECK:       return{{$}}
; CHECK-NEXT:  BB15_5:
; CHECK-NOT:   block
; CHECK:       BB15_6:
; CHECK-NOT:   block
; CHECK:       return{{$}}
; OPT-LABEL: test6:
; OPT:       BB15_1:
; OPT-NEXT:  block BB15_6{{$}}
; OPT-NEXT:  block BB15_5{{$}}
; OPT-NEXT:  loop  BB15_4{{$}}
; OPT-NOT:   block
; OPT:       br_if {{[^,]*}}, BB15_6{{$}}
; OPT-NOT:   block
; OPT:       br_if {{[^,]*}}, BB15_5{{$}}
; OPT-NOT:   block
; OPT:       br_if {{[^,]*}}, BB15_1{{$}}
; OPT-NEXT:  BB15_4:
; OPT-NOT:   block
; OPT:       return{{$}}
; OPT-NEXT:  BB15_5:
; OPT-NOT:   block
; OPT:       BB15_6:
; OPT-NOT:   block
; OPT:       return{{$}}
define void @test6(i1 %p, i1 %q) {
entry:
  br label %header

header:
  store volatile i32 0, i32* null
  br i1 %p, label %more, label %second

more:
  store volatile i32 1, i32* null
  br i1 %q, label %evenmore, label %first

evenmore:
  store volatile i32 1, i32* null
  br i1 %q, label %header, label %return

return:
  store volatile i32 2, i32* null
  ret void

first:
  store volatile i32 3, i32* null
  br label %second

second:
  store volatile i32 4, i32* null
  ret void
}

; Test a case where there are multiple backedges and multiple loop exits
; that end in unreachable.

; CHECK-LABEL: test7:
; CHECK:       BB16_1:
; CHECK-NEXT:  loop BB16_5{{$}}
; CHECK-NOT:   block
; CHECK:       block BB16_4{{$}}
; CHECK-NEXT:  br_if {{[^,]*}}, BB16_4{{$}}
; CHECK-NOT:   block
; CHECK:       br_if {{[^,]*}}, BB16_1{{$}}
; CHECK-NOT:   block
; CHECK:       unreachable
; CHECK_NEXT:  BB16_4:
; CHECK-NOT:   block
; CHECK:       br_if {{[^,]*}}, BB16_1{{$}}
; CHECK-NEXT:  BB16_5:
; CHECK-NOT:   block
; CHECK:       unreachable
; OPT-LABEL: test7:
; OPT:       BB16_1:
; OPT-NEXT:  loop BB16_5{{$}}
; OPT-NOT:   block
; OPT:       block BB16_4{{$}}
; OPT-NOT:   block
; OPT:       br_if {{[^,]*}}, BB16_4{{$}}
; OPT-NOT:   block
; OPT:       br_if {{[^,]*}}, BB16_1{{$}}
; OPT-NOT:   block
; OPT:       unreachable
; OPT_NEXT:  BB16_4:
; OPT-NOT:   block
; OPT:       br_if {{[^,]*}}, BB16_1{{$}}
; OPT-NEXT:  BB16_5:
; OPT-NOT:   block
; OPT:       unreachable
define void @test7(i1 %tobool2, i1 %tobool9) {
entry:
  store volatile i32 0, i32* null
  br label %loop

loop:
  store volatile i32 1, i32* null
  br i1 %tobool2, label %l1, label %l0

l0:
  store volatile i32 2, i32* null
  br i1 %tobool9, label %loop, label %u0

l1:
  store volatile i32 3, i32* null
  br i1 %tobool9, label %loop, label %u1

u0:
  store volatile i32 4, i32* null
  unreachable

u1:
  store volatile i32 5, i32* null
  unreachable
}

; Test an interesting case using nested loops and switches.

; CHECK-LABEL: test8:
; CHECK:       BB17_1:
; CHECK-NEXT:  loop     BB17_4{{$}}
; CHECK-NEXT:  block    BB17_3{{$}}
; CHECK-NOT:   block
; CHECK:       br_if    {{[^,]*}}, BB17_3{{$}}
; CHECK-NOT:   block
; CHECK:       br_if    {{[^,]*}}, BB17_1{{$}}
; CHECK-NEXT:  BB17_3:
; CHECK-NEXT:  loop     BB17_4{{$}}
; CHECK-NEXT:  br_if    {{[^,]*}}, BB17_3{{$}}
; CHECK-NEXT:  br       BB17_1{{$}}
; CHECK-NEXT:  BB17_4:
; OPT-LABEL: test8:
; OPT:       BB17_1:
; OPT-NEXT:  loop     BB17_4{{$}}
; OPT-NEXT:  block    BB17_3{{$}}
; OPT-NOT:   block
; OPT:       br_if    {{[^,]*}}, BB17_3{{$}}
; OPT-NOT:   block
; OPT:       br_if    {{[^,]*}}, BB17_1{{$}}
; OPT-NEXT:  BB17_3:
; OPT-NEXT:  loop     BB17_4{{$}}
; OPT-NEXT:  br_if    {{[^,]*}}, BB17_3{{$}}
; OPT-NEXT:  br       BB17_1{{$}}
; OPT-NEXT:  BB17_4:
define i32 @test8() {
bb:
  br label %bb1

bb1:
  br i1 undef, label %bb2, label %bb3

bb2:
  switch i8 undef, label %bb1 [
    i8 44, label %bb2
  ]

bb3:
  switch i8 undef, label %bb1 [
    i8 44, label %bb2
  ]
}

; Test an interesting case using nested loops that share a bottom block.

; CHECK-LABEL: test9:
; CHECK:       BB18_1:
; CHECK-NEXT:  loop      BB18_5{{$}}
; CHECK-NOT:   block
; CHECK:       br_if     {{[^,]*}}, BB18_5{{$}}
; CHECK-NEXT:  BB18_2:
; CHECK-NEXT:  loop      BB18_5{{$}}
; CHECK-NOT:   block
; CHECK:       block     BB18_4{{$}}
; CHECK-NOT:   block
; CHECK:       br_if     {{[^,]*}}, BB18_4{{$}}
; CHECK-NOT:   block
; CHECK:       br_if     {{[^,]*}}, BB18_2{{$}}
; CHECK-NEXT:  br        BB18_1{{$}}
; CHECK-NEXT:  BB18_4:
; CHECK-NOT:   block
; CHECK:       br_if     {{[^,]*}}, BB18_2{{$}}
; CHECK-NEXT:  br        BB18_1{{$}}
; CHECK-NEXT:  BB18_5:
; CHECK-NOT:   block
; CHECK:       return{{$}}
; OPT-LABEL: test9:
; OPT:       BB18_1:
; OPT-NEXT:  loop      BB18_5{{$}}
; OPT-NOT:   block
; OPT:       br_if     {{[^,]*}}, BB18_5{{$}}
; OPT-NEXT:  BB18_2:
; OPT-NEXT:  loop      BB18_5{{$}}
; OPT-NOT:   block
; OPT:       block     BB18_4{{$}}
; OPT-NOT:   block
; OPT:       br_if     {{[^,]*}}, BB18_4{{$}}
; OPT-NOT:   block
; OPT:       br_if     {{[^,]*}}, BB18_2{{$}}
; OPT-NEXT:  br        BB18_1{{$}}
; OPT-NEXT:  BB18_4:
; OPT-NOT:   block
; OPT:       br_if     {{[^,]*}}, BB18_2{{$}}
; OPT-NEXT:  br        BB18_1{{$}}
; OPT-NEXT:  BB18_5:
; OPT-NOT:   block
; OPT:       return{{$}}
declare i1 @a()
define void @test9() {
entry:
  store volatile i32 0, i32* null
  br label %header

header:
  store volatile i32 1, i32* null
  %call4 = call i1 @a()
  br i1 %call4, label %header2, label %end

header2:
  store volatile i32 2, i32* null
  %call = call i1 @a()
  br i1 %call, label %if.then, label %if.else

if.then:
  store volatile i32 3, i32* null
  %call3 = call i1 @a()
  br i1 %call3, label %header2, label %header

if.else:
  store volatile i32 4, i32* null
  %call2 = call i1 @a()
  br i1 %call2, label %header2, label %header

end:
  store volatile i32 5, i32* null
  ret void
}

; Test an interesting case involving nested loops sharing a loop bottom,
; and loop exits to a block with unreachable.

; CHECK-LABEL: test10:
; CHECK:       BB19_1:
; CHECK-NEXT:  loop     BB19_7{{$}}
; CHECK-NOT:   block
; CHECK:       br_if    {{[^,]*}}, BB19_1{{$}}
; CHECK-NEXT:  BB19_2:
; CHECK-NEXT:  block    BB19_6{{$}}
; CHECK-NEXT:  loop     BB19_5{{$}}
; CHECK-NOT:   block
; CHECK:       BB19_3:
; CHECK-NEXT:  loop     BB19_5{{$}}
; CHECK-NOT:   block
; CHECK:       br_if    {{[^,]*}}, BB19_1{{$}}
; CHECK-NOT:   block
; CHECK:       tableswitch  {{[^,]*}}, BB19_3, BB19_3, BB19_5, BB19_1, BB19_2, BB19_6{{$}}
; CHECK-NEXT:  BB19_5:
; CHECK-NEXT:  return{{$}}
; CHECK-NEXT:  BB19_6:
; CHECK-NOT:   block
; CHECK:       br       BB19_1{{$}}
; CHECK-NEXT:  BB19_7:
; OPT-LABEL: test10:
; OPT:       BB19_1:
; OPT-NEXT:  loop     BB19_7{{$}}
; OPT-NOT:   block
; OPT:       br_if    {{[^,]*}}, BB19_1{{$}}
; OPT-NEXT:  BB19_2:
; OPT-NEXT:  block    BB19_6{{$}}
; OPT-NEXT:  loop     BB19_5{{$}}
; OPT-NOT:   block
; OPT:       BB19_3:
; OPT-NEXT:  loop     BB19_5{{$}}
; OPT-NOT:   block
; OPT:       br_if    {{[^,]*}}, BB19_1{{$}}
; OPT-NOT:   block
; OPT:       tableswitch  {{[^,]*}}, BB19_3, BB19_3, BB19_5, BB19_1, BB19_2, BB19_6{{$}}
; OPT-NEXT:  BB19_5:
; OPT-NEXT:  return{{$}}
; OPT-NEXT:  BB19_6:
; OPT-NOT:   block
; OPT:       br       BB19_1{{$}}
; OPT-NEXT:  BB19_7:
define void @test10() {
bb0:
  br label %bb1

bb1:
  %tmp = phi i32 [ 2, %bb0 ], [ 3, %bb3 ]
  %tmp3 = phi i32 [ undef, %bb0 ], [ %tmp11, %bb3 ]
  %tmp4 = icmp eq i32 %tmp3, 0
  br i1 %tmp4, label %bb4, label %bb2

bb2:
  br label %bb3

bb3:
  %tmp11 = phi i32 [ 1, %bb5 ], [ 0, %bb2 ]
  br label %bb1

bb4:
  %tmp6 = phi i32 [ %tmp9, %bb5 ], [ 4, %bb1 ]
  %tmp7 = phi i32 [ %tmp6, %bb5 ], [ %tmp, %bb1 ]
  br label %bb5

bb5:
  %tmp9 = phi i32 [ %tmp6, %bb5 ], [ %tmp7, %bb4 ]
  switch i32 %tmp9, label %bb2 [
    i32 0, label %bb5
    i32 1, label %bb6
    i32 3, label %bb4
    i32 4, label %bb3
  ]

bb6:
  ret void
}

; Test a CFG DAG with interesting merging.

; CHECK-LABEL: test11:
; CHECK:       block        BB20_8{{$}}
; CHECK-NEXT:  block        BB20_7{{$}}
; CHECK-NEXT:  block        BB20_6{{$}}
; CHECK-NEXT:  block        BB20_4{{$}}
; CHECK-NEXT:  br_if        {{[^,]*}}, BB20_4{{$}}
; CHECK-NOT:   block
; CHECK:       block        BB20_3{{$}}
; CHECK-NEXT:  br_if        {{[^,]*}}, BB20_3{{$}}
; CHECK-NOT:   block
; CHECK:       br_if        {{[^,]*}}, BB20_6{{$}}
; CHECK-NEXT:  BB20_3:
; CHECK-NOT:   block
; CHECK:       return{{$}}
; CHECK-NEXT:  BB20_4:
; CHECK-NOT:   block
; CHECK:       br_if        {{[^,]*}}, BB20_8{{$}}
; CHECK-NOT:   block
; CHECK:       br_if        {{[^,]*}}, BB20_7{{$}}
; CHECK-NEXT:  BB20_6:
; CHECK-NOT:   block
; CHECK:       return{{$}}
; CHECK-NEXT:  BB20_7:
; CHECK-NOT:   block
; CHECK:       return{{$}}
; CHECK-NEXT:  BB20_8:
; CHECK-NOT:   block
; CHECK:       return{{$}}
; OPT-LABEL: test11:
; OPT:       block        BB20_8{{$}}
; OPT-NEXT:  block        BB20_4{{$}}
; OPT-NEXT:  br_if        $0, BB20_4{{$}}
; OPT-NOT:   block
; OPT:       block        BB20_3{{$}}
; OPT-NEXT:  br_if        $0, BB20_3{{$}}
; OPT-NOT:   block
; OPT:       br_if        $0, BB20_8{{$}}
; OPT-NEXT:  BB20_3:
; OPT-NOT:   block
; OPT:       return{{$}}
; OPT-NEXT:  BB20_4:
; OPT-NOT:   block
; OPT:       block        BB20_6{{$}}
; OPT-NOT:   block
; OPT:       br_if        $pop9, BB20_6{{$}}
; OPT-NOT:   block
; OPT:       return{{$}}
; OPT-NEXT:  BB20_6:
; OPT-NOT:   block
; OPT:       br_if        $0, BB20_8{{$}}
; OPT-NOT:   block
; OPT:       return{{$}}
; OPT-NEXT:  BB20_8:
; OPT-NOT:   block
; OPT:       return{{$}}
define void @test11() {
bb0:
  store volatile i32 0, i32* null
  br i1 undef, label %bb1, label %bb4
bb1:
  store volatile i32 1, i32* null
  br i1 undef, label %bb3, label %bb2
bb2:
  store volatile i32 2, i32* null
  br i1 undef, label %bb3, label %bb7
bb3:
  store volatile i32 3, i32* null
  ret void
bb4:
  store volatile i32 4, i32* null
  br i1 undef, label %bb8, label %bb5
bb5:
  store volatile i32 5, i32* null
  br i1 undef, label %bb6, label %bb7
bb6:
  store volatile i32 6, i32* null
  ret void
bb7:
  store volatile i32 7, i32* null
  ret void
bb8:
  store volatile i32 8, i32* null
  ret void
}

; CHECK-LABEL: test12:
; CHECK:       BB21_1:
; CHECK-NEXT:  loop        BB21_8{{$}}
; CHECK-NOT:   block
; CHECK:       block       BB21_7{{$}}
; CHECK-NEXT:  block       BB21_6{{$}}
; CHECK-NEXT:  block       BB21_4{{$}}
; CHECK-NEXT:  br_if       {{[^,]*}}, BB21_4{{$}}
; CHECK-NOT:   block
; CHECK:       br_if       {{[^,]*}}, BB21_7{{$}}
; CHECK-NOT:   block
; CHECK:       br_if       {{[^,]*}}, BB21_7{{$}}
; CHECK-NEXT:  br          BB21_6{{$}}
; CHECK-NEXT:  BB21_4:
; CHECK-NOT:   block
; CHECK:       br_if       {{[^,]*}}, BB21_7{{$}}
; CHECK-NOT:   block
; CHECK:       br_if       {{[^,]*}}, BB21_7{{$}}
; CHECK-NEXT:  BB21_6:
; CHECK-NEXT:  return{{$}}
; CHECK-NEXT:  BB21_7:
; CHECK-NOT:   block
; CHECK:       br          BB21_1{{$}}
; CHECK-NEXT:  BB21_8:
; OPT-LABEL: test12:
; OPT:       BB21_1:
; OPT-NEXT:  loop        BB21_8{{$}}
; OPT-NOT:   block
; OPT:       block       BB21_7{{$}}
; OPT-NEXT:  block       BB21_6{{$}}
; OPT-NEXT:  block       BB21_4{{$}}
; OPT-NEXT:  br_if       {{[^,]*}}, BB21_4{{$}}
; OPT-NOT:   block
; OPT:       br_if       {{[^,]*}}, BB21_7{{$}}
; OPT-NOT:   block
; OPT:       br_if       {{[^,]*}}, BB21_7{{$}}
; OPT-NEXT:  br          BB21_6{{$}}
; OPT-NEXT:  BB21_4:
; OPT-NOT:   block
; OPT:       br_if       {{[^,]*}}, BB21_7{{$}}
; OPT-NOT:   block
; OPT:       br_if       {{[^,]*}}, BB21_7{{$}}
; OPT-NEXT:  BB21_6:
; OPT-NEXT:  return{{$}}
; OPT-NEXT:  BB21_7:
; OPT-NOT:   block
; OPT:       br          BB21_1{{$}}
; OPT-NEXT:  BB21_8:
define void @test12(i8* %arg) {
bb:
  br label %bb1

bb1:
  %tmp = phi i32 [ 0, %bb ], [ %tmp5, %bb4 ]
  %tmp2 = getelementptr i8, i8* %arg, i32 %tmp
  %tmp3 = load i8, i8* %tmp2
  switch i8 %tmp3, label %bb7 [
    i8 42, label %bb4
    i8 76, label %bb4
    i8 108, label %bb4
    i8 104, label %bb4
  ]

bb4:
  %tmp5 = add i32 %tmp, 1
  br label %bb1

bb7:
  ret void
}

; A block can be "branched to" from another even if it is also reachable via
; fallthrough from the other. This would normally be optimized away, so use
; optnone to disable optimizations to test this case.

; CHECK-LABEL: test13:
; CHECK-NEXT:  .local i32{{$}}
; CHECK:       block BB22_2{{$}}
; CHECK:       br_if $pop4, BB22_2{{$}}
; CHECK-NEXT:  return{{$}}
; CHECK-NEXT:  BB22_2:
; CHECK:       block BB22_4{{$}}
; CHECK-NEXT:  br_if $0, BB22_4{{$}}
; CHECK:       BB22_4:
; CHECK:       block BB22_5{{$}}
; CHECK:       br_if $pop6, BB22_5{{$}}
; CHECK-NEXT:  BB22_5:
; CHECK-NEXT:  unreachable{{$}}
; OPT-LABEL: test13:
; OPT-NEXT:  .local i32{{$}}
; OPT:       block BB22_2{{$}}
; OPT:       br_if $pop4, BB22_2{{$}}
; OPT-NEXT:  return{{$}}
; OPT-NEXT:  BB22_2:
; OPT:       block BB22_4{{$}}
; OPT-NEXT:  br_if $0, BB22_4{{$}}
; OPT:       BB22_4:
; OPT:       block BB22_5{{$}}
; OPT:       br_if $pop6, BB22_5{{$}}
; OPT-NEXT:  BB22_5:
; OPT-NEXT:  unreachable{{$}}
define void @test13() noinline optnone {
bb:
  br i1 undef, label %bb5, label %bb2
bb1:
  unreachable
bb2:
  br i1 undef, label %bb3, label %bb4
bb3:
  br label %bb4
bb4:
  %tmp = phi i1 [ false, %bb2 ], [ false, %bb3 ]
  br i1 %tmp, label %bb1, label %bb1
bb5:
  ret void
}
