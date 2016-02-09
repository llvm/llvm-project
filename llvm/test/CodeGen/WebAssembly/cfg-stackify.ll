; RUN: llc < %s -asm-verbose=false -disable-block-placement -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs | FileCheck -check-prefix=OPT %s

; Test the CFG stackifier pass.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
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
; CHECK: br 0{{$}}
; CHECK: return{{$}}
; OPT-LABEL: test0:
; OPT: loop
; OPT-NOT: br
; OPT: i32.add
; OPT-NEXT: i32.ge_s
; OPT-NEXT: br_if
; OPT-NOT: br
; OPT: call
; OPT: br 0{{$}}
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
; CHECK: br 0{{$}}
; CHECK: return{{$}}
; OPT-LABEL: test1:
; OPT: loop
; OPT-NOT: br
; OPT: i32.add
; OPT-NEXT: i32.ge_s
; OPT-NEXT: br_if
; OPT-NOT: br
; OPT: call
; OPT: br 0{{$}}
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
; CHECK-NOT: local
; CHECK: block{{$}}
; CHECK: br_if 0, {{[^,]+}}{{$}}
; CHECK: .LBB2_1:
; CHECK: br_if 0, ${{[0-9]+}}{{$}}
; CHECK: .LBB2_2:
; CHECK: return{{$}}
; OPT-LABEL: test2:
; OPT-NOT: local
; OPT: block{{$}}
; OPT: br_if 0, {{[^,]+}}{{$}}
; OPT: .LBB2_1:
; OPT: br_if 0, ${{[0-9]+}}{{$}}
; OPT: .LBB2_2:
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
; CHECK: block{{$}}
; CHECK-NEXT: block{{$}}
; CHECK: br_if 0, ${{[^,]+}}{{$}}
; CHECK: br 1{{$}}
; CHECK: .LBB3_2:
; CHECK-NEXT: end_block{{$}}
; CHECK: block{{$}}
; CHECK: br_if 0, ${{[^,]+}}{{$}}
; CHECK: br 1{{$}}
; CHECK: .LBB3_4:
; CHECK-NEXT: end_block{{$}}
; CHECK: .LBB3_5:
; CHECK-NEXT: end_block{{$}}
; CHECK: i32.const $push{{[0-9]+}}=, 0{{$}}
; CHECK-NEXT: return $pop{{[0-9]+}}{{$}}
; OPT-LABEL: doublediamond:
; OPT: block{{$}}
; OPT-NEXT: block{{$}}
; OPT: br_if 0, ${{[^,]+}}{{$}}
; OPT: block{{$}}
; OPT: br_if 0, ${{[^,]+}}{{$}}
; OPT: br 1{{$}}
; OPT: .LBB3_4:
; OPT: .LBB3_5:
; OPT: i32.const $push{{[0-9]+}}=, 0{{$}}
; OPT-NEXT: return $pop{{[0-9]+}}{{$}}
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
; CHECK: block{{$}}
; CHECK: br_if 0, $1{{$}}
; CHECK: .LBB4_2:
; CHECK: return ${{[0-9]+}}{{$}}
; OPT-LABEL: triangle:
; OPT: block{{$}}
; OPT: br_if 0, $1{{$}}
; OPT: .LBB4_2:
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
; CHECK: block{{$}}
; CHECK: block{{$}}
; CHECK: br_if 0, $1{{$}}
; CHECK: br 1{{$}}
; CHECK: .LBB5_2:
; CHECK: .LBB5_3:
; CHECK: i32.const $push{{[0-9]+}}=, 0{{$}}
; CHECK-NEXT: return $pop{{[0-9]+}}{{$}}
; OPT-LABEL: diamond:
; OPT: block{{$}}
; OPT: block{{$}}
; OPT: br_if 0, {{[^,]+}}{{$}}
; OPT: br 1{{$}}
; OPT: .LBB5_2:
; OPT: .LBB5_3:
; OPT: i32.const $push{{[0-9]+}}=, 0{{$}}
; OPT-NEXT: return $pop{{[0-9]+}}{{$}}
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
; CHECK: .LBB7_1:
; CHECK: i32.store $discard=, 0($0), $pop{{[0-9]+}}{{$}}
; CHECK: br 0{{$}}
; CHECK: .LBB7_2:
; OPT-LABEL: minimal_loop:
; OPT-NOT: br
; OPT: .LBB7_1:
; OPT: i32.store $discard=, 0($0), $pop{{[0-9]+}}{{$}}
; OPT: br 0{{$}}
; OPT: .LBB7_2:
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
; CHECK: .LBB8_1:
; CHECK: loop{{$}}
; CHECK: br_if 0, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: end_loop{{$}}
; CHECK: i32.const $push{{[0-9]+}}=, 0{{$}}
; CHECK-NEXT: return $pop{{[0-9]+}}{{$}}
; OPT-LABEL: simple_loop:
; OPT-NOT: br
; OPT: .LBB8_1:
; OPT: loop{{$}}
; OPT: br_if 0, {{[^,]+}}{{$}}
; OPT-NEXT: end_loop{{$}}
; OPT: i32.const $push{{[0-9]+}}=, 0{{$}}
; OPT-NEXT: return $pop{{[0-9]+}}{{$}}
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
; CHECK: block{{$}}
; CHECK: br_if 0, $0{{$}}
; CHECK: block{{$}}
; CHECK: br_if 0, $1{{$}}
; CHECK: .LBB9_3:
; CHECK: .LBB9_4:
; CHECK: return ${{[0-9]+}}{{$}}
; OPT-LABEL: doubletriangle:
; OPT: block{{$}}
; OPT: br_if 0, $0{{$}}
; OPT: block{{$}}
; OPT: br_if 0, $1{{$}}
; OPT: .LBB9_3:
; OPT: .LBB9_4:
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
; CHECK: block{{$}}
; CHECK: block{{$}}
; CHECK: br_if 0, $0{{$}}
; CHECK: br 1{{$}}
; CHECK: .LBB10_2:
; CHECK: br_if 0, $1{{$}}
; CHECK: .LBB10_4:
; CHECK: i32.const $push{{[0-9]+}}=, 0{{$}}
; CHECK-NEXT: return $pop{{[0-9]+}}{{$}}
; OPT-LABEL: ifelse_earlyexits:
; OPT: block{{$}}
; OPT: block{{$}}
; OPT: br_if 0, {{[^,]+}}{{$}}
; OPT: br_if 1, $1{{$}}
; OPT: br 1{{$}}
; OPT: .LBB10_3:
; OPT: .LBB10_4:
; OPT: i32.const $push{{[0-9]+}}=, 0{{$}}
; OPT-NEXT: return $pop{{[0-9]+}}{{$}}
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
; CHECK: .LBB11_1:
; CHECK: loop{{$}}
; CHECK: block{{$}}
; CHECK: block{{$}}
; CHECK: br_if           0, $0{{$}}
; CHECK: br              1{{$}}
; CHECK: .LBB11_3:
; CHECK: block{{$}}
; CHECK: br_if           0, $1{{$}}
; CHECK: br              1{{$}}
; CHECK: .LBB11_5:
; CHECK: .LBB11_6:
; CHECK: br              0{{$}}
; CHECK: .LBB11_7:
; CHECK-NEXT: end_loop{{$}}
; OPT-LABEL: doublediamond_in_a_loop:
; OPT: .LBB11_1:
; OPT: loop{{$}}
; OPT: block{{$}}
; OPT: block{{$}}
; OPT: br_if           0, {{[^,]+}}{{$}}
; OPT: block{{$}}
; OPT: br_if           0, {{[^,]+}}{{$}}
; OPT: br              2{{$}}
; OPT: .LBB11_4:
; OPT-NEXT: end_block{{$}}
; OPT: br              1{{$}}
; OPT: .LBB11_5:
; OPT-NEXT: end_block{{$}}
; OPT: .LBB11_6:
; OPT-NEXT: end_block{{$}}
; OPT: br              0{{$}}
; OPT: .LBB11_7:
; OPT-NEXT: end_loop{{$}}
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
; CHECK-NEXT: .LBB{{[0-9]+}}_{{[0-9]+}}:
; CHECK-NEXT: loop
; OPT-LABEL: test3:
; OPT: loop
; OPT-NEXT: br_if
; OPT-NEXT: .LBB{{[0-9]+}}_{{[0-9]+}}:
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
; CHECK-NEXT: .param       i32{{$}}
; CHECK:      block{{$}}
; CHECK-NEXT: block{{$}}
; CHECK-NEXT: block{{$}}
; CHECK:      br_if       0, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: block{{$}}
; CHECK:      br_if       0, $pop{{[0-9]+}}{{$}}
; CHECK:      br_if       2, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: .LBB13_3:
; CHECK-NEXT: end_block{{$}}
; CHECK-NEXT: return{{$}}
; CHECK-NEXT: .LBB13_4:
; CHECK:      br_if       1, $pop{{[0-9]+}}{{$}}
; CHECK:      br_if       0, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return{{$}}
; CHECK-NEXT: .LBB13_7:
; CHECK-NEXT: end_block{{$}}
; CHECK-NEXT: return{{$}}
; CHECK-NEXT: .LBB13_8:
; CHECK-NEXT: end_block{{$}}
; CHECK-NEXT: return{{$}}
; OPT-LABEL: test4:
; OPT-NEXT: .param       i32{{$}}
; OPT:      block{{$}}
; OPT-NEXT: block{{$}}
; OPT-NEXT: block{{$}}
; OPT:      br_if       0, $pop{{[0-9]+}}{{$}}
; OPT-NEXT: block{{$}}
; OPT:      br_if       0, $pop{{[0-9]+}}{{$}}
; OPT:      br_if       2, $pop{{[0-9]+}}{{$}}
; OPT-NEXT: .LBB13_3:
; OPT-NEXT: end_block{{$}}
; OPT-NEXT: return{{$}}
; OPT-NEXT: .LBB13_4:
; OPT:      br_if       1, $pop{{[0-9]+}}{{$}}
; OPT:      br_if       0, $pop{{[0-9]+}}{{$}}
; OPT-NEXT: return{{$}}
; OPT-NEXT: .LBB13_7:
; OPT-NEXT: end_block{{$}}
; OPT-NEXT: return{{$}}
; OPT-NEXT: .LBB13_8:
; OPT-NEXT: end_block{{$}}
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
; CHECK:       .LBB14_1:
; CHECK-NEXT:  block{{$}}
; CHECK-NEXT:  loop{{$}}
; CHECK:       br_if 2, {{[^,]+}}{{$}}
; CHECK:       br_if 0, {{[^,]+}}{{$}}
; CHECK-NEXT:  end_loop{{$}}
; CHECK:       return{{$}}
; CHECK-NEXT:  .LBB14_4:
; CHECK:       return{{$}}
; OPT-LABEL: test5:
; OPT:       .LBB14_1:
; OPT-NEXT:  block{{$}}
; OPT-NEXT:  loop{{$}}
; OPT:       br_if 2, {{[^,]+}}{{$}}
; OPT:       br_if 0, {{[^,]+}}{{$}}
; OPT-NEXT:  end_loop{{$}}
; OPT:       return{{$}}
; OPT-NEXT:  .LBB14_4:
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
; CHECK:       .LBB15_1:
; CHECK-NEXT:  block{{$}}
; CHECK-NEXT:  block{{$}}
; CHECK-NEXT:  loop{{$}}
; CHECK-NOT:   block
; CHECK:       br_if 3, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if 2, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if 0, {{[^,]+}}{{$}}
; CHECK-NEXT:  end_loop{{$}}
; CHECK-NOT:   block
; CHECK:       return{{$}}
; CHECK-NEXT:  .LBB15_5:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       .LBB15_6:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       return{{$}}
; OPT-LABEL: test6:
; OPT:       .LBB15_1:
; OPT-NEXT:  block{{$}}
; OPT-NEXT:  block{{$}}
; OPT-NEXT:  loop{{$}}
; OPT-NOT:   block
; OPT:       br_if 3, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       br_if 2, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       br_if 0, {{[^,]+}}{{$}}
; OPT-NEXT:  end_loop{{$}}
; OPT-NOT:   block
; OPT:       return{{$}}
; OPT-NEXT:  .LBB15_5:
; OPT-NEXT:  end_block{{$}}
; OPT-NOT:   block
; OPT:       .LBB15_6:
; OPT-NEXT:  end_block{{$}}
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
; CHECK:       .LBB16_1:
; CHECK-NEXT:  loop{{$}}
; CHECK-NOT:   block
; CHECK:       block{{$}}
; CHECK:       br_if 0, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if 1, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       unreachable
; CHECK-NEXT:  .LBB16_4:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       br_if 0, {{[^,]+}}{{$}}
; CHECK-NEXT:  end_loop{{$}}
; CHECK-NOT:   block
; CHECK:       unreachable
; OPT-LABEL: test7:
; OPT:       .LBB16_1:
; OPT-NEXT:  loop{{$}}
; OPT-NOT:   block
; OPT:       block{{$}}
; OPT-NOT:   block
; OPT:       br_if 0, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       br_if 1, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       unreachable
; OPT-NEXT:  .LBB16_4:
; OPT-NEXT:  end_block{{$}}
; OPT-NOT:   block
; OPT:       br_if 0, {{[^,]+}}{{$}}
; OPT-NEXT:  end_loop{{$}}
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
; CHECK:       .LBB17_1:
; CHECK-NEXT:  loop{{$}}
; CHECK-NEXT:  block{{$}}
; CHECK-NOT:   block
; CHECK:       br_if    0, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if    1, {{[^,]+}}{{$}}
; CHECK-NEXT:  .LBB17_3:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NEXT:  loop{{$}}
; CHECK-NEXT:  i32.const $push{{[^,]+}}, 0{{$}}
; CHECK-NEXT:  br_if    0, {{[^,]+}}{{$}}
; CHECK-NEXT:  br       2{{$}}
; CHECK-NEXT:  .LBB17_4:
; OPT-LABEL: test8:
; OPT:       .LBB17_1:
; OPT-NEXT:  loop{{$}}
; OPT-NEXT:  block{{$}}
; OPT-NOT:   block
; OPT:       br_if    0, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       br_if    1, {{[^,]+}}{{$}}
; OPT-NEXT:  .LBB17_3:
; OPT-NEXT:  end_block{{$}}
; OPT-NEXT:  loop{{$}}
; OPT-NEXT:  i32.const $push{{[^,]+}}, 0{{$}}
; OPT-NEXT:  br_if    0, {{[^,]+}}{{$}}
; OPT-NEXT:  br       2{{$}}
; OPT-NEXT:  .LBB17_4:
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
; CHECK:       .LBB18_1:
; CHECK-NEXT:  loop{{$}}
; CHECK-NOT:   block
; CHECK:       br_if     1, {{[^,]+}}{{$}}
; CHECK-NEXT:  .LBB18_2:
; CHECK-NEXT:  loop{{$}}
; CHECK-NOT:   block
; CHECK:       block{{$}}
; CHECK-NOT:   block
; CHECK:       br_if     0, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if     1, {{[^,]+}}{{$}}
; CHECK-NEXT:  br        3{{$}}
; CHECK-NEXT:  .LBB18_4:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       br_if     0, {{[^,]+}}{{$}}
; CHECK-NEXT:  br        2{{$}}
; CHECK-NEXT:  .LBB18_5:
; CHECK-NOT:   block
; CHECK:       return{{$}}
; OPT-LABEL: test9:
; OPT:       .LBB18_1:
; OPT-NEXT:  loop{{$}}
; OPT-NOT:   block
; OPT:       br_if     1, {{[^,]+}}{{$}}
; OPT-NEXT:  .LBB18_2:
; OPT-NEXT:  loop{{$}}
; OPT-NOT:   block
; OPT:       block{{$}}
; OPT-NOT:   block
; OPT:       br_if     0, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       br_if     1, {{[^,]+}}{{$}}
; OPT-NEXT:  br        3{{$}}
; OPT-NEXT:  .LBB18_4:
; OPT-NEXT:  end_block{{$}}
; OPT-NOT:   block
; OPT:       br_if     0, {{[^,]+}}{{$}}
; OPT-NEXT:  br        2{{$}}
; OPT-NEXT:  .LBB18_5:
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
; CHECK:       .LBB19_1:
; CHECK-NEXT:  loop{{$}}
; CHECK-NOT:   block
; CHECK:       br_if    0, {{[^,]+}}{{$}}
; CHECK-NEXT:  .LBB19_2:
; CHECK-NEXT:  block{{$}}
; CHECK-NEXT:  loop{{$}}
; CHECK-NOT:   block
; CHECK:       .LBB19_3:
; CHECK-NEXT:  loop{{$}}
; CHECK-NOT:   block
; CHECK:       br_if    5, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       tableswitch  {{[^,]+}}, 0, 0, 1, 5, 2, 4{{$}}
; CHECK-NEXT:  .LBB19_5:
; CHECK-NEXT:  end_loop{{$}}
; CHECK-NEXT:  end_loop{{$}}
; CHECK-NEXT:  return{{$}}
; CHECK-NEXT:  .LBB19_6:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       br       0{{$}}
; CHECK-NEXT:  .LBB19_7:
; OPT-LABEL: test10:
; OPT:       .LBB19_1:
; OPT-NEXT:  loop{{$}}
; OPT-NOT:   block
; OPT:       br_if    0, {{[^,]+}}{{$}}
; OPT-NEXT:  .LBB19_2:
; OPT-NEXT:  block{{$}}
; OPT-NEXT:  loop{{$}}
; OPT-NOT:   block
; OPT:       .LBB19_3:
; OPT-NEXT:  loop{{$}}
; OPT-NOT:   block
; OPT:       br_if    5, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       tableswitch  {{[^,]+}}, 0, 0, 1, 5, 2, 4{{$}}
; OPT-NEXT:  .LBB19_5:
; OPT-NEXT:  end_loop{{$}}
; OPT-NEXT:  end_loop{{$}}
; OPT-NEXT:  return{{$}}
; OPT-NEXT:  .LBB19_6:
; OPT-NEXT:  end_block{{$}}
; OPT-NOT:   block
; OPT:       br       0{{$}}
; OPT-NEXT:  .LBB19_7:
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
; CHECK:       block{{$}}
; CHECK-NEXT:  block{{$}}
; CHECK-NEXT:  block{{$}}
; CHECK-NEXT:  block{{$}}
; CHECK:       br_if        0, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       block{{$}}
; CHECK-NEXT:  br_if        0, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if        2, {{[^,]+}}{{$}}
; CHECK-NEXT:  .LBB20_3:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       return{{$}}
; CHECK-NEXT:  .LBB20_4:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       br_if        2, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if        1, {{[^,]+}}{{$}}
; CHECK-NEXT:  .LBB20_6:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       return{{$}}
; CHECK-NEXT:  .LBB20_7:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       return{{$}}
; CHECK-NEXT:  .LBB20_8:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       return{{$}}
; OPT-LABEL: test11:
; OPT:       block{{$}}
; OPT-NEXT:  block{{$}}
; OPT:       br_if        0, $pop{{[0-9]+}}{{$}}
; OPT-NOT:   block
; OPT:       block{{$}}
; OPT-NEXT:  br_if        0, $0{{$}}
; OPT-NOT:   block
; OPT:       br_if        2, {{[^,]+}}{{$}}
; OPT-NEXT:  .LBB20_3:
; OPT-NEXT:  end_block{{$}}
; OPT-NOT:   block
; OPT:       return{{$}}
; OPT-NEXT:  .LBB20_4:
; OPT-NEXT:  end_block{{$}}
; OPT-NOT:   block
; OPT:       block{{$}}
; OPT-NOT:   block
; OPT:       br_if        0, $pop{{[0-9]+}}{{$}}
; OPT-NOT:   block
; OPT:       return{{$}}
; OPT-NEXT:  .LBB20_6:
; OPT-NEXT:  end_block{{$}}
; OPT-NOT:   block
; OPT:       br_if        0, $pop{{[0-9]+}}{{$}}
; OPT-NOT:   block
; OPT:       return{{$}}
; OPT-NEXT:  .LBB20_8:
; OPT-NEXT:  end_block{{$}}
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
; CHECK:       .LBB21_1:
; CHECK-NEXT:  loop{{$}}
; CHECK-NOT:   block
; CHECK:       block{{$}}
; CHECK-NEXT:  block{{$}}
; CHECK-NEXT:  block{{$}}
; CHECK:       br_if       0, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if       2, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if       2, {{[^,]+}}{{$}}
; CHECK-NEXT:  br          1{{$}}
; CHECK-NEXT:  .LBB21_4:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       br_if       1, {{[^,]+}}{{$}}
; CHECK-NOT:   block
; CHECK:       br_if       1, {{[^,]+}}{{$}}
; CHECK-NEXT:  .LBB21_6:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NEXT:  return{{$}}
; CHECK-NEXT:  .LBB21_7:
; CHECK-NEXT:  end_block{{$}}
; CHECK-NOT:   block
; CHECK:       br          0{{$}}
; CHECK-NEXT:  .LBB21_8:
; OPT-LABEL: test12:
; OPT:       .LBB21_1:
; OPT-NEXT:  loop{{$}}
; OPT-NOT:   block
; OPT:       block{{$}}
; OPT-NEXT:  block{{$}}
; OPT-NEXT:  block{{$}}
; OPT:       br_if       0, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       br_if       2, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       br_if       2, {{[^,]+}}{{$}}
; OPT-NEXT:  br          1{{$}}
; OPT-NEXT:  .LBB21_4:
; OPT-NEXT:  end_block{{$}}
; OPT-NOT:   block
; OPT:       br_if       1, {{[^,]+}}{{$}}
; OPT-NOT:   block
; OPT:       br_if       1, {{[^,]+}}{{$}}
; OPT-NEXT:  .LBB21_6:
; OPT-NEXT:  end_block{{$}}
; OPT-NEXT:  return{{$}}
; OPT-NEXT:  .LBB21_7:
; OPT-NEXT:  end_block{{$}}
; OPT-NOT:   block
; OPT:       br          0{{$}}
; OPT-NEXT:  .LBB21_8:
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
; CHECK:       block{{$}}
; CHECK:       br_if 0, $pop5{{$}}
; CHECK-NEXT:  return{{$}}
; CHECK-NEXT:  .LBB22_2:
; CHECK-NEXT:  end_block{{$}}
; CHECK:       block{{$}}
; CHECK-NEXT:  i32.const $push3=, 0{{$}}
; CHECK-NEXT:  br_if 0, $pop3{{$}}
; CHECK:       .LBB22_4:
; CHECK-NEXT:  end_block{{$}}
; CHECK:       block{{$}}
; CHECK:       br_if 0, $pop7{{$}}
; CHECK-NEXT:  end_block{{$}}
; CHECK-NEXT:  unreachable{{$}}
; OPT-LABEL: test13:
; OPT-NEXT:  .local i32{{$}}
; OPT:       block{{$}}
; OPT:       br_if 0, $pop5{{$}}
; OPT-NEXT:  return{{$}}
; OPT-NEXT:  .LBB22_2:
; OPT-NEXT:  end_block{{$}}
; OPT:       block{{$}}
; OPT-NEXT:  i32.const $push3=, 0{{$}}
; OPT-NEXT:  br_if 0, $pop3{{$}}
; OPT:       .LBB22_4:
; OPT-NEXT:  end_block{{$}}
; OPT:       block{{$}}
; OPT:       br_if 0, $pop7{{$}}
; OPT-NEXT:  end_block{{$}}
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

; Test a case with a single-block loop that has another loop
; as a successor. The end_loop for the first loop should go
; before the loop for the second.

; CHECK-LABEL: test14:
; CHECK-NEXT: .LBB23_1:{{$}}
; CHECK-NEXT:     loop{{$}}
; CHECK-NEXT:     i32.const   $push0=, 0{{$}}
; CHECK-NEXT:     br_if       0, $pop0{{$}}
; CHECK-NEXT: .LBB23_2:{{$}}
; CHECK-NEXT:     end_loop{{$}}
; CHECK-NEXT:     loop{{$}}
; CHECK-NEXT:     i32.const   $discard=, 0{{$}}
; CHECK-NEXT:     i32.const   $push1=, 0{{$}}
; CHECK-NEXT:     br_if       0, $pop1{{$}}
; CHECK-NEXT:     end_loop{{$}}
; CHECK-NEXT:     return{{$}}
define void @test14() {
bb:
  br label %bb1

bb1:
  %tmp = bitcast i1 undef to i1
  br i1 %tmp, label %bb3, label %bb1

bb3:
  br label %bb4

bb4:
  br i1 undef, label %bb7, label %bb48

bb7:
  br i1 undef, label %bb12, label %bb12

bb12:
  br i1 undef, label %bb17, label %bb17

bb17:
  br i1 undef, label %bb22, label %bb22

bb22:
  br i1 undef, label %bb27, label %bb27

bb27:
  br i1 undef, label %bb30, label %bb30

bb30:
  br i1 undef, label %bb35, label %bb35

bb35:
  br i1 undef, label %bb38, label %bb38

bb38:
  br i1 undef, label %bb48, label %bb48

bb48:
  %tmp49 = bitcast i1 undef to i1
  br i1 %tmp49, label %bb3, label %bb50

bb50:
  ret void
}
