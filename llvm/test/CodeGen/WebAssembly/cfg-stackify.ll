; RUN: llc < %s -asm-verbose=false -disable-block-placement | FileCheck %s
; RUN: llc < %s -asm-verbose=false | FileCheck -check-prefix=OPT %s

; Test the CFG stackifier pass.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @something()

; Test that loops are made contiguous, even in the presence of split backedges.

; CHECK-LABEL: test0:
; CHECK: loop
; CHECK: i32.add
; CHECK-NOT: br
; CHECK: br_if
; CHECK-NOT: br
; CHECK: call
; CHECK: br BB0_1{{$}}
; CHECK: return{{$}}
; OPT-LABEL: test0:
; OPT: loop
; OPT: i32.add
; OPT-NOT: br
; OPT: br_if
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
; CHECK: i32.add
; CHECK-NOT: br
; CHECK: br_if
; CHECK-NOT: br
; CHECK: call
; CHECK: br BB1_1{{$}}
; CHECK: return{{$}}
; OPT-LABEL: test1:
; OPT: loop
; OPT: i32.add
; OPT-NOT: br
; OPT: br_if
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
; CHECK: br_if {{.*}}, BB2_2{{$}}
; CHECK: BB2_1:
; CHECK: br_if ${{[0-9]+}}, BB2_1{{$}}
; CHECK: BB2_2:
; CHECK: return{{$}}
; OPT-LABEL: test2:
; OPT: block BB2_2{{$}}
; OPT: br_if {{.*}}, BB2_2{{$}}
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
; OPT: br_if {{.*}}, BB3_4{{$}}
; OPT: block BB3_3{{$}}
; OPT: br_if {{.*}}, BB3_3{{$}}
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
; OPT: br_if {{.*}}, BB5_2{{$}}
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
; OPT: br_if {{.*}}, BB8_1{{$}}
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
; OPT: br_if {{.*}}, BB10_3{{$}}
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
; OPT: br_if           {{.*}}, BB11_5{{$}}
; OPT: block           BB11_4{{$}}
; OPT: br_if           {{.*}}, BB11_4{{$}}
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
