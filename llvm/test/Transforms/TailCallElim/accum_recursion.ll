; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s

define i32 @test1_factorial(i32 %x) {
entry:
	%tmp.1 = icmp sgt i32 %x, 0
	br i1 %tmp.1, label %then, label %else
then:
	%tmp.6 = add i32 %x, -1
	%recurse = call i32 @test1_factorial( i32 %tmp.6 )
	%accumulate = mul i32 %recurse, %x
	ret i32 %accumulate
else:
	ret i32 1
}

; CHECK-LABEL: define i32 @test1_factorial(
; CHECK: tailrecurse:
; CHECK: %accumulator.tr = phi i32 [ 1, %entry ], [ %accumulate, %then ]
; CHECK: then:
; CHECK-NOT: %recurse
; CHECK: %accumulate = mul i32 %accumulator.tr, %x.tr
; CHECK: else:
; CHECK: %accumulator.ret.tr = mul i32 %accumulator.tr, 1
; CHECK: ret i32 %accumulator.ret.tr

; This is a more aggressive form of accumulator recursion insertion, which 
; requires noticing that X doesn't change as we perform the tailcall.

define i32 @test2_mul(i32 %x, i32 %y) {
entry:
	%tmp.1 = icmp eq i32 %y, 0
	br i1 %tmp.1, label %return, label %endif
endif:
	%tmp.8 = add i32 %y, -1
	%recurse = call i32 @test2_mul( i32 %x, i32 %tmp.8 )
	%accumulate = add i32 %recurse, %x
	ret i32 %accumulate
return:
	ret i32 %x
}

; CHECK-LABEL: define i32 @test2_mul(
; CHECK: tailrecurse:
; CHECK: %accumulator.tr = phi i32 [ 0, %entry ], [ %accumulate, %endif ]
; CHECK: endif:
; CHECK-NOT: %recurse
; CHECK: %accumulate = add i32 %accumulator.tr, %x
; CHECK: return:
; CHECK: %accumulator.ret.tr = add i32 %accumulator.tr, %x
; CHECK: ret i32 %accumulator.ret.tr

define i64 @test3_fib(i64 %n) nounwind readnone {
entry:
  switch i64 %n, label %bb1 [
    i64 0, label %bb2
    i64 1, label %bb2
  ]

bb1:
  %0 = add i64 %n, -1
  %recurse1 = tail call i64 @test3_fib(i64 %0) nounwind
  %1 = add i64 %n, -2
  %recurse2 = tail call i64 @test3_fib(i64 %1) nounwind
  %accumulate = add nsw i64 %recurse2, %recurse1
  ret i64 %accumulate

bb2:
  ret i64 %n
}

; CHECK-LABEL: define i64 @test3_fib(
; CHECK: tailrecurse:
; CHECK: %accumulator.tr = phi i64 [ 0, %entry ], [ %accumulate, %bb1 ]
; CHECK: bb1:
; CHECK-NOT: %recurse2
; CHECK: %accumulate = add nsw i64 %accumulator.tr, %recurse1
; CHECK: bb2:
; CHECK: %accumulator.ret.tr = add nsw i64 %accumulator.tr, %n.tr
; CHECK: ret i64 %accumulator.ret.tr

define i32 @test4_base_case_call() local_unnamed_addr {
entry:
  %base = call i32 @test4_helper()
  switch i32 %base, label %sw.default [
    i32 1, label %cleanup
    i32 5, label %cleanup
    i32 7, label %cleanup
  ]

sw.default:
  %recurse = call i32 @test4_base_case_call()
  %accumulate = add nsw i32 %recurse, 1
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ %accumulate, %sw.default ], [ %base, %entry ], [ %base, %entry ], [ %base, %entry ]
  ret i32 %retval.0
}

declare i32 @test4_helper()

; CHECK-LABEL: define i32 @test4_base_case_call(
; CHECK: tailrecurse:
; CHECK: %accumulator.tr = phi i32 [ 0, %entry ], [ %accumulate, %sw.default ]
; CHECK: sw.default:
; CHECK-NOT: %recurse
; CHECK: %accumulate = add nsw i32 %accumulator.tr, 1
; CHECK: cleanup:
; CHECK: %accumulator.ret.tr = add nsw i32 %accumulator.tr, %base
; CHECK: ret i32 %accumulator.ret.tr

define i32 @test5_base_case_load(i32* nocapture %A, i32 %n) local_unnamed_addr {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %base = load i32, i32* %A, align 4
  ret i32 %base

if.end:
  %idxprom = zext i32 %n to i64
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %idxprom
  %load = load i32, i32* %arrayidx1, align 4
  %sub = add i32 %n, -1
  %recurse = tail call i32 @test5_base_case_load(i32* %A, i32 %sub)
  %accumulate = add i32 %recurse, %load
  ret i32 %accumulate
}

; CHECK-LABEL: define i32 @test5_base_case_load(
; CHECK: tailrecurse:
; CHECK: %accumulator.tr = phi i32 [ 0, %entry ], [ %accumulate, %if.end ]
; CHECK: if.then:
; CHECK: %accumulator.ret.tr = add i32 %accumulator.tr, %base
; CHECK: ret i32 %accumulator.ret.tr
; CHECK: if.end:
; CHECK-NOT: %recurse
; CHECK: %accumulate = add i32 %accumulator.tr, %load

define i32 @test6_multiple_returns(i32 %x, i32 %y) local_unnamed_addr {
entry:
  switch i32 %x, label %default [
    i32 0, label %case0
    i32 99, label %case99
  ]

case0:
  %helper = call i32 @test6_helper()
  ret i32 %helper

case99:
  %sub1 = add i32 %x, -1
  %recurse1 = call i32 @test6_multiple_returns(i32 %sub1, i32 %y)
  ret i32 18

default:
  %sub2 = add i32 %x, -1
  %recurse2 = call i32 @test6_multiple_returns(i32 %sub2, i32 %y)
  %accumulate = add i32 %recurse2, %y
  ret i32 %accumulate
}

declare i32 @test6_helper()

; CHECK-LABEL: define i32 @test6_multiple_returns(
; CHECK: tailrecurse:
; CHECK: %accumulator.tr = phi i32 [ %accumulator.tr, %case99 ], [ 0, %entry ], [ %accumulate, %default ]
; CHECK: %ret.tr = phi i32 [ poison, %entry ], [ %current.ret.tr, %case99 ], [ %ret.tr, %default ]
; CHECK: %ret.known.tr = phi i1 [ false, %entry ], [ true, %case99 ], [ %ret.known.tr, %default ]
; CHECK: case0:
; CHECK: %accumulator.ret.tr2 = add i32 %accumulator.tr, %helper
; CHECK: %current.ret.tr1 = select i1 %ret.known.tr, i32 %ret.tr, i32 %accumulator.ret.tr2
; CHECK: case99:
; CHECK-NOT: %recurse
; CHECK: %accumulator.ret.tr = add i32 %accumulator.tr, 18
; CHECK: %current.ret.tr = select i1 %ret.known.tr, i32 %ret.tr, i32 %accumulator.ret.tr
; CHECK: default:
; CHECK-NOT: %recurse
; CHECK: %accumulate = add i32 %accumulator.tr, %y

; It is only safe to transform one accumulator per function, make sure we don't
; try to remove more.

define i32 @test7_multiple_accumulators(i32 %a) local_unnamed_addr {
entry:
  %tobool = icmp eq i32 %a, 0
  br i1 %tobool, label %return, label %if.end

if.end:
  %and = and i32 %a, 1
  %tobool1 = icmp eq i32 %and, 0
  %sub = add nsw i32 %a, -1
  br i1 %tobool1, label %if.end3, label %if.then2

if.then2:
  %recurse1 = tail call i32 @test7_multiple_accumulators(i32 %sub)
  %accumulate1 = add nsw i32 %recurse1, 1
  br label %return

if.end3:
  %recurse2 = tail call i32 @test7_multiple_accumulators(i32 %sub)
  %accumulate2 = mul nsw i32 %recurse2, 2
  br label %return

return:
  %retval.0 = phi i32 [ %accumulate1, %if.then2 ], [ %accumulate2, %if.end3 ], [ 0, %entry ]
  ret i32 %retval.0
}

; CHECK-LABEL: define i32 @test7_multiple_accumulators(
; CHECK: tailrecurse:
; CHECK: %accumulator.tr = phi i32 [ 0, %entry ], [ %accumulate1, %if.then2 ]
; CHECK: if.then2:
; CHECK-NOT: %recurse1
; CHECK: %accumulate1 = add nsw i32 %accumulator.tr, 1
; CHECK: if.end3:
; CHECK: %recurse2
; CHECK: %accumulator.ret.tr = add nsw i32 %accumulator.tr, %accumulate2
; CHECK: ret i32 %accumulator.ret.tr
; CHECK: return:
; CHECK: %accumulator.ret.tr1 = add nsw i32 %accumulator.tr, 0
; CHECK: ret i32 %accumulator.ret.tr1
