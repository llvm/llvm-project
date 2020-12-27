; RUN: opt < %s -basic-aa -gvn -S | FileCheck %s

@a = external global i32		; <i32*> [#uses=7]

; CHECK-LABEL: @test1(
define i32 @test1() nounwind {
entry:
	%0 = load i32, i32* @a, align 4
	%1 = icmp eq i32 %0, 4
	br i1 %1, label %bb, label %bb1

bb:		; preds = %entry
	br label %bb8

bb1:		; preds = %entry
	%2 = load i32, i32* @a, align 4
	%3 = icmp eq i32 %2, 5
	br i1 %3, label %bb2, label %bb3

bb2:		; preds = %bb1
	br label %bb8

bb3:		; preds = %bb1
	%4 = load i32, i32* @a, align 4
	%5 = icmp eq i32 %4, 4
; CHECK: br i1 false, label %bb4, label %bb5
	br i1 %5, label %bb4, label %bb5

bb4:		; preds = %bb3
	%6 = load i32, i32* @a, align 4
	%7 = add i32 %6, 5
	br label %bb8

bb5:		; preds = %bb3
	%8 = load i32, i32* @a, align 4
	%9 = icmp eq i32 %8, 5
; CHECK: br i1 false, label %bb6, label %bb7
	br i1 %9, label %bb6, label %bb7

bb6:		; preds = %bb5
	%10 = load i32, i32* @a, align 4
	%11 = add i32 %10, 4
	br label %bb8

bb7:		; preds = %bb5
	%12 = load i32, i32* @a, align 4
	br label %bb8

bb8:		; preds = %bb7, %bb6, %bb4, %bb2, %bb
	%.0 = phi i32 [ %12, %bb7 ], [ %11, %bb6 ], [ %7, %bb4 ], [ 4, %bb2 ], [ 5, %bb ]
	br label %return

return:		; preds = %bb8
	ret i32 %.0
}

declare void @foo(i1)
declare void @bar(i32)

; CHECK-LABEL: @test3(
define void @test3(i32 %x, i32 %y) {
  %xz = icmp eq i32 %x, 0
  %yz = icmp eq i32 %y, 0
  %z = and i1 %xz, %yz
  br i1 %z, label %both_zero, label %nope
both_zero:
  call void @foo(i1 %xz)
; CHECK: call void @foo(i1 true)
  call void @foo(i1 %yz)
; CHECK: call void @foo(i1 true)
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 0)
  call void @bar(i32 %y)
; CHECK: call void @bar(i32 0)
  ret void
nope:
  call void @foo(i1 %z)
; CHECK: call void @foo(i1 false)
  ret void
}

; CHECK-LABEL: @test3_select(
define void @test3_select(i32 %x, i32 %y) {
  %xz = icmp eq i32 %x, 0
  %yz = icmp eq i32 %y, 0
  %z = select i1 %xz, i1 %yz, i1 false
  br i1 %z, label %both_zero, label %nope
both_zero:
  call void @foo(i1 %xz)
; CHECK: call void @foo(i1 true)
  call void @foo(i1 %yz)
; CHECK: call void @foo(i1 true)
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 0)
  call void @bar(i32 %y)
; CHECK: call void @bar(i32 0)
  ret void
nope:
  call void @foo(i1 %z)
; CHECK: call void @foo(i1 false)
  ret void
}

; CHECK-LABEL: @test3_or(
define void @test3_or(i32 %x, i32 %y) {
  %xz = icmp ne i32 %x, 0
  %yz = icmp ne i32 %y, 0
  %z = or i1 %xz, %yz
  br i1 %z, label %nope, label %both_zero
both_zero:
  call void @foo(i1 %xz)
; CHECK: call void @foo(i1 false)
  call void @foo(i1 %yz)
; CHECK: call void @foo(i1 false)
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 0)
  call void @bar(i32 %y)
; CHECK: call void @bar(i32 0)
  ret void
nope:
  call void @foo(i1 %z)
; CHECK: call void @foo(i1 true)
  ret void
}

; CHECK-LABEL: @test3_or_select(
define void @test3_or_select(i32 %x, i32 %y) {
  %xz = icmp ne i32 %x, 0
  %yz = icmp ne i32 %y, 0
  %z = select i1 %xz, i1 true, i1 %yz
  br i1 %z, label %nope, label %both_zero
both_zero:
  call void @foo(i1 %xz)
; CHECK: call void @foo(i1 false)
  call void @foo(i1 %yz)
; CHECK: call void @foo(i1 false)
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 0)
  call void @bar(i32 %y)
; CHECK: call void @bar(i32 0)
  ret void
nope:
  call void @foo(i1 %z)
; CHECK: call void @foo(i1 true)
  ret void
}

; CHECK-LABEL: @test4(
define void @test4(i1 %b, i32 %x) {
  br i1 %b, label %sw, label %case3
sw:
  switch i32 %x, label %default [
    i32 0, label %case0
    i32 1, label %case1
    i32 2, label %case0
    i32 3, label %case3
    i32 4, label %default
  ]
default:
; CHECK: default:
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 %x)
  ret void
case0:
; CHECK: case0:
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 %x)
  ret void
case1:
; CHECK: case1:
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 1)
  ret void
case3:
; CHECK: case3:
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 %x)
  ret void
}

; CHECK-LABEL: @test5(
define i1 @test5(i32 %x, i32 %y) {
  %cmp = icmp eq i32 %x, %y
  br i1 %cmp, label %same, label %different

same:
  %cmp2 = icmp ne i32 %x, %y
; CHECK: ret i1 false
  ret i1 %cmp2

different:
  %cmp3 = icmp eq i32 %x, %y
; CHECK: ret i1 false
  ret i1 %cmp3
}

; CHECK-LABEL: @test6(
define i1 @test6(i32 %x, i32 %y) {
  %cmp2 = icmp ne i32 %x, %y
  %cmp = icmp eq i32 %x, %y
  %cmp3 = icmp eq i32 %x, %y
  br i1 %cmp, label %same, label %different

same:
; CHECK: ret i1 false
  ret i1 %cmp2

different:
; CHECK: ret i1 false
  ret i1 %cmp3
}

; CHECK-LABEL: @test6_fp(
define i1 @test6_fp(float %x, float %y) {
  %cmp2 = fcmp une float %x, %y
  %cmp = fcmp oeq float %x, %y
  %cmp3 = fcmp oeq float  %x, %y
  br i1 %cmp, label %same, label %different

same:
; CHECK: ret i1 false
  ret i1 %cmp2

different:
; CHECK: ret i1 false
  ret i1 %cmp3
}

; CHECK-LABEL: @test7(
define i1 @test7(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  br i1 %cmp, label %same, label %different

same:
  %cmp2 = icmp sle i32 %x, %y
; CHECK: ret i1 false
  ret i1 %cmp2

different:
  %cmp3 = icmp sgt i32 %x, %y
; CHECK: ret i1 false
  ret i1 %cmp3
}

; CHECK-LABEL: @test7_fp(
define i1 @test7_fp(float %x, float %y) {
  %cmp = fcmp ogt float %x, %y
  br i1 %cmp, label %same, label %different

same:
  %cmp2 = fcmp ule float %x, %y
; CHECK: ret i1 false
  ret i1 %cmp2

different:
  %cmp3 = fcmp ogt float %x, %y
; CHECK: ret i1 false
  ret i1 %cmp3
}

; CHECK-LABEL: @test8(
define i1 @test8(i32 %x, i32 %y) {
  %cmp2 = icmp sle i32 %x, %y
  %cmp = icmp sgt i32 %x, %y
  %cmp3 = icmp sgt i32 %x, %y
  br i1 %cmp, label %same, label %different

same:
; CHECK: ret i1 false
  ret i1 %cmp2

different:
; CHECK: ret i1 false
  ret i1 %cmp3
}

; CHECK-LABEL: @test8_fp(
define i1 @test8_fp(float %x, float %y) {
  %cmp2 = fcmp ule float %x, %y
  %cmp = fcmp ogt float %x, %y
  %cmp3 = fcmp ogt float %x, %y
  br i1 %cmp, label %same, label %different

same:
; CHECK: ret i1 false
  ret i1 %cmp2

different:
; CHECK: ret i1 false
  ret i1 %cmp3
}

; PR1768
; CHECK-LABEL: @test9(
define i32 @test9(i32 %i, i32 %j) {
  %cmp = icmp eq i32 %i, %j
  br i1 %cmp, label %cond_true, label %ret

cond_true:
  %diff = sub i32 %i, %j
  ret i32 %diff
; CHECK: ret i32 0

ret:
  ret i32 5
; CHECK: ret i32 5
}

; PR1768
; CHECK-LABEL: @test10(
define i32 @test10(i32 %j, i32 %i) {
  %cmp = icmp eq i32 %i, %j
  br i1 %cmp, label %cond_true, label %ret

cond_true:
  %diff = sub i32 %i, %j
  ret i32 %diff
; CHECK: ret i32 0

ret:
  ret i32 5
; CHECK: ret i32 5
}

declare i32 @yogibar()

; CHECK-LABEL: @test11(
define i32 @test11(i32 %x) {
  %v0 = call i32 @yogibar()
  %v1 = call i32 @yogibar()
  %cmp = icmp eq i32 %v0, %v1
  br i1 %cmp, label %cond_true, label %next

cond_true:
  ret i32 %v1
; CHECK: ret i32 %v0

next:
  %cmp2 = icmp eq i32 %x, %v0
  br i1 %cmp2, label %cond_true2, label %next2

cond_true2:
  ret i32 %v0
; CHECK: ret i32 %x

next2:
  ret i32 0
}

; CHECK-LABEL: @test12(
define i32 @test12(i32 %x) {
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %cond_true, label %cond_false

cond_true:
  br label %ret

cond_false:
  br label %ret

ret:
  %res = phi i32 [ %x, %cond_true ], [ %x, %cond_false ]
; CHECK: %res = phi i32 [ 0, %cond_true ], [ %x, %cond_false ]
  ret i32 %res
}

; On the path from entry->if->end we know that ptr1==ptr2, so we can determine
; that gep2 does not alias ptr1 on that path (as it would require that
; ptr2==ptr2+2), so we can perform PRE of the load.
; CHECK-LABEL: @test13
define i32 @test13(i32* %ptr1, i32* %ptr2) {
; CHECK-LABEL: entry:
entry:
  %gep1 = getelementptr i32, i32* %ptr2, i32 1
  %gep2 = getelementptr i32, i32* %ptr2, i32 2
  %cmp = icmp eq i32* %ptr1, %ptr2
  br i1 %cmp, label %if, label %end

; CHECK: [[CRIT_EDGE:.*]]:
; CHECK: %[[PRE:.*]] = load i32, i32* %gep2, align 4

; CHECK-LABEL: if:
if:
  %val1 = load i32, i32* %gep2, align 4
  br label %end

; CHECK-LABEL: end:
; CHECK: %val2 = phi i32 [ %val1, %if ], [ %[[PRE]], %[[CRIT_EDGE]] ]
; CHECK-NOT: load
end:
  %phi1 = phi i32* [ %ptr1, %if ], [ %gep1, %entry ]
  %phi2 = phi i32 [ %val1, %if ], [ 0, %entry ]
  store i32 0, i32* %phi1, align 4
  %val2 = load i32, i32* %gep2, align 4
  %ret = add i32 %phi2, %val2
  ret i32 %ret
}

; CHECK-LABEL: @test14
define void @test14(i32* %ptr1, i32* noalias %ptr2) {
entry:
  %gep1 = getelementptr inbounds i32, i32* %ptr1, i32 1
  %gep2 = getelementptr inbounds i32, i32* %ptr1, i32 2
  br label %loop

; CHECK-LABEL: loop:
loop:
  %phi1 = phi i32* [ %gep3, %loop.end ], [ %gep1, %entry ]
  br i1 undef, label %if1, label %then

; CHECK: [[CRIT_EDGE:.*]]:
; CHECK: %[[PRE:.*]] = load i32, i32* %gep2, align 4

; CHECK-LABEL: if1:
; CHECK: %val2 = phi i32 [ %[[PRE]], %[[CRIT_EDGE]] ], [ %val3, %loop.end ]
; CHECK-NOT: load
if1:
  %val2 = load i32, i32* %gep2, align 4
  store i32 %val2, i32* %gep2, align 4
  store i32 0, i32* %phi1, align 4
  br label %then

; CHECK-LABEL: then:
then:
  %cmp = icmp eq i32* %gep2, %ptr2
  br i1 %cmp, label %loop.end, label %if2

if2:
  br label %loop.end

loop.end:
  %phi3 = phi i32* [ %gep2, %then ], [ %ptr1, %if2 ]
  %val3 = load i32, i32* %gep2, align 4
  store i32 %val3, i32* %phi3, align 4
  %gep3 = getelementptr inbounds i32, i32* %ptr1, i32 1
  br i1 undef, label %loop, label %if1
}
