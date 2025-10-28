; RUN: opt -passes=jump-threading -S < %s | FileCheck %s

declare i32 @f1()
declare i32 @f2()
declare void @f3()

define i32 @test1(i1 %cond) {
; CHECK-LABEL: @test1(

	br i1 %cond, label %T1, label %F1

T1:
	%v1 = call i32 @f1()
	br label %Merge

F1:
	%v2 = call i32 @f2()
	br label %Merge

Merge:
	%A = phi i1 [true, %T1], [false, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	br i1 %A, label %T2, label %F2

T2:
; CHECK: T2:
; CHECK: ret i32 %v1
	call void @f3()
	ret i32 %B

F2:
; CHECK: F2:
; CHECK: ret i32 %v2
	ret i32 %B
}


;; cond is known false on Entry -> F1 edge!
define i32 @test2(i1 %cond) {
; CHECK-LABEL: @test2(
Entry:
	br i1 %cond, label %T1, label %F1

T1:
; CHECK: %v1 = call i32 @f1()
; CHECK: ret i32 47
	%v1 = call i32 @f1()
	br label %Merge

F1:
	br i1 %cond, label %Merge, label %F2

Merge:
	%B = phi i32 [47, %T1], [192, %F1]
	ret i32 %B

F2:
	call void @f3()
	ret i32 12
}


define i32 @test3(i1 %cond) {
; CHECK-LABEL: @test3(
; CHECK-NEXT: T1:
; CHECK-NEXT: ret i32 42
	br i1 true, label %T1, label %F1

T1:
	ret i32 42

F1:
	ret i32 17
}

define i32 @test4(i1 %cond, i1 %cond2) {
; CHECK-LABEL: @test4(

	br i1 %cond, label %T1, label %F1

T1:
; CHECK:   %v1 = call i32 @f1()
; CHECK-NEXT:   br label %T

	%v1 = call i32 @f1()
	br label %Merge

F1:
	%v2 = call i32 @f2()
; CHECK:   %v2 = call i32 @f2()
; CHECK-NEXT:   br i1 %cond2,
	br label %Merge

Merge:
	%A = phi i1 [undef, %T1], [%cond2, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	br i1 %A, label %T2, label %F2

T2:
	call void @f3()
	ret i32 %B

F2:
	ret i32 %B
}


;; This tests that the branch in 'merge' can be cloned up into T1.
define i32 @test5(i1 %cond, i1 %cond2) {
; CHECK-LABEL: @test5(

	br i1 %cond, label %T1, label %F1

T1:
; CHECK: T1:
; CHECK-NEXT:   %v1 = call i32 @f1()
; CHECK-NEXT:   %cond3 = icmp eq i32 %v1, 412
; CHECK-NEXT:   br i1 %cond3, label %T2, label %F2

	%v1 = call i32 @f1()
        %cond3 = icmp eq i32 %v1, 412
	br label %Merge

F1:
	%v2 = call i32 @f2()
	br label %Merge

Merge:
	%A = phi i1 [%cond3, %T1], [%cond2, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	br i1 %A, label %T2, label %F2

T2:
	call void @f3()
	ret i32 %B

F2:
	ret i32 %B
}


;; Lexically duplicated conditionals should be threaded.


define i32 @test6(i32 %A) {
; CHECK-LABEL: @test6(
	%tmp455 = icmp eq i32 %A, 42
	br i1 %tmp455, label %BB1, label %BB2

; CHECK: call i32 @f2()
; CHECK-NEXT: ret i32 3

; CHECK: call i32 @f1()
; CHECK-NOT: br
; CHECK: call void @f3()
; CHECK-NOT: br
; CHECK: ret i32 4

BB2:
	call i32 @f1()
	br label %BB1


BB1:
	%tmp459 = icmp eq i32 %A, 42
	br i1 %tmp459, label %BB3, label %BB4

BB3:
	call i32 @f2()
        ret i32 3

BB4:
	call void @f3()
	ret i32 4
}


;; This tests that the branch in 'merge' can be cloned up into T1.
;; rdar://7367025
define i32 @test7(i1 %cond, i1 %cond2) {
Entry:
; CHECK-LABEL: @test7(
	%v1 = call i32 @f1()
	br i1 %cond, label %Merge, label %F1

F1:
	%v2 = call i32 @f2()
	br label %Merge

Merge:
	%B = phi i32 [%v1, %Entry], [%v2, %F1]
        %M = icmp ne i32 %B, %v1
        %N = icmp eq i32 %B, 47
        %O = and i1 %M, %N
	br i1 %O, label %T2, label %F2

; CHECK: Merge:
; CHECK-NOT: phi
; CHECK-NEXT:   %v2 = call i32 @f2()

T2:
	call void @f3()
	ret i32 %B

F2:
	ret i32 %B
; CHECK: F2:
; CHECK-NEXT: phi i32
}


declare i1 @test8a()

define i32 @test8b(i1 %cond, i1 %cond2) {
; CHECK-LABEL: @test8b(
T0:
        %A = call i1 @test8a()
	br i1 %A, label %T1, label %F1

; CHECK: T0:
; CHECK-NEXT: call
; CHECK-NEXT: br i1 %A, label %T1, label %Y

T1:
        %B = call i1 @test8a()
	br i1 %B, label %T2, label %F1

; CHECK: T1:
; CHECK-NEXT: call
; CHECK-NEXT: br i1 %B, label %T2, label %Y
T2:
        %C = call i1 @test8a()
	br i1 %cond, label %T3, label %F1

; CHECK: T2:
; CHECK-NEXT: call
; CHECK-NEXT: br i1 %cond, label %T3, label %Y
T3:
        ret i32 0

F1:
        %D = phi i32 [0, %T0], [0, %T1], [1, %T2]
        %E = icmp eq i32 %D, 1
        %F = and i1 %E, %cond
	br i1 %F, label %X, label %Y
X:
        call i1 @test8a()
        ret i32 1
Y:
        ret i32 2
}


;;; Verify that we can handle constraint propagation through "xor x, 1".
define i32 @test9(i1 %cond, i1 %cond2) {
Entry:
; CHECK-LABEL: @test9(
	%v1 = call i32 @f1()
	br i1 %cond, label %Merge, label %F1

; CHECK: Entry:
; CHECK-NEXT:  %v1 = call i32 @f1()
; CHECK-NEXT:  br i1 %cond, label %F2, label %Merge

F1:
	%v2 = call i32 @f2()
	br label %Merge

Merge:
	%B = phi i32 [%v1, %Entry], [%v2, %F1]
        %M = icmp eq i32 %B, %v1
        %M1 = xor i1 %M, 1
        %N = icmp eq i32 %B, 47
        %O = and i1 %M1, %N
	br i1 %O, label %T2, label %F2

; CHECK: Merge:
; CHECK-NOT: phi
; CHECK-NEXT:   %v2 = call i32 @f2()

T2:
	%Q = zext i1 %M to i32
	ret i32 %Q

F2:
	ret i32 %B
; CHECK: F2:
; CHECK-NEXT: phi i32
}



; CHECK: @test10
declare i32 @test10f1()
declare i32 @test10f2()
declare void @test10f3()

;; Non-local condition threading.
define i32 @test10g(i1 %cond) {
; CHECK-LABEL: @test10g(
; CHECK-NEXT:   br i1 %cond, label %T2, label %F2
        br i1 %cond, label %T1, label %F1

T1:
        %v1 = call i32 @test10f1()
        br label %Merge

; CHECK: %v1 = call i32 @test10f1()
; CHECK-NEXT: call void @f3()
; CHECK-NEXT: ret i32 %v1

F1:
        %v2 = call i32 @test10f2()
        br label %Merge

Merge:
        %B = phi i32 [%v1, %T1], [%v2, %F1]
        br i1 %cond, label %T2, label %F2

T2:
        call void @f3()
        ret i32 %B

F2:
        ret i32 %B
}


; Impossible conditional constraints should get threaded.  BB3 is dead here.
define i32 @test11(i32 %A) {
; CHECK-LABEL: @test11(
; CHECK-NEXT: icmp
; CHECK-NEXT: br i1 %tmp455, label %BB4, label %BB2
	%tmp455 = icmp eq i32 %A, 42
	br i1 %tmp455, label %BB1, label %BB2

BB2:
; CHECK: call i32 @f1()
; CHECK-NEXT: ret i32 %C
	%C = call i32 @f1()
	ret i32 %C


BB1:
	%tmp459 = icmp eq i32 %A, 43
	br i1 %tmp459, label %BB3, label %BB4

BB3:
	call i32 @f2()
        ret i32 3

BB4:
	call void @f3()
	ret i32 4
}

;; Correlated value through boolean expression.  GCC PR18046.
define void @test12(i32 %A) {
; CHECK-LABEL: @test12(
entry:
  %cond = icmp eq i32 %A, 0
  br i1 %cond, label %bb, label %bb1
; Should branch to the return block instead of through BB1.
; CHECK: entry:
; CHECK-NEXT: %cond = icmp eq i32 %A, 0
; CHECK-NEXT: br i1 %cond, label %bb1, label %return

bb:
  %B = call i32 @test10f2()
  br label %bb1

bb1:
  %C = phi i32 [ %A, %entry ], [ %B, %bb ]
  %cond4 = icmp eq i32 %C, 0
  br i1 %cond4, label %bb2, label %return

; CHECK: bb1:
; CHECK-NEXT: %B = call i32 @test10f2()
; CHECK-NEXT: %cond4 = icmp eq i32 %B, 0
; CHECK-NEXT: br i1 %cond4, label %bb2, label %return

bb2:
  %D = call i32 @test10f2()
  ret void

return:
  ret void
}


;; Duplicate condition to avoid xor of cond.
;; rdar://7391699
define i32 @test13(i1 %cond, i1 %cond2) {
Entry:
; CHECK-LABEL: @test13(
	%v1 = call i32 @f1()
	br i1 %cond, label %Merge, label %F1

F1:
	br label %Merge

Merge:
	%B = phi i1 [true, %Entry], [%cond2, %F1]
        %C = phi i32 [192, %Entry], [%v1, %F1]
        %M = icmp eq i32 %C, 192
        %N = xor i1 %B, %M
	br i1 %N, label %T2, label %F2

T2:
	ret i32 123

F2:
	ret i32 %v1

; CHECK:   br i1 %cond, label %F2, label %Merge

; CHECK:      Merge:
; CHECK-NEXT:   %M = icmp eq i32 %v1, 192
; CHECK-NEXT:   %N = xor i1 %cond2, %M
; CHECK-NEXT:   br i1 %N, label %T2, label %F2
}

; CHECK-LABEL: @test14(
define i32 @test14(i32 %in) {
entry:
	%A = icmp eq i32 %in, 0
; CHECK: br i1 %A, label %right_ret, label %merge
  br i1 %A, label %left, label %right

; CHECK-NOT: left:
left:
	br label %merge

; CHECK-NOT: right:
right:
  %B = call i32 @f1()
	br label %merge

merge:
; CHECK-NOT: %C = phi i32 [%in, %left], [%B, %right]
	%C = phi i32 [%in, %left], [%B, %right]
	%D = add i32 %C, 1
	%E = icmp eq i32 %D, 2
	br i1 %E, label %left_ret, label %right_ret

; CHECK: left_ret:
left_ret:
	ret i32 0

right_ret:
	ret i32 1
}

; PR5652
; CHECK-LABEL: @test15(
define i32 @test15(i32 %len) {
entry:
; CHECK: icmp ult i32 %len, 13
  %tmp = icmp ult i32 %len, 13
  br i1 %tmp, label %check, label %exit0

exit0:
  ret i32 0

check:
  %tmp9 = icmp ult i32 %len, 21
  br i1 %tmp9, label %exit1, label %exit2

exit2:
; CHECK-NOT: ret i32 2
  ret i32 2

exit1:
  ret i32 1
; CHECK: }
}

;;; Verify that we can handle constraint propagation through cast.
define i32 @test16(i1 %cond) {
Entry:
; CHECK-LABEL: @test16(
	br i1 %cond, label %Merge, label %F1

; CHECK: Entry:
; CHECK-NEXT:  br i1 %cond, label %F2, label %Merge

F1:
	%v1 = call i32 @f1()
	br label %Merge

Merge:
	%B = phi i32 [0, %Entry], [%v1, %F1]
	%M = icmp eq i32 %B, 0
	%M1 = zext i1 %M to i32
	%N = icmp eq i32 %M1, 0
	br i1 %N, label %T2, label %F2

; CHECK: Merge:
; CHECK-NOT: phi
; CHECK-NEXT:   %v1 = call i32 @f1()

T2:
	%Q = call i32 @f2()
	ret i32 %Q

F2:
	ret i32 %B
; CHECK: F2:
; CHECK-NEXT: phi i32
}

; In this test we check that block duplication is inhibited by the presence
; of a function with the 'noduplicate' attribute.

declare void @g()
declare void @j()
declare void @k()

; CHECK-LABEL: define void @h(i32 %p) {
define void @h(i32 %p) {
  %x = icmp ult i32 %p, 5
  br i1 %x, label %l1, label %l2

l1:
  call void @j()
  br label %l3

l2:
  call void @k()
  br label %l3

l3:
; CHECK: call void @g() [[$NOD:#[0-9]+]]
; CHECK-NOT: call void @g() [[$NOD]]
  call void @g() noduplicate
  %y = icmp ult i32 %p, 5
  br i1 %y, label %l4, label %l5

l4:
  call void @j()
  ret void

l5:
  call void @k()
  ret void
; CHECK: }
}

define i1 @trunc_switch(i1 %arg) {
; CHECK-LABEL: @trunc_switch
top:
; CHECK: br i1 %arg, label %exitA, label %exitB
  br i1 %arg, label %common, label %B

B:
  br label %common

common:
  %phi = phi i8 [ 2, %B ], [ 1, %top ]
  %trunc = trunc i8 %phi to i2
; CHECK-NOT: switch
  switch i2 %trunc, label %unreach [
    i2 1, label %exitA
    i2 -2, label %exitB
  ]

unreach:
  unreachable

exitA:
  ret i1 true

exitB:
  ret i1 false
}

; CHECK-LABEL: define void @h_con(i32 %p) {
define void @h_con(i32 %p) {
  %x = icmp ult i32 %p, 5
  br i1 %x, label %l1, label %l2

l1:
  call void @j()
  br label %l3

l2:
  call void @k()
  br label %l3

l3:
; CHECK: call void @g() [[$CON:#[0-9]+]]
; CHECK-NOT: call void @g() [[$CON]]
  call void @g() convergent
  %y = icmp ult i32 %p, 5
  br i1 %y, label %l4, label %l5

l4:
  call void @j()
  ret void

l5:
  call void @k()
  ret void
; CHECK: }
}


; CHECK: attributes [[$NOD]] = { noduplicate }
; CHECK: attributes [[$CON]] = { convergent }
