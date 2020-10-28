; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128:n8:16:32:64"

define i32 @test1(i32 %A, i1 %b) {
BB0:
        br i1 %b, label %BB1, label %BB2

BB1:
        ; Combine away one argument PHI nodes
        %B = phi i32 [ %A, %BB0 ]               
        ret i32 %B

BB2:
        ret i32 %A
; CHECK-LABEL: @test1(
; CHECK: BB1:
; CHECK-NEXT: ret i32 %A
}

define i32 @test2(i32 %A, i1 %b) {
BB0:
        br i1 %b, label %BB1, label %BB2

BB1:
        br label %BB2

BB2:
        ; Combine away PHI nodes with same values
        %B = phi i32 [ %A, %BB0 ], [ %A, %BB1 ]         
        ret i32 %B
; CHECK-LABEL: @test2(
; CHECK: BB2:
; CHECK-NEXT: ret i32 %A
}

define i32 @test3(i32 %A, i1 %b) {
BB0:
        br label %Loop

Loop:
        ; PHI has same value always.
        %B = phi i32 [ %A, %BB0 ], [ %B, %Loop ]
        br i1 %b, label %Loop, label %Exit

Exit:
        ret i32 %B
; CHECK-LABEL: @test3(
; CHECK: Exit:
; CHECK-NEXT: ret i32 %A
}

define i32 @test4(i1 %b) {
BB0:
        ; Loop is unreachable
        ret i32 7

Loop:           ; preds = %L2, %Loop
        ; PHI has same value always.
        %B = phi i32 [ %B, %L2 ], [ %B, %Loop ]         
        br i1 %b, label %L2, label %Loop

L2:             ; preds = %Loop
        br label %Loop
; CHECK-LABEL: @test4(
; CHECK: Loop:
; CHECK-NEXT: br i1 %b
}

define i32 @test5(i32 %A, i1 %b) {
BB0:
        br label %Loop

Loop:           ; preds = %Loop, %BB0
        ; PHI has same value always.
        %B = phi i32 [ %A, %BB0 ], [ undef, %Loop ]             
        br i1 %b, label %Loop, label %Exit

Exit:           ; preds = %Loop
        ret i32 %B
; CHECK-LABEL: @test5(
; CHECK: Loop:
; CHECK-NEXT: br i1 %b
; CHECK: Exit:
; CHECK-NEXT: ret i32 %A
}

define i32 @test6(i16 %A, i1 %b) {
BB0:
        %X = zext i16 %A to i32              
        br i1 %b, label %BB1, label %BB2

BB1:           
        %Y = zext i16 %A to i32              
        br label %BB2

BB2:           
        ;; Suck casts into phi
        %B = phi i32 [ %X, %BB0 ], [ %Y, %BB1 ]         
        ret i32 %B
; CHECK-LABEL: @test6(
; CHECK: BB2:
; CHECK: zext i16 %A to i32
; CHECK-NEXT: ret i32
}

define i32 @test7(i32 %A, i1 %b) {
BB0:
        br label %Loop

Loop:           ; preds = %Loop, %BB0
        ; PHI is dead.
        %B = phi i32 [ %A, %BB0 ], [ %C, %Loop ]                
        %C = add i32 %B, 123            
        br i1 %b, label %Loop, label %Exit

Exit:           ; preds = %Loop
        ret i32 0
; CHECK-LABEL: @test7(
; CHECK: Loop:
; CHECK-NEXT: br i1 %b
}

define i32* @test8({ i32, i32 } *%A, i1 %b) {
BB0:
        %X = getelementptr inbounds { i32, i32 }, { i32, i32 } *%A, i32 0, i32 1
        br i1 %b, label %BB1, label %BB2

BB1:
        %Y = getelementptr { i32, i32 }, { i32, i32 } *%A, i32 0, i32 1
        br label %BB2

BB2:
        ;; Suck GEPs into phi
        %B = phi i32* [ %X, %BB0 ], [ %Y, %BB1 ]
        ret i32* %B
; CHECK-LABEL: @test8(
; CHECK-NOT: phi
; CHECK: BB2:
; CHECK-NEXT: %B = getelementptr { i32, i32 }, { i32, i32 }* %A 
; CHECK-NEXT: ret i32* %B
}

define i32 @test9(i32* %A, i32* %B) {
entry:
  %c = icmp eq i32* %A, null
  br i1 %c, label %bb1, label %bb

bb:
  %C = load i32, i32* %B, align 1
  br label %bb2

bb1:
  %D = load i32, i32* %A, align 1
  br label %bb2

bb2:
  %E = phi i32 [ %C, %bb ], [ %D, %bb1 ]
  ret i32 %E
; CHECK-LABEL: @test9(
; CHECK:       bb2:
; CHECK-NEXT:        phi i32* [ %B, %bb ], [ %A, %bb1 ]
; CHECK-NEXT:   %E = load i32, i32* %{{[^,]*}}, align 1
; CHECK-NEXT:   ret i32 %E

}

define i32 @test10(i32* %A, i32* %B) {
entry:
  %c = icmp eq i32* %A, null
  br i1 %c, label %bb1, label %bb

bb:
  %C = load i32, i32* %B, align 16
  br label %bb2

bb1:
  %D = load i32, i32* %A, align 32
  br label %bb2

bb2:
  %E = phi i32 [ %C, %bb ], [ %D, %bb1 ]
  ret i32 %E
; CHECK-LABEL: @test10(
; CHECK:       bb2:
; CHECK-NEXT:        phi i32* [ %B, %bb ], [ %A, %bb1 ]
; CHECK-NEXT:   %E = load i32, i32* %{{[^,]*}}, align 16
; CHECK-NEXT:   ret i32 %E
}


; PR1777
declare i1 @test11a()

define i1 @test11() {
entry:
  %a = alloca i32
  %i = ptrtoint i32* %a to i64
  %b = call i1 @test11a()
  br i1 %b, label %one, label %two

one:
  %x = phi i64 [%i, %entry], [%y, %two]
  %c = call i1 @test11a()
  br i1 %c, label %two, label %end

two:
  %y = phi i64 [%i, %entry], [%x, %one]
  %d = call i1 @test11a()
  br i1 %d, label %one, label %end

end:
  %f = phi i64 [ %x, %one], [%y, %two]
  ; Change the %f to %i, and the optimizer suddenly becomes a lot smarter
  ; even though %f must equal %i at this point
  %g = inttoptr i64 %f to i32*
  store i32 10, i32* %g
  %z = call i1 @test11a()
  ret i1 %z
; CHECK-LABEL: @test11(
; CHECK-NOT: phi i32
; CHECK: ret i1 %z
}


define i64 @test12(i1 %cond, i8* %Ptr, i64 %Val) {
entry:
  %tmp41 = ptrtoint i8* %Ptr to i64
  %tmp42 = zext i64 %tmp41 to i128
  br i1 %cond, label %end, label %two

two:
  %tmp36 = zext i64 %Val to i128            ; <i128> [#uses=1]
  %tmp37 = shl i128 %tmp36, 64                    ; <i128> [#uses=1]
  %ins39 = or i128 %tmp42, %tmp37                 ; <i128> [#uses=1]
  br label %end

end:
  %tmp869.0 = phi i128 [ %tmp42, %entry ], [ %ins39, %two ]
  %tmp32 = trunc i128 %tmp869.0 to i64            ; <i64> [#uses=1]
  %tmp29 = lshr i128 %tmp869.0, 64                ; <i128> [#uses=1]
  %tmp30 = trunc i128 %tmp29 to i64               ; <i64> [#uses=1]

  %tmp2 = add i64 %tmp32, %tmp30
  ret i64 %tmp2
; CHECK-LABEL: @test12(
; CHECK-NOT: zext
; CHECK: end:
; CHECK-NEXT: phi i64 [ 0, %entry ], [ %Val, %two ]
; CHECK-NOT: phi
; CHECK: ret i64
}

declare void @test13f(double, i32)

define void @test13(i1 %cond, i32 %V1, double %Vald) {
entry:
  %tmp42 = zext i32 %V1 to i128
  br i1 %cond, label %end, label %two

two:
  %Val = bitcast double %Vald to i64
  %tmp36 = zext i64 %Val to i128            ; <i128> [#uses=1]
  %tmp37 = shl i128 %tmp36, 64                    ; <i128> [#uses=1]
  %ins39 = or i128 %tmp42, %tmp37                 ; <i128> [#uses=1]
  br label %end

end:
  %tmp869.0 = phi i128 [ %tmp42, %entry ], [ %ins39, %two ]
  %tmp32 = trunc i128 %tmp869.0 to i32
  %tmp29 = lshr i128 %tmp869.0, 64                ; <i128> [#uses=1]
  %tmp30 = trunc i128 %tmp29 to i64               ; <i64> [#uses=1]
  %tmp31 = bitcast i64 %tmp30 to double
  
  call void @test13f(double %tmp31, i32 %tmp32)
  ret void
; CHECK-LABEL: @test13(
; CHECK-NOT: zext
; CHECK: end:
; CHECK-NEXT: phi double [ 0.000000e+00, %entry ], [ %Vald, %two ]
; CHECK-NEXT: call void @test13f(double {{[^,]*}}, i32 %V1)
; CHECK: ret void
}

define i640 @test14a(i320 %A, i320 %B, i1 %b1) {
BB0:
        %a = zext i320 %A to i640
        %b = zext i320 %B to i640
        br label %Loop

Loop:
        %C = phi i640 [ %a, %BB0 ], [ %b, %Loop ]             
        br i1 %b1, label %Loop, label %Exit

Exit:           ; preds = %Loop
        ret i640 %C
; CHECK-LABEL: @test14a(
; CHECK: Loop:
; CHECK-NEXT: phi i320
}

define i160 @test14b(i320 %A, i320 %B, i1 %b1) {
BB0:
        %a = trunc i320 %A to i160
        %b = trunc i320 %B to i160
        br label %Loop

Loop:
        %C = phi i160 [ %a, %BB0 ], [ %b, %Loop ]             
        br i1 %b1, label %Loop, label %Exit

Exit:           ; preds = %Loop
        ret i160 %C
; CHECK-LABEL: @test14b(
; CHECK: Loop:
; CHECK-NEXT: phi i160
}

declare i64 @test15a(i64)

define i64 @test15b(i64 %A, i1 %b) {
; CHECK-LABEL: @test15b(
entry:
  %i0 = zext i64 %A to i128
  %i1 = shl i128 %i0, 64
  %i = or i128 %i1, %i0
  br i1 %b, label %one, label %two
; CHECK: entry:
; CHECK-NEXT: br i1 %b

one:
  %x = phi i128 [%i, %entry], [%y, %two]
  %x1 = lshr i128 %x, 64
  %x2 = trunc i128 %x1 to i64
  %c = call i64 @test15a(i64 %x2)
  %c1 = zext i64 %c to i128
  br label %two

; CHECK: one:
; CHECK-NEXT: phi i64
; CHECK-NEXT: %c = call i64 @test15a

two:
  %y = phi i128 [%i, %entry], [%c1, %one]
  %y1 = lshr i128 %y, 64
  %y2 = trunc i128 %y1 to i64
  %d = call i64 @test15a(i64 %y2)
  %d1 = trunc i64 %d to i1
  br i1 %d1, label %one, label %end

; CHECK: two:
; CHECK-NEXT: phi i64
; CHECK-NEXT: phi i64
; CHECK-NEXT: %d = call i64 @test15a

end:
  %g = trunc i128 %y to i64
  ret i64 %g
; CHECK: end: 
; CHECK-NEXT: ret i64
}

; PR6512 - Shouldn't merge loads from different addr spaces.
define i32 @test16(i32 addrspace(1)* %pointer1, i32 %flag, i32* %pointer2)
nounwind {
entry:
  %retval = alloca i32, align 4                   ; <i32*> [#uses=2]
  %pointer1.addr = alloca i32 addrspace(1)*, align 4 ; <i32 addrspace(1)**>
  %flag.addr = alloca i32, align 4                ; <i32*> [#uses=2]
  %pointer2.addr = alloca i32*, align 4           ; <i32**> [#uses=2]
  %res = alloca i32, align 4                      ; <i32*> [#uses=4]
  store i32 addrspace(1)* %pointer1, i32 addrspace(1)** %pointer1.addr
  store i32 %flag, i32* %flag.addr
  store i32* %pointer2, i32** %pointer2.addr
  store i32 10, i32* %res
  %tmp = load i32, i32* %flag.addr                     ; <i32> [#uses=1]
  %tobool = icmp ne i32 %tmp, 0                   ; <i1> [#uses=1]
  br i1 %tobool, label %if.then, label %if.else

return:                                           ; preds = %if.end
  %tmp7 = load i32, i32* %retval                       ; <i32> [#uses=1]
  ret i32 %tmp7

if.end:                                           ; preds = %if.else, %if.then
  %tmp6 = load i32, i32* %res                          ; <i32> [#uses=1]
  store i32 %tmp6, i32* %retval
  br label %return

if.then:                                          ; preds = %entry
  %tmp1 = load i32 addrspace(1)*, i32 addrspace(1)** %pointer1.addr  ; <i32 addrspace(1)*>
  %arrayidx = getelementptr i32, i32 addrspace(1)* %tmp1, i32 0 ; <i32 addrspace(1)*> [#uses=1]
  %tmp2 = load i32, i32 addrspace(1)* %arrayidx        ; <i32> [#uses=1]
  store i32 %tmp2, i32* %res
  br label %if.end

if.else:                                          ; preds = %entry
  %tmp3 = load i32*, i32** %pointer2.addr               ; <i32*> [#uses=1]
  %arrayidx4 = getelementptr i32, i32* %tmp3, i32 0    ; <i32*> [#uses=1]
  %tmp5 = load i32, i32* %arrayidx4                    ; <i32> [#uses=1]
  store i32 %tmp5, i32* %res
  br label %if.end
}

; PR4413
declare i32 @ext()
; CHECK-LABEL: @test17(
define i32 @test17(i1 %a) {
entry:
    br i1 %a, label %bb1, label %bb2

bb1:        ; preds = %entry
    %0 = tail call i32 @ext()        ; <i32> [#uses=1]
    br label %bb2

bb2:        ; preds = %bb1, %entry
    %cond = phi i1 [ true, %bb1 ], [ false, %entry ]        ; <i1> [#uses=1]
; CHECK-NOT: phi i1
; CHECK:     %res = phi i32 [ %0, %bb1 ], [ 0, %entry ]
; CHECK:     ret i32 %res
    %val = phi i32 [ %0, %bb1 ], [ 0, %entry ]        ; <i32> [#uses=1]
    %res = select i1 %cond, i32 %val, i32 0        ; <i32> [#uses=1]
    ret i32 %res
}

define i1 @test18(i1 %cond) {
  %zero = alloca i32
  %one = alloca i32
  br i1 %cond, label %true, label %false
true:
  br label %ret
false:
  br label %ret
ret:
  %ptr = phi i32* [ %zero, %true ] , [ %one, %false ]
  %isnull = icmp eq i32* %ptr, null
  ret i1 %isnull
; CHECK-LABEL: @test18(
; CHECK: ret i1 false
}

define i1 @test19(i1 %cond, double %x) {
  br i1 %cond, label %true, label %false
true:
  br label %ret
false:
  br label %ret
ret:
  %p = phi double [ %x, %true ], [ 0x7FF0000000000000, %false ]; RHS = +infty
  %cmp = fcmp ule double %x, %p
  ret i1 %cmp
; CHECK-LABEL: @test19(
; CHECK: ret i1 true
}

define i1 @test20(i1 %cond) {
  %a = alloca i32
  %b = alloca i32
  %c = alloca i32
  br i1 %cond, label %true, label %false
true:
  br label %ret
false:
  br label %ret
ret:
  %p = phi i32* [ %a, %true ], [ %b, %false ]
  %r = icmp eq i32* %p, %c
  ret i1 %r
; CHECK-LABEL: @test20(
; CHECK: ret i1 false
}

define i1 @test21(i1 %c1, i1 %c2) {
  %a = alloca i32
  %b = alloca i32
  %c = alloca i32
  br i1 %c1, label %true, label %false
true:
  br label %loop
false:
  br label %loop
loop:
  %p = phi i32* [ %a, %true ], [ %b, %false ], [ %p, %loop ]
  %r = icmp eq i32* %p, %c
  br i1 %c2, label %ret, label %loop
ret:
  ret i1 %r
; CHECK-LABEL: @test21(
; CHECK: ret i1 false
}

define void @test22() {
; CHECK-LABEL: @test22(
entry:
  br label %loop
loop:
  %phi = phi i32 [ 0, %entry ], [ %y, %loop ]
  %y = add i32 %phi, 1
  %o = or i32 %y, %phi
  %e = icmp eq i32 %o, %y
  br i1 %e, label %loop, label %ret
; CHECK: br i1 %e
ret:
  ret void
}

define i32 @test23(i32 %A, i1 %b, i32 * %P) {
BB0:
        br label %Loop

Loop:           ; preds = %Loop, %BB0
        ; PHI has same value always.
        %B = phi i32 [ %A, %BB0 ], [ 42, %Loop ]
        %D = add i32 %B, 19
        store i32 %D, i32* %P
        br i1 %b, label %Loop, label %Exit

Exit:           ; preds = %Loop
        %E = add i32 %B, 19
        ret i32 %E
; CHECK-LABEL: @test23(
; CHECK: %phi.bo = add i32 %A, 19
; CHECK: Loop:
; CHECK-NEXT: %B = phi i32 [ %phi.bo, %BB0 ], [ 61, %Loop ]
; CHECK: Exit:
; CHECK-NEXT: ret i32 %B
}

define i32 @test24(i32 %A, i1 %cond) {
BB0:
        %X = add nuw nsw i32 %A, 1
        br i1 %cond, label %BB1, label %BB2

BB1:
        %Y = add nuw i32 %A, 1
        br label %BB2

BB2:
        %C = phi i32 [ %X, %BB0 ], [ %Y, %BB1 ]
        ret i32 %C
; CHECK-LABEL: @test24(
; CHECK-NOT: phi
; CHECK: BB2:
; CHECK-NEXT: %C = add nuw i32 %A, 1
; CHECK-NEXT: ret i32 %C
}

; Same as test11, but used to be missed due to a bug.
declare i1 @test25a()

define i1 @test25() {
entry:
  %a = alloca i32
  %i = ptrtoint i32* %a to i64
  %b = call i1 @test25a()
  br i1 %b, label %one, label %two

one:
  %x = phi i64 [%y, %two], [%i, %entry]
  %c = call i1 @test25a()
  br i1 %c, label %two, label %end

two:
  %y = phi i64 [%x, %one], [%i, %entry]
  %d = call i1 @test25a()
  br i1 %d, label %one, label %end

end:
  %f = phi i64 [ %x, %one], [%y, %two]
  ; Change the %f to %i, and the optimizer suddenly becomes a lot smarter
  ; even though %f must equal %i at this point
  %g = inttoptr i64 %f to i32*
  store i32 10, i32* %g
  %z = call i1 @test25a()
  ret i1 %z
; CHECK-LABEL: @test25(
; CHECK-NOT: phi i32
; CHECK: ret i1 %z
}

declare i1 @test26a()

define i1 @test26(i32 %n) {
entry:
  %a = alloca i32
  %i = ptrtoint i32* %a to i64
  %b = call i1 @test26a()
  br label %one

one:
  %x = phi i64 [%y, %two], [%w, %three], [%i, %entry]
  %c = call i1 @test26a()
  switch i32 %n, label %end [
          i32 2, label %two
          i32 3, label %three
  ]

two:
  %y = phi i64 [%x, %one], [%w, %three]
  %d = call i1 @test26a()
  switch i32 %n, label %end [
          i32 10, label %one
          i32 30, label %three
  ]

three:
  %w = phi i64 [%y, %two], [%x, %one]
  %e = call i1 @test26a()
  br i1 %e, label %one, label %two

end:
  %f = phi i64 [ %x, %one], [%y, %two]
  ; Change the %f to %i, and the optimizer suddenly becomes a lot smarter
  ; even though %f must equal %i at this point
  %g = inttoptr i64 %f to i32*
  store i32 10, i32* %g
  %z = call i1 @test26a()
  ret i1 %z
; CHECK-LABEL: @test26(
; CHECK-NOT: phi i32
; CHECK: ret i1 %z
}

; CHECK-LABEL: @test27(
; CHECK: ret i32 undef
define i32 @test27(i1 %b) {
entry:
  br label %done
done:
  %y = phi i32 [ undef, %entry ]
  ret i32 %y
}

; We should be able to fold the zexts to the other side of the phi
; even though there's a constant value input to the phi. This is
; because we can shrink that constant to the smaller phi type.

define i1 @PR24766(i8 %x1, i8 %x2, i8 %condition) {
entry:
  %conv = sext i8 %condition to i32
  switch i32 %conv, label %epilog [
    i32 0, label %sw1
    i32 1, label %sw2
  ]

sw1:
  %cmp1 = icmp eq i8 %x1, %x2
  %frombool1 = zext i1 %cmp1 to i8
  br label %epilog

sw2:
  %cmp2 = icmp sle i8 %x1, %x2
  %frombool2 = zext i1 %cmp2 to i8
  br label %epilog

epilog:
  %conditionMet = phi i8 [ 0, %entry ], [ %frombool2, %sw2 ], [ %frombool1, %sw1 ]
  %tobool = icmp ne i8 %conditionMet, 0
  ret i1 %tobool

; CHECK-LABEL: @PR24766(
; CHECK: %[[RES:.*]] = phi i1 [ false, %entry ], [ %cmp2, %sw2 ], [ %cmp1, %sw1 ]
; CHECK-NEXT: ret i1 %[[RES]] 
}

; Same as above (a phi with more than 2 operands), but no constants
 
define i1 @PR24766_no_constants(i8 %x1, i8 %x2, i8 %condition, i1 %another_condition) {
entry:
  %frombool0 = zext i1 %another_condition to i8
  %conv = sext i8 %condition to i32
  switch i32 %conv, label %epilog [
    i32 0, label %sw1
    i32 1, label %sw2
  ]

sw1:
  %cmp1 = icmp eq i8 %x1, %x2
  %frombool1 = zext i1 %cmp1 to i8
  br label %epilog

sw2:
  %cmp2 = icmp sle i8 %x1, %x2
  %frombool2 = zext i1 %cmp2 to i8
  br label %epilog

epilog:
  %conditionMet = phi i8 [ %frombool0, %entry ], [ %frombool2, %sw2 ], [ %frombool1, %sw1 ]
  %tobool = icmp ne i8 %conditionMet, 0
  ret i1 %tobool

; CHECK-LABEL: @PR24766_no_constants(
; CHECK: %[[RES:.*]] = phi i1 [ %another_condition, %entry ], [ %cmp2, %sw2 ], [ %cmp1, %sw1 ]
; CHECK-NEXT: ret i1 %[[RES]]
}

; Same as above (a phi with more than 2 operands), but two constants

define i1 @PR24766_two_constants(i8 %x1, i8 %x2, i8 %condition) {
entry:
  %conv = sext i8 %condition to i32
  switch i32 %conv, label %epilog [
    i32 0, label %sw1
    i32 1, label %sw2
  ]

sw1:
  %cmp1 = icmp eq i8 %x1, %x2
  %frombool1 = zext i1 %cmp1 to i8
  br label %epilog

sw2:
  %cmp2 = icmp sle i8 %x1, %x2
  %frombool2 = zext i1 %cmp2 to i8
  br label %epilog

epilog:
  %conditionMet = phi i8 [ 0, %entry ], [ 1, %sw2 ], [ %frombool1, %sw1 ]
  %tobool = icmp ne i8 %conditionMet, 0
  ret i1 %tobool

; CHECK-LABEL: @PR24766_two_constants(
; CHECK: %[[RES:.*]] = phi i1 [ false, %entry ], [ true, %sw2 ], [ %cmp1, %sw1 ]
; CHECK-NEXT: ret i1 %[[RES]]
}

; Same as above (a phi with more than 2 operands), but two constants and two variables

define i1 @PR24766_two_constants_two_var(i8 %x1, i8 %x2, i8 %condition) {
entry:
  %conv = sext i8 %condition to i32
  switch i32 %conv, label %epilog [
    i32 0, label %sw1
    i32 1, label %sw2
    i32 2, label %sw3
  ]

sw1:
  %cmp1 = icmp eq i8 %x1, %x2
  %frombool1 = zext i1 %cmp1 to i8
  br label %epilog

sw2:
  %cmp2 = icmp sle i8 %x1, %x2
  %frombool2 = zext i1 %cmp2 to i8
  br label %epilog

sw3:
  %cmp3 = icmp sge i8 %x1, %x2
  %frombool3 = zext i1 %cmp3 to i8
  br label %epilog

epilog:
  %conditionMet = phi i8 [ 0, %entry ], [ %frombool2, %sw2 ], [ %frombool1, %sw1 ], [ 1, %sw3 ]
  %tobool = icmp ne i8 %conditionMet, 0
  ret i1 %tobool

; CHECK-LABEL: @PR24766_two_constants_two_var(
; CHECK: %[[RES:.*]] = phi i1 [ false, %entry ], [ %cmp2, %sw2 ], [ %cmp1, %sw1 ], [ true, %sw3 ]
; CHECK-NEXT: ret i1 %[[RES]]
}

; CHECK-LABEL: phi_allnonzeroconstant
; CHECK-NOT: phi i32
; CHECK: ret i1 false
define i1 @phi_allnonzeroconstant(i1 %c, i32 %a, i32 %b) {
entry:
  br i1 %c, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %if.end

if.else:                                          ; preds = %entry
  call void @dummy()

  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %x.0 = phi i32 [ 1, %if.then ], [ 2, %if.else ]
  %or = or i32 %x.0, %a
  %cmp1 = icmp eq i32 %or, 0
  ret i1 %cmp1
}

declare void @dummy()

; CHECK-LABEL: @phi_knownnonzero_eq
; CHECK-LABEL: if.then:
; CHECK-NOT: select
; CHECK-LABEL: if.end:
; CHECK: phi i32 [ 1, %if.then ]
define i1 @phi_knownnonzero_eq(i32 %n, i32 %s, i32* nocapture readonly %P) {
entry:
  %tobool = icmp slt  i32 %n, %s
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32, i32* %P
  %cmp = icmp eq i32 %n, %0
  %1 = select i1 %cmp, i32 1, i32 2
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi i32 [ %1,  %if.then ], [ %n, %entry ]
  %cmp1 = icmp eq i32 %a.0, 0
  ret i1  %cmp1
}

; CHECK-LABEL: @phi_knownnonzero_ne
; CHECK-LABEL: if.then:
; CHECK-NOT: select
; CHECK-LABEL: if.end:
; CHECK: phi i32 [ 1, %if.then ]
define i1 @phi_knownnonzero_ne(i32 %n, i32 %s, i32* nocapture readonly %P) {
entry:
  %tobool = icmp slt  i32 %n, %s
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32, i32* %P
  %cmp = icmp eq i32 %n, %0
  %1 = select i1 %cmp, i32 1, i32 2
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi i32 [ %1,  %if.then ], [ %n, %entry ]
  %cmp1 = icmp ne i32 %a.0, 0
  ret i1  %cmp1
}

; CHECK-LABEL: @phi_knownnonzero_eq_2
; CHECK-LABEL: if.then:
; CHECK-NOT: select
; CHECK-LABEL: if.end:
; CHECK: phi i32 [ 2, %if.else ]
define i1 @phi_knownnonzero_eq_2(i32 %n, i32 %s, i32* nocapture readonly %P) {
entry:
  %tobool = icmp slt  i32 %n, %s
  br i1 %tobool, label %if.then, label %if.end

if.then:
  %tobool2 = icmp slt  i32 %n, %s
  br i1 %tobool2, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  %0 = load i32, i32* %P
  %cmp = icmp eq i32 %n, %0
  %1 = select i1 %cmp, i32 1, i32 2
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi i32 [ %1,  %if.else], [ %n, %entry ], [2, %if.then]
  %cmp1 = icmp eq i32 %a.0, 0
  ret i1  %cmp1
}

; CHECK-LABEL: @phi_knownnonzero_ne_2
; CHECK-LABEL: if.then:
; CHECK-NOT: select
; CHECK-LABEL: if.end:
; CHECK: phi i32 [ 2, %if.else ]
define i1 @phi_knownnonzero_ne_2(i32 %n, i32 %s, i32* nocapture readonly %P) {
entry:
  %tobool = icmp slt  i32 %n, %s
  br i1 %tobool, label %if.then, label %if.end

if.then:
  %tobool2 = icmp slt  i32 %n, %s
  br i1 %tobool2, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  %0 = load i32, i32* %P
  %cmp = icmp eq i32 %n, %0
  %1 = select i1 %cmp, i32 1, i32 2
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi i32 [ %1,  %if.else], [ %n, %entry ], [2, %if.then]
  %cmp1 = icmp ne i32 %a.0, 0
  ret i1  %cmp1
}
