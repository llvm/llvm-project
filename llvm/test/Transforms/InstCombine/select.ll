; This test makes sure that these instructions are properly eliminated.
; PR1822

; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A, i32 %B) {
        %C = select i1 false, i32 %A, i32 %B            
        ret i32 %C
; CHECK: @test1
; CHECK: ret i32 %B
}

define i32 @test2(i32 %A, i32 %B) {
        %C = select i1 true, i32 %A, i32 %B             
        ret i32 %C
; CHECK: @test2
; CHECK: ret i32 %A
}


define i32 @test3(i1 %C, i32 %I) {
        ; V = I
        %V = select i1 %C, i32 %I, i32 %I               
        ret i32 %V
; CHECK: @test3
; CHECK: ret i32 %I
}

define i1 @test4(i1 %C) {
        ; V = C
        %V = select i1 %C, i1 true, i1 false            
        ret i1 %V
; CHECK: @test4
; CHECK: ret i1 %C
}

define i1 @test5(i1 %C) {
        ; V = !C
        %V = select i1 %C, i1 false, i1 true            
        ret i1 %V
; CHECK: @test5
; CHECK: xor i1 %C, true
; CHECK: ret i1
}

define i32 @test6(i1 %C) { 
        ; V = cast C to int
        %V = select i1 %C, i32 1, i32 0         
        ret i32 %V
; CHECK: @test6
; CHECK: %V = zext i1 %C to i32
; CHECK: ret i32 %V
}

define i1 @test7(i1 %C, i1 %X) {
        ; R = or C, X       
        %R = select i1 %C, i1 true, i1 %X               
        ret i1 %R
; CHECK: @test7
; CHECK: %R = or i1 %C, %X
; CHECK: ret i1 %R
}

define i1 @test8(i1 %C, i1 %X) {
        ; R = and C, X
        %R = select i1 %C, i1 %X, i1 false              
        ret i1 %R
; CHECK: @test8
; CHECK: %R = and i1 %C, %X
; CHECK: ret i1 %R
}

define i1 @test9(i1 %C, i1 %X) {
        ; R = and !C, X
        %R = select i1 %C, i1 false, i1 %X              
        ret i1 %R
; CHECK: @test9
; CHECK: xor i1 %C, true
; CHECK: %R = and i1
; CHECK: ret i1 %R
}

define i1 @test10(i1 %C, i1 %X) {
        ; R = or !C, X
        %R = select i1 %C, i1 %X, i1 true               
        ret i1 %R
; CHECK: @test10
; CHECK: xor i1 %C, true
; CHECK: %R = or i1
; CHECK: ret i1 %R
}

define i32 @test11(i32 %a) {
        %C = icmp eq i32 %a, 0          
        %R = select i1 %C, i32 0, i32 1         
        ret i32 %R
; CHECK: @test11
; CHECK: icmp ne i32 %a, 0
; CHECK: %R = zext i1
; CHECK: ret i32 %R
}

define i32 @test12(i1 %cond, i32 %a) {
        %b = or i32 %a, 1               
        %c = select i1 %cond, i32 %b, i32 %a            
        ret i32 %c
; CHECK: @test12
; CHECK: %b = zext i1 %cond to i32
; CHECK: %c = or i32 %b, %a
; CHECK: ret i32 %c
}

define i32 @test12a(i1 %cond, i32 %a) {
        %b = ashr i32 %a, 1             
        %c = select i1 %cond, i32 %b, i32 %a            
        ret i32 %c
; CHECK: @test12a
; CHECK: %b = zext i1 %cond to i32
; CHECK: %c = ashr i32 %a, %b
; CHECK: ret i32 %c
}

define i32 @test12b(i1 %cond, i32 %a) {
        %b = ashr i32 %a, 1             
        %c = select i1 %cond, i32 %a, i32 %b            
        ret i32 %c
; CHECK: @test12b
; CHECK: zext i1 %cond to i32
; CHECK: %b = xor i32
; CHECK: %c = ashr i32 %a, %b
; CHECK: ret i32 %c
}

define i32 @test13(i32 %a, i32 %b) {
        %C = icmp eq i32 %a, %b         
        %V = select i1 %C, i32 %a, i32 %b               
        ret i32 %V
; CHECK: @test13
; CHECK: ret i32 %b
}

define i32 @test13a(i32 %a, i32 %b) {
        %C = icmp ne i32 %a, %b         
        %V = select i1 %C, i32 %a, i32 %b               
        ret i32 %V
; CHECK: @test13a
; CHECK: ret i32 %a
}

define i32 @test13b(i32 %a, i32 %b) {
        %C = icmp eq i32 %a, %b         
        %V = select i1 %C, i32 %b, i32 %a               
        ret i32 %V
; CHECK: @test13b
; CHECK: ret i32 %a
}

define i1 @test14a(i1 %C, i32 %X) {
        %V = select i1 %C, i32 %X, i32 0                
        ; (X < 1) | !C
        %R = icmp slt i32 %V, 1         
        ret i1 %R
; CHECK: @test14a
; CHECK: icmp slt i32 %X, 1
; CHECK: xor i1 %C, true
; CHECK: or i1
; CHECK: ret i1 %R
}

define i1 @test14b(i1 %C, i32 %X) {
        %V = select i1 %C, i32 0, i32 %X                
        ; (X < 1) | C
        %R = icmp slt i32 %V, 1         
        ret i1 %R
; CHECK: @test14b
; CHECK: icmp slt i32 %X, 1
; CHECK: or i1
; CHECK: ret i1 %R
}

;; Code sequence for (X & 16) ? 16 : 0
define i32 @test15a(i32 %X) {
        %t1 = and i32 %X, 16            
        %t2 = icmp eq i32 %t1, 0                
        %t3 = select i1 %t2, i32 0, i32 16              
        ret i32 %t3
; CHECK: @test15a
; CHECK: %t1 = and i32 %X, 16
; CHECK: ret i32 %t1
}

;; Code sequence for (X & 32) ? 0 : 24
define i32 @test15b(i32 %X) {
        %t1 = and i32 %X, 32            
        %t2 = icmp eq i32 %t1, 0                
        %t3 = select i1 %t2, i32 32, i32 0              
        ret i32 %t3
; CHECK: @test15b
; CHECK: %t1 = and i32 %X, 32
; CHECK: xor i32 %t1, 32
; CHECK: ret i32
}

;; Alternate code sequence for (X & 16) ? 16 : 0
define i32 @test15c(i32 %X) {
        %t1 = and i32 %X, 16            
        %t2 = icmp eq i32 %t1, 16               
        %t3 = select i1 %t2, i32 16, i32 0              
        ret i32 %t3
; CHECK: @test15c
; CHECK: %t1 = and i32 %X, 16
; CHECK: ret i32 %t1
}

;; Alternate code sequence for (X & 16) ? 16 : 0
define i32 @test15d(i32 %X) {
        %t1 = and i32 %X, 16            
        %t2 = icmp ne i32 %t1, 0                
        %t3 = select i1 %t2, i32 16, i32 0              
        ret i32 %t3
; CHECK: @test15d
; CHECK: %t1 = and i32 %X, 16
; CHECK: ret i32 %t1
}

;; (a & 128) ? 256 : 0
define i32 @test15e(i32 %X) {
        %t1 = and i32 %X, 128
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 256, i32 0
        ret i32 %t3
; CHECK: @test15e
; CHECK: %t1 = shl i32 %X, 1
; CHECK: and i32 %t1, 256
; CHECK: ret i32
}

;; (a & 128) ? 0 : 256
define i32 @test15f(i32 %X) {
        %t1 = and i32 %X, 128
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 0, i32 256
        ret i32 %t3
; CHECK: @test15f
; CHECK: %t1 = shl i32 %X, 1
; CHECK: and i32 %t1, 256
; CHECK: xor i32 %{{.*}}, 256
; CHECK: ret i32
}

;; (a & 8) ? -1 : -9
define i32 @test15g(i32 %X) {
        %t1 = and i32 %X, 8
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 -1, i32 -9
        ret i32 %t3
; CHECK: @test15g
; CHECK-NEXT: %1 = or i32 %X, -9
; CHECK-NEXT: ret i32 %1
}

;; (a & 8) ? -9 : -1
define i32 @test15h(i32 %X) {
        %t1 = and i32 %X, 8
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 -9, i32 -1
        ret i32 %t3
; CHECK: @test15h
; CHECK-NEXT: %1 = or i32 %X, -9
; CHECK-NEXT: %2 = xor i32 %1, 8
; CHECK-NEXT: ret i32 %2
}

;; (a & 2) ? 577 : 1089
define i32 @test15i(i32 %X) {
        %t1 = and i32 %X, 2
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 577, i32 1089
        ret i32 %t3
; CHECK: @test15i
; CHECK-NEXT: %t1 = shl i32 %X, 8
; CHECK-NEXT: %1 = and i32 %t1, 512
; CHECK-NEXT: %2 = xor i32 %1, 512
; CHECK-NEXT: %3 = add i32 %2, 577
; CHECK-NEXT: ret i32 %3
}

;; (a & 2) ? 1089 : 577
define i32 @test15j(i32 %X) {
        %t1 = and i32 %X, 2
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 1089, i32 577
        ret i32 %t3
; CHECK: @test15j
; CHECK-NEXT: %t1 = shl i32 %X, 8
; CHECK-NEXT: %1 = and i32 %t1, 512
; CHECK-NEXT: %2 = add i32 %1, 577
; CHECK-NEXT: ret i32 %2
}

define i32 @test16(i1 %C, i32* %P) {
        %P2 = select i1 %C, i32* %P, i32* null          
        %V = load i32* %P2              
        ret i32 %V
; CHECK: @test16
; CHECK-NEXT: %V = load i32* %P
; CHECK: ret i32 %V
}

define i1 @test17(i32* %X, i1 %C) {
        %R = select i1 %C, i32* %X, i32* null           
        %RV = icmp eq i32* %R, null             
        ret i1 %RV
; CHECK: @test17
; CHECK: icmp eq i32* %X, null
; CHECK: xor i1 %C, true
; CHECK: %RV = or i1
; CHECK: ret i1 %RV
}

define i32 @test18(i32 %X, i32 %Y, i1 %C) {
        %R = select i1 %C, i32 %X, i32 0                
        %V = sdiv i32 %Y, %R            
        ret i32 %V
; CHECK: @test18
; CHECK: %V = sdiv i32 %Y, %X
; CHECK: ret i32 %V
}

define i32 @test19(i32 %x) {
        %tmp = icmp ugt i32 %x, 2147483647              
        %retval = select i1 %tmp, i32 -1, i32 0         
        ret i32 %retval
; CHECK: @test19
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: ret i32 
}

define i32 @test20(i32 %x) {
        %tmp = icmp slt i32 %x, 0               
        %retval = select i1 %tmp, i32 -1, i32 0         
        ret i32 %retval
; CHECK: @test20
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: ret i32 
}

define i64 @test21(i32 %x) {
        %tmp = icmp slt i32 %x, 0               
        %retval = select i1 %tmp, i64 -1, i64 0         
        ret i64 %retval
; CHECK: @test21
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: sext i32 
; CHECK-NEXT: ret i64
}

define i16 @test22(i32 %x) {
        %tmp = icmp slt i32 %x, 0               
        %retval = select i1 %tmp, i16 -1, i16 0         
        ret i16 %retval
; CHECK: @test22
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: trunc i32 
; CHECK-NEXT: ret i16
}

define i1 @test23(i1 %a, i1 %b) {
        %c = select i1 %a, i1 %b, i1 %a         
        ret i1 %c
; CHECK: @test23
; CHECK-NEXT: %c = and i1 %a, %b
; CHECK-NEXT: ret i1 %c
}

define i1 @test24(i1 %a, i1 %b) {
        %c = select i1 %a, i1 %a, i1 %b         
        ret i1 %c
; CHECK: @test24
; CHECK-NEXT: %c = or i1 %a, %b
; CHECK-NEXT: ret i1 %c
}

define i32 @test25(i1 %c)  {
entry:
  br i1 %c, label %jump, label %ret
jump:
  br label %ret 
ret:
  %a = phi i1 [true, %jump], [false, %entry]
  %b = select i1 %a, i32 10, i32 20
  ret i32 %b
; CHECK: @test25
; CHECK: %a = phi i32 [ 10, %jump ], [ 20, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test26(i1 %cond)  {
entry:
  br i1 %cond, label %jump, label %ret
jump:
  %c = or i1 false, false
  br label %ret 
ret:
  %a = phi i1 [true, %jump], [%c, %entry]
  %b = select i1 %a, i32 10, i32 20
  ret i32 %b
; CHECK: @test26
; CHECK: %a = phi i32 [ 10, %jump ], [ 20, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test27(i1 %c, i32 %A, i32 %B)  {
entry:
  br i1 %c, label %jump, label %ret
jump:
  br label %ret 
ret:
  %a = phi i1 [true, %jump], [false, %entry]
  %b = select i1 %a, i32 %A, i32 %B
  ret i32 %b
; CHECK: @test27
; CHECK: %a = phi i32 [ %A, %jump ], [ %B, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test28(i1 %cond, i32 %A, i32 %B)  {
entry:
  br i1 %cond, label %jump, label %ret
jump:
  br label %ret 
ret:
  %c = phi i32 [%A, %jump], [%B, %entry]
  %a = phi i1 [true, %jump], [false, %entry]
  %b = select i1 %a, i32 %A, i32 %c
  ret i32 %b
; CHECK: @test28
; CHECK: %a = phi i32 [ %A, %jump ], [ %B, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test29(i1 %cond, i32 %A, i32 %B)  {
entry:
  br i1 %cond, label %jump, label %ret
jump:
  br label %ret 
ret:
  %c = phi i32 [%A, %jump], [%B, %entry]
  %a = phi i1 [true, %jump], [false, %entry]
  br label %next
  
next:
  %b = select i1 %a, i32 %A, i32 %c
  ret i32 %b
; CHECK: @test29
; CHECK: %a = phi i32 [ %A, %jump ], [ %B, %entry ]
; CHECK: ret i32 %a
}


; SMAX(SMAX(x, y), x) -> SMAX(x, y)
define i32 @test30(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %x, i32 %y
  
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x
  ret i32 %retval
; CHECK: @test30
; CHECK: ret i32 %cond
}

; UMAX(UMAX(x, y), x) -> UMAX(x, y)
define i32 @test31(i32 %x, i32 %y) {
  %cmp = icmp ugt i32 %x, %y 
  %cond = select i1 %cmp, i32 %x, i32 %y
  %cmp5 = icmp ugt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x
  ret i32 %retval
; CHECK: @test31
; CHECK: ret i32 %cond
}

; SMIN(SMIN(x, y), x) -> SMIN(x, y)
define i32 @test32(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %y, i32 %x
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %x, i32 %cond
  ret i32 %retval
; CHECK: @test32
; CHECK: ret i32 %cond
}

; MAX(MIN(x, y), x) -> x
define i32 @test33(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %y, i32 %x
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x
  ret i32 %retval
; CHECK: @test33
; CHECK: ret i32 %x
}

; MIN(MAX(x, y), x) -> x
define i32 @test34(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %x, i32 %y
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %x, i32 %cond
  ret i32 %retval
; CHECK: @test34
; CHECK: ret i32 %x
}

define i32 @test35(i32 %x) {
  %cmp = icmp sge i32 %x, 0
  %cond = select i1 %cmp, i32 60, i32 100
  ret i32 %cond
; CHECK: @test35
; CHECK: ashr i32 %x, 31
; CHECK: and i32 {{.*}}, 40
; CHECK: add i32 {{.*}}, 60
; CHECK: ret
}

define i32 @test36(i32 %x) {
  %cmp = icmp slt i32 %x, 0
  %cond = select i1 %cmp, i32 60, i32 100
  ret i32 %cond
; CHECK: @test36
; CHECK: ashr i32 %x, 31
; CHECK: and i32 {{.*}}, -40
; CHECK: add i32 {{.*}}, 100
; CHECK: ret
}

define i32 @test37(i32 %x) {
  %cmp = icmp sgt i32 %x, -1
  %cond = select i1 %cmp, i32 1, i32 -1
  ret i32 %cond
; CHECK: @test37
; CHECK: ashr i32 %x, 31
; CHECK: or i32 {{.*}}, 1
; CHECK: ret
}

define i1 @test38(i1 %cond) {
  %zero = alloca i32
  %one = alloca i32
  %ptr = select i1 %cond, i32* %zero, i32* %one
  %isnull = icmp eq i32* %ptr, null
  ret i1 %isnull
; CHECK: @test38
; CHECK: ret i1 false
}

define i1 @test39(i1 %cond, double %x) {
  %s = select i1 %cond, double %x, double 0x7FF0000000000000 ; RHS = +infty
  %cmp = fcmp ule double %x, %s
  ret i1 %cmp
; CHECK: @test39
; CHECK: ret i1 true
}

define i1 @test40(i1 %cond) {
  %a = alloca i32
  %b = alloca i32
  %c = alloca i32
  %s = select i1 %cond, i32* %a, i32* %b
  %r = icmp eq i32* %s, %c
  ret i1 %r
; CHECK: @test40
; CHECK: ret i1 false
}

define i32 @test41(i1 %cond, i32 %x, i32 %y) {
  %z = and i32 %x, %y
  %s = select i1 %cond, i32 %y, i32 %z
  %r = and i32 %x, %s
  ret i32 %r
; CHECK: @test41
; CHECK-NEXT: and i32 %x, %y
; CHECK-NEXT: ret i32
}

define i32 @test42(i32 %x, i32 %y) {
  %b = add i32 %y, -1
  %cond = icmp eq i32 %x, 0
  %c = select i1 %cond, i32 %b, i32 %y
  ret i32 %c
; CHECK: @test42
; CHECK-NEXT: %cond = icmp eq i32 %x, 0
; CHECK-NEXT: %b = sext i1 %cond to i32
; CHECK-NEXT: %c = add i32 %b, %y
; CHECK-NEXT: ret i32 %c
}

define i64 @test43(i32 %a) nounwind {
	%a_ext = sext i32 %a to i64
	%is_a_nonnegative = icmp sgt i32 %a, -1
	%max = select i1 %is_a_nonnegative, i64 %a_ext, i64 0
	ret i64 %max
; CHECK: @test43
; CHECK-NEXT: %a_ext = sext i32 %a to i64
; CHECK-NEXT: %is_a_nonnegative = icmp slt i64 %a_ext, 0
; CHECK-NEXT: %max = select i1 %is_a_nonnegative, i64 0, i64 %a_ext
; CHECK-NEXT: ret i64 %max
}

define i64 @test44(i32 %a) nounwind {
	%a_ext = sext i32 %a to i64
	%is_a_nonpositive = icmp slt i32 %a, 1
	%min = select i1 %is_a_nonpositive, i64 %a_ext, i64 0
	ret i64 %min
; CHECK: @test44
; CHECK-NEXT: %a_ext = sext i32 %a to i64
; CHECK-NEXT: %is_a_nonpositive = icmp sgt i64 %a_ext, 0
; CHECK-NEXT: %min = select i1 %is_a_nonpositive, i64 0, i64 %a_ext
; CHECK-NEXT: ret i64 %min
}
define i64 @test45(i32 %a) nounwind {
	%a_ext = zext i32 %a to i64
	%is_a_nonnegative = icmp ugt i32 %a, 2
	%max = select i1 %is_a_nonnegative, i64 %a_ext, i64 3
	ret i64 %max
; CHECK: @test45
; CHECK-NEXT: %a_ext = zext i32 %a to i64
; CHECK-NEXT: %is_a_nonnegative = icmp ult i64 %a_ext, 3
; CHECK-NEXT: %max = select i1 %is_a_nonnegative, i64 3, i64 %a_ext
; CHECK-NEXT: ret i64 %max
}

define i64 @test46(i32 %a) nounwind {
	%a_ext = zext i32 %a to i64
	%is_a_nonpositive = icmp ult i32 %a, 3
	%min = select i1 %is_a_nonpositive, i64 %a_ext, i64 2
	ret i64 %min
; CHECK: @test46
; CHECK-NEXT: %a_ext = zext i32 %a to i64
; CHECK-NEXT: %is_a_nonpositive = icmp ugt i64 %a_ext, 2
; CHECK-NEXT: %min = select i1 %is_a_nonpositive, i64 2, i64 %a_ext
; CHECK-NEXT: ret i64 %min
}
define i64 @test47(i32 %a) nounwind {
	%a_ext = sext i32 %a to i64
	%is_a_nonnegative = icmp ugt i32 %a, 2
	%max = select i1 %is_a_nonnegative, i64 %a_ext, i64 3
	ret i64 %max
; CHECK: @test47
; CHECK-NEXT: %a_ext = sext i32 %a to i64
; CHECK-NEXT: %is_a_nonnegative = icmp ult i64 %a_ext, 3
; CHECK-NEXT: %max = select i1 %is_a_nonnegative, i64 3, i64 %a_ext
; CHECK-NEXT: ret i64 %max
}

define i64 @test48(i32 %a) nounwind {
	%a_ext = sext i32 %a to i64
	%is_a_nonpositive = icmp ult i32 %a, 3
	%min = select i1 %is_a_nonpositive, i64 %a_ext, i64 2
	ret i64 %min
; CHECK: @test48
; CHECK-NEXT: %a_ext = sext i32 %a to i64
; CHECK-NEXT: %is_a_nonpositive = icmp ugt i64 %a_ext, 2
; CHECK-NEXT: %min = select i1 %is_a_nonpositive, i64 2, i64 %a_ext
; CHECK-NEXT: ret i64 %min
}

define i64 @test49(i32 %a) nounwind {
	%a_ext = sext i32 %a to i64
	%is_a_nonpositive = icmp ult i32 %a, 3
	%min = select i1 %is_a_nonpositive, i64 2, i64 %a_ext
	ret i64 %min
; CHECK: @test49
; CHECK-NEXT: %a_ext = sext i32 %a to i64
; CHECK-NEXT: %is_a_nonpositive = icmp ugt i64 %a_ext, 2
; CHECK-NEXT: %min = select i1 %is_a_nonpositive, i64 %a_ext, i64 2
; CHECK-NEXT: ret i64 %min
}
define i64 @test50(i32 %a) nounwind {
	%is_a_nonpositive = icmp ult i32 %a, 3
	%a_ext = sext i32 %a to i64
	%min = select i1 %is_a_nonpositive, i64 2, i64 %a_ext
	ret i64 %min
; CHECK: @test50
; CHECK-NEXT: %a_ext = sext i32 %a to i64
; CHECK-NEXT: %is_a_nonpositive = icmp ugt i64 %a_ext, 2
; CHECK-NEXT: %min = select i1 %is_a_nonpositive, i64 %a_ext, i64 2
; CHECK-NEXT: ret i64 %min
}

; PR8994

; This select instruction can't be eliminated because trying to do so would
; change the number of vector elements. This used to assert.
define i48 @test51(<3 x i1> %icmp, <3 x i16> %tmp) {
; CHECK: @test51
  %select = select <3 x i1> %icmp, <3 x i16> zeroinitializer, <3 x i16> %tmp
; CHECK: select <3 x i1>
  %tmp2 = bitcast <3 x i16> %select to i48
  ret i48 %tmp2
}

; PR8575

define i32 @test52(i32 %n, i32 %m) nounwind {
; CHECK: @test52
  %cmp = icmp sgt i32 %n, %m
  %. = select i1 %cmp, i32 1, i32 3
  %add = add nsw i32 %., 3
  %storemerge = select i1 %cmp, i32 %., i32 %add
; CHECK: select i1 %cmp, i32 1, i32 6
  ret i32 %storemerge
}

; PR9454
define i32 @test53(i32 %x) nounwind {
  %and = and i32 %x, 2
  %cmp = icmp eq i32 %and, %x
  %sel = select i1 %cmp, i32 2, i32 1
  ret i32 %sel
; CHECK: @test53
; CHECK: select i1 %cmp
; CHECK: ret
}

define i32 @test54(i32 %X, i32 %Y) {
  %A = ashr exact i32 %X, %Y
  %B = icmp eq i32 %A, 0
  %C = select i1 %B, i32 %A, i32 1
  ret i32 %C
; CHECK: @test54
; CHECK-NOT: ashr
; CHECK-NOT: select
; CHECK: icmp ne i32 %X, 0
; CHECK: zext 
; CHECK: ret
}

define i1 @test55(i1 %X, i32 %Y, i32 %Z) {
  %A = ashr exact i32 %Y, %Z
  %B = select i1 %X, i32 %Y, i32 %A
  %C = icmp eq i32 %B, 0
  ret i1 %C
; CHECK: @test55
; CHECK-NOT: ashr
; CHECK-NOT: select
; CHECK: icmp eq
; CHECK: ret i1
}

define i32 @test56(i16 %x) nounwind {
  %tobool = icmp eq i16 %x, 0
  %conv = zext i16 %x to i32
  %cond = select i1 %tobool, i32 0, i32 %conv
  ret i32 %cond
; CHECK: @test56
; CHECK-NEXT: zext
; CHECK-NEXT: ret
}

define i32 @test57(i32 %x, i32 %y) nounwind {
  %and = and i32 %x, %y
  %tobool = icmp eq i32 %x, 0
  %.and = select i1 %tobool, i32 0, i32 %and
  ret i32 %.and
; CHECK: @test57
; CHECK-NEXT: and i32 %x, %y
; CHECK-NEXT: ret
}

define i32 @test58(i16 %x) nounwind {
  %tobool = icmp ne i16 %x, 1
  %conv = zext i16 %x to i32
  %cond = select i1 %tobool, i32 %conv, i32 1
  ret i32 %cond
; CHECK: @test58
; CHECK-NEXT: zext
; CHECK-NEXT: ret
}

define i32 @test59(i32 %x, i32 %y) nounwind {
  %and = and i32 %x, %y
  %tobool = icmp ne i32 %x, %y
  %.and = select i1 %tobool, i32 %and, i32 %y
  ret i32 %.and
; CHECK: @test59
; CHECK-NEXT: and i32 %x, %y
; CHECK-NEXT: ret
}

define i1 @test60(i32 %x, i1* %y) nounwind {
  %cmp = icmp eq i32 %x, 0
  %load = load i1* %y, align 1
  %cmp1 = icmp slt i32 %x, 1
  %sel = select i1 %cmp, i1 %load, i1 %cmp1
  ret i1 %sel
; CHECK: @test60
; CHECK: select
}

@glbl = constant i32 10
define i32 @test61(i32* %ptr) {
  %A = load i32* %ptr
  %B = icmp eq i32* %ptr, @glbl
  %C = select i1 %B, i32 %A, i32 10
  ret i32 %C
; CHECK: @test61
; CHECK: ret i32 10
}

define i1 @test62(i1 %A, i1 %B) {
        %not = xor i1 %A, true
        %C = select i1 %A, i1 %not, i1 %B             
        ret i1 %C
; CHECK: @test62
; CHECK: %not = xor i1 %A, true
; CHECK: %C = and i1 %not, %B
; CHECK: ret i1 %C
}

define i1 @test63(i1 %A, i1 %B) {
        %not = xor i1 %A, true
        %C = select i1 %A, i1 %B, i1 %not         
        ret i1 %C
; CHECK: @test63
; CHECK: %not = xor i1 %A, true
; CHECK: %C = or i1 %B, %not
; CHECK: ret i1 %C
}

; PR14131
define void @test64(i32 %p, i16 %b) noreturn nounwind {
entry:
  %p.addr.0.insert.mask = and i32 %p, -65536
  %conv2 = and i32 %p, 65535
  br i1 undef, label %lor.rhs, label %lor.end

lor.rhs:
  %p.addr.0.extract.trunc = trunc i32 %p.addr.0.insert.mask to i16
  %phitmp = zext i16 %p.addr.0.extract.trunc to i32
  br label %lor.end

lor.end:
  %t.1 = phi i32 [ 0, %entry ], [ %phitmp, %lor.rhs ]
  %conv6 = zext i16 %b to i32
  %div = udiv i32 %conv6, %t.1
  %tobool8 = icmp eq i32 %div, 0
  %cmp = icmp eq i32 %t.1, 0
  %cmp12 = icmp ult i32 %conv2, 2
  %cmp.sink = select i1 %tobool8, i1 %cmp12, i1 %cmp
  br i1 %cmp.sink, label %cond.end17, label %cond.false16

cond.false16:
  br label %cond.end17

cond.end17:
  br label %while.body

while.body:
  br label %while.body
; CHECK: @test64
; CHECK-NOT: select
}
