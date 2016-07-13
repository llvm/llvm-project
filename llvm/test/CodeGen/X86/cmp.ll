; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -show-mc-encoding | FileCheck %s

define i32 @test1(i32 %X, i32* %y) nounwind {
	%tmp = load i32, i32* %y		; <i32> [#uses=1]
	%tmp.upgrd.1 = icmp eq i32 %tmp, 0		; <i1> [#uses=1]
	br i1 %tmp.upgrd.1, label %ReturnBlock, label %cond_true

cond_true:		; preds = %0
	ret i32 1

ReturnBlock:		; preds = %0
	ret i32 0
; CHECK-LABEL: test1:
; CHECK: cmpl	$0, (%rsi)
}

define i32 @test2(i32 %X, i32* %y) nounwind {
	%tmp = load i32, i32* %y		; <i32> [#uses=1]
	%tmp1 = shl i32 %tmp, 3		; <i32> [#uses=1]
	%tmp1.upgrd.2 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.2, label %ReturnBlock, label %cond_true

cond_true:		; preds = %0
	ret i32 1

ReturnBlock:		; preds = %0
	ret i32 0
; CHECK-LABEL: test2:
; CHECK: testl	$536870911, (%rsi)
}

define i8 @test2b(i8 %X, i8* %y) nounwind {
	%tmp = load i8, i8* %y		; <i8> [#uses=1]
	%tmp1 = shl i8 %tmp, 3		; <i8> [#uses=1]
	%tmp1.upgrd.2 = icmp eq i8 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.2, label %ReturnBlock, label %cond_true

cond_true:		; preds = %0
	ret i8 1

ReturnBlock:		; preds = %0
	ret i8 0
; CHECK-LABEL: test2b:
; CHECK: testb	$31, (%rsi)
}

define i64 @test3(i64 %x) nounwind {
  %t = icmp eq i64 %x, 0
  %r = zext i1 %t to i64
  ret i64 %r
; CHECK-LABEL: test3:
; CHECK:  xorl %eax, %eax
; CHECK: 	testq	%rdi, %rdi
; CHECK: 	sete	%al
; CHECK: 	ret
}

define i64 @test4(i64 %x) nounwind {
  %t = icmp slt i64 %x, 1
  %r = zext i1 %t to i64
  ret i64 %r
; CHECK-LABEL: test4:
; CHECK:  xorl %eax, %eax
; CHECK: 	testq	%rdi, %rdi
; CHECK: 	setle	%al
; CHECK: 	ret
}


define i32 @test5(double %A) nounwind  {
 entry:
 %tmp2 = fcmp ogt double %A, 1.500000e+02; <i1> [#uses=1]
 %tmp5 = fcmp ult double %A, 7.500000e+01; <i1> [#uses=1]
 %bothcond = or i1 %tmp2, %tmp5; <i1> [#uses=1]
 br i1 %bothcond, label %bb8, label %bb12

 bb8:; preds = %entry
 %tmp9 = tail call i32 (...) @foo( ) nounwind ; <i32> [#uses=1]
 ret i32 %tmp9

 bb12:; preds = %entry
 ret i32 32
; CHECK-LABEL: test5:
; CHECK: ucomisd	LCPI5_0(%rip), %xmm0
; CHECK: ucomisd	LCPI5_1(%rip), %xmm0
}

declare i32 @foo(...)

define i32 @test6() nounwind align 2 {
  %A = alloca {i64, i64}, align 8
  %B = getelementptr inbounds {i64, i64}, {i64, i64}* %A, i64 0, i32 1
  %C = load i64, i64* %B
  %D = icmp eq i64 %C, 0
  br i1 %D, label %T, label %F
T:
  ret i32 1
  
F:
  ret i32 0
; CHECK-LABEL: test6:
; CHECK: cmpq	$0, -8(%rsp)
; CHECK: encoding: [0x48,0x83,0x7c,0x24,0xf8,0x00]
}

; rdar://11866926
define i32 @test7(i64 %res) nounwind {
entry:
; CHECK-LABEL: test7:
; CHECK-NOT: movabsq
; CHECK: shrq $32, %rdi
; CHECK: sete
  %lnot = icmp ult i64 %res, 4294967296
  %lnot.ext = zext i1 %lnot to i32
  ret i32 %lnot.ext
}

define i32 @test8(i64 %res) nounwind {
entry:
; CHECK-LABEL: test8:
; CHECK-NOT: movabsq
; CHECK: shrq $32, %rdi
; CHECK: cmpq $3, %rdi
  %lnot = icmp ult i64 %res, 12884901888
  %lnot.ext = zext i1 %lnot to i32
  ret i32 %lnot.ext
}

define i32 @test9(i64 %res) nounwind {
entry:
; CHECK-LABEL: test9:
; CHECK-NOT: movabsq
; CHECK: shrq $33, %rdi
; CHECK: sete
  %lnot = icmp ult i64 %res, 8589934592
  %lnot.ext = zext i1 %lnot to i32
  ret i32 %lnot.ext
}

define i32 @test10(i64 %res) nounwind {
entry:
; CHECK-LABEL: test10:
; CHECK-NOT: movabsq
; CHECK: shrq $32, %rdi
; CHECK: setne
  %lnot = icmp uge i64 %res, 4294967296
  %lnot.ext = zext i1 %lnot to i32
  ret i32 %lnot.ext
}

; rdar://9758774
define i32 @test11(i64 %l) nounwind {
entry:
; CHECK-LABEL: test11:
; CHECK-NOT: movabsq
; CHECK-NOT: andq
; CHECK: shrq $47, %rdi
; CHECK: cmpq $1, %rdi
  %shr.mask = and i64 %l, -140737488355328
  %cmp = icmp eq i64 %shr.mask, 140737488355328
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @test12() uwtable ssp {
; CHECK-LABEL: test12:
; CHECK: testb
  %1 = call zeroext i1 @test12b()
  br i1 %1, label %2, label %3

; <label>:2                                       ; preds = %0
  ret i32 1

; <label>:3                                       ; preds = %0
  ret i32 2
}

declare zeroext i1 @test12b()

define i32 @test13(i32 %mask, i32 %base, i32 %intra) {
  %and = and i32 %mask, 8
  %tobool = icmp ne i32 %and, 0
  %cond = select i1 %tobool, i32 %intra, i32 %base
  ret i32 %cond

; CHECK-LABEL: test13:
; CHECK: testb	$8, %dil
; CHECK: cmovnel
}

define i32 @test14(i32 %mask, i32 %base, i32 %intra) #0 {
  %s = lshr i32 %mask, 7
  %tobool = icmp sgt i32 %s, -1
  %cond = select i1 %tobool, i32 %intra, i32 %base
  ret i32 %cond

; CHECK-LABEL: test14:
; CHECK: 	shrl	$7, %edi
; CHECK-NEXT: 	cmovnsl	%edx, %esi
}

; PR19964
define zeroext i1 @test15(i32 %bf.load, i32 %n) {
  %bf.lshr = lshr i32 %bf.load, 16
  %cmp2 = icmp eq i32 %bf.lshr, 0
  %cmp5 = icmp uge i32 %bf.lshr, %n
  %.cmp5 = or i1 %cmp2, %cmp5
  ret i1 %.cmp5

; CHECK-LABEL: test15:
; CHECK:  shrl	$16, %edi
; CHECK:  cmpl	%esi, %edi
}

define i8 @test16(i16 signext %L) {
  %lshr  = lshr i16 %L, 15
  %trunc = trunc i16 %lshr to i8
  %not   = xor i8 %trunc, 1
  ret i8 %not

; CHECK-LABEL: test16:
; CHECK:  testw   %di, %di
; CHECK:  setns   %al
}

define i8 @test17(i32 %L) {
  %lshr  = lshr i32 %L, 31
  %trunc = trunc i32 %lshr to i8
  %not   = xor i8 %trunc, 1
  ret i8 %not

; CHECK-LABEL: test17:
; CHECK:  testl   %edi, %edi
; CHECK:  setns   %al
}

define i8 @test18(i64 %L) {
  %lshr  = lshr i64 %L, 63
  %trunc = trunc i64 %lshr to i8
  %not   = xor i8 %trunc, 1
  ret i8 %not

; CHECK-LABEL: test18:
; CHECK:  testq   %rdi, %rdi
; CHECK:  setns   %al
}

define zeroext i1 @test19(i32 %L) {
  %lshr  = lshr i32 %L, 31
  %trunc = trunc i32 %lshr to i1
  %not   = xor i1 %trunc, 1
  ret i1 %not

; CHECK-LABEL: test19:
; CHECK:  testl   %edi, %edi
; CHECK:  setns   %al
}

@d = global i8 0, align 1

; This test failed due to incorrect handling of "shift + icmp" sequence
define void @test20(i32 %bf.load, i8 %x1, i8* %b_addr) {
  %bf.shl = shl i32 %bf.load, 8
  %bf.ashr = ashr exact i32 %bf.shl, 8
  %tobool4 = icmp ne i32 %bf.ashr, 0
  %conv = zext i1 %tobool4 to i32
  %conv6 = zext i8 %x1 to i32
  %add = add nuw nsw i32 %conv, %conv6
  %tobool7 = icmp ne i32 %add, 0
  %frombool = zext i1 %tobool7 to i8
  store i8 %frombool, i8* %b_addr, align 1
  %tobool14 = icmp ne i32 %bf.shl, 0
  %frombool15 = zext i1 %tobool14 to i8
  store i8 %frombool15, i8* @d, align 1
  ret void

; CHECK-LABEL: test20
; CHECK: andl
; CHECK: setne
; CHECK: addl
; CHECK: setne
; CHECK: testl
; CHECK: setne
}