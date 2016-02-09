; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -mcpu=generic | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -mcpu=atom | FileCheck -check-prefix=ATOM %s
; PR5757

%0 = type { i64, i32 }

define i32 @test1(%0* %p, %0* %q, i1 %r) nounwind {
  %t0 = load %0, %0* %p
  %t1 = load %0, %0* %q
  %t4 = select i1 %r, %0 %t0, %0 %t1
  %t5 = extractvalue %0 %t4, 1
  ret i32 %t5
; CHECK-LABEL: test1:
; CHECK: cmovneq %rdi, %rsi
; CHECK: movl (%rsi), %eax

; ATOM-LABEL: test1:
; ATOM: cmovneq %rdi, %rsi
; ATOM: movl (%rsi), %eax
}


; PR2139
define i32 @test2() nounwind {
entry:
	%tmp73 = tail call i1 @return_false()		; <i8> [#uses=1]
	%g.0 = select i1 %tmp73, i16 0, i16 -480		; <i16> [#uses=2]
	%tmp7778 = sext i16 %g.0 to i32		; <i32> [#uses=1]
	%tmp80 = shl i32 %tmp7778, 3		; <i32> [#uses=2]
	%tmp87 = icmp sgt i32 %tmp80, 32767		; <i1> [#uses=1]
	br i1 %tmp87, label %bb90, label %bb91
bb90:		; preds = %bb84, %bb72
	unreachable
bb91:		; preds = %bb84
	ret i32 0
; CHECK-LABEL: test2:
; CHECK: cmovnew
; CHECK: cwtl

; ATOM-LABEL: test2:
; ATOM: cmovnew
; ATOM: cwtl
}

declare i1 @return_false()


;; Select between two floating point constants.
define float @test3(i32 %x) nounwind readnone {
entry:
	%0 = icmp eq i32 %x, 0		; <i1> [#uses=1]
	%iftmp.0.0 = select i1 %0, float 4.200000e+01, float 2.300000e+01		; <float> [#uses=1]
	ret float %iftmp.0.0
; CHECK-LABEL: test3:
; CHECK: movss	{{.*}},4), %xmm0

; ATOM-LABEL: test3:
; ATOM: movss  {{.*}},4), %xmm0
}

define signext i8 @test4(i8* nocapture %P, double %F) nounwind readonly {
entry:
	%0 = fcmp olt double %F, 4.200000e+01		; <i1> [#uses=1]
	%iftmp.0.0 = select i1 %0, i32 4, i32 0		; <i32> [#uses=1]
	%1 = getelementptr i8, i8* %P, i32 %iftmp.0.0		; <i8*> [#uses=1]
	%2 = load i8, i8* %1, align 1		; <i8> [#uses=1]
	ret i8 %2
; CHECK-LABEL: test4:
; CHECK: movb	({{.*}},4), %al

; ATOM-LABEL: test4:
; ATOM: movb ({{.*}},4), %al
}

define void @test5(i1 %c, <2 x i16> %a, <2 x i16> %b, <2 x i16>* %p) nounwind {
  %x = select i1 %c, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %x, <2 x i16>* %p
  ret void
; CHECK-LABEL: test5:

; ATOM-LABEL: test5:
}

define void @test6(i32 %C, <4 x float>* %A, <4 x float>* %B) nounwind {
        %tmp = load <4 x float>, <4 x float>* %A             ; <<4 x float>> [#uses=1]
        %tmp3 = load <4 x float>, <4 x float>* %B            ; <<4 x float>> [#uses=2]
        %tmp9 = fmul <4 x float> %tmp3, %tmp3            ; <<4 x float>> [#uses=1]
        %tmp.upgrd.1 = icmp eq i32 %C, 0                ; <i1> [#uses=1]
        %iftmp.38.0 = select i1 %tmp.upgrd.1, <4 x float> %tmp9, <4 x float> %tmp               ; <<4 x float>> [#uses=1]
        store <4 x float> %iftmp.38.0, <4 x float>* %A
        ret void
; Verify that the fmul gets sunk into the one part of the diamond where it is
; needed.
; CHECK-LABEL: test6:
; CHECK: je
; CHECK: ret
; CHECK: mulps
; CHECK: ret

; ATOM-LABEL: test6:
; ATOM: je
; ATOM: ret
; ATOM: mulps
; ATOM: ret
}

; Select with fp80's
define x86_fp80 @test7(i32 %tmp8) nounwind {
        %tmp9 = icmp sgt i32 %tmp8, -1          ; <i1> [#uses=1]
        %retval = select i1 %tmp9, x86_fp80 0xK4005B400000000000000, x86_fp80 0xK40078700000000000000
        ret x86_fp80 %retval
; CHECK-LABEL: test7:
; CHECK: leaq
; CHECK: fldt (%r{{.}}x,%r{{.}}x)

; ATOM-LABEL: test7:
; ATOM: leaq
; ATOM: fldt (%r{{.}}x,%r{{.}}x)
}

; widening select v6i32 and then a sub
define void @test8(i1 %c, <6 x i32>* %dst.addr, <6 x i32> %src1,<6 x i32> %src2) nounwind {
	%x = select i1 %c, <6 x i32> %src1, <6 x i32> %src2
	%val = sub <6 x i32> %x, < i32 1, i32 1, i32 1, i32 1, i32 1, i32 1 >
	store <6 x i32> %val, <6 x i32>* %dst.addr
	ret void

; CHECK-LABEL: test8:

; ATOM-LABEL: test8:
}


;; Test integer select between values and constants.

define i64 @test9(i64 %x, i64 %y) nounwind readnone ssp noredzone {
  %cmp = icmp ne i64 %x, 0
  %cond = select i1 %cmp, i64 %y, i64 -1
  ret i64 %cond
; CHECK-LABEL: test9:
; CHECK: cmpq	$1, %rdi
; CHECK: sbbq	%rax, %rax
; CHECK: orq	%rsi, %rax
; CHECK: ret

; ATOM-LABEL: test9:
; ATOM: cmpq   $1, %rdi
; ATOM: sbbq   %rax, %rax
; ATOM: orq    %rsi, %rax
; ATOM: ret
}

;; Same as test9
define i64 @test9a(i64 %x, i64 %y) nounwind readnone ssp noredzone {
  %cmp = icmp eq i64 %x, 0
  %cond = select i1 %cmp, i64 -1, i64 %y
  ret i64 %cond
; CHECK-LABEL: test9a:
; CHECK: cmpq	$1, %rdi
; CHECK: sbbq	%rax, %rax
; CHECK: orq	%rsi, %rax
; CHECK: ret

; ATOM-LABEL: test9a:
; ATOM: cmpq   $1, %rdi
; ATOM: sbbq   %rax, %rax
; ATOM: orq    %rsi, %rax
; ATOM: ret
}

define i64 @test9b(i64 %x, i64 %y) nounwind readnone ssp noredzone {
  %cmp = icmp eq i64 %x, 0
  %A = sext i1 %cmp to i64
  %cond = or i64 %y, %A
  ret i64 %cond
; CHECK-LABEL: test9b:
; CHECK: cmpq	$1, %rdi
; CHECK: sbbq	%rax, %rax
; CHECK: orq	%rsi, %rax
; CHECK: ret

; ATOM-LABEL: test9b:
; ATOM: cmpq   $1, %rdi
; ATOM: sbbq   %rax, %rax
; ATOM: orq    %rsi, %rax
; ATOM: ret
}

;; Select between -1 and 1.
define i64 @test10(i64 %x, i64 %y) nounwind readnone ssp noredzone {
  %cmp = icmp eq i64 %x, 0
  %cond = select i1 %cmp, i64 -1, i64 1
  ret i64 %cond
; CHECK-LABEL: test10:
; CHECK: cmpq	$1, %rdi
; CHECK: sbbq	%rax, %rax
; CHECK: orq	$1, %rax
; CHECK: ret

; ATOM-LABEL: test10:
; ATOM: cmpq   $1, %rdi
; ATOM: sbbq   %rax, %rax
; ATOM: orq    $1, %rax
; ATOM: ret
}



define i64 @test11(i64 %x, i64 %y) nounwind readnone ssp noredzone {
  %cmp = icmp eq i64 %x, 0
  %cond = select i1 %cmp, i64 %y, i64 -1
  ret i64 %cond
; CHECK-LABEL: test11:
; CHECK: cmpq	$1, %rdi
; CHECK: sbbq	%rax, %rax
; CHECK: notq %rax
; CHECK: orq	%rsi, %rax
; CHECK: ret

; ATOM-LABEL: test11:
; ATOM: cmpq   $1, %rdi
; ATOM: sbbq   %rax, %rax
; ATOM: notq %rax
; ATOM: orq    %rsi, %rax
; ATOM: ret
}

define i64 @test11a(i64 %x, i64 %y) nounwind readnone ssp noredzone {
  %cmp = icmp ne i64 %x, 0
  %cond = select i1 %cmp, i64 -1, i64 %y
  ret i64 %cond
; CHECK-LABEL: test11a:
; CHECK: cmpq	$1, %rdi
; CHECK: sbbq	%rax, %rax
; CHECK: notq %rax
; CHECK: orq	%rsi, %rax
; CHECK: ret

; ATOM-LABEL: test11a:
; ATOM: cmpq   $1, %rdi
; ATOM: sbbq   %rax, %rax
; ATOM: notq %rax
; ATOM: orq    %rsi, %rax
; ATOM: ret
}


declare noalias i8* @_Znam(i64) noredzone

define noalias i8* @test12(i64 %count) nounwind ssp noredzone {
entry:
  %A = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %count, i64 4)
  %B = extractvalue { i64, i1 } %A, 1
  %C = extractvalue { i64, i1 } %A, 0
  %D = select i1 %B, i64 -1, i64 %C
  %call = tail call noalias i8* @_Znam(i64 %D) nounwind noredzone
  ret i8* %call
; CHECK-LABEL: test12:
; CHECK: mulq
; CHECK: movq $-1, %[[R:r..]]
; CHECK: cmovnoq	%rax, %[[R]]
; CHECK: jmp	__Znam

; ATOM-LABEL: test12:
; ATOM: mulq
; ATOM: movq $-1, %rdi
; ATOM: cmovnoq        %rax, %rdi
; ATOM: jmp    __Znam
}

declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) nounwind readnone

define i32 @test13(i32 %a, i32 %b) nounwind {
  %c = icmp ult i32 %a, %b
  %d = sext i1 %c to i32
  ret i32 %d
; CHECK-LABEL: test13:
; CHECK: cmpl
; CHECK-NEXT: sbbl
; CHECK-NEXT: ret

; ATOM-LABEL: test13:
; ATOM: cmpl
; ATOM-NEXT: sbbl
; ATOM: ret
}

define i32 @test14(i32 %a, i32 %b) nounwind {
  %c = icmp uge i32 %a, %b
  %d = sext i1 %c to i32
  ret i32 %d
; CHECK-LABEL: test14:
; CHECK: cmpl
; CHECK-NEXT: sbbl
; CHECK-NEXT: notl
; CHECK-NEXT: ret

; ATOM-LABEL: test14:
; ATOM: cmpl
; ATOM-NEXT: sbbl
; ATOM-NEXT: notl
; ATOM: ret
}

; rdar://10961709
define i32 @test15(i32 %x) nounwind {
entry:
  %cmp = icmp ne i32 %x, 0
  %sub = sext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: test15:
; CHECK: negl
; CHECK: sbbl

; ATOM-LABEL: test15:
; ATOM: negl
; ATOM: sbbl
}

define i64 @test16(i64 %x) nounwind uwtable readnone ssp {
entry:
  %cmp = icmp ne i64 %x, 0
  %conv1 = sext i1 %cmp to i64
  ret i64 %conv1
; CHECK-LABEL: test16:
; CHECK: negq
; CHECK: sbbq

; ATOM-LABEL: test16:
; ATOM: negq
; ATOM: sbbq
}

define i16 @test17(i16 %x) nounwind {
entry:
  %cmp = icmp ne i16 %x, 0
  %sub = sext i1 %cmp to i16
  ret i16 %sub
; CHECK-LABEL: test17:
; CHECK: negw
; CHECK: sbbw

; ATOM-LABEL: test17:
; ATOM: negw
; ATOM: sbbw
}

define i8 @test18(i32 %x, i8 zeroext %a, i8 zeroext %b) nounwind {
  %cmp = icmp slt i32 %x, 15
  %sel = select i1 %cmp, i8 %a, i8 %b
  ret i8 %sel
; CHECK-LABEL: test18:
; CHECK: cmpl $15, %edi
; CHECK: cmovgel %edx

; ATOM-LABEL: test18:
; ATOM: cmpl $15, %edi
; ATOM: cmovgel %edx
}

; CHECK-LABEL: @trunc_select_miscompile
; CHECK-NOT: sarb
define i32 @trunc_select_miscompile(i32 %a, i1 zeroext %cc) {
  %tmp1 = select i1 %cc, i32 3, i32 2
  %tmp2 = shl i32 %a, %tmp1
  ret i32 %tmp2
}

define void @test19() {
; This is a massive reduction of an llvm-stress test case that generates
; interesting chains feeding setcc and eventually a f32 select operation. This
; is intended to exercise the SELECT formation in the DAG combine simplifying
; a simplified select_cc node. If it it regresses and is no longer triggering
; that code path, it can be deleted.
;
; CHECK-LABEL: @test19
; CHECK: testb
; CHECK: cmpl
; CHECK: ucomiss

BB:
  br label %CF

CF:
  %Cmp10 = icmp ule i8 undef, undef
  br i1 %Cmp10, label %CF, label %CF250

CF250:
  %E12 = extractelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 2
  %Cmp32 = icmp ugt i1 %Cmp10, false
  br i1 %Cmp32, label %CF, label %CF242

CF242:
  %Cmp38 = icmp uge i32 %E12, undef
  %FC = uitofp i1 %Cmp38 to float
  %Sl59 = select i1 %Cmp32, float %FC, float undef
  %Cmp60 = fcmp ugt float undef, undef
  br i1 %Cmp60, label %CF242, label %CF244

CF244:
  %B122 = fadd float %Sl59, undef
  ret void
}
