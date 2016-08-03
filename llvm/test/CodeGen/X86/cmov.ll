; RUN: llc < %s -verify-machineinstrs -mtriple=x86_64-apple-darwin10 -disable-cgp-select2branch | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define i32 @test1(i32 %x, i32 %n, i32 %w, i32* %vp) nounwind readnone {
entry:
; CHECK-LABEL: test1:
; CHECK: btl
; CHECK-NEXT: movl	$12, %eax
; CHECK-NEXT: cmovael	(%rcx), %eax
; CHECK-NEXT: ret

	%0 = lshr i32 %x, %n		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %1, 0		; <i1> [#uses=1]
        %v = load i32, i32* %vp
	%.0 = select i1 %toBool, i32 %v, i32 12		; <i32> [#uses=1]
	ret i32 %.0
}
define i32 @test2(i32 %x, i32 %n, i32 %w, i32* %vp) nounwind readnone {
entry:
; CHECK-LABEL: test2:
; CHECK: btl
; CHECK-NEXT: movl	$12, %eax
; CHECK-NEXT: cmovbl	(%rcx), %eax
; CHECK-NEXT: ret

	%0 = lshr i32 %x, %n		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %1, 0		; <i1> [#uses=1]
        %v = load i32, i32* %vp
	%.0 = select i1 %toBool, i32 12, i32 %v		; <i32> [#uses=1]
	ret i32 %.0
}


; x86's 32-bit cmov zeroes the high 32 bits of the destination. Make
; sure CodeGen takes advantage of that to avoid an unnecessary
; zero-extend (movl) after the cmov.

declare void @bar(i64) nounwind

define void @test3(i64 %a, i64 %b, i1 %p) nounwind {
; CHECK-LABEL: test3:
; CHECK:      cmov{{n?}}el %[[R1:e..]], %[[R2:e..]]
; CHECK-NOT:  movl
; CHECK:      call

  %c = trunc i64 %a to i32
  %d = trunc i64 %b to i32
  %e = select i1 %p, i32 %c, i32 %d
  %f = zext i32 %e to i64
  call void @bar(i64 %f)
  ret void
}



; CodeGen shouldn't try to do a setne after an expanded 8-bit conditional
; move without recomputing EFLAGS, because the expansion of the conditional
; move with control flow may clobber EFLAGS (e.g., with xor, to set the
; register to zero).

; The test is a little awkward; the important part is that there's a test before the
; setne.
; PR4814


@g_3 = external global i8                         ; <i8*> [#uses=1]
@g_96 = external global i8                        ; <i8*> [#uses=2]
@g_100 = external global i8                       ; <i8*> [#uses=2]
@_2E_str = external constant [15 x i8], align 1   ; <[15 x i8]*> [#uses=1]

define i32 @test4() nounwind {
entry:
  %0 = load i8, i8* @g_3, align 1                     ; <i8> [#uses=2]
  %1 = sext i8 %0 to i32                          ; <i32> [#uses=1]
  %.lobit.i = lshr i8 %0, 7                       ; <i8> [#uses=1]
  %tmp.i = zext i8 %.lobit.i to i32               ; <i32> [#uses=1]
  %tmp.not.i = xor i32 %tmp.i, 1                  ; <i32> [#uses=1]
  %iftmp.17.0.i.i = ashr i32 %1, %tmp.not.i       ; <i32> [#uses=1]
  %retval56.i.i = trunc i32 %iftmp.17.0.i.i to i8 ; <i8> [#uses=1]
  %2 = icmp eq i8 %retval56.i.i, 0                ; <i1> [#uses=2]
  %g_96.promoted.i = load i8, i8* @g_96               ; <i8> [#uses=3]
  %3 = icmp eq i8 %g_96.promoted.i, 0             ; <i1> [#uses=2]
  br i1 %3, label %func_4.exit.i, label %bb.i.i.i

bb.i.i.i:                                         ; preds = %entry
  %4 = load volatile i8, i8* @g_100, align 1          ; <i8> [#uses=0]
  br label %func_4.exit.i

; CHECK-LABEL: test4:
; CHECK: g_100
; CHECK: testb
; CHECK-NOT: xor
; CHECK: setne
; CHECK: testb

func_4.exit.i:                                    ; preds = %bb.i.i.i, %entry
  %.not.i = xor i1 %2, true                       ; <i1> [#uses=1]
  %brmerge.i = or i1 %3, %.not.i                  ; <i1> [#uses=1]
  %.mux.i = select i1 %2, i8 %g_96.promoted.i, i8 0 ; <i8> [#uses=1]
  br i1 %brmerge.i, label %func_1.exit, label %bb.i.i

bb.i.i:                                           ; preds = %func_4.exit.i
  %5 = load volatile i8, i8* @g_100, align 1          ; <i8> [#uses=0]
  br label %func_1.exit

func_1.exit:                                      ; preds = %bb.i.i, %func_4.exit.i
  %g_96.tmp.0.i = phi i8 [ %g_96.promoted.i, %bb.i.i ], [ %.mux.i, %func_4.exit.i ] ; <i8> [#uses=2]
  store i8 %g_96.tmp.0.i, i8* @g_96
  %6 = zext i8 %g_96.tmp.0.i to i32               ; <i32> [#uses=1]
  %7 = tail call i32 (i8*, ...) @printf(i8* noalias getelementptr ([15 x i8], [15 x i8]* @_2E_str, i64 0, i64 0), i32 %6) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind


; Should compile to setcc | -2.
; rdar://6668608
define i32 @test5(i32* nocapture %P) nounwind readonly {
entry:
; CHECK-LABEL: test5:
; CHECK:  xorl %eax, %eax
; CHECK: 	setg	%al
; CHECK:	orl	$-2, %eax
; CHECK:	ret

	%0 = load i32, i32* %P, align 4		; <i32> [#uses=1]
	%1 = icmp sgt i32 %0, 41		; <i1> [#uses=1]
	%iftmp.0.0 = select i1 %1, i32 -1, i32 -2		; <i32> [#uses=1]
	ret i32 %iftmp.0.0
}

define i32 @test6(i32* nocapture %P) nounwind readonly {
entry:
; CHECK-LABEL: test6:
; CHECK:  xorl %eax, %eax
; CHECK: 	setl	%al
; CHECK:	leal	4(%rax,%rax,8), %eax
; CHECK:        ret
	%0 = load i32, i32* %P, align 4		; <i32> [#uses=1]
	%1 = icmp sgt i32 %0, 41		; <i1> [#uses=1]
	%iftmp.0.0 = select i1 %1, i32 4, i32 13		; <i32> [#uses=1]
	ret i32 %iftmp.0.0
}


; Don't try to use a 16-bit conditional move to do an 8-bit select,
; because it isn't worth it. Just use a branch instead.
define i8 @test7(i1 inreg %c, i8 inreg %a, i8 inreg %b) nounwind {
; CHECK-LABEL: test7:
; CHECK:     testb	$1, %dil
; CHECK-NEXT:     jne	LBB

  %d = select i1 %c, i8 %a, i8 %b
  ret i8 %d
}
