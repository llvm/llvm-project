; RUN: llc < %s -mcpu=generic -mtriple=i686-pc-linux-gnu -asm-verbose=0 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"

define zeroext i16 @test1(i16 zeroext %x) nounwind {
entry:
	%div = udiv i16 %x, 33
	ret i16 %div
; CHECK: test1:
; CHECK: imull	$63551, %eax, %eax
; CHECK-NEXT: shrl	$21, %eax
; CHECK-NEXT: ret
}

define zeroext i16 @test2(i8 signext %x, i16 zeroext %c) nounwind readnone ssp noredzone {
entry:
  %div = udiv i16 %c, 3
  ret i16 %div

; CHECK: test2:
; CHECK: imull	$43691, %eax, %eax
; CHECK-NEXT: shrl	$17, %eax
; CHECK-NEXT: ret
}

define zeroext i8 @test3(i8 zeroext %x, i8 zeroext %c) nounwind readnone ssp noredzone {
entry:
  %div = udiv i8 %c, 3
  ret i8 %div

; CHECK: test3:
; CHECK: movzbl  8(%esp), %eax
; CHECK-NEXT: imull	$171, %eax, %eax
; CHECK-NEXT: shrl	$9, %eax
; CHECK-NEXT: ret
}

define signext i16 @test4(i16 signext %x) nounwind {
entry:
	%div = sdiv i16 %x, 33		; <i32> [#uses=1]
	ret i16 %div
; CHECK: test4:
; CHECK: imull	$1986, %eax, %
}

define i32 @test5(i32 %A) nounwind {
        %tmp1 = udiv i32 %A, 1577682821         ; <i32> [#uses=1]
        ret i32 %tmp1
; CHECK: test5:
; CHECK: movl	$365384439, %eax
; CHECK: mull	4(%esp)
}

define signext i16 @test6(i16 signext %x) nounwind {
entry:
  %div = sdiv i16 %x, 10
  ret i16 %div
; CHECK: test6:
; CHECK: imull	$26215, %eax, %eax
; CHECK: shrl	$31, %ecx
; CHECK: sarl	$18, %eax
}

define i32 @test7(i32 %x) nounwind {
  %div = udiv i32 %x, 28
  ret i32 %div
; CHECK: test7:
; CHECK: shrl $2
; CHECK: movl $613566757
; CHECK: mull
; CHECK-NOT: shrl
; CHECK: ret
}

; PR13326
define i8 @test8(i8 %x) nounwind {
  %div = udiv i8 %x, 78
  ret i8 %div
; CHECK: test8:
; CHECK: shrb %
; CHECK: imull $211
; CHECK: shrl $13
; CHECK: ret
}

define i8 @test9(i8 %x) nounwind {
  %div = udiv i8 %x, 116
  ret i8 %div
; CHECK: test9:
; CHECK: shrb $2
; CHECK: imull $71
; CHECK: shrl $11
; CHECK: ret
}
