; RUN: llc -mcpu=generic -march=x86 < %s | FileCheck %s

define i64 @test1(i32 %xx, i32 %test) nounwind {
  %conv = zext i32 %xx to i64
  %and = and i32 %test, 7
  %sh_prom = zext i32 %and to i64
  %shl = shl i64 %conv, %sh_prom
  ret i64 %shl
; CHECK: test1:
; CHECK: shll	%cl, %eax
; CHECK: shrl	%edx
; CHECK: xorb	$31
; CHECK: shrl	%cl, %edx
}

define i64 @test2(i64 %xx, i32 %test) nounwind {
  %and = and i32 %test, 7
  %sh_prom = zext i32 %and to i64
  %shl = shl i64 %xx, %sh_prom
  ret i64 %shl
; CHECK: test2:
; CHECK: shll	%cl, %esi
; CHECK: shrl	%edx
; CHECK: xorb	$31
; CHECK: shrl	%cl, %edx
; CHECK: orl	%esi, %edx
; CHECK: shll	%cl, %eax
}

define i64 @test3(i64 %xx, i32 %test) nounwind {
  %and = and i32 %test, 7
  %sh_prom = zext i32 %and to i64
  %shr = lshr i64 %xx, %sh_prom
  ret i64 %shr
; CHECK: test3:
; CHECK: shrl	%cl, %esi
; CHECK: leal	(%edx,%edx), %eax
; CHECK: xorb	$31, %cl
; CHECK: shll	%cl, %eax
; CHECK: orl	%esi, %eax
; CHECK: shrl	%cl, %edx
}

define i64 @test4(i64 %xx, i32 %test) nounwind {
  %and = and i32 %test, 7
  %sh_prom = zext i32 %and to i64
  %shr = ashr i64 %xx, %sh_prom
  ret i64 %shr
; CHECK: test4:
; CHECK: shrl	%cl, %esi
; CHECK: leal	(%edx,%edx), %eax
; CHECK: xorb	$31, %cl
; CHECK: shll	%cl, %eax
; CHECK: orl	%esi, %eax
; CHECK: sarl	%cl, %edx
}
