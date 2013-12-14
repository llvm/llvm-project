; RUN: llc -mcpu=generic -mtriple=i686-unknown-unknown < %s | FileCheck %s
define i64 @test1(i32 %xx, i32 %test) nounwind {
  %conv = zext i32 %xx to i64
  %and = and i32 %test, 7
  %sh_prom = zext i32 %and to i64
  %shl = shl i64 %conv, %sh_prom
  ret i64 %shl
; CHECK-LABEL: test1:
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
; CHECK-LABEL: test2:
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
; CHECK-LABEL: test3:
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
; CHECK-LABEL: test4:
; CHECK: shrl	%cl, %esi
; CHECK: leal	(%edx,%edx), %eax
; CHECK: xorb	$31, %cl
; CHECK: shll	%cl, %eax
; CHECK: orl	%esi, %eax
; CHECK: sarl	%cl, %edx
}

; PR14668
define <2 x i64> @test5(<2 x i64> %A, <2 x i64> %B) {
  %shl = shl <2 x i64> %A, %B
  ret <2 x i64> %shl
; CHECK: test5
; CHECK: shl
; CHECK: shldl
; CHECK: shl
; CHECK: shldl
}

; PR16108
define i32 @test6() {
  %x = alloca i32, align 4
  %t = alloca i64, align 8
  store i32 1, i32* %x, align 4
  store i64 1, i64* %t, align 8  ;; DEAD
  %load = load i32* %x, align 4
  %shl = shl i32 %load, 8
  %add = add i32 %shl, -224
  %sh_prom = zext i32 %add to i64
  %shl1 = shl i64 1, %sh_prom
  %cmp = icmp ne i64 %shl1, 4294967296
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret i32 1

if.end:                                           ; preds = %entry
  ret i32 0

; CHECK-LABEL: test6:
; CHECK-NOT: andb $31
; CHECK: sete
; CHECK: movzbl
; CHECK: xorl $1
; CHECK: orl
}
