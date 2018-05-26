; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s


define i64 @test1(i8* %data) {
; CHECK-LABEL: test1:
; CHECK:       movzbl
; CHECK-NEXT:  shlq
; CHECK-NEXT:  andl
; CHECK-NEXT:  retq
entry:
  %bf.load = load i8, i8* %data, align 4
  %bf.clear = shl i8 %bf.load, 2
  %0 = and i8 %bf.clear, 60
  %mul = zext i8 %0 to i64
  ret i64 %mul
}

define i8* @test2(i8* %data) {
; CHECK-LABEL: test2:
; CHECK:       movzbl
; CHECK-NEXT:  andl
; CHECK-NEXT:  leaq
; CHECK-NEXT:  retq
entry:
  %bf.load = load i8, i8* %data, align 4
  %bf.clear = shl i8 %bf.load, 2
  %0 = and i8 %bf.clear, 60
  %mul = zext i8 %0 to i64
  %add.ptr = getelementptr inbounds i8, i8* %data, i64 %mul
  ret i8* %add.ptr
}

; If the shift op is SHL, the logic op can only be AND.
define i64 @test3(i8* %data) {
; CHECK-LABEL: test3:
; CHECK:       movb
; CHECK-NEXT:  shlb
; CHECK-NEXT:  xorb
; CHECK-NEXT:  movzbl
; CHECK-NEXT:  retq
entry:
  %bf.load = load i8, i8* %data, align 4
  %bf.clear = shl i8 %bf.load, 2
  %0 = xor i8 %bf.clear, 60
  %mul = zext i8 %0 to i64
  ret i64 %mul
}

define i64 @test4(i8* %data) {
; CHECK-LABEL: test4:
; CHECK:       movzbl
; CHECK-NEXT:  shrq
; CHECK-NEXT:  andl
; CHECK-NEXT:  retq
entry:
  %bf.load = load i8, i8* %data, align 4
  %bf.clear = lshr i8 %bf.load, 2
  %0 = and i8 %bf.clear, 60
  %1 = zext i8 %0 to i64
  ret i64 %1
}

define i64 @test5(i8* %data) {
; CHECK-LABEL: test5:
; CHECK:       movzbl
; CHECK-NEXT:  shrq
; CHECK-NEXT:  xorq
; CHECK-NEXT:  retq
entry:
  %bf.load = load i8, i8* %data, align 4
  %bf.clear = lshr i8 %bf.load, 2
  %0 = xor i8 %bf.clear, 60
  %1 = zext i8 %0 to i64
  ret i64 %1
}

define i64 @test6(i8* %data) {
; CHECK-LABEL: test6:
; CHECK:       movzbl
; CHECK-NEXT:  shrq
; CHECK-NEXT:  orq
; CHECK-NEXT:  retq
entry:
  %bf.load = load i8, i8* %data, align 4
  %bf.clear = lshr i8 %bf.load, 2
  %0 = or i8 %bf.clear, 60
  %1 = zext i8 %0 to i64
  ret i64 %1
}

; Don't do the folding if the other operand isn't a constant.
define i64 @test7(i8* %data, i8 %logop) {
; CHECK-LABEL: test7:
; CHECK:       movb
; CHECK-NEXT:  shrb
; CHECK-NEXT:  orb
; CHECK-NEXT:  movzbl
; CHECK-NEXT:  retq
entry:
  %bf.load = load i8, i8* %data, align 4
  %bf.clear = lshr i8 %bf.load, 2
  %0 = or i8 %bf.clear, %logop
  %1 = zext i8 %0 to i64
  ret i64 %1
}

; Load is folded with sext.
define i64 @test8(i8* %data) {
; CHECK-LABEL: test8:
; CHECK:       movsbl
; CHECK-NEXT:  movzwl
; CHECK-NEXT:  shrl
; CHECK-NEXT:  orl
entry:
  %bf.load = load i8, i8* %data, align 4
  %ext = sext i8 %bf.load to i16
  %bf.clear = lshr i16 %ext, 2
  %0 = or i16 %bf.clear, 60
  %1 = zext i16 %0 to i64
  ret i64 %1
}

