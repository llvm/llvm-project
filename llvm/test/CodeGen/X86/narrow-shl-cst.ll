; RUN: llc < %s -march=x86-64 | FileCheck %s
; PR5039

define i32 @test1(i32 %x) nounwind {
  %and = shl i32 %x, 10
  %shl = and i32 %and, 31744
  ret i32 %shl
; CHECK-LABEL: test1:
; CHECK: andl $31
; CHECK: shll $10
}

define i32 @test2(i32 %x) nounwind {
  %or = shl i32 %x, 10
  %shl = or i32 %or, 31744
  ret i32 %shl
; CHECK-LABEL: test2:
; CHECK: orl $31
; CHECK: shll $10
}

define i32 @test3(i32 %x) nounwind {
  %xor = shl i32 %x, 10
  %shl = xor i32 %xor, 31744
  ret i32 %shl
; CHECK-LABEL: test3:
; CHECK: xorl $31
; CHECK: shll $10
}

define i64 @test4(i64 %x) nounwind {
  %and = shl i64 %x, 40
  %shl = and i64 %and, 264982302294016
  ret i64 %shl
; CHECK-LABEL: test4:
; CHECK: andq $241
; CHECK: shlq $40
}

define i64 @test5(i64 %x) nounwind {
  %and = shl i64 %x, 40
  %shl = and i64 %and, 34084860461056
  ret i64 %shl
; CHECK-LABEL: test5:
; CHECK: andq $31
; CHECK: shlq $40
}

define i64 @test6(i64 %x) nounwind {
  %and = shl i64 %x, 32
  %shl = and i64 %and, -281474976710656
  ret i64 %shl
; CHECK-LABEL: test6:
; CHECK: andq $-65536
; CHECK: shlq $32
}

define i64 @test7(i64 %x) nounwind {
  %or = shl i64 %x, 40
  %shl = or i64 %or, 264982302294016
  ret i64 %shl
; CHECK-LABEL: test7:
; CHECK: orq $241
; CHECK: shlq $40
}

define i64 @test8(i64 %x) nounwind {
  %or = shl i64 %x, 40
  %shl = or i64 %or, 34084860461056
  ret i64 %shl
; CHECK-LABEL: test8:
; CHECK: orq $31
; CHECK: shlq $40
}

define i64 @test9(i64 %x) nounwind {
  %xor = shl i64 %x, 40
  %shl = xor i64 %xor, 264982302294016
  ret i64 %shl
; CHECK-LABEL: test9:
; CHECK: orq $241
; CHECK: shlq $40
}

define i64 @test10(i64 %x) nounwind {
  %xor = shl i64 %x, 40
  %shl = xor i64 %xor, 34084860461056
  ret i64 %shl
; CHECK-LABEL: test10:
; CHECK: xorq $31
; CHECK: shlq $40
}

define i64 @test11(i64 %x) nounwind {
  %xor = shl i64 %x, 33
  %shl = xor i64 %xor, -562949953421312
  ret i64 %shl
; CHECK-LABEL: test11:
; CHECK: xorq $-65536
; CHECK: shlq $33
}

; PR23098
define i32 @test12(i32 %x, i32* %y) nounwind {
  %and = shl i32 %x, 1
  %shl = and i32 %and, 255
  store i32 %shl, i32* %y
  ret i32 %shl
; CHECK-LABEL: test12:
; CHECK: andl $127
; CHECK-NEXT: addl
; CHECK-NOT: shl
}

define i64 @test13(i64 %x, i64* %y) nounwind {
  %and = shl i64 %x, 1
  %shl = and i64 %and, 255
  store i64 %shl, i64* %y
  ret i64 %shl
; CHECK-LABEL: test13:
; CHECK: andq $127
; CHECK-NEXT: addq
; CHECK-NOT: shl
}
