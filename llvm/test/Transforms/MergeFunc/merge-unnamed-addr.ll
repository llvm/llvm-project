; RUN: opt -S -mergefunc < %s | FileCheck %s

; CHECK-NOT: @b

@x = constant { i32 (i32)*, i32 (i32)* } { i32 (i32)* @a, i32 (i32)* @b }
; CHECK: { i32 (i32)* @a, i32 (i32)* @a }

define internal i32 @a(i32 %a) unnamed_addr {
  %b = xor i32 %a, 0
  %c = xor i32 %b, 0
  ret i32 %c
}

define internal i32 @b(i32 %a) unnamed_addr {
  %b = xor i32 %a, 0
  %c = xor i32 %b, 0
  ret i32 %c
}
