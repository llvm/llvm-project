; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

; CHECK-NOT: @b

@x = constant { ptr, ptr } { ptr @a, ptr @b }
; CHECK: { ptr @a, ptr @a }

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
