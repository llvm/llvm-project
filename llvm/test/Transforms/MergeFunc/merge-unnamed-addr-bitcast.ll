; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

%A = type { i32 }
%B = type { i32 }

; CHECK-NOT: @b

@x = constant { ptr, ptr }
  { ptr @a,
    ptr @b }
; CHECK: { ptr @a, ptr @a }

define internal i32 @a(%A) unnamed_addr {
  extractvalue %A %0, 0
  xor i32 %2, 0
  ret i32 %3
}

define internal i32 @b(%B) unnamed_addr {
  extractvalue %B %0, 0
  xor i32 %2, 0
  ret i32 %3
}

define i32 @c(i32) {
  insertvalue %B undef, i32 %0, 0
  call i32 @b(%B %2)
; CHECK: call i32 @a(%B %2)
  ret i32 %3
}
