; RUN: llc -mtriple=hexagon-unknown-linux-musl < %s | FileCheck %s -check-prefix=MUSL
; RUN: llc -mtriple=hexagon-unknown-none-elf   < %s | FileCheck %s -check-prefix=NONMUSL

; MUSL-NOT: memw
; NONMUSL: memw

declare i32 @f0(i32 %a0, ...)

define i32 @f1(i32 %a0, i32 %a1) #0 {
b1:
  %v7 = call i32 (i32, ...) @f0(i32 %a0, i32 %a1)
  ret i32 %v7
}

attributes #0 = { nounwind }
