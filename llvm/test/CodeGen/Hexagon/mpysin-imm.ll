; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; M2_mpysin takes 8-bit unsigned immediates and is not extendable.
; CHECK-NOT: = -mpyi(r{{[0-9]*}},#1536)

target triple = "hexagon-unknown--elf"

@g0 = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = mul nsw i32 %a0, 1536
  store i32 %v0, ptr @g0, align 4
  %v1 = sub nsw i32 0, %v0
  ret i32 %v1
}

attributes #0 = { nounwind }
