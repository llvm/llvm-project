; RUN: llc -march=hexagon -O0 %s -o - | llvm-mc -arch=hexagon -filetype=obj | llvm-objdump -d - | FileCheck %s

; CHECK: immext(#16777216)
; CHECK-NEXT: r0 = add(r0,##16777279)

define void @f0(i32 %a0) {
b0:
  %v0 = add i32 16777279, %a0
  %v1 = alloca i32, align 4
  store i32 %v0, i32* %v1, align 4
  ret void
}
