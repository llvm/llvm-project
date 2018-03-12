; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]+}} = sub(#0,r{{[0-9]+}})

; Function Attrs: nounwind
define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = alloca i32, align 4
  store i32 %a0, i32* %v0, align 4
  %v1 = load i32, i32* %v0, align 4
  %v2 = sub nsw i32 0, %v1
  ret i32 %v2
}

attributes #0 = { nounwind }
