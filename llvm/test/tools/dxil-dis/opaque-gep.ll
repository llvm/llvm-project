; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

define i32 @fn(ptr %0)  {
  %2 = getelementptr i32, ptr %0, i32 4
  %3 = load i32, ptr %2
  ret i32 %3
}

; CHECK:        define internal i32 @fn(i32*)
; CHECK-NEXT:   %2 = getelementptr i32, i32* %0, i32 4
; CHECK-NEXT:   %3 = load i32, i32* %2, align 4

define i32 @fn2(ptr addrspace(1) %0)  {
  %2 = getelementptr i32, ptr addrspace(1) %0, i32 4
  %3 = load i32, ptr addrspace(1) %2
  ret i32 %3
}

; CHECK:        define internal i32 @fn2(i32 addrspace(1)*)
; CHECK-NEXT:   %2 = getelementptr i32, i32 addrspace(1)* %0, i32 4
; CHECK-NEXT:   %3 = load i32, i32 addrspace(1)* %2, align 4
