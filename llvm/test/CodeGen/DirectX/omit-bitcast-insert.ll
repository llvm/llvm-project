; RUN: llc --filetype=asm %s -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

define i64 @test(ptr %p) #0 {
  %v = load i64, ptr %p
  ret i64 %v
}

; CHECK: define i64 @test(ptr %p) #0 {
; CHECK-NEXT: %v = load i64, ptr %p, align 8
; CHECK-NEXT: ret i64 %v

define i64 @test2(ptr %p) #0 {
  store i64 0, ptr %p
  %v = load i64, ptr %p
  ret i64 %v
}

; CHECK: define i64 @test2(ptr %p) #0 {
; CHECK-NEXT: store i64 0, ptr %p
; CHECK-NEXT: %v = load i64, ptr %p, align 8
; CHECK-NEXT: ret i64 %v

define i32 @test3(ptr %0) #0 {
  %2 = getelementptr i32, ptr %0, i32 4
  %3 = load i32, ptr %2
  ret i32 %3
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}

; CHECK: define i32 @test3(ptr %0) #0 {
; CHECK-NEXT: %2 = getelementptr i32, ptr %0, i32 4
; CHECK-NEXT: %3 = load i32, ptr %2
