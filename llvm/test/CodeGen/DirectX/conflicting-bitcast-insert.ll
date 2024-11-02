; RUN: llc --filetype=asm %s -o - | FileCheck %s
target triple = "dxil-unknown-unknown"

define i64 @test(ptr %p) {
  store i32 0, ptr %p
  %v = load i64, ptr %p
  ret i64 %v
}

; CHECK: define i64 @test(ptr %p) {
; CHECK-NEXT: %1 = bitcast ptr %p to ptr
; CHECK-NEXT: store i32 0, ptr %1, align 4
; CHECK-NEXT: %2 = bitcast ptr %p to ptr
; CHECK-NEXT: %3 = load i64, ptr %2, align 8

define i64 @testGEP(ptr %p) {
  %ptr = getelementptr i32, ptr %p, i32 4
  %val = load i64, ptr %p
  ret i64 %val
}

; CHECK: define i64 @testGEP(ptr %p) {
; CHECK-NEXT:   %1 = bitcast ptr %p to ptr
; CHECK-NEXT:   %ptr = getelementptr i32, ptr %1, i32 4
; CHECK-NEXT:   %2 = bitcast ptr %p to ptr
; CHECK-NEXT:   %3 = load i64, ptr %2, align 8
