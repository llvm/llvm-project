; RUN: llc -mtriple=hexagon -o - %s | FileCheck %s

; Test coverage for HexagonConstExtenders: exercise constant extender
; optimization for store-immediate patterns. Multiple uses of the same
; large immediate constant in store operations should share an extender.

; CHECK-LABEL: test_store_imm:
; CHECK: ##
define void @test_store_imm(ptr %p) #0 {
entry:
  %p0 = getelementptr i8, ptr %p, i32 100000
  %p1 = getelementptr i8, ptr %p, i32 100004
  %p2 = getelementptr i8, ptr %p, i32 100008
  %p3 = getelementptr i8, ptr %p, i32 100012
  store i8 1, ptr %p0, align 1
  store i8 2, ptr %p1, align 1
  store i8 3, ptr %p2, align 1
  store i8 4, ptr %p3, align 1
  ret void
}

; Exercise constant extender optimization with multiple loads from a
; global array with large offsets.
@big_array = external global [100000 x i32], align 4

; CHECK-LABEL: test_load_large_offsets:
; CHECK: ##
define i32 @test_load_large_offsets() #0 {
entry:
  %p0 = getelementptr [100000 x i32], ptr @big_array, i32 0, i32 25000
  %p1 = getelementptr [100000 x i32], ptr @big_array, i32 0, i32 25001
  %p2 = getelementptr [100000 x i32], ptr @big_array, i32 0, i32 25002
  %p3 = getelementptr [100000 x i32], ptr @big_array, i32 0, i32 25003
  %v0 = load i32, ptr %p0, align 4
  %v1 = load i32, ptr %p1, align 4
  %v2 = load i32, ptr %p2, align 4
  %v3 = load i32, ptr %p3, align 4
  %s0 = add i32 %v0, %v1
  %s1 = add i32 %s0, %v2
  %s2 = add i32 %s1, %v3
  ret i32 %s2
}

; Exercise constant extender with arithmetic using large immediates.
; CHECK-LABEL: test_arith_large_imm:
; CHECK: ##
define i32 @test_arith_large_imm(i32 %a, i32 %b, i32 %c) #0 {
entry:
  %add1 = add i32 %a, 100000
  %add2 = add i32 %b, 100000
  %add3 = add i32 %c, 100000
  %s1 = add i32 %add1, %add2
  %s2 = add i32 %s1, %add3
  ret i32 %s2
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
