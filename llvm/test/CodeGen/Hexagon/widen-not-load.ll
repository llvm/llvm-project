; Test that double word post increment load is not generated.
; REQUIRES: asserts

; REQUIRES: asserts
; RUN: llc -mtriple=hexagon -O2 -debug-only=hexagon-load-store-widening \
; RUN:      %s -o 2>&1 - | FileCheck %s

; Loads with positive invalid postinc is not widened
define ptr @test1() {
; CHECK-LABEL: test1
; CHECK-NOT: memd(r{{[0-9]+}}++
entry:
  %0 = load ptr, ptr null, align 4
  %b = getelementptr i8, ptr %0, i32 20
  %1 = load i32, ptr %0, align 8
  %c = getelementptr i8, ptr %0, i32 4
  %2 = load i32, ptr %c, align 4
  %call55 = call i8 @foo(ptr %b, i32 %1, i32 %2)
  ret ptr null
}

; Loads with negative invalid postinc is not widened
define ptr @test2() {
; CHECK-LABEL: test2
; CHECK-NOT: memd(r{{[0-9]+}}++
entry:
  %0 = load ptr, ptr null, align 4
  %b = getelementptr i8, ptr %0, i32 -20
  %1 = load i32, ptr %0, align 8
  %c = getelementptr i8, ptr %0, i32 4
  %2 = load i32, ptr %c, align 4
  %call55 = call i8 @foo(ptr %b, i32 %1, i32 %2)
  ret ptr null
}

; Loads with valid positive postinc is widened
define ptr @test3() {
; CHECK-LABEL: test3
; CHECK: memd
entry:
  %0 = load ptr, ptr null, align 4
  %b = getelementptr i8, ptr %0, i32 24
  %1 = load i32, ptr %0, align 8
  %c = getelementptr i8, ptr %0, i32 4
  %2 = load i32, ptr %c, align 4
  %call55 = call i8 @foo(ptr %b, i32 %1, i32 %2)
  ret ptr null
}

; Loads with valid negative postinc is widened
define ptr @test4() {
; CHECK-LABEL: test4
; CHECK: memd
entry:
  %0 = load ptr, ptr null, align 4
  %b = getelementptr i8, ptr %0, i32 -24
  %1 = load i32, ptr %0, align 8
  %c = getelementptr i8, ptr %0, i32 4
  %2 = load i32, ptr %c, align 4
  %call55 = call i8 @foo(ptr %b, i32 %1, i32 %2)
  ret ptr null
}

declare i8 @foo(ptr, i32, i32)
