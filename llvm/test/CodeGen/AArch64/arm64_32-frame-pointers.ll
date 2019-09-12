; RUN: llc -mtriple=arm64_32-apple-ios8.0 %s -o - | FileCheck %s

; We're provoking LocalStackSlotAllocation to create some shared frame bases
; here: it wants multiple <fi#N> using instructions that can be satisfied by a
; single base, but not within the addressing-mode.
;
; When that happens it's important that we don't mix our pointer sizes
; (e.g. try to create an ldr from a w-register base).
define i8 @test_register_wrangling() {
; CHECK-LABEL: test_register_wrangling:
; CHECK: add [[TMP:x[0-9]+]], sp,
; CHECK: add x[[BASE:[0-9]+]], [[TMP]],
; CHECK: ldrb {{w[0-9]+}}, [x[[BASE]], #1]
; CHECK: ldrb {{w[0-9]+}}, [x[[BASE]]]

  %var1 = alloca i8, i32 4100
  %var3 = alloca i8
  %dummy = alloca i8, i32 4100

  %var1p1 = getelementptr i8, i8* %var1, i32 1
  %val1 = load i8, i8* %var1
  %val2 = load i8, i8* %var3

  %sum = add i8 %val1, %val2
  ret i8 %sum
}
