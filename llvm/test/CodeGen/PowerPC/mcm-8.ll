; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=medium < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=large < %s | FileCheck %s

; Test correct code generation for medium and large code model
; for loading a variable with available-externally linkage.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@x = available_externally constant [13 x i8] c"St9bad_alloc\00"

define signext i8 @test_avext() nounwind {
entry:
  %0 = getelementptr inbounds [13 x i8], [13 x i8]* @x, i32 0, i32 0
  %1 = load i8, i8* %0, align 1
  ret i8 %1
}

; CHECK-LABEL: test_avext:
; CHECK: addis [[REG1:[0-9]+]], 2, .LC[[TOCNUM:[0-9]+]]@toc@ha
; CHECK: ld [[REG2:[0-9]+]], .LC[[TOCNUM]]@toc@l([[REG1]])
; CHECK: lbz {{[0-9]+}}, 0([[REG2]])
; CHECK: .section .toc
; CHECK: .LC[[TOCNUM]]:
; CHECK: .tc {{[a-z0-9A-Z_.]+}}[TC],{{[a-z0-9A-Z_.]+}}
