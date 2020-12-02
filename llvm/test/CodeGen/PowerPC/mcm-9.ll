; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=medium <%s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=large <%s | FileCheck %s

; Test correct code generation for medium and large code model
; for loading and storing an aliased external variable.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@a = external global i32

define signext i32 @test_external() nounwind {
entry:
  %0 = load i32, i32* @a, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @a, align 4
  ret i32 %0
}

; CHECK-LABEL: test_external:
; CHECK: addis [[REG1:[0-9]+]], 2, .LC[[TOCNUM:[0-9]+]]@toc@ha
; CHECK: ld [[REG2:[0-9]+]], .LC[[TOCNUM]]@toc@l([[REG1]])
; CHECK: lwz {{[0-9]+}}, 0([[REG2]])
; CHECK: stw {{[0-9]+}}, 0([[REG2]])
; CHECK: .section .toc
; CHECK: .LC[[TOCNUM]]:
; CHECK: .tc {{[a-z0-9A-Z_.]+}}[TC],{{[a-z0-9A-Z_.]+}}
