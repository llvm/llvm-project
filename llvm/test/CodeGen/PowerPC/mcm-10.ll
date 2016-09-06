; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O1 -code-model=medium <%s | FileCheck %s

; Test peephole optimization for medium code model (32-bit TOC offsets)
; for loading and storing a static variable scoped to a function.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@test_fn_static.si = internal global i32 0, align 4

define signext i32 @test_fn_static() nounwind {
entry:
  %0 = load i32, i32* @test_fn_static.si, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @test_fn_static.si, align 4
  ret i32 %0
}

; CHECK-LABEL: test_fn_static:
; CHECK: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK: lwa {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
; CHECK-NOT: extsw
; CHECK: stw {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
; CHECK: .type [[VAR]],@object
; CHECK: .lcomm [[VAR]],4,4
