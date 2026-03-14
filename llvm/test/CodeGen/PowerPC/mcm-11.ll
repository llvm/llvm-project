; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -O1 -code-model=medium <%s | FileCheck %s

; Test peephole optimization for medium code model (32-bit TOC offsets)
; for loading and storing a file-scope static variable.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@gi = dso_local global i32 5, align 4

define dso_local signext i32 @test_file_static() nounwind {
entry:
  %0 = load i32, ptr @gi, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @gi, align 4
  ret i32 %0
}

; CHECK-LABEL: test_file_static:
; CHECK: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK: lwa {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
; CHECK-NOT: extsw
; CHECK: stw {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
; CHECK: .type [[VAR]],@object
; CHECK: .data
; CHECK: .globl [[VAR]]
; CHECK: [[VAR]]:
; CHECK: .long 5
