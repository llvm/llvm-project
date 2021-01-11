; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=medium <%s | FileCheck -check-prefix=MEDIUM %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=large <%s | FileCheck -check-prefix=LARGE %s

; Test correct code generation for medium and large code model
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

; MEDIUM-LABEL: test_fn_static:
; MEDIUM: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; MEDIUM: addi [[REG2:[0-9]+]], [[REG1]], [[VAR]]@toc@l
; MEDIUM: lwz {{[0-9]+}}, 0([[REG2]])
; MEDIUM: addis [[REG3:[0-9]+]], 2, [[VAR]]@toc@ha
; MEDIUM: addi [[REG4:[0-9]+]], [[REG3]], [[VAR]]@toc@l
; MEDIUM: stw {{[0-9]+}}, 0([[REG4]])
; MEDIUM: .type [[VAR]],@object
; MEDIUM: .lcomm [[VAR]],4,4

; LARGE-LABEL: test_fn_static:
; LARGE: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; LARGE: ld [[REG2:[0-9]+]], [[VAR]]@toc@l([[REG1]])
; LARGE: lwz {{[0-9]+}}, 0([[REG2]])
; LARGE: addis [[REG3:[0-9]+]], 2, [[VAR]]@toc@ha
; LARGE: ld [[REG4:[0-9]+]], [[VAR]]@toc@l([[REG3]])
; LARGE: stw {{[0-9]+}}, 0([[REG4]])
; LARGE:      .type test_fn_static.si,@object
; LARGE-NEXT: .lcomm test_fn_static.si,4,4

; LARGE:      .section .toc,"aw",@progbits
; LARGE-NEXT: [[VAR]]:
; LARGE-NEXT: .tc test_fn_static.si[TC],test_fn_static.si
