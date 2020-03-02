; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=medium <%s | FileCheck -check-prefix=MEDIUM %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=large <%s | FileCheck -check-prefix=LARGE %s

; Test correct code generation for medium and large code model
; for loading and storing a file-scope static variable.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@gi = global i32 5, align 4

define signext i32 @test_file_static() nounwind {
entry:
  %0 = load i32, i32* @gi, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @gi, align 4
  ret i32 %0
}

; MEDIUM-LABEL: test_file_static:
; MEDIUM: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; MEDIUM: addi [[REG2:[0-9]+]], [[REG1]], [[VAR]]@toc@l
; MEDIUM: lwz {{[0-9]+}}, 0([[REG2]])
; MEDIUM: stw {{[0-9]+}}, 0([[REG2]])
; MEDIUM: .type [[VAR]],@object
; MEDIUM: .data
; MEDIUM: .globl [[VAR]]
; MEDIUM: [[VAR]]:
; MEDIUM: .long 5

; LARGE-LABEL: test_file_static:
; LARGE: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; LARGE: ld [[REG2:[0-9]+]], [[VAR]]@toc@l([[REG1]])
; LARGE: lwz {{[0-9]+}}, 0([[REG2]])
; LARGE: stw {{[0-9]+}}, 0([[REG2]])
; LARGE:      .type gi,@object
; LARGE-NEXT: .data
; LARGE-NEXT: .globl gi
; LARGE-NEXT: .p2align 2
; LARGE-NEXT: gi:
; LARGE-NEXT: .long 5

; LARGE:      .section .toc,"aw",@progbits
; LARGE-NEXT: [[VAR]]:
; LARGE-NEXT: .tc gi[TC],gi
