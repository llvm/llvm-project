; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=medium \
; RUN:   -fast-isel=false -mattr=-vsx <%s | FileCheck -check-prefix=MEDIUM %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=medium \
; RUN:   -fast-isel=false -mattr=+vsx <%s | FileCheck -check-prefix=MEDIUM-VSX %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=large \
; RUN:   -fast-isel=false -mattr=-vsx <%s | FileCheck -check-prefix=LARGE %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=large \
; RUN:   -fast-isel=false -mattr=+vsx <%s | FileCheck -check-prefix=LARGE-VSX %s
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -O0 -code-model=medium \
; RUN:   -fast-isel=false -mattr=+vsx <%s | FileCheck -check-prefix=MEDIUM-P9 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -O0 -code-model=large \
; RUN:   -fast-isel=false -mattr=+vsx <%s | FileCheck -check-prefix=LARGE-P9 %s

; Test correct code generation for medium and large code model
; for loading a value from the constant pool (TOC-relative).

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @test_double_const() nounwind {
entry:
  ret double 0x3F4FD4920B498CF0
}

; MEDIUM: [[VAR:[a-z0-9A-Z_.]+]]:
; MEDIUM: .quad 0x3f4fd4920b498cf0
; MEDIUM-LABEL: test_double_const:
; MEDIUM: addis [[REG1:[0-9]+]], 2, [[VAR]]@toc@ha
; MEDIUM: addi [[REG2:[0-9]+]], [[REG1]], [[VAR]]@toc@l
; MEDIUM: lfd {{[0-9]+}}, 0([[REG2]])

; MEDIUM-VSX: [[VAR:[a-z0-9A-Z_.]+]]:
; MEDIUM-VSX: .quad 0x3f4fd4920b498cf0
; MEDIUM-VSX-LABEL: test_double_const:
; MEDIUM-VSX: addis [[REG1:[0-9]+]], 2, [[VAR]]@toc@ha
; MEDIUM-VSX: lfd {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])

; LARGE: [[VAR:[a-z0-9A-Z_.]+]]:
; LARGE: .quad 0x3f4fd4920b498cf0
; LARGE-LABEL: test_double_const:
; LARGE: addis [[REG1:[0-9]+]], 2, [[VAR2:[a-z0-9A-Z_.]+]]@toc@ha
; LARGE: ld [[REG2:[0-9]+]], [[VAR2]]@toc@l([[REG1]])
; LARGE: lfd {{[0-9]+}}, 0([[REG2]])

; LARGE-VSX: [[VAR:[a-z0-9A-Z_.]+]]:
; LARGE-VSX: .quad 0x3f4fd4920b498cf0
; LARGE-VSX-LABEL: test_double_const:
; LARGE-VSX: addis [[REG1:[0-9]+]], 2, [[VAR2:[a-z0-9A-Z_.]+]]@toc@ha
; LARGE-VSX: ld [[REG2:[0-9]+]], [[VAR2]]@toc@l([[REG1]])
; LARGE-VSX: lfdx {{[0-9]+}}, 0, [[REG2]]

; MEDIUM-P9: [[VAR:[a-z0-9A-Z_.]+]]:
; MEDIUM-P9: .quad 0x3f4fd4920b498cf0
; MEDIUM-P9-LABEL: test_double_const:
; MEDIUM-P9: addis [[REG1:[0-9]+]], 2, [[VAR]]@toc@ha
; MEDIUM-P9: addi [[REG2:[0-9]+]], [[REG1]], [[VAR]]@toc@l
; MEDIUM-P9: lfd {{[0-9]+}}, 0([[REG2]])

; LARGE-P9: [[VAR:[a-z0-9A-Z_.]+]]:
; LARGE-P9: .quad 0x3f4fd4920b498cf0
; LARGE-P9-LABEL: test_double_const:
; LARGE-P9: addis [[REG1:[0-9]+]], 2, [[VAR2:[a-z0-9A-Z_.]+]]@toc@ha
; LARGE-P9: ld [[REG2:[0-9]+]], [[VAR2]]@toc@l([[REG1]])
; LARGE-P9: lfd {{[0-9]+}}, 0([[REG2]])
