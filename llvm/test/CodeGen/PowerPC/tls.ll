; RUN: llc -relocation-model=static -verify-machineinstrs -O0 < %s -march=ppc64 -mcpu=ppc64 | FileCheck -check-prefix=OPT0 %s
; RUN: llc -relocation-model=static -verify-machineinstrs -O1 < %s -march=ppc64 -mcpu=ppc64 | FileCheck -check-prefix=OPT1 %s
; RUN: llc -verify-machineinstrs -O0 < %s -march=ppc32 -mcpu=ppc | FileCheck -check-prefix=OPT0-PPC32 %s

target triple = "powerpc64-unknown-linux-gnu"

@a = thread_local global i32 0, align 4

;OPT0-LABEL:          localexec:
;OPT1-LABEL:          localexec:
define i32 @localexec() nounwind {
entry:
;OPT0:          addis [[REG1:[0-9]+]], 13, a@tprel@ha
;OPT0-NEXT:     addi [[REG1]], [[REG1]], a@tprel@l
;OPT0-NEXT:     li [[REG2:[0-9]+]], 42
;OPT0:          stw [[REG2]], 0([[REG1]])
;OPT1:          addis [[REG1:[0-9]+]], 13, a@tprel@ha
;OPT1-NEXT:     li [[REG2:[0-9]+]], 42
;OPT1-NEXT:     stw [[REG2]], a@tprel@l([[REG1]])
  store i32 42, i32* @a, align 4
  ret i32 0
}

; Test correct assembly code generation for thread-local storage
; using the initial-exec model.

@a2 = external thread_local global i32

define signext i32 @main2() nounwind {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32, i32* @a2, align 4
  ret i32 %0
}

; OPT1-LABEL: main2:
; OPT1: addis [[REG1:[0-9]+]], 2, a2@got@tprel@ha
; OPT1: ld [[REG2:[0-9]+]], a2@got@tprel@l([[REG1]])
; OPT1: add {{[0-9]+}}, [[REG2]], a2@tls

;OPT0-PPC32-LABEL:    main2:
;OPT0-PPC32:       li [[REG1:[0-9]+]], _GLOBAL_OFFSET_TABLE_@l
;OPT0-PPC32:       addis [[REG1]], [[REG1]], _GLOBAL_OFFSET_TABLE_@ha
;OPT0-PPC32:       lwz [[REG2:[0-9]+]], a2@got@tprel@l([[REG1]])
;OPT0-PPC32:       add 3, [[REG2]], a2@tls
