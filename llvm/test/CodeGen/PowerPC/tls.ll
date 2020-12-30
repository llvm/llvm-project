; RUN: llc -relocation-model=static -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -O0 < %s -mcpu=ppc64 | FileCheck -check-prefix=OPT0 %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -O1 < %s -mcpu=ppc64 | FileCheck -check-prefix=OPT1 %s
; RUN: llc -verify-machineinstrs -O0 < %s -mtriple=ppc32-- -mcpu=ppc | FileCheck -check-prefix=OPT0-PPC32 %s
; RUN: llc -relocation-model=pic -verify-machineinstrs -O0 < %s -mtriple=ppc32-- -mcpu=ppc | FileCheck -check-prefix=OPT0-PPC32-PIC %s

@a = dso_local thread_local global i32 0, align 4

;OPT0-LABEL:          localexec:
;OPT1-LABEL:          localexec:
define dso_local i32 @localexec() nounwind {
entry:
;OPT0:          addis [[REG1:[1-9][0-9]*]], 13, a@tprel@ha
;OPT0-NEXT:     addi [[REG2:[1-9][0-9]*]], [[REG1]], a@tprel@l
;OPT0-NEXT:     li [[REG3:[0-9]+]], 42
;OPT0:          stw [[REG3]], 0([[REG2]])
;OPT1:          addis [[REG1:[1-9][0-9]*]], 13, a@tprel@ha
;OPT1-NEXT:     li [[REG3:[0-9]+]], 42
;OPT1:     stw [[REG3]], a@tprel@l([[REG1]])
  store i32 42, i32* @a, align 4
  ret i32 0
}

; Test correct assembly code generation for thread-local storage
; using the initial-exec model.

@a2 = external thread_local(initialexec) global i32

define dso_local signext i32 @main2() nounwind {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32, i32* @a2, align 4
  ret i32 %0
}

; OPT1-LABEL: main2:
; OPT1: addis [[REG1:[1-9][0-9]*]], 2, a2@got@tprel@ha
; OPT1: ld [[REG2:[0-9]+]], a2@got@tprel@l([[REG1]])
; OPT1: add {{[0-9]+}}, [[REG2]], a2@tls

;OPT0-PPC32-LABEL:    main2:
;OPT0-PPC32:       li [[REG1:[1-9][0-9]*]], _GLOBAL_OFFSET_TABLE_@l
;OPT0-PPC32:       addis [[REG1]], [[REG1]], _GLOBAL_OFFSET_TABLE_@ha
;OPT0-PPC32:       lwz [[REG2:[1-9][0-9]*]], a2@got@tprel([[REG1]])
;OPT0-PPC32:       add 3, [[REG2]], a2@tls

;OPT0-PPC32-PIC-LABEL:  main2:
;OPT0-PPC32-PIC:        .long _GLOBAL_OFFSET_TABLE_-{{.*}}
;OPT0-PPC32-PIC-NOT:    li {{[0-9]+}}, _GLOBAL_OFFSET_TABLE_@l
;OPT0-PPC32-PIC-NOT:    addis {{[0-9]+}}, {{[1-9][0-9*]}}, _GLOBAL_OFFSET_TABLE_@ha
;OPT0-PPC32-PIC-NOT:    bl __tls_get_addr(a2@tlsgd)@PLT
;OPT0-PPC32-PIC:        lwz {{[0-9]+}}, a2@got@tprel({{[1-9][0-9]*}})
