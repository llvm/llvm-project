; RUN: llc < %s -mtriple=powerpc -relocation-model=pic | \
; RUN:    FileCheck -check-prefixes=SMALL,SMALL-BSS %s
; RUN: llc < %s -mtriple=powerpc -relocation-model=pic -mattr=+secure-plt | \
; RUN:    FileCheck -check-prefixes=SMALL,SMALL-SECURE %s
@bar = common global i32 0, align 4

declare i32 @call_foo(i32, ...)

define i32 @foo() {
entry:
  %0 = load i32, ptr @bar, align 4
  %call = call i32 (i32, ...) @call_foo(i32 %0, i32 0, i32 1, i32 2, i32 4, i32 8, i32 16, i32 32, i32 64)
  ret i32 0
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIC Level", i32 1}
; SMALL-LABEL: foo:
; SMALL:         stwu 1, -32(1)
; SMALL:         stw 30, 24(1)
; SMALL-BSS:     bl _GLOBAL_OFFSET_TABLE_@local-4
; SMALL-SECURE:  bl .L0$pb
; SMALL:         mflr 30
; SMALL-SECURE:  addis 30, 30, _GLOBAL_OFFSET_TABLE_-.L0$pb@ha
; SMALL-SECURE:  addi 30, 30, _GLOBAL_OFFSET_TABLE_-.L0$pb@l
; SMALL-DAG:     stw {{[0-9]+}}, 8(1)
; SMALL-DAG:     lwz [[VREG:[0-9]+]], bar@GOT(30)
; SMALL-DAG:     lwz {{[0-9]+}}, 0([[VREG]])
; SMALL:         bl call_foo@PLT{{$}}
; SMALL:         lwz 30, 24(1)
