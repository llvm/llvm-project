; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic | FileCheck -check-prefix=SMALL-BSS %s
@bar = common global i32 0, align 4

declare i32 @call_foo(i32, ...)

define i32 @foo() {
entry:
  %0 = load i32, i32* @bar, align 4
  %call = call i32 (i32, ...) @call_foo(i32 %0, i32 0, i32 1, i32 2, i32 4, i32 8, i32 16, i32 32, i32 64)
  ret i32 0
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIC Level", i32 1}
; SMALL-BSS-LABEL:foo:
; SMALL-BSS:         stwu 1, -32(1)
; SMALL-BSS:         stw 30, 24(1)
; SMALL-BSS:         bl _GLOBAL_OFFSET_TABLE_@local-4
; SMALL-BSS:         mflr 30
; SMALL-BSS-DAG:     stw {{[0-9]+}}, 8(1)
; SMALL-BSS-DAG:     lwz [[VREG:[0-9]+]], bar@GOT(30)
; SMALL-BSS-DAG:     lwz {{[0-9]+}}, 0([[VREG]])
; SMALL-BSS:         bl call_foo@PLT
; SMALL-BSS:         lwz 30, -8(1)
