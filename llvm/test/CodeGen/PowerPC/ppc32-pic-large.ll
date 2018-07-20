; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic | FileCheck -check-prefix=LARGE-BSS %s
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -mattr=+secure-plt -relocation-model=pic | FileCheck -check-prefix=LARGE-SECUREPLT %s
@bar = common global i32 0, align 4

declare i32 @call_foo(i32, ...)

define i32 @foo() {
entry:
  %0 = load i32, i32* @bar, align 4
  %call = call i32 (i32, ...) @call_foo(i32 %0, i32 0, i32 1, i32 2, i32 4, i32 8, i32 16, i32 32, i32 64)
  ret i32 %0
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIC Level", i32 2}
; LARGE-BSS:       [[POFF:\.L[0-9]+\$poff]]:
; LARGE-BSS-NEXT:    .long .LTOC-[[PB:\.L[0-9]+\$pb]]
; LARGE-BSS-NEXT:  foo:
; LARGE-BSS:         stwu 1, -32(1)
; LARGE-BSS:         stw 30, 24(1)
; LARGE-BSS:         bl [[PB]]
; LARGE-BSS-NEXT:  [[PB]]:
; LARGE-BSS:         mflr 30
; LARGE-BSS:         lwz [[REG:[0-9]+]], [[POFF]]-[[PB]](30)
; LARGE-BSS-NEXT:    add 30, [[REG]], 30
; LARGE-BSS-DAG:     lwz [[VREG:[0-9]+]], [[VREF:\.LC[0-9]+]]-.LTOC(30)
; LARGE-BSS-DAG:     lwz {{[0-9]+}}, 0([[VREG]])
; LARGE-BSS-DAG:     stw {{[0-9]+}}, 8(1)
; LARGE-BSS:         lwz 30, 24(1)
; LARGE-BSS:       [[VREF]]:
; LARGE-BSS-NEXT:     .p2align 2
; LARGE-BSS-NEXT:    .long bar
; LARGE-SECUREPLT:   addis 30, 30, .LTOC-.L0$pb@ha
; LARGE-SECUREPLT:   addi 30, 30, .LTOC-.L0$pb@l
; LARGE-SECUREPLT:   bl call_foo@PLT+32768
