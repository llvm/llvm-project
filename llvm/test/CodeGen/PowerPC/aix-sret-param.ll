; RUN: llc -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec \
; RUN: -stop-after=machine-cp --verify-machineinstrs < %s | FileCheck \
; RUN: --check-prefixes=MIR,MIR32 %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec \
; RUN: -stop-after=machine-cp --verify-machineinstrs < %s | FileCheck \
; RUN: --check-prefixes=MIR,MIR64 %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec \
; RUN: --verify-machineinstrs < %s | FileCheck --check-prefixes=ASM,ASM32 %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec \
; RUN: --verify-machineinstrs < %s | FileCheck --check-prefixes=ASM,ASM64 %s

%struct.S = type { i8 }
%struct.T = type { double, i32, i32, i32, float }

define void @test1() {
entry:
  %s = alloca %struct.S, align 4
  call void @foo(%struct.S* sret(%struct.S) %s)
  ret void
}

define void @test2() {
entry:
  %t = alloca %struct.T, align 8
  call void @bar(%struct.T* sret(%struct.T) %t)
  ret void
}

declare void @foo(%struct.S* sret(%struct.S))
declare void @bar(%struct.T* sret(%struct.T))

; MIR:      name:            test1
; MIR:      stack:
; MIR-NEXT:  - { id: 0, name: s, type: default, offset: 0, size: 1, alignment: 8,

; MIR32:      bb.0.entry:
; MIR32-NEXT:   ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; MIR32-NEXT:   renamable $r3 = ADDI %stack.0.s, 0
; MIR32-NEXT:   BL_NOP <mcsymbol .foo[PR]>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1
; MIR32-NEXT:   ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; MIR64:      bb.0.entry:
; MIR64-NEXT:   ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; MIR64-NEXT:   renamable $x3 = ADDI8 %stack.0.s, 0
; MIR64-NEXT:   BL8_NOP <mcsymbol .foo[PR]>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; MIR64-NEXT:   ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1


; ASM-LABEL: .test1:

; ASM32:       stwu 1, -64(1)
; ASM32-NEXT:  addi 3, 1, 56
; ASM32-NEXT:  bl .foo[PR]
; ASM32-NEXT:  nop
; ASM32-NEXT:  addi 1, 1, 64

; ASM64:       stdu 1, -128(1)
; ASM64-NEXT:  addi 3, 1, 120
; ASM64-NEXT:  bl .foo[PR]
; ASM64-NEXT:  nop
; ASM64-NEXT:  addi 1, 1, 128



; MIR:      name:            test2
; MIR:      stack:
; MIR-NEXT:   - { id: 0, name: t, type: default, offset: 0, size: 24, alignment: 8,

; MIR32:       bb.0.entry:
; MIR32-NEXT:    ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; MIR32-NEXT:    renamable $r3 = ADDI %stack.0.t, 0
; MIR32-NEXT:    BL_NOP <mcsymbol .bar[PR]>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1
; MIR32-NEXT:    ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; MIR64:      bb.0.entry:
; MIR64-NEXT:   ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; MIR64-NEXT:   renamable $x3 = ADDI8 %stack.0.t, 0
; MIR64-NEXT:   BL8_NOP <mcsymbol .bar[PR]>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; MIR64-NEXT:   ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1


; ASM-LABEL: .test2:

; ASM32:        stwu 1, -80(1)
; ASM32-NEXT:   addi 3, 1, 56
; ASM32-NEXT:   bl .bar[PR]
; ASM32-NEXT:   nop
; ASM32-NEXT:   addi 1, 1, 80


; ASM64:        stdu 1, -144(1)
; ASM64-NEXT:   addi 3, 1, 120
; ASM64-NEXT:   bl .bar[PR]
; ASM64-NEXT:   nop
; ASM64-NEXT:   addi 1, 1, 144
