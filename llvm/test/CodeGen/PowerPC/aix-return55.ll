; RUN: llc -mcpu=pwr4 -mattr=-altivec -mtriple=powerpc-ibm-aix-xcoff \
; RUN:     -verify-machineinstrs -data-sections=false -xcoff-traceback-table=false < %s | FileCheck %s
; RUN: llc -mcpu=pwr4 -mattr=-altivec -mtriple=powerpc-ibm-aix-xcoff \
; RUN:     -verify-machineinstrs -data-sections=false -xcoff-traceback-table=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefixes=CHECKOBJ,CHECKOBJ32 %s
; RUN: llvm-readobj -S %t.o | FileCheck --check-prefixes=CHECKSECT,CHECKSECT32 %s

; RUN: llc -mcpu=pwr4 -mattr=-altivec -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:     -verify-machineinstrs -data-sections=false -xcoff-traceback-table=false \
; RUN:     -filetype=obj -o %t64.o < %s
; RUN: llvm-objdump -D %t64.o | FileCheck --check-prefixes=CHECKOBJ,CHECKOBJ64 %s
; RUN: llvm-readobj -S %t64.o | FileCheck --check-prefixes=CHECKSECT,CHECKSECT64 %s

@a = global i64 320255973571806, align 8
@d = global double 5.000000e+00, align 8
@strA = private unnamed_addr constant [10 x i8] c"hellowor\0A\00", align 1

define dso_local signext i32 @foo() {
entry:
  ret i32 55
; CHECK-LABEL: .foo:
; CHECK: li 3, 55
; CHECK: blr
}

;CHECKOBJ32:    00000000 <.foo>:
;CHECKOBJ64:    0000000000000000 <.foo>:
;CHECKOBJ-NEXT:       0: 38 60 00 37                    li 3, 55
;CHECKOBJ-NEXT:       4: 4e 80 00 20                    blr{{[[:space:]] *}}
;CHECKOBJ32-NEXT: 00000008 <.rodata>:
;CHECKOBJ64-NEXT: 0000000000000008 <.rodata>:
;CHECKOBJ-NEXT:       8: 68 65 6c 6c                   xori 5, 3, 27756
;CHECKOBJ-NEXT:       c: 6f 77 6f 72 xoris 23, 27, 28530
;CHECKOBJ-NEXT:      10: 0a 00 00 00 tdlti 0, 0{{[[:space:]] *}}
;CHECKOBJ-NEXT:   Disassembly of section .data:{{[[:space:]] *}}
;CHECKOBJ32-NEXT: 00000018 <a>:
;CHECKOBJ64-NEXT: 0000000000000018 <a>:
;CHECKOBJ-NEXT:      18: 00 01 23 45                   <unknown>
;CHECKOBJ-NEXT:      1c: 67 8a bc de                   oris 10, 28, 48350{{[[:space:]] *}}
;CHECKOBJ32-NEXT: 00000020 <d>:
;CHECKOBJ64-NEXT: 0000000000000020 <d>:
;CHECKOBJ-NEXT:      20: 40 14 00 00                   bdnzf   20, 0x20
;CHECKOBJ-NEXT:      24: 00 00 00 00                   <unknown>{{[[:space:]] *}}
;CHECKOBJ32-NEXT: 00000028 <foo>:
;CHECKOBJ32-NEXT:    28: 00 00 00 00                   <unknown>
;CHECKOBJ32-NEXT:    2c: 00 00 00 34                   <unknown>
;CHECKOBJ32-NEXT:    30: 00 00 00 00                   <unknown>
;CHECKOBJ64-NEXT: 0000000000000028 <foo>:
;CHECKOBJ64-NEXT:    ...
;CHECKOBJ64-NEXT:    34: 00 00 00 40  	<unknown>

;CHECKSECT: Sections [
;CHECKSECT-NEXT:   Section {
;CHECKSECT-NEXT:     Index: 1
;CHECKSECT-NEXT:     Name: .text
;CHECKSECT-NEXT:     PhysicalAddress: 0x0
;CHECKSECT-NEXT:     VirtualAddress: 0x0
;CHECKSECT-NEXT:     Size: 0x14
;CHECKSECT32-NEXT:   RawDataOffset: 0x64
;CHECKSECT64-NEXT:   RawDataOffset: 0xA8
;CHECKSECT-NEXT:     RelocationPointer: 0x0
;CHECKSECT-NEXT:     LineNumberPointer: 0x0
;CHECKSECT-NEXT:     NumberOfRelocations: 0
;CHECKSECT-NEXT:     NumberOfLineNumbers: 0
;CHECKSECT-NEXT:     Type: STYP_TEXT (0x20)
;CHECKSECT-NEXT:   }
;CHECKSECT-NEXT:   Section {
;CHECKSECT-NEXT:     Index: 2
;CHECKSECT-NEXT:     Name: .data
;CHECKSECT-NEXT:     PhysicalAddress: 0x18
;CHECKSECT-NEXT:     VirtualAddress: 0x18
;CHECKSECT32-NEXT:   Size: 0x1C
;CHECKSECT32-NEXT:   RawDataOffset: 0x78
;CHECKSECT32-NEXT:   RelocationPointer: 0x94
;CHECKSECT64-NEXT:   Size: 0x28
;CHECKSECT64-NEXT:   RawDataOffset: 0xBC
;CHECKSECT64-NEXT:   RelocationPointer: 0xE4
;CHECKSECT-NEXT:     LineNumberPointer: 0x0
;CHECKSECT-NEXT:     NumberOfRelocations: 2
;CHECKSECT-NEXT:     NumberOfLineNumbers: 0
;CHECKSECT-NEXT:     Type: STYP_DATA (0x40)
;CHECKSECT-NEXT:   }
;CHECKSECT-NEXT: ]
