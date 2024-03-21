; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -data-sections=false < %s | FileCheck --check-prefix=ASM %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump --syms %t.o | FileCheck --check-prefix=SYM %s

@ConstVector = global <2 x i64> <i64 12, i64 34>, align 4
@var1 = alias i64, getelementptr inbounds (<2 x i64>, ptr @ConstVector, i32 0, i32 1)
define void @foo1(i64 %a1) {
  store i64 %a1, ptr getelementptr inbounds (<2 x i64>, ptr @ConstVector, i32 0, i32 1), align 4
  ret void
}

; ASM:           .globl     ConstVector  # @ConstVector
; ASM-NEXT:      .globl     var1
; ASM-NEXT:      .align     4
; ASM-NEXT: ConstVector:
; ASM-NEXT:      .vbyte     4, 0         # 0xc
; ASM-NEXT:      .vbyte     4, 12
; ASM-NEXT: var1:
; ASM-NEXT:      .vbyte     4, 0         # 0x22
; ASM-NEXT:      .vbyte     4, 34

@ConstDataSeq = global [2 x i64] [i64 12, i64 34], align 4
@var2 = alias i64, getelementptr inbounds ([2 x i64], ptr @ConstDataSeq, i32 0, i32 1)
define void @foo2(i64 %a1) {
  store i64 %a1, ptr getelementptr inbounds ([2 x i64], ptr @ConstDataSeq, i32 0, i32 1), align 4
  ret void
}

; ASM:           .globl     ConstDataSeq   # @ConstDataSeq
; ASM-NEXT:      .globl     var2
; ASM-NEXT:      .align     3
; ASM-NEXT: ConstDataSeq:
; ASM-NEXT:      .vbyte     4, 0           # 0xc
; ASM-NEXT:      .vbyte     4, 12
; ASM-NEXT: var2:
; ASM-NEXT:      .vbyte     4, 0           # 0x22
; ASM-NEXT:      .vbyte     4, 34

%struct.B = type { i64 }
@ConstArray = global [2 x %struct.B] [%struct.B {i64 12}, %struct.B {i64 34}], align 4
@var3 = alias %struct.B, ptr @ConstArray
define void @foo3(%struct.B %a1) {
  store %struct.B %a1, ptr getelementptr inbounds ([2 x %struct.B], ptr @ConstArray, i32 0, i32 1), align 4
  ret void
}

; ASM:           .globl     ConstArray  # @ConstArray
; ASM-NEXT:      .globl     var3
; ASM-NEXT:      .align     3
; ASM-NEXT: ConstArray:
; ASM-NEXT: var3:
; ASM-NEXT:      .vbyte     4, 0        # 0xc
; ASM-NEXT:      .vbyte     4, 12
; ASM-NEXT:      .vbyte     4, 0        # 0x22
; ASM-NEXT:      .vbyte     4, 34

; SYM:      SYMBOL TABLE:
; SYM-NEXT: 00000000      df *DEBUG*	00000000 .file
; SYM-NEXT: 00000000 l       .text	0000008a 
; SYM-NEXT: 00000000 g     F .text (csect: ) 	00000000 .foo1
; SYM-NEXT: 00000030 g     F .text (csect: ) 	00000000 .foo2
; SYM-NEXT: 00000060 g     F .text (csect: ) 	00000000 .foo3
; SYM-NEXT: 00000090 l       .data	00000030 .data
; SYM-NEXT: 00000090 g     O .data (csect: .data) 	00000000 ConstVector
; SYM-NEXT: 00000098 g     O .data (csect: .data) 	00000000 var1
; SYM-NEXT: 000000a0 g     O .data (csect: .data) 	00000000 ConstDataSeq
; SYM-NEXT: 000000a8 g     O .data (csect: .data) 	00000000 var2
; SYM-NEXT: 000000b0 g     O .data (csect: .data) 	00000000 ConstArray
; SYM-NEXT: 000000b0 g     O .data (csect: .data) 	00000000 var3
; SYM-NEXT: 000000c0 g     O .data	0000000c foo1
; SYM-NEXT: 000000cc g     O .data	0000000c foo2
; SYM-NEXT: 000000d8 g     O .data	0000000c foo3
; SYM-NEXT: 000000e4 l       .data	00000000 TOC
; SYM-NEXT: 000000e4 l     O .data	00000004 ConstVector
; SYM-NEXT: 000000e8 l     O .data	00000004 ConstDataSeq
; SYM-NEXT: 000000ec l     O .data	00000004 ConstArray
