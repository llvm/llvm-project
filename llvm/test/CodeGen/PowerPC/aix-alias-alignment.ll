; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -data-sections=false < %s | FileCheck --check-prefix=ASM %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -data-sections=true < %s | FileCheck --check-prefix=ASM-DATASECT %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -dr %t.o | FileCheck --check-prefix=OBJ %s
; RUN: llvm-objdump --syms %t.o | FileCheck --check-prefix=SYM %s

@_MergedGlobals = global <{ i32, i32 }> <{ i32 1, i32 2 }>, align 4
@var1 = alias i32, getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 0)
@var2 = alias i32, getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 1)
@var3 = alias i32, ptr @var2

define void @foo(i32 %a1, i32 %a2, i32 %a3) {
  store i32 %a1, ptr getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 0), align 4
  store i32 %a2, ptr getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 1), align 4
  ret void
}

; ASM:              # -- End function
; ASM-NEXT:         .csect .data[RW],2
; ASM-NEXT:         .globl  _MergedGlobals # @_MergedGlobals
; ASM-NEXT:         .globl  var1
; ASM-NEXT:         .globl  var2
; ASM-NEXT:         .globl  var3
; ASM-NEXT:         .align  2
; ASM-NEXT: _MergedGlobals:
; ASM-NEXT: var1:
; ASM-NEXT:         .vbyte  4, 1 # 0x1
; ASM-NEXT: var2:
; ASM-NEXT: var3:
; ASM-NEXT:         .vbyte  4, 2 # 0x2
; ASM-NEXT:         .toc
; ASM-NEXT: L..C0:
; ASM-NEXT:         .tc _MergedGlobals[TC],_MergedGlobals

; ASM-DATASECT:              # -- End function
; ASM-DATASECT-NEXT:         .csect _MergedGlobals[RW],2
; ASM-DATASECT-NEXT:         .globl  _MergedGlobals[RW] # @_MergedGlobals
; ASM-DATASECT-NEXT:         .globl  var1
; ASM-DATASECT-NEXT:         .globl  var2
; ASM-DATASECT-NEXT:         .globl  var3
; ASM-DATASECT-NEXT:         .align  2
; ASM-DATASECT-NEXT: var1:
; ASM-DATASECT-NEXT:         .vbyte  4, 1 # 0x1
; ASM-DATASECT-NEXT: var2:
; ASM-DATASECT-NEXT: var3:
; ASM-DATASECT-NEXT:         .vbyte  4, 2 # 0x2
; ASM-DATASECT-NEXT:         .toc
; ASM-DATASECT-NEXT: L..C0:
; ASM-DATASECT-NEXT:         .tc _MergedGlobals[TC],_MergedGlobals[RW]

; OBJ:      00000000 <.foo>:
; OBJ-NEXT:        0: 80 a2 00 00  	lwz 5, 0(2)
; OBJ-NEXT: 			00000002:  R_TOC	_MergedGlobals
; OBJ-NEXT:        4: 90 65 00 00  	stw 3, 0(5)
; OBJ-NEXT:        8: 90 85 00 04  	stw 4, 4(5)
; OBJ-NEXT:        c: 4e 80 00 20  	blr

; SYM:      SYMBOL TABLE:
; SYM-NEXT: 00000000      df *DEBUG*	00000000 <stdin>
; SYM-NEXT: 00000000 l       .text	00000029 
; SYM-NEXT: 00000000 g     F .text (csect: ) 	00000000 .foo
; SYM-NEXT: 0000002c l       .data	00000008 .data
; SYM-NEXT: 0000002c g     O .data (csect: .data) 	00000000 _MergedGlobals
; SYM-NEXT: 0000002c g     O .data (csect: .data) 	00000000 var1
; SYM-NEXT: 00000030 g     O .data (csect: .data) 	00000000 var2
; SYM-NEXT: 00000030 g     O .data (csect: .data) 	00000000 var3
; SYM-NEXT: 00000034 g     O .data	0000000c foo
; SYM-NEXT: 00000040 l       .data	00000000 TOC
; SYM-NEXT: 00000040 l     O .data	00000004 _MergedGlobals
