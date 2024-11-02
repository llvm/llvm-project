; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -function-sections < %s | \
; RUN:   FileCheck --check-prefix=ASM %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -function-sections < %s | \
; RUN:   FileCheck --check-prefix=ASM %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -function-sections -xcoff-traceback-table=true \
; RUN:     -filetype=obj -o %t32.o < %s
; RUN: llvm-objdump --syms --reloc --symbol-description %t32.o | \
; RUN:   FileCheck --check-prefix=XCOFF32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -function-sections -xcoff-traceback-table=true \
; RUN:     -filetype=obj -o %t64.o < %s
; RUN: llvm-objdump --syms --reloc --symbol-description %t64.o | \
; RUN:   FileCheck --check-prefix=XCOFF64 %s

@alias_foo = alias void (...), bitcast (void ()* @foo to void (...)*)

define void @foo() {
entry:
  ret void
}

define hidden void @hidden_foo() {
entry:
  ret void
}

define void @bar() {
entry:
  call void @foo()
  call void @static_overalign_foo()
  call void bitcast (void (...)* @alias_foo to void ()*)()
  call void bitcast (void (...)* @extern_foo to void ()*)()
  call void @hidden_foo()
  ret void
}

declare void @extern_foo(...)

define internal void @static_overalign_foo() align 64 {
entry:
  ret void
}

; ASM:        .csect .foo[PR],5
; ASM-NEXT:  	.globl	foo[DS]                         # -- Begin function foo
; ASM-NEXT:  	.globl	.foo[PR]
; ASM-NEXT:  	.align	4
; ASM-NEXT:  	.csect foo[DS]
; ASM-NEXT:  alias_foo:                                # @foo
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, .foo[PR]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, TOC[TC0]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, 0
; ASM-NEXT:  	.csect .foo[PR],5
; ASM-NEXT:  .alias_foo:
; ASM-NEXT:  # %bb.0:                                # %entry
; ASM-NEXT:  	blr
; ASM:        .csect .hidden_foo[PR],5
; ASM-NEXT:  	.globl	hidden_foo[DS],hidden           # -- Begin function hidden_foo
; ASM-NEXT:  	.globl	.hidden_foo[PR],hidden
; ASM-NEXT:  	.align	4
; ASM-NEXT:  	.csect hidden_foo[DS]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, .hidden_foo[PR]              # @hidden_foo
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, TOC[TC0]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, 0
; ASM-NEXT:  	.csect .hidden_foo[PR]
; ASM-NEXT:  # %bb.0:                                # %entry
; ASM-NEXT:  	blr
; ASM:        .csect .bar[PR],5
; ASM-NEXT:  	.globl	bar[DS]                         # -- Begin function bar
; ASM-NEXT:  	.globl	.bar[PR]
; ASM-NEXT:  	.align	4
; ASM-NEXT:  	.csect bar[DS]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, .bar[PR]                     # @bar
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, TOC[TC0]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, 0
; ASM-NEXT:  	.csect .bar[PR],5
; ASM-NEXT:  # %bb.0:                                # %entry
; ASM:        bl .foo[PR]
; ASM-NEXT:  	nop
; ASM-NEXT:  	bl .static_overalign_foo[PR]
; ASM-NEXT:  	nop
; ASM-NEXT:  	bl .alias_foo
; ASM-NEXT:  	nop
; ASM-NEXT:  	bl .extern_foo
; ASM-NEXT:  	nop
; ASM-NEXT:  	bl .hidden_foo[PR]
; ASM-NEXT:  	nop
; ASM:        .csect .static_overalign_foo[PR],6
; ASM-NEXT:  	.lglobl	static_overalign_foo[DS]                  # -- Begin function static_overalign_foo
; ASM-NEXT:  	.lglobl	.static_overalign_foo[PR]
; ASM-NEXT:  	.align	6
; ASM-NEXT:  	.csect static_overalign_foo[DS]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, .static_overalign_foo[PR]              # @static_overalign_foo
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, TOC[TC0]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, 0
; ASM-NEXT:  	.csect .static_overalign_foo[PR],6
; ASM-NEXT:  # %bb.0:                                # %entry
; ASM-NEXT:  	blr
; ASM:        .extern	.extern_foo
; ASM-NEXT:  	.extern	extern_foo[DS]
; ASM-NEXT:  	.globl	alias_foo
; ASM-NEXT:  	.globl	.alias_foo

; XCOFF32:      SYMBOL TABLE:
; XCOFF32-NEXT: 00000000      df *DEBUG*	00000000 (idx: 0) <stdin>
; XCOFF32-NEXT: 00000000         *UND*	00000000 (idx: 1) .extern_foo[PR]
; XCOFF32-NEXT: 00000000         *UND*	00000000 (idx: 3) extern_foo[DS]
; XCOFF32-NEXT: 00000000 l       .text	00000000 (idx: 5) .text[PR]
; XCOFF32-NEXT: 00000000 g       .text	00000019 (idx: 7) .foo[PR]
; XCOFF32-NEXT: 00000000 g     F .text (csect: (idx: 7) .foo[PR]) 	00000000 (idx: 9) .alias_foo
; XCOFF32-NEXT: 00000020 g       .text	00000020 .hidden (idx: 11) .hidden_foo[PR]
; XCOFF32-NEXT: 00000040 g       .text	00000059 (idx: 13) .bar[PR]
; XCOFF32-NEXT: 000000c0 l       .text	0000002a (idx: 15) .static_overalign_foo[PR]
; XCOFF32-NEXT: 000000ec g     O .data	0000000c (idx: 17) foo[DS]
; XCOFF32-NEXT: 000000ec g     O .data (csect: (idx: 17) foo[DS]) 	00000000 (idx: 19) alias_foo
; XCOFF32-NEXT: 000000f8 g     O .data	0000000c .hidden (idx: 21) hidden_foo[DS]
; XCOFF32-NEXT: 00000104 g     O .data	0000000c (idx: 23) bar[DS]
; XCOFF32-NEXT: 00000110 l     O .data	0000000c (idx: 25) static_overalign_foo[DS]
; XCOFF32-NEXT: 0000011c l       .data	00000000 (idx: 27) TOC[TC0]

; XCOFF32:      RELOCATION RECORDS FOR [.text]:
; XCOFF32-NEXT: OFFSET   TYPE                     VALUE
; XCOFF32-NEXT: 0000004c R_RBR                    (idx: 7) .foo[PR]
; XCOFF32-NEXT: 00000054 R_RBR                    (idx: 15) .static_overalign_foo[PR]
; XCOFF32-NEXT: 0000005c R_RBR                    (idx: 9) .alias_foo
; XCOFF32-NEXT: 00000064 R_RBR                    (idx: 1) .extern_foo[PR]
; XCOFF32-NEXT: 0000006c R_RBR                    (idx: 11) .hidden_foo[PR]
; XCOFF32:      RELOCATION RECORDS FOR [.data]:
; XCOFF32-NEXT: OFFSET   TYPE                     VALUE
; XCOFF32-NEXT: 00000000 R_POS                    (idx: 7) .foo[PR]
; XCOFF32-NEXT: 00000004 R_POS                    (idx: 27) TOC[TC0]
; XCOFF32-NEXT: 0000000c R_POS                    (idx: 11) .hidden_foo[PR]
; XCOFF32-NEXT: 00000010 R_POS                    (idx: 27) TOC[TC0]
; XCOFF32-NEXT: 00000018 R_POS                    (idx: 13) .bar[PR]
; XCOFF32-NEXT: 0000001c R_POS                    (idx: 27) TOC[TC0]
; XCOFF32-NEXT: 00000024 R_POS                    (idx: 15) .static_overalign_foo[PR]
; XCOFF32-NEXT: 00000028 R_POS                    (idx: 27) TOC[TC0]


; XCOFF64:      SYMBOL TABLE:
; XCOFF64-NEXT: 0000000000000000      df *DEBUG*	0000000000000000 (idx: 0) <stdin>
; XCOFF64-NEXT: 0000000000000000         *UND*	0000000000000000 (idx: 1) .extern_foo[PR]
; XCOFF64-NEXT: 0000000000000000         *UND*	0000000000000000 (idx: 3) extern_foo[DS]
; XCOFF64-NEXT: 0000000000000000 l       .text	0000000000000000 (idx: 5) .text[PR]
; XCOFF64-NEXT: 0000000000000000 g       .text	0000000000000019 (idx: 7) .foo[PR]
; XCOFF64-NEXT: 0000000000000000 g     F .text (csect: (idx: 7) .foo[PR]) 	0000000000000000 (idx: 9) .alias_foo
; XCOFF64-NEXT: 0000000000000020 g       .text	0000000000000020 .hidden (idx: 11) .hidden_foo[PR]
; XCOFF64-NEXT: 0000000000000040 g       .text	0000000000000059 (idx: 13) .bar[PR]
; XCOFF64-NEXT: 00000000000000c0 l       .text	000000000000002a (idx: 15) .static_overalign_foo[PR]
; XCOFF64-NEXT: 00000000000000f0 g     O .data	0000000000000018 (idx: 17) foo[DS]
; XCOFF64-NEXT: 00000000000000f0 g     O .data (csect: (idx: 17) foo[DS]) 	0000000000000000 (idx: 19) alias_foo
; XCOFF64-NEXT: 0000000000000108 g     O .data	0000000000000018 .hidden (idx: 21) hidden_foo[DS]
; XCOFF64-NEXT: 0000000000000120 g     O .data	0000000000000018 (idx: 23) bar[DS]
; XCOFF64-NEXT: 0000000000000138 l     O .data	0000000000000018 (idx: 25) static_overalign_foo[DS]
; XCOFF64-NEXT: 0000000000000150 l       .data	0000000000000000 (idx: 27) TOC[TC0]

; XCOFF64:      RELOCATION RECORDS FOR [.text]:
; XCOFF64-NEXT: OFFSET           TYPE                     VALUE
; XCOFF64-NEXT: 000000000000004c R_RBR                    (idx: 7) .foo[PR]
; XCOFF64-NEXT: 0000000000000054 R_RBR                    (idx: 15) .static_overalign_foo[PR]
; XCOFF64-NEXT: 000000000000005c R_RBR                    (idx: 9) .alias_foo
; XCOFF64-NEXT: 0000000000000064 R_RBR                    (idx: 1) .extern_foo[PR]
; XCOFF64-NEXT: 000000000000006c R_RBR                    (idx: 11) .hidden_foo[PR]
; XCOFF64:      RELOCATION RECORDS FOR [.data]:
; XCOFF64-NEXT: OFFSET           TYPE                     VALUE
; XCOFF64-NEXT: 0000000000000000 R_POS                    (idx: 7) .foo[PR]
; XCOFF64-NEXT: 0000000000000008 R_POS                    (idx: 27) TOC[TC0]
; XCOFF64-NEXT: 0000000000000018 R_POS                    (idx: 11) .hidden_foo[PR]
; XCOFF64-NEXT: 0000000000000020 R_POS                    (idx: 27) TOC[TC0]
; XCOFF64-NEXT: 0000000000000030 R_POS                    (idx: 13) .bar[PR]
; XCOFF64-NEXT: 0000000000000038 R_POS                    (idx: 27) TOC[TC0]
; XCOFF64-NEXT: 0000000000000048 R_POS                    (idx: 15) .static_overalign_foo[PR]
; XCOFF64-NEXT: 0000000000000050 R_POS                    (idx: 27) TOC[TC0]
