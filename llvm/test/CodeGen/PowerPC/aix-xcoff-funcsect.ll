; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -function-sections < %s | \
; RUN:   FileCheck --check-prefix=ASM %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -function-sections < %s | \
; RUN:   FileCheck --check-prefix=ASM %s

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

; ASM:        .csect .foo[PR],2
; ASM-NEXT:  	.globl	foo[DS]                         # -- Begin function foo
; ASM-NEXT:  	.globl	.foo[PR]
; ASM-NEXT:  	.align	4
; ASM-NEXT:  	.csect foo[DS]
; ASM-NEXT:  alias_foo:                                # @foo
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, .foo[PR]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, TOC[TC0]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, 0
; ASM-NEXT:  	.csect .foo[PR],2
; ASM-NEXT:  .alias_foo:
; ASM-NEXT:  # %bb.0:                                # %entry
; ASM-NEXT:  	blr
; ASM:        .csect .hidden_foo[PR],2
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
; ASM:        .csect .bar[PR],2
; ASM-NEXT:  	.globl	bar[DS]                         # -- Begin function bar
; ASM-NEXT:  	.globl	.bar[PR]
; ASM-NEXT:  	.align	4
; ASM-NEXT:  	.csect bar[DS]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, .bar[PR]                     # @bar
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, TOC[TC0]
; ASM-NEXT:  	.vbyte	{{[0-9]+}}, 0
; ASM-NEXT:  	.csect .bar[PR],2
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
