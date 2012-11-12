; RUN: llc < %s -mtriple=i386-unknown-linux-gnu | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=i386-apple-darwin9.7 | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu -fdata-sections | FileCheck %s -check-prefix=LINUX-SECTIONS


; int G1;
@G1 = common global i32 0

; LINUX: .type   G1,@object
; LINUX: .comm  G1,4,4

; DARWIN: .comm	_G1,4,2




; const int G2 __attribute__((weak)) = 42;
@G2 = weak_odr unnamed_addr constant i32 42	


; TODO: linux drops this into .rodata, we drop it into ".gnu.linkonce.r.G2"

; DARWIN: .section __TEXT,__const_coal,coalesced
; DARWIN: _G2:
; DARWIN:    .long 42


; int * const G3 = &G1;
@G3 = unnamed_addr constant i32* @G1

; DARWIN: .section        __DATA,__const
; DARWIN: .globl _G3
; DARWIN: _G3:
; DARWIN:     .long _G1

; LINUX:   .section        .rodata,"a",@progbits
; LINUX:   .globl  G3

; LINUX-SECTIONS: .section        .rodata.G3,"a",@progbits
; LINUX-SECTIONS: .globl  G3


; _Complex long long const G4 = 34;
@G4 = unnamed_addr constant {i64,i64} { i64 34, i64 0 }

; DARWIN: .section        __TEXT,__literal16,16byte_literals
; DARWIN: _G4:
; DARWIN:     .long 34


; int G5 = 47;
@G5 = global i32 47

; LINUX: .data
; LINUX: .globl G5
; LINUX: G5:
; LINUX:    .long 47

; DARWIN: .section        __DATA,__data
; DARWIN: .globl _G5
; DARWIN: _G5:
; DARWIN:    .long 47


; PR4584
@"foo bar" = linkonce global i32 42

; LINUX: .type	foo_20_bar,@object
; LINUX: .section .data.foo_20_bar,"aGw",@progbits,foo_20_bar,comdat
; LINUX: .weak	foo_20_bar
; LINUX: foo_20_bar:

; DARWIN: .section		__DATA,__datacoal_nt,coalesced
; DARWIN: .globl	"_foo bar"
; DARWIN:	.weak_definition "_foo bar"
; DARWIN: "_foo bar":

; PR4650
@G6 = weak_odr unnamed_addr constant [1 x i8] c"\01"

; LINUX:   .type	G6,@object
; LINUX:   .section	.rodata.G6,"aG",@progbits,G6,comdat
; LINUX:   .weak	G6
; LINUX: G6:
; LINUX:   .byte	1
; LINUX:   .size	G6, 1

; DARWIN:  .section __TEXT,__const_coal,coalesced
; DARWIN:  .globl _G6
; DARWIN:  .weak_definition _G6
; DARWIN:_G6:
; DARWIN:  .byte 1


@G7 = unnamed_addr constant [10 x i8] c"abcdefghi\00"

; DARWIN:	__TEXT,__cstring,cstring_literals
; DARWIN:	.globl _G7
; DARWIN: _G7:
; DARWIN:	.asciz	"abcdefghi"

; LINUX:	.section	.rodata.str1.1,"aMS",@progbits,1
; LINUX:	.globl G7
; LINUX: G7:
; LINUX:	.asciz	"abcdefghi"

; LINUX-SECTIONS: .section        .rodata.G7,"aMS",@progbits,1
; LINUX-SECTIONS:	.globl G7


@G8 = unnamed_addr constant [4 x i16] [ i16 1, i16 2, i16 3, i16 0 ]

; DARWIN:	.section	__TEXT,__const
; DARWIN:	.globl _G8
; DARWIN: _G8:

; LINUX:	.section	.rodata.str2.2,"aMS",@progbits,2
; LINUX:	.globl G8
; LINUX:G8:

@G9 = unnamed_addr constant [4 x i32] [ i32 1, i32 2, i32 3, i32 0 ]

; DARWIN:	.globl _G9
; DARWIN: _G9:

; LINUX:	.section	.rodata.str4.4,"aMS",@progbits,4
; LINUX:	.globl G9
; LINUX:G9


@G10 = weak global [100 x i32] zeroinitializer, align 32 ; <[100 x i32]*> [#uses=0]


; DARWIN: 	.section	__DATA,__datacoal_nt,coalesced
; DARWIN: .globl _G10
; DARWIN:	.weak_definition _G10
; DARWIN:	.align	5
; DARWIN: _G10:
; DARWIN:	.space	400

; LINUX:	.bss
; LINUX:	.weak	G10
; LINUX:	.align	32
; LINUX: G10:
; LINUX:	.zero	400



;; Zero sized objects should round up to 1 byte in zerofill directives.
; rdar://7886017
@G11 = global [0 x i32] zeroinitializer
@G12 = global {} zeroinitializer
@G13 = global { [0 x {}] } zeroinitializer

; DARWIN: .globl _G11
; DARWIN: .zerofill __DATA,__common,_G11,1,2
; DARWIN: .globl _G12
; DARWIN: .zerofill __DATA,__common,_G12,1,3
; DARWIN: .globl _G13
; DARWIN: .zerofill __DATA,__common,_G13,1,3
