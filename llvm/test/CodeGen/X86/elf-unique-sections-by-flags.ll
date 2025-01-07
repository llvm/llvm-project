; Test that global values with the same specified section produces multiple
; sections with different sets of flags, depending on the properties (mutable,
; executable) of the global value.

; RUN: llc < %s | FileCheck %s
; RUN: llc -function-sections < %s | FileCheck %s --check-prefix=CHECK --check-prefix=FNSECTIONS
target triple="x86_64-unknown-unknown-elf"

; Normal function goes in .text, or in it's own named section with -function-sections.
define i32 @fn_text() {
    entry:
    ret i32 0
}
; FNSECTIONS:   .section	.text.fn_text,"ax",@progbits{{$}}
; CHECK:        .globl fn_text
; CHECK:        fn_text:

; A second function placed in .text, to check the behaviour with -function-sections.
; It should be emitted to a new section with a new name, not expected to require unique.
define i32 @fn_text2() {
    entry:
    ret i32 0
}
; FNSECTIONS:   .section	.text.fn_text2,"ax",@progbits{{$}}
; CHECK:        .globl fn_text2
; CHECK:        fn_text2:

; Functions in user defined executable sections
define i32 @fn_s1() section "s1" {
    entry:
    ret i32 0
}
; CHECK:        .section s1,"ax",@progbits{{$}}
; CHECK-NEXT:   .globl fn_s1
; CHECK:        fn_s1:

define i32 @fn_s2() section "s2" {
    entry:
    ret i32 0
}
; CHECK:        .section s2,"ax",@progbits{{$}}
; CHECK-NEXT:   .globl fn_s2
; CHECK:        fn_s2:

; A second function in s2 should share the same .section
define i32 @fn2_s2() section "s2" {
    entry:
    ret i32 0
}
; CHECK-NOT:    .section
; CHECK:        .globl fn2_s2
; CHECK:        fn2_s2:

; Values that share a section name with a function are placed in different sections without executable flag
@rw_s1 = global i32 10, section "s1", align 4
@ro_s2 = constant i32 10, section "s2", align 4
; CHECK:        .section s1,"aw",@progbits,unique,[[#UNIQUE_S1_aw:]]
; CHECK-NEXT:   .globl rw_s1
; CHECK:        rw_s1:
; CHECK:        .section s2,"a",@progbits,unique,[[#UNIQUE_S2_a:]]
; CHECK-NEXT:   .globl ro_s2
; CHECK:        ro_s2:

; Placing another value in the same section with the same flags uses the same unique ID
@rw2_s1 = global i32 10, section "s1", align 4
@ro2_s2 = constant i32 10, section "s2", align 4
; CHECK:        .section s1,"aw",@progbits,unique,[[#UNIQUE_S1_aw]]
; CHECK-NEXT:   .globl rw2_s1
; CHECK:        rw2_s1:
; CHECK:        .section s2,"a",@progbits,unique,[[#UNIQUE_S2_a]]
; CHECK-NEXT:   .globl ro2_s2
; CHECK:        ro2_s2:

; Normal user defined section, first is the generic section, second should be unique
@ro_s3 = constant i32 10, section "s3", align 4
@rw_s3 = global i32 10, section "s3", align 4
; CHECK:        .section s3,"a",@progbits{{$}}
; CHECK-NEXT:   .globl ro_s3
; CHECK:        ro_s3:
; CHECK:        .section s3,"aw",@progbits,unique,[[#U:]]
; CHECK-NEXT:   .globl rw_s3
; CHECK:        rw_s3:

; Values declared without explicit sections go into compatible default sections and don't require unique
@rw_nosec = global i32 10, align 4
@ro_nosec = constant i32 10, align 4
; CHECK:        .data{{$}}
; CHECK-NEXT:   .globl rw_nosec
; CHECK:        rw_nosec:
; CHECK:        .section .rodata,"a",@progbits{{$}}
; CHECK-NEXT:   .globl ro_nosec
; CHECK:        ro_nosec:

; Explicitly placed in .rodata with writeable set. The writable section should be uniqued, not the default ro section, even if it comes first.
@rw_rodata = global [2 x i32] zeroinitializer, section ".rodata", align 4
@ro_rodata = constant [2 x i32] zeroinitializer, section ".rodata", align 4
; CHECK:        .section .rodata,"aw",@progbits,unique,[[#U+1]]{{$}}
; CHECK-NEXT:   .globl rw_rodata{{$}}
; CHECK:        rw_rodata:
; CHECK:        .section .rodata,"a",@progbits{{$}}
; CHECK-NEXT:   .globl ro_rodata{{$}}
; CHECK:        ro_rodata:

; Writable symbols in writable default sections; no need to unique
@w_sdata = global [4 x i32] zeroinitializer, section ".sdata", align 4
@w_sbss = global [4 x i32] zeroinitializer, section ".sbss", align 4
; CHECK:        .section .sdata,"aw",@progbits{{$}}
; CHECK-NEXT:   .globl w_sdata{{$}}
; CHECK:        w_sdata:
; CHECK:        .section .sbss,"aw",@nobits{{$}}
; CHECK-NEXT:   .globl w_sbss{{$}}
; CHECK:        w_sbss:

; Multiple .text sections are emitted for read-only and read-write sections using .text name.
@rw_text = global i32 10, section ".text", align 4
@ro_text = constant i32 10, section ".text", align 4
; CHECK:        .section .text,"aw",@progbits,unique,[[#U+2]]
; CHECK-NEXT:   .globl rw_text
; CHECK:        rw_text:
; CHECK:        .section .text,"a",@progbits,unique,[[#U+3]]
; CHECK-NEXT:   .globl ro_text
; CHECK:        ro_text:

; A read-only .data section is emitted
@ro_data = constant i32 10, section ".data", align 4
; CHECK:        .section .data,"a",@progbits,unique,[[#U+4]]
; CHECK-NEXT:   .globl ro_data
; CHECK:        ro_data:

; TLS and non-TLS symbols cannot live in the same section
@tls_var = thread_local global i32 10, section "s4", align 4
@non_tls_var = global i32 10, section "s4", align 4
; CHECK:        .section s4,"awT",@progbits{{$}}
; CHECK-NEXT:   .globl tls_var
; CHECK:        tls_var:
; CHECK:        .section s4,"aw",@progbits,unique,[[#U+5]]
; CHECK-NEXT:   .globl non_tls_var
; CHECK:        non_tls_var:
