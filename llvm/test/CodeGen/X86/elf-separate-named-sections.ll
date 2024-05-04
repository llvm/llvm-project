; Test that global values with explicit sections are placed into unique sections.

; RUN: llc < %s | FileCheck %s
; RUN: llc -separate-named-sections < %s | FileCheck %s --check-prefix=SEPARATE
target triple="x86_64-unknown-unknown-elf"

define i32 @f() section "custom_text" {
    entry:
    ret i32 0
}

define i32 @g() section "custom_text" {
    entry:
    ret i32 0
}

; CHECK: .section custom_text,"ax",@progbits{{$}}
; CHECK: f:
; CHECK: g:

; SEPARATE: .section custom_text,"ax",@progbits,unique,1{{$}}
; SEPARATE: f:
; SEPARATE: .section custom_text,"ax",@progbits,unique,2{{$}}
; SEPARATE: g:

@i = global i32 0, section "custom_data", align 8
@j = global i32 0, section "custom_data", align 8

; CHECK: .section custom_data,"aw",@progbits{{$}}
; CHECK: i:
; CHECK: j:

; SEPARATE: .section custom_data,"aw",@progbits,unique,3{{$}}
; SEPARATE: i:
; SEPARATE: .section custom_data,"aw",@progbits,unique,4{{$}}
; SEPARATE: j:
