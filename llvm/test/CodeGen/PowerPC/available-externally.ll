; RUN: llc < %s -relocation-model=static | FileCheck %s -check-prefix=STATIC
; RUN: llc < %s -relocation-model=pic -mtriple=powerpc-apple-darwin9 | FileCheck %s -check-prefix=PIC
; RUN: llc < %s -relocation-model=pic -mtriple=powerpc-unknown-linux | FileCheck %s -check-prefix=PICELF
; RUN: llc < %s -relocation-model=pic -mtriple=powerpc64-apple-darwin9 | FileCheck %s -check-prefix=PIC64
; RUN: llc < %s -relocation-model=dynamic-no-pic -mtriple=powerpc-apple-darwin9 | FileCheck %s -check-prefix=DYNAMIC
; RUN: llc < %s -relocation-model=dynamic-no-pic -mtriple=powerpc64-apple-darwin9 | FileCheck %s -check-prefix=DYNAMIC64
; PR4482
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "powerpc-apple-darwin9"

define i32 @foo(i64 %x) nounwind {
entry:
; STATIC: _foo:
; STATIC: bl _exact_log2
; STATIC: blr
; STATIC: .subsections_via_symbols

; PIC: _foo:
; PIC: bl _exact_log2
; PIC: blr

; PICELF: foo:
; PICELF: bl exact_log2@PLT
; PICELF: blr

; PIC64: _foo:
; PIC64: bl _exact_log2
; PIC64: blr

; DYNAMIC: _foo:
; DYNAMIC: bl _exact_log2
; DYNAMIC: blr

; DYNAMIC64: _foo:
; DYNAMIC64: bl _exact_log2
; DYNAMIC64: blr

        %A = call i32 @exact_log2(i64 %x) nounwind
	ret i32 %A
}

define available_externally i32 @exact_log2(i64 %x) nounwind {
entry:
	ret i32 42
}


; PIC: .subsections_via_symbols


; PIC64: .subsections_via_symbols
