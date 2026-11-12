; RUN: llc -verify-machineinstrs < %s | FileCheck %s -check-prefix=STATIC
; RUN: llc -verify-machineinstrs < %s -relocation-model=static | FileCheck %s -check-prefix=STATIC
; RUN: llc -verify-machineinstrs < %s -relocation-model=pic -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=PIC
; RUN: llc -verify-machineinstrs < %s -relocation-model=pic -mtriple=powerpc-unknown-linux | FileCheck %s -check-prefix=PICELF
; RUN: llc -verify-machineinstrs < %s -relocation-model=pic -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=PIC64

;;; KB: These two tests currently cause an assertion. It seems as though we cannot have a non DSOLocal symbol with dynamic-no-pic.
;;;     I need to ask Sean about this.
;;; RUN-NOT: llc -verify-machineinstrs < %s -relocation-model=dynamic-no-pic -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=DYNAMIC
;;; RUN-NOT: llc -verify-machineinstrs < %s -relocation-model=dynamic-no-pic -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=DYNAMIC64
; PR4482
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "powerpc-unknown-linux-gnu"

define i32 @foo(i64 %x) nounwind {
entry:
; STATIC: foo:
; STATIC: bl exact_log2
; STATIC: blr

; PIC: foo:
; PIC: bl exact_log2@PLT
; PIC: blr

; PICELF: foo:
; PICELF: bl exact_log2@PLT
; PICELF: blr

; PIC64: foo:
; PIC64: bl exact_log2
; PIC64: blr

; DYNAMIC: foo:
; DYNAMIC: bl exact_log2@PLT
; DYNAMIC: blr

; DYNAMIC64: foo:
; DYNAMIC64: bl exact_log2@PPLT
; DYNAMIC64: blr

        %A = call i32 @exact_log2(i64 %x) nounwind
	ret i32 %A
}

define available_externally i32 @exact_log2(i64 %x) nounwind {
entry:
	ret i32 42
}


