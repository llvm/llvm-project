; Make sure LTO succeeds even if %t.bc contains a GlobalVariable F and
; %t2.bc cointains a Function F with the same GUID.
;
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/guid_collision.ll -o %t2.bc
; RUN: llvm-lto2 run %t.bc %t2.bc -o %t.out \
; RUN: -r=%t.bc,dummy,px -r=%t2.bc,dummy2,px

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; The source for the GUID for this symbol will be -:F
source_filename = "-"
@F = internal constant i8 0

; Needed to give llvm-lto2 something to do
@dummy = global i32 0
