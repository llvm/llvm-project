; Make sure LTO succeeds even if %t.bc contains a GlobalVariable F and
; %t2.bc cointains a Function F with the same GUID.
;
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/guid_collision.ll -o %t2.bc
; RUN: llvm-lto2 run %t.bc %t2.bc -o %t.out -save-temps \
; RUN: -r=%t.bc,H,px -r=%t.bc,G, -r=%t2.bc,G,px
; RUN: llvm-dis -o - %t.out.1.3.import.bc | FileCheck %s

; RUN: llvm-lto2 run %t.bc %t2.bc -o %t.out -thinlto-distributed-indexes \
; RUN: -r=%t.bc,H,px -r=%t.bc,G, -r=%t2.bc,G,px
; RUN: opt -function-import -import-all-index -summary-file %t.bc.thinlto.bc %t.bc -o %t.out
; RUN: llvm-dis -o - %t.out | FileCheck %s

; Sanity check that G was imported
; CHECK: define available_externally i64 @G

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; The source for the GUID for this symbol will be -:F
source_filename = "-"

@F = internal constant i8 0

; Provide a global that has the same name as one from the module we import G
; from, to test handling of a global variable with an entry in the distributed
; index but not with a copy in the source module (since we can't import
; appending linkage globals).
@llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer

define i64 @H() {
  call i64 @G()
  ret i64 0
}

declare i64 @G()
