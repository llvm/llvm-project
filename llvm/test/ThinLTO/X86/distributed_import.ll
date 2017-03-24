; Test distributed build thin link output from llvm-lto2

; Generate bitcode files with summary, as well as minimized bitcode without
; the debug metadata for the thin link.
; RUN: opt -thinlto-bc %s -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %p/Inputs/distributed_import.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc

; First perform the thin link on the normal bitcode file.
; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -thinlto-distributed-indexes \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px
; RUN: opt -function-import -summary-file %t1.bc.thinlto.bc %t1.bc -o %t1.out
; RUN: opt -function-import -summary-file %t2.bc.thinlto.bc %t2.bc -o %t2.out
; RUN: llvm-dis -o - %t2.out | FileCheck %s

; Save the generated index files.
; RUN: cp %t1.bc.thinlto.bc %t1.bc.thinlto.bc.orig
; RUN: cp %t2.bc.thinlto.bc %t2.bc.thinlto.bc.orig

; Copy the minimized bitcode to the regular bitcode path so the module
; paths in the index are the same (save the regular bitcode for use again
; further down).
; RUN: cp %t1.bc %t1.bc.sv
; RUN: cp %t1.thinlink.bc %t1.bc
; RUN: cp %t2.bc %t2.bc.sv
; RUN: cp %t2.thinlink.bc %t2.bc

; Next perform the thin link on the minimized bitcode files, and compare dumps
; of the resulting indexes to the above dumps to ensure they are identical.
; RUN: rm -f %t1.bc.thinlto.bc %t2.bc.thinlto.bc
; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -thinlto-distributed-indexes \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px
; RUN: diff %t1.bc.thinlto.bc.orig %t1.bc.thinlto.bc
; RUN: diff %t2.bc.thinlto.bc.orig %t2.bc.thinlto.bc

; Make sure importing occurs as expected
; RUN: cp %t1.bc.sv %t1.bc
; RUN: cp %t2.bc.sv %t2.bc
; RUN: opt -function-import -summary-file %t2.bc.thinlto.bc %t2.bc -o %t2.out
; RUN: llvm-dis -o - %t2.out | FileCheck %s

; CHECK: @G.llvm.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i32 @g(...)

define void @f() {
entry:
  call i32 (...) @g()
  ret void
}

!llvm.dbg.cu = !{}

!1 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!1}
