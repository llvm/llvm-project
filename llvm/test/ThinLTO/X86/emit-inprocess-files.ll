; Test InProcessThinLTO thin link output from llvm-lto2
; Partially copied from distributed_import.ll, and added checks for {dis|en}abling imports

; RUN: rm -f %t1.bc.thinlto.bc %t2.bc.thinlto.bc %t.out.1 %t.out.2 %t1.bc.imports %t2.bc.imports

; Generate bitcode files with summary, as well as minimized bitcode containing just the summary
; RUN: opt -thinlto-bc %s -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %p/Inputs/distributed_import.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc

; First perform the thin link on the normal bitcode file using
; -thinlto-distributed-indexes, collecting outputs to be compared with later.
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -thinlto-distributed-indexes \
; RUN:     -thinlto-emit-imports \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,analias, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px \
; RUN:     -r=%t2.bc,analias,px \
; RUN:     -r=%t2.bc,aliasee,px
; RUN: mv %t1.bc.thinlto.bc %t1.bc.thinlto.bc.orig
; RUN: mv %t2.bc.thinlto.bc %t2.bc.thinlto.bc.orig
; RUN: mv %t1.bc.imports %t1.bc.imports.orig
; RUN: mv %t2.bc.imports %t2.bc.imports.orig

; Now use -thinlto-emit-indexes instead.
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -thinlto-emit-indexes \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,analias, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px \
; RUN:     -r=%t2.bc,analias,px \
; RUN:     -r=%t2.bc,aliasee,px \
; RUN:     -o=%t.out

; Since InProcessThinLTO ran, there should be output
; RUN: ls %t.out.1
; RUN: ls %t.out.2

; Ensure imports weren't generated since -thinlto-emit-imports wasn't specified
; RUN: not ls %t1.bc.imports
; RUN: not ls %t2.bc.imports

; Compare the generated index files.
; RUN: diff %t1.bc.thinlto.bc %t1.bc.thinlto.bc.orig
; RUN: diff %t2.bc.thinlto.bc %t2.bc.thinlto.bc.orig

; RUN: rm -f %t1.bc.thinlto.bc %t2.bc.thinlto.bc %t.out.1 %t.out.2

; Do the thin link again but also emit imports files now
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -thinlto-emit-indexes \
; RUN:     -thinlto-emit-imports \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,analias, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px \
; RUN:     -r=%t2.bc,analias,px \
; RUN:     -r=%t2.bc,aliasee,px \
; RUN:     -o=%t.out

; Check the output
; RUN: ls %t.out.1
; RUN: ls %t.out.2
; RUN: diff %t1.bc.thinlto.bc %t1.bc.thinlto.bc.orig
; RUN: diff %t2.bc.thinlto.bc %t2.bc.thinlto.bc.orig
; RUN: diff %t1.bc.imports %t1.bc.imports.orig
; RUN: diff %t2.bc.imports %t2.bc.imports.orig

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare i32 @g(...)
declare void @analias(...)

define void @f() {
entry:
  call i32 (...) @g()
  call void (...) @analias()
  ret void
}

!llvm.dbg.cu = !{}

!1 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!1}
