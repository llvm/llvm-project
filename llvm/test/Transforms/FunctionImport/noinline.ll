; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.main.bc
; RUN: opt -module-summary %p/Inputs/noinline.ll -o %t.inputs.noinline.bc
; RUN: llvm-lto -thinlto -o %t.summary %t.main.bc %t.inputs.noinline.bc

; Attempt the import now, ensure below that file containing noinline
; is not imported by default but imported with -force-import-all.

; RUN: opt -passes=function-import -summary-file %t.summary.thinlto.bc %t.main.bc -S 2>&1 \
; RUN:   | FileCheck -check-prefix=NOIMPORT %s
; RUN: opt -passes=function-import -force-import-all -summary-file %t.summary.thinlto.bc \
; RUN:   %t.main.bc -S 2>&1 | FileCheck -check-prefix=IMPORT %s

define i32 @main() #0 {
entry:
  %f = alloca i64, align 8
  call void @foo(i64* %f)
  ret i32 0
}

; NOIMPORT: declare void @foo(i64*)
; IMPORT: define available_externally void @foo
declare void @foo(i64*) #1
