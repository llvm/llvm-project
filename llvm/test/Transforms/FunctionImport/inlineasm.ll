; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/inlineasm.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Attempt the import now, ensure below that file containing inline assembly
; is not imported from. Otherwise we would need to promote its local variable
; used in the inline assembly, which would not see the rename.
; RUN: opt -passes=function-import -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s

define i32 @main() #0 {
entry:
  %f = alloca i64, align 8
  call void @foo(ptr %f)
  ret i32 0
}

; CHECK: declare void @foo(ptr)
declare void @foo(ptr) #1
