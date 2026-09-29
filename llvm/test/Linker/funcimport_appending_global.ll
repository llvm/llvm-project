; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_appending_global.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Do the import now
; RUN: llvm-link %t.bc -summary-index=%t3.thinlto.bc -import=foo:%t2.bc -S | FileCheck %s

; Ensure that global constructor (appending linkage) is not imported
; CHECK-NOT: @llvm.global_ctors = {{.*}}@foo

declare void @f()
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @f, ptr null}]

define i32 @main() {
entry:
  call void @foo()
  ret i32 0
}

declare void @foo()
