; RUN: llvm-as -function-summary %s -o %t.bc
; RUN: llvm-as -function-summary %p/Inputs/funcimport_appending_global.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Do the import now
; RUN: llvm-link %t.bc -functionindex=%t3.thinlto.bc -import=foo:%t2.bc -S | FileCheck %s

; Ensure that global constructor (appending linkage) is not imported
; CHECK-NOT: @llvm.global_ctors = {{.*}}@foo

declare void @f()
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @f, i8* null}]

define i32 @main() {
entry:
  call void @foo()
  ret i32 0
}

declare void @foo()
