;; Test to ensure dropping of type tests can handle a select feeding the assume.
;; This pattern is created by SimplifyCFG when merging two type test + assume
;; sequences from different branches.
; RUN: opt -S -passes=drop-type-tests %s | FileCheck %s

; CHECK-LABEL: define void @test
define void @test(ptr %p, i1 %cond) {
  %tt1 = call i1 @llvm.type.test(ptr %p, metadata !"_ZTS4Base")
  %tt2 = call i1 @llvm.type.test(ptr %p, metadata !"_ZTS7Derived")
; CHECK-NOT: @llvm.type.test
  %sel = select i1 %cond, i1 %tt2, i1 %tt1
  call void @llvm.assume(i1 %sel)
; CHECK: %sel = select i1 %cond, i1 true, i1 true
; CHECK: call void @llvm.assume(i1 %sel)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)
