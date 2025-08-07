; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s

define void @test(ptr %g) {
; CHECK-LABEL: test
; CHECK: NoAlias:      i32* %g, i32* %p1.p
  %p1.p = call ptr @llvm.experimental.provenance.begin(ptr %g)
  %r1 = load i32, ptr %g
  %r2 = load i32, ptr %p1.p
  call void @llvm.experimental.provenance.end(ptr %p1.p)
  ret void
}