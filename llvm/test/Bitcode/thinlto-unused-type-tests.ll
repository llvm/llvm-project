; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s

; CHECK-NOT: <TYPE_TESTS
define void @f() {
  %p = call i1 @llvm.type.test(i8* null, metadata !"foo")
  %q = call i1 @llvm.type.test(i8* null, metadata !"bar")
  call void @llvm.assume(i1 %q)
  ret void
}

declare i1 @llvm.type.test(i8*, metadata) nounwind readnone
declare void @llvm.assume(i1)
