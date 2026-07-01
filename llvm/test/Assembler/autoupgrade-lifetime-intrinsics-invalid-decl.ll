; Test to verify that autoupgrade does not end up creating an invalid
; declaration.
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK-NOT: declare void @llvm.lifetime.start.i64(i64)
; CHECK-NOT: declare void @llvm.lifetime.end.i64(i64)

define void @tests.lifetime.start.end() {
  %a = alloca i8
  call void @llvm.lifetime.start(i64 1, ptr %a)
  store i8 0, ptr %a
  call void @llvm.lifetime.end(i64 1, ptr %a)
  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr)
declare void @llvm.lifetime.end.p0(i64, ptr)
