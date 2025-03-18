; RUN: opt -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 < %s | FileCheck %s

declare void @callee(ptr)

; CHECK-LABEL: Function: test
; CHECK: Both ModRef: Ptr: i32* %a.gep <-> call void @callee(ptr %gep)
define void @test(i1 %c, ptr %arg) {
  %a = alloca [2 x i32]
  %a.gep = getelementptr i8, ptr %a, i64 4
  %sel = select i1 %c, ptr %arg, ptr null
  %gep = getelementptr i8, ptr %sel, i64 4
  call void @callee(ptr %gep)
  %l = load i32, ptr %a.gep
  ret void
}
