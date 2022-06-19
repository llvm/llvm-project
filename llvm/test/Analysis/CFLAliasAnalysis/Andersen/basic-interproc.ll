; This testcase ensures that CFL AA won't be too conservative when trying to do
; interprocedural analysis on simple callee

; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: noop_callee
; CHECK: MayAlias: i32* %arg1, i32* %arg2
define void @noop_callee(i32* %arg1, i32* %arg2) {
  store i32 0, i32* %arg1
  store i32 0, i32* %arg2
  ret void
}
; CHECK-LABEL: Function: test_noop
; CHECK: NoAlias: i32* %a, i32* %b
define void @test_noop() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  load i32, i32* %a
  load i32, i32* %b
  call void @noop_callee(i32* %a, i32* %b)

  ret void
}
