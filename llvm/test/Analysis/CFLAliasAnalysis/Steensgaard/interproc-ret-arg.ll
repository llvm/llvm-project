; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to return one of its parameters

; RUN: opt < %s -disable-basic-aa -cfl-steens-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

define i32* @return_arg_callee(i32* %arg1, i32* %arg2) {
  ret i32* %arg1
}
; CHECK-LABEL: Function: test_return_arg
; CHECK: NoAlias: i32* %a, i32* %b
; CHECK: MayAlias: i32* %a, i32* %c
; CHECK: NoAlias: i32* %b, i32* %c

; Temporarily disable modref checks
; NoModRef: Ptr: i32* %b <-> %c = call i32* @return_arg_callee(i32* %a, i32* %b)
define void @test_return_arg() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4

  %c = call i32* @return_arg_callee(i32* %a, i32* %b)

  ret void
}