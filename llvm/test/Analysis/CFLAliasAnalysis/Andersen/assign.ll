; This testcase ensures that CFL AA handles assignment in an inclusion-based 
; manner

; RUN: opt < %s -disable-basicaa -cfl-anders-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test_assign
; CHECK: NoAlias: i64* %a, i64* %b
; CHECK: NoAlias: i32* %c, i64* %b
; CHECK: NoAlias: i32* %d, i64* %a
; CHECK: NoAlias: i32* %c, i32* %d
; CHECK: MayAlias: i32* %e, i64* %a
; CHECK: MayAlias: i32* %e, i64* %b
; CHECK: MayAlias: i32* %c, i32* %e
; CHECK: MayAlias: i32* %d, i32* %e
define void @test_assign(i1 %cond) {
  %a = alloca i64, align 8
  %b = alloca i64, align 8

  %c = bitcast i64* %a to i32*
  %d = bitcast i64* %b to i32*
  %e = select i1 %cond, i32* %c, i32* %d
  ret void
}