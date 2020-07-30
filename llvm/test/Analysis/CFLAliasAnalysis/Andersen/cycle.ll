; This testcase ensures that CFL AA handles assignment cycles correctly

; RUN: opt < %s -disable-basic-aa -cfl-anders-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test_cycle
; CHECK: NoAlias: i64* %a, i64** %b
; CHECK: NoAlias: i64* %a, i64*** %c
; CHECK: NoAlias: i64** %b, i64*** %c
; CHECK: NoAlias: i64* %a, i64**** %d
; CHECK: NoAlias: i64** %b, i64**** %d
; CHECK: NoAlias: i64*** %c, i64**** %d
; CHECK: NoAlias: i64* %a, i64* %e
; CHECK: NoAlias: i64* %e, i64** %b
; CHECK: NoAlias: i64* %e, i64*** %c
; CHECK: MayAlias: i64* %a, i64* %f
; CHECK: NoAlias: i64* %f, i64** %b
; CHECK: NoAlias: i64* %f, i64*** %c
; CHECK: MayAlias: i64* %f, i64**** %d
; CHECK: MayAlias: i64* %e, i64* %f
define void @test_cycle() {
  %a = alloca i64, align 8
  %b = alloca i64*, align 8
  %c = alloca i64**, align 8
  %d = alloca i64***, align 8
  store i64* %a, i64** %b
  store i64** %b, i64*** %c
  store i64*** %c, i64**** %d

  %e = bitcast i64**** %d to i64*
  store i64* %e, i64** %b
  %f = load i64*, i64** %b
  ret void
}
