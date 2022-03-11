; This testcase ensures that CFL AA correctly handles simple memory alias 
; pattern

; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test_memalias
; CHECK: NoAlias: i64* %a, i64** %b
; CHECK: NoAlias: i32** %c, i64* %a
; CHECK: MayAlias: i32* %d, i64* %a
; CHECK: NoAlias: i32* %d, i64** %b
; CHECK: NoAlias: i32* %d, i32** %c
define void @test_memalias() {
  %a = alloca i64, align 8
  %b = alloca i64*, align 8
  store i64* %a, i64** %b

  %c = bitcast i64** %b to i32**
  %d = load i32*, i32** %c
  ret void
}