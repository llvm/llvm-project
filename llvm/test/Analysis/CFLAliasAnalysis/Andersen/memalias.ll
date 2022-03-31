; This testcase ensures that CFL AA correctly handles simple memory alias 
; pattern

; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test_memalias
; CHECK: NoAlias: i64* %a, i64** %b
; CHECK: NoAlias: i64* %a, i32** %c
; CHECK: MayAlias: i64* %a, i32* %d
; CHECK: NoAlias: i64** %b, i32* %d
; CHECK: NoAlias: i32** %c, i32* %d
define void @test_memalias() {
  %a = alloca i64, align 8
  %b = alloca i64*, align 8
  load i64, i64* %a
  store i64* %a, i64** %b

  %c = bitcast i64** %b to i32**
  %d = load i32*, i32** %c
  load i32, i32* %d
  ret void
}
