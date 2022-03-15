; RUN: opt %s -S -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s
;
; Regression: we weren't properly checking constexpr selects.

@g = extern_weak dso_local global i32, align 4
@g2 = extern_weak dso_local global i32, align 4
@g3 = extern_weak dso_local global i32, align 4

; CHECK-LABEL: Function: foo
; CHECK-DAG: NoAlias: i32** %a, i32* select (i1 icmp ne (i32* @g, i32* null), i32* @g2, i32* @g3)
; CHECK-DAG: NoAlias: i32** %a, i32* %b
; CHECK-DAG: MayAlias: i32* %b, i32* select (i1 icmp ne (i32* @g, i32* null), i32* @g2, i32* @g3)
; CHECK-DAG: NoAlias: i32** %a, i32* @g2
; CHECK-DAG: MayAlias: i32* @g2, i32* select (i1 icmp ne (i32* @g, i32* null), i32* @g2, i32* @g3)
; CHECK-DAG: MayAlias: i32* %b, i32* @g2
; CHECK-DAG: NoAlias: i32** %a, i32* @g3
; CHECK-DAG: MayAlias: i32* @g3, i32* select (i1 icmp ne (i32* @g, i32* null), i32* @g2, i32* @g3)
; CHECK-DAG: MayAlias: i32* %b, i32* @g3
; CHECK-DAG: MayAlias: i32* @g2, i32* @g3

define void @foo() {
entry:
  %a = alloca i32*, align 8
  load i32, i32* select (i1 icmp ne (i32* @g, i32* null), i32* @g2, i32* @g3)
  store i32* select (i1 icmp ne (i32* @g, i32* null), i32* @g2, i32* @g3), i32** %a
  %b = load i32*, i32** %a
  %c = load i32, i32* %b
  %d = load i32, i32* @g2
  %e = load i32, i32* @g3
  ret void
}
