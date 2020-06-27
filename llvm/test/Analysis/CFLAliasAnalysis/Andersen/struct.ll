; Ensures that our struct ops are sane.

; RUN: opt < %s -disable-basic-aa -cfl-anders-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; Since we ignore non-pointer values, we effectively ignore extractvalue
; instructions. This means that %c "doesn't exist" in test_structure's graph,
; so we currently get MayAlias.
; XFAIL: *

; CHECK-LABEL: Function: test_structure
; CHECK: NoAlias: i64** %c, { i64**, i64** }* %a
define void @test_structure() {
  %a = alloca {i64**, i64**}, align 8
  %b = load {i64**, i64**}, {i64**, i64**}* %a
  %c = extractvalue {i64**, i64**} %b, 0
  ret void
}
