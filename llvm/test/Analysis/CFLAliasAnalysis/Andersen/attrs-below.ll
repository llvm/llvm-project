; This testcase ensures that AliasAttrs are propagated not only on the same 
; level but also downward.

; RUN: opt < %s -disable-basicaa -cfl-anders-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test_attr_below
; CHECK: MayAlias: i64* %q, i64*** %p
; CHECK: NoAlias: i64* %esc, i64*** %p
; CHECK: NoAlias: i64* %esc, i64* %q

; CHECK: MayAlias: i64* %unknown, i64*** %p
; CHECK: MayAlias: i64* %q, i64* %unknown
; CHECK: MayAlias: i64* %esc, i64* %unknown
; CHECK: MayAlias: i64* %q, i64** %pdrf
; CHECK: MayAlias: i64* %esc, i64** %pdrf
; CHECK: MayAlias: i64* %unknown, i64** %pdrf
; CHECK: MayAlias: i64* %pdrf2, i64* %q
; CHECK: MayAlias: i64* %esc, i64* %pdrf2
; CHECK: MayAlias: i64* %pdrf2, i64* %unknown
define void @test_attr_below(i64*** %p, i64* %q) {
  %esc = alloca i64, align 8
  %escint = ptrtoint i64* %esc to i64
  %unknown = inttoptr i64 %escint to i64*

  %pdrf = load i64**, i64*** %p
  %pdrf2 = load i64*, i64** %pdrf

  ret void
}

; CHECK-LABEL: Function: test_attr_assign_below
; CHECK: MayAlias: i64** %sel, i64*** %p
; CHECK: MayAlias: i64* %q, i64** %sel
; CHECK: MayAlias: i64** %a, i64** %sel
; CHECK: MayAlias: i64** %pdrf, i64** %sel

; CHECK: MayAlias: i64** %c, i64*** %p
; CHECK: MayAlias: i64* %q, i64** %c
; CHECK: MayAlias: i64** %a, i64** %c
; CHECK: MayAlias: i64** %c, i64** %pdrf
; CHECK: MayAlias: i64** %c, i64** %sel

; CHECK: MayAlias: i64* %d, i64*** %p
; CHECK: MayAlias: i64* %d, i64* %q
; CHECK: MayAlias: i64* %d, i64** %pdrf
; CHECK: MayAlias: i64* %d, i64** %sel
define void @test_attr_assign_below(i64*** %p, i64* %q, i1 %cond) {
  %a = alloca i64*, align 8
  %pdrf = load i64**, i64*** %p
  %sel = select i1 %cond, i64** %a, i64** %pdrf

  %b = alloca i64**, align 8
  store i64** %sel, i64*** %b

  %c = load i64**, i64*** %b
  %d = load i64*, i64** %c

  ret void
}

