; RUN: opt %s -passes=aa-eval -disable-output -print-all-alias-modref-info 2>&1 | FileCheck %s

; CHECK-LABEL: Function: patatino
; CHECK: NoAlias: i1** %G22, i1*** %G45

define void @patatino() {
BB:
  %G22 = getelementptr i1*, i1** undef, i8 -1
  %B1 = mul i66 undef, 9223372036854775808
  %G45 = getelementptr i1**, i1*** undef, i66 %B1
  load i1*, i1** %G22
  load i1**, i1*** %G45
  ret void
}
