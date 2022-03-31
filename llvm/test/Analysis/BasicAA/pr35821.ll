; RUN: opt %s -passes=aa-eval -disable-output -print-all-alias-modref-info 2>&1 | FileCheck %s

; CHECK-LABEL: Function: patatino
; CHECK: NoAlias: i1* %G26, i1** %G47

define void @patatino() {
  %G26 = getelementptr i1, i1* undef, i1 undef
  %B20 = shl i8 -128, 16
  %G47 = getelementptr i1*, i1** undef, i8 %B20
  load i1, i1* %G26
  load i1*, i1** %G47
  ret void
}
