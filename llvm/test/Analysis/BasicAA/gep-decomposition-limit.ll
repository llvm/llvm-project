; RUN: opt -S -basic-aa -aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test
;; Before limit:
; CHECK-DAG: MustAlias: i8* %gep.add5, i8* %gep.inc5
; CHECK-DAG: NoAlias: i8* %gep.inc3, i8* %gep.inc5
; CHECK-DAG: NoAlias: i8* %gep.inc4, i8* %gep.inc5
;; At limit:
; CHECK-DAG: MustAlias: i8* %gep.add6, i8* %gep.inc6
; CHECK-DAG: NoAlias: i8* %gep.inc4, i8* %gep.inc6
; CHECK-DAG: NoAlias: i8* %gep.inc5, i8* %gep.inc6
;; After limit:
; CHECK-DAG: MayAlias: i8* %gep.add7, i8* %gep.inc7
; CHECK-DAG: MayAlias: i8* %gep.inc5, i8* %gep.inc7
; CHECK-DAG: NoAlias: i8* %gep.inc6, i8* %gep.inc7

define void @test(i8* %base) {
  %gep.add5 = getelementptr i8, i8* %base, i64 5
  %gep.add6 = getelementptr i8, i8* %base, i64 6
  %gep.add7 = getelementptr i8, i8* %base, i64 7

  %gep.inc1 = getelementptr i8, i8* %base, i64 1
  %gep.inc2 = getelementptr i8, i8* %gep.inc1, i64 1
  %gep.inc3 = getelementptr i8, i8* %gep.inc2, i64 1
  %gep.inc4 = getelementptr i8, i8* %gep.inc3, i64 1
  %gep.inc5 = getelementptr i8, i8* %gep.inc4, i64 1
  %gep.inc6 = getelementptr i8, i8* %gep.inc5, i64 1
  %gep.inc7 = getelementptr i8, i8* %gep.inc6, i64 1

  ret void
}
