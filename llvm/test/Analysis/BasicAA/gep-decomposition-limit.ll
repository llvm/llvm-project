; RUN: opt -S -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test
;; Before limit:
; CHECK-DAG: MustAlias: i8* %gep.add9, i8* %gep.inc9
; CHECK-DAG: NoAlias: i8* %gep.inc7, i8* %gep.inc9
; CHECK-DAG: NoAlias: i8* %gep.inc8, i8* %gep.inc9
;; At limit:
; CHECK-DAG: MustAlias: i8* %gep.add10, i8* %gep.inc10
; CHECK-DAG: NoAlias: i8* %gep.inc10, i8* %gep.inc8
; CHECK-DAG: NoAlias: i8* %gep.inc10, i8* %gep.inc9
;; After limit:
; CHECK-DAG: MayAlias: i8* %gep.add11, i8* %gep.inc11
; CHECK-DAG: MayAlias: i8* %gep.inc11, i8* %gep.inc9
; CHECK-DAG: NoAlias: i8* %gep.inc10, i8* %gep.inc11

define void @test(ptr %base) {
  %gep.add9 = getelementptr i8, ptr %base, i64 9
  %gep.add10 = getelementptr i8, ptr %base, i64 10
  %gep.add11 = getelementptr i8, ptr %base, i64 11

  %gep.inc1 = getelementptr i8, ptr %base, i64 1
  %gep.inc2 = getelementptr i8, ptr %gep.inc1, i64 1
  %gep.inc3 = getelementptr i8, ptr %gep.inc2, i64 1
  %gep.inc4 = getelementptr i8, ptr %gep.inc3, i64 1
  %gep.inc5 = getelementptr i8, ptr %gep.inc4, i64 1
  %gep.inc6 = getelementptr i8, ptr %gep.inc5, i64 1
  %gep.inc7 = getelementptr i8, ptr %gep.inc6, i64 1
  %gep.inc8 = getelementptr i8, ptr %gep.inc7, i64 1
  %gep.inc9 = getelementptr i8, ptr %gep.inc8, i64 1
  %gep.inc10 = getelementptr i8, ptr %gep.inc9, i64 1
  %gep.inc11 = getelementptr i8, ptr %gep.inc10, i64 1

  load i8, ptr %gep.add9
  load i8, ptr %gep.add10
  load i8, ptr %gep.add11
  load i8, ptr %gep.inc3
  load i8, ptr %gep.inc4
  load i8, ptr %gep.inc5
  load i8, ptr %gep.inc6
  load i8, ptr %gep.inc7
  load i8, ptr %gep.inc8
  load i8, ptr %gep.inc9
  load i8, ptr %gep.inc10
  load i8, ptr %gep.inc11

  ret void
}
