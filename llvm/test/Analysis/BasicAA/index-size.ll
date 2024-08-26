; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output %s 2>&1 | FileCheck %s

target datalayout = "p:64:64:64:32"

; gep.1 and gep.2 must alias, because they are truncated to the index size
; (32-bit), not the pointer size (64-bit).
define void @mustalias_due_to_index_size(ptr %ptr) {
; CHECK-LABEL: Function: mustalias_due_to_index_size
; CHECK-NEXT: MustAlias: i8* %gep.1, i8* %ptr
; CHECK-NEXT: MustAlias: i8* %gep.2, i8* %ptr
; CHECK-NEXT: MustAlias: i8* %gep.1, i8* %gep.2
;
  load i8, ptr %ptr
  %gep.1 = getelementptr i8, ptr %ptr, i64 4294967296
  store i8 0, ptr %gep.1
  %gep.2 = getelementptr i8, ptr %ptr, i64 0
  store i8 1, ptr %gep.2
  ret void
}
