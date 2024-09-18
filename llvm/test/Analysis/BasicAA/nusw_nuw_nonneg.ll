; RUN: opt < %s -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: test
; CHECK: NoAlias:	i8* %p.minus.2, i8* %p.plus.2
; CHECK: MayAlias:	i8* %p.idx.maybeneg, i8* %p.minus.2
; CHECK: MayAlias:	i8* %p.idx.maybeneg, i8* %p.plus.2
; CHECK: NoAlias:	i8* %p.idx.nneg, i8* %p.minus.2
; CHECK: MayAlias:	i8* %p.idx.nneg, i8* %p.plus.2
; CHECK: MustAlias:	i8* %p.idx.maybeneg, i8* %p.idx.nneg
define void @test(ptr %p, i64 %idx) {
  %p.minus.2 = getelementptr i8, ptr %p, i64 -2
  %p.plus.2 = getelementptr i8, ptr %p, i64 2
  %p.idx.maybeneg = getelementptr inbounds i8, ptr %p, i64 %idx
  %p.idx.nneg = getelementptr nuw nusw i8, ptr %p, i64 %idx
  load i8, ptr %p.minus.2
  load i8, ptr %p.plus.2
  load i8, ptr %p.idx.maybeneg
  load i8, ptr %p.idx.nneg
  ret void
}
