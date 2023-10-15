; RUN: opt %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "p:64:64:64"

; CHECK-LABEL: Function: pr69096
; FIXME: This should be MayAlias. %p == %scevgep.i when %a == -1.
; CHECK: NoAlias:     i8* %p, i16* %scevgep.i

define i32 @pr69096(i16 %a, ptr %p) {
entry:
  %0 = load i8, ptr %p, align 2
  %dec.i = add i8 %0, -1
  %cmp636.i = icmp eq i16 %a, -1
  br i1 %cmp636.i, label %for.cond2.for.inc29_crit_edge.i, label %n.exit

for.cond2.for.inc29_crit_edge.i:
  %conv3.i = zext i16 %a to i64
  %sub.i.i = shl i64 %conv3.i, 56
  %sub21.i = shl nuw nsw i64 %conv3.i, 2
  %1 = getelementptr i8, ptr %p, i64 %sub21.i
  %2 = getelementptr i8, ptr %1, i64 -262140
  %3 = getelementptr i8, ptr %2, i64 %sub.i.i
  %scevgep.i = getelementptr i8, ptr %3, i64 72057594037927936
  store i16 1285, ptr %scevgep.i, align 2
  br label %n.exit

n.exit:
  %4 = load i8, ptr %p, align 2
  %conv = sext i8 %4 to i32
  ret i32 %conv
}
