; RUN: llc -mtriple=s390x-linux-gnu -mcpu=zEC12 < %s  | FileCheck %s
;
; Check that DAGCombiner doesn't cause a crash by eventually producing a
; PCREL_OFFSET node with a freeze operand.

@a = dso_local global [6 x [6 x [3 x i8]]] zeroinitializer, align 2
@b = dso_local local_unnamed_addr global i32 0, align 4

define void @fun(i8 noundef zeroext %g) {
; CHECK-LABEL: fun
entry:
  %agg.tmp.ensured.sroa.0 = alloca i8, align 2
  %conv = zext i8 %g to i64
  %0 = inttoptr i64 %conv to ptr
  %.fr = freeze ptr getelementptr inbounds ([6 x [6 x [3 x i8]]], ptr @a, i64 0, i64 1, i64 2, i64 1)
  %cmp = icmp eq ptr %.fr, %0
  %1 = load i8, ptr getelementptr inbounds ([6 x [6 x [3 x i8]]], ptr @a, i64 0, i64 5, i64 4, i64 2), align 2
  %conv2 = zext i8 %1 to i32
  br i1 %cmp, label %for.cond.us, label %for.cond

for.cond.us:                                      ; preds = %entry, %for.cond.us
  store i32 %conv2, ptr @b, align 4
  ret void

for.cond:                                         ; preds = %entry, %for.cond
  store i32 0, ptr @b, align 4
  ret void
}
