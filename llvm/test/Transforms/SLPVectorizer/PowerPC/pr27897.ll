; RUN: opt -S -mtriple=powerpc64-linux-gnu -mcpu=pwr8 -mattr=+vsx -slp-vectorizer < %s | FileCheck %s

%struct.A = type { i8*, i8* }

define i64 @foo(%struct.A* nocapture readonly %this) {
entry:
  %end.i = getelementptr inbounds %struct.A, %struct.A* %this, i64 0, i32 1
  %0 = bitcast i8** %end.i to i64*
  %1 = load i64, i64* %0, align 8
  %2 = bitcast %struct.A* %this to i64*
  %3 = load i64, i64* %2, align 8
  %sub.ptr.sub.i = sub i64 %1, %3
  %cmp = icmp sgt i64 %sub.ptr.sub.i, 9
  br i1 %cmp, label %return, label %lor.lhs.false

lor.lhs.false:
  %4 = inttoptr i64 %3 to i8*
  %5 = inttoptr i64 %1 to i8*
  %cmp2 = icmp ugt i8* %5, %4
  %. = select i1 %cmp2, i64 2, i64 -1
  ret i64 %.

return:
  ret i64 2
}

; CHECK: load i64
; CHECK-NOT: load <2 x i64>
; CHECK-NOT: extractelement
