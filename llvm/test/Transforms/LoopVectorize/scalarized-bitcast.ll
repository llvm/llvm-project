; REQUIRES: asserts
; RUN: opt -opaque-pointers=0 -passes=loop-vectorize -force-vector-width=2 -debug-only=loop-vectorize -S -o - < %s 2>&1 | FileCheck %s

%struct.foo = type { i32, i64 }

; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %0 = bitcast i64* %b to i32*

; The bitcast below will be scalarized due to the predication in the loop. Bitcasts
; between pointer types should be treated as free, despite the scalarization.
define void @foo(%struct.foo* noalias nocapture %in, i32* noalias nocapture readnone %out, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %if.end
  %i.012 = phi i64 [ %inc, %if.end ], [ 0, %entry ]
  %b = getelementptr inbounds %struct.foo, %struct.foo* %in, i64 %i.012, i32 1
  %0 = bitcast i64* %b to i32*
  %a = getelementptr inbounds %struct.foo, %struct.foo* %in, i64 %i.012, i32 0
  %1 = load i32, i32* %a, align 8
  %tobool.not = icmp eq i32 %1, 0
  br i1 %tobool.not, label %if.end, label %land.lhs.true

land.lhs.true:                                    ; preds = %for.body
  %2 = load i32, i32* %0, align 4
  %cmp2 = icmp sgt i32 %2, 0
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %sub = add nsw i32 %2, -1
  store i32 %sub, i32* %0, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %for.body
  %inc = add nuw nsw i64 %i.012, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %if.end
  ret void
}
