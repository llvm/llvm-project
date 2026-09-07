; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-width=2 -debug-only=loop-vectorize --disable-output -S -o - < %s 2>&1 | FileCheck %s

%struct.foo = type { i32, i64 }

; CHECK: Cost of 0 for VF 2: WIDEN-GEP ir<%b> = getelementptr inbounds ir<%in>, ir<%i.012>, ir<1>

; The bitcast below is a no-op between identical pointer types and is folded
; away, so it costs nothing even though the loop is predicated.
define void @foo(ptr noalias nocapture %in, ptr noalias nocapture readnone %out, i64 %n) {
entry:
  br label %for.body

for.body:
  %i.012 = phi i64 [ %inc, %if.end ], [ 0, %entry ]
  %b = getelementptr inbounds %struct.foo, ptr %in, i64 %i.012, i32 1
  %0 = bitcast ptr %b to ptr
  %a = getelementptr inbounds %struct.foo, ptr %in, i64 %i.012, i32 0
  %1 = load i32, ptr %a, align 8
  %tobool.not = icmp eq i32 %1, 0
  br i1 %tobool.not, label %if.end, label %land.lhs.true

land.lhs.true:
  %2 = load i32, ptr %0, align 4
  %cmp2 = icmp sgt i32 %2, 0
  br i1 %cmp2, label %if.then, label %if.end

if.then:
  %sub = add nsw i32 %2, -1
  store i32 %sub, ptr %0, align 4
  br label %if.end

if.end:
  %inc = add nuw nsw i64 %i.012, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret void
}
