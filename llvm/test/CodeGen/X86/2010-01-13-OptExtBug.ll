; RUN: llc < %s -mtriple=i386-pc-linux-gnu
; PR6027

%class.OlsonTimeZone = type { i16, ptr, ptr, i16 }

define void @XX(ptr %this) align 2 {
entry:
  %call = tail call ptr @_Z15uprv_malloc_4_2v()
  %tmp = getelementptr inbounds %class.OlsonTimeZone, ptr %this, i32 0, i32 3
  %tmp2 = load i16, ptr %tmp
  %tmp626 = load i16, ptr %this
  %cmp27 = icmp slt i16 %tmp2, %tmp626
  br i1 %cmp27, label %bb.nph, label %for.end

for.cond:
  %tmp6 = load i16, ptr %this
  %cmp = icmp slt i16 %inc, %tmp6
  %indvar.next = add i32 %indvar, 1
  br i1 %cmp, label %for.body, label %for.end

bb.nph:
  %tmp10 = getelementptr inbounds %class.OlsonTimeZone, ptr %this, i32 0, i32 2
  %tmp17 = getelementptr inbounds %class.OlsonTimeZone, ptr %this, i32 0, i32 1
  %tmp29 = sext i16 %tmp2 to i32
  %tmp31 = add i16 %tmp2, 1
  %tmp32 = zext i16 %tmp31 to i32
  br label %for.body

for.body:
  %indvar = phi i32 [ 0, %bb.nph ], [ %indvar.next, %for.cond ]
  %tmp30 = add i32 %indvar, %tmp29
  %tmp33 = add i32 %indvar, %tmp32
  %inc = trunc i32 %tmp33 to i16
  %tmp11 = load ptr, ptr %tmp10
  %arrayidx = getelementptr i8, ptr %tmp11, i32 %tmp30
  %tmp12 = load i8, ptr %arrayidx
  br label %for.cond

for.end:
  ret void
}

declare ptr @_Z15uprv_malloc_4_2v()
