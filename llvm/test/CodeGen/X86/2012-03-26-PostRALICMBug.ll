; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -stats 2>&1 | \
; RUN:   not grep "Number of machine instructions hoisted out of loops post regalloc"

; rdar://11095580

%struct.ref_s = type { %union.color_sample, i16, i16 }
%union.color_sample = type { i64 }

@table = external global [3891 x i64]

declare i32 @foo()

define i32 @zarray(%struct.ref_s* nocapture %op) nounwind ssp {
entry:
  %call = tail call i32 @foo()
  %tmp = ashr i32 %call, 31
  %0 = and i32 %tmp, 1396
  %index9 = add i32 %0, 2397
  indirectbr i8* undef, [label %return, label %if.end]

if.end:                                           ; preds = %entry
  %size5 = getelementptr inbounds %struct.ref_s* %op, i64 0, i32 2
  %tmp6 = load i16* %size5, align 2
  %tobool1 = icmp eq i16 %tmp6, 0
  %1 = select i1 %tobool1, i32 1396, i32 -1910
  %index10 = add i32 %index9, %1
  indirectbr i8* undef, [label %return, label %while.body.lr.ph]

while.body.lr.ph:                                 ; preds = %if.end
  %refs = bitcast %struct.ref_s* %op to %struct.ref_s**
  %tmp9 = load %struct.ref_s** %refs, align 8
  %tmp4 = zext i16 %tmp6 to i64
  %index13 = add i32 %index10, 1658
  %2 = sext i32 %index13 to i64
  %3 = getelementptr [3891 x i64]* @table, i64 0, i64 %2
  %blockaddress14 = load i64* %3, align 8
  %4 = inttoptr i64 %blockaddress14 to i8*
  indirectbr i8* %4, [label %while.body]

while.body:                                       ; preds = %while.body, %while.body.lr.ph
  %index7 = phi i32 [ %index15, %while.body ], [ %index13, %while.body.lr.ph ]
  %indvar = phi i64 [ %indvar.next, %while.body ], [ 0, %while.body.lr.ph ]
  %type_attrs = getelementptr %struct.ref_s* %tmp9, i64 %indvar, i32 1
  store i16 32, i16* %type_attrs, align 2
  %indvar.next = add i64 %indvar, 1
  %exitcond5 = icmp eq i64 %indvar.next, %tmp4
  %tmp7 = select i1 %exitcond5, i32 1648, i32 0
  %index15 = add i32 %index7, %tmp7
  %tmp8 = select i1 %exitcond5, i64 13, i64 0
  %5 = sext i32 %index15 to i64
  %6 = getelementptr [3891 x i64]* @table, i64 0, i64 %5
  %blockaddress16 = load i64* %6, align 8
  %7 = inttoptr i64 %blockaddress16 to i8*
  indirectbr i8* %7, [label %return, label %while.body]

return:                                           ; preds = %while.body, %if.end, %entry
  %retval.0 = phi i32 [ %call, %entry ], [ 0, %if.end ], [ 0, %while.body ]
  ret i32 %retval.0
}
