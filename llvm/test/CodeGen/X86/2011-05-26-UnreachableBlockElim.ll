; RUN: llc < %s -verify-coalescing
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.0"

%struct.attrib = type { i32, i32 }
%struct.dfa = type { [80 x i8], i32, ptr, i32, i32, ptr, i32, i32 }
%struct.state = type { i32, [4 x i32] }

@aux_temp = external global %struct.dfa, align 8

declare i64 @llvm.objectsize.i64.p0(ptr, i1) nounwind readnone

declare void @__memset_chk() nounwind

define void @dfa_add_string() nounwind uwtable ssp {
entry:
  br label %if.end.i

if.end.i:                                         ; preds = %entry
  %idxprom.i = add i64 0, 1
  br i1 undef, label %land.end.thread.i, label %land.end.i

land.end.thread.i:                                ; preds = %if.end.i
  %0 = call i64 @llvm.objectsize.i64.p0(ptr undef, i1 false) nounwind
  %cmp1710.i = icmp eq i64 %0, -1
  br i1 %cmp1710.i, label %cond.false156.i, label %cond.true138.i

land.end.i:                                       ; preds = %if.end.i
  %1 = call i64 @llvm.objectsize.i64.p0(ptr undef, i1 false) nounwind
  %cmp17.i = icmp eq i64 %1, -1
  br i1 %cmp17.i, label %cond.false156.i, label %cond.true138.i

cond.true138.i:                                   ; preds = %for.end.i, %land.end.thread.i
  call void @__memset_chk() nounwind
  br label %cond.end166.i

cond.false156.i:                                  ; preds = %for.end.i, %land.end.thread.i
  %idxprom1114.i = phi i64 [ undef, %land.end.thread.i ], [ %idxprom.i, %land.end.i ]
  call void @__memset_chk() nounwind
  br label %cond.end166.i

cond.end166.i:                                    ; preds = %cond.false156.i, %cond.true138.i
  %idxprom1113.i = phi i64 [ %idxprom1114.i, %cond.false156.i ], [ undef, %cond.true138.i ]
  %tmp235.i = load ptr, ptr getelementptr inbounds (%struct.dfa, ptr @aux_temp, i64 0, i32 2), align 8
  %att.i = getelementptr inbounds %struct.state, ptr %tmp235.i, i64 %idxprom1113.i, i32 0
  store i32 0, ptr %att.i, align 4
  ret void
}
