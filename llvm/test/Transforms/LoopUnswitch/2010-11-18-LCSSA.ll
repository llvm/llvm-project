; RUN: opt < %s -loop-unswitch
; RUN: opt < %s -loop-unswitch -enable-mssa-loop-dependency=true -verify-memoryssa
; PR8622
@g_38 = external global i32, align 4

define void @func_67(i32 %p_68.coerce) nounwind {
entry:
  br i1 true, label %for.end12, label %bb.nph

bb.nph:                                           ; preds = %entry
  %g_38.promoted = load i32, i32* @g_38
  br label %for.body

for.body:                                         ; preds = %for.cond, %bb.nph
  %tobool.i = icmp eq i32 %p_68.coerce, 1
  %xor4.i = xor i32 %p_68.coerce, 1
  %call1 = select i1 %tobool.i, i32 0, i32 %xor4.i
  br label %for.cond

for.cond:                                         ; preds = %for.body
  br i1 true, label %for.cond.for.end12_crit_edge, label %for.body

for.cond.for.end12_crit_edge:                     ; preds = %for.cond
  store i32 %call1, i32* @g_38
  br label %for.end12

for.end12:                                        ; preds = %for.cond.for.end12_crit_edge, %entry
  ret void
}
