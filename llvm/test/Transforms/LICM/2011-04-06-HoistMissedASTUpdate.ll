; RUN: opt < %s -passes=licm -S | FileCheck %s
; PR9630

@g_39 = external global i16, align 2

declare ptr @func_84(ptr nocapture) nounwind readonly

declare ptr @func_108(ptr nocapture) nounwind readonly

define void @func() nounwind {
entry:
  br label %for.body4.lr.ph

for.body4.lr.ph:
  br label %for.body4

; CHECK: for.body4:
; CHECK: load volatile i16, ptr @g_39

for.body4:
  %l_612.11 = phi ptr [ undef, %for.body4.lr.ph ], [ %call19, %for.body4 ]
  %tmp7 = load volatile i16, ptr @g_39, align 2
  %call = call ptr @func_108(ptr undef)
  %call19 = call ptr @func_84(ptr %call)
  br i1 false, label %for.body4, label %for.cond.loopexit

for.cond.loopexit:
  br i1 false, label %for.body4.lr.ph, label %for.end26

for.end26:
  ret void
}
