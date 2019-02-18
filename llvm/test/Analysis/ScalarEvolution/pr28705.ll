; PR28705
; RUN: opt < %s -indvars -S | FileCheck %s

; Check IndVarSimplify doesn't replace external use of the induction var
; "%inc.i.i" with "%.sroa.speculated + 1" because it is not profitable.
;
; CHECK-LABEL: @foo(
; CHECK: %[[EXIT:.+]] = phi i32 [ %inc.i.i, %for.body650 ]
; CHECK: %DB.sroa.9.0.lcssa = phi i32 [ 1, %entry ], [ %[[EXIT]], %loopexit ]
;
define void @foo(i32 %sub.ptr.div.i, i8* %ref.i1174) local_unnamed_addr {
entry:
  %cmp.i1137 = icmp ugt i32 %sub.ptr.div.i, 3
  %.sroa.speculated = select i1 %cmp.i1137, i32 3, i32 %sub.ptr.div.i
  %cmp6483126 = icmp eq i32 %.sroa.speculated, 0
  br i1 %cmp6483126, label %XZ.exit, label %for.body650.lr.ph

for.body650.lr.ph:
  br label %for.body650

loopexit:
  %inc.i.i.lcssa = phi i32 [ %inc.i.i, %for.body650 ]
  br label %XZ.exit

XZ.exit:
  %DB.sroa.9.0.lcssa = phi i32 [ 1, %entry ], [ %inc.i.i.lcssa, %loopexit ]
  br label %end

for.body650:
  %iv = phi i32 [ 0, %for.body650.lr.ph ], [ %inc655, %for.body650 ]
  %iv2 = phi i32 [ 1, %for.body650.lr.ph ], [ %inc.i.i, %for.body650 ]
  %arrayidx.i.i1105 = getelementptr inbounds i8, i8* %ref.i1174, i32 %iv2
  store i8 7, i8* %arrayidx.i.i1105, align 1
  %inc.i.i = add i32 %iv2, 1
  %inc655 = add i32 %iv, 1
  %cmp648 = icmp eq i32 %inc655, %.sroa.speculated
  br i1 %cmp648, label %loopexit, label %for.body650

end:
  ret void
}
