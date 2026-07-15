; RUN: opt %loadNPMPolly '-passes=polly-custom<import-jscop>' -polly-import-jscop-postfix=transformed -polly-print-import-jscop -disable-output < %s | FileCheck %s

define void @change-array-dims(ptr noalias nonnull %A, ptr noalias nonnull %B) {
entry:
  br label %outer.preheader

outer.preheader:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %outer.preheader], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 2
  br i1 %j.cmp, label %inner.preheader, label %outer.exit


    inner.preheader:
      br label %inner.for

    inner.for:
      %i = phi i32 [0, %inner.preheader], [%i.inc, %inner.inc]
      br label %body



        body:
          %mul = mul nsw i32 %j, 4
          %add = add nsw i32 %mul, %i
          %A_idx = getelementptr inbounds double, ptr %A, i32 %add
          store double 42.0, ptr %A_idx
          br label %inner.inc



    inner.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, 4
      br i1 %i.cmp, label %inner.for, label %inner.exit

    inner.exit:
      br label %outer.inc



outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}


; CHECK:      Stmt_body
; CHECK:          MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       { Stmt_body[i0, i1] -> MemRef_A[4i0 + i1] };
; CHECK-NEXT:       new: { Stmt_body[i0, i1] -> MemRef_A[i0, i1] };
