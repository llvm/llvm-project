; RUN: opt -S %s -passes=deadargelim -o - | FileCheck %s
; In that test @internal_fct is used by an instruction
; we don't know how to rewrite (the comparison that produces
; %cmp1).
; Because of that use, we used to bail out on removing the
; unused arguments for this function.
; Yet, we should still be able to rewrite the direct calls that are
; statically known, by replacing the related arguments with poison.
; This is what we check on the call that produces %res2.

define i32 @call_indirect(ptr readnone %fct_ptr, i32 %arg1, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: @call_indirect(
; CHECK-NEXT:    [[CMP0:%.*]] = icmp eq ptr [[FCT_PTR:%.*]], @external_fct
; CHECK-NEXT:    br i1 [[CMP0]], label [[CALL_EXT:%.*]], label [[CHK2:%.*]]
; CHECK:       call_ext:
; CHECK-NEXT:    [[RES1:%.*]] = tail call i32 @external_fct(i32 poison, i32 [[ARG2:%.*]], i32 poison)
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       chk2:
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq ptr [[FCT_PTR]], @internal_fct
; CHECK-NEXT:    br i1 [[CMP1]], label [[CALL_INT:%.*]], label [[CALL_OTHER:%.*]]
; CHECK:       call_int:
; CHECK-NEXT:    [[RES2:%.*]] = tail call i32 @internal_fct(i32 poison, i32 [[ARG2]], i32 poison)
; CHECK-NEXT:    br label [[END]]
; CHECK:       call_other:
; CHECK-NEXT:    [[RES3:%.*]] = tail call i32 @other_fct.argelim(i32 [[ARG2]])
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[FINAL_RES:%.*]] = phi i32 [ [[RES1]], [[CALL_EXT]] ], [ [[RES2]], [[CALL_INT]] ], [ [[RES3]], [[CALL_OTHER]] ]
; CHECK-NEXT:    ret i32 [[FINAL_RES]]
;
  %cmp0 = icmp eq ptr %fct_ptr, @external_fct
  br i1 %cmp0, label %call_ext, label %chk2

call_ext:
  %res1 = tail call i32 @external_fct(i32 %arg1, i32 %arg2, i32 %arg3)
  br label %end

chk2:
  %cmp1 = icmp eq ptr %fct_ptr, @internal_fct
  br i1 %cmp1, label %call_int, label %call_other

call_int:
  %res2 = tail call i32 @internal_fct(i32 %arg1, i32 %arg2, i32 %arg3)
  br label %end

call_other:
  %res3 = tail call i32 @other_fct(i32 %arg1, i32 %arg2, i32 %arg3)
  br label %end

end:
  %final_res = phi i32 [%res1, %call_ext], [%res2, %call_int], [%res3, %call_other]
  ret i32 %final_res
}


define i32 @external_fct(i32 %unused_arg1, i32 %arg2, i32 %unused_arg3) {
  ret i32 %arg2
}

define internal i32 @internal_fct(i32 %unused_arg1, i32 %arg2, i32 %unused_arg3) {
  ret i32 %arg2
}

define internal i32 @other_fct(i32 %unused_arg1, i32 %arg2, i32 %unused_arg3) {
  ret i32 %arg2
}

