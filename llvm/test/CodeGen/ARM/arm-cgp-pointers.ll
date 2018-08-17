; RUN: llc -mtriple=thumbv8 -arm-disable-cgp=false %s -o - | FileCheck %s
; RUN: llc -mtriple=armv8 -arm-disable-cgp=false %s -o - | FileCheck %s

; CHECK-LABEL: phi_pointers
; CHECK-NOT: uxt
define void @phi_pointers(i16* %a, i16* %b, i8 zeroext %M, i8 zeroext %N) {
entry:
  %add = add nuw i8 %M, 1
  %and = and i8 %add, 1
  %cmp = icmp ugt i8 %add, %N
  %base = select i1 %cmp, i16* %a, i16* %b
  %other = select i1 %cmp, i16* %b, i16* %b
  br label %loop

loop:
  %ptr = phi i16* [ %base, %entry ], [ %gep, %loop ]
  %idx = phi i8 [ %and, %entry ], [ %inc, %loop ]
  %load = load i16, i16* %ptr, align 2
  %inc = add nuw nsw i8 %idx, 1
  %gep = getelementptr inbounds i16, i16* %ptr, i8 %inc
  %cond = icmp eq i16* %gep, %other
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: phi_pointers_null
; CHECK-NOT: uxt
define void @phi_pointers_null(i16* %a, i16* %b, i8 zeroext %M, i8 zeroext %N) {
entry:
  %add = add nuw i8 %M, 1
  %and = and i8 %add, 1
  %cmp = icmp ugt i8 %add, %N
  %base = select i1 %cmp, i16* %a, i16* %b
  %other = select i1 %cmp, i16* %b, i16* %b
  %cmp.1 = icmp eq i16* %base, %other
  br i1 %cmp.1, label %fail, label %loop

fail:
  br label %loop

loop:
  %ptr = phi i16* [ %base, %entry ], [ null, %fail ], [ %gep, %if.then ]
  %idx = phi i8 [ %and, %entry ], [ 0, %fail ], [ %inc, %if.then ]
  %undef = icmp eq i16* %ptr, undef
  br i1 %undef, label %exit, label %if.then

if.then:
  %load = load i16, i16* %ptr, align 2
  %inc = add nuw nsw i8 %idx, 1
  %gep = getelementptr inbounds i16, i16* %ptr, i8 %inc
  %cond = icmp eq i16* %gep, %other
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

declare i8 @do_something_with_ptr(i8, i16*)

; CHECK-LABEL: call_pointer
; CHECK-NOT: uxt
define i8 @call_pointer(i8 zeroext %x, i8 zeroext %y, i16* %a, i16* %b) {
  %or = or i8 %x, %y
  %shr = lshr i8 %or, 1
  %add = add nuw i8 %shr, 2
  %cmp = icmp ne i8 %add, 0
  %ptr = select i1 %cmp, i16* %a, i16* %b
  %call = tail call zeroext i8 @do_something_with_ptr(i8 %shr, i16* %ptr)
  ret i8 %call
}

; CHECK-LABEL: pointer_to_pointer
; CHECK-NOT: uxt
define i16 @pointer_to_pointer(i16** %arg, i16 zeroext %limit) {
entry:
  %addr = load i16*, i16** %arg
  %val = load i16, i16* %addr
  %add = add nuw i16 %val, 7
  %cmp = icmp ult i16 %add, 256
  %res = select i1 %cmp, i16 128, i16 255
  ret i16 %res
}

; CHECK-LABEL: gep_2d_array
; CHECK-NOT: uxt
define i8 @gep_2d_array(i8** %a, i8 zeroext %arg) {
entry:
  %arrayidx.us = getelementptr inbounds i8*, i8** %a, i32 0
  %0 = load i8*, i8** %arrayidx.us, align 4
  %1 = load i8, i8* %0, align 1
  %sub = sub nuw i8 %1, 1
  %cmp = icmp ult i8 %sub, %arg
  %res = select i1 %cmp, i8 27, i8 54
  ret i8 %res
}

; CHECK-LABEL: gep_2d_array_loop
; CHECK-NOT: uxt
define void @gep_2d_array_loop(i16** nocapture readonly %a, i16** nocapture readonly %b, i32 %N) {
entry:
  %cmp30 = icmp eq i32 %N, 0
  br i1 %cmp30, label %for.cond.cleanup, label %for.cond1.preheader.us

for.cond1.preheader.us:
  %y.031.us = phi i32 [ %inc13.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %entry ]
  br label %for.body4.us

for.body4.us:
  %x.029.us = phi i32 [ 0, %for.cond1.preheader.us ], [ %inc.us, %for.body4.us ]
  %arrayidx.us = getelementptr inbounds i16*, i16** %a, i32 %x.029.us
  %0 = load i16*, i16** %arrayidx.us, align 4
  %arrayidx5.us = getelementptr inbounds i16, i16* %0, i32 %y.031.us
  %1 = load i16, i16* %arrayidx5.us, align 2
  %dec.us = add nuw i16 %1, -1
  %cmp6.us = icmp ult i16 %dec.us, 16383
  %shl.us = shl nuw i16 %dec.us, 2
  %spec.select.us = select i1 %cmp6.us, i16 %shl.us, i16 %dec.us
  %arrayidx10.us = getelementptr inbounds i16*, i16** %b, i32 %x.029.us
  %2 = load i16*, i16** %arrayidx10.us, align 4
  %arrayidx11.us = getelementptr inbounds i16, i16* %2, i32 %y.031.us
  store i16 %spec.select.us, i16* %arrayidx11.us, align 2
  %inc.us = add nuw i32 %x.029.us, 1
  %exitcond = icmp eq i32 %inc.us, %N
  br i1 %exitcond, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us

for.cond1.for.cond.cleanup3_crit_edge.us:
  %inc13.us = add nuw i32 %y.031.us, 1
  %exitcond32 = icmp eq i32 %inc13.us, %N
  br i1 %exitcond32, label %for.cond.cleanup, label %for.cond1.preheader.us

for.cond.cleanup:
  ret void
}
