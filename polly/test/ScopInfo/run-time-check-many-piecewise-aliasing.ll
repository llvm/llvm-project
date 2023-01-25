; RUN: opt %loadPolly -polly-print-detect -disable-output < %s \
; RUN: | FileCheck %s -check-prefix=DETECT
; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; DETECT: Valid Region for Scop: for => return
;
; Check that this SCoP is allowed, even though the number of disjunct memory accesses of A
; is 11, greater than RunTimeChecksMaxAccessDisjuncts.
;
; CHECK: Function: func
; CHECK-NEXT: Region: %for---%return

define void @func(i64 %n, ptr nonnull %A, ptr nonnull %B, i64 %d) {
entry:
	  br label %for
for:
  %j = phi i64 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i64 %j, %n
  br i1 %j.cmp, label %body, label %exit

body:
      %add.i.i = add nsw i64 1, %j
      %sub.i.i = sub nsw i64 %add.i.i, 1
      %cmp.i.i.i = icmp sgt i64 %sub.i.i, 0
      %cond.i.i.i = select i1 %cmp.i.i.i, i64 %sub.i.i, i64 0
      %mul.i.i = mul nsw i64 %cond.i.i.i, 7
      %sub1.i.i = sub nsw i64 1, %j
      %add2.i.i = add nsw i64 %sub1.i.i, 1
      %cmp.i8.i.i = icmp sgt i64 %add2.i.i, 0
      %cond.i11.i.i = select i1 %cmp.i8.i.i, i64 %add2.i.i, i64 0
      %mul4.i.i = mul nsw i64 %cond.i11.i.i, 7
      %add5.i.i = add nsw i64 %mul.i.i, %mul4.i.i
      %add.i113.i = add nsw i64 1, %j
      %sub.i114.i = sub nsw i64 %add.i113.i, 3
      %cmp.i.i115.i = icmp sgt i64 %sub.i114.i, 0
      %cond.i.i118.i = select i1 %cmp.i.i115.i, i64 %sub.i114.i, i64 0
      %mul.i119.i = mul nsw i64 %cond.i.i118.i, 9
      %sub1.i120.i = sub nsw i64 1, %j
      %add2.i121.i = add nsw i64 %sub1.i120.i, 3
      %cmp.i8.i122.i = icmp sgt i64 %add2.i121.i, 0
      %cond.i11.i126.i = select i1 %cmp.i8.i122.i, i64 %add2.i121.i, i64 0
      %mul4.i127.i = mul nsw i64 %cond.i11.i126.i, 9
      %add5.i128.i = add nsw i64 %mul.i119.i, %mul4.i127.i
      %add.i = add nsw i64 %add5.i.i, %add5.i128.i
      %add.i89.i = add nsw i64 1, %j
      %sub.i90.i = sub nsw i64 %add.i89.i, 4
      %cmp.i.i91.i = icmp sgt i64 %sub.i90.i, 0
      %cond.i.i94.i = select i1 %cmp.i.i91.i, i64 %sub.i90.i, i64 0
      %mul.i95.i = mul nsw i64 %cond.i.i94.i, 11
      %sub1.i96.i = sub nsw i64 1, %j
      %add2.i97.i = add nsw i64 %sub1.i96.i, 4
      %cmp.i8.i98.i = icmp sgt i64 %add2.i97.i, 0
      %cond.i11.i102.i = select i1 %cmp.i8.i98.i, i64 %add2.i97.i, i64 0
      %mul4.i103.i = mul nsw i64 %cond.i11.i102.i, 11
      %add5.i104.i = add nsw i64 %mul.i95.i, %mul4.i103.i
      %add3.i = add nsw i64 %add.i, %add5.i104.i
      %add.i65.i = add nsw i64 1, %j
      %sub.i66.i = sub nsw i64 %add.i65.i, 6
      %cmp.i.i67.i = icmp sgt i64 %sub.i66.i, 0
      %cond.i.i70.i = select i1 %cmp.i.i67.i, i64 %sub.i66.i, i64 0
      %mul.i71.i = mul nsw i64 %cond.i.i70.i, 13
      %sub1.i72.i = sub nsw i64 1, %j
      %add2.i73.i = add nsw i64 %sub1.i72.i, 6
      %cmp.i8.i74.i = icmp sgt i64 %add2.i73.i, 0
      %cond.i11.i78.i = select i1 %cmp.i8.i74.i, i64 %add2.i73.i, i64 0
      %mul4.i79.i = mul nsw i64 %cond.i11.i78.i, 13
      %add5.i80.i = add nsw i64 %mul.i71.i, %mul4.i79.i
      %add5.i = add nsw i64 %add3.i, %add5.i80.i
      %add.i41.i = add nsw i64 1, %j
      %sub.i42.i = sub nsw i64 %add.i41.i, 8
      %cmp.i.i43.i = icmp sgt i64 %sub.i42.i, 0
      %cond.i.i46.i = select i1 %cmp.i.i43.i, i64 %sub.i42.i, i64 0
      %mul.i47.i = mul nsw i64 %cond.i.i46.i, 17
      %sub1.i48.i = sub nsw i64 1, %j
      %add2.i49.i = add nsw i64 %sub1.i48.i, 8
      %cmp.i8.i50.i = icmp sgt i64 %add2.i49.i, 0
      %cond.i11.i54.i = select i1 %cmp.i8.i50.i, i64 %add2.i49.i, i64 0
      %mul4.i55.i = mul nsw i64 %cond.i11.i54.i, 17
      %add5.i56.i = add nsw i64 %mul.i47.i, %mul4.i55.i
      %add7.i = add nsw i64 %add5.i, %add5.i56.i
      %add.i17.i = add nsw i64 1, %j
      %sub.i18.i = sub nsw i64 %add.i17.i, 10
      %cmp.i.i19.i = icmp sgt i64 %sub.i18.i, 0
      %cond.i.i22.i = select i1 %cmp.i.i19.i, i64 %sub.i18.i, i64 0
      %mul.i23.i = mul nsw i64 %cond.i.i22.i, 19
      %sub1.i24.i = sub nsw i64 1, %j
      %add2.i25.i = add nsw i64 %sub1.i24.i, 10
      %cmp.i8.i26.i = icmp sgt i64 %add2.i25.i, 0
      %cond.i11.i30.i = select i1 %cmp.i8.i26.i, i64 %add2.i25.i, i64 0
      %mul4.i31.i = mul nsw i64 %cond.i11.i30.i, 19
      %add5.i32.i = add nsw i64 %mul.i23.i, %mul4.i31.i
      %idxprom = add nsw i64 %add7.i, %add5.i32.i

      %A_idx = getelementptr inbounds double, ptr %A, i64 %idxprom
      %val = load double, ptr %A_idx
      %B_idx = getelementptr inbounds double, ptr %B, i64 %j
      store double %val, ptr %B_idx
      br label %inc

inc:
	%j.inc = add nuw nsw i64 %j, 1
	br label %for

exit:
	br label %return
return:
	ret void
}

