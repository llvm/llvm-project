; RUN: opt %loadPolly -pass-remarks-analysis="polly-scops" -polly-print-scops \
; RUN:     -polly-invariant-load-hoisting=true \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Low complexity assumption: {  : false }
;
; The IR is a modified version of the following C:
;
;    void f(int *A) {
;    Begin:
;      if (A[0] == 1 | A[1] == 1 | A[2] == 1 | A[3] == 1 | A[4] == 1 | A[5] == 1 |
;          A[6] == 1 | A[7] == 1 | A[8] == 1 | A[9] == 1 | A[10] == 1 | A[11] == 1 |
;          A[12] == 1 | A[13] == 1 | A[14] == 1 | A[15] == 1 | A[16] == 1 |
;          A[17] == 1 | A[18] == 1 | A[19] == 1 | A[20] == 1 | A[21] == 1 |
;          A[22] == 1 | A[23]) {
;        A[-1]++;
;      } else {
;        A[-1]--;
;      }
;    End:
;      return;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
entry:
  br label %Begin

Begin:                                            ; preds = %entry
  %tmp = load i32, ptr %A, align 4
  %cmp = icmp eq i32 %tmp, 1
  %arrayidx1 = getelementptr inbounds i32, ptr %A, i64 1
  %tmp1 = load i32, ptr %arrayidx1, align 4
  %cmp2 = icmp eq i32 %tmp1, 1
  %or = or i1 %cmp, %cmp2
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 2
  %tmp2 = load i32, ptr %arrayidx4, align 4
  %cmp5 = icmp eq i32 %tmp2, 1
  %or7 = or i1 %or, %cmp5
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i64 3
  %tmp3 = load i32, ptr %arrayidx8, align 4
  %cmp9 = icmp eq i32 %tmp3, 1
  %or11 = or i1 %or7, %cmp9
  %arrayidx12 = getelementptr inbounds i32, ptr %A, i64 4
  %tmp4 = load i32, ptr %arrayidx12, align 4
  %cmp13 = icmp eq i32 %tmp4, 1
  %or15 = or i1 %or11, %cmp13
  %arrayidx16 = getelementptr inbounds i32, ptr %A, i64 5
  %tmp5 = load i32, ptr %arrayidx16, align 4
  %cmp17 = icmp eq i32 %tmp5, 1
  %or19 = or i1 %or15, %cmp17
  %arrayidx20 = getelementptr inbounds i32, ptr %A, i64 6
  %tmp6 = load i32, ptr %arrayidx20, align 4
  %cmp21 = icmp eq i32 %tmp6, 1
  %or23 = or i1 %or19, %cmp21
  %arrayidx24 = getelementptr inbounds i32, ptr %A, i64 7
  %tmp7 = load i32, ptr %arrayidx24, align 4
  %cmp25 = icmp eq i32 %tmp7, 1
  %or27 = or i1 %or23, %cmp25
  %arrayidx28 = getelementptr inbounds i32, ptr %A, i64 8
  %tmp8 = load i32, ptr %arrayidx28, align 4
  %cmp29 = icmp eq i32 %tmp8, 1
  %or31 = or i1 %or27, %cmp29
  %arrayidx32 = getelementptr inbounds i32, ptr %A, i64 9
  %tmp9 = load i32, ptr %arrayidx32, align 4
  %cmp33 = icmp eq i32 %tmp9, 1
  %or35 = or i1 %or31, %cmp33
  %arrayidx36 = getelementptr inbounds i32, ptr %A, i64 10
  %tmp10 = load i32, ptr %arrayidx36, align 4
  %cmp37 = icmp eq i32 %tmp10, 1
  %or39 = or i1 %or35, %cmp37
  %arrayidx40 = getelementptr inbounds i32, ptr %A, i64 11
  %tmp11 = load i32, ptr %arrayidx40, align 4
  %cmp41 = icmp eq i32 %tmp11, 1
  %or43 = or i1 %or39, %cmp41
  %arrayidx44 = getelementptr inbounds i32, ptr %A, i64 12
  %tmp12 = load i32, ptr %arrayidx44, align 4
  %cmp45 = icmp eq i32 %tmp12, 1
  %or47 = or i1 %or43, %cmp45
  %arrayidx48 = getelementptr inbounds i32, ptr %A, i64 13
  %tmp13 = load i32, ptr %arrayidx48, align 4
  %cmp49 = icmp eq i32 %tmp13, 1
  %or51 = or i1 %or47, %cmp49
  %arrayidx52 = getelementptr inbounds i32, ptr %A, i64 14
  %tmp14 = load i32, ptr %arrayidx52, align 4
  %cmp53 = icmp eq i32 %tmp14, 1
  %or55 = or i1 %or51, %cmp53
  %arrayidx56 = getelementptr inbounds i32, ptr %A, i64 15
  %tmp15 = load i32, ptr %arrayidx56, align 4
  %cmp57 = icmp eq i32 %tmp15, 1
  %or59 = or i1 %or55, %cmp57
  %arrayidx60 = getelementptr inbounds i32, ptr %A, i64 16
  %tmp16 = load i32, ptr %arrayidx60, align 4
  %cmp61 = icmp eq i32 %tmp16, 1
  %or63 = or i1 %or59, %cmp61
  %arrayidx64 = getelementptr inbounds i32, ptr %A, i64 17
  %tmp17 = load i32, ptr %arrayidx64, align 4
  %cmp65 = icmp eq i32 %tmp17, 1
  %or67 = or i1 %or63, %cmp65
  %arrayidx68 = getelementptr inbounds i32, ptr %A, i64 18
  %tmp18 = load i32, ptr %arrayidx68, align 4
  %cmp69 = icmp eq i32 %tmp18, 1
  %or71 = or i1 %or67, %cmp69
  %arrayidx72 = getelementptr inbounds i32, ptr %A, i64 19
  %tmp19 = load i32, ptr %arrayidx72, align 4
  %cmp73 = icmp eq i32 %tmp19, 1
  %or75 = or i1 %or71, %cmp73
  %arrayidx76 = getelementptr inbounds i32, ptr %A, i64 20
  %tmp20 = load i32, ptr %arrayidx76, align 4
  %cmp77 = icmp eq i32 %tmp20, 1
  %or79 = or i1 %or75, %cmp77
  %arrayidx80 = getelementptr inbounds i32, ptr %A, i64 21
  %tmp21 = load i32, ptr %arrayidx80, align 4
  %cmp81 = icmp eq i32 %tmp21, 1
  %or83 = or i1 %or79, %cmp81
  %arrayidx84 = getelementptr inbounds i32, ptr %A, i64 22
  %tmp22 = load i32, ptr %arrayidx84, align 4
  %cmp85 = icmp eq i32 %tmp22, 1
  %or87 = or i1 %or83, %cmp85
  %arrayidx88 = getelementptr inbounds i32, ptr %A, i64 23
  %tmp23 = load i32, ptr %arrayidx88, align 4
  %cmp88 = icmp eq i32 %tmp23, 1
  %or89 = or i1 %or87, %cmp88
  br i1 %or89, label %if.else, label %if.then

if.then:                                          ; preds = %Begin
  %arrayidx90 = getelementptr inbounds i32, ptr %A, i64 -1
  %tmp24 = load i32, ptr %arrayidx90, align 4
  %inc = add nsw i32 %tmp24, 1
  store i32 %inc, ptr %arrayidx90, align 4
  br label %if.end

if.else:                                          ; preds = %Begin
  %arrayidx91 = getelementptr inbounds i32, ptr %A, i64 -1
  %tmp25 = load i32, ptr %arrayidx91, align 4
  %dec = add nsw i32 %tmp25, -1
  store i32 %dec, ptr %arrayidx91, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %End

End:                                              ; preds = %if.end
  ret void
}
