; RUN: opt %loadPolly -disable-output -polly-print-scops \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
; Check that propagation of domains from A(X) to A(X+1) will keep the
; domains small and concise.
;
; CHECK:         Assumed Context:
; CHECK-NEXT:    [tmp5, tmp, tmp8, tmp11, tmp14, tmp17, tmp20, tmp23, tmp26] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [tmp5, tmp, tmp8, tmp11, tmp14, tmp17, tmp20, tmp23, tmp26] -> {  : false }
;
; CHECK:         Stmt_FINAL
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [tmp5, tmp, tmp8, tmp11, tmp14, tmp17, tmp20, tmp23, tmp26] -> { Stmt_FINAL[] };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [tmp5, tmp, tmp8, tmp11, tmp14, tmp17, tmp20, tmp23, tmp26] -> { Stmt_FINAL[] -> [16] };
;
;
;    void f(short *restrict In, int *restrict Out) {
;      int InV, V, Idx;
;      Idx = 0;
;      V = 999;
;
;    A0:
;      InV = In[Idx++];
;      if (InV < V + 42) {
;      B0:
;        V = V + 42;
;        Out[V]++;
;      } else {
;      C0:
;        V = InV;
;        Out[V]--;
;      }
;
;    A1:
;      InV = In[Idx++];
;      if (InV < V + 42) {
;      B1:
;        V = V + 42;
;        Out[V]++;
;      } else {
;      C1:
;        V = InV;
;        Out[V]--;
;      }
;      V = 999;
;
;    A2:
;      InV = In[Idx++];
;      if (InV < V + 42) {
;      B2:
;        V = V + 42;
;        Out[V]++;
;      } else {
;      C2:
;        V = InV;
;        Out[V]--;
;      }
;
;    A3:
;      InV = In[Idx++];
;      if (InV < V + 42) {
;      B3:
;        V = V + 42;
;        Out[V]++;
;      } else {
;      C3:
;        V = InV;
;        Out[V]--;
;      }
;      V = 999;
;
;    A4:
;      InV = In[Idx++];
;      if (InV < V + 42) {
;      B4:
;        V = V + 42;
;        Out[V]++;
;      } else {
;      C4:
;        V = InV;
;        Out[V]--;
;      }
;
;    A5:
;      InV = In[Idx++];
;      if (InV < V + 42) {
;      B5:
;        V = V + 42;
;        Out[V]++;
;      } else {
;      C5:
;        V = InV;
;        Out[V]--;
;      }
;      V = 999;
;
;    A6:
;      InV = In[Idx++];
;      if (InV < V + 42) {
;      B6:
;        V = V + 42;
;        Out[V]++;
;      } else {
;      C6:
;        V = InV;
;        Out[V]--;
;      }
;
;    A7:
;      InV = In[Idx++];
;      if (InV < V + 42) {
;      B7:
;        V = V + 42;
;        Out[V]++;
;      } else {
;      C7:
;        V = InV;
;        Out[V]--;
;      }
;      V = 999;
;
;    A8:
;      InV = In[Idx++];
;      if (InV < V + 42) {
;      B8:
;        V = V + 42;
;        Out[V]++;
;      } else {
;      C8:
;        V = InV;
;        Out[V]--;
;      }
;    FINAL:
;      Out[V]++;
;
;    ScopExit:
;      return;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i16* noalias %In, i32* noalias %Out) {
entry:
  %tmp = load i16, i16* %In, align 2
  %conv = sext i16 %tmp to i32
  %cmp = icmp slt i16 %tmp, 1041
  br i1 %cmp, label %B0, label %C0

B0:                                               ; preds = %entry
  %arrayidx4 = getelementptr inbounds i32, i32* %Out, i64 1041
  %tmp3 = load i32, i32* %arrayidx4, align 4
  %inc5 = add nsw i32 %tmp3, 1
  store i32 %inc5, i32* %arrayidx4, align 4
  br label %A1

C0:                                               ; preds = %entry
  %idxprom6 = sext i16 %tmp to i64
  %arrayidx7 = getelementptr inbounds i32, i32* %Out, i64 %idxprom6
  %tmp4 = load i32, i32* %arrayidx7, align 4
  %dec = add nsw i32 %tmp4, -1
  store i32 %dec, i32* %arrayidx7, align 4
  br label %A1

A1:                                               ; preds = %B0, %C0
  %V.0 = phi i32 [ 1041, %B0 ], [ %conv, %C0 ]
  %arrayidx10 = getelementptr inbounds i16, i16* %In, i64 1
  %tmp5 = load i16, i16* %arrayidx10, align 2
  %conv11 = sext i16 %tmp5 to i32
  %add12 = add nsw i32 %V.0, 42
  %cmp13 = icmp slt i32 %conv11, %add12
  br i1 %cmp13, label %B1, label %C1

B1:                                               ; preds = %A1
  %add16 = add nsw i32 %V.0, 42
  %idxprom17 = sext i32 %add16 to i64
  %arrayidx18 = getelementptr inbounds i32, i32* %Out, i64 %idxprom17
  %tmp6 = load i32, i32* %arrayidx18, align 4
  %inc19 = add nsw i32 %tmp6, 1
  store i32 %inc19, i32* %arrayidx18, align 4
  br label %A2

C1:                                               ; preds = %A1
  %idxprom21 = sext i16 %tmp5 to i64
  %arrayidx22 = getelementptr inbounds i32, i32* %Out, i64 %idxprom21
  %tmp7 = load i32, i32* %arrayidx22, align 4
  %dec23 = add nsw i32 %tmp7, -1
  store i32 %dec23, i32* %arrayidx22, align 4
  br label %A2

A2:                                               ; preds = %B1, %C1
  %arrayidx27 = getelementptr inbounds i16, i16* %In, i64 2
  %tmp8 = load i16, i16* %arrayidx27, align 2
  %conv28 = sext i16 %tmp8 to i32
  %cmp30 = icmp slt i16 %tmp8, 1041
  br i1 %cmp30, label %B2, label %C2

B2:                                               ; preds = %A2
  %arrayidx35 = getelementptr inbounds i32, i32* %Out, i64 1041
  %tmp9 = load i32, i32* %arrayidx35, align 4
  %inc36 = add nsw i32 %tmp9, 1
  store i32 %inc36, i32* %arrayidx35, align 4
  br label %A3

C2:                                               ; preds = %A2
  %idxprom38 = sext i16 %tmp8 to i64
  %arrayidx39 = getelementptr inbounds i32, i32* %Out, i64 %idxprom38
  %tmp10 = load i32, i32* %arrayidx39, align 4
  %dec40 = add nsw i32 %tmp10, -1
  store i32 %dec40, i32* %arrayidx39, align 4
  br label %A3

A3:                                               ; preds = %B2, %C2
  %V.1 = phi i32 [ 1041, %B2 ], [ %conv28, %C2 ]
  %arrayidx44 = getelementptr inbounds i16, i16* %In, i64 3
  %tmp11 = load i16, i16* %arrayidx44, align 2
  %conv45 = sext i16 %tmp11 to i32
  %add46 = add nsw i32 %V.1, 42
  %cmp47 = icmp slt i32 %conv45, %add46
  br i1 %cmp47, label %B3, label %C3

B3:                                               ; preds = %A3
  %add50 = add nsw i32 %V.1, 42
  %idxprom51 = sext i32 %add50 to i64
  %arrayidx52 = getelementptr inbounds i32, i32* %Out, i64 %idxprom51
  %tmp12 = load i32, i32* %arrayidx52, align 4
  %inc53 = add nsw i32 %tmp12, 1
  store i32 %inc53, i32* %arrayidx52, align 4
  br label %A4

C3:                                               ; preds = %A3
  %idxprom55 = sext i16 %tmp11 to i64
  %arrayidx56 = getelementptr inbounds i32, i32* %Out, i64 %idxprom55
  %tmp13 = load i32, i32* %arrayidx56, align 4
  %dec57 = add nsw i32 %tmp13, -1
  store i32 %dec57, i32* %arrayidx56, align 4
  br label %A4

A4:                                               ; preds = %B3, %C3
  %arrayidx61 = getelementptr inbounds i16, i16* %In, i64 4
  %tmp14 = load i16, i16* %arrayidx61, align 2
  %conv62 = sext i16 %tmp14 to i32
  %cmp64 = icmp slt i16 %tmp14, 1041
  br i1 %cmp64, label %B4, label %C4

B4:                                               ; preds = %A4
  %arrayidx69 = getelementptr inbounds i32, i32* %Out, i64 1041
  %tmp15 = load i32, i32* %arrayidx69, align 4
  %inc70 = add nsw i32 %tmp15, 1
  store i32 %inc70, i32* %arrayidx69, align 4
  br label %A5

C4:                                               ; preds = %A4
  %idxprom72 = sext i16 %tmp14 to i64
  %arrayidx73 = getelementptr inbounds i32, i32* %Out, i64 %idxprom72
  %tmp16 = load i32, i32* %arrayidx73, align 4
  %dec74 = add nsw i32 %tmp16, -1
  store i32 %dec74, i32* %arrayidx73, align 4
  %phitmp = add nsw i32 %conv62, 42
  br label %A5

A5:                                               ; preds = %B4, %C4
  %V.2 = phi i32 [ 1083, %B4 ], [ %phitmp, %C4 ]
  %arrayidx78 = getelementptr inbounds i16, i16* %In, i64 5
  %tmp17 = load i16, i16* %arrayidx78, align 2
  %conv79 = sext i16 %tmp17 to i32
  %cmp81 = icmp slt i32 %conv79, %V.2
  br i1 %cmp81, label %B5, label %C5

B5:                                               ; preds = %A5
  %idxprom85 = sext i32 %V.2 to i64
  %arrayidx86 = getelementptr inbounds i32, i32* %Out, i64 %idxprom85
  %tmp18 = load i32, i32* %arrayidx86, align 4
  %inc87 = add nsw i32 %tmp18, 1
  store i32 %inc87, i32* %arrayidx86, align 4
  br label %A6

C5:                                               ; preds = %A5
  %idxprom89 = sext i16 %tmp17 to i64
  %arrayidx90 = getelementptr inbounds i32, i32* %Out, i64 %idxprom89
  %tmp19 = load i32, i32* %arrayidx90, align 4
  %dec91 = add nsw i32 %tmp19, -1
  store i32 %dec91, i32* %arrayidx90, align 4
  br label %A6

A6:                                               ; preds = %B5, %C5
  %arrayidx95 = getelementptr inbounds i16, i16* %In, i64 6
  %tmp20 = load i16, i16* %arrayidx95, align 2
  %conv96 = sext i16 %tmp20 to i32
  %cmp98 = icmp slt i16 %tmp20, 1041
  br i1 %cmp98, label %B6, label %C6

B6:                                               ; preds = %A6
  %arrayidx103 = getelementptr inbounds i32, i32* %Out, i64 1041
  %tmp21 = load i32, i32* %arrayidx103, align 4
  %inc104 = add nsw i32 %tmp21, 1
  store i32 %inc104, i32* %arrayidx103, align 4
  br label %A7

C6:                                               ; preds = %A6
  %idxprom106 = sext i16 %tmp20 to i64
  %arrayidx107 = getelementptr inbounds i32, i32* %Out, i64 %idxprom106
  %tmp22 = load i32, i32* %arrayidx107, align 4
  %dec108 = add nsw i32 %tmp22, -1
  store i32 %dec108, i32* %arrayidx107, align 4
  %phitmp1 = add nsw i32 %conv96, 42
  br label %A7

A7:                                               ; preds = %B6, %C6
  %V.3 = phi i32 [ 1083, %B6 ], [ %phitmp1, %C6 ]
  %arrayidx112 = getelementptr inbounds i16, i16* %In, i64 7
  %tmp23 = load i16, i16* %arrayidx112, align 2
  %conv113 = sext i16 %tmp23 to i32
  %cmp115 = icmp slt i32 %conv113, %V.3
  br i1 %cmp115, label %B7, label %C7

B7:                                               ; preds = %A7
  %idxprom119 = sext i32 %V.3 to i64
  %arrayidx120 = getelementptr inbounds i32, i32* %Out, i64 %idxprom119
  %tmp24 = load i32, i32* %arrayidx120, align 4
  %inc121 = add nsw i32 %tmp24, 1
  store i32 %inc121, i32* %arrayidx120, align 4
  br label %A8

C7:                                               ; preds = %A7
  %idxprom123 = sext i16 %tmp23 to i64
  %arrayidx124 = getelementptr inbounds i32, i32* %Out, i64 %idxprom123
  %tmp25 = load i32, i32* %arrayidx124, align 4
  %dec125 = add nsw i32 %tmp25, -1
  store i32 %dec125, i32* %arrayidx124, align 4
  br label %A8

A8:                                               ; preds = %B7, %C7
  %arrayidx129 = getelementptr inbounds i16, i16* %In, i64 8
  %tmp26 = load i16, i16* %arrayidx129, align 2
  %cmp132 = icmp slt i16 %tmp26, 1041
  br i1 %cmp132, label %B8, label %C8

B8:                                               ; preds = %A8
  %arrayidx137 = getelementptr inbounds i32, i32* %Out, i64 1041
  %tmp27 = load i32, i32* %arrayidx137, align 4
  %inc138 = add nsw i32 %tmp27, 1
  store i32 %inc138, i32* %arrayidx137, align 4
  br label %FINAL

C8:                                               ; preds = %A8
  %idxprom140 = sext i16 %tmp26 to i64
  %arrayidx141 = getelementptr inbounds i32, i32* %Out, i64 %idxprom140
  %tmp28 = load i32, i32* %arrayidx141, align 4
  %dec142 = add nsw i32 %tmp28, -1
  store i32 %dec142, i32* %arrayidx141, align 4
  %phitmp2 = sext i16 %tmp26 to i64
  br label %FINAL

FINAL:                                        ; preds = %C8, %B8
  %V.4 = phi i64 [ 1041, %B8 ], [ %phitmp2, %C8 ]
  %arrayidx145 = getelementptr inbounds i32, i32* %Out, i64 %V.4
  %tmp29 = load i32, i32* %arrayidx145, align 4
  %inc146 = add nsw i32 %tmp29, 1
  store i32 %inc146, i32* %arrayidx145, align 4
  br label %ScopExit

ScopExit:
  ret void
}
