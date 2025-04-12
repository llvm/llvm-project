; RUN: opt %loadNPMPolly -pass-remarks-analysis="polly-scops" '-passes=print<polly-function-scops>' \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Low complexity assumption: {  : false }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.bc_struct.0.2.4.6.13.20.27.43.44.46.50.52.58.60.81.89.90.99.107.108.109.111.116.118.149 = type { i32, i32, i32, i32, [1024 x i8] }

; Function Attrs: nounwind uwtable
define void @bc_multiply(ptr readonly %n1, i32 %scale) #0 {
entry:
  %0 = load i32, ptr undef, align 4
  %1 = load i32, ptr undef, align 4
  %2 = load i32, ptr undef, align 4
  %add3 = add nsw i32 %2, %1
  %cmp = icmp sgt i32 %0, %2
  %. = select i1 %cmp, i32 %0, i32 %2
  %cmp12 = icmp slt i32 %., %scale
  %scale.. = select i1 %cmp12, i32 %scale, i32 %.
  %cmp26 = icmp sgt i32 0, %scale..
  %scale...add7 = select i1 %cmp26, i32 %scale.., i32 0
  %sub = sub nsw i32 0, %scale...add7
  %add.ptr = getelementptr inbounds %struct.bc_struct.0.2.4.6.13.20.27.43.44.46.50.52.58.60.81.89.90.99.107.108.109.111.116.118.149, ptr %n1, i64 0, i32 4, i64 0
  %add.ptr59 = getelementptr inbounds i8, ptr %add.ptr, i64 -1
  %idx.ext62 = sext i32 %add3 to i64
  %cmp70140 = icmp sgt i32 %sub, 0
  br label %for.body104.lr.ph

for.body104.lr.ph:                                ; preds = %entry
  %3 = add i32 0, -1
  %4 = sub i32 %3, %scale...add7
  %5 = add i32 %4, 1
  %6 = sext i32 %5 to i64
  br label %for.body104

for.body104:                                      ; preds = %while.end146, %for.body104.lr.ph
  %indvars.iv = phi i64 [ %6, %for.body104.lr.ph ], [ poison, %while.end146 ]
  %7 = sub nsw i64 %indvars.iv, %idx.ext62
  %cmp107 = icmp slt i64 %7, -1
  %.op = xor i64 %7, -1
  %idx.neg116 = select i1 %cmp107, i64 0, i64 %.op
  %add.ptr117 = getelementptr inbounds i8, ptr %add.ptr59, i64 %idx.neg116
  br label %while.body138

while.body138:                                    ; preds = %while.body138, %for.body104
  %n1ptr.1126 = phi ptr [ %incdec.ptr139, %while.body138 ], [ %add.ptr117, %for.body104 ]
  %incdec.ptr139 = getelementptr inbounds i8, ptr %n1ptr.1126, i64 -1
  %cmp132 = icmp uge ptr %incdec.ptr139, null
  %cmp135 = icmp slt i64 0, -1
  %or.cond99 = and i1 %cmp135, %cmp132
  br i1 %or.cond99, label %while.body138, label %while.end146

while.end146:                                     ; preds = %while.body138
  br i1 undef, label %free_num.exit, label %for.body104

free_num.exit:                                    ; preds = %while.end146
  ret void
}
