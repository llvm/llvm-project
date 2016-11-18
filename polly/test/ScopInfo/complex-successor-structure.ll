; RUN: opt %loadPolly -pass-remarks-analysis="polly-scops" -polly-scops \
; RUN: -polly-invariant-load-hoisting=true \
; RUN:     < %s 2>&1 | FileCheck %s

; We build a scop from the region for.body->B13. The CFG is of the
; following form. The test checks that the condition construction does not take
; a huge amount of time. While we can propagate the domain constraints from
; B(X) to B(X+1) the conditions in B(X+1) will exponentially grow the number
; of needed constraints (it is basically the condition of B(X) + one smax),
; thus we should bail out at some point.
;
; CHECK: Low complexity assumption: {  : 1 = 0 }

;      |
;    for.body <--+
;      |         |
;      |---------+
;      |
;     \ /
;    if.entry --+
;      |        |
;      A0       |
;      |        |
;      B0 <-----+
;      |  \
;      |   \
;      A1   \
;      |    |
;      |    |
;      B1<--+
;      |  \
;      |   \
;      A2   \
;      |    |
;      |    |
;      B2<--+
;      |  \
;      |   \
;      A3   \
;      |    |
;      |    |
;      B3<--+
;      |  \
;      |   \
;      A4   \
;      |    |
;      |    |
;      B4<--+
;      |  \
;      |   \
;      A5   \
;      |    |
;      |    |
;      B5<--+
;      |  \
;      |   \
;      A6   \
;      |    |
;      |    |
;      B6<--+
;      |  \
;      |   \
;      A7   \
;      |    |
;      |    |
;      B7<--+
;      |  \
;      |   \
;      A8   \
;      |    |
;      |    |
;      B8<--+
;      |  \
;      |   \
;      A9   \
;      |    |
;      |    |
;      B9<--+
;      |  \
;      |   \
;      A10  \
;      |    |
;      |    |
;      B10<-+
;      |  \
;      |   \
;      A11  \
;      |    |
;      |    |
;      B11<-+
;      |  \
;      |   \
;      A12  \
;      |    |
;      |    |
;      B12<-+
;      |  \
;      |   \
;      A13  \
;      |    |
;      |    |
;      B13<-+

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"
target triple = "thumbv7--linux-android"

@Table1 = external global [2304 x i16], align 2
@Table2 = external global [1792 x i16], align 2
@Table3 = external global [16 x i16], align 2

define void @foo(i16* nocapture readonly %indice, i16* nocapture %Output, i16* nocapture readonly %In1, i16* nocapture readonly %In2, i16 signext %var, i16 signext %var2) {
entry:
  %.reg2mem158 = alloca i16
  %.reg2mem156 = alloca i16
  %.reg2mem154 = alloca i16
  %.reg2mem152 = alloca i16
  %.reg2mem150 = alloca i16
  %.reg2mem = alloca i16
  %Temp_Ref = alloca [16 x i16], align 2
  %0 = bitcast [16 x i16]* %Temp_Ref to i8*
  %cmp = icmp eq i16 %var, 0
  br label %for.body

for.body:                                       ; preds = %for.body, %entry
  %i.2138 = phi i32 [ %inc47, %for.body ], [ 0, %entry ]
  %arrayidx28 = getelementptr inbounds [16 x i16], [16 x i16]* @Table3, i32 0, i32 %i.2138
  %1 = load i16, i16* %arrayidx28, align 2
  %conv29 = sext i16 %1 to i32
  %arrayidx36 = getelementptr inbounds i16, i16* %In2, i32 %i.2138
  %2 = load i16, i16* %arrayidx36, align 2
  %conv37 = sext i16 %2 to i32
  %shl38147 = add nsw i32 %conv37, %conv29
  %add35.1 = add nuw nsw i32 %i.2138, 16
  %arrayidx36.1 = getelementptr inbounds i16, i16* %In2, i32 %add35.1
  %3 = load i16, i16* %arrayidx36.1, align 2
  %conv37.1 = sext i16 %3 to i32
  %shl38.1148 = add nsw i32 %conv37.1, %shl38147
  %add35.2 = add nuw nsw i32 %i.2138, 32
  %arrayidx36.2 = getelementptr inbounds i16, i16* %In2, i32 %add35.2
  %4 = load i16, i16* %arrayidx36.2, align 2
  %conv37.2 = sext i16 %4 to i32
  %shl38.2149 = add nsw i32 %conv37.2, %shl38.1148
  %add39.2 = shl i32 %shl38.2149, 14
  %add43 = add nsw i32 %add39.2, 32768
  %shr129 = lshr i32 %add43, 16
  %conv44 = trunc i32 %shr129 to i16
  %arrayidx45 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 %i.2138
  store i16 %conv44, i16* %arrayidx45, align 2
  %inc47 = add nuw nsw i32 %i.2138, 1
  %exitcond144 = icmp eq i32 %i.2138, 15
  br i1 %exitcond144, label %if.entry, label %for.body

if.entry:                             ; preds = %for.body
  %5 = load i16, i16* %In1, align 2
  %conv54 = sext i16 %5 to i32
  %mul55 = mul nsw i32 %conv54, 29491
  %shr56127 = lshr i32 %mul55, 15
  %arrayidx57 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 0
  %6 = load i16, i16* %arrayidx57, align 2
  %conv58 = sext i16 %6 to i32
  %mul59 = mul nsw i32 %conv58, 3277
  %shr60128 = lshr i32 %mul59, 15
  %add61 = add nuw nsw i32 %shr60128, %shr56127
  %conv62 = trunc i32 %add61 to i16
  store i16 %conv62, i16* %Output, align 2
  %arrayidx53.1 = getelementptr inbounds i16, i16* %In1, i32 1
  %7 = load i16, i16* %arrayidx53.1, align 2
  %conv54.1 = sext i16 %7 to i32
  %mul55.1 = mul nsw i32 %conv54.1, 29491
  %shr56127.1 = lshr i32 %mul55.1, 15
  %arrayidx57.1 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 1
  %8 = load i16, i16* %arrayidx57.1, align 2
  %conv58.1 = sext i16 %8 to i32
  %mul59.1 = mul nsw i32 %conv58.1, 3277
  %shr60128.1 = lshr i32 %mul59.1, 15
  %add61.1 = add nuw nsw i32 %shr60128.1, %shr56127.1
  %conv62.1 = trunc i32 %add61.1 to i16
  %arrayidx63.1 = getelementptr inbounds i16, i16* %Output, i32 1
  store i16 %conv62.1, i16* %arrayidx63.1, align 2
  %arrayidx53.2 = getelementptr inbounds i16, i16* %In1, i32 2
  %9 = load i16, i16* %arrayidx53.2, align 2
  %conv54.2 = sext i16 %9 to i32
  %mul55.2 = mul nsw i32 %conv54.2, 29491
  %shr56127.2 = lshr i32 %mul55.2, 15
  %arrayidx57.2 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 2
  %10 = load i16, i16* %arrayidx57.2, align 2
  %conv58.2 = sext i16 %10 to i32
  %mul59.2 = mul nsw i32 %conv58.2, 3277
  %shr60128.2 = lshr i32 %mul59.2, 15
  %add61.2 = add nuw nsw i32 %shr60128.2, %shr56127.2
  %conv62.2 = trunc i32 %add61.2 to i16
  %arrayidx63.2 = getelementptr inbounds i16, i16* %Output, i32 2
  store i16 %conv62.2, i16* %arrayidx63.2, align 2
  %arrayidx53.3 = getelementptr inbounds i16, i16* %In1, i32 3
  %11 = load i16, i16* %arrayidx53.3, align 2
  %conv54.3 = sext i16 %11 to i32
  %mul55.3 = mul nsw i32 %conv54.3, 29491
  %shr56127.3 = lshr i32 %mul55.3, 15
  %arrayidx57.3 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 3
  %12 = load i16, i16* %arrayidx57.3, align 2
  %conv58.3 = sext i16 %12 to i32
  %mul59.3 = mul nsw i32 %conv58.3, 3277
  %shr60128.3 = lshr i32 %mul59.3, 15
  %add61.3 = add nuw nsw i32 %shr60128.3, %shr56127.3
  %conv62.3 = trunc i32 %add61.3 to i16
  %arrayidx63.3 = getelementptr inbounds i16, i16* %Output, i32 3
  store i16 %conv62.3, i16* %arrayidx63.3, align 2
  %arrayidx53.4 = getelementptr inbounds i16, i16* %In1, i32 4
  %13 = load i16, i16* %arrayidx53.4, align 2
  %conv54.4 = sext i16 %13 to i32
  %mul55.4 = mul nsw i32 %conv54.4, 29491
  %shr56127.4 = lshr i32 %mul55.4, 15
  %arrayidx57.4 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 4
  %14 = load i16, i16* %arrayidx57.4, align 2
  %conv58.4 = sext i16 %14 to i32
  %mul59.4 = mul nsw i32 %conv58.4, 3277
  %shr60128.4 = lshr i32 %mul59.4, 15
  %add61.4 = add nuw nsw i32 %shr60128.4, %shr56127.4
  %conv62.4 = trunc i32 %add61.4 to i16
  %arrayidx63.4 = getelementptr inbounds i16, i16* %Output, i32 4
  store i16 %conv62.4, i16* %arrayidx63.4, align 2
  %arrayidx53.5 = getelementptr inbounds i16, i16* %In1, i32 5
  %15 = load i16, i16* %arrayidx53.5, align 2
  %conv54.5 = sext i16 %15 to i32
  %mul55.5 = mul nsw i32 %conv54.5, 29491
  %shr56127.5 = lshr i32 %mul55.5, 15
  %arrayidx57.5 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 5
  %16 = load i16, i16* %arrayidx57.5, align 2
  %conv58.5 = sext i16 %16 to i32
  %mul59.5 = mul nsw i32 %conv58.5, 3277
  %shr60128.5 = lshr i32 %mul59.5, 15
  %add61.5 = add nuw nsw i32 %shr60128.5, %shr56127.5
  %conv62.5 = trunc i32 %add61.5 to i16
  %arrayidx63.5 = getelementptr inbounds i16, i16* %Output, i32 5
  store i16 %conv62.5, i16* %arrayidx63.5, align 2
  %arrayidx53.6 = getelementptr inbounds i16, i16* %In1, i32 6
  %17 = load i16, i16* %arrayidx53.6, align 2
  %conv54.6 = sext i16 %17 to i32
  %mul55.6 = mul nsw i32 %conv54.6, 29491
  %shr56127.6 = lshr i32 %mul55.6, 15
  %arrayidx57.6 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 6
  %18 = load i16, i16* %arrayidx57.6, align 2
  %conv58.6 = sext i16 %18 to i32
  %mul59.6 = mul nsw i32 %conv58.6, 3277
  %shr60128.6 = lshr i32 %mul59.6, 15
  %add61.6 = add nuw nsw i32 %shr60128.6, %shr56127.6
  %conv62.6 = trunc i32 %add61.6 to i16
  %arrayidx63.6 = getelementptr inbounds i16, i16* %Output, i32 6
  store i16 %conv62.6, i16* %arrayidx63.6, align 2
  %arrayidx53.7 = getelementptr inbounds i16, i16* %In1, i32 7
  %19 = load i16, i16* %arrayidx53.7, align 2
  %conv54.7 = sext i16 %19 to i32
  %mul55.7 = mul nsw i32 %conv54.7, 29491
  %shr56127.7 = lshr i32 %mul55.7, 15
  %arrayidx57.7 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 7
  %20 = load i16, i16* %arrayidx57.7, align 2
  %conv58.7 = sext i16 %20 to i32
  %mul59.7 = mul nsw i32 %conv58.7, 3277
  %shr60128.7 = lshr i32 %mul59.7, 15
  %add61.7 = add nuw nsw i32 %shr60128.7, %shr56127.7
  %conv62.7 = trunc i32 %add61.7 to i16
  %arrayidx63.7 = getelementptr inbounds i16, i16* %Output, i32 7
  store i16 %conv62.7, i16* %arrayidx63.7, align 2
  %arrayidx53.8 = getelementptr inbounds i16, i16* %In1, i32 8
  %21 = load i16, i16* %arrayidx53.8, align 2
  %conv54.8 = sext i16 %21 to i32
  %mul55.8 = mul nsw i32 %conv54.8, 29491
  %shr56127.8 = lshr i32 %mul55.8, 15
  %arrayidx57.8 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 8
  %22 = load i16, i16* %arrayidx57.8, align 2
  %conv58.8 = sext i16 %22 to i32
  %mul59.8 = mul nsw i32 %conv58.8, 3277
  %shr60128.8 = lshr i32 %mul59.8, 15
  %add61.8 = add nuw nsw i32 %shr60128.8, %shr56127.8
  %conv62.8 = trunc i32 %add61.8 to i16
  %arrayidx63.8 = getelementptr inbounds i16, i16* %Output, i32 8
  store i16 %conv62.8, i16* %arrayidx63.8, align 2
  %arrayidx53.9 = getelementptr inbounds i16, i16* %In1, i32 9
  %23 = load i16, i16* %arrayidx53.9, align 2
  %conv54.9 = sext i16 %23 to i32
  %mul55.9 = mul nsw i32 %conv54.9, 29491
  %shr56127.9 = lshr i32 %mul55.9, 15
  %arrayidx57.9 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 9
  %24 = load i16, i16* %arrayidx57.9, align 2
  %conv58.9 = sext i16 %24 to i32
  %mul59.9 = mul nsw i32 %conv58.9, 3277
  %shr60128.9 = lshr i32 %mul59.9, 15
  %add61.9 = add nuw nsw i32 %shr60128.9, %shr56127.9
  %conv62.9 = trunc i32 %add61.9 to i16
  %arrayidx63.9 = getelementptr inbounds i16, i16* %Output, i32 9
  store i16 %conv62.9, i16* %arrayidx63.9, align 2
  %arrayidx53.10 = getelementptr inbounds i16, i16* %In1, i32 10
  %25 = load i16, i16* %arrayidx53.10, align 2
  %conv54.10 = sext i16 %25 to i32
  %mul55.10 = mul nsw i32 %conv54.10, 29491
  %shr56127.10 = lshr i32 %mul55.10, 15
  %arrayidx57.10 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 10
  %26 = load i16, i16* %arrayidx57.10, align 2
  %conv58.10 = sext i16 %26 to i32
  %mul59.10 = mul nsw i32 %conv58.10, 3277
  %shr60128.10 = lshr i32 %mul59.10, 15
  %add61.10 = add nuw nsw i32 %shr60128.10, %shr56127.10
  %conv62.10 = trunc i32 %add61.10 to i16
  %arrayidx63.10 = getelementptr inbounds i16, i16* %Output, i32 10
  store i16 %conv62.10, i16* %arrayidx63.10, align 2
  %arrayidx53.11 = getelementptr inbounds i16, i16* %In1, i32 11
  %27 = load i16, i16* %arrayidx53.11, align 2
  %conv54.11 = sext i16 %27 to i32
  %mul55.11 = mul nsw i32 %conv54.11, 29491
  %shr56127.11 = lshr i32 %mul55.11, 15
  %arrayidx57.11 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 11
  %28 = load i16, i16* %arrayidx57.11, align 2
  %conv58.11 = sext i16 %28 to i32
  %mul59.11 = mul nsw i32 %conv58.11, 3277
  %shr60128.11 = lshr i32 %mul59.11, 15
  %add61.11 = add nuw nsw i32 %shr60128.11, %shr56127.11
  %conv62.11 = trunc i32 %add61.11 to i16
  %arrayidx63.11 = getelementptr inbounds i16, i16* %Output, i32 11
  store i16 %conv62.11, i16* %arrayidx63.11, align 2
  %arrayidx53.12 = getelementptr inbounds i16, i16* %In1, i32 12
  %29 = load i16, i16* %arrayidx53.12, align 2
  %conv54.12 = sext i16 %29 to i32
  %mul55.12 = mul nsw i32 %conv54.12, 29491
  %shr56127.12 = lshr i32 %mul55.12, 15
  %arrayidx57.12 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 12
  %30 = load i16, i16* %arrayidx57.12, align 2
  %conv58.12 = sext i16 %30 to i32
  %mul59.12 = mul nsw i32 %conv58.12, 3277
  %shr60128.12 = lshr i32 %mul59.12, 15
  %add61.12 = add nuw nsw i32 %shr60128.12, %shr56127.12
  %conv62.12 = trunc i32 %add61.12 to i16
  %arrayidx63.12 = getelementptr inbounds i16, i16* %Output, i32 12
  store i16 %conv62.12, i16* %arrayidx63.12, align 2
  %arrayidx53.13 = getelementptr inbounds i16, i16* %In1, i32 13
  %31 = load i16, i16* %arrayidx53.13, align 2
  %conv54.13 = sext i16 %31 to i32
  %mul55.13 = mul nsw i32 %conv54.13, 29491
  %shr56127.13 = lshr i32 %mul55.13, 15
  %arrayidx57.13 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 13
  %32 = load i16, i16* %arrayidx57.13, align 2
  %conv58.13 = sext i16 %32 to i32
  %mul59.13 = mul nsw i32 %conv58.13, 3277
  %shr60128.13 = lshr i32 %mul59.13, 15
  %add61.13 = add nuw nsw i32 %shr60128.13, %shr56127.13
  %conv62.13 = trunc i32 %add61.13 to i16
  %arrayidx63.13 = getelementptr inbounds i16, i16* %Output, i32 13
  store i16 %conv62.13, i16* %arrayidx63.13, align 2
  %arrayidx53.14 = getelementptr inbounds i16, i16* %In1, i32 14
  %33 = load i16, i16* %arrayidx53.14, align 2
  %conv54.14 = sext i16 %33 to i32
  %mul55.14 = mul nsw i32 %conv54.14, 29491
  %shr56127.14 = lshr i32 %mul55.14, 15
  %arrayidx57.14 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 14
  %34 = load i16, i16* %arrayidx57.14, align 2
  %conv58.14 = sext i16 %34 to i32
  %mul59.14 = mul nsw i32 %conv58.14, 3277
  %shr60128.14 = lshr i32 %mul59.14, 15
  %add61.14 = add nuw nsw i32 %shr60128.14, %shr56127.14
  %conv62.14 = trunc i32 %add61.14 to i16
  %arrayidx63.14 = getelementptr inbounds i16, i16* %Output, i32 14
  store i16 %conv62.14, i16* %arrayidx63.14, align 2
  %arrayidx53.15 = getelementptr inbounds i16, i16* %In1, i32 15
  %35 = load i16, i16* %arrayidx53.15, align 2
  %conv54.15 = sext i16 %35 to i32
  %mul55.15 = mul nsw i32 %conv54.15, 29491
  %shr56127.15 = lshr i32 %mul55.15, 15
  %arrayidx57.15 = getelementptr inbounds [16 x i16], [16 x i16]* %Temp_Ref, i32 0, i32 15
  %36 = load i16, i16* %arrayidx57.15, align 2
  %conv58.15 = sext i16 %36 to i32
  %mul59.15 = mul nsw i32 %conv58.15, 3277
  %shr60128.15 = lshr i32 %mul59.15, 15
  %add61.15 = add nuw nsw i32 %shr60128.15, %shr56127.15
  %conv62.15 = trunc i32 %add61.15 to i16
  %arrayidx63.15 = getelementptr inbounds i16, i16* %Output, i32 15
  store i16 %conv62.15, i16* %arrayidx63.15, align 2
  store i16 %conv62.9, i16* %.reg2mem
  store i16 %conv62.10, i16* %.reg2mem150
  store i16 %conv62.11, i16* %.reg2mem152
  store i16 %conv62.12, i16* %.reg2mem154
  store i16 %conv62.13, i16* %.reg2mem156
  store i16 %conv62.14, i16* %.reg2mem158
  %.reload159 = load i16, i16* %.reg2mem158
  %.reload157 = load i16, i16* %.reg2mem156
  %.reload155 = load i16, i16* %.reg2mem154
  %.reload153 = load i16, i16* %.reg2mem152
  %.reload151 = load i16, i16* %.reg2mem150
  %.reload = load i16, i16* %.reg2mem
  %37 = load i16, i16* %In1, align 2
  %cmp77 = icmp slt i16 %37, 128
  br i1 %cmp77, label %A0, label %B0

A0:                                        ; preds = %if.entry
  store i16 128, i16* %Output, align 2
  br label %B0

B0:                                         ; preds = %A, %if.entry
  %38 = phi i16 [ 128, %A0 ], [ %37, %if.entry ]
  %add84 = add i16 %38, 128
  %arrayidx74.1 = getelementptr inbounds i16, i16* %Output, i32 1
  %39 = load i16, i16* %arrayidx74.1, align 2
  %cmp77.1 = icmp slt i16 %39, %add84
  br i1 %cmp77.1, label %A1, label %B1

A1:                                      ; preds = %B
  store i16 %add84, i16* %arrayidx74.1, align 2
  br label %B1

B1:                                       ; preds = %A1, %B
  %40 = phi i16 [ %add84, %A1 ], [ %39, %B0 ]
  %add84.1 = add i16 %40, 128
  %arrayidx74.2 = getelementptr inbounds i16, i16* %Output, i32 2
  %41 = load i16, i16* %arrayidx74.2, align 2
  %cmp77.2 = icmp slt i16 %41, %add84.1
  br i1 %cmp77.2, label %A2, label %B2

A2:                                      ; preds = %B1
  store i16 %add84.1, i16* %arrayidx74.2, align 2
  br label %B2

B2:                                       ; preds = %A2, %B1
  %42 = phi i16 [ %add84.1, %A2 ], [ %41, %B1 ]
  %add84.2 = add i16 %42, 128
  %arrayidx74.3 = getelementptr inbounds i16, i16* %Output, i32 3
  %43 = load i16, i16* %arrayidx74.3, align 2
  %cmp77.3 = icmp slt i16 %43, %add84.2
  br i1 %cmp77.3, label %A3, label %B3

A3:                                      ; preds = %B2
  store i16 %add84.2, i16* %arrayidx74.3, align 2
  br label %B3

B3:                                       ; preds = %A3, %B2
  %44 = phi i16 [ %add84.2, %A3 ], [ %43, %B2 ]
  %add84.3 = add i16 %44, 128
  %arrayidx74.4 = getelementptr inbounds i16, i16* %Output, i32 4
  %45 = load i16, i16* %arrayidx74.4, align 2
  %cmp77.4 = icmp slt i16 %45, %add84.3
  br i1 %cmp77.4, label %A4, label %B4

A4:                                      ; preds = %B3
  store i16 %add84.3, i16* %arrayidx74.4, align 2
  br label %B4

B4:                                       ; preds = %A4, %B3
  %46 = phi i16 [ %add84.3, %A4 ], [ %45, %B3 ]
  %add84.4 = add i16 %46, 128
  %arrayidx74.5 = getelementptr inbounds i16, i16* %Output, i32 5
  %47 = load i16, i16* %arrayidx74.5, align 2
  %cmp77.5 = icmp slt i16 %47, %add84.4
  br i1 %cmp77.5, label %A5, label %B5

A5:                                      ; preds = %B4
  store i16 %add84.4, i16* %arrayidx74.5, align 2
  br label %B5

B5:                                       ; preds = %A5, %B4
  %48 = phi i16 [ %add84.4, %A5 ], [ %47, %B4 ]
  %add84.5 = add i16 %48, 128
  %arrayidx74.6 = getelementptr inbounds i16, i16* %Output, i32 6
  %49 = load i16, i16* %arrayidx74.6, align 2
  %cmp77.6 = icmp slt i16 %49, %add84.5
  br i1 %cmp77.6, label %A6, label %B6

A6:                                      ; preds = %B5
  store i16 %add84.5, i16* %arrayidx74.6, align 2
  br label %B6

B6:                                       ; preds = %A6, %B5
  %50 = phi i16 [ %add84.5, %A6 ], [ %49, %B5 ]
  %add84.6 = add i16 %50, 128
  %arrayidx74.7 = getelementptr inbounds i16, i16* %Output, i32 7
  %51 = load i16, i16* %arrayidx74.7, align 2
  %cmp77.7 = icmp slt i16 %51, %add84.6
  br i1 %cmp77.7, label %A7, label %B7

A7:                                      ; preds = %B6
  store i16 %add84.6, i16* %arrayidx74.7, align 2
  br label %B7

B7:                                       ; preds = %A7, %B6
  %52 = phi i16 [ %add84.6, %A7 ], [ %51, %B6 ]
  %add84.7 = add i16 %52, 128
  %arrayidx74.8 = getelementptr inbounds i16, i16* %Output, i32 8
  %53 = load i16, i16* %arrayidx74.8, align 2
  %cmp77.8 = icmp slt i16 %53, %add84.7
  br i1 %cmp77.8, label %A8, label %B8

A8:                                      ; preds = %B7
  store i16 %add84.7, i16* %arrayidx74.8, align 2
  br label %B8

B8:                                       ; preds = %A8, %B7
  %54 = phi i16 [ %add84.7, %A8 ], [ %53, %B7 ]
  %add84.8 = add i16 %54, 128
  %cmp77.9 = icmp slt i16 %.reload, %add84.8
  br i1 %cmp77.9, label %A9, label %B9

A9:                                      ; preds = %B8
  %arrayidx74.9 = getelementptr inbounds i16, i16* %Output, i32 9
  store i16 %add84.8, i16* %arrayidx74.9, align 2
  br label %B9

B9:                                       ; preds = %A9, %B8
  %55 = phi i16 [ %add84.8, %A9 ], [ %.reload, %B8 ]
  %add84.9 = add i16 %55, 128
  %cmp77.10 = icmp slt i16 %.reload151, %add84.9
  br i1 %cmp77.10, label %A10, label %B10

A10:                                     ; preds = %B9
  %arrayidx74.10 = getelementptr inbounds i16, i16* %Output, i32 10
  store i16 %add84.9, i16* %arrayidx74.10, align 2
  br label %B10

B10:                                      ; preds = %A10, %B9
  %56 = phi i16 [ %add84.9, %A10 ], [ %.reload151, %B9 ]
  %add84.10 = add i16 %56, 128
  %cmp77.11 = icmp slt i16 %.reload153, %add84.10
  br i1 %cmp77.11, label %A11, label %B11

A11:                                     ; preds = %B10
  %arrayidx74.11 = getelementptr inbounds i16, i16* %Output, i32 11
  store i16 %add84.10, i16* %arrayidx74.11, align 2
  br label %B11

B11:                                      ; preds = %A11, %B10
  %57 = phi i16 [ %add84.10, %A11 ], [ %.reload153, %B10 ]
  %add84.11 = add i16 %57, 128
  %cmp77.12 = icmp slt i16 %.reload155, %add84.11
  br i1 %cmp77.12, label %A12, label %B13

A12:                                     ; preds = %B11
  %arrayidx74.12 = getelementptr inbounds i16, i16* %Output, i32 12
  store i16 %add84.11, i16* %arrayidx74.12, align 2
  br label %B13

B13:                                      ; preds = %A12, %B13
  ret void
}
