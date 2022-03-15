; RUN: opt %loadPolly -basic-aa -polly-opt-isl -polly-vectorizer=polly -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [1536 x [1536 x float]] zeroinitializer, align 16
@B = common global [1536 x [1536 x float]] zeroinitializer, align 16
@stdout = external global %struct._IO_FILE*
@.str = private unnamed_addr constant [5 x i8] c"%lf \00", align 1
@C = common global [1536 x [1536 x float]] zeroinitializer, align 16
@.str1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

define void @init_array() nounwind uwtable {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc17, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc18, %for.inc17 ]
  %cmp = icmp slt i32 %i.0, 1536
  br i1 %cmp, label %for.body, label %for.end19

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp2 = icmp slt i32 %j.0, 1536
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %mul = mul nsw i32 %i.0, %j.0
  %rem = srem i32 %mul, 1024
  %add = add nsw i32 1, %rem
  %conv = sitofp i32 %add to double
  %div = fdiv double %conv, 2.000000e+00
  %conv4 = fptrunc double %div to float
  %idxprom = sext i32 %j.0 to i64
  %idxprom5 = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i32 0, i64 %idxprom5
  %arrayidx6 = getelementptr inbounds [1536 x float], [1536 x float]* %arrayidx, i32 0, i64 %idxprom
  store float %conv4, float* %arrayidx6, align 4
  %mul7 = mul nsw i32 %i.0, %j.0
  %rem8 = srem i32 %mul7, 1024
  %add9 = add nsw i32 1, %rem8
  %conv10 = sitofp i32 %add9 to double
  %div11 = fdiv double %conv10, 2.000000e+00
  %conv12 = fptrunc double %div11 to float
  %idxprom13 = sext i32 %j.0 to i64
  %idxprom14 = sext i32 %i.0 to i64
  %arrayidx15 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i32 0, i64 %idxprom14
  %arrayidx16 = getelementptr inbounds [1536 x float], [1536 x float]* %arrayidx15, i32 0, i64 %idxprom13
  store float %conv12, float* %arrayidx16, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc17

for.inc17:                                        ; preds = %for.end
  %inc18 = add nsw i32 %i.0, 1
  br label %for.cond

for.end19:                                        ; preds = %for.cond
  ret void
}

define void @print_array() nounwind uwtable {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc11, %for.inc10 ]
  %cmp = icmp slt i32 %i.0, 1536
  br i1 %cmp, label %for.body, label %for.end12

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp2 = icmp slt i32 %j.0, 1536
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  %idxprom = sext i32 %j.0 to i64
  %idxprom4 = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i32 0, i64 %idxprom4
  %arrayidx5 = getelementptr inbounds [1536 x float], [1536 x float]* %arrayidx, i32 0, i64 %idxprom
  %1 = load float, float* %arrayidx5, align 4
  %conv = fpext float %1 to double
  %call = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %0, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i32 0, i32 0), double %conv)
  %rem = srem i32 %j.0, 80
  %cmp6 = icmp eq i32 %rem, 79
  br i1 %cmp6, label %if.then, label %if.end

if.then:                                          ; preds = %for.body3
  %2 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  %call8 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %2, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str1, i32 0, i32 0))
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body3
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  %3 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  %call9 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %3, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str1, i32 0, i32 0))
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %inc11 = add nsw i32 %i.0, 1
  br label %for.cond

for.end12:                                        ; preds = %for.cond
  ret void
}

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...)

define i32 @main() nounwind uwtable {
entry:
  call void @init_array()
  br label %for.cond

for.cond:                                         ; preds = %for.inc28, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc29, %for.inc28 ]
  %cmp = icmp slt i32 %i.0, 1536
  br i1 %cmp, label %for.body, label %for.end30

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc25, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc26, %for.inc25 ]
  %cmp2 = icmp slt i32 %j.0, 1536
  br i1 %cmp2, label %for.body3, label %for.end27

for.body3:                                        ; preds = %for.cond1
  %idxprom = sext i32 %j.0 to i64
  %idxprom4 = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i32 0, i64 %idxprom4
  %arrayidx5 = getelementptr inbounds [1536 x float], [1536 x float]* %arrayidx, i32 0, i64 %idxprom
  store float 0.000000e+00, float* %arrayidx5, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i32 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %cmp7 = icmp slt i32 %k.0, 1536
  br i1 %cmp7, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond6
  %idxprom9 = sext i32 %j.0 to i64
  %idxprom10 = sext i32 %i.0 to i64
  %arrayidx11 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i32 0, i64 %idxprom10
  %arrayidx12 = getelementptr inbounds [1536 x float], [1536 x float]* %arrayidx11, i32 0, i64 %idxprom9
  %0 = load float, float* %arrayidx12, align 4
  %idxprom13 = sext i32 %k.0 to i64
  %idxprom14 = sext i32 %i.0 to i64
  %arrayidx15 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i32 0, i64 %idxprom14
  %arrayidx16 = getelementptr inbounds [1536 x float], [1536 x float]* %arrayidx15, i32 0, i64 %idxprom13
  %1 = load float, float* %arrayidx16, align 4
  %idxprom17 = sext i32 %j.0 to i64
  %idxprom18 = sext i32 %k.0 to i64
  %arrayidx19 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i32 0, i64 %idxprom18
  %arrayidx20 = getelementptr inbounds [1536 x float], [1536 x float]* %arrayidx19, i32 0, i64 %idxprom17
  %2 = load float, float* %arrayidx20, align 4
  %mul = fmul float %1, %2
  %add = fadd float %0, %mul
  %idxprom21 = sext i32 %j.0 to i64
  %idxprom22 = sext i32 %i.0 to i64
  %arrayidx23 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i32 0, i64 %idxprom22
  %arrayidx24 = getelementptr inbounds [1536 x float], [1536 x float]* %arrayidx23, i32 0, i64 %idxprom21
  store float %add, float* %arrayidx24, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body8
  %inc = add nsw i32 %k.0, 1
  br label %for.cond6

for.end:                                          ; preds = %for.cond6
  br label %for.inc25

for.inc25:                                        ; preds = %for.end
  %inc26 = add nsw i32 %j.0, 1
  br label %for.cond1

for.end27:                                        ; preds = %for.cond1
  br label %for.inc28

for.inc28:                                        ; preds = %for.end27
  %inc29 = add nsw i32 %i.0, 1
  br label %for.cond

for.end30:                                        ; preds = %for.cond
  ret i32 0
}
