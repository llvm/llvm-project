; ModuleID = 'matmul.s'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [1536 x [1536 x float]] zeroinitializer, align 16
@B = common global [1536 x [1536 x float]] zeroinitializer, align 16
@stdout = external global %struct._IO_FILE*
@.str = private unnamed_addr constant [5 x i8] c"%lf \00", align 1
@C = common global [1536 x [1536 x float]] zeroinitializer, align 16
@.str1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

; Function Attrs: nounwind uwtable
define void @init_array() #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc17, %entry
  %0 = phi i64 [ %indvar.next2, %for.inc17 ], [ 0, %entry ]
  %exitcond3 = icmp ne i64 %0, 1536
  br i1 %exitcond3, label %for.body, label %for.end19

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %for.body ]
  %arrayidx6 = getelementptr [1536 x [1536 x float]]* @A, i64 0, i64 %0, i64 %indvar
  %arrayidx16 = getelementptr [1536 x [1536 x float]]* @B, i64 0, i64 %0, i64 %indvar
  %1 = mul i64 %0, %indvar
  %mul = trunc i64 %1 to i32
  %exitcond = icmp ne i64 %indvar, 1536
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %rem = srem i32 %mul, 1024
  %add = add nsw i32 1, %rem
  %conv = sitofp i32 %add to double
  %div = fdiv double %conv, 2.000000e+00
  %conv4 = fptrunc double %div to float
  store float %conv4, float* %arrayidx6, align 4
  %rem8 = srem i32 %mul, 1024
  %add9 = add nsw i32 1, %rem8
  %conv10 = sitofp i32 %add9 to double
  %div11 = fdiv double %conv10, 2.000000e+00
  %conv12 = fptrunc double %div11 to float
  store float %conv12, float* %arrayidx16, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %indvar.next = add i64 %indvar, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc17

for.inc17:                                        ; preds = %for.end
  %indvar.next2 = add i64 %0, 1
  br label %for.cond

for.end19:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind uwtable
define void @print_array() #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %indvar1 = phi i64 [ %indvar.next2, %for.inc10 ], [ 0, %entry ]
  %exitcond3 = icmp ne i64 %indvar1, 1536
  br i1 %exitcond3, label %for.body, label %for.end12

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %for.body ]
  %arrayidx5 = getelementptr [1536 x [1536 x float]]* @C, i64 0, i64 %indvar1, i64 %indvar
  %j.0 = trunc i64 %indvar to i32
  %exitcond = icmp ne i64 %indvar, 1536
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %0 = load %struct._IO_FILE** @stdout, align 8
  %1 = load float* %arrayidx5, align 4
  %conv = fpext float %1 to double
  %call = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %0, i8* getelementptr inbounds ([5 x i8]* @.str, i32 0, i32 0), double %conv)
  %rem = srem i32 %j.0, 80
  %cmp6 = icmp eq i32 %rem, 79
  br i1 %cmp6, label %if.then, label %if.end

if.then:                                          ; preds = %for.body3
  %2 = load %struct._IO_FILE** @stdout, align 8
  %call8 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %2, i8* getelementptr inbounds ([2 x i8]* @.str1, i32 0, i32 0))
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body3
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvar.next = add i64 %indvar, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  %3 = load %struct._IO_FILE** @stdout, align 8
  %call9 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %3, i8* getelementptr inbounds ([2 x i8]* @.str1, i32 0, i32 0))
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %indvar.next2 = add i64 %indvar1, 1
  br label %for.cond

for.end12:                                        ; preds = %for.cond
  ret void
}

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...) #1

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  call void @init_array()
  br label %for.cond

for.cond:                                         ; preds = %for.inc28, %entry
  %indvar3 = phi i64 [ %indvar.next4, %for.inc28 ], [ 0, %entry ]
  %exitcond6 = icmp ne i64 %indvar3, 1536
  br i1 %exitcond6, label %for.body, label %for.end30

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc25, %for.body
  %indvar1 = phi i64 [ %indvar.next2, %for.inc25 ], [ 0, %for.body ]
  %arrayidx5 = getelementptr [1536 x [1536 x float]]* @C, i64 0, i64 %indvar3, i64 %indvar1
  %exitcond5 = icmp ne i64 %indvar1, 1536
  br i1 %exitcond5, label %for.body3, label %for.end27

for.body3:                                        ; preds = %for.cond1
  store float 0.000000e+00, float* %arrayidx5, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %for.body3
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %for.body3 ]
  %arrayidx16 = getelementptr [1536 x [1536 x float]]* @A, i64 0, i64 %indvar3, i64 %indvar
  %arrayidx20 = getelementptr [1536 x [1536 x float]]* @B, i64 0, i64 %indvar, i64 %indvar1
  %exitcond = icmp ne i64 %indvar, 1536
  br i1 %exitcond, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond6
  %0 = load float* %arrayidx5, align 4
  %1 = load float* %arrayidx16, align 4
  %2 = load float* %arrayidx20, align 4
  %mul = fmul float %1, %2
  %add = fadd float %0, %mul
  store float %add, float* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body8
  %indvar.next = add i64 %indvar, 1
  br label %for.cond6

for.end:                                          ; preds = %for.cond6
  br label %for.inc25

for.inc25:                                        ; preds = %for.end
  %indvar.next2 = add i64 %indvar1, 1
  br label %for.cond1

for.end27:                                        ; preds = %for.cond1
  br label %for.inc28

for.inc28:                                        ; preds = %for.end27
  %indvar.next4 = add i64 %indvar3, 1
  br label %for.cond

for.end30:                                        ; preds = %for.cond
  ret i32 0
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
