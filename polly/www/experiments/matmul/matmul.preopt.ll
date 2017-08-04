; ModuleID = 'matmul.s'
source_filename = "matmul.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [1536 x [1536 x float]] zeroinitializer, align 16
@B = common global [1536 x [1536 x float]] zeroinitializer, align 16
@stdout = external global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [5 x i8] c"%lf \00", align 1
@C = common global [1536 x [1536 x float]] zeroinitializer, align 16
@.str.1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

; Function Attrs: nounwind uwtable
define void @init_array() #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry.split, %for.inc17
  %indvars.iv5 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next6, %for.inc17 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %0 = mul nuw nsw i64 %indvars.iv, %indvars.iv5
  %1 = trunc i64 %0 to i32
  %rem = srem i32 %1, 1024
  %add = add nsw i32 %rem, 1
  %conv = sitofp i32 %add to double
  %div = fmul double %conv, 5.000000e-01
  %conv4 = fptrunc double %div to float
  %arrayidx6 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %indvars.iv5, i64 %indvars.iv
  store float %conv4, float* %arrayidx6, align 4
  %2 = mul nuw nsw i64 %indvars.iv, %indvars.iv5
  %3 = trunc i64 %2 to i32
  %rem8 = srem i32 %3, 1024
  %add9 = add nsw i32 %rem8, 1
  %conv10 = sitofp i32 %add9 to double
  %div11 = fmul double %conv10, 5.000000e-01
  %conv12 = fptrunc double %div11 to float
  %arrayidx16 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %indvars.iv5, i64 %indvars.iv
  store float %conv12, float* %arrayidx16, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1536
  br i1 %exitcond, label %for.body3, label %for.inc17

for.inc17:                                        ; preds = %for.body3
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 1
  %exitcond7 = icmp ne i64 %indvars.iv.next6, 1536
  br i1 %exitcond7, label %for.cond1.preheader, label %for.end19

for.end19:                                        ; preds = %for.inc17
  ret void
}

; Function Attrs: nounwind uwtable
define void @print_array() #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry.split, %for.end
  %indvars.iv6 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next7, %for.end ]
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.inc
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.inc ]
  %1 = phi %struct._IO_FILE* [ %0, %for.cond1.preheader ], [ %5, %for.inc ]
  %arrayidx5 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %indvars.iv6, i64 %indvars.iv
  %2 = load float, float* %arrayidx5, align 4
  %conv = fpext float %2 to double
  %call = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %1, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), double %conv) #2
  %3 = trunc i64 %indvars.iv to i32
  %rem = srem i32 %3, 80
  %cmp6 = icmp eq i32 %rem, 79
  br i1 %cmp6, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body3
  %4 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  %fputc3 = tail call i32 @fputc(i32 10, %struct._IO_FILE* %4)
  br label %for.inc

for.inc:                                          ; preds = %for.body3, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  %exitcond = icmp ne i64 %indvars.iv.next, 1536
  br i1 %exitcond, label %for.body3, label %for.end

for.end:                                          ; preds = %for.inc
  %.lcssa = phi %struct._IO_FILE* [ %5, %for.inc ]
  %fputc = tail call i32 @fputc(i32 10, %struct._IO_FILE* %.lcssa)
  %indvars.iv.next7 = add nuw nsw i64 %indvars.iv6, 1
  %exitcond8 = icmp ne i64 %indvars.iv.next7, 1536
  br i1 %exitcond8, label %for.cond1.preheader, label %for.end12

for.end12:                                        ; preds = %for.end
  ret void
}

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...) #1

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @init_array()
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry.split, %for.inc28
  %indvars.iv7 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next8, %for.inc28 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.inc25
  %indvars.iv4 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next5, %for.inc25 ]
  %arrayidx5 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %indvars.iv7, i64 %indvars.iv4
  store float 0.000000e+00, float* %arrayidx5, align 4
  br label %for.body8

for.body8:                                        ; preds = %for.body3, %for.body8
  %indvars.iv = phi i64 [ 0, %for.body3 ], [ %indvars.iv.next, %for.body8 ]
  %arrayidx12 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %indvars.iv7, i64 %indvars.iv4
  %0 = load float, float* %arrayidx12, align 4
  %arrayidx16 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %indvars.iv7, i64 %indvars.iv
  %1 = load float, float* %arrayidx16, align 4
  %arrayidx20 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %indvars.iv, i64 %indvars.iv4
  %2 = load float, float* %arrayidx20, align 4
  %mul = fmul float %1, %2
  %add = fadd float %0, %mul
  %arrayidx24 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %indvars.iv7, i64 %indvars.iv4
  store float %add, float* %arrayidx24, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1536
  br i1 %exitcond, label %for.body8, label %for.inc25

for.inc25:                                        ; preds = %for.body8
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1
  %exitcond6 = icmp ne i64 %indvars.iv.next5, 1536
  br i1 %exitcond6, label %for.body3, label %for.inc28

for.inc28:                                        ; preds = %for.inc25
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1
  %exitcond9 = icmp ne i64 %indvars.iv.next8, 1536
  br i1 %exitcond9, label %for.cond1.preheader, label %for.end30

for.end30:                                        ; preds = %for.inc28
  ret i32 0
}

; Function Attrs: nounwind
declare i64 @fwrite(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) #2

; Function Attrs: nounwind
declare i32 @fputc(i32, %struct._IO_FILE* nocapture) #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (http://llvm.org/git/clang.git 081569d9a29c7bc827b2d41f8e62891bbc895e2f) (http://llvm.org/git/llvm.git e117e506536626352e8e47f6c72cd6e2a276622c)"}
