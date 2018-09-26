; ModuleID = '<stdin>'
source_filename = "matmul.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common dso_local local_unnamed_addr global [1536 x [1536 x float]] zeroinitializer, align 16
@B = common dso_local local_unnamed_addr global [1536 x [1536 x float]] zeroinitializer, align 16
@stdout = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [5 x i8] c"%lf \00", align 1
@C = common dso_local local_unnamed_addr global [1536 x [1536 x float]] zeroinitializer, align 16

; Function Attrs: noinline norecurse nounwind uwtable writeonly
define dso_local void @init_array() local_unnamed_addr #0 {
entry:
  br label %polly.loop_header

polly.exiting:                                    ; preds = %polly.loop_exit3
  ret void

polly.loop_header:                                ; preds = %polly.loop_exit3, %entry
  %polly.indvar = phi i64 [ 0, %entry ], [ %polly.indvar_next, %polly.loop_exit3 ]
  %0 = trunc i64 %polly.indvar to i32
  br label %polly.loop_header1

polly.loop_exit3:                                 ; preds = %polly.loop_header1
  %polly.indvar_next = add nuw nsw i64 %polly.indvar, 1
  %exitcond1 = icmp eq i64 %polly.indvar_next, 1536
  br i1 %exitcond1, label %polly.exiting, label %polly.loop_header

polly.loop_header1:                               ; preds = %polly.loop_header1, %polly.loop_header
  %polly.indvar4 = phi i64 [ 0, %polly.loop_header ], [ %polly.indvar_next5.1, %polly.loop_header1 ]
  %1 = trunc i64 %polly.indvar4 to i32
  %2 = mul nuw nsw i32 %1, %0
  %3 = and i32 %2, 1022
  %4 = or i32 %3, 1
  %p_conv = sitofp i32 %4 to double
  %p_div = fmul double %p_conv, 5.000000e-01
  %p_conv4 = fptrunc double %p_div to float
  %scevgep7 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %polly.indvar, i64 %polly.indvar4
  store float %p_conv4, float* %scevgep7, align 8, !alias.scope !2, !noalias !4
  %scevgep9 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar, i64 %polly.indvar4
  store float %p_conv4, float* %scevgep9, align 8, !alias.scope !5, !noalias !6
  %polly.indvar_next5 = or i64 %polly.indvar4, 1
  %5 = trunc i64 %polly.indvar_next5 to i32
  %6 = mul nuw nsw i32 %5, %0
  %7 = and i32 %6, 1023
  %8 = add nuw nsw i32 %7, 1
  %p_conv.1 = sitofp i32 %8 to double
  %p_div.1 = fmul double %p_conv.1, 5.000000e-01
  %p_conv4.1 = fptrunc double %p_div.1 to float
  %scevgep7.1 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %polly.indvar, i64 %polly.indvar_next5
  store float %p_conv4.1, float* %scevgep7.1, align 4, !alias.scope !2, !noalias !4
  %scevgep9.1 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar, i64 %polly.indvar_next5
  store float %p_conv4.1, float* %scevgep9.1, align 4, !alias.scope !5, !noalias !6
  %polly.indvar_next5.1 = add nuw nsw i64 %polly.indvar4, 2
  %exitcond.1 = icmp eq i64 %polly.indvar_next5.1, 1536
  br i1 %exitcond.1, label %polly.loop_exit3, label %polly.loop_header1
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @print_array() local_unnamed_addr #1 {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.end, %entry
  %indvars.iv6 = phi i64 [ 0, %entry ], [ %indvars.iv.next7, %for.end ]
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  br label %for.body3

for.body3:                                        ; preds = %for.inc, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.inc ]
  %1 = phi %struct._IO_FILE* [ %0, %for.cond1.preheader ], [ %5, %for.inc ]
  %arrayidx5 = getelementptr inbounds [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %indvars.iv6, i64 %indvars.iv
  %2 = load float, float* %arrayidx5, align 4
  %conv = fpext float %2 to double
  %call = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %1, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), double %conv) #4
  %3 = trunc i64 %indvars.iv to i32
  %rem = urem i32 %3, 80
  %cmp6 = icmp eq i32 %rem, 79
  br i1 %cmp6, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body3
  %4 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  %fputc3 = tail call i32 @fputc(i32 10, %struct._IO_FILE* %4)
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  %exitcond = icmp eq i64 %indvars.iv.next, 1536
  br i1 %exitcond, label %for.end, label %for.body3

for.end:                                          ; preds = %for.inc
  %fputc = tail call i32 @fputc(i32 10, %struct._IO_FILE* %5)
  %indvars.iv.next7 = add nuw nsw i64 %indvars.iv6, 1
  %exitcond8 = icmp eq i64 %indvars.iv.next7, 1536
  br i1 %exitcond8, label %for.end12, label %for.cond1.preheader

for.end12:                                        ; preds = %for.end
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #3 {
entry:
  tail call void @init_array()
  call void @llvm.memset.p0i8.i64(i8* align 16 bitcast ([1536 x [1536 x float]]* @C to i8*), i8 0, i64 9437184, i1 false)
  br label %polly.loop_header8

polly.exiting:                                    ; preds = %polly.loop_exit16
  ret i32 0

polly.loop_header8:                               ; preds = %polly.loop_exit16, %entry
  %polly.indvar11 = phi i64 [ %polly.indvar_next12, %polly.loop_exit16 ], [ 0, %entry ]
  br label %polly.loop_header14

polly.loop_exit16:                                ; preds = %polly.loop_exit22
  %polly.indvar_next12 = add nuw nsw i64 %polly.indvar11, 1
  %exitcond2 = icmp eq i64 %polly.indvar_next12, 1536
  br i1 %exitcond2, label %polly.exiting, label %polly.loop_header8

polly.loop_header14:                              ; preds = %polly.loop_exit22, %polly.loop_header8
  %polly.indvar17 = phi i64 [ 0, %polly.loop_header8 ], [ %polly.indvar_next18, %polly.loop_exit22 ]
  %scevgep29 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %polly.indvar11, i64 %polly.indvar17
  %_p_scalar_30 = load float, float* %scevgep29, align 4, !alias.scope !7, !noalias !9
  %broadcast.splatinsert10 = insertelement <4 x float> undef, float %_p_scalar_30, i32 0
  %broadcast.splat11 = shufflevector <4 x float> %broadcast.splatinsert10, <4 x float> undef, <4 x i32> zeroinitializer
  %broadcast.splatinsert12 = insertelement <4 x float> undef, float %_p_scalar_30, i32 0
  %broadcast.splat13 = shufflevector <4 x float> %broadcast.splatinsert12, <4 x float> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %polly.loop_header14
  %index = phi i64 [ 0, %polly.loop_header14 ], [ %index.next.1, %vector.body ]
  %0 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar11, i64 %index
  %1 = bitcast float* %0 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %1, align 16, !alias.scope !10, !noalias !12
  %2 = getelementptr float, float* %0, i64 4
  %3 = bitcast float* %2 to <4 x float>*
  %wide.load7 = load <4 x float>, <4 x float>* %3, align 16, !alias.scope !10, !noalias !12
  %4 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar17, i64 %index
  %5 = bitcast float* %4 to <4 x float>*
  %wide.load8 = load <4 x float>, <4 x float>* %5, align 16, !alias.scope !11, !noalias !13
  %6 = getelementptr float, float* %4, i64 4
  %7 = bitcast float* %6 to <4 x float>*
  %wide.load9 = load <4 x float>, <4 x float>* %7, align 16, !alias.scope !11, !noalias !13
  %8 = fmul <4 x float> %broadcast.splat11, %wide.load8
  %9 = fmul <4 x float> %broadcast.splat13, %wide.load9
  %10 = fadd <4 x float> %wide.load, %8
  %11 = fadd <4 x float> %wide.load7, %9
  %12 = bitcast float* %0 to <4 x float>*
  store <4 x float> %10, <4 x float>* %12, align 16, !alias.scope !10, !noalias !12
  %13 = bitcast float* %2 to <4 x float>*
  store <4 x float> %11, <4 x float>* %13, align 16, !alias.scope !10, !noalias !12
  %index.next = or i64 %index, 8
  %14 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar11, i64 %index.next
  %15 = bitcast float* %14 to <4 x float>*
  %wide.load.1 = load <4 x float>, <4 x float>* %15, align 16, !alias.scope !10, !noalias !12
  %16 = getelementptr float, float* %14, i64 4
  %17 = bitcast float* %16 to <4 x float>*
  %wide.load7.1 = load <4 x float>, <4 x float>* %17, align 16, !alias.scope !10, !noalias !12
  %18 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar17, i64 %index.next
  %19 = bitcast float* %18 to <4 x float>*
  %wide.load8.1 = load <4 x float>, <4 x float>* %19, align 16, !alias.scope !11, !noalias !13
  %20 = getelementptr float, float* %18, i64 4
  %21 = bitcast float* %20 to <4 x float>*
  %wide.load9.1 = load <4 x float>, <4 x float>* %21, align 16, !alias.scope !11, !noalias !13
  %22 = fmul <4 x float> %broadcast.splat11, %wide.load8.1
  %23 = fmul <4 x float> %broadcast.splat13, %wide.load9.1
  %24 = fadd <4 x float> %wide.load.1, %22
  %25 = fadd <4 x float> %wide.load7.1, %23
  %26 = bitcast float* %14 to <4 x float>*
  store <4 x float> %24, <4 x float>* %26, align 16, !alias.scope !10, !noalias !12
  %27 = bitcast float* %16 to <4 x float>*
  store <4 x float> %25, <4 x float>* %27, align 16, !alias.scope !10, !noalias !12
  %index.next.1 = add nuw nsw i64 %index, 16
  %28 = icmp eq i64 %index.next.1, 1536
  br i1 %28, label %polly.loop_exit22, label %vector.body, !llvm.loop !14

polly.loop_exit22:                                ; preds = %vector.body
  %polly.indvar_next18 = add nuw nsw i64 %polly.indvar17, 1
  %exitcond1 = icmp eq i64 %polly.indvar_next18, 1536
  br i1 %exitcond1, label %polly.loop_exit16, label %polly.loop_header14
}

; Function Attrs: nounwind
declare i32 @fputc(i32, %struct._IO_FILE* nocapture) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #5

attributes #0 = { noinline norecurse nounwind uwtable writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "polly-optimized" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "polly-optimized" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.0 (trunk 342834) (llvm/trunk 342856)"}
!2 = distinct !{!2, !3, !"polly.alias.scope.MemRef_A"}
!3 = distinct !{!3, !"polly.alias.scope.domain"}
!4 = !{!5}
!5 = distinct !{!5, !3, !"polly.alias.scope.MemRef_B"}
!6 = !{!2}
!7 = distinct !{!7, !8, !"polly.alias.scope.MemRef_A"}
!8 = distinct !{!8, !"polly.alias.scope.domain"}
!9 = !{!10, !11}
!10 = distinct !{!10, !8, !"polly.alias.scope.MemRef_C"}
!11 = distinct !{!11, !8, !"polly.alias.scope.MemRef_B"}
!12 = !{!7, !11}
!13 = !{!10, !7}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.isvectorized", i32 1}
