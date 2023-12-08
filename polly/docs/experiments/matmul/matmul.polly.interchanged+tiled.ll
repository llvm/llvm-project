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

polly.loop_header8:                               ; preds = %entry, %polly.loop_exit16
  %indvars.iv4 = phi i64 [ 64, %entry ], [ %indvars.iv.next5, %polly.loop_exit16 ]
  %polly.indvar11 = phi i64 [ 0, %entry ], [ %polly.indvar_next12, %polly.loop_exit16 ]
  br label %polly.loop_header14

polly.loop_exit16:                                ; preds = %polly.loop_exit22
  %polly.indvar_next12 = add nuw nsw i64 %polly.indvar11, 64
  %polly.loop_cond13 = icmp ult i64 %polly.indvar_next12, 1536
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 64
  br i1 %polly.loop_cond13, label %polly.loop_header8, label %polly.exiting

polly.loop_header14:                              ; preds = %polly.loop_header8, %polly.loop_exit22
  %polly.indvar17 = phi i64 [ 0, %polly.loop_header8 ], [ %polly.indvar_next18, %polly.loop_exit22 ]
  %offset.idx.1 = or i64 %polly.indvar17, 4
  %offset.idx.2 = or i64 %polly.indvar17, 8
  %offset.idx.3 = or i64 %polly.indvar17, 12
  %offset.idx.4 = or i64 %polly.indvar17, 16
  %offset.idx.5 = or i64 %polly.indvar17, 20
  %offset.idx.6 = or i64 %polly.indvar17, 24
  %offset.idx.7 = or i64 %polly.indvar17, 28
  %offset.idx.8 = or i64 %polly.indvar17, 32
  %offset.idx.9 = or i64 %polly.indvar17, 36
  %offset.idx.10 = or i64 %polly.indvar17, 40
  %offset.idx.11 = or i64 %polly.indvar17, 44
  %offset.idx.12 = or i64 %polly.indvar17, 48
  %offset.idx.13 = or i64 %polly.indvar17, 52
  %offset.idx.14 = or i64 %polly.indvar17, 56
  %offset.idx.15 = or i64 %polly.indvar17, 60
  br label %polly.loop_header20

polly.loop_exit22:                                ; preds = %polly.loop_exit28
  %polly.indvar_next18 = add nuw nsw i64 %polly.indvar17, 64
  %polly.loop_cond19 = icmp ult i64 %polly.indvar_next18, 1536
  br i1 %polly.loop_cond19, label %polly.loop_header14, label %polly.loop_exit16

polly.loop_header20:                              ; preds = %polly.loop_header14, %polly.loop_exit28
  %indvars.iv1 = phi i64 [ 64, %polly.loop_header14 ], [ %indvars.iv.next2, %polly.loop_exit28 ]
  %polly.indvar23 = phi i64 [ 0, %polly.loop_header14 ], [ %polly.indvar_next24, %polly.loop_exit28 ]
  br label %polly.loop_header26

polly.loop_exit28:                                ; preds = %polly.loop_exit34
  %polly.indvar_next24 = add nuw nsw i64 %polly.indvar23, 64
  %polly.loop_cond25 = icmp ult i64 %polly.indvar_next24, 1536
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 64
  br i1 %polly.loop_cond25, label %polly.loop_header20, label %polly.loop_exit22

polly.loop_header26:                              ; preds = %polly.loop_exit34, %polly.loop_header20
  %polly.indvar29 = phi i64 [ %polly.indvar11, %polly.loop_header20 ], [ %polly.indvar_next30, %polly.loop_exit34 ]
  %0 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %polly.indvar17
  %1 = bitcast float* %0 to <4 x float>*
  %2 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.1
  %3 = bitcast float* %2 to <4 x float>*
  %4 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.2
  %5 = bitcast float* %4 to <4 x float>*
  %6 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.3
  %7 = bitcast float* %6 to <4 x float>*
  %8 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.4
  %9 = bitcast float* %8 to <4 x float>*
  %10 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.5
  %11 = bitcast float* %10 to <4 x float>*
  %12 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.6
  %13 = bitcast float* %12 to <4 x float>*
  %14 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.7
  %15 = bitcast float* %14 to <4 x float>*
  %16 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.8
  %17 = bitcast float* %16 to <4 x float>*
  %18 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.9
  %19 = bitcast float* %18 to <4 x float>*
  %20 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.10
  %21 = bitcast float* %20 to <4 x float>*
  %22 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.11
  %23 = bitcast float* %22 to <4 x float>*
  %24 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.12
  %25 = bitcast float* %24 to <4 x float>*
  %26 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.13
  %27 = bitcast float* %26 to <4 x float>*
  %28 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.14
  %29 = bitcast float* %28 to <4 x float>*
  %30 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.15
  %31 = bitcast float* %30 to <4 x float>*
  %.promoted = load <4 x float>, <4 x float>* %1, align 16, !alias.scope !7, !noalias !9
  %.promoted14 = load <4 x float>, <4 x float>* %3, align 16, !alias.scope !7, !noalias !9
  %.promoted17 = load <4 x float>, <4 x float>* %5, align 16, !alias.scope !7, !noalias !9
  %.promoted20 = load <4 x float>, <4 x float>* %7, align 16, !alias.scope !7, !noalias !9
  %.promoted23 = load <4 x float>, <4 x float>* %9, align 16, !alias.scope !7, !noalias !9
  %.promoted26 = load <4 x float>, <4 x float>* %11, align 16, !alias.scope !7, !noalias !9
  %.promoted29 = load <4 x float>, <4 x float>* %13, align 16, !alias.scope !7, !noalias !9
  %.promoted32 = load <4 x float>, <4 x float>* %15, align 16, !alias.scope !7, !noalias !9
  %.promoted35 = load <4 x float>, <4 x float>* %17, align 16, !alias.scope !7, !noalias !9
  %.promoted38 = load <4 x float>, <4 x float>* %19, align 16, !alias.scope !7, !noalias !9
  %.promoted41 = load <4 x float>, <4 x float>* %21, align 16, !alias.scope !7, !noalias !9
  %.promoted44 = load <4 x float>, <4 x float>* %23, align 16, !alias.scope !7, !noalias !9
  %.promoted47 = load <4 x float>, <4 x float>* %25, align 16, !alias.scope !7, !noalias !9
  %.promoted50 = load <4 x float>, <4 x float>* %27, align 16, !alias.scope !7, !noalias !9
  %.promoted53 = load <4 x float>, <4 x float>* %29, align 16, !alias.scope !7, !noalias !9
  %.promoted56 = load <4 x float>, <4 x float>* %31, align 16, !alias.scope !7, !noalias !9
  br label %vector.ph

polly.loop_exit34:                                ; preds = %vector.ph
  store <4 x float> %35, <4 x float>* %1, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %39, <4 x float>* %3, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %43, <4 x float>* %5, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %47, <4 x float>* %7, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %51, <4 x float>* %9, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %55, <4 x float>* %11, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %59, <4 x float>* %13, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %63, <4 x float>* %15, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %67, <4 x float>* %17, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %71, <4 x float>* %19, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %75, <4 x float>* %21, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %79, <4 x float>* %23, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %83, <4 x float>* %25, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %87, <4 x float>* %27, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %91, <4 x float>* %29, align 16, !alias.scope !7, !noalias !9
  store <4 x float> %95, <4 x float>* %31, align 16, !alias.scope !7, !noalias !9
  %polly.indvar_next30 = add nuw nsw i64 %polly.indvar29, 1
  %exitcond6 = icmp eq i64 %polly.indvar_next30, %indvars.iv4
  br i1 %exitcond6, label %polly.loop_exit28, label %polly.loop_header26

vector.ph:                                        ; preds = %polly.loop_header26, %vector.ph
  %wide.load.1557 = phi <4 x float> [ %.promoted56, %polly.loop_header26 ], [ %95, %vector.ph ]
  %wide.load.1454 = phi <4 x float> [ %.promoted53, %polly.loop_header26 ], [ %91, %vector.ph ]
  %wide.load.1351 = phi <4 x float> [ %.promoted50, %polly.loop_header26 ], [ %87, %vector.ph ]
  %wide.load.1248 = phi <4 x float> [ %.promoted47, %polly.loop_header26 ], [ %83, %vector.ph ]
  %wide.load.1145 = phi <4 x float> [ %.promoted44, %polly.loop_header26 ], [ %79, %vector.ph ]
  %wide.load.1042 = phi <4 x float> [ %.promoted41, %polly.loop_header26 ], [ %75, %vector.ph ]
  %wide.load.939 = phi <4 x float> [ %.promoted38, %polly.loop_header26 ], [ %71, %vector.ph ]
  %wide.load.836 = phi <4 x float> [ %.promoted35, %polly.loop_header26 ], [ %67, %vector.ph ]
  %wide.load.733 = phi <4 x float> [ %.promoted32, %polly.loop_header26 ], [ %63, %vector.ph ]
  %wide.load.630 = phi <4 x float> [ %.promoted29, %polly.loop_header26 ], [ %59, %vector.ph ]
  %wide.load.527 = phi <4 x float> [ %.promoted26, %polly.loop_header26 ], [ %55, %vector.ph ]
  %wide.load.424 = phi <4 x float> [ %.promoted23, %polly.loop_header26 ], [ %51, %vector.ph ]
  %wide.load.321 = phi <4 x float> [ %.promoted20, %polly.loop_header26 ], [ %47, %vector.ph ]
  %wide.load.218 = phi <4 x float> [ %.promoted17, %polly.loop_header26 ], [ %43, %vector.ph ]
  %wide.load.115 = phi <4 x float> [ %.promoted14, %polly.loop_header26 ], [ %39, %vector.ph ]
  %wide.load13 = phi <4 x float> [ %.promoted, %polly.loop_header26 ], [ %35, %vector.ph ]
  %polly.indvar35 = phi i64 [ %polly.indvar23, %polly.loop_header26 ], [ %polly.indvar_next36, %vector.ph ]
  %scevgep47 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %polly.indvar29, i64 %polly.indvar35
  %_p_scalar_48 = load float, float* %scevgep47, align 4, !alias.scope !10, !noalias !12
  %broadcast.splatinsert11 = insertelement <4 x float> undef, float %_p_scalar_48, i32 0
  %broadcast.splat12 = shufflevector <4 x float> %broadcast.splatinsert11, <4 x float> undef, <4 x i32> zeroinitializer
  %32 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %polly.indvar17
  %33 = bitcast float* %32 to <4 x float>*
  %wide.load10 = load <4 x float>, <4 x float>* %33, align 16, !alias.scope !11, !noalias !13
  %34 = fmul <4 x float> %broadcast.splat12, %wide.load10
  %35 = fadd <4 x float> %wide.load13, %34
  %36 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.1
  %37 = bitcast float* %36 to <4 x float>*
  %wide.load10.1 = load <4 x float>, <4 x float>* %37, align 16, !alias.scope !11, !noalias !13
  %38 = fmul <4 x float> %broadcast.splat12, %wide.load10.1
  %39 = fadd <4 x float> %wide.load.115, %38
  %40 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.2
  %41 = bitcast float* %40 to <4 x float>*
  %wide.load10.2 = load <4 x float>, <4 x float>* %41, align 16, !alias.scope !11, !noalias !13
  %42 = fmul <4 x float> %broadcast.splat12, %wide.load10.2
  %43 = fadd <4 x float> %wide.load.218, %42
  %44 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.3
  %45 = bitcast float* %44 to <4 x float>*
  %wide.load10.3 = load <4 x float>, <4 x float>* %45, align 16, !alias.scope !11, !noalias !13
  %46 = fmul <4 x float> %broadcast.splat12, %wide.load10.3
  %47 = fadd <4 x float> %wide.load.321, %46
  %48 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.4
  %49 = bitcast float* %48 to <4 x float>*
  %wide.load10.4 = load <4 x float>, <4 x float>* %49, align 16, !alias.scope !11, !noalias !13
  %50 = fmul <4 x float> %broadcast.splat12, %wide.load10.4
  %51 = fadd <4 x float> %wide.load.424, %50
  %52 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.5
  %53 = bitcast float* %52 to <4 x float>*
  %wide.load10.5 = load <4 x float>, <4 x float>* %53, align 16, !alias.scope !11, !noalias !13
  %54 = fmul <4 x float> %broadcast.splat12, %wide.load10.5
  %55 = fadd <4 x float> %wide.load.527, %54
  %56 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.6
  %57 = bitcast float* %56 to <4 x float>*
  %wide.load10.6 = load <4 x float>, <4 x float>* %57, align 16, !alias.scope !11, !noalias !13
  %58 = fmul <4 x float> %broadcast.splat12, %wide.load10.6
  %59 = fadd <4 x float> %wide.load.630, %58
  %60 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.7
  %61 = bitcast float* %60 to <4 x float>*
  %wide.load10.7 = load <4 x float>, <4 x float>* %61, align 16, !alias.scope !11, !noalias !13
  %62 = fmul <4 x float> %broadcast.splat12, %wide.load10.7
  %63 = fadd <4 x float> %wide.load.733, %62
  %64 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.8
  %65 = bitcast float* %64 to <4 x float>*
  %wide.load10.8 = load <4 x float>, <4 x float>* %65, align 16, !alias.scope !11, !noalias !13
  %66 = fmul <4 x float> %broadcast.splat12, %wide.load10.8
  %67 = fadd <4 x float> %wide.load.836, %66
  %68 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.9
  %69 = bitcast float* %68 to <4 x float>*
  %wide.load10.9 = load <4 x float>, <4 x float>* %69, align 16, !alias.scope !11, !noalias !13
  %70 = fmul <4 x float> %broadcast.splat12, %wide.load10.9
  %71 = fadd <4 x float> %wide.load.939, %70
  %72 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.10
  %73 = bitcast float* %72 to <4 x float>*
  %wide.load10.10 = load <4 x float>, <4 x float>* %73, align 16, !alias.scope !11, !noalias !13
  %74 = fmul <4 x float> %broadcast.splat12, %wide.load10.10
  %75 = fadd <4 x float> %wide.load.1042, %74
  %76 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.11
  %77 = bitcast float* %76 to <4 x float>*
  %wide.load10.11 = load <4 x float>, <4 x float>* %77, align 16, !alias.scope !11, !noalias !13
  %78 = fmul <4 x float> %broadcast.splat12, %wide.load10.11
  %79 = fadd <4 x float> %wide.load.1145, %78
  %80 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.12
  %81 = bitcast float* %80 to <4 x float>*
  %wide.load10.12 = load <4 x float>, <4 x float>* %81, align 16, !alias.scope !11, !noalias !13
  %82 = fmul <4 x float> %broadcast.splat12, %wide.load10.12
  %83 = fadd <4 x float> %wide.load.1248, %82
  %84 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.13
  %85 = bitcast float* %84 to <4 x float>*
  %wide.load10.13 = load <4 x float>, <4 x float>* %85, align 16, !alias.scope !11, !noalias !13
  %86 = fmul <4 x float> %broadcast.splat12, %wide.load10.13
  %87 = fadd <4 x float> %wide.load.1351, %86
  %88 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.14
  %89 = bitcast float* %88 to <4 x float>*
  %wide.load10.14 = load <4 x float>, <4 x float>* %89, align 16, !alias.scope !11, !noalias !13
  %90 = fmul <4 x float> %broadcast.splat12, %wide.load10.14
  %91 = fadd <4 x float> %wide.load.1454, %90
  %92 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.15
  %93 = bitcast float* %92 to <4 x float>*
  %wide.load10.15 = load <4 x float>, <4 x float>* %93, align 16, !alias.scope !11, !noalias !13
  %94 = fmul <4 x float> %broadcast.splat12, %wide.load10.15
  %95 = fadd <4 x float> %wide.load.1557, %94
  %polly.indvar_next36 = add nuw nsw i64 %polly.indvar35, 1
  %exitcond3 = icmp eq i64 %polly.indvar_next36, %indvars.iv1
  br i1 %exitcond3, label %polly.loop_exit34, label %vector.ph
}

; Function Attrs: nounwind
declare i32 @fputc(i32, %struct._IO_FILE* nocapture) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #5

attributes #0 = { noinline norecurse nounwind uwtable writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "polly-optimized" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "polly-optimized" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
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
!7 = distinct !{!7, !8, !"polly.alias.scope.MemRef_C"}
!8 = distinct !{!8, !"polly.alias.scope.domain"}
!9 = !{!10, !11}
!10 = distinct !{!10, !8, !"polly.alias.scope.MemRef_A"}
!11 = distinct !{!11, !8, !"polly.alias.scope.MemRef_B"}
!12 = !{!7, !11}
!13 = !{!7, !10}
