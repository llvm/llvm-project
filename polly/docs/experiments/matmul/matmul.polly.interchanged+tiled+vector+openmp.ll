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

; Function Attrs: noinline nounwind uwtable
define dso_local void @init_array() local_unnamed_addr #0 {
entry:
  %polly.par.userContext = alloca {}, align 8
  %polly.par.userContext1 = bitcast {}* %polly.par.userContext to i8*
  call void @GOMP_parallel_loop_runtime_start(void (i8*)* nonnull @init_array_polly_subfn, i8* nonnull %polly.par.userContext1, i32 0, i64 0, i64 1536, i64 1) #3
  call void @init_array_polly_subfn(i8* nonnull %polly.par.userContext1) #3
  call void @GOMP_parallel_end() #3
  ret void
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
  %call = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %1, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), double %conv) #3
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

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 {
entry:
  %polly.par.userContext3 = alloca {}, align 8
  tail call void @init_array()
  %polly.par.userContext1 = bitcast {}* %polly.par.userContext3 to i8*
  call void @GOMP_parallel_loop_runtime_start(void (i8*)* nonnull @main_polly_subfn, i8* nonnull %polly.par.userContext1, i32 0, i64 0, i64 1536, i64 1) #3
  call void @main_polly_subfn(i8* nonnull %polly.par.userContext1) #3
  call void @GOMP_parallel_end() #3
  call void @GOMP_parallel_loop_runtime_start(void (i8*)* nonnull @main_polly_subfn_1, i8* nonnull %polly.par.userContext1, i32 0, i64 0, i64 1536, i64 64) #3
  call void @main_polly_subfn_1(i8* nonnull %polly.par.userContext1) #3
  call void @GOMP_parallel_end() #3
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @fputc(i32, %struct._IO_FILE* nocapture) local_unnamed_addr #3

define internal void @init_array_polly_subfn(i8* nocapture readnone %polly.par.userContext) #4 {
polly.par.setup:
  %polly.par.LBPtr = alloca i64, align 8
  %polly.par.UBPtr = alloca i64, align 8
  %0 = call i8 @GOMP_loop_runtime_next(i64* nonnull %polly.par.LBPtr, i64* nonnull %polly.par.UBPtr)
  %1 = icmp eq i8 %0, 0
  br i1 %1, label %polly.par.exit, label %polly.par.loadIVBounds

polly.par.exit:                                   ; preds = %polly.par.checkNext.loopexit, %polly.par.setup
  call void @GOMP_loop_end_nowait()
  ret void

polly.par.checkNext.loopexit:                     ; preds = %polly.loop_exit4
  %2 = call i8 @GOMP_loop_runtime_next(i64* nonnull %polly.par.LBPtr, i64* nonnull %polly.par.UBPtr)
  %3 = icmp eq i8 %2, 0
  br i1 %3, label %polly.par.exit, label %polly.par.loadIVBounds

polly.par.loadIVBounds:                           ; preds = %polly.par.setup, %polly.par.checkNext.loopexit
  %polly.par.LB = load i64, i64* %polly.par.LBPtr, align 8
  %polly.par.UB = load i64, i64* %polly.par.UBPtr, align 8
  %polly.par.UBAdjusted = add i64 %polly.par.UB, -1
  br label %polly.loop_header

polly.loop_header:                                ; preds = %polly.par.loadIVBounds, %polly.loop_exit4
  %polly.indvar = phi i64 [ %polly.par.LB, %polly.par.loadIVBounds ], [ %polly.indvar_next, %polly.loop_exit4 ]
  %4 = trunc i64 %polly.indvar to i32
  br label %polly.loop_header2

polly.loop_exit4:                                 ; preds = %polly.loop_header2
  %polly.indvar_next = add nsw i64 %polly.indvar, 1
  %polly.loop_cond = icmp slt i64 %polly.indvar, %polly.par.UBAdjusted
  br i1 %polly.loop_cond, label %polly.loop_header, label %polly.par.checkNext.loopexit

polly.loop_header2:                               ; preds = %polly.loop_header2, %polly.loop_header
  %polly.indvar5 = phi i64 [ 0, %polly.loop_header ], [ %polly.indvar_next6, %polly.loop_header2 ]
  %5 = trunc i64 %polly.indvar5 to i32
  %6 = mul i32 %5, %4
  %7 = and i32 %6, 1023
  %8 = add nuw nsw i32 %7, 1
  %p_conv = sitofp i32 %8 to double
  %p_div = fmul double %p_conv, 5.000000e-01
  %p_conv4 = fptrunc double %p_div to float
  %scevgep8 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %polly.indvar, i64 %polly.indvar5
  store float %p_conv4, float* %scevgep8, align 4, !alias.scope !2, !noalias !4
  %scevgep10 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar, i64 %polly.indvar5
  store float %p_conv4, float* %scevgep10, align 4, !alias.scope !5, !noalias !6
  %polly.indvar_next6 = add nuw nsw i64 %polly.indvar5, 1
  %exitcond = icmp eq i64 %polly.indvar_next6, 1536
  br i1 %exitcond, label %polly.loop_exit4, label %polly.loop_header2
}

declare i8 @GOMP_loop_runtime_next(i64*, i64*) local_unnamed_addr

declare void @GOMP_loop_end_nowait() local_unnamed_addr

declare void @GOMP_parallel_loop_runtime_start(void (i8*)*, i8*, i32, i64, i64, i64) local_unnamed_addr

declare void @GOMP_parallel_end() local_unnamed_addr

define internal void @main_polly_subfn(i8* nocapture readnone %polly.par.userContext) #4 {
polly.par.setup:
  %polly.par.LBPtr = alloca i64, align 8
  %polly.par.UBPtr = alloca i64, align 8
  %0 = call i8 @GOMP_loop_runtime_next(i64* nonnull %polly.par.LBPtr, i64* nonnull %polly.par.UBPtr)
  %1 = icmp eq i8 %0, 0
  br i1 %1, label %polly.par.exit, label %polly.par.loadIVBounds

polly.par.exit:                                   ; preds = %polly.par.loadIVBounds, %polly.par.setup
  call void @GOMP_loop_end_nowait()
  ret void

polly.par.loadIVBounds:                           ; preds = %polly.par.setup, %polly.par.loadIVBounds
  %polly.par.LB = load i64, i64* %polly.par.LBPtr, align 8
  %polly.par.UB = load i64, i64* %polly.par.UBPtr, align 8
  %polly.par.UBAdjusted = add i64 %polly.par.UB, -1
  %scevgep2 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.par.LB, i64 0
  %scevgep23 = bitcast float* %scevgep2 to i8*
  %2 = icmp sgt i64 %polly.par.LB, %polly.par.UBAdjusted
  %smax = select i1 %2, i64 %polly.par.LB, i64 %polly.par.UBAdjusted
  %3 = add i64 %smax, 1
  %4 = sub i64 %3, %polly.par.LB
  %5 = mul i64 %4, 6144
  call void @llvm.memset.p0i8.i64(i8* align 16 %scevgep23, i8 0, i64 %5, i1 false)
  %6 = call i8 @GOMP_loop_runtime_next(i64* nonnull %polly.par.LBPtr, i64* nonnull %polly.par.UBPtr)
  %7 = icmp eq i8 %6, 0
  br i1 %7, label %polly.par.exit, label %polly.par.loadIVBounds
}

define internal void @main_polly_subfn_1(i8* nocapture readnone %polly.par.userContext) #4 {
polly.par.setup:
  %polly.par.LBPtr = alloca i64, align 8
  %polly.par.UBPtr = alloca i64, align 8
  %0 = call i8 @GOMP_loop_runtime_next(i64* nonnull %polly.par.LBPtr, i64* nonnull %polly.par.UBPtr)
  %1 = icmp eq i8 %0, 0
  br i1 %1, label %polly.par.exit, label %polly.par.loadIVBounds

polly.par.exit:                                   ; preds = %polly.par.checkNext.loopexit, %polly.par.setup
  call void @GOMP_loop_end_nowait()
  ret void

polly.par.checkNext.loopexit:                     ; preds = %polly.loop_exit4
  %2 = call i8 @GOMP_loop_runtime_next(i64* nonnull %polly.par.LBPtr, i64* nonnull %polly.par.UBPtr)
  %3 = icmp eq i8 %2, 0
  br i1 %3, label %polly.par.exit, label %polly.par.loadIVBounds

polly.par.loadIVBounds:                           ; preds = %polly.par.setup, %polly.par.checkNext.loopexit
  %polly.par.LB = load i64, i64* %polly.par.LBPtr, align 8
  %polly.par.UB = load i64, i64* %polly.par.UBPtr, align 8
  %polly.par.UBAdjusted = add i64 %polly.par.UB, -1
  br label %polly.loop_header

polly.loop_header:                                ; preds = %polly.loop_exit4, %polly.par.loadIVBounds
  %polly.indvar = phi i64 [ %polly.par.LB, %polly.par.loadIVBounds ], [ %polly.indvar_next, %polly.loop_exit4 ]
  %4 = add nsw i64 %polly.indvar, 63
  br label %polly.loop_header2

polly.loop_exit4:                                 ; preds = %polly.loop_exit10
  %polly.indvar_next = add nsw i64 %polly.indvar, 64
  %polly.loop_cond = icmp sgt i64 %polly.indvar_next, %polly.par.UBAdjusted
  br i1 %polly.loop_cond, label %polly.par.checkNext.loopexit, label %polly.loop_header

polly.loop_header2:                               ; preds = %polly.loop_header, %polly.loop_exit10
  %indvar = phi i64 [ 0, %polly.loop_header ], [ %indvar.next, %polly.loop_exit10 ]
  %polly.indvar5 = phi i64 [ 0, %polly.loop_header ], [ %polly.indvar_next6, %polly.loop_exit10 ]
  %5 = shl i64 %indvar, 6
  %offset.idx.1 = or i64 %5, 16
  %offset.idx.2 = or i64 %5, 32
  %offset.idx.3 = or i64 %5, 48
  br label %polly.loop_header8

polly.loop_exit10:                                ; preds = %polly.loop_exit16
  %polly.indvar_next6 = add nuw nsw i64 %polly.indvar5, 64
  %polly.loop_cond7 = icmp ult i64 %polly.indvar_next6, 1536
  %indvar.next = add i64 %indvar, 1
  br i1 %polly.loop_cond7, label %polly.loop_header2, label %polly.loop_exit4

polly.loop_header8:                               ; preds = %polly.loop_header2, %polly.loop_exit16
  %indvars.iv3 = phi i64 [ 64, %polly.loop_header2 ], [ %indvars.iv.next4, %polly.loop_exit16 ]
  %polly.indvar11 = phi i64 [ 0, %polly.loop_header2 ], [ %polly.indvar_next12, %polly.loop_exit16 ]
  br label %polly.loop_header14

polly.loop_exit16:                                ; preds = %polly.loop_exit22
  %polly.indvar_next12 = add nuw nsw i64 %polly.indvar11, 64
  %polly.loop_cond13 = icmp ult i64 %polly.indvar_next12, 1536
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 64
  br i1 %polly.loop_cond13, label %polly.loop_header8, label %polly.loop_exit10

polly.loop_header14:                              ; preds = %polly.loop_header8, %polly.loop_exit22
  %polly.indvar17 = phi i64 [ %polly.indvar_next18, %polly.loop_exit22 ], [ %polly.indvar, %polly.loop_header8 ]
  %6 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar17, i64 %5
  %7 = bitcast float* %6 to <16 x float>*
  %8 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar17, i64 %offset.idx.1
  %9 = bitcast float* %8 to <16 x float>*
  %10 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar17, i64 %offset.idx.2
  %11 = bitcast float* %10 to <16 x float>*
  %12 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar17, i64 %offset.idx.3
  %13 = bitcast float* %12 to <16 x float>*
  %.promoted = load <16 x float>, <16 x float>* %7, align 4, !alias.scope !7, !noalias !9
  %.promoted16 = load <16 x float>, <16 x float>* %9, align 4, !alias.scope !7, !noalias !9
  %.promoted18 = load <16 x float>, <16 x float>* %11, align 4, !alias.scope !7, !noalias !9
  %.promoted20 = load <16 x float>, <16 x float>* %13, align 4, !alias.scope !7, !noalias !9
  br label %vector.ph

polly.loop_exit22:                                ; preds = %vector.ph
  store <16 x float> %interleaved.vec, <16 x float>* %7, align 4, !alias.scope !7, !noalias !9
  store <16 x float> %interleaved.vec.1, <16 x float>* %9, align 4, !alias.scope !7, !noalias !9
  store <16 x float> %interleaved.vec.2, <16 x float>* %11, align 4, !alias.scope !7, !noalias !9
  store <16 x float> %interleaved.vec.3, <16 x float>* %13, align 4, !alias.scope !7, !noalias !9
  %polly.indvar_next18 = add nsw i64 %polly.indvar17, 1
  %polly.loop_cond19 = icmp slt i64 %polly.indvar17, %4
  br i1 %polly.loop_cond19, label %polly.loop_header14, label %polly.loop_exit16

vector.ph:                                        ; preds = %polly.loop_header14, %vector.ph
  %wide.vec.321 = phi <16 x float> [ %.promoted20, %polly.loop_header14 ], [ %interleaved.vec.3, %vector.ph ]
  %wide.vec.219 = phi <16 x float> [ %.promoted18, %polly.loop_header14 ], [ %interleaved.vec.2, %vector.ph ]
  %wide.vec.117 = phi <16 x float> [ %.promoted16, %polly.loop_header14 ], [ %interleaved.vec.1, %vector.ph ]
  %wide.vec15 = phi <16 x float> [ %.promoted, %polly.loop_header14 ], [ %interleaved.vec, %vector.ph ]
  %polly.indvar23 = phi i64 [ %polly.indvar11, %polly.loop_header14 ], [ %polly.indvar_next24, %vector.ph ]
  %scevgep40 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %polly.indvar17, i64 %polly.indvar23
  %_p_scalar_41 = load float, float* %scevgep40, align 4, !alias.scope !10, !noalias !12
  %broadcast.splatinsert13 = insertelement <4 x float> undef, float %_p_scalar_41, i32 0
  %broadcast.splat14 = shufflevector <4 x float> %broadcast.splatinsert13, <4 x float> undef, <4 x i32> zeroinitializer
  %strided.vec = shufflevector <16 x float> %wide.vec15, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec5 = shufflevector <16 x float> %wide.vec15, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec6 = shufflevector <16 x float> %wide.vec15, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec7 = shufflevector <16 x float> %wide.vec15, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %14 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar23, i64 %5
  %15 = bitcast float* %14 to <16 x float>*
  %wide.vec8 = load <16 x float>, <16 x float>* %15, align 16, !alias.scope !11, !noalias !13
  %strided.vec9 = shufflevector <16 x float> %wide.vec8, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec10 = shufflevector <16 x float> %wide.vec8, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec11 = shufflevector <16 x float> %wide.vec8, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec12 = shufflevector <16 x float> %wide.vec8, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %16 = fmul <4 x float> %broadcast.splat14, %strided.vec9
  %17 = fadd <4 x float> %strided.vec, %16
  %18 = fmul <4 x float> %broadcast.splat14, %strided.vec10
  %19 = fadd <4 x float> %strided.vec5, %18
  %20 = fmul <4 x float> %broadcast.splat14, %strided.vec11
  %21 = fadd <4 x float> %strided.vec6, %20
  %22 = fmul <4 x float> %broadcast.splat14, %strided.vec12
  %23 = fadd <4 x float> %strided.vec7, %22
  %24 = shufflevector <4 x float> %17, <4 x float> %19, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %25 = shufflevector <4 x float> %21, <4 x float> %23, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec = shufflevector <8 x float> %24, <8 x float> %25, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %strided.vec.1 = shufflevector <16 x float> %wide.vec.117, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec5.1 = shufflevector <16 x float> %wide.vec.117, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec6.1 = shufflevector <16 x float> %wide.vec.117, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec7.1 = shufflevector <16 x float> %wide.vec.117, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %26 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar23, i64 %offset.idx.1
  %27 = bitcast float* %26 to <16 x float>*
  %wide.vec8.1 = load <16 x float>, <16 x float>* %27, align 16, !alias.scope !11, !noalias !13
  %strided.vec9.1 = shufflevector <16 x float> %wide.vec8.1, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec10.1 = shufflevector <16 x float> %wide.vec8.1, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec11.1 = shufflevector <16 x float> %wide.vec8.1, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec12.1 = shufflevector <16 x float> %wide.vec8.1, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %28 = fmul <4 x float> %broadcast.splat14, %strided.vec9.1
  %29 = fadd <4 x float> %strided.vec.1, %28
  %30 = fmul <4 x float> %broadcast.splat14, %strided.vec10.1
  %31 = fadd <4 x float> %strided.vec5.1, %30
  %32 = fmul <4 x float> %broadcast.splat14, %strided.vec11.1
  %33 = fadd <4 x float> %strided.vec6.1, %32
  %34 = fmul <4 x float> %broadcast.splat14, %strided.vec12.1
  %35 = fadd <4 x float> %strided.vec7.1, %34
  %36 = shufflevector <4 x float> %29, <4 x float> %31, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %37 = shufflevector <4 x float> %33, <4 x float> %35, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec.1 = shufflevector <8 x float> %36, <8 x float> %37, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %strided.vec.2 = shufflevector <16 x float> %wide.vec.219, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec5.2 = shufflevector <16 x float> %wide.vec.219, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec6.2 = shufflevector <16 x float> %wide.vec.219, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec7.2 = shufflevector <16 x float> %wide.vec.219, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %38 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar23, i64 %offset.idx.2
  %39 = bitcast float* %38 to <16 x float>*
  %wide.vec8.2 = load <16 x float>, <16 x float>* %39, align 16, !alias.scope !11, !noalias !13
  %strided.vec9.2 = shufflevector <16 x float> %wide.vec8.2, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec10.2 = shufflevector <16 x float> %wide.vec8.2, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec11.2 = shufflevector <16 x float> %wide.vec8.2, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec12.2 = shufflevector <16 x float> %wide.vec8.2, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %40 = fmul <4 x float> %broadcast.splat14, %strided.vec9.2
  %41 = fadd <4 x float> %strided.vec.2, %40
  %42 = fmul <4 x float> %broadcast.splat14, %strided.vec10.2
  %43 = fadd <4 x float> %strided.vec5.2, %42
  %44 = fmul <4 x float> %broadcast.splat14, %strided.vec11.2
  %45 = fadd <4 x float> %strided.vec6.2, %44
  %46 = fmul <4 x float> %broadcast.splat14, %strided.vec12.2
  %47 = fadd <4 x float> %strided.vec7.2, %46
  %48 = shufflevector <4 x float> %41, <4 x float> %43, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %49 = shufflevector <4 x float> %45, <4 x float> %47, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec.2 = shufflevector <8 x float> %48, <8 x float> %49, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %strided.vec.3 = shufflevector <16 x float> %wide.vec.321, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec5.3 = shufflevector <16 x float> %wide.vec.321, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec6.3 = shufflevector <16 x float> %wide.vec.321, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec7.3 = shufflevector <16 x float> %wide.vec.321, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %50 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar23, i64 %offset.idx.3
  %51 = bitcast float* %50 to <16 x float>*
  %wide.vec8.3 = load <16 x float>, <16 x float>* %51, align 16, !alias.scope !11, !noalias !13
  %strided.vec9.3 = shufflevector <16 x float> %wide.vec8.3, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec10.3 = shufflevector <16 x float> %wide.vec8.3, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec11.3 = shufflevector <16 x float> %wide.vec8.3, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec12.3 = shufflevector <16 x float> %wide.vec8.3, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %52 = fmul <4 x float> %broadcast.splat14, %strided.vec9.3
  %53 = fadd <4 x float> %strided.vec.3, %52
  %54 = fmul <4 x float> %broadcast.splat14, %strided.vec10.3
  %55 = fadd <4 x float> %strided.vec5.3, %54
  %56 = fmul <4 x float> %broadcast.splat14, %strided.vec11.3
  %57 = fadd <4 x float> %strided.vec6.3, %56
  %58 = fmul <4 x float> %broadcast.splat14, %strided.vec12.3
  %59 = fadd <4 x float> %strided.vec7.3, %58
  %60 = shufflevector <4 x float> %53, <4 x float> %55, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %61 = shufflevector <4 x float> %57, <4 x float> %59, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec.3 = shufflevector <8 x float> %60, <8 x float> %61, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %polly.indvar_next24 = add nuw nsw i64 %polly.indvar23, 1
  %exitcond = icmp eq i64 %polly.indvar_next24, %indvars.iv3
  br i1 %exitcond, label %polly.loop_exit22, label %vector.ph
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #5

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "polly-optimized" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { "polly.skip.fn" }
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
