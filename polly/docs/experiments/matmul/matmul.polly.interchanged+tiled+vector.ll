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
  br i1 %exitcond1, label %polly.exiting, label %polly.loop_header, !llvm.loop !2

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
  store float %p_conv4, float* %scevgep7, align 8, !alias.scope !3, !noalias !5, !llvm.mem.parallel_loop_access !2
  %scevgep9 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar, i64 %polly.indvar4
  store float %p_conv4, float* %scevgep9, align 8, !alias.scope !6, !noalias !7, !llvm.mem.parallel_loop_access !2
  %polly.indvar_next5 = or i64 %polly.indvar4, 1
  %5 = trunc i64 %polly.indvar_next5 to i32
  %6 = mul nuw nsw i32 %5, %0
  %7 = and i32 %6, 1023
  %8 = add nuw nsw i32 %7, 1
  %p_conv.1 = sitofp i32 %8 to double
  %p_div.1 = fmul double %p_conv.1, 5.000000e-01
  %p_conv4.1 = fptrunc double %p_div.1 to float
  %scevgep7.1 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %polly.indvar, i64 %polly.indvar_next5
  store float %p_conv4.1, float* %scevgep7.1, align 4, !alias.scope !3, !noalias !5, !llvm.mem.parallel_loop_access !2
  %scevgep9.1 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar, i64 %polly.indvar_next5
  store float %p_conv4.1, float* %scevgep9.1, align 4, !alias.scope !6, !noalias !7, !llvm.mem.parallel_loop_access !2
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
  %indvars.iv5 = phi i64 [ 64, %entry ], [ %indvars.iv.next6, %polly.loop_exit16 ]
  %polly.indvar11 = phi i64 [ 0, %entry ], [ %polly.indvar_next12, %polly.loop_exit16 ]
  br label %polly.loop_header14

polly.loop_exit16:                                ; preds = %polly.loop_exit22
  %polly.indvar_next12 = add nuw nsw i64 %polly.indvar11, 64
  %polly.loop_cond13 = icmp ult i64 %polly.indvar_next12, 1536
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 64
  br i1 %polly.loop_cond13, label %polly.loop_header8, label %polly.exiting, !llvm.loop !8

polly.loop_header14:                              ; preds = %polly.loop_header8, %polly.loop_exit22
  %indvar = phi i64 [ 0, %polly.loop_header8 ], [ %indvar.next, %polly.loop_exit22 ]
  %polly.indvar17 = phi i64 [ 0, %polly.loop_header8 ], [ %polly.indvar_next18, %polly.loop_exit22 ]
  %0 = shl i64 %indvar, 6
  %offset.idx.1 = or i64 %0, 16
  %offset.idx.2 = or i64 %0, 32
  %offset.idx.3 = or i64 %0, 48
  br label %polly.loop_header20

polly.loop_exit22:                                ; preds = %polly.loop_exit28
  %polly.indvar_next18 = add nuw nsw i64 %polly.indvar17, 64
  %polly.loop_cond19 = icmp ult i64 %polly.indvar_next18, 1536
  %indvar.next = add i64 %indvar, 1
  br i1 %polly.loop_cond19, label %polly.loop_header14, label %polly.loop_exit16

polly.loop_header20:                              ; preds = %polly.loop_header14, %polly.loop_exit28
  %indvars.iv3 = phi i64 [ 64, %polly.loop_header14 ], [ %indvars.iv.next4, %polly.loop_exit28 ]
  %polly.indvar23 = phi i64 [ 0, %polly.loop_header14 ], [ %polly.indvar_next24, %polly.loop_exit28 ]
  br label %polly.loop_header26

polly.loop_exit28:                                ; preds = %polly.loop_exit34
  %polly.indvar_next24 = add nuw nsw i64 %polly.indvar23, 64
  %polly.loop_cond25 = icmp ult i64 %polly.indvar_next24, 1536
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 64
  br i1 %polly.loop_cond25, label %polly.loop_header20, label %polly.loop_exit22

polly.loop_header26:                              ; preds = %polly.loop_exit34, %polly.loop_header20
  %polly.indvar29 = phi i64 [ %polly.indvar11, %polly.loop_header20 ], [ %polly.indvar_next30, %polly.loop_exit34 ]
  %1 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %0
  %2 = bitcast float* %1 to <16 x float>*
  %3 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.1
  %4 = bitcast float* %3 to <16 x float>*
  %5 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.2
  %6 = bitcast float* %5 to <16 x float>*
  %7 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %polly.indvar29, i64 %offset.idx.3
  %8 = bitcast float* %7 to <16 x float>*
  %.promoted = load <16 x float>, <16 x float>* %2, align 4, !alias.scope !9, !noalias !11
  %.promoted22 = load <16 x float>, <16 x float>* %4, align 4, !alias.scope !9, !noalias !11
  %.promoted24 = load <16 x float>, <16 x float>* %6, align 4, !alias.scope !9, !noalias !11
  %.promoted26 = load <16 x float>, <16 x float>* %8, align 4, !alias.scope !9, !noalias !11
  br label %vector.ph

polly.loop_exit34:                                ; preds = %vector.ph
  store <16 x float> %interleaved.vec, <16 x float>* %2, align 4, !alias.scope !9, !noalias !11
  store <16 x float> %interleaved.vec.1, <16 x float>* %4, align 4, !alias.scope !9, !noalias !11
  store <16 x float> %interleaved.vec.2, <16 x float>* %6, align 4, !alias.scope !9, !noalias !11
  store <16 x float> %interleaved.vec.3, <16 x float>* %8, align 4, !alias.scope !9, !noalias !11
  %polly.indvar_next30 = add nuw nsw i64 %polly.indvar29, 1
  %exitcond7 = icmp eq i64 %polly.indvar_next30, %indvars.iv5
  br i1 %exitcond7, label %polly.loop_exit28, label %polly.loop_header26

vector.ph:                                        ; preds = %polly.loop_header26, %vector.ph
  %wide.vec.327 = phi <16 x float> [ %.promoted26, %polly.loop_header26 ], [ %interleaved.vec.3, %vector.ph ]
  %wide.vec.225 = phi <16 x float> [ %.promoted24, %polly.loop_header26 ], [ %interleaved.vec.2, %vector.ph ]
  %wide.vec.123 = phi <16 x float> [ %.promoted22, %polly.loop_header26 ], [ %interleaved.vec.1, %vector.ph ]
  %wide.vec21 = phi <16 x float> [ %.promoted, %polly.loop_header26 ], [ %interleaved.vec, %vector.ph ]
  %polly.indvar35 = phi i64 [ %polly.indvar23, %polly.loop_header26 ], [ %polly.indvar_next36, %vector.ph ]
  %scevgep53 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %polly.indvar29, i64 %polly.indvar35
  %_p_scalar_54 = load float, float* %scevgep53, align 4, !alias.scope !12, !noalias !14, !llvm.mem.parallel_loop_access !8
  %broadcast.splatinsert19 = insertelement <4 x float> undef, float %_p_scalar_54, i32 0
  %broadcast.splat20 = shufflevector <4 x float> %broadcast.splatinsert19, <4 x float> undef, <4 x i32> zeroinitializer
  %strided.vec = shufflevector <16 x float> %wide.vec21, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec11 = shufflevector <16 x float> %wide.vec21, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec12 = shufflevector <16 x float> %wide.vec21, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec13 = shufflevector <16 x float> %wide.vec21, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %9 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %0
  %10 = bitcast float* %9 to <16 x float>*
  %wide.vec14 = load <16 x float>, <16 x float>* %10, align 16, !alias.scope !13, !noalias !15
  %strided.vec15 = shufflevector <16 x float> %wide.vec14, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec16 = shufflevector <16 x float> %wide.vec14, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec17 = shufflevector <16 x float> %wide.vec14, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec18 = shufflevector <16 x float> %wide.vec14, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %11 = fmul <4 x float> %broadcast.splat20, %strided.vec15
  %12 = fadd <4 x float> %strided.vec, %11
  %13 = fmul <4 x float> %broadcast.splat20, %strided.vec16
  %14 = fadd <4 x float> %strided.vec11, %13
  %15 = fmul <4 x float> %broadcast.splat20, %strided.vec17
  %16 = fadd <4 x float> %strided.vec12, %15
  %17 = fmul <4 x float> %broadcast.splat20, %strided.vec18
  %18 = fadd <4 x float> %strided.vec13, %17
  %19 = shufflevector <4 x float> %12, <4 x float> %14, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %20 = shufflevector <4 x float> %16, <4 x float> %18, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec = shufflevector <8 x float> %19, <8 x float> %20, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %strided.vec.1 = shufflevector <16 x float> %wide.vec.123, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec11.1 = shufflevector <16 x float> %wide.vec.123, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec12.1 = shufflevector <16 x float> %wide.vec.123, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec13.1 = shufflevector <16 x float> %wide.vec.123, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %21 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.1
  %22 = bitcast float* %21 to <16 x float>*
  %wide.vec14.1 = load <16 x float>, <16 x float>* %22, align 16, !alias.scope !13, !noalias !15
  %strided.vec15.1 = shufflevector <16 x float> %wide.vec14.1, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec16.1 = shufflevector <16 x float> %wide.vec14.1, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec17.1 = shufflevector <16 x float> %wide.vec14.1, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec18.1 = shufflevector <16 x float> %wide.vec14.1, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %23 = fmul <4 x float> %broadcast.splat20, %strided.vec15.1
  %24 = fadd <4 x float> %strided.vec.1, %23
  %25 = fmul <4 x float> %broadcast.splat20, %strided.vec16.1
  %26 = fadd <4 x float> %strided.vec11.1, %25
  %27 = fmul <4 x float> %broadcast.splat20, %strided.vec17.1
  %28 = fadd <4 x float> %strided.vec12.1, %27
  %29 = fmul <4 x float> %broadcast.splat20, %strided.vec18.1
  %30 = fadd <4 x float> %strided.vec13.1, %29
  %31 = shufflevector <4 x float> %24, <4 x float> %26, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %32 = shufflevector <4 x float> %28, <4 x float> %30, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec.1 = shufflevector <8 x float> %31, <8 x float> %32, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %strided.vec.2 = shufflevector <16 x float> %wide.vec.225, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec11.2 = shufflevector <16 x float> %wide.vec.225, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec12.2 = shufflevector <16 x float> %wide.vec.225, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec13.2 = shufflevector <16 x float> %wide.vec.225, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %33 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.2
  %34 = bitcast float* %33 to <16 x float>*
  %wide.vec14.2 = load <16 x float>, <16 x float>* %34, align 16, !alias.scope !13, !noalias !15
  %strided.vec15.2 = shufflevector <16 x float> %wide.vec14.2, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec16.2 = shufflevector <16 x float> %wide.vec14.2, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec17.2 = shufflevector <16 x float> %wide.vec14.2, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec18.2 = shufflevector <16 x float> %wide.vec14.2, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %35 = fmul <4 x float> %broadcast.splat20, %strided.vec15.2
  %36 = fadd <4 x float> %strided.vec.2, %35
  %37 = fmul <4 x float> %broadcast.splat20, %strided.vec16.2
  %38 = fadd <4 x float> %strided.vec11.2, %37
  %39 = fmul <4 x float> %broadcast.splat20, %strided.vec17.2
  %40 = fadd <4 x float> %strided.vec12.2, %39
  %41 = fmul <4 x float> %broadcast.splat20, %strided.vec18.2
  %42 = fadd <4 x float> %strided.vec13.2, %41
  %43 = shufflevector <4 x float> %36, <4 x float> %38, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %44 = shufflevector <4 x float> %40, <4 x float> %42, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec.2 = shufflevector <8 x float> %43, <8 x float> %44, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %strided.vec.3 = shufflevector <16 x float> %wide.vec.327, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec11.3 = shufflevector <16 x float> %wide.vec.327, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec12.3 = shufflevector <16 x float> %wide.vec.327, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec13.3 = shufflevector <16 x float> %wide.vec.327, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %45 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %polly.indvar35, i64 %offset.idx.3
  %46 = bitcast float* %45 to <16 x float>*
  %wide.vec14.3 = load <16 x float>, <16 x float>* %46, align 16, !alias.scope !13, !noalias !15
  %strided.vec15.3 = shufflevector <16 x float> %wide.vec14.3, <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.vec16.3 = shufflevector <16 x float> %wide.vec14.3, <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %strided.vec17.3 = shufflevector <16 x float> %wide.vec14.3, <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %strided.vec18.3 = shufflevector <16 x float> %wide.vec14.3, <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %47 = fmul <4 x float> %broadcast.splat20, %strided.vec15.3
  %48 = fadd <4 x float> %strided.vec.3, %47
  %49 = fmul <4 x float> %broadcast.splat20, %strided.vec16.3
  %50 = fadd <4 x float> %strided.vec11.3, %49
  %51 = fmul <4 x float> %broadcast.splat20, %strided.vec17.3
  %52 = fadd <4 x float> %strided.vec12.3, %51
  %53 = fmul <4 x float> %broadcast.splat20, %strided.vec18.3
  %54 = fadd <4 x float> %strided.vec13.3, %53
  %55 = shufflevector <4 x float> %48, <4 x float> %50, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %56 = shufflevector <4 x float> %52, <4 x float> %54, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec.3 = shufflevector <8 x float> %55, <8 x float> %56, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %polly.indvar_next36 = add nuw nsw i64 %polly.indvar35, 1
  %exitcond = icmp eq i64 %polly.indvar_next36, %indvars.iv3
  br i1 %exitcond, label %polly.loop_exit34, label %vector.ph
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
!2 = distinct !{!2}
!3 = distinct !{!3, !4, !"polly.alias.scope.MemRef_A"}
!4 = distinct !{!4, !"polly.alias.scope.domain"}
!5 = !{!6}
!6 = distinct !{!6, !4, !"polly.alias.scope.MemRef_B"}
!7 = !{!3}
!8 = distinct !{!8}
!9 = distinct !{!9, !10, !"polly.alias.scope.MemRef_C"}
!10 = distinct !{!10, !"polly.alias.scope.domain"}
!11 = !{!12, !13}
!12 = distinct !{!12, !10, !"polly.alias.scope.MemRef_A"}
!13 = distinct !{!13, !10, !"polly.alias.scope.MemRef_B"}
!14 = !{!9, !13}
!15 = !{!9, !12}
