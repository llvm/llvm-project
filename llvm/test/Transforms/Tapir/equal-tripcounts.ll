; RUN: opt < %s -loop-spawning-ti -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.COMPLEX = type { float, float }

@.str = private unnamed_addr constant [15 x i8] c"n=%d error=%e\0A\00", align 1
@.str.2 = private unnamed_addr constant [10 x i8] c"%f + %fi\0A\00", align 1
@.str.4 = private unnamed_addr constant [9 x i8] c"n=%d ok\0A\00", align 1
@str = private unnamed_addr constant [4 x i8] c"ct:\00"
@str.18 = private unnamed_addr constant [5 x i8] c"seq:\00"

; Function Attrs: nounwind uwtable
define void @test_correctness() local_unnamed_addr #1 {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %call = tail call noalias i8* @malloc(i64 6400) #5
  %0 = bitcast i8* %call to %struct.COMPLEX*
  %call1 = tail call noalias i8* @malloc(i64 6400) #5
  %1 = bitcast i8* %call1 to %struct.COMPLEX*
  %call2 = tail call noalias i8* @malloc(i64 6400) #5
  %2 = bitcast i8* %call2 to %struct.COMPLEX*
  %call3 = tail call noalias i8* @malloc(i64 6400) #5
  %3 = bitcast i8* %call3 to %struct.COMPLEX*
  br label %for.body

for.body:                                         ; preds = %for.inc121, %entry
  %indvar = phi i64 [ %indvar.next, %for.inc121 ], [ 0, %entry ]
  %indvars.iv217 = phi i64 [ %indvars.iv.next218, %for.inc121 ], [ 1, %entry ]
  %4 = add i64 %indvar, 1
  %xtraiter = and i64 %4, 1
  %5 = icmp eq i64 %indvar, 0
  br i1 %5, label %for.end.unr-lcssa, label %for.body.new

for.body.new:                                     ; preds = %for.body
  %unroll_iter = sub i64 %4, %xtraiter
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body.new
  %indvars.iv = phi i64 [ 0, %for.body.new ], [ %indvars.iv.next.1, %for.body6 ]
  %niter = phi i64 [ %unroll_iter, %for.body.new ], [ %niter.nsub.1, %for.body6 ]
  %6 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %6 to float
  %re = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %1, i64 %indvars.iv, i32 0
  store float %conv, float* %re, align 4, !tbaa !10
  %re9 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %0, i64 %indvars.iv, i32 0
  store float %conv, float* %re9, align 4, !tbaa !10
  %im = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %1, i64 %indvars.iv, i32 1
  store float 0.000000e+00, float* %im, align 4, !tbaa !13
  %im14 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %0, i64 %indvars.iv, i32 1
  store float 0.000000e+00, float* %im14, align 4, !tbaa !13
  %indvars.iv.next = or i64 %indvars.iv, 1
  %7 = trunc i64 %indvars.iv.next to i32
  %conv.1 = sitofp i32 %7 to float
  %re.1 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %1, i64 %indvars.iv.next, i32 0
  store float %conv.1, float* %re.1, align 4, !tbaa !10
  %re9.1 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %0, i64 %indvars.iv.next, i32 0
  store float %conv.1, float* %re9.1, align 4, !tbaa !10
  %im.1 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %1, i64 %indvars.iv.next, i32 1
  store float 0.000000e+00, float* %im.1, align 4, !tbaa !13
  %im14.1 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %0, i64 %indvars.iv.next, i32 1
  store float 0.000000e+00, float* %im14.1, align 4, !tbaa !13
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv, 2
  %niter.nsub.1 = add i64 %niter, -2
  %niter.ncmp.1 = icmp eq i64 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %for.end.unr-lcssa, label %for.body6

for.end.unr-lcssa:                                ; preds = %for.body6, %for.body
  %indvars.iv.unr = phi i64 [ 0, %for.body ], [ %indvars.iv.next.1, %for.body6 ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %for.end, label %for.body6.epil

for.body6.epil:                                   ; preds = %for.end.unr-lcssa
  %8 = trunc i64 %indvars.iv.unr to i32
  %conv.epil = sitofp i32 %8 to float
  %re.epil = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %1, i64 %indvars.iv.unr, i32 0
  store float %conv.epil, float* %re.epil, align 4, !tbaa !10
  %re9.epil = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %0, i64 %indvars.iv.unr, i32 0
  store float %conv.epil, float* %re9.epil, align 4, !tbaa !10
  %im.epil = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %1, i64 %indvars.iv.unr, i32 1
  store float 0.000000e+00, float* %im.epil, align 4, !tbaa !13
  %im14.epil = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %0, i64 %indvars.iv.unr, i32 1
  store float 0.000000e+00, float* %im14.epil, align 4, !tbaa !13
  br label %for.end

for.end:                                          ; preds = %for.end.unr-lcssa, %for.body6.epil
  %9 = trunc i64 %indvars.iv217 to i32
  tail call void @cilk_fft(i32 %9, %struct.COMPLEX* nonnull %0, %struct.COMPLEX* %2)
  %conv4.i.i = sitofp i32 %9 to double
; CHECK: for.end:
; CHECK: call fastcc void @test_correctness_pfor.detach.us.i.ls2(i64 0,
; CHECK: {{i64 %indvars.iv217|i64 %4}}
  br label %pfor.detach.us.i

pfor.detach.us.i:                                 ; preds = %pfor.inc.us.i, %for.end
  %indvars.iv.i = phi i64 [ %indvars.iv.next.i, %pfor.inc.us.i ], [ 0, %for.end ]
  detach within %syncreg.i, label %for.body.lr.ph.i.us.i, label %pfor.inc.us.i

for.body.lr.ph.i.us.i:                            ; preds = %pfor.detach.us.i
  %10 = trunc i64 %indvars.iv.i to i32
  br label %for.body.i.us.i

for.body.i.us.i:                                  ; preds = %for.body.i.us.i, %for.body.lr.ph.i.us.i
  %indvars.iv.i.us.i = phi i64 [ 0, %for.body.lr.ph.i.us.i ], [ %indvars.iv.next.i.us.i, %for.body.i.us.i ]
  %sum.sroa.6.071.i.us.i = phi float [ 0.000000e+00, %for.body.lr.ph.i.us.i ], [ %add40.i.us.i, %for.body.i.us.i ]
  %sum.sroa.0.070.i.us.i = phi float [ 0.000000e+00, %for.body.lr.ph.i.us.i ], [ %add.i.us.i, %for.body.i.us.i ]
  %11 = trunc i64 %indvars.iv.i.us.i to i32
  %mul1.i.us.i = mul nsw i32 %11, %10
  %rem.i.us.i = srem i32 %mul1.i.us.i, %9
  %conv2.i.us.i = sitofp i32 %rem.i.us.i to double
  %mul3.i.us.i = fmul double %conv2.i.us.i, 0x401921FB60000000
  %div.i.us.i = fdiv double %mul3.i.us.i, %conv4.i.i
  %call.i.us.i = tail call double @cos(double %div.i.us.i) #5
  %conv5.i.us.i = fptrunc double %call.i.us.i to float
  %call15.i.us.i = tail call double @sin(double %div.i.us.i) #5
  %12 = fptrunc double %call15.i.us.i to float
  %conv16.i.us.i = fsub float -0.000000e+00, %12
  %re18.i.us.i = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %1, i64 %indvars.iv.i.us.i, i32 0
  %13 = load float, float* %re18.i.us.i, align 4, !tbaa !10
  %mul20.i.us.i = fmul float %13, %conv5.i.us.i
  %im23.i.us.i = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %1, i64 %indvars.iv.i.us.i, i32 1
  %14 = load float, float* %im23.i.us.i, align 4, !tbaa !13
  %mul25.i.us.i = fmul float %14, %conv16.i.us.i
  %sub26.i.us.i = fsub float %mul20.i.us.i, %mul25.i.us.i
  %add.i.us.i = fadd float %sum.sroa.0.070.i.us.i, %sub26.i.us.i
  %mul32.i.us.i = fmul float %14, %conv5.i.us.i
  %mul37.i.us.i = fmul float %13, %conv16.i.us.i
  %add38.i.us.i = fadd float %mul37.i.us.i, %mul32.i.us.i
  %add40.i.us.i = fadd float %sum.sroa.6.071.i.us.i, %add38.i.us.i
  %indvars.iv.next.i.us.i = add nuw nsw i64 %indvars.iv.i.us.i, 1
  %exitcond.i.us.i = icmp eq i64 %indvars.iv.next.i.us.i, %indvars.iv217
  br i1 %exitcond.i.us.i, label %test_fft_elem.exit.loopexit.us.i, label %for.body.i.us.i

pfor.inc.us.i:                                    ; preds = %test_fft_elem.exit.loopexit.us.i, %pfor.detach.us.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, %indvars.iv217
  br i1 %exitcond.i, label %pfor.cond.cleanup.i, label %pfor.detach.us.i, !llvm.loop !17

test_fft_elem.exit.loopexit.us.i:                 ; preds = %for.body.i.us.i
  %sum.sroa.0.0..sroa_idx.i.us.i = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %3, i64 %indvars.iv.i, i32 0
  store float %add.i.us.i, float* %sum.sroa.0.0..sroa_idx.i.us.i, align 4
  %sum.sroa.6.0..sroa_idx50.i.us.i = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %3, i64 %indvars.iv.i, i32 1
  store float %add40.i.us.i, float* %sum.sroa.6.0..sroa_idx50.i.us.i, align 4
  reattach within %syncreg.i, label %pfor.inc.us.i

pfor.cond.cleanup.i:                              ; preds = %pfor.inc.us.i
  sync within %syncreg.i, label %pfor.cond.cleanup.i.split

pfor.cond.cleanup.i.split:                        ; preds = %pfor.cond.cleanup.i
  br label %for.body18

for.body18:                                       ; preds = %pfor.cond.cleanup.i.split, %for.body18
  %indvars.iv205 = phi i64 [ %indvars.iv.next206, %for.body18 ], [ 0, %pfor.cond.cleanup.i.split ]
  %error.0200 = phi double [ %error.1, %for.body18 ], [ 0.000000e+00, %pfor.cond.cleanup.i.split ]
  %re21 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %2, i64 %indvars.iv205, i32 0
  %re24 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %3, i64 %indvars.iv205, i32 0
  %15 = bitcast float* %re21 to <2 x float>*
  %16 = load <2 x float>, <2 x float>* %15, align 4, !tbaa !14
  %17 = bitcast float* %re24 to <2 x float>*
  %18 = load <2 x float>, <2 x float>* %17, align 4, !tbaa !14
  %19 = fsub <2 x float> %16, %18
  %20 = fmul <2 x float> %19, %19
  %21 = extractelement <2 x float> %20, i32 0
  %22 = extractelement <2 x float> %20, i32 1
  %add = fadd float %21, %22
  %conv47 = fpext float %add to double
  %call48 = tail call double @sqrt(double %conv47) #5
  %23 = fmul <2 x float> %18, %18
  %24 = extractelement <2 x float> %23, i32 0
  %25 = extractelement <2 x float> %23, i32 1
  %add63 = fadd float %24, %25
  %conv64 = fpext float %add63 to double
  %call65 = tail call double @sqrt(double %conv64) #5
  %cmp66 = fcmp olt double %call65, -1.000000e-10
  %cmp68 = fcmp ogt double %call65, 1.000000e-10
  %or.cond = or i1 %cmp66, %cmp68
  %div = fdiv double %call48, %call65
  %a.0 = select i1 %or.cond, double %div, double %call48
  %cmp70 = fcmp ogt double %a.0, %error.0200
  %error.1 = select i1 %cmp70, double %a.0, double %error.0200
  %indvars.iv.next206 = add nuw nsw i64 %indvars.iv205, 1
  %exitcond208 = icmp eq i64 %indvars.iv.next206, %indvars.iv217
  br i1 %exitcond208, label %for.end76, label %for.body18

for.end76:                                        ; preds = %for.body18
  %cmp77 = fcmp ogt double %error.1, 1.000000e-03
  br i1 %cmp77, label %if.then79, label %if.end115

if.then79:                                        ; preds = %for.end76
  %call80 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0), i32 %9, double %error.1)
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @str, i64 0, i64 0))
  br label %for.body85

for.body85:                                       ; preds = %for.body85, %if.then79
  %indvars.iv209 = phi i64 [ 0, %if.then79 ], [ %indvars.iv.next210, %for.body85 ]
  %re88 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %3, i64 %indvars.iv209, i32 0
  %26 = load float, float* %re88, align 4, !tbaa !10
  %conv89 = fpext float %26 to double
  %im92 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %3, i64 %indvars.iv209, i32 1
  %27 = load float, float* %im92, align 4, !tbaa !13
  %conv93 = fpext float %27 to double
  %call94 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.2, i64 0, i64 0), double %conv89, double %conv93)
  %indvars.iv.next210 = add nuw nsw i64 %indvars.iv209, 1
  %exitcond212 = icmp eq i64 %indvars.iv.next210, %indvars.iv217
  br i1 %exitcond212, label %for.end97, label %for.body85

for.end97:                                        ; preds = %for.body85
  %puts197 = tail call i32 @puts(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @str.18, i64 0, i64 0))
  br label %for.body102

for.body102:                                      ; preds = %for.body102, %for.end97
  %indvars.iv213 = phi i64 [ 0, %for.end97 ], [ %indvars.iv.next214, %for.body102 ]
  %re105 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %2, i64 %indvars.iv213, i32 0
  %28 = load float, float* %re105, align 4, !tbaa !10
  %conv106 = fpext float %28 to double
  %im109 = getelementptr inbounds %struct.COMPLEX, %struct.COMPLEX* %2, i64 %indvars.iv213, i32 1
  %29 = load float, float* %im109, align 4, !tbaa !13
  %conv110 = fpext float %29 to double
  %call111 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.2, i64 0, i64 0), double %conv106, double %conv110)
  %indvars.iv.next214 = add nuw nsw i64 %indvars.iv213, 1
  %exitcond216 = icmp eq i64 %indvars.iv.next214, %indvars.iv217
  br i1 %exitcond216, label %if.end115, label %for.body102

if.end115:                                        ; preds = %for.body102, %for.end76
  %rem = urem i32 %9, 10
  %cmp116 = icmp eq i32 %rem, 0
  br i1 %cmp116, label %if.then118, label %for.inc121

if.then118:                                       ; preds = %if.end115
  %call119 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.4, i64 0, i64 0), i32 %9)
  br label %for.inc121

for.inc121:                                       ; preds = %if.end115, %if.then118
  %indvars.iv.next218 = add nuw nsw i64 %indvars.iv217, 1
  %exitcond219 = icmp eq i64 %indvars.iv.next218, 800
  %indvar.next = add i64 %indvar, 1
  br i1 %exitcond219, label %for.end123, label %for.body

for.end123:                                       ; preds = %for.inc121
  ret void
}

; CHECK: define internal fastcc void @test_correctness_pfor.detach.us.i.ls2(i64 %indvars.iv.i.start.ls2, i64 %end.ls2
; CHECK: pfor.inc.us.i.ls2:
; CHECK: %exitcond.i.ls2 = icmp eq i64 %indvars.iv.next.i.ls2, %end.ls2

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #2

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #5

; Function Attrs: nounwind
declare double @sqrt(double) local_unnamed_addr #3

; Function Attrs: nounwind
declare double @cos(double) local_unnamed_addr #3

; Function Attrs: nounwind
declare double @sin(double) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
declare void @cilk_fft(i32 %n, %struct.COMPLEX* %in, %struct.COMPLEX* %out) local_unnamed_addr #1

attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!2 = !{!3, !4, i64 0}
!3 = !{!"timeval", !4, i64 0, !4, i64 8}
!4 = !{!"long", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 8}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !5, i64 0}
!10 = !{!11, !12, i64 0}
!11 = !{!"", !12, i64 0, !12, i64 4}
!12 = !{!"float", !5, i64 0}
!13 = !{!11, !12, i64 4}
!14 = !{!12, !12, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"tapir.loop.spawn.strategy", i32 1}
!17 = distinct !{!17, !16}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.unroll.disable"}
!20 = !{!4, !4, i64 0}
!21 = !{!22, !22, i64 0}
!22 = !{!"any pointer", !5, i64 0}
!23 = distinct !{!23, !19}
!24 = distinct !{!24, !19}
!25 = distinct !{!25, !19}
