; RUN: llc < %s -march=x86 -regalloc=greedy -stop-after=greedy | FileCheck %s
; Make sure bad eviction sequence doesnt occur

; Fix for bugzilla 26810.
; This test is meant to make sure bad eviction sequence like the one described
; below does not occur
;
; movapd	%xmm7, 160(%esp)        # 16-byte Spill
; movapd	%xmm5, %xmm7
; movapd	%xmm4, %xmm5
; movapd	%xmm3, %xmm4
; movapd	%xmm2, %xmm3
; some_inst
; movapd	%xmm3, %xmm2
; movapd	%xmm4, %xmm3
; movapd	%xmm5, %xmm4
; movapd	%xmm7, %xmm5
; movapd	160(%esp), %xmm7        # 16-byte Reload

; Make sure we have no redundant copies in the problematic code section
; CHECK-LABEL: name: loop
; CHECK: bb.2.for.body:
; CHECK: SUBPDrr
; CHECK-NEXT: MOVAPSmr
; CHECK-NEXT: MULPDrm
; CHECK-NEXT: MOVAPSrm
; CHECK-NEXT: ADDPDrr
; CHECK-NEXT: MOVAPSmr
; CHECK-NEXT: ADD32ri8

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-linux-gnu"

%struct._iobuf = type { i8* }

$"\01??_C@_01NOFIACDB@w?$AA@" = comdat any

$"\01??_C@_09LAIDGMDM@?1dev?1null?$AA@" = comdat any

@"\01?v@@3PAU__m128d@@A" = global [8 x <2 x double>] zeroinitializer, align 16
@"\01?m1@@3PAU__m128d@@A" = local_unnamed_addr global [76800000 x <2 x double>] zeroinitializer, align 16
@"\01?m2@@3PAU__m128d@@A" = local_unnamed_addr global [8 x <2 x double>] zeroinitializer, align 16
@"\01??_C@_01NOFIACDB@w?$AA@" = linkonce_odr unnamed_addr constant [2 x i8] c"w\00", comdat, align 1
@"\01??_C@_09LAIDGMDM@?1dev?1null?$AA@" = linkonce_odr unnamed_addr constant [10 x i8] c"/dev/null\00", comdat, align 1

; Function Attrs: norecurse
define i32 @main() local_unnamed_addr #0 {
entry:
  tail call void @init()
  %0 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 0), align 16, !tbaa !8
  %1 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 1), align 16, !tbaa !8
  %2 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 2), align 16, !tbaa !8
  %3 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 3), align 16, !tbaa !8
  %4 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 4), align 16, !tbaa !8
  %5 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 5), align 16, !tbaa !8
  %6 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 6), align 16, !tbaa !8
  %7 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 7), align 16, !tbaa !8
  %.promoted.i = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 0), align 16, !tbaa !8
  %.promoted51.i = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 1), align 16, !tbaa !8
  %.promoted53.i = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 2), align 16, !tbaa !8
  %.promoted55.i = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 3), align 16, !tbaa !8
  %.promoted57.i = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 4), align 16, !tbaa !8
  %.promoted59.i = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 5), align 16, !tbaa !8
  %.promoted61.i = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 6), align 16, !tbaa !8
  %.promoted63.i = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 7), align 16, !tbaa !8
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %add.i64.i = phi <2 x double> [ %.promoted63.i, %entry ], [ %add.i.i, %for.body.i ]
  %add.i3662.i = phi <2 x double> [ %.promoted61.i, %entry ], [ %add.i36.i, %for.body.i ]
  %add.i3860.i = phi <2 x double> [ %.promoted59.i, %entry ], [ %add.i38.i, %for.body.i ]
  %add.i4058.i = phi <2 x double> [ %.promoted57.i, %entry ], [ %add.i40.i, %for.body.i ]
  %add.i4256.i = phi <2 x double> [ %.promoted55.i, %entry ], [ %add.i42.i, %for.body.i ]
  %add.i4454.i = phi <2 x double> [ %.promoted53.i, %entry ], [ %add.i44.i, %for.body.i ]
  %add.i4652.i = phi <2 x double> [ %.promoted51.i, %entry ], [ %add.i46.i, %for.body.i ]
  %add.i4850.i = phi <2 x double> [ %.promoted.i, %entry ], [ %add.i48.i, %for.body.i ]
  %i.049.i = phi i32 [ 0, %entry ], [ %inc.i, %for.body.i ]
  %arrayidx.i = getelementptr inbounds [76800000 x <2 x double>], [76800000 x <2 x double>]* @"\01?m1@@3PAU__m128d@@A", i32 0, i32 %i.049.i
  %8 = load <2 x double>, <2 x double>* %arrayidx.i, align 16, !tbaa !8
  %mul.i.i = fmul <2 x double> %0, %8
  %add.i48.i = fadd <2 x double> %add.i4850.i, %mul.i.i
  %mul.i47.i = fmul <2 x double> %1, %8
  %add.i46.i = fadd <2 x double> %add.i4652.i, %mul.i47.i
  %mul.i45.i = fmul <2 x double> %2, %8
  %add.i44.i = fadd <2 x double> %add.i4454.i, %mul.i45.i
  %mul.i43.i = fmul <2 x double> %3, %8
  %add.i42.i = fadd <2 x double> %add.i4256.i, %mul.i43.i
  %mul.i41.i = fmul <2 x double> %4, %8
  %add.i40.i = fadd <2 x double> %add.i4058.i, %mul.i41.i
  %mul.i39.i = fmul <2 x double> %5, %8
  %add.i38.i = fadd <2 x double> %add.i3860.i, %mul.i39.i
  %mul.i37.i = fmul <2 x double> %6, %8
  %add.i36.i = fsub <2 x double> %add.i3662.i, %mul.i37.i
  %mul.i35.i = fmul <2 x double> %7, %8
  %add.i.i = fadd <2 x double> %add.i64.i, %mul.i35.i
  %inc.i = add nuw nsw i32 %i.049.i, 1
  %exitcond.i = icmp eq i32 %inc.i, 76800000
  br i1 %exitcond.i, label %loop.exit, label %for.body.i

loop.exit:                           ; preds = %for.body.i
  store <2 x double> %add.i48.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 0), align 16, !tbaa !8
  store <2 x double> %add.i46.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 1), align 16, !tbaa !8
  store <2 x double> %add.i46.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 1), align 16, !tbaa !8
  store <2 x double> %add.i44.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 2), align 16, !tbaa !8
  store <2 x double> %add.i42.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 3), align 16, !tbaa !8
  store <2 x double> %add.i40.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 4), align 16, !tbaa !8
  store <2 x double> %add.i38.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 5), align 16, !tbaa !8
  store <2 x double> %add.i36.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 6), align 16, !tbaa !8
  store <2 x double> %add.i.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 7), align 16, !tbaa !8
  %call.i = tail call %struct._iobuf* @fopen(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @"\01??_C@_09LAIDGMDM@?1dev?1null?$AA@", i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @"\01??_C@_01NOFIACDB@w?$AA@", i32 0, i32 0)) #7
  %call1.i = tail call i32 @fwrite(i8* bitcast ([8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A" to i8*), i32 16, i32 8, %struct._iobuf* %call.i) #7
  %call2.i = tail call i32 @fclose(%struct._iobuf* %call.i) #7
  ret i32 0
}

define void @init() local_unnamed_addr #1 {
entry:
  call void @llvm.memset.p0i8.i32(i8* align 16 bitcast ([8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A" to i8*), i8 0, i32 128, i1 false)
  %call.i = tail call i64 @_time64(i64* null)
  %conv = trunc i64 %call.i to i32
  tail call void @srand(i32 %conv)
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %entry
  %i2.051 = phi i32 [ 0, %entry ], [ %inc14, %for.body6 ]
  %call7 = tail call i32 @rand()
  %conv8 = sitofp i32 %call7 to double
  %tmp.sroa.0.0.vec.insert = insertelement <2 x double> undef, double %conv8, i32 0
  %call9 = tail call i32 @rand()
  %conv10 = sitofp i32 %call9 to double
  %tmp.sroa.0.8.vec.insert = insertelement <2 x double> %tmp.sroa.0.0.vec.insert, double %conv10, i32 1
  %arrayidx12 = getelementptr inbounds [76800000 x <2 x double>], [76800000 x <2 x double>]* @"\01?m1@@3PAU__m128d@@A", i32 0, i32 %i2.051
  store <2 x double> %tmp.sroa.0.8.vec.insert, <2 x double>* %arrayidx12, align 16, !tbaa !8
  %inc14 = add nuw nsw i32 %i2.051, 1
  %exitcond = icmp eq i32 %inc14, 76800000
  br i1 %exitcond, label %for.body21.preheader, label %for.body6

for.body21.preheader:                             ; preds = %for.body6
  %call25 = tail call i32 @rand()
  %conv26 = sitofp i32 %call25 to double
  %tmp23.sroa.0.0.vec.insert = insertelement <2 x double> undef, double %conv26, i32 0
  %call28 = tail call i32 @rand()
  %conv29 = sitofp i32 %call28 to double
  %tmp23.sroa.0.8.vec.insert = insertelement <2 x double> %tmp23.sroa.0.0.vec.insert, double %conv29, i32 1
  store <2 x double> %tmp23.sroa.0.8.vec.insert, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 0), align 16, !tbaa !8
  %call25.1 = tail call i32 @rand()
  %conv26.1 = sitofp i32 %call25.1 to double
  %tmp23.sroa.0.0.vec.insert.1 = insertelement <2 x double> undef, double %conv26.1, i32 0
  %call28.1 = tail call i32 @rand()
  %conv29.1 = sitofp i32 %call28.1 to double
  %tmp23.sroa.0.8.vec.insert.1 = insertelement <2 x double> %tmp23.sroa.0.0.vec.insert.1, double %conv29.1, i32 1
  store <2 x double> %tmp23.sroa.0.8.vec.insert.1, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 1), align 16, !tbaa !8
  %call25.2 = tail call i32 @rand()
  %conv26.2 = sitofp i32 %call25.2 to double
  %tmp23.sroa.0.0.vec.insert.2 = insertelement <2 x double> undef, double %conv26.2, i32 0
  %call28.2 = tail call i32 @rand()
  %conv29.2 = sitofp i32 %call28.2 to double
  %tmp23.sroa.0.8.vec.insert.2 = insertelement <2 x double> %tmp23.sroa.0.0.vec.insert.2, double %conv29.2, i32 1
  store <2 x double> %tmp23.sroa.0.8.vec.insert.2, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 2), align 16, !tbaa !8
  %call25.3 = tail call i32 @rand()
  %conv26.3 = sitofp i32 %call25.3 to double
  %tmp23.sroa.0.0.vec.insert.3 = insertelement <2 x double> undef, double %conv26.3, i32 0
  %call28.3 = tail call i32 @rand()
  %conv29.3 = sitofp i32 %call28.3 to double
  %tmp23.sroa.0.8.vec.insert.3 = insertelement <2 x double> %tmp23.sroa.0.0.vec.insert.3, double %conv29.3, i32 1
  store <2 x double> %tmp23.sroa.0.8.vec.insert.3, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 3), align 16, !tbaa !8
  %call25.4 = tail call i32 @rand()
  %conv26.4 = sitofp i32 %call25.4 to double
  %tmp23.sroa.0.0.vec.insert.4 = insertelement <2 x double> undef, double %conv26.4, i32 0
  %call28.4 = tail call i32 @rand()
  %conv29.4 = sitofp i32 %call28.4 to double
  %tmp23.sroa.0.8.vec.insert.4 = insertelement <2 x double> %tmp23.sroa.0.0.vec.insert.4, double %conv29.4, i32 1
  store <2 x double> %tmp23.sroa.0.8.vec.insert.4, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 4), align 16, !tbaa !8
  %call25.5 = tail call i32 @rand()
  %conv26.5 = sitofp i32 %call25.5 to double
  %tmp23.sroa.0.0.vec.insert.5 = insertelement <2 x double> undef, double %conv26.5, i32 0
  %call28.5 = tail call i32 @rand()
  %conv29.5 = sitofp i32 %call28.5 to double
  %tmp23.sroa.0.8.vec.insert.5 = insertelement <2 x double> %tmp23.sroa.0.0.vec.insert.5, double %conv29.5, i32 1
  store <2 x double> %tmp23.sroa.0.8.vec.insert.5, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 5), align 16, !tbaa !8
  %call25.6 = tail call i32 @rand()
  %conv26.6 = sitofp i32 %call25.6 to double
  %tmp23.sroa.0.0.vec.insert.6 = insertelement <2 x double> undef, double %conv26.6, i32 0
  %call28.6 = tail call i32 @rand()
  %conv29.6 = sitofp i32 %call28.6 to double
  %tmp23.sroa.0.8.vec.insert.6 = insertelement <2 x double> %tmp23.sroa.0.0.vec.insert.6, double %conv29.6, i32 1
  store <2 x double> %tmp23.sroa.0.8.vec.insert.6, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 6), align 16, !tbaa !8
  %call25.7 = tail call i32 @rand()
  %conv26.7 = sitofp i32 %call25.7 to double
  %tmp23.sroa.0.0.vec.insert.7 = insertelement <2 x double> undef, double %conv26.7, i32 0
  %call28.7 = tail call i32 @rand()
  %conv29.7 = sitofp i32 %call28.7 to double
  %tmp23.sroa.0.8.vec.insert.7 = insertelement <2 x double> %tmp23.sroa.0.0.vec.insert.7, double %conv29.7, i32 1
  store <2 x double> %tmp23.sroa.0.8.vec.insert.7, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 7), align 16, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind
define void @loop() local_unnamed_addr #2 {
entry:
  %0 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 0), align 16, !tbaa !8
  %1 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 1), align 16, !tbaa !8
  %2 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 2), align 16, !tbaa !8
  %3 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 3), align 16, !tbaa !8
  %4 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 4), align 16, !tbaa !8
  %5 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 5), align 16, !tbaa !8
  %6 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 6), align 16, !tbaa !8
  %7 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?m2@@3PAU__m128d@@A", i32 0, i32 7), align 16, !tbaa !8
  %.promoted = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 0), align 16, !tbaa !8
  %.promoted51 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 1), align 16, !tbaa !8
  %.promoted53 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 2), align 16, !tbaa !8
  %.promoted55 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 3), align 16, !tbaa !8
  %.promoted57 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 4), align 16, !tbaa !8
  %.promoted59 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 5), align 16, !tbaa !8
  %.promoted61 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 6), align 16, !tbaa !8
  %.promoted63 = load <2 x double>, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 7), align 16, !tbaa !8
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  store <2 x double> %add.i48, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 0), align 16, !tbaa !8
  store <2 x double> %add.i46, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 1), align 16, !tbaa !8
  store <2 x double> %add.i44, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 2), align 16, !tbaa !8
  store <2 x double> %add.i42, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 3), align 16, !tbaa !8
  store <2 x double> %add.i40, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 4), align 16, !tbaa !8
  store <2 x double> %add.i38, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 5), align 16, !tbaa !8
  store <2 x double> %add.i36, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 6), align 16, !tbaa !8
  store <2 x double> %add.i, <2 x double>* getelementptr inbounds ([8 x <2 x double>], [8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A", i32 0, i32 7), align 16, !tbaa !8
  ret void

for.body:                                         ; preds = %for.body, %entry
  %add.i64 = phi <2 x double> [ %.promoted63, %entry ], [ %add.i, %for.body ]
  %add.i3662 = phi <2 x double> [ %.promoted61, %entry ], [ %add.i36, %for.body ]
  %add.i3860 = phi <2 x double> [ %.promoted59, %entry ], [ %add.i38, %for.body ]
  %add.i4058 = phi <2 x double> [ %.promoted57, %entry ], [ %add.i40, %for.body ]
  %add.i4256 = phi <2 x double> [ %.promoted55, %entry ], [ %add.i42, %for.body ]
  %add.i4454 = phi <2 x double> [ %.promoted53, %entry ], [ %add.i44, %for.body ]
  %add.i4652 = phi <2 x double> [ %.promoted51, %entry ], [ %add.i46, %for.body ]
  %add.i4850 = phi <2 x double> [ %.promoted, %entry ], [ %add.i48, %for.body ]
  %i.049 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [76800000 x <2 x double>], [76800000 x <2 x double>]* @"\01?m1@@3PAU__m128d@@A", i32 0, i32 %i.049
  %8 = load <2 x double>, <2 x double>* %arrayidx, align 16, !tbaa !8
  %mul.i = fmul <2 x double> %8, %0
  %add.i48 = fadd <2 x double> %add.i4850, %mul.i
  %mul.i47 = fmul <2 x double> %8, %1
  %add.i46 = fadd <2 x double> %add.i4652, %mul.i47
  %mul.i45 = fmul <2 x double> %8, %2
  %add.i44 = fadd <2 x double> %add.i4454, %mul.i45
  %mul.i43 = fmul <2 x double> %8, %3
  %add.i42 = fadd <2 x double> %add.i4256, %mul.i43
  %mul.i41 = fmul <2 x double> %8, %4
  %add.i40 = fadd <2 x double> %add.i4058, %mul.i41
  %mul.i39 = fmul <2 x double> %8, %5
  %add.i38 = fadd <2 x double> %add.i3860, %mul.i39
  %mul.i37 = fmul <2 x double> %8, %6
  %add.i36 = fsub <2 x double> %add.i3662, %mul.i37
  %mul.i35 = fmul <2 x double> %8, %7
  %add.i = fadd <2 x double> %add.i64, %mul.i35
  %inc = add nuw nsw i32 %i.049, 1
  %exitcond = icmp eq i32 %inc, 76800000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind
define void @"\01?dump@@YAXXZ"() local_unnamed_addr #3 {
entry:
  %call = tail call %struct._iobuf* @fopen(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @"\01??_C@_09LAIDGMDM@?1dev?1null?$AA@", i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @"\01??_C@_01NOFIACDB@w?$AA@", i32 0, i32 0))
  %call1 = tail call i32 @fwrite(i8* bitcast ([8 x <2 x double>]* @"\01?v@@3PAU__m128d@@A" to i8*), i32 16, i32 8, %struct._iobuf* %call)
  %call2 = tail call i32 @fclose(%struct._iobuf* %call)
  ret void
}

declare void @srand(i32) local_unnamed_addr #4

declare i32 @rand() local_unnamed_addr #4

; Function Attrs: nounwind
declare noalias %struct._iobuf* @fopen(i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #5

; Function Attrs: nounwind
declare i32 @fwrite(i8* nocapture, i32, i32, %struct._iobuf* nocapture) local_unnamed_addr #5

; Function Attrs: nounwind
declare i32 @fclose(%struct._iobuf* nocapture) local_unnamed_addr #5

declare i64 @_time64(i64*) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #6

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly nounwind }
attributes #7 = { nounwind }

!llvm.linker.options = !{!0, !1, !2, !3, !4}
!llvm.module.flags = !{!5, !6}
!llvm.ident = !{!7}

!0 = !{!"/FAILIFMISMATCH:\22_MSC_VER=1900\22"}
!1 = !{!"/FAILIFMISMATCH:\22_ITERATOR_DEBUG_LEVEL=0\22"}
!2 = !{!"/FAILIFMISMATCH:\22RuntimeLibrary=MT_StaticRelease\22"}
!3 = !{!"/DEFAULTLIB:libcpmt.lib"}
!4 = !{!"/FAILIFMISMATCH:\22_CRT_STDIO_ISO_WIDE_SPECIFIERS=0\22"}
!5 = !{i32 1, !"NumRegisterParameters", i32 0}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 5.0.0 (cfe/trunk 305640)"}
!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
