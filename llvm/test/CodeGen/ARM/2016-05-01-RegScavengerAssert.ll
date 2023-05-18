; RUN: llc < %s
; This test has around 4000 bytes of local variables and it also stresses register allocation
; to force a register scavenging. It tests if the stack is treated as "BigStack" and thus
; spill slots are reserved. If not, reg scavenger will assert.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"
target triple = "thumbv7--linux-android"

%struct.r = type { ptr, [10 x [9 x float]], [10 x [9 x float]], [101 x [9 x float]], [101 x [9 x float]], i32, i32, i32, i32, i32, [8 x [2 x i32]], [432 x float], [432 x i32], [10 x ptr], [10 x ptr], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], [10 x i32], i32, i32, i32, i32, float, float, i32, i32, [9 x float], ptr, float }

define void @foo(ptr %this, ptr %srcR, ptr %srcC, ptr %tempPntsX, ptr %tY, ptr %ms, ptr %sX, ptr %sY, ptr dereferenceable(4) %num, ptr %tm, i32 %SR, i32 %lev, ptr %tdata, i32 %siW, i32 %pyw, i32 %pyh, ptr %sdata) #0 align 2 {
entry:
 %sFV = alloca [49 x float], align 4
 %tFV = alloca [49 x float], align 4
 %TIM = alloca [9 x float], align 4
 %sort_tmp = alloca [432 x float], align 4
 %msDiffs = alloca [432 x float], align 4
 %TM.sroa.0.0.copyload = load float, ptr %tm, align 4
 %TM.sroa.8.0.copyload = load float, ptr null, align 4
 %TM.sroa.9.0..sroa_idx813 = getelementptr inbounds float, ptr %tm, i32 6
 %TM.sroa.9.0.copyload = load float, ptr %TM.sroa.9.0..sroa_idx813, align 4
 %TM.sroa.11.0.copyload = load float, ptr undef, align 4
 br i1 undef, label %for.body.lr.ph, label %if.then343

for.body.lr.ph:                  ; preds = %entry
 %arrayidx8 = getelementptr inbounds %struct.r, ptr %this, i32 0, i32 1, i32 %lev, i32 0
 %arrayidx12 = getelementptr inbounds %struct.r, ptr %this, i32 0, i32 1, i32 %lev, i32 6
 %arrayidx15 = getelementptr inbounds %struct.r, ptr %this, i32 0, i32 1, i32 %lev, i32 4
 %arrayidx20 = getelementptr inbounds %struct.r, ptr %this, i32 0, i32 1, i32 %lev, i32 7
 %arrayidx24 = getelementptr inbounds %struct.r, ptr %this, i32 0, i32 2, i32 %lev, i32 0
 %arrayidx28 = getelementptr inbounds %struct.r, ptr %this, i32 0, i32 2, i32 %lev, i32 6
 %arrayidx32 = getelementptr inbounds %struct.r, ptr %this, i32 0, i32 2, i32 %lev, i32 4
 %arrayidx36 = getelementptr inbounds %struct.r, ptr %this, i32 0, i32 2, i32 %lev, i32 7
 %arrayidx84 = getelementptr inbounds [9 x float], ptr %TIM, i32 0, i32 6
 %arrayidx92 = getelementptr inbounds [9 x float], ptr %TIM, i32 0, i32 7
 %add116 = add nsw i32 %pyh, 15
 br label %for.body

for.body:                     ; preds = %for.cond.cleanup40, %for.body.lr.ph
 %arrayidx.phi = phi ptr [ %sX, %for.body.lr.ph ], [ %arrayidx.inc, %for.cond.cleanup40 ]
 %arrayidx4.phi = phi ptr [ %sY, %for.body.lr.ph ], [ %arrayidx4.inc, %for.cond.cleanup40 ]
 %0 = load float, ptr %arrayidx.phi, align 4
 %1 = load float, ptr %arrayidx4.phi, align 4
 %2 = load float, ptr %arrayidx12, align 4
 %add = fadd fast float 0.000000e+00, %2
 %3 = load float, ptr %arrayidx20, align 4
 %add21 = fadd fast float 0.000000e+00, %3
 %mul3.i = fmul fast float %add21, %TM.sroa.8.0.copyload
 %add.i = fadd fast float 0.000000e+00, %TM.sroa.11.0.copyload
 %add5.i = fadd fast float %add.i, %mul3.i
 %conv6.i = fdiv fast float 1.000000e+00, %add5.i
 %mul8.i = fmul fast float %add, %TM.sroa.0.0.copyload
 %add11.i = fadd fast float %mul8.i, %TM.sroa.9.0.copyload
 %add13.i = fadd fast float %add11.i, 0.000000e+00
 %4 = load float, ptr %arrayidx24, align 4
 %mul14.i = fmul fast float %add13.i, %4
 %mul25 = fmul fast float %mul14.i, %conv6.i
 %add29 = fadd fast float %mul25, 0.000000e+00
 %arrayidx.inc = getelementptr float, ptr %arrayidx.phi, i32 1
 %arrayidx4.inc = getelementptr float, ptr %arrayidx4.phi, i32 1
 %conv64.1 = sitofp i32 undef to float
 %conv64.6 = sitofp i32 undef to float
 br label %for.body41

for.cond.cleanup40:                ; preds = %for.body41
 %call = call fast float undef(ptr nonnull %this, ptr undef, i32 49)
 br label %for.body

for.body41:                    ; preds = %for.cond.cleanup56.for.body41_crit_edge, %for.body
 %5 = phi float [ 0.000000e+00, %for.body ], [ %.pre, %for.cond.cleanup56.for.body41_crit_edge ]
 %sFVData.0840 = phi ptr [ undef, %for.body ], [ undef, %for.cond.cleanup56.for.body41_crit_edge ]
 %dx.0838 = phi i32 [ -3, %for.body ], [ undef, %for.cond.cleanup56.for.body41_crit_edge ]
 %conv42 = sitofp i32 %dx.0838 to float
 %add43 = fadd fast float %conv42, %add29
 %conv44 = fptosi float %add43 to i32
 %conv48 = sitofp i32 %conv44 to float
 %mul49 = fmul fast float %5, %conv48
 %add53 = fadd fast float %mul49, 0.000000e+00
 %conv111 = fptosi float undef to i32
 %cond = select i1 undef, i32 %conv111, i32 -16
 %cond.add116 = select i1 undef, i32 %cond, i32 %add116
 %cmp132 = icmp sgt i32 undef, -16
 %cond137 = select i1 %cmp132, i32 undef, i32 -16
 %cond153 = select i1 undef, i32 %cond137, i32 undef
 %add.ptr = getelementptr inbounds i8, ptr %sdata, i32 %cond153
 %mul154 = mul nsw i32 %cond.add116, %siW
 %add.ptr155 = getelementptr inbounds i8, ptr %add.ptr, i32 %mul154
 %6 = load i8, ptr %add.ptr155, align 1
 %conv157 = uitofp i8 %6 to float
 %incdec.ptr = getelementptr inbounds float, ptr %sFVData.0840, i32 1
 store float %conv157, ptr %sFVData.0840, align 4
 %7 = load float, ptr %arrayidx15, align 4
 %mul65.1 = fmul fast float %7, %conv64.1
 %8 = load float, ptr %arrayidx20, align 4
 %add69.1 = fadd fast float %mul65.1, %8
 %conv78.1 = fdiv fast float 1.000000e+00, 0.000000e+00
 %9 = load float, ptr undef, align 4
 %mul80.1 = fmul fast float %9, %add53
 %10 = load float, ptr undef, align 4
 %mul82.1 = fmul fast float %10, %add69.1
 %add83.1 = fadd fast float %mul82.1, %mul80.1
 %11 = load float, ptr %arrayidx84, align 4
 %add85.1 = fadd fast float %add83.1, %11
 %mul86.1 = fmul fast float %add85.1, %conv78.1
 %12 = load float, ptr %arrayidx92, align 4
 %add93.1 = fadd fast float 0.000000e+00, %12
 %mul94.1 = fmul fast float %add93.1, %conv78.1
 %13 = load float, ptr %arrayidx24, align 4
 %mul98.1 = fmul fast float %mul86.1, %13
 %14 = load float, ptr %arrayidx28, align 4
 %add102.1 = fadd fast float %mul98.1, %14
 %15 = load float, ptr %arrayidx32, align 4
 %mul106.1 = fmul fast float %mul94.1, %15
 %16 = load float, ptr %arrayidx36, align 4
 %add110.1 = fadd fast float %mul106.1, %16
 %conv111.1 = fptosi float %add102.1 to i32
 %conv112.1 = fptosi float %add110.1 to i32
 %cond.1 = select i1 undef, i32 %conv111.1, i32 -16
 %cond.add116.1 = select i1 undef, i32 %cond.1, i32 %add116
 %cond137.1 = select i1 undef, i32 %conv112.1, i32 -16
 %cond153.1 = select i1 undef, i32 %cond137.1, i32 undef
 %add.ptr.1 = getelementptr inbounds i8, ptr %sdata, i32 %cond153.1
 %mul154.1 = mul nsw i32 %cond.add116.1, %siW
 %add.ptr155.1 = getelementptr inbounds i8, ptr %add.ptr.1, i32 %mul154.1
 %17 = load i8, ptr %add.ptr155.1, align 1
 %conv157.1 = uitofp i8 %17 to float
 %incdec.ptr.1 = getelementptr inbounds float, ptr %sFVData.0840, i32 2
 store float %conv157.1, ptr %incdec.ptr, align 4
 %conv112.2 = fptosi float undef to i32
 %cond137.2 = select i1 undef, i32 %conv112.2, i32 -16
 %cond153.2 = select i1 undef, i32 %cond137.2, i32 undef
 %add.ptr.2 = getelementptr inbounds i8, ptr %sdata, i32 %cond153.2
 %18 = load i8, ptr %add.ptr.2, align 1
 %conv157.2 = uitofp i8 %18 to float
 %incdec.ptr.2 = getelementptr inbounds float, ptr %sFVData.0840, i32 3
 store float %conv157.2, ptr %incdec.ptr.1, align 4
 %cmp132.3 = icmp sgt i32 undef, -16
 %cond137.3 = select i1 %cmp132.3, i32 undef, i32 -16
 %cond153.3 = select i1 undef, i32 %cond137.3, i32 undef
 %add.ptr.3 = getelementptr inbounds i8, ptr %sdata, i32 %cond153.3
 %19 = load i8, ptr %add.ptr.3, align 1
 %conv157.3 = uitofp i8 %19 to float
 store float %conv157.3, ptr %incdec.ptr.2, align 4
 %incdec.ptr.5 = getelementptr inbounds float, ptr %sFVData.0840, i32 6
 %20 = load float, ptr %arrayidx15, align 4
 %mul65.6 = fmul fast float %20, %conv64.6
 %21 = load float, ptr %arrayidx20, align 4
 %add69.6 = fadd fast float %mul65.6, %21
 %conv78.6 = fdiv fast float 1.000000e+00, 0.000000e+00
 %22 = load float, ptr undef, align 4
 %mul82.6 = fmul fast float %22, %add69.6
 %add83.6 = fadd fast float %mul82.6, 0.000000e+00
 %23 = load float, ptr %arrayidx84, align 4
 %add85.6 = fadd fast float %add83.6, %23
 %mul86.6 = fmul fast float %add85.6, %conv78.6
 %24 = load float, ptr %arrayidx24, align 4
 %mul98.6 = fmul fast float %mul86.6, %24
 %25 = load float, ptr %arrayidx28, align 4
 %add102.6 = fadd fast float %mul98.6, %25
 %conv111.6 = fptosi float %add102.6 to i32
 %conv112.6 = fptosi float undef to i32
 %cond.6 = select i1 undef, i32 %conv111.6, i32 -16
 %cond.add116.6 = select i1 undef, i32 %cond.6, i32 %add116
 %cmp132.6 = icmp sgt i32 %conv112.6, -16
 %cond137.6 = select i1 %cmp132.6, i32 %conv112.6, i32 -16
 %cond153.6 = select i1 undef, i32 %cond137.6, i32 undef
 %add.ptr.6 = getelementptr inbounds i8, ptr %sdata, i32 %cond153.6
 %mul154.6 = mul nsw i32 %cond.add116.6, %siW
 %add.ptr155.6 = getelementptr inbounds i8, ptr %add.ptr.6, i32 %mul154.6
 %26 = load i8, ptr %add.ptr155.6, align 1
 %conv157.6 = uitofp i8 %26 to float
 store float %conv157.6, ptr %incdec.ptr.5, align 4
 %exitcond874 = icmp eq i32 %dx.0838, 3
 br i1 %exitcond874, label %for.cond.cleanup40, label %for.cond.cleanup56.for.body41_crit_edge

for.cond.cleanup56.for.body41_crit_edge:     ; preds = %for.body41
 %.pre = load float, ptr %arrayidx8, align 4
 br label %for.body41

if.then343:                    ; preds = %entry
 ret void
}

attributes #0 = { sspstrong uwtable "frame-pointer"="none" "target-cpu"="cortex-a7" }

!1 = !{!2, !2, i64 0}
!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!3, !3, i64 0}
