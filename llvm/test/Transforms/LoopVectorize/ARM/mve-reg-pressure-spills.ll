; RUN: opt -mcpu=cortex-m55 -passes=loop-vectorize -disable-output -debug-only=loop-vectorize,vplan -vectorizer-consider-reg-pressure=false %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NOPRESSURE
; RUN: opt -mcpu=cortex-m55 -passes=loop-vectorize -disable-output -debug-only=loop-vectorize,vplan -vectorizer-consider-reg-pressure=true %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-PRESSURE

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-unknown-none-eabihf"

; In this function the spills make it not profitable to vectorize if considering
; register pressure.
define void @spills_not_profitable(ptr %in1, ptr %in2, ptr %out, i32 %n) {
; CHECK-LABEL: LV: Checking a loop in 'spills_not_profitable'
; CHECK: LV: Scalar loop costs: 86
; CHECK-NOPRESSURE: Cost for VF 2: 394 (Estimated cost per lane: 197.0)
; CHECK-NOPRESSURE: Cost for VF 4: 338 (Estimated cost per lane: 84.5)
; CHECK-NOPRESSURE: LV: Selecting VF: 4
; CHECK-PRESSURE: LV(REG): Cost of 300 from 25 spills of Generic::VectorRC
; CHECK-PRESSURE-NEXT: Cost for VF 2: 694 (Estimated cost per lane: 347.0)
; CHECK-PRESSURE: LV(REG): Cost of 100 from 25 spills of Generic::VectorRC
; CHECK-PRESSURE-NEXT: Cost for VF 4: 438 (Estimated cost per lane: 109.5)
; CHECK-PRESSURE: LV: Selecting VF: 1
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %exit, label %for.body

for.body:
  %i = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %x4 = phi float [ %x4.next, %for.body ], [ 0.000000e+00, %entry ]
  %x3 = phi float [ %x3.next, %for.body ], [ 0.000000e+00, %entry ]
  %x2 = phi float [ %x2.next, %for.body ], [ 0.000000e+00, %entry ]
  %x1 = phi float [ %x1.next, %for.body ], [ 0.000000e+00, %entry ]
  %x0 = phi float [ %x0.next, %for.body ], [ 0.000000e+00, %entry ]
  %acc7 = phi float [ %acc7.next, %for.body ], [ 0.000000e+00, %entry ]
  %acc6 = phi float [ %acc6.next, %for.body ], [ 0.000000e+00, %entry ]
  %acc5 = phi float [ %acc5.next, %for.body ], [ 0.000000e+00, %entry ]
  %acc4 = phi float [ %acc4.next, %for.body ], [ 0.000000e+00, %entry ]
  %acc3 = phi float [ %acc3.next, %for.body ], [ 0.000000e+00, %entry ]
  %acc2 = phi float [ %acc2.next, %for.body ], [ 0.000000e+00, %entry ]
  %acc1 = phi float [ %acc1.next, %for.body ], [ 0.000000e+00, %entry ]
  %acc0 = phi float [ %acc0.next, %for.body ], [ 0.000000e+00, %entry ]
  %in1.addr = phi ptr [ %in1.addr.next, %for.body ], [ %in1, %entry ]
  %in2.addr = phi ptr [ %in2.addr.next, %for.body ], [ %in2, %entry ]
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %in1.addr, i32 4
  %0 = load float, ptr %in1.addr, align 4
  %incdec.ptr1 = getelementptr inbounds nuw i8, ptr %in2.addr, i32 4
  %1 = load float, ptr %in2.addr, align 4
  %mul = fmul fast float %0, %x0
  %add = fadd fast float %mul, %acc0
  %mul2 = fmul fast float %0, %x1
  %add3 = fadd fast float %mul2, %acc1
  %mul4 = fmul fast float %0, %x2
  %add5 = fadd fast float %mul4, %acc2
  %mul6 = fmul fast float %0, %x3
  %add7 = fadd fast float %mul6, %acc3
  %mul8 = fmul fast float %0, %x4
  %add9 = fadd fast float %mul8, %acc4
  %mul10 = fmul fast float %1, %0
  %add11 = fadd fast float %mul10, %acc7
  %incdec.ptr12 = getelementptr inbounds nuw i8, ptr %in1.addr, i32 8
  %2 = load float, ptr %incdec.ptr, align 4
  %incdec.ptr13 = getelementptr inbounds nuw i8, ptr %in2.addr, i32 8
  %x0.next = load float, ptr %incdec.ptr1, align 4
  %mul14 = fmul fast float %2, %x1
  %add15 = fadd fast float %add, %mul14
  %mul16 = fmul fast float %2, %x2
  %add17 = fadd fast float %add3, %mul16
  %mul18 = fmul fast float %2, %x3
  %add19 = fadd fast float %add5, %mul18
  %mul20 = fmul fast float %2, %x4
  %add21 = fadd fast float %add7, %mul20
  %mul22 = fmul fast float %2, %1
  %add23 = fadd fast float %mul22, %acc6
  %mul24 = fmul fast float %x0.next, %2
  %add25 = fadd fast float %add11, %mul24
  %incdec.ptr26 = getelementptr inbounds nuw i8, ptr %in1.addr, i32 12
  %4 = load float, ptr %incdec.ptr12, align 4
  %incdec.ptr27 = getelementptr inbounds nuw i8, ptr %in2.addr, i32 12
  %x1.next = load float, ptr %incdec.ptr13, align 4
  %mul28 = fmul fast float %4, %x2
  %add29 = fadd fast float %add15, %mul28
  %mul30 = fmul fast float %4, %x3
  %add31 = fadd fast float %add17, %mul30
  %mul32 = fmul fast float %4, %x4
  %add33 = fadd fast float %add19, %mul32
  %mul34 = fmul fast float %4, %1
  %add35 = fadd fast float %mul34, %acc5
  %mul36 = fmul fast float %4, %x0.next
  %add37 = fadd fast float %add23, %mul36
  %mul38 = fmul fast float %x1.next, %4
  %add39 = fadd fast float %add25, %mul38
  %incdec.ptr40 = getelementptr inbounds nuw i8, ptr %in1.addr, i32 16
  %6 = load float, ptr %incdec.ptr26, align 4
  %incdec.ptr41 = getelementptr inbounds nuw i8, ptr %in2.addr, i32 16
  %x2.next = load float, ptr %incdec.ptr27, align 4
  %mul42 = fmul fast float %6, %x3
  %add43 = fadd fast float %add29, %mul42
  %mul44 = fmul fast float %6, %x4
  %acc1.next = fadd fast float %add31, %mul44
  %mul46 = fmul fast float %6, %1
  %add47 = fadd fast float %add9, %mul46
  %mul48 = fmul fast float %6, %x0.next
  %add49 = fadd fast float %add35, %mul48
  %mul50 = fmul fast float %6, %x1.next
  %add51 = fadd fast float %add37, %mul50
  %mul52 = fmul fast float %x2.next, %6
  %add53 = fadd fast float %add39, %mul52
  %incdec.ptr54 = getelementptr inbounds nuw i8, ptr %in1.addr, i32 20
  %8 = load float, ptr %incdec.ptr40, align 4
  %incdec.ptr55 = getelementptr inbounds nuw i8, ptr %in2.addr, i32 20
  %x3.next = load float, ptr %incdec.ptr41, align 4
  %mul56 = fmul fast float %8, %x4
  %acc0.next = fadd fast float %add43, %mul56
  %mul58 = fmul fast float %8, %1
  %add59 = fadd fast float %add21, %mul58
  %mul60 = fmul fast float %8, %x0.next
  %add61 = fadd fast float %add47, %mul60
  %mul62 = fmul fast float %8, %x1.next
  %add63 = fadd fast float %add49, %mul62
  %mul64 = fmul fast float %8, %x2.next
  %add65 = fadd fast float %add51, %mul64
  %mul66 = fmul fast float %x3.next, %8
  %add67 = fadd fast float %add53, %mul66
  %in1.addr.next = getelementptr inbounds nuw i8, ptr %in1.addr, i32 24
  %10 = load float, ptr %incdec.ptr54, align 4
  %in2.addr.next = getelementptr inbounds nuw i8, ptr %in2.addr, i32 24
  %x4.next = load float, ptr %incdec.ptr55, align 4
  %mul70 = fmul fast float %10, %1
  %acc2.next = fadd fast float %add33, %mul70
  %mul72 = fmul fast float %10, %x0.next
  %acc3.next = fadd fast float %add59, %mul72
  %mul74 = fmul fast float %10, %x1.next
  %acc4.next = fadd fast float %add61, %mul74
  %mul76 = fmul fast float %10, %x2.next
  %acc5.next = fadd fast float %add63, %mul76
  %mul78 = fmul fast float %10, %x3.next
  %acc6.next = fadd fast float %add65, %mul78
  %mul80 = fmul fast float %x4.next, %10
  %acc7.next = fadd fast float %add67, %mul80
  %inc = add nuw i32 %i, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  %acc0.exit = phi float [ 0.000000e+00, %entry ], [ %acc0.next, %for.body ]
  %acc1.exit = phi float [ 0.000000e+00, %entry ], [ %acc1.next, %for.body ]
  %acc2.exit = phi float [ 0.000000e+00, %entry ], [ %acc2.next, %for.body ]
  %acc3.exit = phi float [ 0.000000e+00, %entry ], [ %acc3.next, %for.body ]
  %acc4.exit = phi float [ 0.000000e+00, %entry ], [ %acc4.next, %for.body ]
  %acc5.exit = phi float [ 0.000000e+00, %entry ], [ %acc5.next, %for.body ]
  %acc6.exit = phi float [ 0.000000e+00, %entry ], [ %acc6.next, %for.body ]
  %acc7.exit = phi float [ 0.000000e+00, %entry ], [ %acc7.next, %for.body ]
  store float %acc0.exit, ptr %out, align 4
  %arrayidx82 = getelementptr inbounds nuw i8, ptr %out, i32 4
  store float %acc1.exit, ptr %arrayidx82, align 4
  %arrayidx83 = getelementptr inbounds nuw i8, ptr %out, i32 8
  store float %acc2.exit, ptr %arrayidx83, align 4
  %arrayidx84 = getelementptr inbounds nuw i8, ptr %out, i32 12
  store float %acc3.exit, ptr %arrayidx84, align 4
  %arrayidx85 = getelementptr inbounds nuw i8, ptr %out, i32 16
  store float %acc4.exit, ptr %arrayidx85, align 4
  %arrayidx86 = getelementptr inbounds nuw i8, ptr %out, i32 20
  store float %acc5.exit, ptr %arrayidx86, align 4
  %arrayidx87 = getelementptr inbounds nuw i8, ptr %out, i32 24
  store float %acc6.exit, ptr %arrayidx87, align 4
  %arrayidx88 = getelementptr inbounds nuw i8, ptr %out, i32 28
  store float %acc7.exit, ptr %arrayidx88, align 4
  ret void
}

; In this function we have spills but it is still profitable to vectorize when
; considering register pressure.
define void @spills_profitable(ptr %in1, ptr %in2, ptr %out, i32 %n, i32 %m) {
; CHECK-LABEL: LV: Checking a loop in 'spills_profitable'
; CHECK: LV: Scalar loop costs: 54
; CHECK-NOPRESSURE: Cost for VF 2: 1530 (Estimated cost per lane: 765.0)
; CHECK-NOPRESSURE: Cost for VF 4: 38 (Estimated cost per lane: 9.5)
; CHECK-PRESSURE: LV(REG): Cost of 8 from 2 spills of Generic::ScalarRC
; CHECK-PRESSURE-NEXT: Cost for VF 2: 1538 (Estimated cost per lane: 769.0)
; CHECK-PRESSURE: LV(REG): Cost of 24 from 3 spills of Generic::VectorRC
; CHECK-PRESSURE-NEXT: Cost for VF 4: 62 (Estimated cost per lane: 15.5)
; CHECK: LV: Selecting VF: 4
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %exit, label %for.body.preheader

for.body.preheader:
  %add.ptr3.idx = mul i32 %m, 12
  %add.ptr3 = getelementptr inbounds nuw i8, ptr %in1, i32 %add.ptr3.idx
  %add.ptr1.idx = shl i32 %m, 3
  %add.ptr1 = getelementptr inbounds nuw i8, ptr %in1, i32 %add.ptr1.idx
  %add.ptr = getelementptr inbounds nuw i32, ptr %in1, i32 %m
  br label %for.body

for.body:
  %i = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %acc3 = phi i64 [ %acc3.next, %for.body ], [ 0, %for.body.preheader ]
  %acc2 = phi i64 [ %acc2.next, %for.body ], [ 0, %for.body.preheader ]
  %acc1 = phi i64 [ %acc1.next, %for.body ], [ 0, %for.body.preheader ]
  %acc0 = phi i64 [ %acc0.next, %for.body ], [ 0, %for.body.preheader ]
  %in2.addr = phi ptr [ %in2.addr.next, %for.body ], [ %in2, %for.body.preheader ]
  %px3 = phi ptr [ %px3.next, %for.body ], [ %add.ptr3, %for.body.preheader ]
  %px2 = phi ptr [ %px2.next, %for.body ], [ %add.ptr1, %for.body.preheader ]
  %px1 = phi ptr [ %px1.next, %for.body ], [ %add.ptr, %for.body.preheader ]
  %px0 = phi ptr [ %px0.next, %for.body ], [ %in1, %for.body.preheader ]
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %in2.addr, i32 4
  %0 = load i32, ptr %in2.addr, align 4
  %incdec.ptr4 = getelementptr inbounds nuw i8, ptr %px0, i32 4
  %1 = load i32, ptr %px0, align 4
  %incdec.ptr5 = getelementptr inbounds nuw i8, ptr %px1, i32 4
  %2 = load i32, ptr %px1, align 4
  %incdec.ptr6 = getelementptr inbounds nuw i8, ptr %px2, i32 4
  %3 = load i32, ptr %px2, align 4
  %incdec.ptr7 = getelementptr inbounds nuw i8, ptr %px3, i32 4
  %4 = load i32, ptr %px3, align 4
  %conv = sext i32 %1 to i64
  %conv8 = sext i32 %0 to i64
  %mul9 = mul nsw i64 %conv, %conv8
  %add = add nsw i64 %mul9, %acc0
  %conv10 = sext i32 %2 to i64
  %mul12 = mul nsw i64 %conv10, %conv8
  %add13 = add nsw i64 %mul12, %acc1
  %conv14 = sext i32 %3 to i64
  %mul16 = mul nsw i64 %conv14, %conv8
  %add17 = add nsw i64 %mul16, %acc2
  %conv18 = sext i32 %4 to i64
  %mul20 = mul nsw i64 %conv18, %conv8
  %add21 = add nsw i64 %mul20, %acc3
  %in2.addr.next = getelementptr inbounds nuw i8, ptr %in2.addr, i32 8
  %5 = load i32, ptr %incdec.ptr, align 4
  %px0.next = getelementptr inbounds nuw i8, ptr %px0, i32 8
  %6 = load i32, ptr %incdec.ptr4, align 4
  %px1.next = getelementptr inbounds nuw i8, ptr %px1, i32 8
  %7 = load i32, ptr %incdec.ptr5, align 4
  %px2.next = getelementptr inbounds nuw i8, ptr %px2, i32 8
  %8 = load i32, ptr %incdec.ptr6, align 4
  %px3.next = getelementptr inbounds nuw i8, ptr %px3, i32 8
  %9 = load i32, ptr %incdec.ptr7, align 4
  %conv27 = sext i32 %6 to i64
  %conv28 = sext i32 %5 to i64
  %mul29 = mul nsw i64 %conv27, %conv28
  %acc0.next = add nsw i64 %add, %mul29
  %conv31 = sext i32 %7 to i64
  %mul33 = mul nsw i64 %conv31, %conv28
  %acc1.next = add nsw i64 %add13, %mul33
  %conv35 = sext i32 %8 to i64
  %mul37 = mul nsw i64 %conv35, %conv28
  %acc2.next = add nsw i64 %add17, %mul37
  %conv39 = sext i32 %9 to i64
  %mul41 = mul nsw i64 %conv39, %conv28
  %acc3.next = add nsw i64 %add21, %mul41
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  %acc0.exit = phi i64 [ 0, %entry ], [ %acc0.next, %for.body ]
  %acc1.exit = phi i64 [ 0, %entry ], [ %acc1.next, %for.body ]
  %acc2.exit = phi i64 [ 0, %entry ], [ %acc2.next, %for.body ]
  %acc3.exit = phi i64 [ 0, %entry ], [ %acc3.next, %for.body ]
  store i64 %acc0.exit, ptr %out, align 8
  %arrayidx43 = getelementptr inbounds nuw i8, ptr %out, i32 8
  store i64 %acc1.exit, ptr %arrayidx43, align 8
  %arrayidx44 = getelementptr inbounds nuw i8, ptr %out, i32 16
  store i64 %acc2.exit, ptr %arrayidx44, align 8
  %arrayidx45 = getelementptr inbounds nuw i8, ptr %out, i32 24
  store i64 %acc3.exit, ptr %arrayidx45, align 8
  ret void
}
