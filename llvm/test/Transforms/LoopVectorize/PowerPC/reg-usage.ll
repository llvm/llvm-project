; RUN: opt < %s -debug-only=loop-vectorize -passes='function(loop-vectorize),default<O2>' -vectorizer-maximize-bandwidth -mtriple=powerpc64-unknown-linux -S -mcpu=pwr8 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-PWR8
; RUN: opt < %s -debug-only=loop-vectorize -passes='function(loop-vectorize),default<O2>' -vectorizer-maximize-bandwidth -mtriple=powerpc64le-unknown-linux -S -mcpu=pwr9 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-PWR9
; REQUIRES: asserts

@a = global [1024 x i8] zeroinitializer, align 16
@b = global [1024 x i8] zeroinitializer, align 16

define i32 @foo() {
; CHECK-LABEL: foo

; CHECK-PWR8: Executing best plan with VF=16, UF=4

; CHECK-PWR9: Executing best plan with VF=8, UF=8


entry:
  br label %for.body

for.cond.cleanup:
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %s.015 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds [1024 x i8], ptr @b, i64 0, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  %sub = sub nsw i32 %conv, %conv3
  %ispos = icmp sgt i32 %sub, -1
  %neg = sub nsw i32 0, %sub
  %2 = select i1 %ispos, i32 %sub, i32 %neg
  %add = add nsw i32 %2, %s.015
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define i32 @goo() {
; For indvars.iv used in a computating chain only feeding into getelementptr or cmp,
; it will not have vector version and the vector register usage will not exceed the
; available vector register number.

; CHECK-LABEL: goo

; CHECK: Executing best plan with VF=16, UF=4

entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %s.015 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %tmp1 = add nsw i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %tmp1
  %tmp = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %tmp to i32
  %tmp2 = add nsw i64 %indvars.iv, 2
  %arrayidx2 = getelementptr inbounds [1024 x i8], ptr @b, i64 0, i64 %tmp2
  %tmp3 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %tmp3 to i32
  %sub = sub nsw i32 %conv, %conv3
  %ispos = icmp sgt i32 %sub, -1
  %neg = sub nsw i32 0, %sub
  %tmp4 = select i1 %ispos, i32 %sub, i32 %neg
  %add = add nsw i32 %tmp4, %s.015
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define i64 @bar(ptr nocapture %a) {
; CHECK-LABEL: bar

; CHECK: Executing best plan with VF=2, UF=12

entry:
  br label %for.body

for.cond.cleanup:
  %add2.lcssa = phi i64 [ %add2, %for.body ]
  ret i64 %add2.lcssa

for.body:
  %i.012 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi i64 [ 0, %entry ], [ %add2, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %i.012
  %0 = load i64, ptr %arrayidx, align 8
  %add = add nsw i64 %0, %i.012
  store i64 %add, ptr %arrayidx, align 8
  %add2 = add nsw i64 %add, %s.011
  %inc = add nuw nsw i64 %i.012, 1
  %exitcond = icmp eq i64 %inc, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

@d = external global [0 x i64], align 8
@e = external global [0 x i32], align 4
@c = external global [0 x i32], align 4

define void @hoo(i32 %n) {
; CHECK-LABEL: hoo
; CHECK: Executing best plan with VF=1, UF=12

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [0 x i64], ptr @d, i64 0, i64 %indvars.iv
  %tmp = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds [0 x i32], ptr @e, i64 0, i64 %tmp
  %tmp1 = load i32, ptr %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds [0 x i32], ptr @c, i64 0, i64 %indvars.iv
  store i32 %tmp1, ptr %arrayidx3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define float @float_(ptr nocapture readonly %a, ptr nocapture readonly %b, i32 %n) {
;CHECK-LABEL: float_
;CHECK: LV(REG): VF = 1
;CHECK: LV(REG): Found max usage: 2 item
;CHECK-NEXT: LV(REG): RegisterClass: PPC::GPRRC, 2 registers
;CHECK-NEXT: LV(REG): RegisterClass: PPC::VSXRC, 3 registers
;CHECK: LV(REG): Found invariant usage: 1 item
;CHECK-NEXT: LV(REG): RegisterClass: PPC::GPRRC, 1 registers

entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %preheader, label %for.end

preheader:
  %t0 = sext i32 %n to i64
  br label %for

for:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %for ]
  %s.02 = phi float [ 0.0, %preheader ], [ %add4, %for ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %t1 = load float, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %t2 = load float, ptr %arrayidx3, align 4
  %add = fadd fast float %t1, %s.02
  %add4 = fadd fast float %add, %t2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 32
  %cmp1 = icmp slt i64 %indvars.iv.next, %t0
  br i1 %cmp1, label %for, label %loopexit

loopexit:
  %add4.lcssa = phi float [ %add4, %for ]
  br label %for.end

for.end:
  %s.0.lcssa = phi float [ 0.0, %entry ], [ %add4.lcssa, %loopexit ]
  ret float %s.0.lcssa
}


define void @double_(ptr nocapture %A, i32 %n) nounwind uwtable ssp {
;CHECK-LABEL: double_
;CHECK-PWR8: LV(REG): VF = 2
;CHECK-PWR8: LV(REG): Found max usage: 2 item
;CHECK-PWR8-NEXT: LV(REG): RegisterClass: PPC::GPRRC, 2 registers
;CHECK-PWR8-NEXT: LV(REG): RegisterClass: PPC::VSXRC, 5 registers
;CHECK-PWR8: LV(REG): Found invariant usage: 1 item
;CHECK-PWR8-NEXT: LV(REG): RegisterClass: PPC::GPRRC, 1 registers

;CHECK-PWR9: LV(REG): VF = 1
;CHECK-PWR9: LV(REG): Found max usage: 2 item
;CHECK-PWR9-NEXT: LV(REG): RegisterClass: PPC::GPRRC, 2 registers
;CHECK-PWR9-NEXT: LV(REG): RegisterClass: PPC::VSXRC, 5 registers
;CHECK-PWR9: LV(REG): Found invariant usage: 1 item
;CHECK-PWR9-NEXT: LV(REG): RegisterClass: PPC::GPRRC, 1 registers

  %1 = sext i32 %n to i64
  br label %2

; <label>:2                                       ; preds = %2, %0
  %indvars.iv = phi i64 [ %indvars.iv.next, %2 ], [ %1, %0 ]
  %3 = getelementptr inbounds double, ptr %A, i64 %indvars.iv
  %4 = load double, ptr %3, align 8
  %5 = fadd double %4, 3.000000e+00
  %6 = fmul double %4, 2.000000e+00
  %7 = fadd double %5, %6
  %8 = fadd double %7, 2.000000e+00
  %9 = fmul double %8, 5.000000e-01
  %10 = fadd double %6, %9
  %11 = fsub double %10, %5
  %12 = fadd double %4, %11
  %13 = fdiv double %8, %12
  %14 = fmul double %13, %8
  %15 = fmul double %6, %14
  %16 = fmul double %5, %15
  %17 = fadd double %16, -3.000000e+00
  %18 = fsub double %4, %5
  %19 = fadd double %6, %18
  %20 = fadd double %13, %19
  %21 = fadd double %20, %17
  %22 = fadd double %21, 3.000000e+00
  %23 = fmul double %4, %22
  store double %23, ptr %3, align 8
  %indvars.iv.next = add i64 %indvars.iv, -1
  %24 = trunc i64 %indvars.iv to i32
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %26, label %2

; <label>:26                                      ; preds = %2
  ret void
}

define ppc_fp128 @fp128_(ptr nocapture %n, ppc_fp128 %d) nounwind readonly {
;CHECK-LABEL: fp128_
;CHECK: LV(REG): VF = 1
;CHECK: LV(REG): Found max usage: 2 item
;CHECK: LV(REG): RegisterClass: PPC::GPRRC, 2 registers
;CHECK: LV(REG): RegisterClass: PPC::VRRC, 2 registers
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi ppc_fp128 [ %d, %entry ], [ %sub, %for.body ]
  %arrayidx = getelementptr inbounds ppc_fp128, ptr %n, i32 %i.06
  %0 = load ppc_fp128, ptr %arrayidx, align 8
  %sub = fsub fast ppc_fp128 %x.05, %0
  %inc = add nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret ppc_fp128 %sub
}


define void @fp16_(ptr nocapture readonly %pIn, ptr nocapture %pOut, i32 %numRows, i32 %numCols, i32 %scale.coerce) #0 {
;CHECK-LABEL: fp16_
;CHECK: LV(REG): VF = 1
;CHECK: LV(REG): Found max usage: 2 item
;CHECK: LV(REG): RegisterClass: PPC::GPRRC, 4 registers
;CHECK: LV(REG): RegisterClass: PPC::VSXRC, 2 registers
entry:
  %tmp.0.extract.trunc = trunc i32 %scale.coerce to i16
  %0 = bitcast i16 %tmp.0.extract.trunc to half
  %mul = mul i32 %numCols, %numRows
  %shr = lshr i32 %mul, 2
  %cmp26 = icmp eq i32 %shr, 0
  br i1 %cmp26, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %pIn.addr.029 = phi ptr [ %add.ptr, %while.body ], [ %pIn, %entry ]
  %pOut.addr.028 = phi ptr [ %add.ptr7, %while.body ], [ %pOut, %entry ]
  %blkCnt.027 = phi i32 [ %dec, %while.body ], [ %shr, %entry ]
  %1 = load half, ptr %pIn.addr.029, align 2
  %arrayidx2 = getelementptr inbounds half, ptr %pIn.addr.029, i32 1
  %2 = load half, ptr %arrayidx2, align 2
  %mul3 = fmul half %1, %0
  %mul4 = fmul half %2, %0
  store half %mul3, ptr %pOut.addr.028, align 2
  %arrayidx6 = getelementptr inbounds half, ptr %pOut.addr.028, i32 1
  store half %mul4, ptr %arrayidx6, align 2
  %add.ptr = getelementptr inbounds half, ptr %pIn.addr.029, i32 2
  %add.ptr7 = getelementptr inbounds half, ptr %pOut.addr.028, i32 2
  %dec = add nsw i32 %blkCnt.027, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}
