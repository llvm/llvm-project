; RUN: llc < %s -mcpu=cortex-a57 -aarch64-a57-fp-load-balancing-override=1 -aarch64-a57-fp-load-balancing-force-all -enable-misched=false -enable-post-misched=false | FileCheck %s --check-prefix CHECK --check-prefix CHECK-BALFP --check-prefix CHECK-EVEN
; RUN: llc < %s -mcpu=cortex-a57 -aarch64-a57-fp-load-balancing-override=2 -aarch64-a57-fp-load-balancing-force-all -enable-misched=false -enable-post-misched=false | FileCheck %s --check-prefix CHECK --check-prefix CHECK-BALFP --check-prefix CHECK-ODD
; RUN: llc < %s -mcpu=cortex-a53 -aarch64-a57-fp-load-balancing-override=1 -aarch64-a57-fp-load-balancing-force-all -enable-misched=false -enable-post-misched=false | FileCheck %s --check-prefix CHECK --check-prefix CHECK-A53 --check-prefix CHECK-EVEN
; RUN: llc < %s -mcpu=cortex-a53 -aarch64-a57-fp-load-balancing-override=2 -aarch64-a57-fp-load-balancing-force-all -enable-misched=false -enable-post-misched=false | FileCheck %s --check-prefix CHECK --check-prefix CHECK-A53 --check-prefix CHECK-ODD

; The following tests use the balance-fp-ops feature, and should be independent of
; the target cpu.

; RUN: llc < %s -mtriple=aarch64-linux-gnueabi -mattr=+balance-fp-ops -aarch64-a57-fp-load-balancing-override=1 -aarch64-a57-fp-load-balancing-force-all -enable-misched=false -enable-post-misched=false | FileCheck %s --check-prefix CHECK --check-prefix CHECK-EVEN --check-prefix CHECK-BALFP
; RUN: llc < %s -mtriple=aarch64-linux-gnueabi -mattr=+balance-fp-ops  -aarch64-a57-fp-load-balancing-override=2 -aarch64-a57-fp-load-balancing-force-all -enable-misched=false -enable-post-misched=false | FileCheck %s --check-prefix CHECK --check-prefix CHECK-ODD --check-prefix CHECK-BALFP

; Test the AArch64A57FPLoadBalancing pass. This pass relies heavily on register allocation, so
; our test strategy is to:
;   * Force the pass to always perform register swapping even if the dest register is of the
;     correct color already (-force-all)
;   * Force the pass to ignore all hints it obtained from regalloc (-deterministic-balance),
;     and run it twice, once where it always hints odd, and once where it always hints even.
;
; We then use regex magic to check that in the two cases the register allocation is
; different; this is what gives us the testing coverage and distinguishes cases where
; the pass has done some work versus accidental regalloc.

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; Non-overlapping groups - shouldn't need any changing at all.

; CHECK-LABEL: f1:
; CHECK-EVEN: fmadd [[x:d[0-9]*[02468]]]
; CHECK-ODD: fmadd [[x:d[0-9]*[13579]]]
; CHECK: fmadd [[x]]
; CHECK: fmsub [[x]]
; CHECK: fmadd [[x]]
; CHECK: str [[x]]

define void @f1(double* nocapture readonly %p, double* nocapture %q) #0 {
entry:
  %0 = load double, double* %p, align 8
  %arrayidx1 = getelementptr inbounds double, double* %p, i64 1
  %1 = load double, double* %arrayidx1, align 8
  %arrayidx2 = getelementptr inbounds double, double* %p, i64 2
  %2 = load double, double* %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds double, double* %p, i64 3
  %3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %p, i64 4
  %4 = load double, double* %arrayidx4, align 8
  %mul = fmul fast double %0, %1
  %add = fadd fast double %mul, %4
  %mul5 = fmul fast double %1, %2
  %add6 = fadd fast double %mul5, %add
  %mul7 = fmul fast double %1, %3
  %sub = fsub fast double %add6, %mul7
  %mul8 = fmul fast double %2, %3
  %add9 = fadd fast double %mul8, %sub
  store double %add9, double* %q, align 8
  %arrayidx11 = getelementptr inbounds double, double* %p, i64 5
  %5 = load double, double* %arrayidx11, align 8
  %arrayidx12 = getelementptr inbounds double, double* %p, i64 6
  %6 = load double, double* %arrayidx12, align 8
  %arrayidx13 = getelementptr inbounds double, double* %p, i64 7
  %7 = load double, double* %arrayidx13, align 8
  %mul15 = fmul fast double %6, %7
  %mul16 = fmul fast double %0, %5
  %add17 = fadd fast double %mul16, %mul15
  %mul18 = fmul fast double %5, %6
  %add19 = fadd fast double %mul18, %add17
  %arrayidx20 = getelementptr inbounds double, double* %q, i64 1
  store double %add19, double* %arrayidx20, align 8
  ret void
}

; Overlapping groups - coloring needed.

; CHECK-LABEL: f2:
; CHECK-EVEN: fmadd [[x:d[0-9]*[02468]]]
; CHECK-EVEN: fmul [[y:d[0-9]*[13579]]]
; CHECK-ODD: fmadd [[x:d[0-9]*[13579]]]
; CHECK-ODD: fmul [[y:d[0-9]*[02468]]]
; CHECK: fmadd [[x]]
; CHECK: fmadd [[y]]
; CHECK: fmsub [[x]]
; CHECK: fmadd [[y]]
; CHECK: fmadd [[x]]
; CHECK-BALFP: stp [[x]], [[y]]
; CHECK-A53-DAG: str [[x]]
; CHECK-A53-DAG: str [[y]]

define void @f2(double* nocapture readonly %p, double* nocapture %q) #0 {
entry:
  %0 = load double, double* %p, align 8
  %arrayidx1 = getelementptr inbounds double, double* %p, i64 1
  %1 = load double, double* %arrayidx1, align 8
  %arrayidx2 = getelementptr inbounds double, double* %p, i64 2
  %2 = load double, double* %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds double, double* %p, i64 3
  %3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %p, i64 4
  %4 = load double, double* %arrayidx4, align 8
  %arrayidx5 = getelementptr inbounds double, double* %p, i64 5
  %5 = load double, double* %arrayidx5, align 8
  %arrayidx6 = getelementptr inbounds double, double* %p, i64 6
  %6 = load double, double* %arrayidx6, align 8
  %arrayidx7 = getelementptr inbounds double, double* %p, i64 7
  %7 = load double, double* %arrayidx7, align 8
  %mul = fmul fast double %0, %1
  %add = fadd fast double %mul, %7
  %mul8 = fmul fast double %5, %6
  %mul9 = fmul fast double %1, %2
  %add10 = fadd fast double %mul9, %add
  %mul11 = fmul fast double %3, %4
  %add12 = fadd fast double %mul11, %mul8
  %mul13 = fmul fast double %1, %3
  %sub = fsub fast double %add10, %mul13
  %mul14 = fmul fast double %4, %5
  %add15 = fadd fast double %mul14, %add12
  %mul16 = fmul fast double %2, %3
  %add17 = fadd fast double %mul16, %sub
  store double %add17, double* %q, align 8
  %arrayidx19 = getelementptr inbounds double, double* %q, i64 1
  store double %add15, double* %arrayidx19, align 8
  ret void
}

; Dest register is live on block exit - fixup needed.

; CHECK-LABEL: f3:
; CHECK-EVEN: fmadd [[x:d[0-9]*[02468]]]
; CHECK-ODD: fmadd [[x:d[0-9]*[13579]]]
; CHECK: fmadd [[x]]
; CHECK: fmsub [[x]]
; CHECK: fmadd [[y:d[0-9]+]], {{.*}}, [[x]]
; CHECK: str [[y]]

define void @f3(double* nocapture readonly %p, double* nocapture %q) #0 {
entry:
  %0 = load double, double* %p, align 8
  %arrayidx1 = getelementptr inbounds double, double* %p, i64 1
  %1 = load double, double* %arrayidx1, align 8
  %arrayidx2 = getelementptr inbounds double, double* %p, i64 2
  %2 = load double, double* %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds double, double* %p, i64 3
  %3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %p, i64 4
  %4 = load double, double* %arrayidx4, align 8
  %mul = fmul fast double %0, %1
  %add = fadd fast double %mul, %4
  %mul5 = fmul fast double %1, %2
  %add6 = fadd fast double %mul5, %add
  %mul7 = fmul fast double %1, %3
  %sub = fsub fast double %add6, %mul7
  %mul8 = fmul fast double %2, %3
  %add9 = fadd fast double %mul8, %sub
  %cmp = fcmp oeq double %3, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void bitcast (void (...)* @g to void ()*)() #2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  store double %add9, double* %q, align 8
  ret void
}

declare void @g(...) #1

; Single precision version of f2.

; CHECK-LABEL: f4:
; CHECK-EVEN: fmadd [[x:s[0-9]*[02468]]]
; CHECK-EVEN: fmul [[y:s[0-9]*[13579]]]
; CHECK-ODD: fmadd [[x:s[0-9]*[13579]]]
; CHECK-ODD: fmul [[y:s[0-9]*[02468]]]
; CHECK: fmadd [[x]]
; CHECK: fmadd [[y]]
; CHECK: fmsub [[x]]
; CHECK: fmadd [[y]]
; CHECK: fmadd [[x]]
; CHECK-BALFP: stp [[x]], [[y]]
; CHECK-A53-DAG: str [[x]]
; CHECK-A53-DAG: str [[y]]

define void @f4(float* nocapture readonly %p, float* nocapture %q) #0 {
entry:
  %0 = load float, float* %p, align 4
  %arrayidx1 = getelementptr inbounds float, float* %p, i64 1
  %1 = load float, float* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds float, float* %p, i64 2
  %2 = load float, float* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds float, float* %p, i64 3
  %3 = load float, float* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds float, float* %p, i64 4
  %4 = load float, float* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds float, float* %p, i64 5
  %5 = load float, float* %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds float, float* %p, i64 6
  %6 = load float, float* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds float, float* %p, i64 7
  %7 = load float, float* %arrayidx7, align 4
  %mul = fmul fast float %0, %1
  %add = fadd fast float %mul, %7
  %mul8 = fmul fast float %5, %6
  %mul9 = fmul fast float %1, %2
  %add10 = fadd fast float %mul9, %add
  %mul11 = fmul fast float %3, %4
  %add12 = fadd fast float %mul11, %mul8
  %mul13 = fmul fast float %1, %3
  %sub = fsub fast float %add10, %mul13
  %mul14 = fmul fast float %4, %5
  %add15 = fadd fast float %mul14, %add12
  %mul16 = fmul fast float %2, %3
  %add17 = fadd fast float %mul16, %sub
  store float %add17, float* %q, align 4
  %arrayidx19 = getelementptr inbounds float, float* %q, i64 1
  store float %add15, float* %arrayidx19, align 4
  ret void
}

; Single precision version of f3

; CHECK-LABEL: f5:
; CHECK-EVEN: fmadd [[x:s[0-9]*[02468]]]
; CHECK-ODD: fmadd [[x:s[0-9]*[13579]]]
; CHECK: fmadd [[x]]
; CHECK: fmsub [[x]]
; CHECK: fmadd [[y:s[0-9]+]], {{.*}}, [[x]]
; CHECK: str [[y]]

define void @f5(float* nocapture readonly %p, float* nocapture %q) #0 {
entry:
  %0 = load float, float* %p, align 4
  %arrayidx1 = getelementptr inbounds float, float* %p, i64 1
  %1 = load float, float* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds float, float* %p, i64 2
  %2 = load float, float* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds float, float* %p, i64 3
  %3 = load float, float* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds float, float* %p, i64 4
  %4 = load float, float* %arrayidx4, align 4
  %mul = fmul fast float %0, %1
  %add = fadd fast float %mul, %4
  %mul5 = fmul fast float %1, %2
  %add6 = fadd fast float %mul5, %add
  %mul7 = fmul fast float %1, %3
  %sub = fsub fast float %add6, %mul7
  %mul8 = fmul fast float %2, %3
  %add9 = fadd fast float %mul8, %sub
  %cmp = fcmp oeq float %3, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void bitcast (void (...)* @g to void ()*)() #2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  store float %add9, float* %q, align 4
  ret void
}

; Test that regmask clobbering stops a chain sequence.

; CHECK-LABEL: f6:
; CHECK-EVEN: fmadd [[x:d[0-9]*[02468]]]
; CHECK-ODD: fmadd [[x:d[0-9]*[13579]]]
; CHECK: fmadd [[x]]
; CHECK: fmsub [[x]]
; CHECK: fmadd d0, {{.*}}, [[x]]
; CHECK: bl hh
; CHECK: str d0

define void @f6(double* nocapture readonly %p, double* nocapture %q) #0 {
entry:
  %0 = load double, double* %p, align 8
  %arrayidx1 = getelementptr inbounds double, double* %p, i64 1
  %1 = load double, double* %arrayidx1, align 8
  %arrayidx2 = getelementptr inbounds double, double* %p, i64 2
  %2 = load double, double* %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds double, double* %p, i64 3
  %3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %p, i64 4
  %4 = load double, double* %arrayidx4, align 8
  %mul = fmul fast double %0, %1
  %add = fadd fast double %mul, %4
  %mul5 = fmul fast double %1, %2
  %add6 = fadd fast double %mul5, %add
  %mul7 = fmul fast double %1, %3
  %sub = fsub fast double %add6, %mul7
  %mul8 = fmul fast double %2, %3
  %add9 = fadd fast double %mul8, %sub
  %call = tail call double @hh(double %add9) #2
  store double %call, double* %q, align 8
  ret void
}

declare double @hh(double) #1

; Check that we correctly deal with repeated operands.
; The following testcase creates:
;   %d1 = FADDDrr killed %d0, %d0
; We'll get a crash if we naively look at the first operand, remove it
; from the substitution list then look at the second operand.

; CHECK: fmadd [[x:d[0-9]+]]
; CHECK: fadd d1, [[x]], [[x]]

define void @f7(double* nocapture readonly %p, double* nocapture %q) #0 {
entry:
  %0 = load double, double* %p, align 8
  %arrayidx1 = getelementptr inbounds double, double* %p, i64 1
  %1 = load double, double* %arrayidx1, align 8
  %arrayidx2 = getelementptr inbounds double, double* %p, i64 2
  %2 = load double, double* %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds double, double* %p, i64 3
  %3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %p, i64 4
  %4 = load double, double* %arrayidx4, align 8
  %mul = fmul fast double %0, %1
  %add = fadd fast double %mul, %4
  %mul5 = fmul fast double %1, %2
  %add6 = fadd fast double %mul5, %add
  %mul7 = fmul fast double %1, %3
  %sub = fsub fast double %add6, %mul7
  %mul8 = fmul fast double %2, %3
  %add9 = fadd fast double %mul8, %sub
  %add10 = fadd fast double %add9, %add9
  call void @hhh(double 0.0, double %add10)
  ret void
}

declare void @hhh(double, double)

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

