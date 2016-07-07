; RUN: llc -mtriple=arm64-apple-ios -mcpu=cyclone < %s | FileCheck %s
; rdar://11481771
; rdar://13713797

define void @t1() nounwind ssp {
entry:
; CHECK-LABEL: t1:
; CHECK-NOT: fmov
; CHECK: movi.2d v0, #0000000000000000
; CHECK: movi.2d v1, #0000000000000000
; CHECK: movi.2d v2, #0000000000000000
; CHECK: movi.2d v3, #0000000000000000
  tail call void @bar(double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00) nounwind
  ret void
}

define void @t2() nounwind ssp {
entry:
; CHECK-LABEL: t2:
; CHECK-NOT: mov w0, wzr
; CHECK: mov w0, #0
; CHECK: mov w1, #0
  tail call void @bari(i32 0, i32 0) nounwind
  ret void
}

define void @t3() nounwind ssp {
entry:
; CHECK-LABEL: t3:
; CHECK-NOT: mov x0, xzr
; CHECK: mov x0, #0
; CHECK: mov x1, #0
  tail call void @barl(i64 0, i64 0) nounwind
  ret void
}

define void @t4() nounwind ssp {
; CHECK-LABEL: t4:
; CHECK-NOT: fmov
; CHECK: movi.2d v0, #0000000000000000
; CHECK: movi.2d v1, #0000000000000000
  tail call void @barf(float 0.000000e+00, float 0.000000e+00) nounwind
  ret void
}

declare void @bar(double, double, double, double)
declare void @bari(i32, i32)
declare void @barl(i64, i64)
declare void @barf(float, float)

; We used to produce spills+reloads for a Q register with zero cycle zeroing
; enabled.
; CHECK-LABEL: foo:
; CHECK-NOT: str {{q[0-9]+}}
; CHECK-NOT: ldr {{q[0-9]+}}
define double @foo(i32 %n) {
entry:
  br label %for.body

for.body:
  %phi0 = phi double [ 1.0, %entry ], [ %v0, %for.body ]
  %i.076 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %conv21 = sitofp i32 %i.076 to double
  %call = tail call fast double @sin(double %conv21)
  %cmp.i = fcmp fast olt double %phi0, %call
  %v0 = select i1 %cmp.i, double %call, double %phi0
  %inc = add nuw nsw i32 %i.076, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret double %v0
}

declare double @sin(double)
