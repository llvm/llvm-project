; RUN: llc -mtriple=hexagon-unknown-elf < %s | FileCheck %s

; CHECK-LABEL: ldf
; CHECK: memw(r{{[0-9]+}}++#4)
; CHECK: memw(r{{[0-9]+}}++#4)
define float @ldf(float* nocapture readonly %x, float* nocapture readonly %y) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:
  %arrayidx.phi = phi float* [ %x, %entry ], [ %arrayidx.inc, %for.body ]
  %arrayidx1.phi = phi float* [ %y, %entry ], [ %arrayidx1.inc, %for.body ]
  %i.09 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %acc.08 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %0 = load float, float* %arrayidx.phi, align 4
  %1 = load float, float* %arrayidx1.phi, align 4
  %mul = fmul contract float %0, %1
  %add = fadd contract float %acc.08, %mul
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, 1024
  %arrayidx.inc = getelementptr float, float* %arrayidx.phi, i32 1
  %arrayidx1.inc = getelementptr float, float* %arrayidx1.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %add
}

; CHECK-LABEL: ldd
; CHECK: memd(r{{[0-9]+}}++#8)
; CHECK: memd(r{{[0-9]+}}++#8)
define double @ldd(double* nocapture readonly %x, double* nocapture readonly %y) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:
  %arrayidx.phi = phi double* [ %x, %entry ], [ %arrayidx.inc, %for.body ]
  %arrayidx1.phi = phi double* [ %y, %entry ], [ %arrayidx1.inc, %for.body ]
  %i.09 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %acc.08 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %0 = load double, double* %arrayidx.phi, align 8
  %1 = load double, double* %arrayidx1.phi, align 8
  %mul = fmul contract double %0, %1
  %add = fadd contract double %acc.08, %mul
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, 1024
  %arrayidx.inc = getelementptr double, double* %arrayidx.phi, i32 1
  %arrayidx1.inc = getelementptr double, double* %arrayidx1.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret double %add
}

; CHECK-LABEL: stf
; CHECK: memw(r{{[0-9]+}}++#4)
define double* @stf(float* returned %p) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:
  %arrayidx.phi = phi float* [ %arrayidx.inc, %for.body ], [ %p, %entry ]
  %call = tail call float @foof() #2
  store float %call, float* %arrayidx.phi, align 8
  %arrayidx.inc = getelementptr float, float* %arrayidx.phi, i32 1
  br label %for.body
}

declare float @foof() local_unnamed_addr #1

; CHECK-LABEL: std
; CHECK: memd(r{{[0-9]+}}++#8)
define double* @std(double* returned %p) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:
  %arrayidx.phi = phi double* [ %arrayidx.inc, %for.body ], [ %p, %entry ]
  %call = tail call double @food() #2
  store double %call, double* %arrayidx.phi, align 8
  %arrayidx.inc = getelementptr double, double* %arrayidx.phi, i32 1
  br label %for.body
}

declare double @food() local_unnamed_addr #1

