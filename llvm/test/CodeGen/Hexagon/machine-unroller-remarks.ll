; RUN: llc -march=hexagon -O3 -enable-machine-unroller \
; RUN: -pass-remarks=machine-unroller -pass-remarks-missed=machine-unroller \
; RUN: < %s 2>&1 | FileCheck %s
; CHECK: remark: {{.*}}: Unrolled loop by factor 2 (ResMII improved from {{[0-9]+}} to {{[0-9]+}})

; RUN: llc -march=hexagon -O3 -enable-machine-unroller \
; RUN: -machine-unroller-threshold=1 -pass-remarks=machine-unroller \
; RUN: -pass-remarks-missed=machine-unroller < %s 2>&1 |\
; RUN:  FileCheck %s --check-prefix=MISSED
; MISSED: remark: {{.*}}: Unable to unroll loop by factor 2: unrolled size {{[0-9]+}} exceeds threshold 1
; MISSED: remark: {{.*}}: Unable to unroll loop by factor 4: unrolled size {{[0-9]+}} exceeds threshold 1

define float @test(i32 %n, float %da, float* noalias nocapture readonly %dx, i32 %incx, float* noalias nocapture %dy, i32 %incy) local_unnamed_addr {
entry:
  %cmp = icmp slt i32 %n, 1
  %cmp1 = fcmp oeq float %da, 0.000000e+00
  %or.cond45 = or i1 %cmp, %cmp1
  br i1 %or.cond45, label %if.then6, label %if.end3

if.end3:
  %cmp4 = icmp ne i32 %incx, 1
  %cmp5 = icmp ne i32 %incy, 1
  %or.cond = or i1 %cmp4, %cmp5
  br i1 %or.cond, label %if.then6, label %for.body.lr.ph

if.then6:
  ret float 0.000000e+00

for.body.lr.ph:
  %0 = load float, float* %dy, align 4
  br label %for.body

for.body:
  %arrayidx18.phi = phi float* [ %dx, %for.body.lr.ph ], [ %arrayidx18.inc, %for.body ]
  %arrayidx21.phi = phi float* [ %dy, %for.body.lr.ph ], [ %arrayidx21.inc, %for.body ]
  %i.047 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %1 = load float, float* %arrayidx18.phi, align 4
  %mul19 = fmul float %1, %da
  %add20 = fadd float %0, %mul19
  store float %add20, float* %arrayidx21.phi, align 4
  %inc = add nuw nsw i32 %i.047, 1
  %exitcond = icmp eq i32 %inc, %n
  %arrayidx18.inc = getelementptr float, float* %arrayidx18.phi, i32 32
  %arrayidx21.inc = getelementptr float, float* %arrayidx21.phi, i32 32
  br i1 %exitcond, label %if.then6, label %for.body
}


