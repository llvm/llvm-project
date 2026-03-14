; RUN: llc -O3 -mtriple=thumbv7em -mcpu=cortex-m4 %s -o - | FileCheck %s --check-prefix=CHECK-REG-PRESSURE
; RUN: llc -O3 -mtriple=thumbv7eb %s -o - | FileCheck %s --check-prefix=CHECK-UNSUPPORTED
; RUN: llc -O3 -mtriple=thumbv8m.main -mattr=+dsp -arm-parallel-dsp-load-limit=20 %s -o - | FileCheck %s --check-prefix=CHECK

; CHECK-UNSUPPORTED-LABEL: unroll_n_jam_smlad
; CHECK-UNSUPPORTED-NOT: smlad r{{.}}

; Test that the duplicate loads are removed, which allows parallel dsp to find
; the parallel operations.

; CHECK-LABEL: unroll_n_jam_smlad
define void @unroll_n_jam_smlad(ptr %res, ptr %A, ptr %B, i32 %N, i32 %idx) {
entry:
  %xtraiter306.i = and i32 %N, 3
  %unroll_iter310.i = sub i32 %N, %xtraiter306.i
  %arrayidx.us.i117.i = getelementptr inbounds i32, ptr %res, i32 %idx
  store i32 0, ptr %arrayidx.us.i117.i, align 4
  %mul.us.i118.i = mul i32 %idx, %N
  %inc11.us.i.i = or disjoint i32 %idx, 1
  %arrayidx.us.i117.1.i = getelementptr inbounds i32, ptr %res, i32 %inc11.us.i.i
  store i32 0, ptr %arrayidx.us.i117.1.i, align 4
  %mul.us.i118.1.i = mul i32 %inc11.us.i.i, %N
  %inc11.us.i.1.i = or disjoint i32 %idx, 2
  %arrayidx.us.i117.2.i = getelementptr inbounds i32, ptr %res, i32 %inc11.us.i.1.i
  store i32 0, ptr %arrayidx.us.i117.2.i, align 4
  %mul.us.i118.2.i = mul i32 %inc11.us.i.1.i, %N
  %inc11.us.i.2.i = or disjoint i32 %idx, 3
  %arrayidx.us.i117.3.i = getelementptr inbounds i32, ptr %res, i32 %inc11.us.i.2.i
  store i32 0, ptr %arrayidx.us.i117.3.i, align 4
  %mul.us.i118.3.i = mul i32 %inc11.us.i.2.i, %N
  %inc11.us.i.3.i = add i32 %idx, 4
  br label %for.body

; TODO: CSE, or something similar, is required to remove the duplicate loads.
; CHECK: %for.body
; CHECK: smlad
; CHECK: smlad
; CHECK-NOT: smlad r{{.*}}

; CHECK-REG-PRESSURE: .LBB0_1:
; CHECK-REG-PRESSURE-NOT: call i32 @llvm.arm.smlad
; CHECK-REG-PRESSURE: ldr{{.*}}, [sp
; CHECK-REG-PRESSURE: ldr{{.*}}, [sp
; CHECK-REG-PRESSURE: ldr{{.*}}, [sp
; CHECK-REG-PRESSURE: ldr{{.*}}, [sp
; CHECK-REG-PRESSURE: ldr{{.*}}, [sp
; CHECK-REG-PRESSURE-NOT: ldr{{.*}}, [sp
; CHECK-REG-PRESSURE: bne .LBB0_1

for.body:
  %A3 = phi i32 [ %add9.us.i.3361.i, %for.body ], [ 0, %entry ]
  %j.026.us.i.i = phi i32 [ %inc.us.i.3362.i, %for.body ], [ 0, %entry ]
  %A4 = phi i32 [ %add9.us.i.1.3.i, %for.body ], [ 0, %entry ]
  %A5 = phi i32 [ %add9.us.i.2.3.i, %for.body ], [ 0, %entry ]
  %A6 = phi i32 [ %add9.us.i.3.3.i, %for.body ], [ 0, %entry ]
  %niter335.i = phi i32 [ %niter335.nsub.3.i, %for.body ], [ %unroll_iter310.i, %entry ]
  %add.us.i.i = add i32 %j.026.us.i.i, %mul.us.i118.i
  %arrayidx4.us.i.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.i
  %A7 = load i16, ptr %arrayidx4.us.i.i, align 2
  %conv.us.i.i = sext i16 %A7 to i32
  %arrayidx5.us.i.i = getelementptr inbounds i16, ptr %B, i32 %j.026.us.i.i
  %A8 = load i16, ptr %arrayidx5.us.i.i, align 2
  %conv6.us.i.i = sext i16 %A8 to i32
  %mul7.us.i.i = mul nsw i32 %conv6.us.i.i, %conv.us.i.i
  %add9.us.i.i = add nsw i32 %mul7.us.i.i, %A3
  %inc.us.i.i = or disjoint i32 %j.026.us.i.i, 1
  %add.us.i.1.i = add i32 %j.026.us.i.i, %mul.us.i118.1.i
  %arrayidx4.us.i.1.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.1.i
  %A9 = load i16, ptr %arrayidx4.us.i.1.i, align 2
  %conv.us.i.1.i = sext i16 %A9 to i32
  %arrayidx5.us.i.1.i = getelementptr inbounds i16, ptr %B, i32 %j.026.us.i.i
  %B0 = load i16, ptr %arrayidx5.us.i.1.i, align 2
  %conv6.us.i.1.i = sext i16 %B0 to i32
  %mul7.us.i.1.i = mul nsw i32 %conv6.us.i.1.i, %conv.us.i.1.i
  %add9.us.i.1.i = add nsw i32 %mul7.us.i.1.i, %A4
  %inc.us.i.1.i = or disjoint i32 %j.026.us.i.i, 1
  %add.us.i.2.i = add i32 %j.026.us.i.i, %mul.us.i118.2.i
  %arrayidx4.us.i.2.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.2.i
  %B1 = load i16, ptr %arrayidx4.us.i.2.i, align 2
  %conv.us.i.2.i = sext i16 %B1 to i32
  %arrayidx5.us.i.2.i = getelementptr inbounds i16, ptr %B, i32 %j.026.us.i.i
  %B2 = load i16, ptr %arrayidx5.us.i.2.i, align 2
  %conv6.us.i.2.i = sext i16 %B2 to i32
  %mul7.us.i.2.i = mul nsw i32 %conv6.us.i.2.i, %conv.us.i.2.i
  %add9.us.i.2.i = add nsw i32 %mul7.us.i.2.i, %A5
  %inc.us.i.2.i = or disjoint i32 %j.026.us.i.i, 1
  %add.us.i.3.i = add i32 %j.026.us.i.i, %mul.us.i118.3.i
  %arrayidx4.us.i.3.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.3.i
  %B3 = load i16, ptr %arrayidx4.us.i.3.i, align 2
  %conv.us.i.3.i = sext i16 %B3 to i32
  %arrayidx5.us.i.3.i = getelementptr inbounds i16, ptr %B, i32 %j.026.us.i.i
  %B4 = load i16, ptr %arrayidx5.us.i.3.i, align 2
  %conv6.us.i.3.i = sext i16 %B4 to i32
  %mul7.us.i.3.i = mul nsw i32 %conv6.us.i.3.i, %conv.us.i.3.i
  %add9.us.i.3.i = add nsw i32 %mul7.us.i.3.i, %A6
  %inc.us.i.3.i = or disjoint i32 %j.026.us.i.i, 1
  %add.us.i.1337.i = add i32 %inc.us.i.i, %mul.us.i118.i
  %arrayidx4.us.i.1338.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.1337.i
  %B5 = load i16, ptr %arrayidx4.us.i.1338.i, align 2
  %conv.us.i.1339.i = sext i16 %B5 to i32
  %arrayidx5.us.i.1340.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.i
  %B6 = load i16, ptr %arrayidx5.us.i.1340.i, align 2
  %conv6.us.i.1341.i = sext i16 %B6 to i32
  %mul7.us.i.1342.i = mul nsw i32 %conv6.us.i.1341.i, %conv.us.i.1339.i
  %add9.us.i.1343.i = add nsw i32 %mul7.us.i.1342.i, %add9.us.i.i
  %inc.us.i.1344.i = or disjoint i32 %j.026.us.i.i, 2
  %add.us.i.1.1.i = add i32 %inc.us.i.1.i, %mul.us.i118.1.i
  %arrayidx4.us.i.1.1.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.1.1.i
  %B7 = load i16, ptr %arrayidx4.us.i.1.1.i, align 2
  %conv.us.i.1.1.i = sext i16 %B7 to i32
  %arrayidx5.us.i.1.1.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.1.i
  %B6.dup = load i16, ptr %arrayidx5.us.i.1.1.i, align 2
  %conv6.us.i.1.1.i = sext i16 %B6.dup to i32
  %mul7.us.i.1.1.i = mul nsw i32 %conv6.us.i.1.1.i, %conv.us.i.1.1.i
  %add9.us.i.1.1.i = add nsw i32 %mul7.us.i.1.1.i, %add9.us.i.1.i
  %inc.us.i.1.1.i = or disjoint i32 %j.026.us.i.i, 2
  %add.us.i.2.1.i = add i32 %inc.us.i.2.i, %mul.us.i118.2.i
  %arrayidx4.us.i.2.1.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.2.1.i
  %B9 = load i16, ptr %arrayidx4.us.i.2.1.i, align 2
  %conv.us.i.2.1.i = sext i16 %B9 to i32
  %arrayidx5.us.i.2.1.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.2.i
  %B6.dup.i = load i16, ptr %arrayidx5.us.i.2.1.i, align 2
  %conv6.us.i.2.1.i = sext i16 %B6.dup.i to i32
  %mul7.us.i.2.1.i = mul nsw i32 %conv6.us.i.2.1.i, %conv.us.i.2.1.i
  %add9.us.i.2.1.i = add nsw i32 %mul7.us.i.2.1.i, %add9.us.i.2.i
  %inc.us.i.2.1.i = or disjoint i32 %j.026.us.i.i, 2
  %add.us.i.3.1.i = add i32 %inc.us.i.3.i, %mul.us.i118.3.i
  %arrayidx4.us.i.3.1.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.3.1.i
  %B11 = load i16, ptr %arrayidx4.us.i.3.1.i, align 2
  %conv.us.i.3.1.i = sext i16 %B11 to i32
  %arrayidx5.us.i.3.1.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.3.i
  %B6.dup.i.i = load i16, ptr %arrayidx5.us.i.3.1.i, align 2
  %conv6.us.i.3.1.i = sext i16 %B6.dup.i.i to i32
  %mul7.us.i.3.1.i = mul nsw i32 %conv6.us.i.3.1.i, %conv.us.i.3.1.i
  %add9.us.i.3.1.i = add nsw i32 %mul7.us.i.3.1.i, %add9.us.i.3.i
  %inc.us.i.3.1.i = or disjoint i32 %j.026.us.i.i, 2
  %add.us.i.2346.i = add i32 %inc.us.i.1344.i, %mul.us.i118.i
  %arrayidx4.us.i.2347.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.2346.i
  %B13 = load i16, ptr %arrayidx4.us.i.2347.i, align 2
  %conv.us.i.2348.i = sext i16 %B13 to i32
  %arrayidx5.us.i.2349.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.1344.i
  %B14 = load i16, ptr %arrayidx5.us.i.2349.i, align 2
  %conv6.us.i.2350.i = sext i16 %B14 to i32
  %mul7.us.i.2351.i = mul nsw i32 %conv6.us.i.2350.i, %conv.us.i.2348.i
  %add9.us.i.2352.i = add nsw i32 %mul7.us.i.2351.i, %add9.us.i.1343.i
  %inc.us.i.2353.i = or disjoint i32 %j.026.us.i.i, 3
  %add.us.i.1.2.i = add i32 %inc.us.i.1.1.i, %mul.us.i118.1.i
  %arrayidx4.us.i.1.2.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.1.2.i
  %B15 = load i16, ptr %arrayidx4.us.i.1.2.i, align 2
  %conv.us.i.1.2.i = sext i16 %B15 to i32
  %arrayidx5.us.i.1.2.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.1.1.i
  %B14.dup = load i16, ptr %arrayidx5.us.i.1.2.i, align 2
  %conv6.us.i.1.2.i = sext i16 %B14.dup to i32
  %mul7.us.i.1.2.i = mul nsw i32 %conv6.us.i.1.2.i, %conv.us.i.1.2.i
  %add9.us.i.1.2.i = add nsw i32 %mul7.us.i.1.2.i, %add9.us.i.1.1.i
  %inc.us.i.1.2.i = or disjoint i32 %j.026.us.i.i, 3
  %add.us.i.2.2.i = add i32 %inc.us.i.2.1.i, %mul.us.i118.2.i
  %arrayidx4.us.i.2.2.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.2.2.i
  %B17 = load i16, ptr %arrayidx4.us.i.2.2.i, align 2
  %conv.us.i.2.2.i = sext i16 %B17 to i32
  %arrayidx5.us.i.2.2.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.2.1.i
  %B14.dup.i = load i16, ptr %arrayidx5.us.i.2.2.i, align 2
  %conv6.us.i.2.2.i = sext i16 %B14.dup.i to i32
  %mul7.us.i.2.2.i = mul nsw i32 %conv6.us.i.2.2.i, %conv.us.i.2.2.i
  %add9.us.i.2.2.i = add nsw i32 %mul7.us.i.2.2.i, %add9.us.i.2.1.i
  %inc.us.i.2.2.i = or disjoint i32 %j.026.us.i.i, 3
  %add.us.i.3.2.i = add i32 %inc.us.i.3.1.i, %mul.us.i118.3.i
  %arrayidx4.us.i.3.2.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.3.2.i
  %B19 = load i16, ptr %arrayidx4.us.i.3.2.i, align 2
  %conv.us.i.3.2.i = sext i16 %B19 to i32
  %arrayidx5.us.i.3.2.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.3.1.i
  %B14.dup.i.i = load i16, ptr %arrayidx5.us.i.3.2.i, align 2
  %conv6.us.i.3.2.i = sext i16 %B14.dup.i.i to i32
  %mul7.us.i.3.2.i = mul nsw i32 %conv6.us.i.3.2.i, %conv.us.i.3.2.i
  %add9.us.i.3.2.i = add nsw i32 %mul7.us.i.3.2.i, %add9.us.i.3.1.i
  %inc.us.i.3.2.i = or disjoint i32 %j.026.us.i.i, 3
  %add.us.i.3355.i = add i32 %inc.us.i.2353.i, %mul.us.i118.i
  %arrayidx4.us.i.3356.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.3355.i
  %B21 = load i16, ptr %arrayidx4.us.i.3356.i, align 2
  %conv.us.i.3357.i = sext i16 %B21 to i32
  %arrayidx5.us.i.3358.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.2353.i
  %B22 = load i16, ptr %arrayidx5.us.i.3358.i, align 2
  %conv6.us.i.3359.i = sext i16 %B22 to i32
  %mul7.us.i.3360.i = mul nsw i32 %conv6.us.i.3359.i, %conv.us.i.3357.i
  %add9.us.i.3361.i = add nsw i32 %mul7.us.i.3360.i, %add9.us.i.2352.i
  %inc.us.i.3362.i = add i32 %j.026.us.i.i, 4
  %add.us.i.1.3.i = add i32 %inc.us.i.1.2.i, %mul.us.i118.1.i
  %arrayidx4.us.i.1.3.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.1.3.i
  %B23 = load i16, ptr %arrayidx4.us.i.1.3.i, align 2
  %conv.us.i.1.3.i = sext i16 %B23 to i32
  %arrayidx5.us.i.1.3.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.1.2.i
  %B22.dup = load i16, ptr %arrayidx5.us.i.1.3.i, align 2
  %conv6.us.i.1.3.i = sext i16 %B22.dup to i32
  %mul7.us.i.1.3.i = mul nsw i32 %conv6.us.i.1.3.i, %conv.us.i.1.3.i
  %add9.us.i.1.3.i = add nsw i32 %mul7.us.i.1.3.i, %add9.us.i.1.2.i
  %add.us.i.2.3.i = add i32 %inc.us.i.2.2.i, %mul.us.i118.2.i
  %arrayidx4.us.i.2.3.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.2.3.i
  %B25 = load i16, ptr %arrayidx4.us.i.2.3.i, align 2
  %conv.us.i.2.3.i = sext i16 %B25 to i32
  %arrayidx5.us.i.2.3.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.2.2.i
  %B22.dup.i = load i16, ptr %arrayidx5.us.i.2.3.i, align 2
  %conv6.us.i.2.3.i = sext i16 %B22.dup.i to i32
  %mul7.us.i.2.3.i = mul nsw i32 %conv6.us.i.2.3.i, %conv.us.i.2.3.i
  %add9.us.i.2.3.i = add nsw i32 %mul7.us.i.2.3.i, %add9.us.i.2.2.i
  %add.us.i.3.3.i = add i32 %inc.us.i.3.2.i, %mul.us.i118.3.i
  %arrayidx4.us.i.3.3.i = getelementptr inbounds i16, ptr %A, i32 %add.us.i.3.3.i
  %B27 = load i16, ptr %arrayidx4.us.i.3.3.i, align 2
  %conv.us.i.3.3.i = sext i16 %B27 to i32
  %arrayidx5.us.i.3.3.i = getelementptr inbounds i16, ptr %B, i32 %inc.us.i.3.2.i
  %B22.dup.i.i = load i16, ptr %arrayidx5.us.i.3.3.i, align 2
  %conv6.us.i.3.3.i = sext i16 %B22.dup.i.i to i32
  %mul7.us.i.3.3.i = mul nsw i32 %conv6.us.i.3.3.i, %conv.us.i.3.3.i
  %add9.us.i.3.3.i = add nsw i32 %mul7.us.i.3.3.i, %add9.us.i.3.2.i
  %niter335.nsub.3.i = add i32 %niter335.i, -4
  %niter335.ncmp.3.i = icmp eq i32 %niter335.nsub.3.i, 0
  br i1 %niter335.ncmp.3.i, label %exit, label %for.body

exit:
  store i32 %add9.us.i.3361.i, ptr %res, align 4
  %arrayidx.out.1.i = getelementptr inbounds i32, ptr %res, i32 1
  store i32 %add9.us.i.1.3.i, ptr %arrayidx.out.1.i, align 4
  %arrayidx.out.2.i = getelementptr inbounds i32, ptr %res, i32 2
  store i32 %add9.us.i.2.3.i, ptr %arrayidx.out.2.i, align 4
  %arrayidx.out.3.i = getelementptr inbounds i32, ptr %res, i32 3
  store i32 %add9.us.i.3.3.i, ptr %arrayidx.out.3.i, align 4
  ret void
}
