; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -mattr=+v,+zvfbfmin -debug-only=loop-vectorize,vplan --disable-output -riscv-v-register-bit-width-lmul=1 -S < %s 2>&1 | FileCheck %s

define void @add(ptr noalias nocapture readonly %src1, ptr noalias nocapture readonly %src2, i32 signext %size, ptr noalias nocapture writeonly %result) {
; CHECK-LABEL: add
; CHECK:       LV(REG): VF = vscale x 4
; CHECK-NEXT:  LV(REG): Found max usage: 2 item
; CHECK-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 6 registers
; CHECK-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 4 registers
; CHECK-NEXT:  LV(REG): Found invariant usage: 1 item
; CHECK-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 1 registers

entry:
  %conv = zext i32 %size to i64
  %cmp10.not = icmp eq i32 %size, 0
  br i1 %cmp10.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.011 = phi i64 [ %add4, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds bfloat, ptr %src1, i64 %i.011
  %0 = load bfloat, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds bfloat, ptr %src2, i64 %i.011
  %1 = load bfloat, ptr %arrayidx2, align 4
  %add = fadd bfloat %0, %1
  %arrayidx3 = getelementptr inbounds bfloat, ptr %result, i64 %i.011
  store bfloat %add, ptr %arrayidx3, align 4
  %add4 = add nuw nsw i64 %i.011, 1
  %exitcond.not = icmp eq i64 %add4, %conv
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
