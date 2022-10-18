; REQUIRES: asserts
; RUN: opt -loop-vectorize -mtriple riscv64-linux-gnu \
; RUN:   -mattr=+v,+d -debug-only=loop-vectorize \
; RUN:   -riscv-v-vector-bits-min=128 -force-vector-width=1 \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCALAR
; RUN: opt -loop-vectorize -mtriple riscv64-linux-gnu \
; RUN:   -mattr=+v,+d -debug-only=loop-vectorize \
; RUN:   -riscv-v-vector-bits-min=128 -riscv-v-register-bit-width-lmul=1 \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LMUL1
; RUN: opt -loop-vectorize -mtriple riscv64-linux-gnu \
; RUN:   -mattr=+v,+d -debug-only=loop-vectorize \
; RUN:   -riscv-v-vector-bits-min=128 -riscv-v-register-bit-width-lmul=2 \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LMUL2
; RUN: opt -loop-vectorize -mtriple riscv64-linux-gnu \
; RUN:   -mattr=+v,+d -debug-only=loop-vectorize \
; RUN:   -riscv-v-vector-bits-min=128 -riscv-v-register-bit-width-lmul=4 \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LMUL4
; RUN: opt -loop-vectorize -mtriple riscv64-linux-gnu \
; RUN:   -mattr=+v,+d -debug-only=loop-vectorize \
; RUN:   -riscv-v-vector-bits-min=128 -riscv-v-register-bit-width-lmul=8 \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LMUL8

define void @add(float* noalias nocapture readonly %src1, float* noalias nocapture readonly %src2, i32 signext %size, float* noalias nocapture writeonly %result) {
; CHECK-LABEL: add
; CHECK-SCALAR:      LV(REG): Found max usage: 2 item
; CHECK-SCALAR-NEXT: LV(REG): RegisterClass: RISCV::GPRRC, 2 registers
; CHECK-SCALAR-NEXT: LV(REG): RegisterClass: RISCV::FPRRC, 2 registers
; CHECK-SCALAR-NEXT: LV(REG): Found invariant usage: 1 item
; CHECK-SCALAR-NEXT: LV(REG): RegisterClass: RISCV::GPRRC, 1 registers
; CHECK-LMUL1:       LV(REG): Found max usage: 2 item
; CHECK-LMUL1-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 2 registers
; CHECK-LMUL1-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 2 registers
; CHECK-LMUL1-NEXT:  LV(REG): Found invariant usage: 1 item
; CHECK-LMUL1-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 2 registers
; CHECK-LMUL2:       LV(REG): Found max usage: 2 item
; CHECK-LMUL2-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 2 registers
; CHECK-LMUL2-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 4 registers
; CHECK-LMUL2-NEXT:  LV(REG): Found invariant usage: 1 item
; CHECK-LMUL2-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 4 registers
; CHECK-LMUL4:       LV(REG): Found max usage: 2 item
; CHECK-LMUL4-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 2 registers
; CHECK-LMUL4-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 8 registers
; CHECK-LMUL4-NEXT:  LV(REG): Found invariant usage: 1 item
; CHECK-LMUL4-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 8 registers
; CHECK-LMUL8:       LV(REG): Found max usage: 2 item
; CHECK-LMUL8-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 2 registers
; CHECK-LMUL8-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 16 registers
; CHECK-LMUL8-NEXT:  LV(REG): Found invariant usage: 1 item
; CHECK-LMUL8-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 16 registers

entry:
  %conv = zext i32 %size to i64
  %cmp10.not = icmp eq i32 %size, 0
  br i1 %cmp10.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.011 = phi i64 [ %add4, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %src1, i64 %i.011
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %src2, i64 %i.011
  %1 = load float, float* %arrayidx2, align 4
  %add = fadd float %0, %1
  %arrayidx3 = getelementptr inbounds float, float* %result, i64 %i.011
  store float %add, float* %arrayidx3, align 4
  %add4 = add nuw nsw i64 %i.011, 1
  %exitcond.not = icmp eq i64 %add4, %conv
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @goo(i32** nocapture noundef %a, i32 noundef signext %n) {
; CHECK-LABEL: goo
; CHECK-SCALAR:      LV(REG): Found max usage: 1 item
; CHECK-SCALAR-NEXT: LV(REG): RegisterClass: RISCV::GPRRC, 3 registers
; CHECK-LMUL1:       LV(REG): Found max usage: 2 item
; CHECK-LMUL1-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 2 registers
; CHECK-LMUL1-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 1 registers
; CHECK-LMUL2:       LV(REG): Found max usage: 2 item
; CHECK-LMUL2-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 2 registers
; CHECK-LMUL2-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 2 registers
; CHECK-LMUL4:       LV(REG): Found max usage: 2 item
; CHECK-LMUL4-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 2 registers
; CHECK-LMUL4-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 4 registers
; CHECK-LMUL8:       LV(REG): Found max usage: 2 item
; CHECK-LMUL8-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 2 registers
; CHECK-LMUL8-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 8 registers
entry:
  %cmp3 = icmp sgt i32 %n, 0
  br i1 %cmp3, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32*, i32** %a, i64 %indvars.iv
  %0 = load i32*, i32** %arrayidx, align 8
  %add.ptr = getelementptr inbounds i32, i32* %0, i64 1
  store i32* %add.ptr, i32** %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
