; REQUIRES: asserts
; RUN: opt --passes=loop-vectorize --mtriple loongarch64-linux-gnu \
; RUN:   --mattr=+lsx -debug-only=loop-vectorize --force-vector-width=1 \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCALAR
; RUN: opt --passes=loop-vectorize --mtriple loongarch64-linux-gnu \
; RUN:   --mattr=+lsx -debug-only=loop-vectorize --force-vector-width=4 \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-VECTOR

define void @bar(ptr %A, i32 signext %n) {
; CHECK-LABEL: bar
; CHECK-SCALAR:      LV(REG): Found max usage: 2 item
; CHECK-SCALAR-NEXT: LV(REG): RegisterClass: LoongArch::GPRRC, 2 registers
; CHECK-SCALAR-NEXT: LV(REG): RegisterClass: LoongArch::FPRRC, 1 registers
; CHECK-SCALAR-NEXT: LV(REG): Found invariant usage: 1 item
; CHECK-SCALAR-NEXT: LV(REG): RegisterClass: LoongArch::GPRRC, 1 registers
; CHECK-SCALAR-NEXT: LV: The target has 30 registers of LoongArch::GPRRC register class
; CHECK-SCALAR-NEXT: LV: The target has 32 registers of LoongArch::FPRRC register class
; CHECK-VECTOR:      LV(REG): Found max usage: 1 item
; CHECK-VECTOR-NEXT: LV(REG): RegisterClass: LoongArch::VRRC, 3 registers
; CHECK-VECTOR-NEXT: LV(REG): Found invariant usage: 1 item
; CHECK-VECTOR-NEXT: LV(REG): RegisterClass: LoongArch::GPRRC, 1 registers
; CHECK-VECTOR-NEXT: LV: The target has 32 registers of LoongArch::VRRC register class

entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to float
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  store float %conv, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
