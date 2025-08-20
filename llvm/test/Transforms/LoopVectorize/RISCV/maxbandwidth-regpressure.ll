; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -mattr=+v -vectorizer-maximize-bandwidth -debug-only=loop-vectorize,vplan -disable-output -force-vector-interleave=1 -enable-epilogue-vectorization=false -S < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-REGS-VP
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -mattr=+v -vectorizer-maximize-bandwidth -debug-only=loop-vectorize -disable-output -force-target-num-vector-regs=1 -force-vector-interleave=1 -enable-epilogue-vectorization=false -S < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-NOREGS-VP
define i32 @dotp(ptr %a, ptr %b) {
; CHECK-REGS-VP:      LV(REG): VF = vscale x 16
; CHECK-REGS-VP-NEXT: LV(REG): Found max usage: 2 item
; CHECK-REGS-VP-NEXT: LV(REG): RegisterClass: RISCV::GPRRC, 6 registers
; CHECK-REGS-VP-NEXT: LV(REG): RegisterClass: RISCV::VRRC, 24 registers
; CHECK-REGS-VP-NEXT: LV(REG): Found invariant usage: 1 item
; CHECK-REGS-VP-NEXT: LV(REG): RegisterClass: RISCV::GPRRC, 1 registers
; CHECK-REGS-VP: LV: Selecting VF: vscale x 16.
;
; CHECK-NOREGS-VP: LV(REG): Not considering vector loop of width vscale x 8 because it uses too many registers
; CHECK-NOREGS-VP: LV(REG): Not considering vector loop of width vscale x 16 because it uses too many registers
; CHECK-NOREGS-VP: LV: Selecting VF: vscale x 4.
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %accum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %gep.a = getelementptr i8, ptr %a, i64 %iv
  %load.a = load i8, ptr %gep.a, align 1
  %ext.a = zext i8 %load.a to i32
  %gep.b = getelementptr i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b, align 1
  %ext.b = zext i8 %load.b to i32
  %mul = mul i32 %ext.b, %ext.a
  %sub = sub i32 0, %mul
  %add = add i32 %accum, %sub
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %for.exit, label %for.body

for.exit:                        ; preds = %for.body
  ret i32 %add
}
