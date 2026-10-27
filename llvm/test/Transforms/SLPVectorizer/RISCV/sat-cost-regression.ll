; RUN: opt -mtriple=riscv64 -mattr=+v,+zbb -passes='default<O3>' -S %s | FileCheck %s

; Regression test for LLVM issue #209237.
; Scalar saturating arithmetic should not be unprofitably
; combined into a vector saturating operation.

; CHECK-LABEL: define i16 @lattice_step(
; CHECK: call i16 @llvm.sadd.sat.i16(
; CHECK: call i16 @llvm.sadd.sat.i16(
; CHECK-NOT: call <2 x i16> @llvm.sadd.sat.v2i16(
; CHECK: ret i16

define i16 @lattice_step(i16 %di, i16 %ui, i16 %rpi, ptr %u, ptr %s) {
  %rpi.ext = sext i16 %rpi to i64
  %di.ext = sext i16 %di to i64
  %ui.ext = sext i16 %ui to i64
  %m1.shl = shl nsw i64 %di.ext, 33
  %m1 = mul i64 %m1.shl, %rpi.ext
  %m1.rnd = add i64 %m1, 140737488355328
  %m1.shr = lshr i64 %m1.rnd, 48
  %m1.t = trunc nuw i64 %m1.shr to i16
  %sav = tail call i16 @llvm.sadd.sat.i16(i16 %m1.t, i16 %ui)
  %m2.shl = shl nsw i64 %ui.ext, 33
  %m2 = mul i64 %m2.shl, %rpi.ext
  %m2.rnd = add i64 %m2, 140737488355328
  %m2.shr = lshr i64 %m2.rnd, 48
  %m2.t = trunc nuw i64 %m2.shr to i16
  %di.next = tail call i16 @llvm.sadd.sat.i16(i16 %m2.t, i16 %di)
  store i16 %sav, ptr %u, align 2
  store i16 %di.next, ptr %s, align 2
  ret i16 %di.next
}

declare i16 @llvm.sadd.sat.i16(i16, i16)
