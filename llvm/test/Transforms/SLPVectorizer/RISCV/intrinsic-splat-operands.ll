; RUN: opt -p slp-vectorizer -mtriple=riscv64 -mattr=+v -riscv-v-slp-max-vf=2 \
; RUN:   -debug-only=SLP -S -disable-output < %s 2>&1 | FileCheck %s --check-prefix=SLP

target triple = "riscv64-unknown-linux-gnu"

declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.ssub.sat.i32(i32, i32)
declare i32 @llvm.fshl.i32(i32, i32, i32)

; The splat is operand 0 of a commutative, splattable intrinsic.
; SLP: Adding cost 0 for bundle Idx: 2, n=2 [i32 %splat, ..].
define void @smax_splat_first(ptr %out, ptr %in, i32 %splat) {
entry:
  %in0 = load i32, ptr %in, align 4
  %in.next = getelementptr i32, ptr %in, i64 1
  %in1 = load i32, ptr %in.next, align 4
  %result0 = call i32 @llvm.smax.i32(i32 %splat, i32 %in0)
  %result1 = call i32 @llvm.smax.i32(i32 %splat, i32 %in1)
  store i32 %result0, ptr %out, align 4
  %out.next = getelementptr i32, ptr %out, i64 1
  store i32 %result1, ptr %out.next, align 4
  ret void
}

; The splat is operand 1 of a splattable intrinsic.
; SLP: Adding cost 0 for bundle Idx: 3, n=2 [i32 %splat, ..].
define void @ssub_splat_second(ptr %out, ptr %in, i32 %splat) {
entry:
  %in0 = load i32, ptr %in, align 4
  %in.next = getelementptr i32, ptr %in, i64 1
  %in1 = load i32, ptr %in.next, align 4
  %result0 = call i32 @llvm.ssub.sat.i32(i32 %in0, i32 %splat)
  %result1 = call i32 @llvm.ssub.sat.i32(i32 %in1, i32 %splat)
  store i32 %result0, ptr %out, align 4
  %out.next = getelementptr i32, ptr %out, i64 1
  store i32 %result1, ptr %out.next, align 4
  ret void
}

; The intrinsic is splattable, but operand 0 is not a splattable operand.
; SLP: Adding cost 2 for bundle Idx: 2, n=2 [i32 %splat, ..].
define void @ssub_splat_first(ptr %out, ptr %in, i32 %splat) {
entry:
  %in0 = load i32, ptr %in, align 4
  %in.next = getelementptr i32, ptr %in, i64 1
  %in1 = load i32, ptr %in.next, align 4
  %result0 = call i32 @llvm.ssub.sat.i32(i32 %splat, i32 %in0)
  %result1 = call i32 @llvm.ssub.sat.i32(i32 %splat, i32 %in1)
  store i32 %result0, ptr %out, align 4
  %out.next = getelementptr i32, ptr %out, i64 1
  store i32 %result1, ptr %out.next, align 4
  ret void
}

; fshl is not one of the intrinsics with a splattable operand on RISC-V.
; SLP: Adding cost 2 for bundle Idx: 2, n=2 [i32 %splat, ..].
define void @fshl_not_splattable(ptr %out, ptr %in, i32 %splat) {
entry:
  %in0 = load i32, ptr %in, align 4
  %in.next = getelementptr i32, ptr %in, i64 1
  %in1 = load i32, ptr %in.next, align 4
  %shift0 = load i32, ptr %in, align 4
  %shift.next = getelementptr i32, ptr %in, i64 1
  %shift1 = load i32, ptr %shift.next, align 4
  %result0 = call i32 @llvm.fshl.i32(i32 %splat, i32 %in0, i32 %shift0)
  %result1 = call i32 @llvm.fshl.i32(i32 %splat, i32 %in1, i32 %shift1)
  store i32 %result0, ptr %out, align 4
  %out.next = getelementptr i32, ptr %out, i64 1
  store i32 %result1, ptr %out.next, align 4
  ret void
}

; shl operand 0 is not splattable
; SLP: Adding cost 2 for bundle Idx: 2, n=2 [i32 %splat, ..].
define void @shl_not_splattable(ptr %out, ptr %in, i32 %splat) {
entry:
  %in0 = load i32, ptr %in, align 4
  %in.next = getelementptr i32, ptr %in, i64 1
  %in1 = load i32, ptr %in.next, align 4
  %shift0 = load i32, ptr %in, align 4
  %shift.next = getelementptr i32, ptr %in, i64 1
  %shift1 = load i32, ptr %shift.next, align 4
  %result0 = lshr i32 %splat, %shift0
  %result1 = lshr i32 %splat, %shift1
  store i32 %result0, ptr %out, align 4
  %out.next = getelementptr i32, ptr %out, i64 1
  store i32 %result1, ptr %out.next, align 4
  ret void
}
