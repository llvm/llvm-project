; RUN: opt -p slp-vectorizer -mtriple=riscv64 -mattr=+v -riscv-v-slp-max-vf=2 -S -disable-output < %s -pass-remarks-output=%t
; RUN: FileCheck %s --check-prefix REMARK --input-file=%t

; The splat is operand 0 of a commutative, splattable intrinsic.
; REMARK-LABEL: --- !Passed
; REMARK-NEXT: Pass:            slp-vectorizer
; REMARK-NEXT: Name:            StoresVectorized
; REMARK-NEXT: Function:        smax_splat_first
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          'Stores SLP vectorized with cost '
; REMARK-NEXT:   - Cost:            '-5'
; REMARK-NEXT:   - String:          ' and with tree size '
; REMARK-NEXT:   - TreeSize:        '4'
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
; REMARK-LABEL: --- !Passed
; REMARK-NEXT: Pass:            slp-vectorizer
; REMARK-NEXT: Name:            StoresVectorized
; REMARK-NEXT: Function:        ssub_splat_second
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          'Stores SLP vectorized with cost '
; REMARK-NEXT:   - Cost:            '-15'
; REMARK-NEXT:   - String:          ' and with tree size '
; REMARK-NEXT:   - TreeSize:        '4'
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
; REMARK-LABEL: --- !Passed
; REMARK-NEXT: Pass:            slp-vectorizer
; REMARK-NEXT: Name:            StoresVectorized
; REMARK-NEXT: Function:        ssub_splat_first
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          'Stores SLP vectorized with cost '
; REMARK-NEXT:   - Cost:            '-13'
; REMARK-NEXT:   - String:          ' and with tree size '
; REMARK-NEXT:   - TreeSize:        '4'
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
; REMARK-LABEL: --- !Passed
; REMARK-NEXT: Pass:            slp-vectorizer
; REMARK-NEXT: Name:            StoresVectorized
; REMARK-NEXT: Function:        fshl_not_splattable
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          'Stores SLP vectorized with cost '
; REMARK-NEXT:   - Cost:            '-8'
; REMARK-NEXT:   - String:          ' and with tree size '
; REMARK-NEXT:   - TreeSize:        '5'
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
; REMARK-LABEL: --- !Passed
; REMARK-NEXT: Pass:            slp-vectorizer
; REMARK-NEXT: Name:            StoresVectorized
; REMARK-NEXT: Function:        shl_not_splattable
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          'Stores SLP vectorized with cost '
; REMARK-NEXT:   - Cost:            '-1'
; REMARK-NEXT:   - String:          ' and with tree size '
; REMARK-NEXT:   - TreeSize:        '4'
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
