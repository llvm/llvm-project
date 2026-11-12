; RUN: not llubi --verbose --entry-function=evl_too_large < %s 2>&1 | FileCheck %s --check-prefix=EVL
; RUN: not llubi --verbose --entry-function=poison_evl < %s 2>&1 | FileCheck %s --check-prefix=POISON-EVL
; RUN: not llubi --verbose --entry-function=poison_mask_lane < %s 2>&1 | FileCheck %s --check-prefix=POISON-MASK-LANE
; RUN: not llubi --verbose --entry-function=poison_stride < %s 2>&1 | FileCheck %s --check-prefix=POISON-STRIDE
; RUN: not llubi --verbose --entry-function=out_of_bounds < %s 2>&1 | FileCheck %s --check-prefix=OOB
; RUN: not llubi --verbose --entry-function=misaligned_stride < %s 2>&1 | FileCheck %s --check-prefix=MISALIGNED-STRIDE
; RUN: not llubi --verbose --entry-function=align_attr < %s 2>&1 | FileCheck %s --check-prefix=ALIGN-ATTR

@a = global [8 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7]

define void @evl_too_large() {
; EVL: Entering function: evl_too_large
; EVL-NEXT: Stacktrace:
; EVL-NEXT: #0   %r = call <4 x i32> @llvm.experimental.vp.strided.load.v4i32.p0.i64(ptr @a, i64 4, <4 x i1> splat (i1 true), i32 5) at @evl_too_large <stdin>:{{[0-9]+}}
; EVL-NEXT: Immediate UB detected: llvm.experimental.vp.strided.load with explicit vector length 5 greater than the number of elements 4.
; EVL-NEXT: error: Execution of function 'evl_too_large' failed.
  %r = call <4 x i32> @llvm.experimental.vp.strided.load(ptr @a, i64 4, <4 x i1> splat (i1 true), i32 5)
  ret void
}

define void @poison_evl() {
; POISON-EVL: Entering function: poison_evl
; POISON-EVL-NEXT: Stacktrace:
; POISON-EVL-NEXT: #0   %r = call <4 x i32> @llvm.experimental.vp.strided.load.v4i32.p0.i64(ptr @a, i64 4, <4 x i1> splat (i1 true), i32 poison) at @poison_evl <stdin>:{{[0-9]+}}
; POISON-EVL-NEXT: Immediate UB detected: llvm.experimental.vp.strided.load with poison explicit vector length.
; POISON-EVL-NEXT: error: Execution of function 'poison_evl' failed.
  %r = call <4 x i32> @llvm.experimental.vp.strided.load(ptr @a, i64 4, <4 x i1> splat (i1 true), i32 poison)
  ret void
}

define void @poison_mask_lane() {
; POISON-MASK-LANE: Entering function: poison_mask_lane
; POISON-MASK-LANE-NEXT: Stacktrace:
; POISON-MASK-LANE-NEXT: #0   %r = call <4 x i32> @llvm.experimental.vp.strided.load.v4i32.p0.i64(ptr @a, i64 4, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2) at @poison_mask_lane <stdin>:{{[0-9]+}}
; POISON-MASK-LANE-NEXT: Immediate UB detected: Poison mask lane in llvm.experimental.vp.strided.load.
; POISON-MASK-LANE-NEXT: error: Execution of function 'poison_mask_lane' failed.
  %r = call <4 x i32> @llvm.experimental.vp.strided.load(ptr @a, i64 4, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2)
  ret void
}

; A poison stride makes every lane's pointer poison.
define void @poison_stride() {
; POISON-STRIDE: Entering function: poison_stride
; POISON-STRIDE-NEXT: Stacktrace:
; POISON-STRIDE-NEXT: #0   %r = call <4 x i32> @llvm.experimental.vp.strided.load.v4i32.p0.i64(ptr @a, i64 poison, <4 x i1> splat (i1 true), i32 1) at @poison_stride <stdin>:{{[0-9]+}}
; POISON-STRIDE-NEXT: Immediate UB detected: Invalid memory access with a poison pointer.
; POISON-STRIDE-NEXT: error: Execution of function 'poison_stride' failed.
  %r = call <4 x i32> @llvm.experimental.vp.strided.load(ptr @a, i64 poison, <4 x i1> splat (i1 true), i32 1)
  ret void
}

define void @out_of_bounds() {
; OOB: Entering function: out_of_bounds
; OOB-NEXT: Stacktrace:
; OOB-NEXT: #0   %r = call <4 x i32> @llvm.experimental.vp.strided.load.v4i32.p0.i64(ptr @a, i64 16, <4 x i1> splat (i1 true), i32 3) at @out_of_bounds <stdin>:{{[0-9]+}}
; OOB-NEXT: Immediate UB detected: Memory access is out of bounds. Accessed size: 4, Address: 0x30, Object base: 0x10, Object size: 32.
; OOB-NEXT: error: Execution of function 'out_of_bounds' failed.
  %r = call <4 x i32> @llvm.experimental.vp.strided.load(ptr @a, i64 16, <4 x i1> splat (i1 true), i32 3)
  ret void
}

; Without an align attribute each lane's pointer must have the ABI alignment of
; i32, so a stride of 2 makes lane 1's pointer poison.
define void @misaligned_stride() {
; MISALIGNED-STRIDE: Entering function: misaligned_stride
; MISALIGNED-STRIDE-NEXT: Stacktrace:
; MISALIGNED-STRIDE-NEXT: #0   %r = call <4 x i32> @llvm.experimental.vp.strided.load.v4i32.p0.i64(ptr @a, i64 2, <4 x i1> splat (i1 true), i32 2) at @misaligned_stride <stdin>:{{[0-9]+}}
; MISALIGNED-STRIDE-NEXT: Immediate UB detected: Invalid memory access with a poison pointer.
; MISALIGNED-STRIDE-NEXT: error: Execution of function 'misaligned_stride' failed.
  %r = call <4 x i32> @llvm.experimental.vp.strided.load(ptr @a, i64 2, <4 x i1> splat (i1 true), i32 2)
  ret void
}

; The align attribute on the base pointer applies to every lane's pointer.
define void @align_attr() {
; ALIGN-ATTR: Entering function: align_attr
; ALIGN-ATTR-NEXT: Stacktrace:
; ALIGN-ATTR-NEXT: #0   %r = call <4 x i32> @llvm.experimental.vp.strided.load.v4i32.p0.i64(ptr align 8 @a, i64 4, <4 x i1> splat (i1 true), i32 2) at @align_attr <stdin>:{{[0-9]+}}
; ALIGN-ATTR-NEXT: Immediate UB detected: Invalid memory access with a poison pointer.
; ALIGN-ATTR-NEXT: error: Execution of function 'align_attr' failed.
  %r = call <4 x i32> @llvm.experimental.vp.strided.load(ptr align 8 @a, i64 4, <4 x i1> splat (i1 true), i32 2)
  ret void
}
