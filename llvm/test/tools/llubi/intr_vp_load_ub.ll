; RUN: not llubi --verbose --entry-function=evl_too_large < %s 2>&1 | FileCheck %s --check-prefix=EVL
; RUN: not llubi --verbose --vscale=2 --entry-function=evl_too_large_scalable < %s 2>&1 | FileCheck %s --check-prefix=EVL-SCALABLE
; RUN: not llubi --verbose --entry-function=poison_evl < %s 2>&1 | FileCheck %s --check-prefix=POISON-EVL
; RUN: not llubi --verbose --entry-function=poison_mask_lane < %s 2>&1 | FileCheck %s --check-prefix=POISON-MASK-LANE
; RUN: not llubi --verbose --entry-function=out_of_bounds < %s 2>&1 | FileCheck %s --check-prefix=OOB
; RUN: not llubi --verbose --entry-function=misaligned < %s 2>&1 | FileCheck %s --check-prefix=MISALIGNED

@a = global [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16

define void @evl_too_large() {
; EVL: Entering function: evl_too_large
; EVL-NEXT: Stacktrace:
; EVL-NEXT: #0   %r = call <4 x i32> @llvm.vp.load.v4i32.p0(ptr @a, <4 x i1> splat (i1 true), i32 5) at @evl_too_large <stdin>:{{[0-9]+}}
; EVL-NEXT: Immediate UB detected: llvm.vp.load with explicit vector length 5 greater than the number of elements 4.
; EVL-NEXT: error: Execution of function 'evl_too_large' failed.
  %r = call <4 x i32> @llvm.vp.load(ptr @a, <4 x i1> splat (i1 true), i32 5)
  ret void
}

define void @evl_too_large_scalable() {
; EVL-SCALABLE: Entering function: evl_too_large_scalable
; EVL-SCALABLE-NEXT: Stacktrace:
; EVL-SCALABLE-NEXT: #0   %r = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0(ptr @a, <vscale x 2 x i1> splat (i1 true), i32 5) at @evl_too_large_scalable <stdin>:{{[0-9]+}}
; EVL-SCALABLE-NEXT: Immediate UB detected: llvm.vp.load with explicit vector length 5 greater than the number of elements 4.
; EVL-SCALABLE-NEXT: error: Execution of function 'evl_too_large_scalable' failed.
  %r = call <vscale x 2 x i32> @llvm.vp.load(ptr @a, <vscale x 2 x i1> splat (i1 true), i32 5)
  ret void
}

define void @poison_evl() {
; POISON-EVL: Entering function: poison_evl
; POISON-EVL-NEXT: Stacktrace:
; POISON-EVL-NEXT: #0   %r = call <4 x i32> @llvm.vp.load.v4i32.p0(ptr @a, <4 x i1> splat (i1 true), i32 poison) at @poison_evl <stdin>:{{[0-9]+}}
; POISON-EVL-NEXT: Immediate UB detected: llvm.vp.load with poison explicit vector length.
; POISON-EVL-NEXT: error: Execution of function 'poison_evl' failed.
  %r = call <4 x i32> @llvm.vp.load(ptr @a, <4 x i1> splat (i1 true), i32 poison)
  ret void
}

define void @poison_mask_lane() {
; POISON-MASK-LANE: Entering function: poison_mask_lane
; POISON-MASK-LANE-NEXT: Stacktrace:
; POISON-MASK-LANE-NEXT: #0   %r = call <4 x i32> @llvm.vp.load.v4i32.p0(ptr @a, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2) at @poison_mask_lane <stdin>:{{[0-9]+}}
; POISON-MASK-LANE-NEXT: Immediate UB detected: Poison mask lane in llvm.vp.load.
; POISON-MASK-LANE-NEXT: error: Execution of function 'poison_mask_lane' failed.
  %r = call <4 x i32> @llvm.vp.load(ptr @a, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2)
  ret void
}

define void @out_of_bounds() {
; OOB: Entering function: out_of_bounds
; OOB-NEXT:   %tail = getelementptr i32, ptr @a, i64 2 => ptr 0x18 [@a + 8]
; OOB-NEXT: Stacktrace:
; OOB-NEXT: #0   %r = call <4 x i32> @llvm.vp.load.v4i32.p0(ptr align 4 %tail, <4 x i1> splat (i1 true), i32 3) at @out_of_bounds <stdin>:{{[0-9]+}}
; OOB-NEXT: Immediate UB detected: Memory access is out of bounds. Accessed size: 4, Address: 0x20, Object base: 0x10, Object size: 16.
; OOB-NEXT: error: Execution of function 'out_of_bounds' failed.
  %tail = getelementptr i32, ptr @a, i64 2
  %r = call <4 x i32> @llvm.vp.load(ptr align 4 %tail, <4 x i1> splat (i1 true), i32 3)
  ret void
}

; Without an align attribute the pointer must have the ABI alignment of
; <2 x i32>, so the misaligned pointer is poison.
define void @misaligned() {
; MISALIGNED: Entering function: misaligned
; MISALIGNED-NEXT:   %p = getelementptr i32, ptr @a, i64 1 => ptr 0x14 [@a + 4]
; MISALIGNED-NEXT: Stacktrace:
; MISALIGNED-NEXT: #0   %r = call <2 x i32> @llvm.vp.load.v2i32.p0(ptr %p, <2 x i1> splat (i1 true), i32 1) at @misaligned <stdin>:{{[0-9]+}}
; MISALIGNED-NEXT: Immediate UB detected: Invalid memory access with a poison pointer.
; MISALIGNED-NEXT: error: Execution of function 'misaligned' failed.
  %p = getelementptr i32, ptr @a, i64 1
  %r = call <2 x i32> @llvm.vp.load(ptr %p, <2 x i1> splat (i1 true), i32 1)
  ret void
}
