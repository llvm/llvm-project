; RUN: not llubi --verbose --entry-function=evl_too_large < %s 2>&1 | FileCheck %s --check-prefix=EVL
; RUN: not llubi --verbose --entry-function=poison_evl < %s 2>&1 | FileCheck %s --check-prefix=POISON-EVL
; RUN: not llubi --verbose --entry-function=poison_mask_lane < %s 2>&1 | FileCheck %s --check-prefix=POISON-MASK-LANE
; RUN: not llubi --verbose --entry-function=poison_stride < %s 2>&1 | FileCheck %s --check-prefix=POISON-STRIDE
; RUN: not llubi --verbose --entry-function=constant_mem < %s 2>&1 | FileCheck %s --check-prefix=CONSTANT-MEM
; RUN: not llubi --verbose --entry-function=misaligned_base < %s 2>&1 | FileCheck %s --check-prefix=MISALIGNED-BASE

@a = global [8 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7]
@c = constant [4 x i32] [i32 0, i32 1, i32 2, i32 3]

define void @evl_too_large() {
; EVL: Entering function: evl_too_large
; EVL-NEXT: Stacktrace:
; EVL-NEXT: #0   call void @llvm.experimental.vp.strided.store.v4i32.p0.i64(<4 x i32> zeroinitializer, ptr @a, i64 4, <4 x i1> splat (i1 true), i32 5) at @evl_too_large <stdin>:{{[0-9]+}}
; EVL-NEXT: Immediate UB detected: llvm.experimental.vp.strided.store with explicit vector length 5 greater than the number of elements 4.
; EVL-NEXT: error: Execution of function 'evl_too_large' failed.
  call void @llvm.experimental.vp.strided.store(<4 x i32> zeroinitializer, ptr @a, i64 4, <4 x i1> splat (i1 true), i32 5)
  ret void
}

define void @poison_evl() {
; POISON-EVL: Entering function: poison_evl
; POISON-EVL-NEXT: Stacktrace:
; POISON-EVL-NEXT: #0   call void @llvm.experimental.vp.strided.store.v4i32.p0.i64(<4 x i32> zeroinitializer, ptr @a, i64 4, <4 x i1> splat (i1 true), i32 poison) at @poison_evl <stdin>:{{[0-9]+}}
; POISON-EVL-NEXT: Immediate UB detected: llvm.experimental.vp.strided.store with poison explicit vector length.
; POISON-EVL-NEXT: error: Execution of function 'poison_evl' failed.
  call void @llvm.experimental.vp.strided.store(<4 x i32> zeroinitializer, ptr @a, i64 4, <4 x i1> splat (i1 true), i32 poison)
  ret void
}

define void @poison_mask_lane() {
; POISON-MASK-LANE: Entering function: poison_mask_lane
; POISON-MASK-LANE-NEXT: Stacktrace:
; POISON-MASK-LANE-NEXT: #0   call void @llvm.experimental.vp.strided.store.v4i32.p0.i64(<4 x i32> zeroinitializer, ptr @a, i64 4, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2) at @poison_mask_lane <stdin>:{{[0-9]+}}
; POISON-MASK-LANE-NEXT: Immediate UB detected: Poison mask lane in llvm.experimental.vp.strided.store.
; POISON-MASK-LANE-NEXT: error: Execution of function 'poison_mask_lane' failed.
  call void @llvm.experimental.vp.strided.store(<4 x i32> zeroinitializer, ptr @a, i64 4, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2)
  ret void
}

; A poison stride makes every lane's pointer except lane 0's poison.
define void @poison_stride() {
; POISON-STRIDE: Entering function: poison_stride
; POISON-STRIDE-NEXT: Stacktrace:
; POISON-STRIDE-NEXT: #0   call void @llvm.experimental.vp.strided.store.v4i32.p0.i64(<4 x i32> zeroinitializer, ptr @a, i64 poison, <4 x i1> splat (i1 true), i32 2) at @poison_stride <stdin>:{{[0-9]+}}
; POISON-STRIDE-NEXT: Immediate UB detected: Invalid memory access with a poison pointer.
; POISON-STRIDE-NEXT: error: Execution of function 'poison_stride' failed.
  call void @llvm.experimental.vp.strided.store(<4 x i32> zeroinitializer, ptr @a, i64 poison, <4 x i1> splat (i1 true), i32 2)
  ret void
}

define void @constant_mem() {
; CONSTANT-MEM: Entering function: constant_mem
; CONSTANT-MEM-NEXT: Stacktrace:
; CONSTANT-MEM-NEXT: #0   call void @llvm.experimental.vp.strided.store.v4i32.p0.i64(<4 x i32> zeroinitializer, ptr @c, i64 4, <4 x i1> splat (i1 true), i32 1) at @constant_mem <stdin>:{{[0-9]+}}
; CONSTANT-MEM-NEXT: Immediate UB detected: Try to write to a constant memory object: ptr 0x34 [@c].
; CONSTANT-MEM-NEXT: error: Execution of function 'constant_mem' failed.
  call void @llvm.experimental.vp.strided.store(<4 x i32> zeroinitializer, ptr @c, i64 4, <4 x i1> splat (i1 true), i32 1)
  ret void
}

; Without an align attribute the base pointer must have the ABI alignment of
; i32, even though lane 0 is disabled and lane 1 is aligned.
define void @misaligned_base() {
; MISALIGNED-BASE: Entering function: misaligned_base
; MISALIGNED-BASE-NEXT: %p = getelementptr i8, ptr @a, i64 2 => ptr 0x12 [@a + 2]
; MISALIGNED-BASE-NEXT: Stacktrace:
; MISALIGNED-BASE-NEXT: #0   call void @llvm.experimental.vp.strided.store.v4i32.p0.i64(<4 x i32> zeroinitializer, ptr %p, i64 2, <4 x i1> <i1 false, i1 true, i1 false, i1 false>, i32 2) at @misaligned_base <stdin>:{{[0-9]+}}
; MISALIGNED-BASE-NEXT: Immediate UB detected: Invalid memory access with a poison pointer.
; MISALIGNED-BASE-NEXT: error: Execution of function 'misaligned_base' failed.
  %p = getelementptr i8, ptr @a, i64 2
  call void @llvm.experimental.vp.strided.store(<4 x i32> zeroinitializer, ptr %p, i64 2, <4 x i1> <i1 false, i1 true, i1 false, i1 false>, i32 2)
  ret void
}
