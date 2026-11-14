; RUN: not llubi --verbose --entry-function=evl_too_large < %s 2>&1 | FileCheck %s --check-prefix=EVL
; RUN: not llubi --verbose --entry-function=poison_evl < %s 2>&1 | FileCheck %s --check-prefix=POISON-EVL
; RUN: not llubi --verbose --entry-function=poison_mask_lane < %s 2>&1 | FileCheck %s --check-prefix=POISON-MASK-LANE
; RUN: not llubi --verbose --entry-function=constant_mem < %s 2>&1 | FileCheck %s --check-prefix=CONSTANT-MEM
; RUN: not llubi --verbose --entry-function=misaligned < %s 2>&1 | FileCheck %s --check-prefix=MISALIGNED

@a = global [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16
@c = constant [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16

define void @evl_too_large() {
; EVL: Entering function: evl_too_large
; EVL-NEXT: Stacktrace:
; EVL-NEXT: #0   call void @llvm.vp.store.v4i32.p0(<4 x i32> zeroinitializer, ptr @a, <4 x i1> splat (i1 true), i32 5) at @evl_too_large <stdin>:{{[0-9]+}}
; EVL-NEXT: Immediate UB detected: llvm.vp.store with explicit vector length 5 greater than the number of elements 4.
; EVL-NEXT: error: Execution of function 'evl_too_large' failed.
  call void @llvm.vp.store(<4 x i32> zeroinitializer, ptr @a, <4 x i1> splat (i1 true), i32 5)
  ret void
}

define void @poison_evl() {
; POISON-EVL: Entering function: poison_evl
; POISON-EVL-NEXT: Stacktrace:
; POISON-EVL-NEXT: #0   call void @llvm.vp.store.v4i32.p0(<4 x i32> zeroinitializer, ptr @a, <4 x i1> splat (i1 true), i32 poison) at @poison_evl <stdin>:{{[0-9]+}}
; POISON-EVL-NEXT: Immediate UB detected: llvm.vp.store with poison explicit vector length.
; POISON-EVL-NEXT: error: Execution of function 'poison_evl' failed.
  call void @llvm.vp.store(<4 x i32> zeroinitializer, ptr @a, <4 x i1> splat (i1 true), i32 poison)
  ret void
}

define void @poison_mask_lane() {
; POISON-MASK-LANE: Entering function: poison_mask_lane
; POISON-MASK-LANE-NEXT: Stacktrace:
; POISON-MASK-LANE-NEXT: #0   call void @llvm.vp.store.v4i32.p0(<4 x i32> zeroinitializer, ptr @a, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2) at @poison_mask_lane <stdin>:{{[0-9]+}}
; POISON-MASK-LANE-NEXT: Immediate UB detected: Poison mask lane in llvm.vp.store.
; POISON-MASK-LANE-NEXT: error: Execution of function 'poison_mask_lane' failed.
  call void @llvm.vp.store(<4 x i32> zeroinitializer, ptr @a, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2)
  ret void
}

define void @constant_mem() {
; CONSTANT-MEM: Entering function: constant_mem
; CONSTANT-MEM-NEXT: Stacktrace:
; CONSTANT-MEM-NEXT: #0   call void @llvm.vp.store.v4i32.p0(<4 x i32> zeroinitializer, ptr @c, <4 x i1> splat (i1 true), i32 1) at @constant_mem <stdin>:{{[0-9]+}}
; CONSTANT-MEM-NEXT: Immediate UB detected: Try to write to a constant memory object: ptr 0x30 [@c].
; CONSTANT-MEM-NEXT: error: Execution of function 'constant_mem' failed.
  call void @llvm.vp.store(<4 x i32> zeroinitializer, ptr @c, <4 x i1> splat (i1 true), i32 1)
  ret void
}

; Without an align attribute the pointer must have the ABI alignment of
; <2 x i32>, so the misaligned pointer is poison.
define void @misaligned() {
; MISALIGNED: Entering function: misaligned
; MISALIGNED-NEXT:   %p = getelementptr i32, ptr @a, i64 1 => ptr 0x14 [@a + 4]
; MISALIGNED-NEXT: Stacktrace:
; MISALIGNED-NEXT: #0   call void @llvm.vp.store.v2i32.p0(<2 x i32> zeroinitializer, ptr %p, <2 x i1> splat (i1 true), i32 1) at @misaligned <stdin>:{{[0-9]+}}
; MISALIGNED-NEXT: Immediate UB detected: Invalid memory access with a poison pointer.
; MISALIGNED-NEXT: error: Execution of function 'misaligned' failed.
  %p = getelementptr i32, ptr @a, i64 1
  call void @llvm.vp.store(<2 x i32> zeroinitializer, ptr %p, <2 x i1> splat (i1 true), i32 1)
  ret void
}
