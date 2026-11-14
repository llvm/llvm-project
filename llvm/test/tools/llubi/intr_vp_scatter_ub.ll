; RUN: not llubi --verbose --entry-function=evl_too_large < %s 2>&1 | FileCheck %s --check-prefix=EVL
; RUN: not llubi --verbose --entry-function=poison_evl < %s 2>&1 | FileCheck %s --check-prefix=POISON-EVL
; RUN: not llubi --verbose --entry-function=poison_mask_lane < %s 2>&1 | FileCheck %s --check-prefix=POISON-MASK-LANE
; RUN: not llubi --verbose --entry-function=constant_mem < %s 2>&1 | FileCheck %s --check-prefix=CONSTANT-MEM
; RUN: not llubi --verbose --entry-function=misaligned < %s 2>&1 | FileCheck %s --check-prefix=MISALIGNED

@a = global [4 x i32] [i32 0, i32 1, i32 2, i32 3]
@c = constant [4 x i32] [i32 0, i32 1, i32 2, i32 3]

define void @evl_too_large() {
; EVL: Entering function: evl_too_large
; EVL-NEXT:   %ptrs = getelementptr i32, ptr @a, <4 x i64> <i64 0, i64 1, i64 2, i64 3> => { ptr 0x8 [@a], ptr 0xC [@a + 4], ptr 0x10 [@a + 8], ptr 0x14 [@a + 12] }
; EVL-NEXT: Stacktrace:
; EVL-NEXT: #0   call void @llvm.vp.scatter.v4i32.v4p0(<4 x i32> zeroinitializer, <4 x ptr> %ptrs, <4 x i1> splat (i1 true), i32 5) at @evl_too_large <stdin>:{{[0-9]+}}
; EVL-NEXT: Immediate UB detected: llvm.vp.scatter with explicit vector length 5 greater than the number of elements 4.
; EVL-NEXT: error: Execution of function 'evl_too_large' failed.
  %ptrs = getelementptr i32, ptr @a, <4 x i64> <i64 0, i64 1, i64 2, i64 3>
  call void @llvm.vp.scatter(<4 x i32> zeroinitializer, <4 x ptr> %ptrs, <4 x i1> splat (i1 true), i32 5)
  ret void
}

define void @poison_evl() {
; POISON-EVL: Entering function: poison_evl
; POISON-EVL-NEXT:   %ptrs = getelementptr i32, ptr @a, <4 x i64> <i64 0, i64 1, i64 2, i64 3> => { ptr 0x8 [@a], ptr 0xC [@a + 4], ptr 0x10 [@a + 8], ptr 0x14 [@a + 12] }
; POISON-EVL-NEXT: Stacktrace:
; POISON-EVL-NEXT: #0   call void @llvm.vp.scatter.v4i32.v4p0(<4 x i32> zeroinitializer, <4 x ptr> %ptrs, <4 x i1> splat (i1 true), i32 poison) at @poison_evl <stdin>:{{[0-9]+}}
; POISON-EVL-NEXT: Immediate UB detected: llvm.vp.scatter with poison explicit vector length.
; POISON-EVL-NEXT: error: Execution of function 'poison_evl' failed.
  %ptrs = getelementptr i32, ptr @a, <4 x i64> <i64 0, i64 1, i64 2, i64 3>
  call void @llvm.vp.scatter(<4 x i32> zeroinitializer, <4 x ptr> %ptrs, <4 x i1> splat (i1 true), i32 poison)
  ret void
}

define void @poison_mask_lane() {
; POISON-MASK-LANE: Entering function: poison_mask_lane
; POISON-MASK-LANE-NEXT:   %ptrs = getelementptr i32, ptr @a, <4 x i64> <i64 0, i64 1, i64 2, i64 3> => { ptr 0x8 [@a], ptr 0xC [@a + 4], ptr 0x10 [@a + 8], ptr 0x14 [@a + 12] }
; POISON-MASK-LANE-NEXT: Stacktrace:
; POISON-MASK-LANE-NEXT: #0   call void @llvm.vp.scatter.v4i32.v4p0(<4 x i32> zeroinitializer, <4 x ptr> %ptrs, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2) at @poison_mask_lane <stdin>:{{[0-9]+}}
; POISON-MASK-LANE-NEXT: Immediate UB detected: Poison mask lane in llvm.vp.scatter.
; POISON-MASK-LANE-NEXT: error: Execution of function 'poison_mask_lane' failed.
  %ptrs = getelementptr i32, ptr @a, <4 x i64> <i64 0, i64 1, i64 2, i64 3>
  call void @llvm.vp.scatter(<4 x i32> zeroinitializer, <4 x ptr> %ptrs, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, i32 2)
  ret void
}

define void @constant_mem() {
; CONSTANT-MEM: Entering function: constant_mem
; CONSTANT-MEM-NEXT:   %ptrs = getelementptr i32, ptr @c, <4 x i64> <i64 0, i64 1, i64 2, i64 3> => { ptr 0x1C [@c], ptr 0x20 [@c + 4], ptr 0x24 [@c + 8], ptr 0x28 [@c + 12] }
; CONSTANT-MEM-NEXT: Stacktrace:
; CONSTANT-MEM-NEXT: #0   call void @llvm.vp.scatter.v4i32.v4p0(<4 x i32> zeroinitializer, <4 x ptr> %ptrs, <4 x i1> splat (i1 true), i32 1) at @constant_mem <stdin>:{{[0-9]+}}
; CONSTANT-MEM-NEXT: Immediate UB detected: Try to write to a constant memory object: ptr 0x1C [@c].
; CONSTANT-MEM-NEXT: error: Execution of function 'constant_mem' failed.
  %ptrs = getelementptr i32, ptr @c, <4 x i64> <i64 0, i64 1, i64 2, i64 3>
  call void @llvm.vp.scatter(<4 x i32> zeroinitializer, <4 x ptr> %ptrs, <4 x i1> splat (i1 true), i32 1)
  ret void
}

; Without an align attribute each lane's pointer must have the ABI alignment of
; i32, so the misaligned lane's pointer is poison.
define void @misaligned() {
; MISALIGNED: Entering function: misaligned
; MISALIGNED-NEXT:   %ptrs = getelementptr i8, ptr @a, <2 x i64> <i64 0, i64 2> => { ptr 0x8 [@a], ptr 0xA [@a + 2] }
; MISALIGNED-NEXT: Stacktrace:
; MISALIGNED-NEXT: #0   call void @llvm.vp.scatter.v2i32.v2p0(<2 x i32> zeroinitializer, <2 x ptr> %ptrs, <2 x i1> splat (i1 true), i32 2) at @misaligned <stdin>:{{[0-9]+}}
; MISALIGNED-NEXT: Immediate UB detected: Invalid memory access with a poison pointer.
; MISALIGNED-NEXT: error: Execution of function 'misaligned' failed.
  %ptrs = getelementptr i8, ptr @a, <2 x i64> <i64 0, i64 2>
  call void @llvm.vp.scatter(<2 x i32> zeroinitializer, <2 x ptr> %ptrs, <2 x i1> splat (i1 true), i32 2)
  ret void
}
