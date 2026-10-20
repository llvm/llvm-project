; RUN: not llubi --verbose --entry-function=out_of_bounds < %s 2>&1 | FileCheck %s --check-prefix=OOB
; RUN: not llubi --verbose --entry-function=poison_mask_lane < %s 2>&1 | FileCheck %s --check-prefix=POISON-MASK-LANE
; RUN: not llubi --verbose --entry-function=poison_mask < %s 2>&1 | FileCheck %s --check-prefix=POISON-MASK
; RUN: not llubi --verbose --entry-function=poison_pointer < %s 2>&1 | FileCheck %s --check-prefix=POISON-PTR
; RUN: not llubi --verbose --entry-function=misaligned < %s 2>&1 | FileCheck %s --check-prefix=MISALIGNED
; RUN: not llubi --verbose --entry-function=unsupported_elt_ty < %s 2>&1 | FileCheck %s --check-prefix=ELT-TY

@a = global [4 x i32] [i32 0, i32 1, i32 2, i32 3]

define void @out_of_bounds() {
; OOB: Entering function: out_of_bounds
; OOB-NEXT:   %tail = getelementptr i32, ptr @a, i64 2 => ptr 0x10 [@a + 8]
; OOB-NEXT: Stacktrace:
; OOB-NEXT: #0   %r = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 %tail, <4 x i1> <i1 false, i1 false, i1 true, i1 false>, <4 x i32> poison) at @out_of_bounds <stdin>:{{[0-9]+}}
; OOB-NEXT: Immediate UB detected: Memory access is out of bounds. Accessed size: 4, Address: 0x18, Object base: 0x8, Object size: 16.
; OOB-NEXT: error: Execution of function 'out_of_bounds' failed.
  %tail = getelementptr i32, ptr @a, i64 2
  %r = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 %tail, <4 x i1> <i1 false, i1 false, i1 true, i1 false>, <4 x i32> poison)
  ret void
}

define void @poison_mask_lane() {
; POISON-MASK-LANE: Entering function: poison_mask_lane
; POISON-MASK-LANE-NEXT: Stacktrace:
; POISON-MASK-LANE-NEXT: #0   %r = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 @a, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, <4 x i32> poison) at @poison_mask_lane <stdin>:{{[0-9]+}}
; POISON-MASK-LANE-NEXT: Immediate UB detected: Poison mask lane in llvm.masked.load.
; POISON-MASK-LANE-NEXT: error: Execution of function 'poison_mask_lane' failed.
  %r = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 @a, <4 x i1> <i1 true, i1 poison, i1 false, i1 false>, <4 x i32> poison)
  ret void
}

define void @poison_mask() {
; POISON-MASK: Entering function: poison_mask
; POISON-MASK-NEXT: Stacktrace:
; POISON-MASK-NEXT: #0   %r = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 @a, <4 x i1> poison, <4 x i32> poison) at @poison_mask <stdin>:{{[0-9]+}}
; POISON-MASK-NEXT: Immediate UB detected: Poison mask lane in llvm.masked.load.
; POISON-MASK-NEXT: error: Execution of function 'poison_mask' failed.
  %r = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 @a, <4 x i1> poison, <4 x i32> poison)
  ret void
}

define void @poison_pointer() {
; POISON-PTR: Entering function: poison_pointer
; POISON-PTR-NEXT: Stacktrace:
; POISON-PTR-NEXT: #0   %r = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 poison, <4 x i1> <i1 false, i1 true, i1 false, i1 false>, <4 x i32> poison) at @poison_pointer <stdin>:{{[0-9]+}}
; POISON-PTR-NEXT: Immediate UB detected: llvm.masked.load with poison pointer.
; POISON-PTR-NEXT: error: Execution of function 'poison_pointer' failed.
  %r = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 poison, <4 x i1> <i1 false, i1 true, i1 false, i1 false>, <4 x i32> poison)
  ret void
}

; The align attribute turns the misaligned pointer into poison.
define void @misaligned() {
; MISALIGNED: Entering function: misaligned
; MISALIGNED-NEXT:   %p = getelementptr i32, ptr @a, i64 1 => ptr 0xC [@a + 4]
; MISALIGNED-NEXT: Stacktrace:
; MISALIGNED-NEXT: #0   %r = call <2 x i32> @llvm.masked.load.v2i32.p0(ptr align 8 %p, <2 x i1> <i1 true, i1 false>, <2 x i32> poison) at @misaligned <stdin>:{{[0-9]+}}
; MISALIGNED-NEXT: Immediate UB detected: llvm.masked.load with poison pointer.
; MISALIGNED-NEXT: error: Execution of function 'misaligned' failed.
  %p = getelementptr i32, ptr @a, i64 1
  %r = call <2 x i32> @llvm.masked.load.v2i32.p0(ptr align 8 %p, <2 x i1> <i1 true, i1 false>, <2 x i32> poison)
  ret void
}

define void @unsupported_elt_ty() {
; ELT-TY: Entering function: unsupported_elt_ty
; ELT-TY-NEXT: Stacktrace:
; ELT-TY-NEXT: #0   %r = call <8 x i1> @llvm.masked.load.v8i1.p0(ptr align 1 @a, <8 x i1> splat (i1 true), <8 x i1> poison) at @unsupported_elt_ty <stdin>:{{[0-9]+}}
; ELT-TY-NEXT: Error: llvm.masked.load with element type i1 is not supported.
; ELT-TY-NEXT: error: Execution of function 'unsupported_elt_ty' failed.
  %r = call <8 x i1> @llvm.masked.load.v8i1.p0(ptr align 1 @a, <8 x i1> splat (i1 true), <8 x i1> poison)
  ret void
}
