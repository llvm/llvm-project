; RUN: not llubi --entry-function=sdiv_zero --verbose < %s 2>&1 | FileCheck %s --check-prefix=SDIV_ZERO
; RUN: not llubi --entry-function=sdiv_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=SDIV_POISON
; RUN: not llubi --entry-function=udiv_zero --verbose < %s 2>&1 | FileCheck %s --check-prefix=UDIV_ZERO
; RUN: not llubi --entry-function=udiv_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=UDIV_POISON
; RUN: not llubi --entry-function=srem_zero --verbose < %s 2>&1 | FileCheck %s --check-prefix=SREM_ZERO
; RUN: not llubi --entry-function=srem_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=SREM_POISON
; RUN: not llubi --entry-function=urem_zero --verbose < %s 2>&1 | FileCheck %s --check-prefix=UREM_ZERO
; RUN: not llubi --entry-function=urem_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=UREM_POISON
; RUN: not llubi --entry-function=sdiv_overflow --verbose < %s 2>&1 | FileCheck %s --check-prefix=SDIV_OVERFLOW
; RUN: not llubi --entry-function=srem_overflow --verbose < %s 2>&1 | FileCheck %s --check-prefix=SREM_OVERFLOW
; RUN: not llubi --entry-function=arith_evl_large --verbose < %s 2>&1 | FileCheck %s --check-prefix=ARITH_EVL_LARGE
; RUN: not llubi --entry-function=reduce_evl_large --verbose < %s 2>&1 | FileCheck %s --check-prefix=REDUCE_EVL_LARGE
; RUN: not llubi --entry-function=cttz_evl_large --verbose < %s 2>&1 | FileCheck %s --check-prefix=CTTZ_EVL_LARGE
; RUN: not llubi --entry-function=arith_evl_unsigned --verbose < %s 2>&1 | FileCheck %s --check-prefix=ARITH_EVL_UNSIGNED
; RUN: not llubi --entry-function=reduce_evl_unsigned --verbose < %s 2>&1 | FileCheck %s --check-prefix=REDUCE_EVL_UNSIGNED
; RUN: not llubi --entry-function=cttz_evl_unsigned --verbose < %s 2>&1 | FileCheck %s --check-prefix=CTTZ_EVL_UNSIGNED
; RUN: not llubi --entry-function=arith_evl_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=ARITH_EVL_POISON
; RUN: not llubi --entry-function=reduce_evl_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=REDUCE_EVL_POISON
; RUN: not llubi --entry-function=cttz_evl_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=CTTZ_EVL_POISON
; RUN: not llubi --entry-function=sdiv_mask_zero --verbose < %s 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llubi --entry-function=sdiv_mask_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llubi --entry-function=udiv_mask_zero --verbose < %s 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llubi --entry-function=udiv_mask_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llubi --entry-function=srem_mask_zero --verbose < %s 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llubi --entry-function=srem_mask_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llubi --entry-function=urem_mask_zero --verbose < %s 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llubi --entry-function=urem_mask_poison --verbose < %s 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llubi --entry-function=sdiv_mask_overflow --verbose < %s 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: not llubi --entry-function=sdiv_mask_poison_lhs --verbose < %s 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: not llubi --entry-function=srem_mask_overflow --verbose < %s 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: not llubi --entry-function=srem_mask_poison_lhs --verbose < %s 2>&1 | FileCheck %s --check-prefix=OVERFLOW

define void @sdiv_zero() {
  %r = call <2 x i32> @llvm.vp.sdiv.v2i32(<2 x i32> splat(i32 10), <2 x i32> <i32 1, i32 0>, <2 x i1> splat(i1 true), i32 2)
  ret void
}

define void @sdiv_poison() {
  %r = call <2 x i32> @llvm.vp.sdiv.v2i32(<2 x i32> splat(i32 10), <2 x i32> poison, <2 x i1> splat(i1 true), i32 1)
  ret void
}

define void @udiv_zero() {
  %r = call <2 x i32> @llvm.vp.udiv.v2i32(<2 x i32> splat(i32 10), <2 x i32> <i32 1, i32 0>, <2 x i1> splat(i1 true), i32 2)
  ret void
}

define void @udiv_poison() {
  %r = call <2 x i32> @llvm.vp.udiv.v2i32(<2 x i32> splat(i32 10), <2 x i32> poison, <2 x i1> splat(i1 true), i32 1)
  ret void
}

define void @srem_zero() {
  %r = call <2 x i32> @llvm.vp.srem.v2i32(<2 x i32> splat(i32 10), <2 x i32> <i32 1, i32 0>, <2 x i1> splat(i1 true), i32 2)
  ret void
}

define void @srem_poison() {
  %r = call <2 x i32> @llvm.vp.srem.v2i32(<2 x i32> splat(i32 10), <2 x i32> poison, <2 x i1> splat(i1 true), i32 1)
  ret void
}

define void @urem_zero() {
  %r = call <2 x i32> @llvm.vp.urem.v2i32(<2 x i32> splat(i32 10), <2 x i32> <i32 1, i32 0>, <2 x i1> splat(i1 true), i32 2)
  ret void
}

define void @urem_poison() {
  %r = call <2 x i32> @llvm.vp.urem.v2i32(<2 x i32> splat(i32 10), <2 x i32> poison, <2 x i1> splat(i1 true), i32 1)
  ret void
}

define void @sdiv_overflow() {
  %r = call <2 x i32> @llvm.vp.sdiv.v2i32(<2 x i32> splat(i32 -2147483648), <2 x i32> splat(i32 -1), <2 x i1> splat(i1 true), i32 1)
  ret void
}

define void @srem_overflow() {
  %r = call <2 x i32> @llvm.vp.srem.v2i32(<2 x i32> splat(i32 -2147483648), <2 x i32> splat(i32 -1), <2 x i1> splat(i1 true), i32 1)
  ret void
}

define void @arith_evl_large() {
  %r = call <2 x i32> @llvm.vp.udiv.v2i32(<2 x i32> poison, <2 x i32> poison, <2 x i1> zeroinitializer, i32 3)
  ret void
}

define void @reduce_evl_large() {
  %r = call i32 @llvm.vp.reduce.add.v2i32(i32 1, <2 x i32> poison, <2 x i1> zeroinitializer, i32 3)
  ret void
}

define void @cttz_evl_large() {
  %r = call i32 @llvm.vp.cttz.elts.i32.v2i32(<2 x i32> poison, i1 false, <2 x i1> zeroinitializer, i32 3)
  ret void
}

define void @arith_evl_unsigned() {
  %r = call <2 x i32> @llvm.vp.udiv.v2i32(<2 x i32> poison, <2 x i32> poison, <2 x i1> zeroinitializer, i32 -1)
  ret void
}

define void @reduce_evl_unsigned() {
  %r = call i32 @llvm.vp.reduce.add.v2i32(i32 1, <2 x i32> poison, <2 x i1> zeroinitializer, i32 -1)
  ret void
}

define void @cttz_evl_unsigned() {
  %r = call i32 @llvm.vp.cttz.elts.i32.v2i32(<2 x i32> poison, i1 false, <2 x i1> zeroinitializer, i32 -1)
  ret void
}

define void @arith_evl_poison() {
  %r = call <2 x i32> @llvm.vp.udiv.v2i32(<2 x i32> poison, <2 x i32> poison, <2 x i1> zeroinitializer, i32 poison)
  ret void
}

define void @reduce_evl_poison() {
  %r = call i32 @llvm.vp.reduce.add.v2i32(i32 1, <2 x i32> poison, <2 x i1> zeroinitializer, i32 poison)
  ret void
}

define void @cttz_evl_poison() {
  %r = call i32 @llvm.vp.cttz.elts.i32.v2i32(<2 x i32> poison, i1 false, <2 x i1> zeroinitializer, i32 poison)
  ret void
}

define void @sdiv_mask_zero() {
  %r = call <2 x i32> @llvm.vp.sdiv.v2i32(<2 x i32> splat(i32 8), <2 x i32> <i32 2, i32 0>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @sdiv_mask_poison() {
  %r = call <2 x i32> @llvm.vp.sdiv.v2i32(<2 x i32> splat(i32 8), <2 x i32> <i32 2, i32 poison>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @udiv_mask_zero() {
  %r = call <2 x i32> @llvm.vp.udiv.v2i32(<2 x i32> splat(i32 8), <2 x i32> <i32 2, i32 0>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @udiv_mask_poison() {
  %r = call <2 x i32> @llvm.vp.udiv.v2i32(<2 x i32> splat(i32 8), <2 x i32> <i32 2, i32 poison>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @srem_mask_zero() {
  %r = call <2 x i32> @llvm.vp.srem.v2i32(<2 x i32> splat(i32 8), <2 x i32> <i32 2, i32 0>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @srem_mask_poison() {
  %r = call <2 x i32> @llvm.vp.srem.v2i32(<2 x i32> splat(i32 8), <2 x i32> <i32 2, i32 poison>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @urem_mask_zero() {
  %r = call <2 x i32> @llvm.vp.urem.v2i32(<2 x i32> splat(i32 8), <2 x i32> <i32 2, i32 0>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @urem_mask_poison() {
  %r = call <2 x i32> @llvm.vp.urem.v2i32(<2 x i32> splat(i32 8), <2 x i32> <i32 2, i32 poison>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @sdiv_mask_overflow() {
  %r = call <2 x i32> @llvm.vp.sdiv.v2i32(<2 x i32> <i32 8, i32 -2147483648>, <2 x i32> <i32 2, i32 -1>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @sdiv_mask_poison_lhs() {
  %r = call <2 x i32> @llvm.vp.sdiv.v2i32(<2 x i32> <i32 8, i32 poison>, <2 x i32> <i32 2, i32 -1>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @srem_mask_overflow() {
  %r = call <2 x i32> @llvm.vp.srem.v2i32(<2 x i32> <i32 8, i32 -2147483648>, <2 x i32> <i32 2, i32 -1>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

define void @srem_mask_poison_lhs() {
  %r = call <2 x i32> @llvm.vp.srem.v2i32(<2 x i32> <i32 8, i32 poison>, <2 x i32> <i32 2, i32 -1>, <2 x i1> <i1 true, i1 poison>, i32 2)
  ret void
}

; SDIV_ZERO: Immediate UB detected: Division by zero.
; SDIV_ZERO-NEXT: error: Execution of function 'sdiv_zero' failed.
; SDIV_POISON: Immediate UB detected: Division by zero (refine RHS to 0).
; SDIV_POISON-NEXT: error: Execution of function 'sdiv_poison' failed.
; UDIV_ZERO: Immediate UB detected: Division by zero.
; UDIV_ZERO-NEXT: error: Execution of function 'udiv_zero' failed.
; UDIV_POISON: Immediate UB detected: Division by zero (refine RHS to 0).
; UDIV_POISON-NEXT: error: Execution of function 'udiv_poison' failed.
; SREM_ZERO: Immediate UB detected: Division by zero.
; SREM_ZERO-NEXT: error: Execution of function 'srem_zero' failed.
; SREM_POISON: Immediate UB detected: Division by zero (refine RHS to 0).
; SREM_POISON-NEXT: error: Execution of function 'srem_poison' failed.
; UREM_ZERO: Immediate UB detected: Division by zero.
; UREM_ZERO-NEXT: error: Execution of function 'urem_zero' failed.
; UREM_POISON: Immediate UB detected: Division by zero (refine RHS to 0).
; UREM_POISON-NEXT: error: Execution of function 'urem_poison' failed.
; SDIV_OVERFLOW: Immediate UB detected: Signed division overflow.
; SDIV_OVERFLOW-NEXT: error: Execution of function 'sdiv_overflow' failed.
; SREM_OVERFLOW: Immediate UB detected: Signed division overflow.
; SREM_OVERFLOW-NEXT: error: Execution of function 'srem_overflow' failed.
; ARITH_EVL_LARGE: Immediate UB detected: VP explicit vector length 3 exceeds the number of vector elements 2.
; ARITH_EVL_LARGE-NEXT: error: Execution of function 'arith_evl_large' failed.
; REDUCE_EVL_LARGE: Immediate UB detected: VP explicit vector length 3 exceeds the number of vector elements 2.
; REDUCE_EVL_LARGE-NEXT: error: Execution of function 'reduce_evl_large' failed.
; CTTZ_EVL_LARGE: Immediate UB detected: VP explicit vector length 3 exceeds the number of vector elements 2.
; CTTZ_EVL_LARGE-NEXT: error: Execution of function 'cttz_evl_large' failed.
; ARITH_EVL_UNSIGNED: Immediate UB detected: VP explicit vector length -1 exceeds the number of vector elements 2.
; ARITH_EVL_UNSIGNED-NEXT: error: Execution of function 'arith_evl_unsigned' failed.
; REDUCE_EVL_UNSIGNED: Immediate UB detected: VP explicit vector length -1 exceeds the number of vector elements 2.
; REDUCE_EVL_UNSIGNED-NEXT: error: Execution of function 'reduce_evl_unsigned' failed.
; CTTZ_EVL_UNSIGNED: Immediate UB detected: VP explicit vector length -1 exceeds the number of vector elements 2.
; CTTZ_EVL_UNSIGNED-NEXT: error: Execution of function 'cttz_evl_unsigned' failed.
; ARITH_EVL_POISON: Immediate UB detected: Poison explicit vector length in VP intrinsic.
; ARITH_EVL_POISON-NEXT: error: Execution of function 'arith_evl_poison' failed.
; REDUCE_EVL_POISON: Immediate UB detected: Poison explicit vector length in VP intrinsic.
; REDUCE_EVL_POISON-NEXT: error: Execution of function 'reduce_evl_poison' failed.
; CTTZ_EVL_POISON: Immediate UB detected: Poison explicit vector length in VP intrinsic.
; CTTZ_EVL_POISON-NEXT: error: Execution of function 'cttz_evl_poison' failed.
; ZERO: Immediate UB detected: Division by zero
; ZERO-NEXT: error: Execution of function
; OVERFLOW: Immediate UB detected: Signed division overflow
; OVERFLOW-NEXT: error: Execution of function
