; RUN: not llubi --verbose --entry-function=indirect_mismatch < %s 2>&1 | FileCheck %s --check-prefix=INDIRECT
; RUN: not llubi --verbose --entry-function=direct_mismatch < %s 2>&1 | FileCheck %s --check-prefix=DIRECT
; RUN: not llubi --verbose --entry-function=callsite_fastcc < %s 2>&1 | FileCheck %s --check-prefix=CALLSITE-FASTCC
; RUN: llubi --verbose --entry-function=matching_cc < %s 2>&1 | FileCheck %s --check-prefix=MATCHING
; RUN: not llubi --verbose --entry-function=intrinsic_anyregcc < %s 2>&1 | FileCheck %s --check-prefix=INTRINSIC

@fp_fastcc = global ptr @callee_fastcc

define fastcc void @callee_fastcc() {
  ret void
}

define void @callee_default() {
  ret void
}

define void @indirect_mismatch() {
; INDIRECT: Entering function: indirect_mismatch
; INDIRECT-NEXT:   %p = load ptr, ptr @fp_fastcc, align 8 => ptr 0x8 [@callee_fastcc]
; INDIRECT-NEXT: Stacktrace:
; INDIRECT-NEXT: #0   call void %p() at @indirect_mismatch <stdin>:{{[0-9]+}}
; INDIRECT-NEXT: Immediate UB detected: Mismatched calling convention between callee and callsite.
; INDIRECT-NEXT: error: Execution of function 'indirect_mismatch' failed.
  %p = load ptr, ptr @fp_fastcc
  call void %p()
  ret void
}

define void @direct_mismatch() {
; DIRECT: Entering function: direct_mismatch
; DIRECT-NEXT: Stacktrace:
; DIRECT-NEXT: #0   call void @callee_fastcc() at @direct_mismatch <stdin>:{{[0-9]+}}
; DIRECT-NEXT: Immediate UB detected: Mismatched calling convention between callee and callsite.
; DIRECT-NEXT: error: Execution of function 'direct_mismatch' failed.
  call void @callee_fastcc()
  ret void
}

define void @callsite_fastcc() {
; CALLSITE-FASTCC: Entering function: callsite_fastcc
; CALLSITE-FASTCC-NEXT: Stacktrace:
; CALLSITE-FASTCC-NEXT: #0   call fastcc void @callee_default() at @callsite_fastcc <stdin>:{{[0-9]+}}
; CALLSITE-FASTCC-NEXT: Immediate UB detected: Mismatched calling convention between callee and callsite.
; CALLSITE-FASTCC-NEXT: error: Execution of function 'callsite_fastcc' failed.
  call fastcc void @callee_default()
  ret void
}

define void @matching_cc() {
; MATCHING: Entering function: matching_cc
; MATCHING-NEXT: Entering function: callee_fastcc
; MATCHING-NEXT:   ret void
; MATCHING-NEXT: Exiting function: callee_fastcc
; MATCHING-NEXT:   call fastcc void @callee_fastcc()
; MATCHING-NEXT:   %p = load ptr, ptr @fp_fastcc, align 8 => ptr 0x8 [@callee_fastcc]
; MATCHING-NEXT: Entering function: callee_fastcc
; MATCHING-NEXT:   ret void
; MATCHING-NEXT: Exiting function: callee_fastcc
; MATCHING-NEXT:   call fastcc void %p()
; MATCHING-NEXT:   ret void
; MATCHING-NEXT: Exiting function: matching_cc
  call fastcc void @callee_fastcc()
  %p = load ptr, ptr @fp_fastcc
  call fastcc void %p()
  ret void
}

declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)

define void @intrinsic_anyregcc() {
; INTRINSIC: Entering function: intrinsic_anyregcc
; INTRINSIC-NEXT: Unrecognized instruction:   call anyregcc void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 0, ptr null, i32 0)
; INTRINSIC-NEXT: error: Execution of function 'intrinsic_anyregcc' failed.
  call anyregcc void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 0, ptr null, i32 0)
  ret void
}
