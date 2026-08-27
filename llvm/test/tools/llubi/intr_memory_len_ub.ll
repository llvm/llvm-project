; RUN: sed 's/OP/memcpy_len_overflow/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: sed 's/OP/memset_len_overflow/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=SET

define void @main() {
  call void @OP()
  ret void
}

define void @memcpy_len_overflow() {
  %src = alloca [4 x i8], align 1
  %dst = alloca [4 x i8], align 1
  call void @llvm.memcpy.p0.p0.i128(ptr %dst, ptr %src, i128 18446744073709551616, i1 false)
  ret void
}

define void @memset_len_overflow() {
  %dst = alloca [4 x i8], align 1
  call void @llvm.memset.p0.i128(ptr %dst, i8 0, i128 18446744073709551616, i1 false)
  ret void
}

; COPY: Entering function: main
; COPY: Entering function: memcpy_len_overflow
; COPY: Immediate UB detected: Memory transfer intrinsic length overflows uint64_t.
; COPY: error: Execution of function 'main' failed.

; SET: Entering function: main
; SET: Entering function: memset_len_overflow
; SET: Immediate UB detected: memset called with length overflows uint64_t.
; SET: error: Execution of function 'main' failed.
