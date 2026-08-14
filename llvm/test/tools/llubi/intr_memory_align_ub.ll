; RUN: sed 's/OP/memcpy_misaligned_dst/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=COPY-DST
; RUN: sed 's/OP/memcpy_misaligned_src/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=COPY-SRC
; RUN: sed 's/OP/memset_misaligned_dst/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=SET-DST

define void @main() {
  call void @OP()
  ret void
}

define void @memcpy_misaligned_dst() {
  %src = alloca [4 x i8], align 2
  %dst = alloca [4 x i8], align 2
  %dst1 = getelementptr i8, ptr %dst, i64 1
  call void @llvm.memcpy.p0.p0.i8(ptr align 2 %dst1, ptr align 1 %src, i8 1, i1 false)
  ret void
}

define void @memcpy_misaligned_src() {
  %src = alloca [4 x i8], align 1
  %dst = alloca [4 x i8], align 2
  %src1 = getelementptr i8, ptr %src, i64 1
  call void @llvm.memcpy.p0.p0.i8(ptr align 1 %dst, ptr align 2 %src1, i8 1, i1 false)
  ret void
}

define void @memset_misaligned_dst() {
  %dst = alloca [4 x i8], align 1
  %dst1 = getelementptr i8, ptr %dst, i64 1
  call void @llvm.memset.p0.i8(ptr align 2 %dst1, i8 0, i8 1, i1 false)
  ret void
}

; COPY-DST: Entering function: main
; COPY-DST: Entering function: memcpy_misaligned_dst
; COPY-DST: Immediate UB detected: Memory transfer intrinsic with poison destination pointer.
; COPY-DST: error: Execution of function 'main' failed.

; COPY-SRC: Entering function: main
; COPY-SRC: Entering function: memcpy_misaligned_src
; COPY-SRC: Immediate UB detected: Memory transfer intrinsic with poison source pointer.
; COPY-SRC: error: Execution of function 'main' failed.

; SET-DST: Entering function: main
; SET-DST: Entering function: memset_misaligned_dst
; SET-DST: Immediate UB detected: memset called with poison destination pointer.
; SET-DST: error: Execution of function 'main' failed.
