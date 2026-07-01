; RUN: sed 's/OP/memcpy_overlap/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=OVERLAP
; RUN: sed 's/OP/memcpy_inline_poison_src/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=COPY-SRC
; RUN: sed 's/OP/memset_poison_dst/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=SET-DST
; RUN: sed 's/OP/memset_poison_len/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=SET-LEN

define void @main() {
  call void @OP()
  ret void
}

define void @memcpy_overlap() {
  %buf = alloca [4 x i8], align 1
  %dst = getelementptr i8, ptr %buf, i64 1
  call void @llvm.memcpy.p0.p0.i8(ptr %dst, ptr %buf, i8 2, i1 false)
  ret void
}

define void @memcpy_inline_poison_src() {
  %dst = alloca [4 x i8], align 1
  call void @llvm.memcpy.inline.p0.p0.i8(ptr %dst, ptr poison, i8 1, i1 false)
  ret void
}

define void @memset_poison_dst() {
  call void @llvm.memset.p0.i8(ptr poison, i8 0, i8 1, i1 false)
  ret void
}

define void @memset_poison_len() {
  %dst = alloca [4 x i8], align 1
  call void @llvm.memset.p0.i8(ptr %dst, i8 0, i8 poison, i1 false)
  ret void
}

; OVERLAP: Entering function: main
; OVERLAP: Entering function: memcpy_overlap
; OVERLAP: Immediate UB detected: memcpy with overlapping source and destination.
; OVERLAP: error: Execution of function 'main' failed.

; COPY-SRC: Entering function: main
; COPY-SRC: Entering function: memcpy_inline_poison_src
; COPY-SRC: Immediate UB detected: Memory transfer intrinsic with poison source pointer.
; COPY-SRC: error: Execution of function 'main' failed.

; SET-DST: Entering function: main
; SET-DST: Entering function: memset_poison_dst
; SET-DST: Immediate UB detected: memset called with poison destination pointer.
; SET-DST: error: Execution of function 'main' failed.

; SET-LEN: Entering function: main
; SET-LEN: Entering function: memset_poison_len
; SET-LEN: Immediate UB detected: memset called with poison length.
; SET-LEN: error: Execution of function 'main' failed.
