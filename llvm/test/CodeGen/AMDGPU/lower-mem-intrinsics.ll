; RUN: opt -S -amdgpu-lower-intrinsics %s | FileCheck -check-prefix=OPT %s

declare void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture readonly, i64, i32, i1) #1
declare void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* nocapture, i8 addrspace(3)* nocapture readonly, i32, i32, i1) #1

declare void @llvm.memmove.p1i8.p1i8.i64(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture readonly, i64, i32, i1) #1
declare void @llvm.memset.p1i8.i64(i8 addrspace(1)* nocapture, i8, i64, i32, i1) #1

; Test the upper bound for sizes to leave
; OPT-LABEL: @max_size_small_static_memcpy_caller0(
; OPT: call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 1024, i32 1, i1 false)
define void @max_size_small_static_memcpy_caller0(i8 addrspace(1)* %dst, i8 addrspace(1)* %src) #0 {
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 1024, i32 1, i1 false)
  ret void
}

; Smallest static size which will be expanded
; OPT-LABEL: @min_size_large_static_memcpy_caller0(
; OPT-NOT: call
; OPT: getelementptr
; OPT-NEXT: load i8
; OPT: getelementptr
; OPT-NEXT: store i8
define void @min_size_large_static_memcpy_caller0(i8 addrspace(1)* %dst, i8 addrspace(1)* %src) #0 {
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 1025, i32 1, i1 false)
  ret void
}

; OPT-LABEL: @max_size_small_static_memmove_caller0(
; OPT: call void @llvm.memmove.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 1024, i32 1, i1 false)
define void @max_size_small_static_memmove_caller0(i8 addrspace(1)* %dst, i8 addrspace(1)* %src) #0 {
  call void @llvm.memmove.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 1024, i32 1, i1 false)
  ret void
}

; OPT-LABEL: @min_size_large_static_memmove_caller0(
; OPT-NOT: call
; OPT: getelementptr
; OPT-NEXT: load i8
; OPT: getelementptr
; OPT-NEXT: store i8
define void @min_size_large_static_memmove_caller0(i8 addrspace(1)* %dst, i8 addrspace(1)* %src) #0 {
  call void @llvm.memmove.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 1025, i32 1, i1 false)
  ret void
}

; OPT-LABEL: @max_size_small_static_memset_caller0(
; OPT: call void @llvm.memset.p1i8.i64(i8 addrspace(1)* %dst, i8 %val, i64 1024, i32 1, i1 false)
define void @max_size_small_static_memset_caller0(i8 addrspace(1)* %dst, i8 %val) #0 {
  call void @llvm.memset.p1i8.i64(i8 addrspace(1)* %dst, i8 %val, i64 1024, i32 1, i1 false)
  ret void
}

; OPT-LABEL: @min_size_large_static_memset_caller0(
; OPT-NOT: call
; OPT: getelementptr
; OPT: store i8
define void @min_size_large_static_memset_caller0(i8 addrspace(1)* %dst, i8 %val) #0 {
  call void @llvm.memset.p1i8.i64(i8 addrspace(1)* %dst, i8 %val, i64 1025, i32 1, i1 false)
  ret void
}

; OPT-LABEL: @variable_memcpy_caller0(
; OPT-NOT: call
; OPT: phi
define void @variable_memcpy_caller0(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 %n) #0 {
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 %n, i32 1, i1 false)
  ret void
}

; OPT-LABEL: @variable_memcpy_caller1(
; OPT-NOT: call
; OPT: phi
define void @variable_memcpy_caller1(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 %n) #0 {
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 %n, i32 1, i1 false)
  ret void
}

; OPT-LABEL: @memcpy_multi_use_one_function(
; OPT-NOT: call
; OPT: phi
; OPT-NOT: call
; OPT: phi
; OPT-NOT: call
define void @memcpy_multi_use_one_function(i8 addrspace(1)* %dst0, i8 addrspace(1)* %dst1, i8 addrspace(1)* %src, i64 %n, i64 %m) #0 {
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst0, i8 addrspace(1)* %src, i64 %n, i32 1, i1 false)
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst1, i8 addrspace(1)* %src, i64 %m, i32 1, i1 false)
  ret void
}

; OPT-LABEL: @memcpy_alt_type(
; OPT: phi
; OPT: getelementptr inbounds i8, i8 addrspace(3)*
; OPT: load i8, i8 addrspace(3)*
; OPT: getelementptr inbounds i8, i8 addrspace(1)*
; OPT: store i8
define void @memcpy_alt_type(i8 addrspace(1)* %dst, i8 addrspace(3)* %src, i32 %n) #0 {
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* %dst, i8 addrspace(3)* %src, i32 %n, i32 1, i1 false)
  ret void
}

; One of the uses in the function should be expanded, the other left alone.
; OPT-LABEL: @memcpy_multi_use_one_function_keep_small(
; OPT: getelementptr inbounds i8, i8 addrspace(1)*
; OPT: load i8, i8 addrspace(1)*
; OPT: getelementptr inbounds i8, i8 addrspace(1)*
; OPT: store i8

; OPT: call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst1, i8 addrspace(1)* %src, i64 102, i32 1, i1 false)
define void @memcpy_multi_use_one_function_keep_small(i8 addrspace(1)* %dst0, i8 addrspace(1)* %dst1, i8 addrspace(1)* %src, i64 %n) #0 {
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst0, i8 addrspace(1)* %src, i64 %n, i32 1, i1 false)
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst1, i8 addrspace(1)* %src, i64 102, i32 1, i1 false)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
