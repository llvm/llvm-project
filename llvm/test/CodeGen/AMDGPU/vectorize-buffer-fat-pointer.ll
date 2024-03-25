; RUN: opt -S -mtriple=amdgcn-- -passes=load-store-vectorizer < %s | FileCheck -check-prefix=OPT %s

; OPT-LABEL: @buffer_fat_ptrs(
define void @buffer_fat_ptrs(ptr addrspace(7) %out) {
entry:
  %a1 = getelementptr i32, ptr addrspace(7) %out, i32 1
  %a2 = getelementptr i32, ptr addrspace(7) %out, i32 2
  %a3 = getelementptr i32, ptr addrspace(7) %out, i32 3

; OPT: store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr addrspace(7) %out, align 4
  store i32 0, ptr addrspace(7) %out
  store i32 1, ptr addrspace(7) %a1
  store i32 2, ptr addrspace(7) %a2
  store i32 3, ptr addrspace(7) %a3
  ret void
}

; OPT-LABEL: @buffer_strided_ptrs(
define void @buffer_strided_ptrs(ptr addrspace(9) %out) {
entry:
  %a1 = getelementptr i32, ptr addrspace(9) %out, i32 1
  %a2 = getelementptr i32, ptr addrspace(9) %out, i32 2
  %a3 = getelementptr i32, ptr addrspace(9) %out, i32 3

; OPT: store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr addrspace(9) %out, align 4
  store i32 0, ptr addrspace(9) %out
  store i32 1, ptr addrspace(9) %a1
  store i32 2, ptr addrspace(9) %a2
  store i32 3, ptr addrspace(9) %a3
  ret void
}
