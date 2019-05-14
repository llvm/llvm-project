; RUN: opt -S -mtriple=amdgcn-- -load-store-vectorizer < %s | FileCheck -check-prefix=OPT %s

; OPT-LABEL: @func(
define void @func(i32 addrspace(7)* %out) {
entry:
  %a0 = getelementptr i32, i32 addrspace(7)* %out, i32 0
  %a1 = getelementptr i32, i32 addrspace(7)* %out, i32 1
  %a2 = getelementptr i32, i32 addrspace(7)* %out, i32 2
  %a3 = getelementptr i32, i32 addrspace(7)* %out, i32 3

; OPT: store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <4 x i32> addrspace(7)* %0, align 4
  store i32 0, i32 addrspace(7)* %a0
  store i32 1, i32 addrspace(7)* %a1
  store i32 2, i32 addrspace(7)* %a2
  store i32 3, i32 addrspace(7)* %a3
  ret void
}
