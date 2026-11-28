; RUN: split-file %s %t
; RUN: opt -module-summary -passes='thinlto-pre-link<O2>' %t/a.ll -o %t/a.bc
; RUN: opt -module-summary -passes='thinlto-pre-link<O2>' %t/b.ll -o %t/b.bc
; RUN: llvm-lto2 run %t/a.bc %t/b.bc -o %t/out -save-temps -O2 -mcpu=sm_90 -filetype=asm \
; RUN:   -r=%t/a.bc,kernel,px -r=%t/a.bc,write_dynamic, -r=%t/a.bc,a, \
; RUN:   -r=%t/b.bc,write_dynamic,p -r=%t/b.bc,b,
; RUN: llvm-dis %t/out.1.4.opt.bc -o - | FileCheck %s

; Importing @write_dynamic brings in a second declaration of the dynamic shared
; memory. It is merged before inlining and GVN, so the store it makes is not
; assumed to leave @a unchanged.

; CHECK: @a = external {{.*}}addrspace(3) global [0 x i32], align 16
; CHECK-NOT: @b =
; CHECK-LABEL: define ptx_kernel void @kernel(
; CHECK: %old = load i32, ptr addrspace(3) @a
; CHECK: store i32 %v, ptr addrspace(3) @a
; CHECK-NOT: load
; CHECK: %r = sub i32 %v, %old

;--- a.ll
target triple = "nvptx64-nvidia-cuda"

@a = external addrspace(3) global [0 x i32], align 4

define ptx_kernel void @kernel(ptr addrspace(1) %out, i32 %v) {
  %old = load i32, ptr addrspace(3) @a
  call void @write_dynamic(i32 %v)
  %new = load i32, ptr addrspace(3) @a
  %r = sub i32 %new, %old
  store i32 %r, ptr addrspace(1) %out
  ret void
}

declare void @write_dynamic(i32)

;--- b.ll
target triple = "nvptx64-nvidia-cuda"

@b = external addrspace(3) global [0 x i32], align 16

define void @write_dynamic(i32 %v) alwaysinline {
  store i32 %v, ptr addrspace(3) @b
  ret void
}
