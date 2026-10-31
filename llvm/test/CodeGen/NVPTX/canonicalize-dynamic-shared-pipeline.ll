; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes='default<O0>' %s | FileCheck %s
; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes='default<O2>' %s | FileCheck %s
; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes='thinlto-pre-link<O2>' %s | FileCheck %s
; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes='thinlto<O0>' %s | FileCheck %s
; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes='thinlto<O2>' %s | FileCheck %s
; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes='lto<O0>' %s | FileCheck %s
; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes='lto<O2>' %s | FileCheck %s
; RUN: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 < %s | FileCheck %s --check-prefix=PTX
; RUN: llc -enable-new-pm -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 < %s | FileCheck %s --check-prefix=PTX
; RUN: %if ptxas %{ llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 < %s | %ptxas-verify -arch=sm_90 %}

; The optimization pipelines and both codegen pipelines merge the declarations
; before anything relies on them being distinct.

; CHECK: @a = external addrspace(3) global [0 x i32], align 16
; CHECK-NOT: @b =
; CHECK-LABEL: define {{.*}}void @kernel(
; CHECK: store volatile i32 %v, ptr addrspace(3) @a
; CHECK: load volatile i32, ptr addrspace(3) @a

; PTX: .extern .shared .align 16 .b8 a[];
; PTX-NOT: .extern .shared
; PTX-LABEL: .visible .entry kernel(
; PTX: st.volatile.shared.b32 [a],
; PTX: ld.volatile.shared.b32 %r{{[0-9]+}}, [a];

@a = external addrspace(3) global [0 x i32], align 4
@b = external addrspace(3) global [0 x i32], align 16

define ptx_kernel void @kernel(ptr addrspace(1) %out, i32 %v) {
  store volatile i32 %v, ptr addrspace(3) @a
  %x = load volatile i32, ptr addrspace(3) @b
  store i32 %x, ptr addrspace(1) %out
  ret void
}
