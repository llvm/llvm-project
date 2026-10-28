; RUN: split-file %s %t
; RUN: opt -S -passes=nvptx-canonicalize-dynamic-shared %t/used.ll | FileCheck %t/used.ll
; RUN: opt -S -passes=nvptx-canonicalize-dynamic-shared %t/inline-asm.ll | FileCheck %t/inline-asm.ll
; RUN: opt -S -passes=nvptx-canonicalize-dynamic-shared %t/module-asm.ll | FileCheck %t/module-asm.ll
; RUN: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 %t/inline-asm.ll -o - | FileCheck %t/inline-asm.ll --check-prefix=PTX
; RUN: %if ptxas %{ llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 %t/inline-asm.ll -o - | %ptxas-verify -arch=sm_90 %}

; A declaration that a used list or assembly may still name keeps its symbol,
; but its uses still move to the common declaration.

;--- used.ll
target triple = "nvptx64-nvidia-cuda"

@a = external addrspace(3) global [0 x i32], align 4
@b = external addrspace(3) global [0 x i32], align 4
@c = external addrspace(3) global [0 x i32], align 4
@d = external addrspace(3) global [0 x i32], align 4
@llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @b to ptr)], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @c to ptr)], section "llvm.metadata"

; CHECK: @a = external addrspace(3) global [0 x i32], align 4
; CHECK-NEXT: @b = external addrspace(3) global [0 x i32], align 4
; CHECK-NEXT: @c = external addrspace(3) global [0 x i32], align 4
; CHECK-NEXT: @llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @b to ptr)], section "llvm.metadata"
; CHECK-NEXT: @llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @c to ptr)], section "llvm.metadata"
; CHECK-NOT: @d =
; CHECK-LABEL: define void @uses(
; CHECK-COUNT-4: call void @consume(ptr addrspace(3) @a)
define void @uses() {
  call void @consume(ptr addrspace(3) @a)
  call void @consume(ptr addrspace(3) @b)
  call void @consume(ptr addrspace(3) @c)
  call void @consume(ptr addrspace(3) @d)
  ret void
}

declare void @consume(ptr addrspace(3))

;--- inline-asm.ll
target triple = "nvptx64-nvidia-cuda"

; Inline asm names @b, so @b keeps its symbol and the PTX still declares it.
; Only whole identifiers count: "c_suffix" does not keep @c.
; CHECK: @a = external addrspace(3) global [0 x i32], align 4
; CHECK-NEXT: @b = external addrspace(3) global [0 x i32], align 4
; CHECK-NOT: @c =
; CHECK-LABEL: define ptx_kernel void @kernel(
; CHECK-COUNT-3: ptr addrspace(3) @a

; PTX: .extern .shared .align 4 .b8 a[];
; PTX: .extern .shared .align 4 .b8 b[];
; PTX-NOT: .extern .shared
; PTX: ld.shared.u32 %r{{[0-9]+}}, [b+4]; // c_suffix

@a = external addrspace(3) global [0 x i32], align 4
@b = external addrspace(3) global [0 x i32], align 4
@c = external addrspace(3) global [0 x i32], align 4

define ptx_kernel void @kernel(ptr addrspace(1) %out, i32 %v) {
  store volatile i32 %v, ptr addrspace(3) @a
  %x = load volatile i32, ptr addrspace(3) @b
  %y = load volatile i32, ptr addrspace(3) @c
  %z = call i32 asm sideeffect "ld.shared.u32 $0, [b+4]; // c_suffix", "=r"()
  %s = add i32 %x, %y
  %t = add i32 %s, %z
  store i32 %t, ptr addrspace(1) %out
  ret void
}

;--- module-asm.ll
target triple = "nvptx64-nvidia-cuda"

module asm "// may name any symbol"

; CHECK: @a = external addrspace(3) global [0 x i32], align 4
; CHECK-NEXT: @b = external addrspace(3) global [0 x i32], align 4
; CHECK-LABEL: define void @uses(
; CHECK-COUNT-2: call void @consume(ptr addrspace(3) @a)
@a = external addrspace(3) global [0 x i32], align 4
@b = external addrspace(3) global [0 x i32], align 4

define void @uses() {
  call void @consume(ptr addrspace(3) @a)
  call void @consume(ptr addrspace(3) @b)
  ret void
}

declare void @consume(ptr addrspace(3))
