; RUN: split-file %s %t
; RUN: opt -S -passes=nvptx-canonicalize-dynamic-shared %t/mixed.ll | FileCheck %t/mixed.ll
; RUN: opt -S -passes=nvptx-canonicalize-dynamic-shared %t/all-constant.ll | FileCheck %t/all-constant.ll

; A constant declaration still names the dynamic shared memory, so it is merged
; like any other. The common declaration stays constant only if every merged
; declaration is.

;--- mixed.ll
target triple = "nvptx64-nvidia-cuda"

; CHECK: @a = external addrspace(3) global [0 x i32], align 4
@a = external addrspace(3) constant [0 x i32], align 4
@b = external addrspace(3) global [0 x i32], align 4

; CHECK-LABEL: define i32 @uses(
; CHECK-NEXT: store i32 %v, ptr addrspace(3) @a
; CHECK-NEXT: load i32, ptr addrspace(3) @a
define i32 @uses(i32 %v) {
  store i32 %v, ptr addrspace(3) @b
  %r = load i32, ptr addrspace(3) @a
  ret i32 %r
}

;--- all-constant.ll
target triple = "nvptx64-nvidia-cuda"

; CHECK: @a = external addrspace(3) constant [0 x i32], align 4
@a = external addrspace(3) constant [0 x i32], align 4
@b = external addrspace(3) constant [0 x i32], align 4

; CHECK-LABEL: define i32 @uses(
; CHECK-NEXT: load i32, ptr addrspace(3) @a
; CHECK-NEXT: load i32, ptr addrspace(3) @a
define i32 @uses() {
  %x = load i32, ptr addrspace(3) @a
  %y = load i32, ptr addrspace(3) @b
  %r = add i32 %x, %y
  ret i32 %r
}
