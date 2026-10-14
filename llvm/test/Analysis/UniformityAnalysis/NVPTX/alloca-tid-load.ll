; RUN: opt %s -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s
;
; NVPTX: alloca results are divergence sources (private/local stack). A divergent
; value stored through an alloca pointer must make the subsequent load divergent.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel i32 @alloca_tid_load() {
; CHECK-LABEL: for function 'alloca_tid_load'
entry:
  %tid_value = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %a = alloca i32, align 4
  store i32 %tid_value, ptr %a, align 4
  %v = load i32, ptr %a, align 4
; CHECK: DIVERGENT:   %tid_value = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK: DIVERGENT:   %a = alloca i32, align 4
; CHECK: DIVERGENT:   store i32 %tid_value, ptr %a, align 4
; CHECK: DIVERGENT:   %v = load i32, ptr %a, align 4
  ret i32 %v
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
