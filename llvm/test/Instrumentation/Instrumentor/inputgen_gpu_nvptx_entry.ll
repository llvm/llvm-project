; Verify entry generation uses the NVPTX kernel calling convention.
; RUN: opt < %s -passes=inputgen-gpu -inputgen-gpu-entry-function=vvv_foo -S | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

define hidden i32 @vvv_foo(ptr noundef %a) {
entry:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

; CHECK-LABEL: define ptx_kernel void @__ig_entry_vvv_foo(
