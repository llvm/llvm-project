; RUN: not llc -mtriple=thumbv6m-none-eabi -global-isel=0 < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=thumbv8m.base-none-eabi -global-isel=0 < %s -o /dev/null 2>&1 | FileCheck %s

; Thumb-1 has no load to pc, and these targets have no ARM state to fall back
; to, so there's no trampoline sequence for them.

; CHECK: error: {{.*}}in function init {{.*}}: trampolines are not supported on Thumb-1-only targets

declare void @llvm.init.trampoline(ptr, ptr, ptr)
declare ptr @llvm.adjust.trampoline(ptr)

declare i32 @nested(ptr nest, i32)

define ptr @init(ptr %tramp, ptr %chain) {
  call void @llvm.init.trampoline(ptr %tramp, ptr @nested, ptr %chain)
  %fp = call ptr @llvm.adjust.trampoline(ptr %tramp)
  ret ptr %fp
}
