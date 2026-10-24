; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-before=always-inline %s -o - | FileCheck %s --check-prefix=PRUNED
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-prune-before-inline=false -stop-before=always-inline %s -o - | FileCheck %s --check-prefix=BASE
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
;
; Removing the unused outer wrapper leaves a whole device-call chain with no
; externally reachable root. Prune it before always-inline expands its bodies.
; Keep the live call and externally visible global data.
;
; PRUNED-NOT: define {{.*}}@dead_leaf
; PRUNED-NOT: define {{.*}}@dead_middle
; PRUNED: @keep = global i32 42
; PRUNED: define internal i32 @live_helper
; PRUNED: define agx_kernel void @kernel
; BASE: define internal i32 @dead_leaf
; BASE: define internal i32 @dead_middle
; ASM-LABEL: kernel:
; ASM: iadd {{.*}}, 7

@keep = global i32 42

define i32 @dead_leaf(i32 %x) {
  %value = mul i32 %x, 17
  ret i32 %value
}

define i32 @dead_middle(i32 %x) {
  %value = call i32 @dead_leaf(i32 %x)
  ret i32 %value
}

define i32 @dead_outer(i32 %x) {
  %value = call i32 @dead_middle(i32 %x)
  ret i32 %value
}

define i32 @live_helper(i32 %x) {
  %value = add i32 %x, 7
  ret i32 %value
}

define agx_kernel void @kernel(ptr addrspace(1) %out, i32 %x) {
  %value = call i32 @live_helper(i32 %x)
  store i32 %value, ptr addrspace(1) %out
  ret void
}
