; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-lower-module-lds -amdgpu-enable-object-linking < %s | FileCheck %s

; An internal (static) device function uses an internal LDS variable.
; The per-function struct must get a module-unique hash suffix to avoid
; cross-TU name collisions (two TUs can both have "static void helper()").

@lds_priv = internal addrspace(3) global [32 x i32] poison, align 16

declare void @extern_func()

; Struct name includes the function name AND a module-unique hash.
; CHECK: @[[STRUCT:__amdgpu_lds\.helper\.[a-f0-9]+]] = external {{(dso_local )?}}addrspace(3) global %[[STRUCT]].t, align 16

; Original variable should be removed.
; CHECK-NOT: @lds_priv

; CHECK-LABEL: define internal void @helper()
; CHECK: getelementptr {{.*}} ptr addrspace(3) @[[STRUCT]]

; CHECK-LABEL: define amdgpu_kernel void @kernel()
; CHECK: call void @helper()

; CHECK: !amdgpu.lds.uses = !{{{![0-9]+}}}
; CHECK-DAG: !{ptr @helper, ptr addrspace(3) @[[STRUCT]]}

; CHECK: !{i32 1, !"amdgpu-link-time-lds", i32 1}

define internal void @helper() {
  %gep = getelementptr [32 x i32], ptr addrspace(3) @lds_priv, i32 0, i32 0
  store i32 1, ptr addrspace(3) %gep
  call void @extern_func()
  ret void
}

define amdgpu_kernel void @kernel() {
  call void @helper()
  ret void
}
