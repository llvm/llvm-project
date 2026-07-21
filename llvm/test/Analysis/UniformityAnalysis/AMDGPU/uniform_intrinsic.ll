; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; wave_shuffle(Value, Index): result is divergent only when both Value and
; Index are divergent. A uniform Value read from any lane yields the same
; result, and a uniform Index makes all lanes read the same source lane.

; All kernel args are uniform => result is uniform.
; CHECK-LABEL: UniformityInfo for function 'wave_shuffle_all_uniform':
; CHECK: ALL VALUES UNIFORM
define amdgpu_kernel void @wave_shuffle_all_uniform(ptr addrspace(1) %out, i32 %val, i32 %idx) {
  %v = call i32 @llvm.amdgcn.wave.shuffle(i32 %val, i32 %idx)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; Value is divergent, Index is uniform => result is uniform.
; All lanes read from the same source lane, so the result is the same.
; CHECK-LABEL: UniformityInfo for function 'wave_shuffle_divergent_val_uniform_idx':
; CHECK-NOT: DIVERGENT: {{.*}}wave.shuffle
define amdgpu_kernel void @wave_shuffle_divergent_val_uniform_idx(ptr addrspace(1) %out, i32 %idx) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.wave.shuffle(i32 %tid, i32 %idx)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; Value is uniform, Index is divergent => result is uniform.
; Each lane may read from a different source lane, but Value is the same
; across all lanes so the result is still uniform.
; CHECK-LABEL: UniformityInfo for function 'wave_shuffle_uniform_val_divergent_idx':
; CHECK-NOT: DIVERGENT: {{.*}}wave.shuffle
define amdgpu_kernel void @wave_shuffle_uniform_val_divergent_idx(ptr addrspace(1) %out, i32 %val) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.wave.shuffle(i32 %val, i32 %tid)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; Both Value and Index are divergent => result is divergent.
; CHECK-LABEL: UniformityInfo for function 'wave_shuffle_both_divergent':
; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.wave.shuffle.i32(i32 %tid, i32 %tid)
define amdgpu_kernel void @wave_shuffle_both_divergent(ptr addrspace(1) %out) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.wave.shuffle(i32 %tid, i32 %tid)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.wave.shuffle(i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()
