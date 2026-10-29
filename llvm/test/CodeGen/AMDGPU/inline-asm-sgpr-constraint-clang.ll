; RUN: clang -x ir -S -O3 --target=amdgcn-amd-amdhsa -mcpu=gfx1100 %s -o - | FileCheck %s

; CHECK-LABEL: k:
; CHECK:       v_cvt_u32_f64_e32
; CHECK:       ;;#ASMSTART
; CHECK-NEXT:  s_mov_b32 s0, v{{[0-9]+}}
; CHECK-NEXT:  ;;#ASMEND

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgpu11.00-amd-amdhsa"

define amdgpu_kernel void @k(ptr addrspace(1) %p, double %d) #0 {
  %v = fptoui double %d to i32
  call void asm sideeffect "s_mov_b32 s0, $0", "s"(i32 %v)
  ret void
}