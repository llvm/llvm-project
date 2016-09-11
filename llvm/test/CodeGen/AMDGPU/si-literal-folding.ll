; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}main:
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x3f4353f8, v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0xbf4353f8, v{{[0-9]+}}
define amdgpu_vs void @main(float) {
main_body:
  %1 = fmul float %0, 0x3FE86A7F00000000
  %2 = fmul float %0, 0xBFE86A7F00000000
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %1, float %1, float %2, float %2)
  ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
