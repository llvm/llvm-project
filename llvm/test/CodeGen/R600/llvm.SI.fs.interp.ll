;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-NOT: s_wqm
;CHECK: s_mov_b32
;CHECK: v_interp_p1_f32
;CHECK: v_interp_p2_f32
;CHECK: v_interp_mov_f32

define void @main(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>) #0 {
main_body:
  %5 = call float @llvm.SI.fs.constant(i32 0, i32 0, i32 %3)
  %6 = call float @llvm.SI.fs.interp(i32 0, i32 0, i32 %3, <2 x i32> %4)
  %7 = call float @llvm.SI.fs.interp(i32 1, i32 0, i32 %3, <2 x i32> %4)
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %5, float %6, float %7, float %7)
  ret void
}

declare void @llvm.AMDGPU.shader.type(i32)

; Function Attrs: nounwind readnone
declare float @llvm.SI.fs.constant(i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.SI.fs.interp(i32, i32, i32, <2 x i32>) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }
