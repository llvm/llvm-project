;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK: s_mov_b32
;CHECK-NEXT: v_interp_mov_f32

define void @main(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg) "ShaderType"="0" {
main_body:
  %4 = call float @llvm.SI.fs.constant(i32 0, i32 0, i32 %3)
  %5 = call i32 @llvm.SI.packf16(float %4, float %4)
  %6 = bitcast i32 %5 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %6, float %6, float %6, float %6)
  ret void
}

declare void @llvm.AMDGPU.shader.type(i32)

declare float @llvm.SI.fs.constant(i32, i32, i32) readnone

declare i32 @llvm.SI.packf16(float, float) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
