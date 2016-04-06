;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=GCN %s
;RUN: llc < %s -march=amdgcn -mcpu=kabini -verify-machineinstrs | FileCheck --check-prefix=GCN --check-prefix=16BANK %s
;RUN: llc < %s -march=amdgcn -mcpu=stoney -verify-machineinstrs | FileCheck --check-prefix=GCN --check-prefix=16BANK %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s

;GCN-LABEL: {{^}}main:
;GCN-NOT: s_wqm
;GCN: s_mov_b32
;GCN-NEXT: v_interp_mov_f32
;GCN: v_interp_p1_f32
;GCN: v_interp_p2_f32

define amdgpu_ps void @main(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>) {
main_body:
  %5 = call float @llvm.SI.fs.constant(i32 0, i32 0, i32 %3)
  %6 = call float @llvm.SI.fs.interp(i32 0, i32 0, i32 %3, <2 x i32> %4)
  %7 = call float @llvm.SI.fs.interp(i32 1, i32 0, i32 %3, <2 x i32> %4)
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %5, float %6, float %7, float %7)
  ret void
}

; Thest that v_interp_p1 uses different source and destination registers
; on 16 bank LDS chips.

; 16BANK-LABEL: {{^}}v_interp_p1_bank16_bug:
; 16BANK-NOT: v_interp_p1_f32 [[DST:v[0-9]+]], [[DST]]

define amdgpu_ps void @v_interp_p1_bank16_bug([6 x <16 x i8>] addrspace(2)* byval, [17 x <16 x i8>] addrspace(2)* byval, [17 x <4 x i32>] addrspace(2)* byval, [34 x <8 x i32>] addrspace(2)* byval, float inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, i32, float, float) {
main_body:
  %22 = call float @llvm.SI.fs.interp(i32 0, i32 0, i32 %5, <2 x i32> %7)
  %23 = call float @llvm.SI.fs.interp(i32 1, i32 0, i32 %5, <2 x i32> %7)
  %24 = call float @llvm.SI.fs.interp(i32 2, i32 0, i32 %5, <2 x i32> %7)
  %25 = call float @fabs(float %22)
  %26 = call float @fabs(float %23)
  %27 = call float @fabs(float %24)
  %28 = call i32 @llvm.SI.packf16(float %25, float %26)
  %29 = bitcast i32 %28 to float
  %30 = call i32 @llvm.SI.packf16(float %27, float 1.000000e+00)
  %31 = bitcast i32 %30 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %29, float %31, float %29, float %31)
  ret void
}

; Function Attrs: readnone
declare float @fabs(float) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.SI.packf16(float, float) #0

; Function Attrs: nounwind readnone
declare float @llvm.SI.fs.constant(i32, i32, i32) #0

; Function Attrs: nounwind readnone
declare float @llvm.SI.fs.interp(i32, i32, i32, <2 x i32>) #0

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind readnone }
attributes #1 = { readnone }
