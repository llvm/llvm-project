; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck -check-prefix=SI %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}vector_imin:
; SI: v_min_i32_e32
define void @vector_imin(i32 %p0, i32 %p1, i32 addrspace(1)* %in) #0 {
main_body:
  %load = load i32 addrspace(1)* %in, align 4
  %min = call i32 @llvm.AMDGPU.imin(i32 %p0, i32 %load)
  %bc = bitcast i32 %min to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %bc, float %bc, float %bc, float %bc)
  ret void
}

; SI-LABEL: {{^}}scalar_imin:
; SI: s_min_i32
define void @scalar_imin(i32 %p0, i32 %p1) #0 {
entry:
  %min = call i32 @llvm.AMDGPU.imin(i32 %p0, i32 %p1)
  %bc = bitcast i32 %min to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %bc, float %bc, float %bc, float %bc)
  ret void
}

; Function Attrs: readnone
declare i32 @llvm.AMDGPU.imin(i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!0 = !{!"const", null, i32 1}
