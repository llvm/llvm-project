; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck -check-prefix=SI %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}vector_umin:
; SI: v_min_u32_e32
define void @vector_umin(i32 %p0, i32 %p1, i32 addrspace(1)* %in) #0 {
main_body:
  %load = load i32 addrspace(1)* %in, align 4
  %min = call i32 @llvm.AMDGPU.umin(i32 %p0, i32 %load)
  %bc = bitcast i32 %min to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %bc, float %bc, float %bc, float %bc)
  ret void
}

; SI-LABEL: {{^}}scalar_umin:
; SI: s_min_u32
define void @scalar_umin(i32 %p0, i32 %p1) #0 {
entry:
  %min = call i32 @llvm.AMDGPU.umin(i32 %p0, i32 %p1)
  %bc = bitcast i32 %min to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %bc, float %bc, float %bc, float %bc)
  ret void
}

; SI-LABEL: {{^}}trunc_zext_umin:
; SI: buffer_load_ubyte [[VREG:v[0-9]+]],
; SI: v_min_u32_e32 [[RESULT:v[0-9]+]], 0, [[VREG]]
; SI-NOT: and
; SI: buffer_store_short [[RESULT]],
define void @trunc_zext_umin(i16 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %src) nounwind {
  %tmp5 = load i8 addrspace(1)* %src, align 1
  %tmp2 = zext i8 %tmp5 to i32
  %tmp3 = tail call i32 @llvm.AMDGPU.umin(i32 %tmp2, i32 0) nounwind readnone
  %tmp4 = trunc i32 %tmp3 to i8
  %tmp6 = zext i8 %tmp4 to i16
  store i16 %tmp6, i16 addrspace(1)* %out, align 2
  ret void
}

; Function Attrs: readnone
declare i32 @llvm.AMDGPU.umin(i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!0 = !{!"const", null, i32 1}
