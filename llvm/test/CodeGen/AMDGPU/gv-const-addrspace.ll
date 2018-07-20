; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


@b = internal addrspace(4) constant [1 x i16] [ i16 7 ], align 2

@float_gv = internal unnamed_addr addrspace(4) constant [5 x float] [float 0.0, float 1.0, float 2.0, float 3.0, float 4.0], align 4

; FUNC-LABEL: {{^}}float:
; GCN: s_load_dword

; EG: VTX_READ_32
; EG: @float_gv
; EG-NOT: MOVA_INT
; EG-NOT: MOV
define amdgpu_kernel void @float(float addrspace(1)* %out, i32 %index) {
entry:
  %0 = getelementptr inbounds [5 x float], [5 x float] addrspace(4)* @float_gv, i32 0, i32 %index
  %1 = load float, float addrspace(4)* %0
  store float %1, float addrspace(1)* %out
  ret void
}

@i32_gv = internal unnamed_addr addrspace(4) constant [5 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4], align 4

; FUNC-LABEL: {{^}}i32:

; GCN: s_load_dword

; EG: VTX_READ_32
; EG: @i32_gv
; EG-NOT: MOVA_INT
; EG-NOT: MOV
define amdgpu_kernel void @i32(i32 addrspace(1)* %out, i32 %index) {
entry:
  %0 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(4)* @i32_gv, i32 0, i32 %index
  %1 = load i32, i32 addrspace(4)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}


%struct.foo = type { float, [5 x i32] }

@struct_foo_gv = internal unnamed_addr addrspace(4) constant [1 x %struct.foo] [ %struct.foo { float 16.0, [5 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4] } ]

; FUNC-LABEL: {{^}}struct_foo_gv_load:
; GCN: s_load_dword

; EG: VTX_READ_32
; EG: @struct_foo_gv
; EG-NOT: MOVA_INT
; EG-NOT: MOV
define amdgpu_kernel void @struct_foo_gv_load(i32 addrspace(1)* %out, i32 %index) {
  %gep = getelementptr inbounds [1 x %struct.foo], [1 x %struct.foo] addrspace(4)* @struct_foo_gv, i32 0, i32 0, i32 1, i32 %index
  %load = load i32, i32 addrspace(4)* %gep, align 4
  store i32 %load, i32 addrspace(1)* %out, align 4
  ret void
}

@array_v1_gv = internal addrspace(4) constant [4 x <1 x i32>] [ <1 x i32> <i32 1>,
                                                                <1 x i32> <i32 2>,
                                                                <1 x i32> <i32 3>,
                                                                <1 x i32> <i32 4> ]

; FUNC-LABEL: {{^}}array_v1_gv_load:
; GCN: s_load_dword

; EG: VTX_READ_32
; EG: @array_v1_gv
; EG-NOT: MOVA_INT
; EG-NOT: MOV
define amdgpu_kernel void @array_v1_gv_load(<1 x i32> addrspace(1)* %out, i32 %index) {
  %gep = getelementptr inbounds [4 x <1 x i32>], [4 x <1 x i32>] addrspace(4)* @array_v1_gv, i32 0, i32 %index
  %load = load <1 x i32>, <1 x i32> addrspace(4)* %gep, align 4
  store <1 x i32> %load, <1 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}gv_addressing_in_branch:

; EG: VTX_READ_32
; EG: @float_gv
; EG-NOT: MOVA_INT
define amdgpu_kernel void @gv_addressing_in_branch(float addrspace(1)* %out, i32 %index, i32 %a) {
entry:
  %0 = icmp eq i32 0, %a
  br i1 %0, label %if, label %else

if:
  %1 = getelementptr inbounds [5 x float], [5 x float] addrspace(4)* @float_gv, i32 0, i32 %index
  %2 = load float, float addrspace(4)* %1
  store float %2, float addrspace(1)* %out
  br label %endif

else:
  store float 1.0, float addrspace(1)* %out
  br label %endif

endif:
  ret void
}
