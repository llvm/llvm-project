; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=GCN %s

declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #0
declare void @llvm.amdgcn.exp.compr.v2i16(i32, i32, <2 x i16>, <2 x i16>, i1, i1) #0

; GCN-LABEL: {{^}}test_export_compr_zeroes_v2f16:
; GCN: exp mrt0 off, off, off, off compr{{$}}
; GCN: exp mrt0 off, off, off, off done compr{{$}}
define amdgpu_kernel void @test_export_compr_zeroes_v2f16() #0 {
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 0, <2 x half> zeroinitializer, <2 x half> zeroinitializer, i1 false, i1 false)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 0, <2 x half> zeroinitializer, <2 x half> zeroinitializer, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_en_src0_v2f16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x40003c00
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x44003800
; GCN: exp mrt0 [[SRC0]], [[SRC0]], off, off done compr{{$}}
define amdgpu_kernel void @test_export_compr_en_src0_v2f16() #0 {
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 3, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_en_src1_v2f16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x40003c00
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x44003800
; GCN: exp mrt0 off, off, [[SRC1]], [[SRC1]] done compr{{$}}
define amdgpu_kernel void @test_export_compr_en_src1_v2f16() #0 {
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 12, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_en_src0_src1_v2f16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x40003c00
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x44003800
; GCN: exp mrt0 [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] done compr{{$}}
define amdgpu_kernel void @test_export_compr_en_src0_src1_v2f16() #0 {
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_en_invalid2_v2f16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x40003c00
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x44003800
; GCN: exp mrt0 off, [[SRC0]], off, off done compr{{$}}
define amdgpu_kernel void @test_export_compr_en_invalid2_v2f16() #0 {
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 2, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_en_invalid10_v2f16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x40003c00
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x44003800
; GCN: exp mrt0 off, [[SRC0]], off, [[SRC1]] done compr{{$}}
define amdgpu_kernel void @test_export_compr_en_invalid10_v2f16() #0 {
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 10, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_mrt7_v2f16:
; GCN-DAG: v_mov_b32_e32 [[VHALF:v[0-9]+]], 0x38003800
; GCN: exp mrt7 [[VHALF]], [[VHALF]], [[VHALF]], [[VHALF]] compr{{$}}
; GCN: exp mrt7 [[VHALF]], [[VHALF]], [[VHALF]], [[VHALF]] done compr{{$}}
define amdgpu_kernel void @test_export_compr_mrt7_v2f16() #0 {
  call void @llvm.amdgcn.exp.compr.v2f16(i32 7, i32 15, <2 x half> <half 0.5, half 0.5>, <2 x half> <half 0.5, half 0.5>, i1 false, i1 false)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 7, i32 15, <2 x half> <half 0.5, half 0.5>, <2 x half> <half 0.5, half 0.5>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_z_v2f16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x40003c00
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x44003800
; GCN: exp mrtz [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] compr{{$}}
; GCN: exp mrtz [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] done compr{{$}}
define amdgpu_kernel void @test_export_compr_z_v2f16() #0 {
  call void @llvm.amdgcn.exp.compr.v2f16(i32 8, i32 15, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 false, i1 false)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 8, i32 15, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_vm_v2f16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x40003c00
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x44003800
; GCN: exp mrt0 [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] compr vm{{$}}
; GCN: exp mrt0 [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] done compr vm{{$}}
define amdgpu_kernel void @test_export_compr_vm_v2f16() #0 {
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 false, i1 true)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> <half 1.0, half 2.0>, <2 x half> <half 0.5, half 4.0>, i1 true, i1 true)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_zeroes_v2i16:
; GCN: exp mrt0 off, off, off, off compr{{$}}
; GCN: exp mrt0 off, off, off, off done compr{{$}}
define amdgpu_kernel void @test_export_compr_zeroes_v2i16() #0 {
  call void @llvm.amdgcn.exp.compr.v2i16(i32 0, i32 0, <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, i1 false, i1 false)
  call void @llvm.amdgcn.exp.compr.v2i16(i32 0, i32 0, <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_en_src0_v2i16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x20001
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x40005
; GCN: exp mrt0 [[SRC0]], off, off, off done compr{{$}}
define amdgpu_kernel void @test_export_compr_en_src0_v2i16() #0 {
  call void @llvm.amdgcn.exp.compr.v2i16(i32 0, i32 1, <2 x i16> <i16 1, i16 2>, <2 x i16> <i16 5, i16 4>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_en_src1_v2i16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x20001
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x40005
; GCN: exp mrt0 off, off, [[SRC1]], [[SRC1]] done compr{{$}}
define amdgpu_kernel void @test_export_compr_en_src1_v2i16() #0 {
  call void @llvm.amdgcn.exp.compr.v2i16(i32 0, i32 12, <2 x i16> <i16 1, i16 2>, <2 x i16> <i16 5, i16 4>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_en_src0_src1_v2i16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x20001
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x40005
; GCN: exp mrt0 [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] done compr{{$}}
define amdgpu_kernel void @test_export_compr_en_src0_src1_v2i16() #0 {
  call void @llvm.amdgcn.exp.compr.v2i16(i32 0, i32 15, <2 x i16> <i16 1, i16 2>, <2 x i16> <i16 5, i16 4>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_mrt7_v2i16:
; GCN-DAG: v_mov_b32_e32 [[VI16:v[0-9]+]], 0x50005
; GCN: exp mrt7 [[VI16]], [[VI16]], [[VI16]], [[VI16]] compr{{$}}
; GCN: exp mrt7 [[VI16]], [[VI16]], [[VI16]], [[VI16]] done compr{{$}}
define amdgpu_kernel void @test_export_compr_mrt7_v2i16() #0 {
  call void @llvm.amdgcn.exp.compr.v2i16(i32 7, i32 15, <2 x i16> <i16 5, i16 5>, <2 x i16> <i16 5, i16 5>, i1 false, i1 false)
  call void @llvm.amdgcn.exp.compr.v2i16(i32 7, i32 15, <2 x i16> <i16 5, i16 5>, <2 x i16> <i16 5, i16 5>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_z_v2i16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x20001
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x40005
; GCN: exp mrtz [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] compr{{$}}
; GCN: exp mrtz [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] done compr{{$}}
define amdgpu_kernel void @test_export_compr_z_v2i16() #0 {
  call void @llvm.amdgcn.exp.compr.v2i16(i32 8, i32 15, <2 x i16> <i16 1, i16 2>, <2 x i16> <i16 5, i16 4>, i1 false, i1 false)
  call void @llvm.amdgcn.exp.compr.v2i16(i32 8, i32 15, <2 x i16> <i16 1, i16 2>, <2 x i16> <i16 5, i16 4>, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_compr_vm_v2i16:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 0x20001
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 0x40005
; GCN: exp mrt0 [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] compr vm{{$}}
; GCN: exp mrt0 [[SRC0]], [[SRC0]], [[SRC1]], [[SRC1]] done compr vm{{$}}
define amdgpu_kernel void @test_export_compr_vm_v2i16() #0 {
  call void @llvm.amdgcn.exp.compr.v2i16(i32 0, i32 15, <2 x i16> <i16 1, i16 2>, <2 x i16> <i16 5, i16 4>, i1 false, i1 true)
  call void @llvm.amdgcn.exp.compr.v2i16(i32 0, i32 15, <2 x i16> <i16 1, i16 2>, <2 x i16> <i16 5, i16 4>, i1 true, i1 true)
  ret void
}

attributes #0 = { nounwind }
