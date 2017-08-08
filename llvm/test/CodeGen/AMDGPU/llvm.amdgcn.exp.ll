; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=GCN %s

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #1
declare void @llvm.amdgcn.exp.i32(i32, i32, i32, i32, i32, i32, i1, i1) #1

; GCN-LABEL: {{^}}test_export_zeroes_f32:
; GCN: exp mrt0 off, off, off, off{{$}}
; GCN: exp mrt0 off, off, off, off done{{$}}
define amdgpu_kernel void @test_export_zeroes_f32() #0 {

  call void @llvm.amdgcn.exp.f32(i32 0, i32 0, float 0.0, float 0.0, float 0.0, float 0.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 0, float 0.0, float 0.0, float 0.0, float 0.0, i1 true, i1 false)
  ret void
}

; FIXME: Should not set up registers for the unused source registers.

; GCN-LABEL: {{^}}test_export_en_src0_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], off, off, off done{{$}}
define amdgpu_kernel void @test_export_en_src0_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 1, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src1_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 off, [[SRC1]], off, off done{{$}}
define amdgpu_kernel void @test_export_en_src1_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 2, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src2_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 off, off, [[SRC2]], off done{{$}}
define amdgpu_kernel void @test_export_en_src2_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 4, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src3_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 off, off, off, [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_en_src3_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 8, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src1_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], [[SRC1]], off, off done{{$}}
define amdgpu_kernel void @test_export_en_src0_src1_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 3, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src2_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], off, [[SRC2]], off done{{$}}
define amdgpu_kernel void @test_export_en_src0_src2_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 5, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src3_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], off, off, [[SRC3]]{{$}}
; GCN: exp mrt0 [[SRC0]], off, off, [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_en_src0_src3_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 9, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 9, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src1_src2_src3_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_en_src0_src1_src2_src3_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_mrt7_f32:
; GCN-DAG: v_mov_b32_e32 [[VHALF:v[0-9]+]], 0.5
; GCN: exp mrt7 [[VHALF]], [[VHALF]], [[VHALF]], [[VHALF]]{{$}}
; GCN: exp mrt7 [[VHALF]], [[VHALF]], [[VHALF]], [[VHALF]] done{{$}}
define amdgpu_kernel void @test_export_mrt7_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 7, i32 15, float 0.5, float 0.5, float 0.5, float 0.5, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 7, i32 15, float 0.5, float 0.5, float 0.5, float 0.5, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_z_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrtz [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp mrtz [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_z_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 8, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 8, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_null_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_null_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 9, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 9, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_reserved10_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp invalid_target_10 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp invalid_target_10 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_reserved10_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 10, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 10, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_reserved11_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp invalid_target_11 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp invalid_target_11 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_reserved11_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 11, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 11, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos0_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp pos0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp pos0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_pos0_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos3_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp pos3 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp pos3 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_pos3_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 15, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 15, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_param0_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_param0_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_param31_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_param31_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 63, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 63, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_vm_f32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] vm{{$}}
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done vm{{$}}
define amdgpu_kernel void @test_export_vm_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 true)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 true)
  ret void
}















; GCN-LABEL: {{^}}test_export_zeroes_i32:
; GCN: exp mrt0 off, off, off, off{{$}}
; GCN: exp mrt0 off, off, off, off done{{$}}
define amdgpu_kernel void @test_export_zeroes_i32() #0 {

  call void @llvm.amdgcn.exp.i32(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 true, i1 false)
  ret void
}

; FIXME: Should not set up registers for the unused source registers.

; GCN-LABEL: {{^}}test_export_en_src0_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrt0 [[SRC0]], off, off, off done{{$}}
define amdgpu_kernel void @test_export_en_src0_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 1, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src1_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrt0 off, [[SRC1]], off, off done{{$}}
define amdgpu_kernel void @test_export_en_src1_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 2, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src2_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrt0 off, off, [[SRC2]], off done{{$}}
define amdgpu_kernel void @test_export_en_src2_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 4, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src3_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrt0 off, off, off, [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_en_src3_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 8, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src1_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrt0 [[SRC0]], [[SRC1]], off, off done{{$}}
define amdgpu_kernel void @test_export_en_src0_src1_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 3, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src2_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrt0 [[SRC0]], off, [[SRC2]], off done{{$}}
define amdgpu_kernel void @test_export_en_src0_src2_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 5, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src3_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrt0 [[SRC0]], off, off, [[SRC3]]{{$}}
; GCN: exp mrt0 [[SRC0]], off, off, [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_en_src0_src3_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 9, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 0, i32 9, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src1_src2_src3_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_en_src0_src1_src2_src3_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 0, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_mrt7_i32:
; GCN-DAG: v_mov_b32_e32 [[VHALF:v[0-9]+]], 5
; GCN: exp mrt7 [[VHALF]], [[VHALF]], [[VHALF]], [[VHALF]]{{$}}
; GCN: exp mrt7 [[VHALF]], [[VHALF]], [[VHALF]], [[VHALF]] done{{$}}
define amdgpu_kernel void @test_export_mrt7_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 7, i32 15, i32 5, i32 5, i32 5, i32 5, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 7, i32 15, i32 5, i32 5, i32 5, i32 5, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_z_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrtz [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp mrtz [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_z_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 8, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 8, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_null_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_null_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 9, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 9, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_reserved10_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp invalid_target_10 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp invalid_target_10 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_reserved10_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 10, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 10, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_reserved11_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp invalid_target_11 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp invalid_target_11 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_reserved11_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 11, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 11, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos0_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp pos0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp pos0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_pos0_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 12, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 12, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos3_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp pos3 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp pos3 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_pos3_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 15, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 15, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_param0_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_param0_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 32, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 32, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_param31_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_param31_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 63, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 63, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_vm_i32:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] vm{{$}}
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done vm{{$}}
define amdgpu_kernel void @test_export_vm_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 true)
  call void @llvm.amdgcn.exp.i32(i32 0, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 true)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind inaccessiblememonly }
