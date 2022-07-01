; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefixes=GCN,PREGFX11 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefixes=GCN,GFX10,PREGFX11 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefixes=GCN,GFX11 %s

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #1
declare void @llvm.amdgcn.exp.i32(i32, i32, i32, i32, i32, i32, i1, i1) #1
declare float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32>, i32, i32, i32) #2

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
; PREGFX11-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; PREGFX11-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; PREGFX11-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; PREGFX11-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; PREGFX11: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; PREGFX11: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
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
; PREGFX11-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; PREGFX11-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; PREGFX11-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; PREGFX11-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; PREGFX11: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; PREGFX11: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_param0_f32() #0 {
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float 1.0, float 2.0, float 0.5, float 4.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_param31_f32:
; PREGFX11-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; PREGFX11-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; PREGFX11-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; PREGFX11-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; PREGFX11: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; PREGFX11: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
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
; PREGFX11: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] vm{{$}}
; PREGFX11: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done vm{{$}}
; GFX11: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GFX11: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
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
; PREGFX11-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; PREGFX11-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; PREGFX11-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; PREGFX11-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; PREGFX11: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; PREGFX11: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
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
; PREGFX11-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; PREGFX11-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; PREGFX11-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; PREGFX11-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; PREGFX11: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; PREGFX11: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_param0_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 32, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 false)
  call void @llvm.amdgcn.exp.i32(i32 32, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_param31_i32:
; PREGFX11-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1
; PREGFX11-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2
; PREGFX11-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 5
; PREGFX11-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4
; PREGFX11: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; PREGFX11: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
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
; PREGFX11: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] vm{{$}}
; PREGFX11: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done vm{{$}}
; GFX11: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GFX11: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_vm_i32() #0 {
  call void @llvm.amdgcn.exp.i32(i32 0, i32 15, i32 1, i32 2, i32 5, i32 4, i1 false, i1 true)
  call void @llvm.amdgcn.exp.i32(i32 0, i32 15, i32 1, i32 2, i32 5, i32 4, i1 true, i1 true)
  ret void
}

; GCN-LABEL: {{^}}test_if_export_f32:
; GCN: s_cbranch_execz
; GCN: exp
define amdgpu_ps void @test_if_export_f32(i32 %flag, float %x, float %y, float %z, float %w) #0 {
  %cc = icmp eq i32 %flag, 0
  br i1 %cc, label %end, label %exp

exp:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 false, i1 false)
  br label %end

end:
  ret void
}

; GCN-LABEL: {{^}}test_if_export_vm_f32:
; GCN: s_cbranch_execz
; GCN: exp
define amdgpu_ps void @test_if_export_vm_f32(i32 %flag, float %x, float %y, float %z, float %w) #0 {
  %cc = icmp eq i32 %flag, 0
  br i1 %cc, label %end, label %exp

exp:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 false, i1 true)
  br label %end

end:
  ret void
}

; GCN-LABEL: {{^}}test_if_export_done_f32:
; GCN: s_cbranch_execz
; GCN: exp
define amdgpu_ps void @test_if_export_done_f32(i32 %flag, float %x, float %y, float %z, float %w) #0 {
  %cc = icmp eq i32 %flag, 0
  br i1 %cc, label %end, label %exp

exp:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 false)
  br label %end

end:
  ret void
}

; GCN-LABEL: {{^}}test_if_export_vm_done_f32:
; GCN: s_cbranch_execz
; GCN: exp
define amdgpu_ps void @test_if_export_vm_done_f32(i32 %flag, float %x, float %y, float %z, float %w) #0 {
  %cc = icmp eq i32 %flag, 0
  br i1 %cc, label %end, label %exp

exp:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 true)
  br label %end

end:
  ret void
}

; GCN-LABEL: {{^}}test_export_clustering:
; PREGFX11-DAG: v_mov_b32_e32 [[W0:v[0-9]+]], 0
; PREGFX11-DAG: v_mov_b32_e32 [[W1:v[0-9]+]], 1.0
; PREGFX11-DAG: v_mov_b32_e32 [[X:v[0-9]+]], s0
; PREGFX11-DAG: v_mov_b32_e32 [[Y:v[0-9]+]], s1
; PREGFX11-DAG: v_add_f32_e{{32|64}} [[Z0:v[0-9]+]]
; PREGFX11-DAG: v_sub_f32_e{{32|64}} [[Z1:v[0-9]+]]
; PREGFX11: exp param0 [[X]], [[Y]], [[Z0]], [[W0]]{{$}}
; PREGFX11-NEXT: exp param1 [[X]], [[Y]], [[Z1]], [[W1]] done{{$}}
define amdgpu_kernel void @test_export_clustering(float %x, float %y) #0 {
  %z0 = fadd float %x, %y
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %x, float %y, float %z0, float 0.0, i1 false, i1 false)
  %z1 = fsub float %y, %x
  call void @llvm.amdgcn.exp.f32(i32 33, i32 15, float %x, float %y, float %z1, float 1.0, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos_before_param:
; PREGFX11: exp pos0
; PREGFX11-NOT: s_waitcnt
; PREGFX11: exp param0
define amdgpu_kernel void @test_export_pos_before_param(float %x, float %y) #0 {
  %z0 = fadd float %x, %y
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float 1.0, float 1.0, float 1.0, float %z0, i1 false, i1 false)
  %z1 = fsub float %y, %x
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float 0.0, float 0.0, float 0.0, float %z1, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos4_before_param:
; GFX10: exp pos4
; GFX10-NOT: s_waitcnt
; GFX10: exp param0
define amdgpu_kernel void @test_export_pos4_before_param(float %x, float %y) #0 {
  %z0 = fadd float %x, %y
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float 1.0, float 1.0, float 1.0, float %z0, i1 false, i1 false)
  %z1 = fsub float %y, %x
  call void @llvm.amdgcn.exp.f32(i32 16, i32 15, float 0.0, float 0.0, float 0.0, float %z1, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos_before_param_ordered:
; PREGFX11: exp pos0
; PREGFX11: exp pos1
; PREGFX11: exp pos2
; PREGFX11-NOT: s_waitcnt
; PREGFX11: exp param0
; PREGFX11: exp param1
; PREGFX11: exp param2
define amdgpu_kernel void @test_export_pos_before_param_ordered(float %x, float %y) #0 {
  %z0 = fadd float %x, %y
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float 1.0, float 1.0, float 1.0, float %z0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 33, i32 15, float 1.0, float 1.0, float 1.0, float %z0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 34, i32 15, float 1.0, float 1.0, float 1.0, float %z0, i1 false, i1 false)
  %z1 = fsub float %y, %x
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float 0.0, float 0.0, float 0.0, float %z1, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 13, i32 15, float 0.0, float 0.0, float 0.0, float %z1, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 14, i32 15, float 0.0, float 0.0, float 0.0, float %z1, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos_before_param_across_load:
; PREGFX11: exp pos0
; PREGFX11-NEXT: exp param0
; PREGFX11-NEXT: exp param1
define amdgpu_kernel void @test_export_pos_before_param_across_load(i32 %idx) #0 {
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float 1.0, float 1.0, float 1.0, float 1.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 33, i32 15, float 1.0, float 1.0, float 1.0, float 0.5, i1 false, i1 false)
  %load = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> undef, i32 %idx, i32 0, i32 0)
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float 0.0, float 0.0, float 0.0, float %load, i1 true, i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_export_across_store_load:
; PREGFX11: buffer_store
; PREGFX11: buffer_load
; PREGFX11: exp pos0
; PREGFX11: exp param0
; PREGFX11: exp param1
define amdgpu_kernel void @test_export_across_store_load(i32 %idx, float %v) #0 {
  %data0 = alloca <4 x float>, align 8, addrspace(5)
  %data1 = alloca <4 x float>, align 8, addrspace(5)
  %cmp = icmp eq i32 %idx, 1
  %data = select i1 %cmp, <4 x float> addrspace(5)* %data0, <4 x float> addrspace(5)* %data1
  %sptr = getelementptr inbounds <4 x float>, <4 x float> addrspace(5)* %data, i32 0, i32 0
  store float %v, float addrspace(5)* %sptr, align 8
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float 0.0, float 0.0, float 0.0, float 1.0, i1 true, i1 false)
  %ptr0 = getelementptr inbounds <4 x float>, <4 x float> addrspace(5)* %data0, i32 0, i32 0
  %load0 = load float, float addrspace(5)* %ptr0, align 8
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %load0, float 0.0, float 1.0, float 0.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 33, i32 15, float %load0, float 0.0, float 1.0, float 0.0, i1 false, i1 false)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind inaccessiblememonly }
attributes #2 = { nounwind readnone }
