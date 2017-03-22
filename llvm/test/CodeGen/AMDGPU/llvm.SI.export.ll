; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=GCN %s

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float) #0

; GCN-LABEL: {{^}}test_export_zeroes:
; GCN: exp mrt0 off, off, off, off{{$}}
; GCN: exp mrt0 off, off, off, off done{{$}}
define amdgpu_kernel void @test_export_zeroes() #0 {

  call void @llvm.SI.export(i32 0, i32 0, i32 0, i32 0, i32 0, float 0.0, float 0.0, float 0.0, float 0.0)
  call void @llvm.SI.export(i32 0, i32 0, i32 1, i32 0, i32 0, float 0.0, float 0.0, float 0.0, float 0.0)
  ret void
}

; FIXME: Should not set up registers for the unused source registers.

; GCN-LABEL: {{^}}test_export_en_src0:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], off, off, off done{{$}}
define amdgpu_kernel void @test_export_en_src0() #0 {
  call void @llvm.SI.export(i32 1, i32 0, i32 1, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src1:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 off, [[SRC1]], off, off done{{$}}
define amdgpu_kernel void @test_export_en_src1() #0 {
  call void @llvm.SI.export(i32 2, i32 0, i32 1, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src2:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 off, off, [[SRC2]], off done{{$}}
define amdgpu_kernel void @test_export_en_src2() #0 {
  call void @llvm.SI.export(i32 4, i32 0, i32 1, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src3:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 off, off, off, [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_en_src3() #0 {
  call void @llvm.SI.export(i32 8, i32 0, i32 1, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src1:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], [[SRC1]], off, off done{{$}}
define amdgpu_kernel void @test_export_en_src0_src1() #0 {
  call void @llvm.SI.export(i32 3, i32 0, i32 1, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src2:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], off, [[SRC2]], off done{{$}}
define amdgpu_kernel void @test_export_en_src0_src2() #0 {
  call void @llvm.SI.export(i32 5, i32 0, i32 1, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src3:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], off, off, [[SRC3]]{{$}}
; GCN: exp mrt0 [[SRC0]], off, off, [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_en_src0_src3() #0 {
  call void @llvm.SI.export(i32 9, i32 0, i32 0, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 9, i32 0, i32 1, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_en_src0_src1_src2_src3:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_en_src0_src1_src2_src3() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_mrt7:
; GCN-DAG: v_mov_b32_e32 [[VHALF:v[0-9]+]], 0.5
; GCN: exp mrt7 [[VHALF]], [[VHALF]], [[VHALF]], [[VHALF]]{{$}}
; GCN: exp mrt7 [[VHALF]], [[VHALF]], [[VHALF]], [[VHALF]] done{{$}}
define amdgpu_kernel void @test_export_mrt7() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 7, i32 0, float 0.5, float 0.5, float 0.5, float 0.5)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 7, i32 0, float 0.5, float 0.5, float 0.5, float 0.5)
  ret void
}

; GCN-LABEL: {{^}}test_export_z:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrtz [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp mrtz [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_z() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 8, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 8, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_null:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp null [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_null() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 9, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 9, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_reserved10:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp invalid_target_10 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp invalid_target_10 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_reserved10() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 10, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 10, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_reserved11:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp invalid_target_11 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp invalid_target_11 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_reserved11() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 11, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 11, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos0:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp pos0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp pos0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_pos0() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 12, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_pos3:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp pos3 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp pos3 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_pos3() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 15, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 15, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_param0:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp param0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_param0() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 32, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 32, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_param31:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]]{{$}}
; GCN: exp param31 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done{{$}}
define amdgpu_kernel void @test_export_param31() #0 {
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 63, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 63, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_export_vm:
; GCN-DAG: v_mov_b32_e32 [[SRC0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[SRC1:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[SRC2:v[0-9]+]], 0.5
; GCN-DAG: v_mov_b32_e32 [[SRC3:v[0-9]+]], 4.0
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] vm{{$}}
; GCN: exp mrt0 [[SRC0]], [[SRC1]], [[SRC2]], [[SRC3]] done vm{{$}}
define amdgpu_kernel void @test_export_vm() #0 {
  call void @llvm.SI.export(i32 15, i32 1, i32 0, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float 1.0, float 2.0, float 0.5, float 4.0)
  ret void
}

attributes #0 = { nounwind "ShaderType"="0" }
