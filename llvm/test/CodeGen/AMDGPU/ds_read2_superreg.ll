; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -mattr=+load-store-opt < %s | FileCheck -check-prefix=CI %s

@lds = addrspace(3) global [512 x float] undef, align 4
@lds.v2 = addrspace(3) global [512 x <2 x float>] undef, align 4
@lds.v3 = addrspace(3) global [512 x <3 x float>] undef, align 4
@lds.v4 = addrspace(3) global [512 x <4 x float>] undef, align 4
@lds.v8 = addrspace(3) global [512 x <8 x float>] undef, align 4
@lds.v16 = addrspace(3) global [512 x <16 x float>] undef, align 4

; CI-LABEL: {{^}}simple_read2_v2f32_superreg_align4:
; CI: ds_read2_b32 [[RESULT:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset1:1{{$}}
; CI: s_waitcnt lgkmcnt(0)
; CI: buffer_store_dwordx2 [[RESULT]]
; CI: s_endpgm
define void @simple_read2_v2f32_superreg_align4(<2 x float> addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds  [512 x <2 x float>], [512 x <2 x float>] addrspace(3)* @lds.v2, i32 0, i32 %x.i
  %val0 = load <2 x float>, <2 x float> addrspace(3)* %arrayidx0, align 4
  %out.gep = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %out, i32 %x.i
  store <2 x float> %val0, <2 x float> addrspace(1)* %out.gep
  ret void
}

; CI-LABEL: {{^}}simple_read2_v2f32_superreg:
; CI: ds_read_b64 [[RESULT:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}{{$}}
; CI: s_waitcnt lgkmcnt(0)
; CI: buffer_store_dwordx2 [[RESULT]]
; CI: s_endpgm
define void @simple_read2_v2f32_superreg(<2 x float> addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <2 x float>], [512 x <2 x float>] addrspace(3)* @lds.v2, i32 0, i32 %x.i
  %val0 = load <2 x float>, <2 x float> addrspace(3)* %arrayidx0
  %out.gep = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %out, i32 %x.i
  store <2 x float> %val0, <2 x float> addrspace(1)* %out.gep
  ret void
}

; CI-LABEL: {{^}}simple_read2_v4f32_superreg_align4:
; CI-DAG: ds_read2_b32 v{{\[}}[[REG_X:[0-9]+]]:[[REG_Y:[0-9]+]]{{\]}}, v{{[0-9]+}} offset1:1{{$}}
; CI-DAG: ds_read2_b32 v{{\[}}[[REG_Z:[0-9]+]]:[[REG_W:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:2 offset1:3{{$}}
; CI-DAG: v_add_f32_e32 v[[ADD0:[0-9]+]], v[[REG_Z]], v[[REG_X]]
; CI-DAG: v_add_f32_e32 v[[ADD1:[0-9]+]], v[[REG_W]], v[[REG_Y]]
; CI: v_add_f32_e32 v[[ADD2:[0-9]+]], v[[ADD1]], v[[ADD0]]
; CI: buffer_store_dword v[[ADD2]]
; CI: s_endpgm
define void @simple_read2_v4f32_superreg_align4(float addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <4 x float>], [512 x <4 x float>] addrspace(3)* @lds.v4, i32 0, i32 %x.i
  %val0 = load <4 x float>, <4 x float> addrspace(3)* %arrayidx0, align 4
  %elt0 = extractelement <4 x float> %val0, i32 0
  %elt1 = extractelement <4 x float> %val0, i32 1
  %elt2 = extractelement <4 x float> %val0, i32 2
  %elt3 = extractelement <4 x float> %val0, i32 3

  %add0 = fadd float %elt0, %elt2
  %add1 = fadd float %elt1, %elt3
  %add2 = fadd float %add0, %add1

  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %add2, float addrspace(1)* %out.gep
  ret void
}


; FIXME: the v_lshl_b64 x, x, 32 is a bad way of doing a copy

; CI-LABEL: {{^}}simple_read2_v3f32_superreg_align4:
; CI-DAG: ds_read2_b32 v{{\[}}[[REG_X:[0-9]+]]:[[REG_Y:[0-9]+]]{{\]}}, v{{[0-9]+}} offset1:1{{$}}
; CI-DAG: ds_read_b32 v[[REG_Z:[0-9]+]], v{{[0-9]+}} offset:8{{$}}
; CI: v_lshr_b64 v{{\[}}[[Y_COPY:[0-9]+]]:{{[0-9]+\]}}, v{{\[}}[[REG_X]]:[[REG_Y]]{{\]}}, 32
; CI-DAG: v_add_f32_e32 v[[ADD0:[0-9]+]], v[[REG_Z]], v[[REG_X]]
; CI-DAG: v_add_f32_e32 v[[ADD1:[0-9]+]], v[[Y_COPY]], v[[ADD0]]
; CI: buffer_store_dword v[[ADD1]]
; CI: s_endpgm
define void @simple_read2_v3f32_superreg_align4(float addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <3 x float>], [512 x <3 x float>] addrspace(3)* @lds.v3, i32 0, i32 %x.i
  %val0 = load <3 x float>, <3 x float> addrspace(3)* %arrayidx0, align 4
  %elt0 = extractelement <3 x float> %val0, i32 0
  %elt1 = extractelement <3 x float> %val0, i32 1
  %elt2 = extractelement <3 x float> %val0, i32 2

  %add0 = fadd float %elt0, %elt2
  %add1 = fadd float %add0, %elt1

  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %add1, float addrspace(1)* %out.gep
  ret void
}

; CI-LABEL: {{^}}simple_read2_v4f32_superreg_align8:
; CI-DAG: ds_read2_b64 v{{\[}}[[REG_W:[0-9]+]]:[[REG_Z:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:1{{$}}

; FIXME: These moves shouldn't be necessary, it should be able to
; store the same register if offset1 was the non-zero offset.

; CI: v_mov_b32
; CI: v_mov_b32
; CI: buffer_store_dwordx4
; CI: s_endpgm
define void @simple_read2_v4f32_superreg_align8(<4 x float> addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <4 x float>], [512 x <4 x float>] addrspace(3)* @lds.v4, i32 0, i32 %x.i
  %val0 = load <4 x float>, <4 x float> addrspace(3)* %arrayidx0, align 8
  %out.gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i32 %x.i
  store <4 x float> %val0, <4 x float> addrspace(1)* %out.gep
  ret void
}

; CI-LABEL: {{^}}simple_read2_v4f32_superreg:
; CI: ds_read2_b64 v{{\[}}[[REG_W:[0-9]+]]:[[REG_Z:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:1{{$}}
; CI: v_mov_b32
; CI: v_mov_b32
; CI: buffer_store_dwordx4
; CI: s_endpgm
define void @simple_read2_v4f32_superreg(<4 x float> addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <4 x float>], [512 x <4 x float>] addrspace(3)* @lds.v4, i32 0, i32 %x.i
  %val0 = load <4 x float>, <4 x float> addrspace(3)* %arrayidx0
  %out.gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i32 %x.i
  store <4 x float> %val0, <4 x float> addrspace(1)* %out.gep
  ret void
}

; FIXME: Extra moves shuffling superregister
; CI-LABEL: {{^}}simple_read2_v8f32_superreg:
; CI: ds_read2_b64 v{{\[}}[[REG_ELT3:[0-9]+]]:[[REG_ELT7:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:1 offset1:3{{$}}
; CI: v_mov_b32
; CI: v_mov_b32
; CI: ds_read2_b64 v{{\[}}[[REG_ELT6:[0-9]+]]:[[REG_ELT5:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:2{{$}}
; CI: v_mov_b32
; CI: v_mov_b32
; CI: buffer_store_dwordx4
; CI: buffer_store_dwordx4
; CI: s_endpgm
define void @simple_read2_v8f32_superreg(<8 x float> addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <8 x float>], [512 x <8 x float>] addrspace(3)* @lds.v8, i32 0, i32 %x.i
  %val0 = load <8 x float>, <8 x float> addrspace(3)* %arrayidx0
  %out.gep = getelementptr inbounds <8 x float>, <8 x float> addrspace(1)* %out, i32 %x.i
  store <8 x float> %val0, <8 x float> addrspace(1)* %out.gep
  ret void
}

; FIXME: Extra moves shuffling superregister
; CI-LABEL: {{^}}simple_read2_v16f32_superreg:
; CI: ds_read2_b64 v{{\[}}[[REG_ELT11:[0-9]+]]:[[REG_ELT15:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:1 offset1:3{{$}}
; CI: v_mov_b32
; CI: v_mov_b32
; CI: ds_read2_b64 v{{\[}}[[REG_ELT14:[0-9]+]]:[[REG_ELT13:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:5 offset1:7{{$}}
; CI: ds_read2_b64 v{{\[}}[[REG_ELT14:[0-9]+]]:[[REG_ELT13:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:6 offset1:4{{$}}
; CI: v_mov_b32
; CI: v_mov_b32
; CI: ds_read2_b64 v{{\[}}[[REG_ELT12:[0-9]+]]:[[REG_ELT10:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:2{{$}}
; CI: v_mov_b32
; CI: v_mov_b32

; CI: s_waitcnt lgkmcnt(0)
; CI: buffer_store_dwordx4
; CI: buffer_store_dwordx4
; CI: buffer_store_dwordx4
; CI: buffer_store_dwordx4
; CI: s_endpgm
define void @simple_read2_v16f32_superreg(<16 x float> addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <16 x float>], [512 x <16 x float>] addrspace(3)* @lds.v16, i32 0, i32 %x.i
  %val0 = load <16 x float>, <16 x float> addrspace(3)* %arrayidx0
  %out.gep = getelementptr inbounds <16 x float>, <16 x float> addrspace(1)* %out, i32 %x.i
  store <16 x float> %val0, <16 x float> addrspace(1)* %out.gep
  ret void
}

; Do scalar loads into the super register we need.
; CI-LABEL: {{^}}simple_read2_v2f32_superreg_scalar_loads_align4:
; CI-DAG: ds_read2_b32 v{{\[}}[[REG_ELT0:[0-9]+]]:[[REG_ELT1:[0-9]+]]{{\]}}, v{{[0-9]+}} offset1:1{{$}}
; CI-NOT: v_mov
; CI: buffer_store_dwordx2 v{{\[}}[[REG_ELT0]]:[[REG_ELT1]]{{\]}}
; CI: s_endpgm
define void @simple_read2_v2f32_superreg_scalar_loads_align4(<2 x float> addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  %arrayidx1 = getelementptr inbounds float, float addrspace(3)* %arrayidx0, i32 1

  %val0 = load float, float addrspace(3)* %arrayidx0
  %val1 = load float, float addrspace(3)* %arrayidx1

  %vec.0 = insertelement <2 x float> undef, float %val0, i32 0
  %vec.1 = insertelement <2 x float> %vec.0, float %val1, i32 1

  %out.gep = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %out, i32 %x.i
  store <2 x float> %vec.1, <2 x float> addrspace(1)* %out.gep
  ret void
}

; Do scalar loads into the super register we need.
; CI-LABEL: {{^}}simple_read2_v4f32_superreg_scalar_loads_align4:
; CI-DAG: ds_read2_b32 v{{\[}}[[REG_ELT0:[0-9]+]]:[[REG_ELT1:[0-9]+]]{{\]}}, v{{[0-9]+}} offset1:1{{$}}
; CI-DAG: ds_read2_b32 v{{\[}}[[REG_ELT2:[0-9]+]]:[[REG_ELT3:[0-9]+]]{{\]}}, v{{[0-9]+}} offset0:2 offset1:3{{$}}
; CI-NOT: v_mov
; CI: buffer_store_dwordx4 v{{\[}}[[REG_ELT0]]:[[REG_ELT3]]{{\]}}
; CI: s_endpgm
define void @simple_read2_v4f32_superreg_scalar_loads_align4(<4 x float> addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  %arrayidx1 = getelementptr inbounds float, float addrspace(3)* %arrayidx0, i32 1
  %arrayidx2 = getelementptr inbounds float, float addrspace(3)* %arrayidx0, i32 2
  %arrayidx3 = getelementptr inbounds float, float addrspace(3)* %arrayidx0, i32 3

  %val0 = load float, float addrspace(3)* %arrayidx0
  %val1 = load float, float addrspace(3)* %arrayidx1
  %val2 = load float, float addrspace(3)* %arrayidx2
  %val3 = load float, float addrspace(3)* %arrayidx3

  %vec.0 = insertelement <4 x float> undef, float %val0, i32 0
  %vec.1 = insertelement <4 x float> %vec.0, float %val1, i32 1
  %vec.2 = insertelement <4 x float> %vec.1, float %val2, i32 2
  %vec.3 = insertelement <4 x float> %vec.2, float %val3, i32 3

  %out.gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i32 %x.i
  store <4 x float> %vec.3, <4 x float> addrspace(1)* %out.gep
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.y() #1

; Function Attrs: noduplicate nounwind
declare void @llvm.AMDGPU.barrier.local() #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { noduplicate nounwind }
