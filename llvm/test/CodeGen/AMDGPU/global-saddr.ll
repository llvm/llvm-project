; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -amdgpu-enable-global-sgpr-addr < %s | FileCheck -check-prefix=GFX9 %s

; Test for a conv2d like sequence of loads.

; GFX9: global_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} offset:16{{$}}
; GFX9: global_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}{{$}}
; GFX9: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} offset:32{{$}}
; GFX9: global_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} offset:-16{{$}}
; GFX9: global_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} offset:-32{{$}}
; GFX9: global_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} offset:8{{$}}

define hidden amdgpu_kernel void @simpleSaddrs(i64 addrspace(1)* %dst_image, i64 addrspace(1)* %src_image ) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = zext i32 %id to i64
  %gep = getelementptr i64, i64 addrspace(1)* %src_image, i64 %idx
  %ptr0 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 1
  %load0 = load i64, i64 addrspace(1)* %ptr0
  %ptr1 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 2
  %load1 = load i64, i64 addrspace(1)* %ptr1
  %ptr2 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 3
  %load2 = load i64, i64 addrspace(1)* %ptr2
  %ptr3 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 4
  %load3 = load i64, i64 addrspace(1)* %ptr3
  %ptr4 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 -4
  %load4 = load i64, i64 addrspace(1)* %ptr4
  %ptr5 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 -3
  %load5 = load i64, i64 addrspace(1)* %ptr5
  %ptr6 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 -2
  %load6 = load i64, i64 addrspace(1)* %ptr6
  %ptr7 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 -1
  %load7 = load i64, i64 addrspace(1)* %ptr7
  %ptr8 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 0
  %load8 = load i64, i64 addrspace(1)* %ptr8
  %add0 = add i64 %load1, %load0
  %add1 = add i64 %load3, %load2
  %add2 = add i64 %load5, %load4
  %add3 = add i64 %load7, %load6
  %add4 = add i64 %add0, %load8
  %add5 = add i64 %add2, %add1
  %add6 = add i64 %add4, %add3
  %add7 = add i64 %add6, %add5
  %gep9 = getelementptr i64, i64 addrspace(1)* %dst_image, i64 %idx
  %ptr9 = getelementptr inbounds i64, i64 addrspace(1)* %gep9, i64 1
  store volatile i64 %add7, i64 addrspace(1)* %ptr9

; Test various offset boundaries.
; GFX9: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+}}:{{[0-9]+}}] offset:4088{{$}}
; GFX9: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, off offset:4088{{$}}
; GFX9: global_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+}}:{{[0-9]+}}] offset:2040{{$}}
  %gep11 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 511
  %load11 = load i64, i64 addrspace(1)* %gep11
  %gep12 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 1023
  %load12 = load i64, i64 addrspace(1)* %gep12
  %gep13 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 255
  %load13 = load i64, i64 addrspace(1)* %gep13
  %add11 = add i64 %load11, %load12
  %add12 = add i64 %add11, %load13
  store volatile i64 %add12, i64 addrspace(1)* undef

; GFX9: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
; GFX9: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
; GFX9: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+}}:{{[0-9]+}}] offset:-4096{{$}}
  %gep21 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 -1024
  %load21 = load i64, i64 addrspace(1)* %gep21
  %gep22 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 -2048
  %load22 = load i64, i64 addrspace(1)* %gep22
  %gep23 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 -512
  %load23 = load i64, i64 addrspace(1)* %gep23
  %add21 = add i64 %load22, %load21
  %add22 = add i64 %add21, %load23
  store volatile i64 %add22, i64 addrspace(1)* undef

; GFX9: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+}}:{{[0-9]+}}] offset:2040{{$}}
  %gep31 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 257
  %load31 = load i64, i64 addrspace(1)* %gep31
  %gep32 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 256
  %load32 = load i64, i64 addrspace(1)* %gep32
  %gep33 = getelementptr inbounds i64, i64 addrspace(1)* %gep, i64 255
  %load33 = load i64, i64 addrspace(1)* %gep33
  %add34 = add i64 %load32, %load31
  %add35 = add i64 %add34, %load33
  store volatile i64 %add35, i64 addrspace(1)* undef
  ret void
}

; GFX9: global_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
; GFX9: global_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, off offset:16{{$}}
; GFX9-NEXT: s_waitcnt
; NGFX9-NOT: global_load_dword

define amdgpu_cs void @_amdgpu_cs_main(i64 inreg %arg) {
bb:
  %tmp1 = inttoptr i64 %arg to <4 x i64> addrspace(1)*
  %tmp2 = load <4 x i64>, <4 x i64> addrspace(1)* %tmp1, align 16
  store volatile <4 x i64> %tmp2, <4 x i64> addrspace(1)* undef
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
attributes #0 = { convergent nounwind }
attributes #1 = { nounwind readnone speculatable }
