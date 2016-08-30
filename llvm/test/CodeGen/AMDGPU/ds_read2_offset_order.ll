; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -mattr=+load-store-opt < %s | FileCheck -strict-whitespace -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -mattr=+load-store-opt < %s | FileCheck -strict-whitespace -check-prefix=SI %s


@lds = addrspace(3) global [512 x float] undef, align 4

; offset0 is larger than offset1

; SI-LABEL: {{^}}offset_order:

; SI-DAG: ds_read2_b32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} offset0:2 offset1:3
; SI-DAG: ds_read2_b32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} offset0:12 offset1:14
; SI-DAG: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:44

define void @offset_order(float addrspace(1)* %out) {
entry:
  %ptr0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 0
  %val0 = load float, float addrspace(3)* %ptr0

  %ptr1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 256
  %val1 = load float, float addrspace(3)* %ptr1
  %add1 = fadd float %val0, %val1

  %ptr2 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 3
  %val2 = load float, float addrspace(3)* %ptr2
  %add2 = fadd float %add1, %val2

  %ptr3 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 2
  %val3 = load float, float addrspace(3)* %ptr3
  %add3 = fadd float %add2, %val3

  %ptr4 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 12
  %val4 = load float, float addrspace(3)* %ptr4
  %add4 = fadd float %add3, %val4

  %ptr5 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 14
  %val5 = load float, float addrspace(3)* %ptr5
  %add5 = fadd float %add4, %val5

  %ptr6 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 11
  %val6 = load float, float addrspace(3)* %ptr6
  %add6 = fadd float %add5, %val6
  store float %add6, float addrspace(1)* %out
  ret void
}
