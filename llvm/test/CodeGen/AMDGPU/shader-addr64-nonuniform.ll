; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx600 -verify-machineinstrs <%s | FileCheck -enable-var-scope -check-prefixes=GCN,SICI,SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx700 -verify-machineinstrs <%s | FileCheck -enable-var-scope -check-prefixes=GCN,SICI,CI %s

; Check that an addrspace(1) (const) load with various combinations of
; uniform, nonuniform and constant address components all load with an
; addr64 mubuf with no readfirstlane.

@indexable = internal unnamed_addr addrspace(1) constant [6 x <3 x float>] [<3 x float> <float 1.000000e+00, float 0.000000e+00, float 0.000000e+00>, <3 x float> <float 0.000000e+00, float 1.000000e+00, float 0.000000e+00>, <3 x float> <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, <3 x float> <float 0.000000e+00, float 1.000000e+00, float 1.000000e+00>, <3 x float> <float 1.000000e+00, float 0.000000e+00, float 1.000000e+00>, <3 x float> <float 1.000000e+00, float 1.000000e+00, float 0.000000e+00>]

; GCN-LABEL: {{^}}nonuniform_uniform:
; GCN-NOT: readfirstlane
; SI: buffer_load_dwordx4 {{.*}} addr64
; CI: buffer_load_dwordx3 {{.*}} addr64

define amdgpu_ps float @nonuniform_uniform(i32 %arg18) {
.entry:
  %tmp31 = sext i32 %arg18 to i64
  %tmp32 = getelementptr [6 x <3 x float>], ptr addrspace(1) @indexable, i64 0, i64 %tmp31
  %tmp33 = load volatile <3 x float>, ptr addrspace(1) %tmp32, align 16
  %tmp34 = extractelement <3 x float> %tmp33, i32 0
  ret float %tmp34
}

; GCN-LABEL: {{^}}uniform_nonuniform:
; GCN-NOT: readfirstlane
; SI: buffer_load_dwordx4 {{.*}} addr64
; CI: buffer_load_dwordx3 {{.*}} addr64

define amdgpu_ps float @uniform_nonuniform(i32 inreg %offset, i32 %arg18) {
.entry:
  %tmp1 = zext i32 %arg18 to i64
  %tmp2 = inttoptr i64 %tmp1 to ptr addrspace(1)
  %tmp32 = getelementptr [6 x <3 x float>], ptr addrspace(1) %tmp2, i32 0, i32 %offset
  %tmp33 = load volatile <3 x float>, ptr addrspace(1) %tmp32, align 16
  %tmp34 = extractelement <3 x float> %tmp33, i32 0
  ret float %tmp34
}

; GCN-LABEL: {{^}}const_nonuniform:
; GCN-NOT: readfirstlane
; SI: buffer_load_dwordx4 {{.*}} addr64
; CI: buffer_load_dwordx3 {{.*}} addr64

define amdgpu_ps float @const_nonuniform(i32 %arg18) {
.entry:
  %tmp1 = zext i32 %arg18 to i64
  %tmp2 = inttoptr i64 %tmp1 to ptr addrspace(1)
  %tmp32 = getelementptr [6 x <3 x float>], ptr addrspace(1) %tmp2, i32 0, i32 1
  %tmp33 = load volatile <3 x float>, ptr addrspace(1) %tmp32, align 16
  %tmp34 = extractelement <3 x float> %tmp33, i32 0
  ret float %tmp34
}

; GCN-LABEL: {{^}}nonuniform_nonuniform:
; GCN-NOT: readfirstlane
; SI: buffer_load_dwordx4 {{.*}} addr64
; CI: buffer_load_dwordx3 {{.*}} addr64

define amdgpu_ps float @nonuniform_nonuniform(i32 %offset, i32 %arg18) {
.entry:
  %tmp1 = zext i32 %arg18 to i64
  %tmp2 = inttoptr i64 %tmp1 to ptr addrspace(1)
  %tmp32 = getelementptr [6 x <3 x float>], ptr addrspace(1) %tmp2, i32 0, i32 %offset
  %tmp33 = load volatile <3 x float>, ptr addrspace(1) %tmp32, align 16
  %tmp34 = extractelement <3 x float> %tmp33, i32 0
  ret float %tmp34
}

; GCN-LABEL: {{^}}nonuniform_uniform_const:
; GCN-NOT: readfirstlane
; SICI: buffer_load_dword {{.*}} addr64

define amdgpu_ps float @nonuniform_uniform_const(i32 %arg18) {
.entry:
  %tmp31 = sext i32 %arg18 to i64
  %tmp32 = getelementptr [6 x <3 x float>], ptr addrspace(1) @indexable, i64 0, i64 %tmp31, i64 1
  %tmp33 = load float, ptr addrspace(1) %tmp32, align 4
  ret float %tmp33
}

; GCN-LABEL: {{^}}uniform_nonuniform_const:
; GCN-NOT: readfirstlane
; SICI: buffer_load_dword {{.*}} addr64

define amdgpu_ps float @uniform_nonuniform_const(i32 inreg %offset, i32 %arg18) {
.entry:
  %tmp1 = zext i32 %arg18 to i64
  %tmp2 = inttoptr i64 %tmp1 to ptr addrspace(1)
  %tmp32 = getelementptr [6 x <3 x float>], ptr addrspace(1) %tmp2, i32 0, i32 %offset, i32 1
  %tmp33 = load float, ptr addrspace(1) %tmp32, align 4
  ret float %tmp33
}

; GCN-LABEL: {{^}}nonuniform_nonuniform_const:
; GCN-NOT: readfirstlane
; SICI: buffer_load_dword {{.*}} addr64

define amdgpu_ps float @nonuniform_nonuniform_const(i32 %offset, i32 %arg18) {
.entry:
  %tmp1 = zext i32 %arg18 to i64
  %tmp2 = inttoptr i64 %tmp1 to ptr addrspace(1)
  %tmp32 = getelementptr [6 x <3 x float>], ptr addrspace(1) %tmp2, i32 0, i32 %offset, i32 1
  %tmp33 = load float, ptr addrspace(1) %tmp32, align 4
  ret float %tmp33
}




