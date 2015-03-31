;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}gather4_v2:
;CHECK: image_gather4 {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_v2() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.v2i32(<2 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4:
;CHECK: image_gather4 {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_cl:
;CHECK: image_gather4_cl {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.cl.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_l:
;CHECK: image_gather4_l {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_l() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.l.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_b:
;CHECK: image_gather4_b {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_b() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.b.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_b_cl:
;CHECK: image_gather4_b_cl {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_b_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.b.cl.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_b_cl_v8:
;CHECK: image_gather4_b_cl {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_b_cl_v8() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.b.cl.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_lz_v2:
;CHECK: image_gather4_lz {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_lz_v2() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.lz.v2i32(<2 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_lz:
;CHECK: image_gather4_lz {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_lz() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.lz.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}



;CHECK-LABEL: {{^}}gather4_o:
;CHECK: image_gather4_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.o.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_cl_o:
;CHECK: image_gather4_cl_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_cl_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.cl.o.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_cl_o_v8:
;CHECK: image_gather4_cl_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_cl_o_v8() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.cl.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_l_o:
;CHECK: image_gather4_l_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_l_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.l.o.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_l_o_v8:
;CHECK: image_gather4_l_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_l_o_v8() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.l.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_b_o:
;CHECK: image_gather4_b_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_b_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.b.o.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_b_o_v8:
;CHECK: image_gather4_b_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_b_o_v8() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.b.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_b_cl_o:
;CHECK: image_gather4_b_cl_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_b_cl_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.b.cl.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_lz_o:
;CHECK: image_gather4_lz_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_lz_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.lz.o.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}



;CHECK-LABEL: {{^}}gather4_c:
;CHECK: image_gather4_c {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_cl:
;CHECK: image_gather4_c_cl {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.cl.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_cl_v8:
;CHECK: image_gather4_c_cl {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_cl_v8() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.cl.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_l:
;CHECK: image_gather4_c_l {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_l() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.l.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_l_v8:
;CHECK: image_gather4_c_l {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_l_v8() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.l.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_b:
;CHECK: image_gather4_c_b {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_b() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.b.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_b_v8:
;CHECK: image_gather4_c_b {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_b_v8() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.b.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_b_cl:
;CHECK: image_gather4_c_b_cl {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_b_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.b.cl.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_lz:
;CHECK: image_gather4_c_lz {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_lz() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.lz.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}



;CHECK-LABEL: {{^}}gather4_c_o:
;CHECK: image_gather4_c_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.o.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_o_v8:
;CHECK: image_gather4_c_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_o_v8() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_cl_o:
;CHECK: image_gather4_c_cl_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_cl_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.cl.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_l_o:
;CHECK: image_gather4_c_l_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_l_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.l.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_b_o:
;CHECK: image_gather4_c_b_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_b_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.b.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_b_cl_o:
;CHECK: image_gather4_c_b_cl_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_b_cl_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.b.cl.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_lz_o:
;CHECK: image_gather4_c_lz_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_lz_o() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.lz.o.v4i32(<4 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}gather4_c_lz_o_v8:
;CHECK: image_gather4_c_lz_o {{v\[[0-9]+:[0-9]+\]}}, 1, 0, 0, -1, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @gather4_c_lz_o_v8() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.gather4.c.lz.o.v8i32(<8 x i32> undef, <32 x i8> undef, <16 x i8> undef, i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}



declare <4 x float> @llvm.SI.gather4.v2i32(<2 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.cl.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.l.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.b.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.b.cl.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.b.cl.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.lz.v2i32(<2 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.lz.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1

declare <4 x float> @llvm.SI.gather4.o.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.cl.o.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.cl.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.l.o.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.l.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.b.o.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.b.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.b.cl.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.lz.o.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1

declare <4 x float> @llvm.SI.gather4.c.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.cl.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.cl.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.l.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.l.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.b.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.b.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.b.cl.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.lz.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1

declare <4 x float> @llvm.SI.gather4.c.o.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.cl.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.l.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.b.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.b.cl.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.lz.o.v4i32(<4 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.gather4.c.lz.o.v8i32(<8 x i32>, <32 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }
