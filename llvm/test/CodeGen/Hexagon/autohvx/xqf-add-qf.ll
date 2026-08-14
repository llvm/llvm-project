; Tests strict-ieee mode for XQFloat for inputs to add/sub 16 and 32-bits. Tests for only Strict-IEEE mode.

; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true -hexagon-qfloat-mode=strict-ieee \
; RUN: -mattr=+hvxv79,+hvx-length128B < %s | FileCheck %s --enable-var-scope

; Test qf16 = vadd(hf ,hf) when no input is from adder
define <64 x half> @add_hf(<64 x half> %a0, <64 x half> %a1) #0 {
; CHECK-LABEL: add_hf
; CHECK: {{v[0-9]+}}.qf16 = vadd(v0.hf,v1.hf)
label0:
  %v0 = fadd <64 x half> %a0, %a1
  ret <64 x half> %v0
}

; Test qf32 = vadd(sf ,sf) when no input is from adder
define <32 x float> @add_sf(<32 x float> %a0, <32 x float> %a1) #0 {
; CHECK-LABEL: add_sf
; CHECK: {{v[0-9]+}}.qf32 = vadd(v0.sf,v1.sf)
label1:
  %v0 = fadd <32 x float> %a0, %a1
  ret <32 x float> %v0
}

; Test qf16 = vadd(qf16 ,hf) when first input is from vadd instruction
define <64 x half> @add_qf16_1(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2) #0 {
; CHECK-LABEL: add_qf16_1
; CHECK: [[V0:v[0-9]+]].qf16 = vadd(v0.hf,v1.hf)
; CHECK: [[V1:v[0-9]+]].hf = [[V0]].qf16
; CHECK: qf16 = vadd([[V1]].hf,v2.hf)
label2:
  %v0 = fadd <64 x half> %a0, %a1
  %v1 = fadd <64 x half> %v0, %a2
  ret <64 x half> %v1
}

; Test qf32 = vadd(qf32 ,sf) when first input is from vadd instruction
define <32 x float> @add_qf32_1(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: add_qf32_1
; CHECK: [[V0:v[0-9]+]].qf32 = vadd(v0.sf,v1.sf)
; CHECK: [[V1:v[0-9]+]].sf = [[V0]].qf32
; CHECK: qf32 = vadd([[V1]].sf,v2.sf)
label3:
  %v0 = fadd <32 x float> %a0, %a1
  %v1 = fadd <32 x float> %v0, %a2
  ret <32 x float> %v1
}

; Test qf16 = vadd(qf16 ,qf16) when both inputs are from vadd instruction
define <64 x half> @add_qf16_2(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2) #0 {
; CHECK-LABEL: add_qf16_2
; CHECK-DAG: [[V0:v[0-9]+]].qf16 = vsub(v0.hf,v1.hf)
; CHECK-DAG: [[V2:v[0-9]+]].hf = [[V0]].qf16
; CHECK-DAG: [[V1:v[0-9]+]].qf16 = vadd(v0.hf,v2.hf)
; CHECK-DAG: [[V3:v[0-9]+]].hf = [[V1]].qf16
; CHECK: qf16 = vadd([[V2]].hf,[[V3]].hf)
label4:
  %v1 = fadd <64 x half> %a0, %a2
  %v2 = fsub <64 x half> %a0, %a1
  %v3 = fadd <64 x half> %v2, %v1
  ret <64 x half> %v3
}

; Test qf32 = vadd(qf32 ,qf32) when both inputs are from vadd instruction
define <32 x float> @add_qf32_2(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: add_qf32_2
; CHECK-DAG: [[V0:v[0-9]+]].qf32 = vadd(v0.sf,v1.sf)
; CHECK-DAG: [[V1:v[0-9]+]].qf32 = vadd(v0.sf,v2.sf)
; CHECK-DAG: [[V2:v[0-9]+]].sf = [[V0]].qf32
; CHECK-DAG: [[V3:v[0-9]+]].sf = [[V1]].qf32
; CHECK: qf32 = vadd([[V3]].sf,[[V2]].sf)
label5:
  %v1 = fadd <32 x float> %a0, %a2
  %v2 = fadd <32 x float> %a0, %a1
  %v3 = fadd <32 x float> %v1, %v2
  ret <32 x float> %v3
}

; Test qf16 = vsub(hf , hf) when no input is from adder
define <64 x half> @sub_hf(<64 x half> %a0, <64 x half> %a1) #0 {
; CHECK-LABEL: sub_hf
; CHECK: {{v[0-9]+}}.qf16 = vsub(v0.hf,v1.hf)
label6:
  %v0 = fsub <64 x half> %a0, %a1
  ret <64 x half> %v0
}

; Test qf32 = vsub(sf ,sf) when no input is from adder
define <32 x float> @sub_sf(<32 x float> %a0, <32 x float> %a1) #0 {
; CHECK-LABEL: sub_sf
; CHECK: {{v[0-9]+}}.qf32 = vsub(v0.sf,v1.sf)
label7:
  %v0 = fsub <32 x float> %a0, %a1
  ret <32 x float> %v0
}

; Test qf16 = vsub(qf16 ,hf) when first input is from vsub instruction
define <64 x half> @sub_qf16_1(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2) #0 {
; CHECK-LABEL: sub_qf16_1
; CHECK: [[V0:v[0-9]+]].qf16 = vsub(v0.hf,v1.hf)
; CHECK: [[V1:v[0-9]+]].hf = [[V0]].qf16
; CHECK: qf16 = vsub([[V1]].hf,v2.hf)
label8:
  %v0 = fsub <64 x half> %a0, %a1
  %v1 = fsub <64 x half> %v0, %a2
  ret <64 x half> %v1
}

; Test qf32 = vsub(qf32 ,sf) when first input is from vsub instruction
define <32 x float> @sub_qf32_1(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: sub_qf32_1
; CHECK: [[V0:v[0-9]+]].qf32 = vadd(v0.sf,v1.sf)
; CHECK: [[V1:v[0-9]+]].sf = [[V0]].qf32
; CHECK: qf32 = vsub([[V1]].sf,v2.sf)
label9:
  %v0 = fadd <32 x float> %a0, %a1
  %v1 = fsub <32 x float> %v0, %a2
  ret <32 x float> %v1
}

; Test qf16 = vsub(qf16 ,qf16) when both inputs are from vadd/vsub instruction
define <64 x half> @sub_qf16_2(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2) #0 {
; CHECK-LABEL: sub_qf16_2
; CHECK-DAG: [[V0:v[0-9]+]].qf16 = vadd(v0.hf,v1.hf)
; CHECK-DAG: [[V1:v[0-9]+]].qf16 = vadd(v0.hf,v2.hf)
; CHECK-DAG: [[V2:v[0-9]+]].hf = [[V0]].qf16
; CHECK-DAG: [[V3:v[0-9]+]].hf = [[V1]].qf16
; CHECK: qf16 = vsub([[V2]].hf,[[V3]].hf)
label10:
  %v1 = fadd <64 x half> %a0, %a2
  %v2 = fadd <64 x half> %a0, %a1
  %v3 = fsub <64 x half> %v2, %v1
  ret <64 x half> %v3
}

; Test qf32 = vsub(qf32 ,qf32) when first input is from vadd/vsub instruction
define <32 x float> @add_mul_qf32_2(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: add_mul_qf32_2
; CHECK-DAG: [[R0:r[0-9]+]] = ##2147483648
; CHECK-DAG: [[V0:v[0-9]+]].qf32 = vadd(v0.sf,v2.sf)
; CHECK-DAG: [[V1:v[0-9]+]] = vxor([[V1]],[[V1]])
; CHECK-DAG: [[V2:v[0-9]+]] = vsplat([[R0]])
; CHECK-DAG: [[V3:v[0-9]+]].sf = [[V0]].qf32
; CHECK-DAG: [[V4:v[0-9]+]].qf32 = vmpy([[V1]].sf,[[V2]].sf)
; CHECK-DAG: [[V5:v[0-9]+]].qf32 = vadd([[V4]].qf32,v0.sf)
; CHECK-DAG: [[V6:v[0-9]+]].qf32 = vadd([[V4]].qf32,v1.sf)
; CHECK: [[V7:v[0-9]+]].qf32 = vmpy([[V5]].qf32,[[V6]].qf32)
; CHECK: [[V8:v[0-9]+]].sf = [[V7]].qf32
; CHECK: qf32 = vsub([[V8]].sf,[[V3]].sf)
label11:
  %v1 = fadd <32 x float> %a0, %a2
  %v2 = fmul <32 x float> %a0, %a1
  %v0 = fsub <32 x float> %v2, %v1
  ret <32 x float> %v0
}


attributes #0 = { nofree nosync nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-length128b,+hvx-qfloat,-long-calls" "unsafe-fp-math"="true" }
