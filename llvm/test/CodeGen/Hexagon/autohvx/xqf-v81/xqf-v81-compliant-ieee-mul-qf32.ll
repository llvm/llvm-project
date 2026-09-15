; Tests compliant IEEE mode for XQFloat multiplication 32-bit for v81.

; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -force-hvx-float -enable-xqf-gen=true \
; RUN: -enable-rem-conv=true -hexagon-qfloat-mode=ieee  -mattr=+hvxv81,+hvx-length128B \
; RUN: < %s | FileCheck %s -check-prefix=CHECK

; Test qf32 = vmpy(sf, sf)
; Normalization of inputs
define <32 x float> @mul_add_0(<32 x float> %a0, <32 x float> %a1) #0 {
; CHECK-LABEL: mul_add_0:
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = v0.sf
; CHECK-DAG:     [[V4:v[0-9]+]].qf32 = v1.sf
; CHECK:         qf32 = vmpy([[V3]].qf32,[[V4]].qf32)
label0:
  %v3 = fmul <32 x float> %a0, %a1
  ret <32 x float> %v3
}

; Test qf32 = vmpy(sf ,qf32) when only one input is from vadd instruction
define <32 x float> @mul_add_1(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_add_1:
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = vadd(v0.sf,v2.sf)
; CHECK-DAG:     [[V4:v[0-9]+]].qf32 = v1.sf
; CHECK-DAG:     [[V5:v[0-9]+]].sf = [[V3]].qf32
; CHECK-DAG:     [[V6:v[0-9]+]].qf32 = [[V5]].sf
; CHECK:         qf32 = vmpy([[V4]].qf32,[[V6]].qf32)
label0:
  %v1 = fadd <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %a1, %v1
  ret <32 x float> %v3
}


; Test qf32 = vmpy(qf32 ,qf32) when both inputs are from vadd instruction
define <32 x float> @mul_add_3(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_add_3:
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = vadd(v0.sf,v2.sf)
; CHECK-DAG:     [[V4:v[0-9]+]].qf32 = vadd(v0.sf,v1.sf)
; CHECK-DAG:     [[V5:v[0-9]+]].qf32 = [[V3]].qf32
; CHECK-DAG:     [[V6:v[0-9]+]].qf32 = [[V4]].qf32
; CHECK:         qf32 = vmpy([[V6]].qf32,[[V5]].qf32)
label0:
  %v0 = fadd <32 x float> %a0, %a1
  %v1 = fadd <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %v0, %v1
  ret <32 x float> %v3
}

; Test qf32 = vmpy(qf32 ,qf32) when only first input is from vsub instruction
define <32 x float> @mul_sub_1(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_sub_1:
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = vsub(v0.sf,v2.sf)
; CHECK-DAG:     [[V4:v[0-9]+]].qf32 = v1.sf
; CHECK-DAG:     [[V5:v[0-9]+]].sf = [[V3]].qf32
; CHECK-DAG:     [[V6:v[0-9]+]].qf32 = [[V5]].sf
; CHECK:         qf32 = vmpy([[V4]].qf32,[[V6]].qf32)
label0:
  %v1 = fsub <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %a1, %v1
  ret <32 x float> %v3
}

; Test qf32 = vmpy(qf32 ,qf32) when both inputs are from vsub instruction
define <32 x float> @mul_sub_3(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_sub_3:
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = vsub(v0.sf,v2.sf)
; CHECK-DAG:     [[V4:v[0-9]+]].qf32 = vsub(v0.sf,v1.sf)
; CHECK-DAG:     [[V5:v[0-9]+]].qf32 = [[V3]].qf32
; CHECK-DAG:     [[V6:v[0-9]+]].qf32 = [[V4]].qf32
; CHECK:         qf32 = vmpy([[V6]].qf32,[[V5]].qf32)
label0:
  %v0 = fsub <32 x float> %a0, %a1
  %v1 = fsub <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %v0, %v1
  ret <32 x float> %v3
}

; Test qf32 = vmpy(qf32, qf32) when one is from adder, another from multiplier
define <32 x float> @mul_add_mul(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_add_mul:
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = vadd(v0.sf,v2.sf)
; CHECK-DAG:     [[V4:v[0-9]+]].qf32 = v0.sf
; CHECK-DAG:     [[V5:v[0-9]+]].qf32 = v1.sf
; CHECK-DAG:     [[V6:v[0-9]+]].qf32 = [[V3]].qf32
; CHECK-DAG:     [[V7:v[0-9]+]].qf32 = vmpy([[V4]].qf32,[[V5]].qf32)
; CHECK:         qf32 = vmpy([[V6]].qf32,[[V7]].qf32)
label0:
  %v1 = fadd <32 x float> %a0, %a2
  %v2 = fmul <32 x float> %a0, %a1
  %v3 = fmul <32 x float> %v1, %v2
  ret <32 x float> %v3
}

define <32 x float> @mul_mul_mul(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
label0:
; CHECK-LABEL: mul_mul_mul
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = v0.sf
; CHECK-DAG:     [[V4:v[0-9]+]].qf32 = v2.sf
; CHECK-DAG:     [[V5:v[0-9]+]].qf32 = v1.sf
; CHECK-DAG:     [[V6:v[0-9]+]].qf32 = vmpy([[V3]].qf32,[[V4]].qf32)
; CHECK-DAG:     [[V7:v[0-9]+]].qf32 = vmpy([[V3]].qf32,[[V5]].qf32)
; CHECK:         qf32 = vmpy([[V6]].qf32,[[V7]].qf32)
  %v1 = fmul <32 x float> %a0, %a2
  %v2 = fmul <32 x float> %a0, %a1
  %v3 = fmul <32 x float> %v1, %v2
  ret <32 x float> %v3
}

attributes #0 = { nofree nosync nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv81" "target-features"="+hvx-length128b,+hvx-qfloat,+hvxv81,+v81,-long-calls" "unsafe-fp-math"="true" }
