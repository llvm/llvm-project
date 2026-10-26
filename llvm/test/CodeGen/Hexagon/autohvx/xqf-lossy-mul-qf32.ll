;n Tests lossy-subnormals mode for XQFloat multiplication 32-bit

; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true -enable-rem-conv=true -hexagon-qfloat-mode=lossy  -mattr=+hvxv79,+hvx-length128B < %s | FileCheck %s

; Test qf32 = vmpy(sf, sf)
; Normalization of inputs
define <32 x float> @mul_add_0(<32 x float> %a0, <32 x float> %a1) #0 {
; CHECK-LABEL: mul_add_0:
; CHECK:     qf32 = vmpy(v0.sf,v1.sf)
label0:
  %v3 = fmul <32 x float> %a0, %a1
  ret <32 x float> %v3
}

; Test qf32 = vmpy(sf ,qf32) when only one input is from vadd instruction
define <32 x float> @mul_add_1(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_add_1:
; CHECK:     [[V3:v[0-9]+]].qf32 = vadd(v0.sf,v2.sf)
; CHECK:     [[V4:v[0-9]+]].sf = [[V3]].qf32
; CHECK:     qf32 = vmpy(v1.sf,[[V4]].sf)
label0:
  %v1 = fadd <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %a1, %v1
  ret <32 x float> %v3
}


; Test qf32 = vmpy(qf32 ,qf32) when both inputs are from vadd instruction
define <32 x float> @mul_add_3(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_add_3:
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = vadd(v0.sf,v2.sf)
; CHECK-DAG:     [[R0:r[0-9]+]] = ##2147483648
; CHECK-DAG:     [[V4:v[0-9]+]] = vxor([[V4]],[[V4]])
; CHECK-DAG:     [[V6:v[0-9]+]] = vsplat([[R0]])
; CHECK-DAG:     [[V5:v[0-9]+]].qf32 = vadd(v0.sf,v1.sf)
; CHECK-DAG:     [[V7:v[0-9]+]].qf32 = vmpy([[V4]].sf,[[V6]].sf)
; CHECK-DAG:     [[V8:v[0-9]+]].qf32 = vadd([[V7]].qf32,[[V5]].qf32)
; CHECK-DAG:     [[V9:v[0-9]+]].qf32 = vadd([[V7]].qf32,[[V3]].qf32)
; CHECK:         qf32 = vmpy([[V8]].qf32,[[V9]].qf32)
label0:
  %v0 = fadd <32 x float> %a0, %a1
  %v1 = fadd <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %v0, %v1
  ret <32 x float> %v3
}

; Test qf32 = vmpy(qf32 ,qf32) when only first input is from vsub instruction
define <32 x float> @mul_sub_1(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_sub_1:
; CHECK:     [[V3:v[0-9]+]].qf32 = vsub(v0.sf,v2.sf)
; CHECK:     [[V4:v[0-9]+]].sf = [[V3]].qf32
; CHECK:     qf32 = vmpy(v1.sf,[[V4]].sf)
label0:
  %v1 = fsub <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %a1, %v1
  ret <32 x float> %v3
}

; Test qf32 = vmpy(qf32 ,qf32) when both inputs are from vsub instruction
define <32 x float> @mul_sub_3(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_sub_3:
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = vsub(v0.sf,v2.sf)
; CHECK-DAG:     [[R0:r[0-9]+]] = ##2147483648
; CHECK-DAG:     [[V4:v[0-9]+]] = vxor([[V4]],[[V4]])
; CHECK-DAG:     [[V6:v[0-9]+]] = vsplat([[R0]])
; CHECK-DAG:     [[V5:v[0-9]+]].qf32 = vsub(v0.sf,v1.sf)
; CHECK-DAG:     [[V7:v[0-9]+]].qf32 = vmpy([[V4]].sf,[[V6]].sf)
; CHECK-DAG:     [[V8:v[0-9]+]].qf32 = vadd([[V7]].qf32,[[V5]].qf32)
; CHECK-DAG:     [[V9:v[0-9]+]].qf32 = vadd([[V7]].qf32,[[V3]].qf32)
; CHECK:         qf32 = vmpy([[V8]].qf32,[[V9]].qf32)
label0:
  %v0 = fsub <32 x float> %a0, %a1
  %v1 = fsub <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %v0, %v1
  ret <32 x float> %v3
}

; Test qf32 = vmpy(qf32, qf32) when one is from adder, another from multiplier
define <32 x float> @mul_add_mul(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_add_mul:
; CHECK-DAG:     [[V7:v[0-9]+]].qf32 = vmpy(v0.sf,v1.sf)
; CHECK-DAG:     [[R0:r[0-9]+]] = ##2147483648
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = vadd(v0.sf,v2.sf)
; CHECK-DAG:     [[V4:v[0-9]+]] = vxor([[V4]],[[V4]])
; CHECK-DAG:     [[V5:v[0-9]+]] = vsplat([[R0]])
; CHECK-DAG:     [[V6:v[0-9]+]].qf32 = vmpy([[V4]].sf,[[V5]].sf)
; CHECK-DAG:     [[V8:v[0-9]+]].qf32 = vadd([[V6]].qf32,[[V3]].qf32)
; CHECK-DAG:     qf32 = vmpy([[V8]].qf32,[[V7]].qf32)
label0:
  %v1 = fadd <32 x float> %a0, %a2
  %v2 = fmul <32 x float> %a0, %a1
  %v3 = fmul <32 x float> %v1, %v2
  ret <32 x float> %v3
}

; Test qf32 = vmpy(qf32, qf32) when both are from multiplier
define <32 x float> @mul_mul_mul(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
label0:
; CHECK-LABEL: mul_mul_mul
; CHECK: [[V3:v[0-9]+]].qf32 = vmpy(v0.sf,v1.sf)
; CHECK: [[V4:v[0-9]+]].qf32 = vmpy(v0.sf,v2.sf)
; CHECK: qf32 = vmpy([[V4]].qf32,[[V3]].qf32)
  %v1 = fmul <32 x float> %a0, %a2
  %v2 = fmul <32 x float> %a0, %a1
  %v3 = fmul <32 x float> %v1, %v2
  ret <32 x float> %v3
}

attributes #0 = { nofree nosync nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-length128b,+hvx-qfloat,+hvxv79,+v79,-long-calls" "unsafe-fp-math"="true" }
