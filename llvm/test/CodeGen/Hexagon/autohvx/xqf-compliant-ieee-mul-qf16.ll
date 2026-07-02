; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-rem-conv=true \
; RUN: -enable-xqf-gen=true -hexagon-qfloat-mode=ieee -mattr=+hvxv79,+hvx-length128B < %s | FileCheck %s
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -force-hvx-float -enable-rem-conv=true \
; RUN: -enable-xqf-gen=true -hexagon-qfloat-mode=ieee -mattr=+hvxv81,+hvx-length128B < %s | FileCheck %s

; Test qf16 = vmpy(qf16 ,qf16) when both inputs are from vadd instruction
define <64 x half> @mul_add_3(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2) #0 {
; CHECK-LABEL: mul_add_3:
; CHECK-DAG:     [[V3:v[0-9]+]].qf16 = vadd(v0.hf,v2.hf)
; CHECK-DAG:     [[V4:v[0-9]+]].qf16 = vadd(v0.hf,v1.hf)
; CHECK-DAG:     [[V5:v[0-9]+]] = vxor([[V5]],[[V5]])
; CHECK-DAG:     [[V10:v[0-9]+:[0-9]+]].qf32 = vmpy([[V4]].qf16,[[V3]].qf16)
; CHECK-DAG:     [[V6:v[0-9]+]].hf = [[V10]].qf32
; CHECK-DAG:     qf16 = vsub([[V6]].hf,[[V5]].hf)
label0:
  %v0 = fadd <64 x half> %a0, %a1
  %v1 = fadd <64 x half> %a0, %a2
  %v3 = fmul <64 x half> %v0, %v1
  ret <64 x half> %v3
}

; Test qf32 = vmpy(qf16 ,qf16) when both inputs are from vadd and vmul instruction
define <64 x half> @mul_add_mul(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2) #0 {
; CHECK-LABEL: mul_add_mul:
; CHECK-DAG:     [[V32:v[0-9]+:[0-9]+]].qf32 = vmpy(v0.hf,v2.hf)
; CHECK-DAG:     [[V4:v[0-9]+]].qf16 = vadd(v0.hf,v1.hf)
; CHECK-DAG:     [[V5:v[0-9]+]] = vxor([[V5]],[[V5]])
; CHECK-DAG:     [[V3:v[0-9]+]].hf = [[V32]].qf32
; CHECK-DAG:     [[V6:v[0-9]+]].qf16 = vsub([[V3]].hf,[[V5]].hf)
; CHECK-DAG:     [[V10:v[0-9]+:[0-9]+]].qf32 = vmpy([[V4]].qf16,[[V6]].qf16)
; CHECK-DAG:     [[V7:v[0-9]+]].hf = [[V10]].qf32
; CHECK:         qf16 = vsub([[V7]].hf,[[V5]].hf)
label0:
  %v0 = fadd <64 x half> %a0, %a1
  %v1 = fmul <64 x half> %a0, %a2
  %v3 = fmul <64 x half> %v0, %v1
  ret <64 x half> %v3
}

; Test qf16 = vmpy(sf ,sf)
define <64 x half> @mul_add_0(<64 x half> %a0, <64 x half> %a1) #0 {
; CHECK-LABEL: mul_add_0:
; CHECK-DAG:     [[V10:v[0-9]+:[0-9]+]].qf32 = vmpy(v0.hf,v1.hf)
; CHECK-DAG:     [[V2:v[0-9]+]] = vxor([[V2]],[[V2]])
; CHECK-DAG:     [[V3:v[0-9]+]].hf = [[V10]].qf32
; CHECK:         qf16 = vsub([[V3]].hf,[[V2]].hf)
label0:
  %v3 = fmul <64 x half> %a0, %a1
  ret <64 x half> %v3
}

; Test qf16 = vmpy(qf16 ,qf16) when first input is from vadd instruction
define <64 x half> @mul_add_1(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2) #0 {
; CHECK-LABEL: mul_add_1:
; CHECK-DAG:     [[V3:v[0-9]+]].qf16 = vadd(v0.hf,v1.hf)
; CHECK-DAG:     [[V4:v[0-9]+]] = vxor([[V4]],[[V4]])
; CHECK-DAG:     [[V10:v[0-9]+:[0-9]+]].qf32 = vmpy([[V3]].qf16,v2.hf)
; CHECK-DAG:     [[V5:v[0-9]+]].hf = [[V10]].qf32
; CHECK:         qf16 = vsub([[V5]].hf,[[V4]].hf)
label0:
  %v0 = fadd <64 x half> %a0, %a1
  %v3 = fmul <64 x half> %v0, %a2
  ret <64 x half> %v3
}

; Test qf16 = vmpy(qf16 ,qf16) when second input is from vadd instruction
define <64 x half> @mul_add_2(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2) #0 {
; CHECK-LABEL: mul_add_2:
; CHECK-DAG:     [[V54:v[0-9]+:[0-9]+]].qf32 = vmpy(v0.hf,v2.hf)
; CHECK-DAG:     [[V29:v[0-9]+:[0-9]+]].qf32 = vmpy(v1.hf,v2.hf)
; CHECK-DAG:     [[V30:v[0-9]+]] = vxor([[V30]],[[V30]])
; CHECK-DAG:     [[V3:v[0-9]+]].hf = [[V54]].qf32
; CHECK-DAG:     [[V31:v[0-9]+]].hf = [[V29]].qf32
; CHECK-DAG:     [[V6:v[0-9]+]].qf16 = vsub([[V3]].hf,[[V30]].hf)
; CHECK-DAG:     [[V7:v[0-9]+]].qf16 = vsub([[V31]].hf,[[V30]].hf)
; CHECK-DAG:     [[V32:v[0-9]+:[0-9]+]].qf32 = vmpy([[V6]].qf16,[[V7]].qf16)
; CHECK-DAG:     [[V8:v[0-9]+]].hf = [[V32]].qf32
; CHECK:         qf16 = vsub([[V8]].hf,[[V30]].hf)
label0:
  %v1 = fmul <64 x half> %a0, %a2
  %v2 = fmul <64 x half> %a1, %a2
  %v3 = fmul <64 x half> %v1, %v2
  ret <64 x half> %v3
}

attributes #0 = { nofree nosync nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-length128b,+hvx-qfloat,-long-calls" "unsafe-fp-math"="true" }
