; Tests strict-ieee mode for XQFloat for multiplication 32-bit

; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=strict-ieee -mattr=+hvxv81,+hvx-length128B < %s | FileCheck %s -check-prefix=CHECK

; Test qf32 = vmpy(sf, sf)
; Normalization of inputs
define <32 x float> @mul_add_0(<32 x float> %a0, <32 x float> %a1) #0 {
; CHECK-LABEL: mul_add_0
; CHECK-DAG:     [[V2:v[0-9]+]].qf32 = v0.sf
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = v1.sf
; CHECK:         qf32 = vmpy([[V2]].qf32,[[V3]].qf32)
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
; CHECK-DAG:     [[V5:v[0-9]+]].sf = [[V3]].qf32
; CHECK-DAG:     [[V6:v[0-9]+]].sf = [[V4]].qf32
; CHECK-DAG:     [[V7:v[0-9]+]].qf32 = [[V5]].sf
; CHECK-DAG:     [[V8:v[0-9]+]].qf32 = [[V6]].sf
; CHECK:         qf32 = vmpy([[V8]].qf32,[[V7]].qf32)
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
; CHECK-DAG:     [[V5:v[0-9]+]].sf = [[V3]].qf32
; CHECK-DAG:     [[V6:v[0-9]+]].sf = [[V4]].qf32
; CHECK-DAG:     [[V7:v[0-9]+]].qf32 = [[V5]].sf
; CHECK-DAG:     [[V8:v[0-9]+]].qf32 = [[V6]].sf
; CHECK:         qf32 = vmpy([[V8]].qf32,[[V7]].qf32)
label0:
  %v0 = fsub <32 x float> %a0, %a1
  %v1 = fsub <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %v0, %v1
  ret <32 x float> %v3
}

; Test qf32 = vmpy(qf32 ,qf32) when inputs are from vadd and vmul respectively
; The inputs to both multiplications are converted to IEEE and normalized.
define <32 x float> @mul_add_mul(<32 x float> %a0, <32 x float> %a1, <32 x float> %a2) #0 {
; CHECK-LABEL: mul_add_mul:
; CHECK-DAG:     [[V3:v[0-9]+]].qf32 = v2.sf
; CHECK-DAG:     [[V4:v[0-9]+]].qf32 = v0.sf
; CHECK-DAG:     [[V5:v[0-9]+]].qf32 = vadd(v0.sf,v1.sf)
; CHECK-DAG:     [[V6:v[0-9]+]].qf32 = vmpy([[V4]].qf32,[[V3]].qf32)
; CHECK-DAG:     [[V7:v[0-9]+]].sf = [[V5]].qf32
; CHECK-DAG:     [[V8:v[0-9]+]].sf = [[V6]].qf32
; CHECK-DAG:     [[V9:v[0-9]+]].qf32 = [[V7]].sf
; CHECK-DAG:     [[V10:v[0-9]+]].qf32 = [[V8]].sf
; CHECK:     qf32 = vmpy([[V9]].qf32,[[V10]].qf32)
label0:
  %v0 = fadd <32 x float> %a0, %a1
  %v1 = fmul <32 x float> %a0, %a2
  %v3 = fmul <32 x float> %v0, %v1
  ret <32 x float> %v3
}

; Tests when input to vmul is a mul qf32 and a sf type, and the output stored to memory
define i32 @mul_intrinsic(ptr %output) {
; CHECK-LABEL: mul_intrinsic:
; CHECK: [[V1:v[0-9]+]].qf32 = vmpy(v0.qf32,v0.qf32)
; CHECK: [[V2:v[0-9]+]].sf = [[V1]].qf32
; CHECK: [[V3:v[0-9]+]].qf32 = [[V2]].sf
; CHECK: [[V4:v[0-9]+]].qf32 = vmpy([[V3]].qf32,v0.qf32)
; CHECK: [[V5:v[0-9]+]].sf = [[V4]].qf32
; CHECK: vmemu{{.*}} = [[V5]]
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %0, <32 x i32> zeroinitializer)
  store <32 x i32> %1, ptr %output, align 4
  ret i32 0
}

declare <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32>, <32 x i32>) #0
uselistorder ptr @llvm.hexagon.V6.vmpy.qf32.128B, { 1, 0 }

attributes #0 = { nofree nosync nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+hvx-qfloat,-long-calls" "unsafe-fp-math"="true" }
