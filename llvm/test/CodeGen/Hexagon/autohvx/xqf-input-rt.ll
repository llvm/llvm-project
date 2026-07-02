; Tests qf operations with Rt types along with hf/sf and qf16 operands
; for strict-ieee mode

; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true -hexagon-qfloat-mode=strict-ieee -mattr=+hvxv79,+hvx-length128B < %s | FileCheck %s --check-prefix=STRICT-IEEE
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true -hexagon-qfloat-mode=ieee -mattr=+hvxv79,+hvx-length128B < %s | FileCheck %s --check-prefix=COMPLIANT-IEEE
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true -hexagon-qfloat-mode=lossy -mattr=+hvxv79,+hvx-length128B < %s | FileCheck %s --check-prefix=LOSSY-SUBNORMAL


; Tests qf16 = vmpy(hf, Rt32.hf)
define <32 x i32> @mul_hf_rt(<32 x i32> %a0, i32 %a1) {
; STRICT-IEEE-LABEL: mul_hf_rt
; STRICT-IEEE-DAG: [[V1:v[0-9]+]] = vsplat(r0)
; STRICT-IEEE-DAG: [[V2:v[0-9]+]] = vxor([[V2]],[[V2]])
; STRICT-IEEE: [[V10:v[0-31]+:[0-31]+]].qf32 = vmpy(v0.hf,[[V1]].hf)
; STRICT-IEEE: [[V3:v[0-9]+]].hf = [[V10]].qf32
; STRICT-IEEE: qf16 = vsub([[V3]].hf,[[V2]].hf)

; COMPLIANT-IEEE-LABEL: mul_hf_rt
; COMPLIANT-IEEE-DAG: [[V1:v[0-9]+]] = vsplat(r0)
; COMPLIANT-IEEE-DAG: [[V2:v[0-9]+]] = vxor([[V2]],[[V2]])
; COMPLIANT-IEEE: [[V10:v[0-31]+:[0-31]+]].qf32 = vmpy(v0.hf,[[V1]].hf)
; COMPLIANT-IEEE: [[V3:v[0-9]+]].hf = [[V10]].qf32
; COMPLIANT-IEEE: qf16 = vsub([[V3]].hf,[[V2]].hf)

; LOSSY-SUBNORMAL: qf16 = vmpy(v0.hf,r0.hf)

label0:
  %v0 = call <32 x i32> @llvm.hexagon.V6.vmpy.rt.hf.128B(<32 x i32> %a0, i32 %a1)
  ret <32 x i32> %v0
}


; Tests qf32 = vmpy(sf, Rt32.hf)
define <32 x i32> @mul_sf_rt(<32 x i32> %a0, i32 %a1) {
; STRICT-IEEE-LABEL: mul_sf_rt
; STRICT-IEEE-DAG: [[V2:v[0-9]+]] = vsplat(r0)
; STRICT-IEEE-DAG: [[R2:r[0-9]+]] = ##2147483648
; STRICT-IEEE-DAG: [[V1:v[0-9]+]] = vxor([[V1]],[[V1]])
; STRICT-IEEE-DAG: [[V3:v[0-9]+]] = vsplat([[R2]])
; STRICT-IEEE: [[V4:v[0-9]+]].qf32 = vmpy([[V1]].sf,[[V3]].sf)
; STRICT-IEEE: [[V5:v[0-9]+]].qf32 = vadd([[V4]].qf32,v0.sf)
; STRICT-IEEE: [[V6:v[0-9]+]].qf32 = vadd([[V4]].qf32,[[V2]].sf)
; STRICT-IEEE: qf32 = vmpy([[V5]].qf32,[[V6]].qf32)

; COMPLIANT-IEEE-LABEL: mul_sf_rt
; COMPLIANT-IEEE-DAG: [[V2:v[0-9]+]] = vsplat(r0)
; COMPLIANT-IEEE-DAG: [[R2:r[0-9]+]] = ##2147483648
; COMPLIANT-IEEE-DAG: [[V1:v[0-9]+]] = vxor([[V1]],[[V1]])
; COMPLIANT-IEEE-DAG: [[V3:v[0-9]+]] = vsplat([[R2]])
; COMPLIANT-IEEE: [[V4:v[0-9]+]].qf32 = vmpy([[V1]].sf,[[V3]].sf)
; COMPLIANT-IEEE: [[V5:v[0-9]+]].qf32 = vadd([[V4]].qf32,v0.sf)
; COMPLIANT-IEEE: [[V6:v[0-9]+]].qf32 = vadd([[V4]].qf32,[[V2]].sf)
; COMPLIANT-IEEE: qf32 = vmpy([[V5]].qf32,[[V6]].qf32)

; LOSSY-SUBNORMAL: qf32 = vmpy(v0.sf,r0.sf)

label0:
  %v0 = call <32 x i32> @llvm.hexagon.V6.vmpy.rt.sf.128B(<32 x i32> %a0, i32 %a1)
  ret <32 x i32> %v0
}

declare <32 x i32> @llvm.hexagon.V6.vmpy.rt.hf.128B(<32 x i32>, i32)
declare <32 x i32> @llvm.hexagon.V6.vmpy.rt.sf.128B(<32 x i32>, i32)
