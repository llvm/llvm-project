; RUN: llc -O0 -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: fmaximum_vec32f32
define float @fmaximum_vec32f32(<32 x float> %vec) {
  %res = call float @llvm.vector.reduce.fmaximum.v32f32(<32 x float> %vec)
  ret float %res
}
; CHECK: r{{[0-9]+}} = sfmax(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


; CHECK-LABEL: fmaximum_vec2f16
define half @fmaximum_vec2f16(<2 x half> %vec) {
  %res = call half @llvm.vector.reduce.fmaximum.v2f16(<2 x half> %vec)
  ret half %res
}
; CHECK: call __extendhfsf2
; CHECK: r{{[0-9]+}} = sfmax(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


; CHECK-LABEL: fmaximum_vec64f16
define half @fmaximum_vec64f16(<64 x half> %vec) {
  %res = call half @llvm.vector.reduce.fmaximum.v64f16(<64 x half> %vec)
  ret half %res
}
; CHECK: call __extendhfsf2
; CHECK: r{{[0-9]+}} = sfmax(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


; CHECK-LABEL: fmaximum_float
define float @fmaximum_float(float %a, float %b) {
  %res = call float @llvm.maximum.f32(float %a, float %b)
  ret float %res
}
; CHECK: r{{[0-9]+}} = sfmax(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


; CHECK-LABEL: fmaximum_half
define half @fmaximum_half(half %a, half %b) {
  %res = call half @llvm.maximum.f16(half %a, half %b)
  ret half %res
}
; CHECK: call __extendhfsf2
; CHECK: r{{[0-9]+}} = sfmax(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


declare float @llvm.vector.reduce.fmaximum.v32f32(<32 x float> %vec) #0
declare half @llvm.vector.reduce.fmaximum.v2f16(<2 x half> %vec) #0
declare half @llvm.vector.reduce.fmaximum.v64f16(<64 x half> %vec) #0
declare float @llvm.maximum.f32(float %a, float %b) #0
declare half @llvm.maximum.f16(half %a, half %b) #0

attributes #0 = { nounwind "target-cpu"="hexagonv75" }
