; RUN: llc -O0 -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: fminimum_vec32f32
define float @fminimum_vec32f32(<32 x float> %vec) {
  %res = call float @llvm.vector.reduce.fminimum.v32f32(<32 x float> %vec)
  ret float %res
}
; CHECK: r{{[0-9]+}} = sfmin(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


; CHECK-LABEL: fminimum_vec2f16
define half @fminimum_vec2f16(<2 x half> %vec) {
  %res = call half @llvm.vector.reduce.fminimum.v2f16(<2 x half> %vec)
  ret half %res
}
; CHECK: call __extendhfsf2
; CHECK: r{{[0-9]+}} = sfmin(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


; CHECK-LABEL: fminimum_vec64f16
define half @fminimum_vec64f16(<64 x half> %vec) {
  %res = call half @llvm.vector.reduce.fminimum.v64f16(<64 x half> %vec)
  ret half %res
}
; CHECK: call __extendhfsf2
; CHECK: r{{[0-9]+}} = sfmin(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


; CHECK-LABEL: fminimum_float
define float @fminimum_float(float %a, float %b) {
  %res = call float @llvm.minimum.f32(float %a, float %b)
  ret float %res
}
; CHECK: r{{[0-9]+}} = sfmin(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


; CHECK-LABEL: fminimum_half
define half @fminimum_half(half %a, half %b) {
  %res = call half @llvm.minimum.f16(half %a, half %b)
  ret half %res
}
; CHECK: call __extendhfsf2
; CHECK: r{{[0-9]+}} = sfmin(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.uo(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})


declare float @llvm.vector.reduce.fminimum.v32f32(<32 x float> %vec) #0
declare half @llvm.vector.reduce.fminimum.v2f16(<2 x half> %vec) #0
declare half @llvm.vector.reduce.fminimum.v64f16(<64 x half> %vec) #0
declare float @llvm.minimum.f32(float %a, float %b) #0
declare half @llvm.minimum.f16(half %a, half %b) #0

attributes #0 = { nounwind "target-cpu"="hexagonv75" }
