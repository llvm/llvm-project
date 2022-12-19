; RUN: llc -mtriple=aarch64-linux-gnuabi -O0 -fast-isel < %s | FileCheck %s
; RUN: llc -mtriple=aarch64_be-linux-gnuabi -O0 -fast-isel < %s | FileCheck %s --check-prefix=CHECK-BE

define void @test_bitcast_v8f16_to_v4f32(<8 x half> %a) {
; CHECK-LABEL: test_bitcast_v8f16_to_v4f32:
; CHECK-NOT: st1

; CHECK-BE-LABEL: test_bitcast_v8f16_to_v4f32:
; CHECK-BE: st1

  %x = alloca <4 x float>, align 16
  %y = bitcast <8 x half> %a to <4 x float>
  store <4 x float> %y, ptr %x, align 16
  ret void
}

define void @test_bitcast_v8f16_to_v2f64(<8 x half> %a) {
; CHECK-LABEL: test_bitcast_v8f16_to_v2f64:
; CHECK-NOT: st1

; CHECK-BE-LABEL: test_bitcast_v8f16_to_v2f64:
; CHECK-BE: st1

  %x = alloca <2 x double>, align 16
  %y = bitcast <8 x half> %a to <2 x double>
  store <2 x double> %y, ptr %x, align 16
  ret void
}

define void @test_bitcast_v8f16_to_fp128(<8 x half> %a) {
; CHECK-LABEL: test_bitcast_v8f16_to_fp128:
; CHECK-NOT: st1

; CHECK-BE-LABEL: test_bitcast_v8f16_to_fp128:
; CHECK-BE: st1

  %x = alloca fp128, align 16
  %y = bitcast <8 x half> %a to fp128
  store fp128 %y, ptr %x, align 16
  ret void
}

define void @test_bitcast_v4f16_to_v2f32(<4 x half> %a) {
; CHECK-LABEL: test_bitcast_v4f16_to_v2f32:
; CHECK-NOT: st1

; CHECK-BE-LABEL: test_bitcast_v4f16_to_v2f32:
; CHECK-BE: st1

  %x = alloca <2 x float>, align 8
  %y = bitcast <4 x half> %a to <2 x float>
  store <2 x float> %y, ptr %x, align 8
  ret void
}

define void @test_bitcast_v4f16_to_v1f64(<4 x half> %a) {
; CHECK-LABEL: test_bitcast_v4f16_to_v1f64:
; CHECK-NOT: st1

; CHECK-BE-LABEL: test_bitcast_v4f16_to_v1f64:
; CHECK-BE: st1

  %x = alloca <1 x double>, align 8
  %y = bitcast <4 x half> %a to <1 x double>
  store <1 x double> %y, ptr %x, align 8
  ret void
}
