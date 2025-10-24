; Check if int_to_fp is lowered correctly for v32i32 --> v32f32

; RUN: llc -march=hexagon -stop-after=hexagon-isel -o - %s | FileCheck %s

; CHECK: [[R2:%[0-9]+]]:hvxvr = V6_vd0
; CHECK-NEXT: [[R3:%[0-9]+]]:hvxvr = V6_vavguw %0, [[R2]]
; CHECK-NEXT: [[R4:%[0-9]+]]:hvxvr = V6_vconv_sf_w [[R3]]
; CHECK-NEXT: [[R5:%[0-9]+]]:hvxvr = V6_vsubw %0, [[R3]]
; CHECK-NEXT: [[R6:%[0-9]+]]:hvxvr = V6_vconv_sf_w killed [[R5]]
; CHECK-NEXT: [[R7:%[0-9]+]]:hvxvr = V6_vadd_sf_sf killed [[R4]], killed [[R6]]
; CHECK-NEXT: [[R8:%[0-9]+]]:hvxvr = V6_vavguw %1, [[R2]]
; CHECK-NEXT: [[R9:%[0-9]+]]:hvxvr = V6_vconv_sf_w [[R8]]
; CHECK-NEXT: [[R10:%[0-9]+]]:hvxvr = V6_vsubw %1, [[R8]]
; CHECK-NEXT: [[R11:%[0-9]+]]:hvxvr = V6_vconv_sf_w killed [[R10]]
; CHECK-NEXT: [[R12:%[0-9]+]]:hvxvr = V6_vadd_sf_sf killed [[R9]], killed [[R11]]
; CHECK-NEXT: V6_vmpy_qf32_sf killed [[R7]], killed [[R12]]


target triple = "hexagon"
define <32 x float> @uitofp(<32 x i32> %int0, <32 x i32> %int1) #0
{
   %fp0 = uitofp <32 x i32> %int0 to <32 x float>
   %fp1 = uitofp <32 x i32> %int1 to <32 x float>
   %out = fmul <32 x float> %fp0, %fp1
   ret <32 x float> %out
}
attributes #0 = { nounwind readnone "target-cpu"="hexagonv79" "target-features"="+hvxv79,+hvx-length128b" }
