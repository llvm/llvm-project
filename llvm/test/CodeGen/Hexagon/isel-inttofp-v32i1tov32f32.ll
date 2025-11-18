; Tests lowering of v32i1 to v32f32

; RUN: llc -march=hexagon -mattr=+hvxv79,+hvx-length128b,+hvx-ieee-fp \
; RUN: -stop-after=hexagon-isel %s -o - | FileCheck %s

define <32 x float> @uitofp_i1(<32 x i16> %in0, <32 x i16> %in1) #0 {
; CHECK: name:            uitofp_i1
; CHECK: [[R0:%[0-9]+]]:hvxvr = V6_lvsplatw killed %{{[0-9]+}}
; CHECK-NEXT: [[R1:%[0-9]+]]:intregs = A2_tfrsi 1
; CHECK-NEXT: [[R2:%[0-9]+]]:hvxvr = V6_lvsplatw [[R1]]
; CHECK-NEXT: [[R3:%[0-9]+]]:hvxqr = V6_vandvrt [[R2]], [[R1]]
; CHECK-NEXT: [[R4:%[0-9]+]]:hvxvr = V6_vprefixqw killed [[R3]]
; CHECK-NEXT: [[R5:%[0-9]+]]:hvxvr = V6_vsubw killed [[R4]], [[R2]]
; CHECK-NEXT: [[R6:%[0-9]+]]:hvxvr = V6_vlsrwv killed [[R0]], killed [[R5]]
; CHECK-NEXT: [[R7:%[0-9]+]]:hvxvr = V6_vand killed [[R6]], [[R2]]
; CHECK-NEXT: [[R8:%[0-9]+]]:hvxvr = V6_vconv_sf_w killed [[R7]]
; CHECK-NEXT: hvxvr = V6_vadd_sf_sf [[R8]], [[R8]]
  %q1 = icmp eq <32 x i16> %in0, %in1
  %fp0 = uitofp <32 x i1> %q1 to <32 x float>
  %out = fadd <32 x float> %fp0, %fp0
  ret <32 x float> %out
}

define <32 x float> @sitofp_i1(<32 x i16> %in0, <32 x i16> %in1) #0 {
; CHECK: name:            sitofp_i1
; CHECK: [[R0:%[0-9]+]]:hvxvr = V6_lvsplatw killed %{{[0-9]+}}
; CHECK-NEXT: [[R1:%[0-9]+]]:intregs = A2_tfrsi 1
; CHECK-NEXT: [[R2:%[0-9]+]]:hvxvr = V6_lvsplatw [[R1]]
; CHECK-NEXT: [[R3:%[0-9]+]]:hvxqr = V6_vandvrt [[R2]], [[R1]]
; CHECK-NEXT: [[R4:%[0-9]+]]:hvxvr = V6_vprefixqw killed [[R3]]
; CHECK-NEXT: [[R5:%[0-9]+]]:hvxvr = V6_vsubw killed [[R4]], [[R2]]
; CHECK-NEXT: [[R6:%[0-9]+]]:hvxvr = V6_vlsrwv killed [[R0]], killed [[R5]]
; CHECK-NEXT: [[R7:%[0-9]+]]:hvxvr = V6_vand killed [[R6]], [[R2]]
; CHECK-NEXT: [[R8:%[0-9]+]]:hvxvr = V6_vconv_sf_w killed [[R7]]
; CHECK-NEXT: hvxvr = V6_vadd_sf_sf [[R8]], [[R8]]
  %q1 = icmp eq <32 x i16> %in0, %in1
  %fp0 = sitofp <32 x i1> %q1 to <32 x float>
  %out = fadd <32 x float> %fp0, %fp0
  ret <32 x float> %out
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv79" "target-features"="+hvxv79,+hvx-length128b" }
