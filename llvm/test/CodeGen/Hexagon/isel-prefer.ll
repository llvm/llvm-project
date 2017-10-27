; RUN: llc -march=hexagon < %s | FileCheck %s

@data1 = external global [2 x [31 x i8]], align 8
@data2 = external global [2 x [91 x i8]], align 8

; CHECK-LABEL: Prefer_M4_or_andn:
; CHECK: r2 |= and(r0,~r1)
define i32 @Prefer_M4_or_andn(i32 %a0, i32 %a1, i32 %a2) #0 {
b3:
  %v4 = xor i32 %a1, -1
  %v5 = shl i32 %a2, 5
  %v6 = and i32 %a0, %v4
  %v7 = or i32 %v6, %v5
  ret i32 %v7
}

; CHECK-LABEL: Prefer_M4_mpyri_addi:
; CHECK: add(##data1,mpyi(r0,#31))
define i32 @Prefer_M4_mpyri_addi(i32 %a0) #0 {
b1:
  %v2 = getelementptr inbounds [2 x [31 x i8]], [2 x [31 x i8]]* @data1, i32 0, i32 %a0
  %v3 = ptrtoint [31 x i8]* %v2 to i32
  ret i32 %v3
}

; CHECK-LABEL: Prefer_M4_mpyrr_addi:
; CHECK: add(##data2,mpyi(r0,r1))
define i32 @Prefer_M4_mpyrr_addi(i32 %a0) #0 {
b1:
  %v2 = getelementptr inbounds [2 x [91 x i8]], [2 x [91 x i8]]* @data2, i32 0, i32 %a0
  %v3 = ptrtoint [91 x i8]* %v2 to i32
  ret i32 %v3
}

; CHECK-LABEL: Prefer_S2_tstbit_r:
; CHECK: p0 = tstbit(r0,r1)
define i32 @Prefer_S2_tstbit_r(i32 %a0, i32 %a1) #0 {
b2:
  %v3 = shl i32 1, %a1
  %v4 = and i32 %a0, %v3
  %v5 = icmp ne i32 %v4, 0
  %v6 = zext i1 %v5 to i32
  ret i32 %v6
}

; CHECK-LABEL: Prefer_S2_ntstbit_r:
; CHECK: p0 = !tstbit(r0,r1)
define i32 @Prefer_S2_ntstbit_r(i32 %a0, i32 %a1) #0 {
b2:
  %v3 = shl i32 1, %a1
  %v4 = and i32 %a0, %v3
  %v5 = icmp eq i32 %v4, 0
  %v6 = zext i1 %v5 to i32
  ret i32 %v6
}

attributes #0 = { nounwind readnone }
