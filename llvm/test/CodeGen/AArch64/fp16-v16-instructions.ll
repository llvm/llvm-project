; RUN: llc < %s -mtriple=aarch64-none-eabi | FileCheck %s


define <16 x half> @sitofp_i32(<16 x i32> %a) #0 {
; CHECK-LABEL: sitofp_i32:
; CHECK-DAG: scvtf [[S0:v[0-9]+\.4s]], v0.4s
; CHECK-DAG: scvtf [[S1:v[0-9]+\.4s]], v1.4s
; CHECK-DAG: scvtf [[S2:v[0-9]+\.4s]], v2.4s
; CHECK-DAG: scvtf [[S3:v[0-9]+\.4s]], v3.4s
; CHECK-DAG: fcvtn v0.4h, [[S0]]
; CHECK-DAG: fcvtn v1.4h, [[S2]]
; CHECK-DAG: v[[R1:[0-9]+]].4h, [[S1]]
; CHECK-DAG: v[[R3:[0-9]+]].4h, [[S3]]
; CHECK-DAG: mov v0.d[1], v[[R1]].d[0]
; CHECK-DAG: mov v1.d[1], v[[R3]].d[0]

  %1 = sitofp <16 x i32> %a to <16 x half>
  ret <16 x half> %1
}


define <16 x half> @sitofp_i64(<16 x i64> %a) #0 {
; CHECK-LABEL: sitofp_i64:
; CHECK-DAG: scvtf [[D0:v[0-9]+\.2d]], v0.2d
; CHECK-DAG: scvtf [[D1:v[0-9]+\.2d]], v1.2d
; CHECK-DAG: scvtf [[D2:v[0-9]+\.2d]], v2.2d
; CHECK-DAG: scvtf [[D3:v[0-9]+\.2d]], v3.2d
; CHECK-DAG: scvtf [[D4:v[0-9]+\.2d]], v4.2d
; CHECK-DAG: scvtf [[D5:v[0-9]+\.2d]], v5.2d
; CHECK-DAG: scvtf [[D6:v[0-9]+\.2d]], v6.2d
; CHECK-DAG: scvtf [[D7:v[0-9]+\.2d]], v7.2d

; CHECK-DAG: fcvtn [[S0:v[0-9]+]].2s, [[D0]]
; CHECK-DAG: fcvtn [[S1:v[0-9]+]].2s, [[D2]]
; CHECK-DAG: fcvtn [[S2:v[0-9]+]].2s, [[D4]]
; CHECK-DAG: fcvtn [[S3:v[0-9]+]].2s, [[D6]]

; CHECK-DAG: fcvtn2 [[S0]].4s, [[D1]]
; CHECK-DAG: fcvtn2 [[S1]].4s, [[D3]]
; CHECK-DAG: fcvtn2 [[S2]].4s, [[D5]]
; CHECK-DAG: fcvtn2 [[S3]].4s, [[D7]]

; CHECK-DAG: fcvtn v0.4h, [[S0]].4s
; CHECK-DAG: fcvtn v1.4h, [[S2]].4s
; CHECK-DAG: fcvtn v[[R1:[0-9]+]].4h, [[S1]].4s
; CHECK-DAG: fcvtn v[[R3:[0-9]+]].4h, [[S3]].4s
; CHECK-DAG: mov v0.d[1], v[[R1]].d[0]
; CHECK-DAG: mov v1.d[1], v[[R3]].d[0]

  %1 = sitofp <16 x i64> %a to <16 x half>
  ret <16 x half> %1
}


define <16 x half> @uitofp_i32(<16 x i32> %a) #0 {
; CHECK-LABEL: uitofp_i32:
; CHECK-DAG: ucvtf [[S0:v[0-9]+\.4s]], v0.4s
; CHECK-DAG: ucvtf [[S1:v[0-9]+\.4s]], v1.4s
; CHECK-DAG: ucvtf [[S2:v[0-9]+\.4s]], v2.4s
; CHECK-DAG: ucvtf [[S3:v[0-9]+\.4s]], v3.4s
; CHECK-DAG: fcvtn v0.4h, [[S0]]
; CHECK-DAG: fcvtn v1.4h, [[S2]]
; CHECK-DAG: v[[R1:[0-9]+]].4h, [[S1]]
; CHECK-DAG: v[[R3:[0-9]+]].4h, [[S3]]
; CHECK-DAG: mov v0.d[1], v[[R1]].d[0]
; CHECK-DAG: mov v1.d[1], v[[R3]].d[0]

  %1 = uitofp <16 x i32> %a to <16 x half>
  ret <16 x half> %1
}


define <16 x half> @uitofp_i64(<16 x i64> %a) #0 {
; CHECK-LABEL: uitofp_i64:
; CHECK-DAG: ucvtf [[D0:v[0-9]+\.2d]], v0.2d
; CHECK-DAG: ucvtf [[D1:v[0-9]+\.2d]], v1.2d
; CHECK-DAG: ucvtf [[D2:v[0-9]+\.2d]], v2.2d
; CHECK-DAG: ucvtf [[D3:v[0-9]+\.2d]], v3.2d
; CHECK-DAG: ucvtf [[D4:v[0-9]+\.2d]], v4.2d
; CHECK-DAG: ucvtf [[D5:v[0-9]+\.2d]], v5.2d
; CHECK-DAG: ucvtf [[D6:v[0-9]+\.2d]], v6.2d
; CHECK-DAG: ucvtf [[D7:v[0-9]+\.2d]], v7.2d

; CHECK-DAG: fcvtn [[S0:v[0-9]+]].2s, [[D0]]
; CHECK-DAG: fcvtn [[S1:v[0-9]+]].2s, [[D2]]
; CHECK-DAG: fcvtn [[S2:v[0-9]+]].2s, [[D4]]
; CHECK-DAG: fcvtn [[S3:v[0-9]+]].2s, [[D6]]

; CHECK-DAG: fcvtn2 [[S0]].4s, [[D1]]
; CHECK-DAG: fcvtn2 [[S1]].4s, [[D3]]
; CHECK-DAG: fcvtn2 [[S2]].4s, [[D5]]
; CHECK-DAG: fcvtn2 [[S3]].4s, [[D7]]

; CHECK-DAG: fcvtn v0.4h, [[S0]].4s
; CHECK-DAG: fcvtn v1.4h, [[S2]].4s
; CHECK-DAG: fcvtn v[[R1:[0-9]+]].4h, [[S1]].4s
; CHECK-DAG: fcvtn v[[R3:[0-9]+]].4h, [[S3]].4s
; CHECK-DAG: mov v0.d[1], v[[R1]].d[0]
; CHECK-DAG: mov v1.d[1], v[[R3]].d[0]

  %1 = uitofp <16 x i64> %a to <16 x half>
  ret <16 x half> %1
}

attributes #0 = { nounwind }
