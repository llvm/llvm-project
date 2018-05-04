; RUN: llc -mtriple=thumbv7k-apple-watchos %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V7K
; RUN: llc -mtriple=thumbv7-linux-gnueabihf %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AAPCS
; RUN: llc -mtriple=thumbv7-apple-ios %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-APCS

define <32 x i8> @test_consume_arg([9 x double], <32 x i8> %vec) {
; CHECK-LABEL: test_consume_arg:

; CHECK-V7K: add r[[BASE:[0-9]+]], sp, #16
; CHECK-V7K: vld1.64 {d0, d1}, [r[[BASE]]:128]
; CHECK-V7K: add r[[BASE:[0-9]+]], sp, #32
; CHECK-V7K: vld1.64 {d2, d3}, [r[[BASE]]:128]

; CHECK-AAPCS: add r[[BASE:[0-9]+]], sp, #8
; CHECK-AAPCS: vld1.64 {d0, d1}, [r[[BASE]]]
; CHECK-AAPCS: add r[[BASE:[0-9]+]], sp, #24
; CHECK-AAPCS: vld1.64 {d2, d3}, [r[[BASE]]]

; CHECK-APCS: add r[[BASE:[0-9]+]], sp, #76
; CHECK-APCS: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[BASE]]]
; CHECK-APCS: add r[[BASE:[0-9]+]], sp, #60
; CHECK-APCS: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[BASE]]]

  ret <32 x i8> %vec
}

define void @test_produce_arg() {
; CHECK-LABEL: test_produce_arg:

; CHECK-V7K: add r[[BASE:[0-9]+]], sp, #32
; CHECK-V7K: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[BASE]]:128]
; CHECK-V7K: add r[[BASE:[0-9]+]], sp, #16
; CHECK-V7K: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[BASE]]:128]

; CHECK-AAPCS: add r[[BASE:[0-9]+]], sp, #24
; CHECK-AAPCS: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[BASE]]]
; CHECK-AAPCS: add r[[BASE:[0-9]+]], sp, #8
; CHECK-AAPCS: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[BASE]]]

; CHECK-APCS: add r[[BASE:[0-9]+]], sp, #60
; CHECK-APCS: vst1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[BASE]]]
; CHECK-APCS: mov r[[BASE:[0-9]+]], sp
; CHECK-APCS: str {{r[0-9]+}}, [r[[BASE]]], #76
; CHECK-APCS: vst1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[BASE]]]

call <32 x i8> @test_consume_arg([9 x double] undef, <32 x i8> zeroinitializer)
  ret void
}
