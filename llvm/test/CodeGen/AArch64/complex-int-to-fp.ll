; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

; CHECK: autogen_SD19655
; CHECK: scvtf
; CHECK: ret
define void @autogen_SD19655(<2 x i64>* %addr, <2 x float>* %addrfloat) {
  %T = load <2 x i64>, <2 x i64>* %addr
  %F = sitofp <2 x i64> %T to <2 x float>
  store <2 x float> %F, <2 x float>* %addrfloat
  ret void
}

define <2 x double> @test_signed_v2i32_to_v2f64(<2 x i32> %v) nounwind readnone {
; CHECK-LABEL: test_signed_v2i32_to_v2f64:
; CHECK: sshll.2d [[VAL64:v[0-9]+]], v0, #0
; CHECK-NEXT: scvtf.2d v0, [[VAL64]]
; CHECK-NEXT: ret
  %conv = sitofp <2 x i32> %v to <2 x double>
  ret <2 x double> %conv
}

define <2 x double> @test_unsigned_v2i32_to_v2f64(<2 x i32> %v) nounwind readnone {
; CHECK-LABEL: test_unsigned_v2i32_to_v2f64
; CHECK: ushll.2d [[VAL64:v[0-9]+]], v0, #0
; CHECK-NEXT: ucvtf.2d v0, [[VAL64]]
; CHECK-NEXT: ret
  %conv = uitofp <2 x i32> %v to <2 x double>
  ret <2 x double> %conv
}

define <2 x double> @test_signed_v2i16_to_v2f64(<2 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_signed_v2i16_to_v2f64:
; CHECK: shl.2s [[TMP:v[0-9]+]], v0, #16
; CHECK: sshr.2s [[VAL32:v[0-9]+]], [[TMP]], #16
; CHECK: sshll.2d [[VAL64:v[0-9]+]], [[VAL32]], #0
; CHECK: scvtf.2d v0, [[VAL64]]

  %conv = sitofp <2 x i16> %v to <2 x double>
  ret <2 x double> %conv
}
define <2 x double> @test_unsigned_v2i16_to_v2f64(<2 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_unsigned_v2i16_to_v2f64
; CHECK: movi d[[MASK:[0-9]+]], #0x00ffff0000ffff
; CHECK: and.8b [[VAL32:v[0-9]+]], v0, v[[MASK]]
; CHECK: ushll.2d [[VAL64:v[0-9]+]], [[VAL32]], #0
; CHECK: ucvtf.2d v0, [[VAL64]]

  %conv = uitofp <2 x i16> %v to <2 x double>
  ret <2 x double> %conv
}

define <2 x double> @test_signed_v2i8_to_v2f64(<2 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_signed_v2i8_to_v2f64:
; CHECK: shl.2s [[TMP:v[0-9]+]], v0, #24
; CHECK: sshr.2s [[VAL32:v[0-9]+]], [[TMP]], #24
; CHECK: sshll.2d [[VAL64:v[0-9]+]], [[VAL32]], #0
; CHECK: scvtf.2d v0, [[VAL64]]

  %conv = sitofp <2 x i8> %v to <2 x double>
  ret <2 x double> %conv
}
define <2 x double> @test_unsigned_v2i8_to_v2f64(<2 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_unsigned_v2i8_to_v2f64
; CHECK: movi d[[MASK:[0-9]+]], #0x0000ff000000ff
; CHECK: and.8b [[VAL32:v[0-9]+]], v0, v[[MASK]]
; CHECK: ushll.2d [[VAL64:v[0-9]+]], [[VAL32]], #0
; CHECK: ucvtf.2d v0, [[VAL64]]

  %conv = uitofp <2 x i8> %v to <2 x double>
  ret <2 x double> %conv
}

define <2 x float> @test_signed_v2i64_to_v2f32(<2 x i64> %v) nounwind readnone {
; CHECK-LABEL: test_signed_v2i64_to_v2f32:
; CHECK: scvtf.2d [[VAL64:v[0-9]+]], v0
; CHECK: fcvtn v0.2s, [[VAL64]].2d

  %conv = sitofp <2 x i64> %v to <2 x float>
  ret <2 x float> %conv
}
define <2 x float> @test_unsigned_v2i64_to_v2f32(<2 x i64> %v) nounwind readnone {
; CHECK-LABEL: test_unsigned_v2i64_to_v2f32
; CHECK: ucvtf.2d [[VAL64:v[0-9]+]], v0
; CHECK: fcvtn v0.2s, [[VAL64]].2d

  %conv = uitofp <2 x i64> %v to <2 x float>
  ret <2 x float> %conv
}

define <2 x float> @test_signed_v2i16_to_v2f32(<2 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_signed_v2i16_to_v2f32:
; CHECK: shl.2s [[TMP:v[0-9]+]], v0, #16
; CHECK: sshr.2s [[VAL32:v[0-9]+]], [[TMP]], #16
; CHECK: scvtf.2s v0, [[VAL32]]

  %conv = sitofp <2 x i16> %v to <2 x float>
  ret <2 x float> %conv
}
define <2 x float> @test_unsigned_v2i16_to_v2f32(<2 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_unsigned_v2i16_to_v2f32
; CHECK: movi d[[MASK:[0-9]+]], #0x00ffff0000ffff
; CHECK: and.8b [[VAL32:v[0-9]+]], v0, v[[MASK]]
; CHECK: ucvtf.2s v0, [[VAL32]]

  %conv = uitofp <2 x i16> %v to <2 x float>
  ret <2 x float> %conv
}

define <2 x float> @test_signed_v2i8_to_v2f32(<2 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_signed_v2i8_to_v2f32:
; CHECK: shl.2s [[TMP:v[0-9]+]], v0, #24
; CHECK: sshr.2s [[VAL32:v[0-9]+]], [[TMP]], #24
; CHECK: scvtf.2s v0, [[VAL32]]

  %conv = sitofp <2 x i8> %v to <2 x float>
  ret <2 x float> %conv
}
define <2 x float> @test_unsigned_v2i8_to_v2f32(<2 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_unsigned_v2i8_to_v2f32
; CHECK: movi d[[MASK:[0-9]+]], #0x0000ff000000ff
; CHECK: and.8b [[VAL32:v[0-9]+]], v0, v[[MASK]]
; CHECK: ucvtf.2s v0, [[VAL32]]

  %conv = uitofp <2 x i8> %v to <2 x float>
  ret <2 x float> %conv
}

define <4 x float> @test_signed_v4i16_to_v4f32(<4 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_signed_v4i16_to_v4f32:
; CHECK: sshll.4s [[VAL32:v[0-9]+]], v0, #0
; CHECK: scvtf.4s v0, [[VAL32]]

  %conv = sitofp <4 x i16> %v to <4 x float>
  ret <4 x float> %conv
}

define <4 x float> @test_unsigned_v4i16_to_v4f32(<4 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_unsigned_v4i16_to_v4f32
; CHECK: ushll.4s [[VAL32:v[0-9]+]], v0, #0
; CHECK: ucvtf.4s v0, [[VAL32]]

  %conv = uitofp <4 x i16> %v to <4 x float>
  ret <4 x float> %conv
}

define <4 x float> @test_signed_v4i8_to_v4f32(<4 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_signed_v4i8_to_v4f32:
; CHECK: shl.4h [[TMP:v[0-9]+]], v0, #8
; CHECK: sshr.4h [[VAL16:v[0-9]+]], [[TMP]], #8
; CHECK: sshll.4s [[VAL32:v[0-9]+]], [[VAL16]], #0
; CHECK: scvtf.4s v0, [[VAL32]]

  %conv = sitofp <4 x i8> %v to <4 x float>
  ret <4 x float> %conv
}
define <4 x float> @test_unsigned_v4i8_to_v4f32(<4 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_unsigned_v4i8_to_v4f32
; CHECK: bic.4h v0, #255, lsl #8
; CHECK: ushll.4s [[VAL32:v[0-9]+]], v0, #0
; CHECK: ucvtf.4s v0, [[VAL32]]

  %conv = uitofp <4 x i8> %v to <4 x float>
  ret <4 x float> %conv
}
