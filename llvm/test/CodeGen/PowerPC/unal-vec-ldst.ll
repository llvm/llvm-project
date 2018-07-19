; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define <16 x i8> @test_l_v16i8(<16 x i8>* %p) #0 {
entry:
  %r = load <16 x i8>, <16 x i8>* %p, align 1
  ret <16 x i8> %r

; CHECK-LABEL: @test_l_v16i8
; CHECK-DAG: li [[REG1:[0-9]+]], 15
; CHECK-DAG: lvsl [[REG2:[0-9]+]], 0, 3
; CHECK-DAG: lvx [[REG3:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: lvx [[REG4:[0-9]+]], 0, 3
; CHECK: vperm 2, [[REG4]], [[REG3]], [[REG2]]
; CHECK: blr
}

define <32 x i8> @test_l_v32i8(<32 x i8>* %p) #0 {
entry:
  %r = load <32 x i8>, <32 x i8>* %p, align 1
  ret <32 x i8> %r

; CHECK-LABEL: @test_l_v32i8
; CHECK-DAG: li [[REG1:[0-9]+]], 31
; CHECK-DAG: li [[REG2:[0-9]+]], 16
; CHECK-DAG: lvsl [[REG3:[0-9]+]], 0, 3
; CHECK-DAG: lvx [[REG4:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: lvx [[REG5:[0-9]+]], 3, [[REG2]]
; CHECK-DAG: lvx [[REG6:[0-9]+]], 0, 3
; CHECK-DAG: vperm 3, [[REG5]], [[REG4]], [[REG3]]
; CHECK-DAG: vperm 2, [[REG6]], [[REG5]], [[REG3]]
; CHECK: blr
}

define <8 x i16> @test_l_v8i16(<8 x i16>* %p) #0 {
entry:
  %r = load <8 x i16>, <8 x i16>* %p, align 2
  ret <8 x i16> %r

; CHECK-LABEL: @test_l_v8i16
; CHECK-DAG: li [[REG1:[0-9]+]], 15
; CHECK-DAG: lvsl [[REG2:[0-9]+]], 0, 3
; CHECK-DAG: lvx [[REG3:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: lvx [[REG4:[0-9]+]], 0, 3
; CHECK: vperm 2, [[REG4]], [[REG3]], [[REG2]]
; CHECK: blr
}

define <16 x i16> @test_l_v16i16(<16 x i16>* %p) #0 {
entry:
  %r = load <16 x i16>, <16 x i16>* %p, align 2
  ret <16 x i16> %r

; CHECK-LABEL: @test_l_v16i16
; CHECK-DAG: li [[REG1:[0-9]+]], 31
; CHECK-DAG: li [[REG2:[0-9]+]], 16
; CHECK-DAG: lvsl [[REG3:[0-9]+]], 0, 3
; CHECK-DAG: lvx [[REG4:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: lvx [[REG5:[0-9]+]], 3, [[REG2]]
; CHECK-DAG: lvx [[REG6:[0-9]+]], 0, 3
; CHECK-DAG: vperm 3, [[REG5]], [[REG4]], [[REG3]]
; CHECK-DAG: vperm 2, [[REG6]], [[REG5]], [[REG3]]
; CHECK: blr
}

define <4 x i32> @test_l_v4i32(<4 x i32>* %p) #0 {
entry:
  %r = load <4 x i32>, <4 x i32>* %p, align 4
  ret <4 x i32> %r

; CHECK-LABEL: @test_l_v4i32
; CHECK-DAG: li [[REG1:[0-9]+]], 15
; CHECK-DAG: lvsl [[REG2:[0-9]+]], 0, 3
; CHECK-DAG: lvx [[REG3:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: lvx [[REG4:[0-9]+]], 0, 3
; CHECK: vperm 2, [[REG4]], [[REG3]], [[REG2]]
; CHECK: blr
}

define <8 x i32> @test_l_v8i32(<8 x i32>* %p) #0 {
entry:
  %r = load <8 x i32>, <8 x i32>* %p, align 4
  ret <8 x i32> %r

; CHECK-LABEL: @test_l_v8i32
; CHECK-DAG: li [[REG1:[0-9]+]], 31
; CHECK-DAG: li [[REG2:[0-9]+]], 16
; CHECK-DAG: lvsl [[REG3:[0-9]+]], 0, 3
; CHECK-DAG: lvx [[REG4:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: lvx [[REG5:[0-9]+]], 3, [[REG2]]
; CHECK-DAG: lvx [[REG6:[0-9]+]], 0, 3
; CHECK-DAG: vperm 3, [[REG5]], [[REG4]], [[REG3]]
; CHECK-DAG: vperm 2, [[REG6]], [[REG5]], [[REG3]]
; CHECK: blr
}

define <2 x i64> @test_l_v2i64(<2 x i64>* %p) #0 {
entry:
  %r = load <2 x i64>, <2 x i64>* %p, align 8
  ret <2 x i64> %r

; CHECK-LABEL: @test_l_v2i64
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr
}

define <4 x i64> @test_l_v4i64(<4 x i64>* %p) #0 {
entry:
  %r = load <4 x i64>, <4 x i64>* %p, align 8
  ret <4 x i64> %r

; CHECK-LABEL: @test_l_v4i64
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: lxvd2x 34, 0, 3
; CHECK-DAG: lxvd2x 35, 3, [[REG1]]
; CHECK: blr
}

define <4 x float> @test_l_v4float(<4 x float>* %p) #0 {
entry:
  %r = load <4 x float>, <4 x float>* %p, align 4
  ret <4 x float> %r

; CHECK-LABEL: @test_l_v4float
; CHECK-DAG: li [[REG1:[0-9]+]], 15
; CHECK-DAG: lvsl [[REG2:[0-9]+]], 0, 3
; CHECK-DAG: lvx [[REG3:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: lvx [[REG4:[0-9]+]], 0, 3
; CHECK: vperm 2, [[REG4]], [[REG3]], [[REG2]]
; CHECK: blr
}

define <8 x float> @test_l_v8float(<8 x float>* %p) #0 {
entry:
  %r = load <8 x float>, <8 x float>* %p, align 4
  ret <8 x float> %r

; CHECK-LABEL: @test_l_v8float
; CHECK-DAG: li [[REG1:[0-9]+]], 31
; CHECK-DAG: li [[REG2:[0-9]+]], 16
; CHECK-DAG: lvsl [[REG3:[0-9]+]], 0, 3
; CHECK-DAG: lvx [[REG4:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: lvx [[REG5:[0-9]+]], 3, [[REG2]]
; CHECK-DAG: lvx [[REG6:[0-9]+]], 0, 3
; CHECK-DAG: vperm 3, [[REG5]], [[REG4]], [[REG3]]
; CHECK-DAG: vperm 2, [[REG6]], [[REG5]], [[REG3]]
; CHECK: blr
}

define <2 x double> @test_l_v2double(<2 x double>* %p) #0 {
entry:
  %r = load <2 x double>, <2 x double>* %p, align 8
  ret <2 x double> %r

; CHECK-LABEL: @test_l_v2double
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr
}

define <4 x double> @test_l_v4double(<4 x double>* %p) #0 {
entry:
  %r = load <4 x double>, <4 x double>* %p, align 8
  ret <4 x double> %r

; CHECK-LABEL: @test_l_v4double
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: lxvd2x 34, 0, 3
; CHECK-DAG: lxvd2x 35, 3, [[REG1]]
; CHECK: blr
}

define <16 x i8> @test_l_p8v16i8(<16 x i8>* %p) #2 {
entry:
  %r = load <16 x i8>, <16 x i8>* %p, align 1
  ret <16 x i8> %r

; CHECK-LABEL: @test_l_p8v16i8
; CHECK: lxvw4x 34, 0, 3
; CHECK: blr
}

define <32 x i8> @test_l_p8v32i8(<32 x i8>* %p) #2 {
entry:
  %r = load <32 x i8>, <32 x i8>* %p, align 1
  ret <32 x i8> %r

; CHECK-LABEL: @test_l_p8v32i8
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: lxvw4x 34, 0, 3
; CHECK-DAG: lxvw4x 35, 3, [[REG1]]
; CHECK: blr
}

define <8 x i16> @test_l_p8v8i16(<8 x i16>* %p) #2 {
entry:
  %r = load <8 x i16>, <8 x i16>* %p, align 2
  ret <8 x i16> %r

; CHECK-LABEL: @test_l_p8v8i16
; CHECK: lxvw4x 34, 0, 3
; CHECK: blr
}

define <16 x i16> @test_l_p8v16i16(<16 x i16>* %p) #2 {
entry:
  %r = load <16 x i16>, <16 x i16>* %p, align 2
  ret <16 x i16> %r

; CHECK-LABEL: @test_l_p8v16i16
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: lxvw4x 34, 0, 3
; CHECK-DAG: lxvw4x 35, 3, [[REG1]]
; CHECK: blr
}

define <4 x i32> @test_l_p8v4i32(<4 x i32>* %p) #2 {
entry:
  %r = load <4 x i32>, <4 x i32>* %p, align 4
  ret <4 x i32> %r

; CHECK-LABEL: @test_l_p8v4i32
; CHECK: lxvw4x 34, 0, 3
; CHECK: blr
}

define <8 x i32> @test_l_p8v8i32(<8 x i32>* %p) #2 {
entry:
  %r = load <8 x i32>, <8 x i32>* %p, align 4
  ret <8 x i32> %r

; CHECK-LABEL: @test_l_p8v8i32
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: lxvw4x 34, 0, 3
; CHECK-DAG: lxvw4x 35, 3, [[REG1]]
; CHECK: blr
}

define <2 x i64> @test_l_p8v2i64(<2 x i64>* %p) #2 {
entry:
  %r = load <2 x i64>, <2 x i64>* %p, align 8
  ret <2 x i64> %r

; CHECK-LABEL: @test_l_p8v2i64
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr
}

define <4 x i64> @test_l_p8v4i64(<4 x i64>* %p) #2 {
entry:
  %r = load <4 x i64>, <4 x i64>* %p, align 8
  ret <4 x i64> %r

; CHECK-LABEL: @test_l_p8v4i64
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: lxvd2x 34, 0, 3
; CHECK-DAG: lxvd2x 35, 3, [[REG1]]
; CHECK: blr
}

define <4 x float> @test_l_p8v4float(<4 x float>* %p) #2 {
entry:
  %r = load <4 x float>, <4 x float>* %p, align 4
  ret <4 x float> %r

; CHECK-LABEL: @test_l_p8v4float
; CHECK: lxvw4x 34, 0, 3
; CHECK: blr
}

define <8 x float> @test_l_p8v8float(<8 x float>* %p) #2 {
entry:
  %r = load <8 x float>, <8 x float>* %p, align 4
  ret <8 x float> %r

; CHECK-LABEL: @test_l_p8v8float
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: lxvw4x 34, 0, 3
; CHECK-DAG: lxvw4x 35, 3, [[REG1]]
; CHECK: blr
}

define <2 x double> @test_l_p8v2double(<2 x double>* %p) #2 {
entry:
  %r = load <2 x double>, <2 x double>* %p, align 8
  ret <2 x double> %r

; CHECK-LABEL: @test_l_p8v2double
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr
}

define <4 x double> @test_l_p8v4double(<4 x double>* %p) #2 {
entry:
  %r = load <4 x double>, <4 x double>* %p, align 8
  ret <4 x double> %r

; CHECK-LABEL: @test_l_p8v4double
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: lxvd2x 34, 0, 3
; CHECK-DAG: lxvd2x 35, 3, [[REG1]]
; CHECK: blr
}

define <4 x float> @test_l_qv4float(<4 x float>* %p) #1 {
entry:
  %r = load <4 x float>, <4 x float>* %p, align 4
  ret <4 x float> %r

; CHECK-LABEL: @test_l_qv4float
; CHECK-DAG: li [[REG1:[0-9]+]], 15
; CHECK-DAG: qvlpclsx 0, 0, 3
; CHECK-DAG: qvlfsx [[REG2:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: qvlfsx [[REG3:[0-9]+]], 0, 3
; CHECK: qvfperm 1, [[REG3]], [[REG2]], 0
; CHECK: blr
}

define <8 x float> @test_l_qv8float(<8 x float>* %p) #1 {
entry:
  %r = load <8 x float>, <8 x float>* %p, align 4
  ret <8 x float> %r

; CHECK-LABEL: @test_l_qv8float
; CHECK-DAG: li [[REG1:[0-9]+]], 31
; CHECK-DAG: li [[REG2:[0-9]+]], 16
; CHECK-DAG: qvlfsx [[REG3:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: qvlfsx [[REG4:[0-9]+]], 3, [[REG2]]
; CHECK-DAG: qvlpclsx [[REG5:[0-5]+]], 0, 3
; CHECK-DAG: qvlfsx [[REG6:[0-9]+]], 0, 3
; CHECK-DAG: qvfperm 2, [[REG4]], [[REG3]], [[REG5]]
; CHECK-DAG: qvfperm 1, [[REG6]], [[REG4]], [[REG5]]
; CHECK: blr
}

define <4 x double> @test_l_qv4double(<4 x double>* %p) #1 {
entry:
  %r = load <4 x double>, <4 x double>* %p, align 8
  ret <4 x double> %r

; CHECK-LABEL: @test_l_qv4double
; CHECK-DAG: li [[REG1:[0-9]+]], 31
; CHECK-DAG: qvlpcldx 0, 0, 3
; CHECK-DAG: qvlfdx [[REG2:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: qvlfdx [[REG3:[0-9]+]], 0, 3
; CHECK: qvfperm 1, [[REG3]], [[REG2]], 0
; CHECK: blr
}

define <8 x double> @test_l_qv8double(<8 x double>* %p) #1 {
entry:
  %r = load <8 x double>, <8 x double>* %p, align 8
  ret <8 x double> %r

; CHECK-LABEL: @test_l_qv8double
; CHECK-DAG: li [[REG1:[0-9]+]], 63
; CHECK-DAG: li [[REG2:[0-9]+]], 32
; CHECK-DAG: qvlfdx [[REG3:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: qvlfdx [[REG4:[0-9]+]], 3, [[REG2]]
; CHECK-DAG: qvlpcldx [[REG5:[0-5]+]], 0, 3
; CHECK-DAG: qvlfdx [[REG6:[0-9]+]], 0, 3
; CHECK-DAG: qvfperm 2, [[REG4]], [[REG3]], [[REG5]]
; CHECK-DAG: qvfperm 1, [[REG6]], [[REG4]], [[REG5]]
; CHECK: blr
}

define void @test_s_v16i8(<16 x i8>* %p, <16 x i8> %v) #0 {
entry:
  store <16 x i8> %v, <16 x i8>* %p, align 1
  ret void

; CHECK-LABEL: @test_s_v16i8
; CHECK: stxvw4x 34, 0, 3
; CHECK: blr
}

define void @test_s_v32i8(<32 x i8>* %p, <32 x i8> %v) #0 {
entry:
  store <32 x i8> %v, <32 x i8>* %p, align 1
  ret void

; CHECK-LABEL: @test_s_v32i8
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: stxvw4x 35, 3, [[REG1]]
; CHECK-DAG: stxvw4x 34, 0, 3
; CHECK: blr
}

define void @test_s_v8i16(<8 x i16>* %p, <8 x i16> %v) #0 {
entry:
  store <8 x i16> %v, <8 x i16>* %p, align 2
  ret void

; CHECK-LABEL: @test_s_v8i16
; CHECK: stxvw4x 34, 0, 3
; CHECK: blr
}

define void @test_s_v16i16(<16 x i16>* %p, <16 x i16> %v) #0 {
entry:
  store <16 x i16> %v, <16 x i16>* %p, align 2
  ret void

; CHECK-LABEL: @test_s_v16i16
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: stxvw4x 35, 3, [[REG1]]
; CHECK-DAG: stxvw4x 34, 0, 3
; CHECK: blr
}

define void @test_s_v4i32(<4 x i32>* %p, <4 x i32> %v) #0 {
entry:
  store <4 x i32> %v, <4 x i32>* %p, align 4
  ret void

; CHECK-LABEL: @test_s_v4i32
; CHECK: stxvw4x 34, 0, 3
; CHECK: blr
}

define void @test_s_v8i32(<8 x i32>* %p, <8 x i32> %v) #0 {
entry:
  store <8 x i32> %v, <8 x i32>* %p, align 4
  ret void

; CHECK-LABEL: @test_s_v8i32
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: stxvw4x 35, 3, [[REG1]]
; CHECK-DAG: stxvw4x 34, 0, 3
; CHECK: blr
}

define void @test_s_v2i64(<2 x i64>* %p, <2 x i64> %v) #0 {
entry:
  store <2 x i64> %v, <2 x i64>* %p, align 8
  ret void

; CHECK-LABEL: @test_s_v2i64
; CHECK: stxvd2x 34, 0, 3
; CHECK: blr
}

define void @test_s_v4i64(<4 x i64>* %p, <4 x i64> %v) #0 {
entry:
  store <4 x i64> %v, <4 x i64>* %p, align 8
  ret void

; CHECK-LABEL: @test_s_v4i64
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: stxvd2x 35, 3, [[REG1]]
; CHECK-DAG: stxvd2x 34, 0, 3
; CHECK: blr
}

define void @test_s_v4float(<4 x float>* %p, <4 x float> %v) #0 {
entry:
  store <4 x float> %v, <4 x float>* %p, align 4
  ret void

; CHECK-LABEL: @test_s_v4float
; CHECK: stxvw4x 34, 0, 3
; CHECK: blr
}

define void @test_s_v8float(<8 x float>* %p, <8 x float> %v) #0 {
entry:
  store <8 x float> %v, <8 x float>* %p, align 4
  ret void

; CHECK-LABEL: @test_s_v8float
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: stxvw4x 35, 3, [[REG1]]
; CHECK-DAG: stxvw4x 34, 0, 3
; CHECK: blr
}

define void @test_s_v2double(<2 x double>* %p, <2 x double> %v) #0 {
entry:
  store <2 x double> %v, <2 x double>* %p, align 8
  ret void

; CHECK-LABEL: @test_s_v2double
; CHECK: stxvd2x 34, 0, 3
; CHECK: blr
}

define void @test_s_v4double(<4 x double>* %p, <4 x double> %v) #0 {
entry:
  store <4 x double> %v, <4 x double>* %p, align 8
  ret void

; CHECK-LABEL: @test_s_v4double
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK-DAG: stxvd2x 35, 3, [[REG1]]
; CHECK-DAG: stxvd2x 34, 0, 3
; CHECK: blr
}

define void @test_s_qv4float(<4 x float>* %p, <4 x float> %v) #1 {
entry:
  store <4 x float> %v, <4 x float>* %p, align 4
  ret void

; CHECK-LABEL: @test_s_qv4float
; CHECK-DAG: qvesplati [[REG1:[0-9]+]], 1, 3
; CHECK-DAG: qvesplati [[REG2:[0-9]+]], 1, 2
; CHECK-DAG: qvesplati [[REG3:[0-9]+]], 1, 1
; CHECK-DAG: stfs 1, 0(3)
; CHECK-DAG: stfs [[REG1]], 12(3)
; CHECK-DAG: stfs [[REG2]], 8(3)
; CHECK-DAG: stfs [[REG3]], 4(3)
; CHECK: blr
}

define void @test_s_qv8float(<8 x float>* %p, <8 x float> %v) #1 {
entry:
  store <8 x float> %v, <8 x float>* %p, align 4
  ret void

; CHECK-LABEL: @test_s_qv8float
; CHECK-DAG: qvesplati [[REG1:[0-9]+]], 2, 3
; CHECK-DAG: qvesplati [[REG2:[0-9]+]], 2, 2
; CHECK-DAG: qvesplati [[REG3:[0-9]+]], 2, 1
; CHECK-DAG: qvesplati [[REG4:[0-9]+]], 1, 3
; CHECK-DAG: qvesplati [[REG5:[0-9]+]], 1, 2
; CHECK-DAG: qvesplati [[REG6:[0-9]+]], 1, 1
; CHECK-DAG: stfs 2, 16(3)
; CHECK-DAG: stfs 1, 0(3)
; CHECK-DAG: stfs [[REG1]], 28(3)
; CHECK-DAG: stfs [[REG2]], 24(3)
; CHECK-DAG: stfs [[REG3]], 20(3)
; CHECK-DAG: stfs [[REG4]], 12(3)
; CHECK-DAG: stfs [[REG5]], 8(3)
; CHECK-DAG: stfs [[REG6]], 4(3)
; CHECK: blr
}

define void @test_s_qv4double(<4 x double>* %p, <4 x double> %v) #1 {
entry:
  store <4 x double> %v, <4 x double>* %p, align 8
  ret void

; CHECK-LABEL: @test_s_qv4double
; CHECK-DAG: qvesplati [[REG1:[0-9]+]], 1, 3
; CHECK-DAG: qvesplati [[REG2:[0-9]+]], 1, 2
; CHECK-DAG: qvesplati [[REG3:[0-9]+]], 1, 1
; CHECK-DAG: stfd 1, 0(3)
; CHECK-DAG: stfd [[REG1]], 24(3)
; CHECK-DAG: stfd [[REG2]], 16(3)
; CHECK-DAG: stfd [[REG3]], 8(3)
; CHECK: blr
}

define void @test_s_qv8double(<8 x double>* %p, <8 x double> %v) #1 {
entry:
  store <8 x double> %v, <8 x double>* %p, align 8
  ret void

; CHECK-LABEL: @test_s_qv8double
; CHECK-DAG: qvesplati [[REG1:[0-9]+]], 2, 3
; CHECK-DAG: qvesplati [[REG2:[0-9]+]], 2, 2
; CHECK-DAG: qvesplati [[REG3:[0-9]+]], 2, 1
; CHECK-DAG: qvesplati [[REG4:[0-9]+]], 1, 3
; CHECK-DAG: qvesplati [[REG5:[0-9]+]], 1, 2
; CHECK-DAG: qvesplati [[REG6:[0-9]+]], 1, 1
; CHECK-DAG: stfd 2, 32(3)
; CHECK-DAG: stfd 1, 0(3)
; CHECK-DAG: stfd [[REG1]], 56(3)
; CHECK-DAG: stfd [[REG2]], 48(3)
; CHECK-DAG: stfd [[REG3]], 40(3)
; CHECK-DAG: stfd [[REG4]], 24(3)
; CHECK-DAG: stfd [[REG5]], 16(3)
; CHECK-DAG: stfd [[REG6]], 8(3)
; CHECK: blr
}

attributes #0 = { nounwind "target-cpu"="pwr7" }
attributes #1 = { nounwind "target-cpu"="a2q" }
attributes #2 = { nounwind "target-cpu"="pwr8" }

