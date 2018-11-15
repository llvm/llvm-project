; RUN: llc -verify-machineinstrs -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s -mtriple=powerpc64-linux-gnu -mcpu=pwr8 -mattr=+vsx | FileCheck %s -check-prefix=CHECK-VSX
; RUN: llc -verify-machineinstrs -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s -mtriple=powerpc64-linux-gnu -mcpu=pwr8 -mattr=-vsx | FileCheck %s -check-prefix=CHECK-NOVSX
; RUN: llc -verify-machineinstrs -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s -mtriple=powerpc64le-linux-gnu -mcpu=pwr8 -mattr=+vsx | FileCheck %s -check-prefix=CHECK-VSX
; RUN: llc -verify-machineinstrs -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s -mtriple=powerpc64le-linux-gnu -mcpu=pwr8 -mattr=-vsx | FileCheck %s -check-prefix=CHECK-NOVSX

define <4 x float> @test1(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d) {
entry:
  %m = fcmp oeq <4 x float> %c, %d
  %v = select <4 x i1> %m, <4 x float> %a, <4 x float> %b
  ret <4 x float> %v
}
; CHECK-VSX-LABLE: test1
; CHECK-VSX: xvcmpeqsp [[REG1:(vs|v)[0-9]+]], v4, v5
; CHECK-VSX: xxsel v2, v3, v2, [[REG1]]
; CHECK-VSX: blr

; CHECK-NOVSX-LABLE: test1
; CHECK-NOVSX: vcmpeqfp v[[REG1:[0-9]+]], v4, v5
; CHECK-NOVSX: vsel v2, v3, v2, v[[REG1]]
; CHECK-NOVSX: blr

define <2 x double> @test2(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %d) {
entry:
  %m = fcmp oeq <2 x double> %c, %d
  %v = select <2 x i1> %m, <2 x double> %a, <2 x double> %b
  ret <2 x double> %v
}
; CHECK-VSX-LABLE: test2
; CHECK-VSX: xvcmpeqdp [[REG1:(vs|v)[0-9]+]], v4, v5
; CHECK-VSX: xxsel v2, v3, v2, [[REG1]]
; CHECK-VSX: blr

; CHECK-NOVSX-LABLE: test2
; CHECK-NOVSX: fcmp
; CHECK-NOVSX: fcmp
; CHECK-NOVSX: blr

define <16 x i8> @test3(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, <16 x i8> %d) {
entry:
  %m = icmp eq <16 x i8> %c, %d
  %v = select <16 x i1> %m, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %v
}
; CHECK-VSX-LABLE: test3
; CHECK-VSX: vcmpequb v[[REG1:[0-9]+]], v4, v5
; CHECK-VSX: xxsel v2, v3, v2, v[[REG1]]
; CHECK-VSX: blr

; CHECK-NOVSX-LABLE: test3
; CHECK-NOVSX: vcmpequb v[[REG1:[0-9]+]], v4, v5
; CHECK-NOVSX: vsel v2, v3, v2, v[[REG1]]
; CHECK-NOVSX: blr

define <8 x i16> @test4(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c, <8 x i16> %d) {
entry:
  %m = icmp eq <8 x i16> %c, %d
  %v = select <8 x i1> %m, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %v
}
; CHECK-VSX-LABLE: test4
; CHECK-VSX: vcmpequh v[[REG1:[0-9]+]], v4, v5
; CHECK-VSX: xxsel v2, v3, v2, v[[REG1]]
; CHECK-VSX: blr

; CHECK-NOVSX-LABLE: test4
; CHECK-NOVSX: vcmpequh v[[REG1:[0-9]+]], v4, v5
; CHECK-NOVSX: vsel v2, v3, v2, v[[REG1]]
; CHECK-NOVSX: blr

define <4 x i32> @test5(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c, <4 x i32> %d) {
entry:
  %m = icmp eq <4 x i32> %c, %d
  %v = select <4 x i1> %m, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %v
}
; CHECK-VSX-LABLE: test5
; CHECK-VSX: vcmpequw v[[REG1:[0-9]+]], v4, v5
; CHECK-VSX: xxsel v2, v3, v2, v[[REG1]]
; CHECK-VSX: blr

; CHECK-NOVSX-LABLE: test5
; CHECK-NOVSX: vcmpequw v[[REG1:[0-9]+]], v4, v5
; CHECK-NOVSX: vsel v2, v3, v2, v[[REG1]]
; CHECK-NOVSX: blr

define <2 x i64> @test6(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c, <2 x i64> %d) {
entry:
  %m = icmp eq <2 x i64> %c, %d
  %v = select <2 x i1> %m, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %v
}
; CHECK-VSX-LABLE: test6
; CHECK-VSX: vcmpequd v[[REG1:[0-9]+]], v4, v5
; CHECK-VSX: xxsel v2, v3, v2, v[[REG1]]
; CHECK-VSX: blr

; CHECK-NOVSX-LABLE: test6
; CHECK-NOVSX: vcmpequd v[[REG1:[0-9]+]], v4, v5
; CHECK-NOVSX: vsel v2, v3, v2, v[[REG1]]
; CHECK-NOVSX: blr
