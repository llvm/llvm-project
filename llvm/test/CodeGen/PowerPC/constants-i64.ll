; RUN: llc -verify-machineinstrs -mcpu=ppc64 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define i64 @cn1() #0 {
entry:
  ret i64 281474976710655

; CHECK-LABEL: @cn1
; CHECK: lis [[REG1:[0-9]+]], -1
; CHECK: rldicr 3, [[REG1]], 48, 63
; CHECK: blr
}

; Function Attrs: nounwind readnone
define i64 @cnb() #0 {
entry:
  ret i64 281474976710575

; CHECK-LABEL: @cnb
; CHECK: lis [[REG1:[0-9]+]], -81
; CHECK: rldicr 3, [[REG1]], 48, 63
; CHECK: blr
}

; Function Attrs: nounwind readnone
define i64 @f2(i64 %x) #0 {
entry:
  ret i64 -68719476736

; CHECK-LABEL: @f2
; CHECK: li [[REG1:[0-9]+]], -1
; CHECK: sldi 3, [[REG1]], 36
; CHECK: blr
}

; Function Attrs: nounwind readnone
define i64 @f2a(i64 %x) #0 {
entry:
  ret i64 -361850994688

; CHECK-LABEL: @f2a
; CHECK: li [[REG1:[0-9]+]], -337
; CHECK: sldi 3, [[REG1]], 30
; CHECK: blr
}

; Function Attrs: nounwind readnone
define i64 @f2n(i64 %x) #0 {
entry:
  ret i64 68719476735

; CHECK-LABEL: @f2n
; CHECK: lis [[REG1:[0-9]+]], -4096
; CHECK: rldicr 3, [[REG1]], 36, 63
; CHECK: blr
}

; Function Attrs: nounwind readnone
define i64 @f3(i64 %x) #0 {
entry:
  ret i64 8589934591

; CHECK-LABEL: @f3
; CHECK: lis [[REG1:[0-9]+]], -32768
; CHECK: rldicr 3, [[REG1]], 33, 63
; CHECK: blr
}

; Function Attrs: nounwind readnone
define i64 @cn2n() #0 {
entry:
  ret i64 -1407374887747585

; CHECK-LABEL: @cn2n
; CHECK: lis [[REG1:[0-9]+]], -5121
; CHECK: ori [[REG2:[0-9]+]], [[REG1]], 65534
; CHECK: rldicr 3, [[REG2]], 22, 63
; CHECK: blr
}

define i64 @uint32_1() #0 {
entry:
  ret i64 3900000000

; CHECK-LABEL: @uint32_1
; CHECK: lis [[REG1:[0-9]+]], 232
; CHECK: ori [[REG2:[0-9]+]], [[REG1]], 30023
; CHECK: sldi 3, [[REG2]], 8
; CHECK: blr
}

define i32 @uint32_1_i32() #0 {
entry:
  ret i32 -394967296

; CHECK-LABEL: @uint32_1_i32
; CHECK: lis [[REG1:[0-9]+]], 232
; CHECK: ori [[REG2:[0-9]+]], [[REG1]], 30023
; CHECK: sldi 3, [[REG2]], 8
; CHECK: blr
}

define i64 @uint32_2() #0 {
entry:
  ret i64 4294967295

; CHECK-LABEL: @uint32_2
; CHECK: li [[REG1:[0-9]+]], 0
; CHECK: oris [[REG2:[0-9]+]], [[REG1]], 65535
; CHECK: ori 3, [[REG2]], 65535
; CHECK: blr
}

define i32 @uint32_2_i32() #0 {
entry:
  ret i32 -1

; CHECK-LABEL: @uint32_2_i32
; CHECK: li [[REG1:[0-9]+]], 0
; CHECK: oris [[REG2:[0-9]+]], [[REG1]], 65535
; CHECK: ori 3, [[REG2]], 65535
; CHECK: blr
}

define i64 @uint32_3() #0 {
entry:
  ret i64 2147483648

; CHECK-LABEL: @uint32_3
; CHECK: li [[REG1:[0-9]+]], 1
; CHECK: sldi 3, [[REG1]], 31
; CHECK: blr
}

define i64 @uint32_4() #0 {
entry:
  ret i64 124800000032

; CHECK-LABEL: @uint32_4
; CHECK: li [[REG1:[0-9]+]], 29
; CHECK: sldi [[REG2:[0-9]+]], [[REG1]], 32
; CHECK: oris [[REG3:[0-9]+]], [[REG2]], 3752
; CHECK: ori 3, [[REG3]], 57376
; CHECK: blr
}

define i64 @cn_ones_1() #0 {
entry:
  ret i64 10460594175

; CHECK-LABEL: @cn_ones_1
; CHECK: li [[REG1:[0-9]+]], 2
; CHECK: sldi [[REG2:[0-9]+]], [[REG1]], 32
; CHECK: oris [[REG3:[0-9]+]], [[REG2]], 28543
; CHECK: ori 3, [[REG3]], 65535
; CHECK: blr
}

define i64 @cn_ones_2() #0 {
entry:
  ret i64 10459119615

; CHECK-LABEL: @cn_ones_2
; CHECK: li [[REG1:[0-9]+]], 2
; CHECK: sldi [[REG2:[0-9]+]], [[REG1]], 32
; CHECK: oris [[REG3:[0-9]+]], [[REG2]], 28521
; CHECK: ori 3, [[REG3]], 32767
; CHECK: blr
}

attributes #0 = { nounwind readnone }
