; Test that a bitcast of a vector of i1 to an integer (a "movemask") uses
; VECTOR BIT PERMUTE on z14 and later.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s --check-prefix=POOL
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s --check-prefix=Z13

; The index vectors in the constant pools.  A byte of 128 selects a zero bit.
; POOL: .LCPI0_0:
; POOL-NEXT: .byte 0
; POOL-NEXT: .byte 8
; POOL-NEXT: .byte 16
; POOL-NEXT: .byte 24
; POOL-NEXT: .byte 32
; POOL-NEXT: .byte 40
; POOL-NEXT: .byte 48
; POOL-NEXT: .byte 56
; POOL-NEXT: .byte 64
; POOL-NEXT: .byte 72
; POOL-NEXT: .byte 80
; POOL-NEXT: .byte 88
; POOL-NEXT: .byte 96
; POOL-NEXT: .byte 104
; POOL-NEXT: .byte 112
; POOL-NEXT: .byte 120
; POOL: .LCPI1_0:
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 0
; POOL-NEXT: .byte 16
; POOL-NEXT: .byte 32
; POOL-NEXT: .byte 48
; POOL-NEXT: .byte 64
; POOL-NEXT: .byte 80
; POOL-NEXT: .byte 96
; POOL-NEXT: .byte 112
; POOL: .LCPI2_0:
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 0
; POOL-NEXT: .byte 32
; POOL-NEXT: .byte 64
; POOL-NEXT: .byte 96
; POOL: .LCPI3_0:
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 0
; POOL-NEXT: .byte 64

; Byte lanes.  Lane 0 ends up in the most significant bit of the result.
define i16 @f1(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: f1:
; CHECK: larl [[REG:%r[0-5]]], .LCPI0_0
; CHECK: vl [[IDX:%v[0-9]+]], 0([[REG]]), 3
; CHECK: vceqb [[CMP:%v[0-9]+]], %v24, %v26
; CHECK: vbperm [[PERM:%v[0-9]+]], [[CMP]], [[IDX]]
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
; Z13-LABEL: f1:
; Z13-NOT: vbperm
  %cmp = icmp eq <16 x i8> %a, %b
  %mask = bitcast <16 x i1> %cmp to i16
  ret i16 %mask
}

; Halfword lanes.
define i8 @f2(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: f2:
; CHECK: vceqh [[CMP:%v[0-9]+]], %v24, %v26
; CHECK: vbperm [[PERM:%v[0-9]+]], [[CMP]], {{%v[0-9]+}}
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
  %cmp = icmp eq <8 x i16> %a, %b
  %mask = bitcast <8 x i1> %cmp to i8
  ret i8 %mask
}

; Word lanes, using a floating-point compare.
define i4 @f3(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: f3:
; CHECK: vfchsb [[CMP:%v[0-9]+]], %v26, %v24
; CHECK: vbperm [[PERM:%v[0-9]+]], [[CMP]], {{%v[0-9]+}}
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
  %cmp = fcmp olt <4 x float> %a, %b
  %mask = bitcast <4 x i1> %cmp to i4
  ret i4 %mask
}

; Doubleword lanes.
define i2 @f4(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: f4:
; CHECK: vfchdb [[CMP:%v[0-9]+]], %v24, %v26
; CHECK: vbperm [[PERM:%v[0-9]+]], [[CMP]], {{%v[0-9]+}}
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
  %cmp = fcmp ogt <2 x double> %a, %b
  %mask = bitcast <2 x i1> %cmp to i2
  ret i2 %mask
}

; A reduction of the compare result must still use the condition code.
define i1 @f5(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: f5:
; CHECK: vceqbs
; CHECK-NOT: vbperm
; CHECK: br %r14
  %cmp = icmp eq <16 x i8> %a, %b
  %mask = bitcast <16 x i1> %cmp to i16
  %res = icmp ne i16 %mask, 0
  ret i1 %res
}
