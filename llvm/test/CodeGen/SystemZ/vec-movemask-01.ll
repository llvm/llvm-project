; Test that a bitcast of a vector of i1 to an integer (a "movemask") uses
; VECTOR BIT PERMUTE on z14 and later.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s --check-prefix=POOL
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s --check-prefix=Z13

; The index vectors in the constant pools.  The low bit of each lane is
; selected, and a byte of 128 selects a zero bit.
; POOL: .LCPI0_0:
; POOL-NEXT: .byte 7
; POOL-NEXT: .byte 15
; POOL-NEXT: .byte 23
; POOL-NEXT: .byte 31
; POOL-NEXT: .byte 39
; POOL-NEXT: .byte 47
; POOL-NEXT: .byte 55
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 71
; POOL-NEXT: .byte 79
; POOL-NEXT: .byte 87
; POOL-NEXT: .byte 95
; POOL-NEXT: .byte 103
; POOL-NEXT: .byte 111
; POOL-NEXT: .byte 119
; POOL-NEXT: .byte 127
; POOL: .LCPI1_0:
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 15
; POOL-NEXT: .byte 31
; POOL-NEXT: .byte 47
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 79
; POOL-NEXT: .byte 95
; POOL-NEXT: .byte 111
; POOL-NEXT: .byte 127
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
; POOL-NEXT: .byte 31
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 95
; POOL-NEXT: .byte 127
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
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 127
; POOL: .LCPI5_0:
; POOL-NEXT: .byte 7
; POOL-NEXT: .byte 15
; POOL-NEXT: .byte 23
; POOL-NEXT: .byte 31
; POOL-NEXT: .byte 39
; POOL-NEXT: .byte 47
; POOL-NEXT: .byte 55
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 71
; POOL-NEXT: .byte 79
; POOL-NEXT: .byte 87
; POOL-NEXT: .byte 95
; POOL-NEXT: .byte 103
; POOL-NEXT: .byte 111
; POOL-NEXT: .byte 119
; POOL-NEXT: .byte 127
; POOL: .LCPI6_0:
; POOL-NEXT: .byte 7
; POOL-NEXT: .byte 15
; POOL-NEXT: .byte 23
; POOL-NEXT: .byte 31
; POOL-NEXT: .byte 39
; POOL-NEXT: .byte 47
; POOL-NEXT: .byte 55
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 71
; POOL-NEXT: .byte 79
; POOL-NEXT: .byte 87
; POOL-NEXT: .byte 95
; POOL-NEXT: .byte 103
; POOL-NEXT: .byte 111
; POOL-NEXT: .byte 119
; POOL-NEXT: .byte 127
; POOL: .LCPI7_0:
; POOL-NEXT: .byte 7
; POOL-NEXT: .byte 15
; POOL-NEXT: .byte 23
; POOL-NEXT: .byte 31
; POOL-NEXT: .byte 39
; POOL-NEXT: .byte 47
; POOL-NEXT: .byte 55
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 71
; POOL-NEXT: .byte 79
; POOL-NEXT: .byte 87
; POOL-NEXT: .byte 95
; POOL-NEXT: .byte 103
; POOL-NEXT: .byte 111
; POOL-NEXT: .byte 119
; POOL-NEXT: .byte 127
; POOL: .LCPI8_0:
; POOL-NEXT: .byte 7
; POOL-NEXT: .byte 15
; POOL-NEXT: .byte 23
; POOL-NEXT: .byte 31
; POOL-NEXT: .byte 39
; POOL-NEXT: .byte 47
; POOL-NEXT: .byte 55
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 71
; POOL-NEXT: .byte 79
; POOL-NEXT: .byte 87
; POOL-NEXT: .byte 95
; POOL-NEXT: .byte 103
; POOL-NEXT: .byte 111
; POOL-NEXT: .byte 119
; POOL-NEXT: .byte 127
; POOL: .LCPI9_0:
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 128
; POOL-NEXT: .byte 15
; POOL-NEXT: .byte 31
; POOL-NEXT: .byte 47
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 79
; POOL-NEXT: .byte 95
; POOL-NEXT: .byte 111
; POOL-NEXT: .byte 127
; POOL: .LCPI12_0:
; POOL-NEXT: .byte 7
; POOL-NEXT: .byte 15
; POOL-NEXT: .byte 23
; POOL-NEXT: .byte 31
; POOL-NEXT: .byte 39
; POOL-NEXT: .byte 47
; POOL-NEXT: .byte 55
; POOL-NEXT: .byte 63
; POOL-NEXT: .byte 71
; POOL-NEXT: .byte 79
; POOL-NEXT: .byte 87
; POOL-NEXT: .byte 95
; POOL-NEXT: .byte 103
; POOL-NEXT: .byte 111
; POOL-NEXT: .byte 119
; POOL-NEXT: .byte 127

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

; A truncation to i1 is done by VBPERM itself, so the byte vector is used
; directly without being widened to a mask first.
define i16 @f6(<16 x i8> %a) {
; CHECK-LABEL: f6:
; CHECK-NOT: ves
; CHECK: vbperm [[PERM:%v[0-9]+]], %v24, {{%v[0-9]+}}
; CHECK-NOT: ves
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
  %t = trunc <16 x i8> %a to <16 x i1>
  %mask = bitcast <16 x i1> %t to i16
  ret i16 %mask
}

; ... also if the truncated value is not a compare result.
define i16 @f7(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: f7:
; CHECK-NOT: ves
; CHECK: vsel [[SEL:%v[0-9]+]], %v24, %v26, {{%v[0-9]+}}
; CHECK-NOT: ves
; CHECK: vbperm [[PERM:%v[0-9]+]], [[SEL]], {{%v[0-9]+}}
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
  %cmp = icmp eq <16 x i8> %c, zeroinitializer
  %sel = select <16 x i1> %cmp, <16 x i8> %a, <16 x i8> %b
  %t = trunc <16 x i8> %sel to <16 x i1>
  %mask = bitcast <16 x i1> %t to i16
  ret i16 %mask
}

; ... and for a vector of i1 that is passed as an argument.
define i16 @f8(<16 x i1> %a) {
; CHECK-LABEL: f8:
; CHECK-NOT: ves
; CHECK: vbperm [[PERM:%v[0-9]+]], %v24, {{%v[0-9]+}}
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
  %mask = bitcast <16 x i1> %a to i16
  ret i16 %mask
}

; A source that is neither a truncation nor a compare.  Only the low bit of
; each lane is defined, and it is the one that is read.
define i16 @f9(<16 x i1> %a, <16 x i1> %b) {
; CHECK-LABEL: f9:
; CHECK-NOT: ves
; CHECK: vn [[AND:%v[0-9]+]], %v24, %v26
; CHECK: vbperm [[PERM:%v[0-9]+]], [[AND]], {{%v[0-9]+}}
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
  %and = and <16 x i1> %a, %b
  %mask = bitcast <16 x i1> %and to i16
  ret i16 %mask
}

; A short vector does not fill a vector register.  The lanes are widened to
; halfwords first (the upper bits of the halfwords are don't-care), and the
; low bit of each halfword is selected.
define i8 @f10(<8 x i8> %a) {
; CHECK-LABEL: f10:
; CHECK: vmrhb [[EXT:%v[0-9]+]], %v24, %v24
; CHECK: vbperm [[PERM:%v[0-9]+]], [[EXT]], {{%v[0-9]+}}
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
  %t = trunc <8 x i8> %a to <8 x i1>
  %mask = bitcast <8 x i1> %t to i8
  ret i8 %mask
}

; If only one or two lanes of the mask are used, extracting them individually
; is cheaper than permuting all of them.
define i16 @f11(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: f11:
; CHECK: vceqb [[CMP:%v[0-9]+]], %v24, %v26
; CHECK-NOT: vbperm
; CHECK: vlgvb
; CHECK-NOT: vbperm
; CHECK: br %r14
  %cmp = icmp eq <16 x i8> %a, %b
  %mask = bitcast <16 x i1> %cmp to i16
  %res = and i16 %mask, 32768
  ret i16 %res
}

define i16 @f12(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: f12:
; CHECK: vceqb [[CMP:%v[0-9]+]], %v24, %v26
; CHECK-NOT: vbperm
; CHECK: br %r14
  %cmp = icmp eq <16 x i8> %a, %b
  %mask = bitcast <16 x i1> %cmp to i16
  %res = and i16 %mask, 40960
  ret i16 %res
}

; ... but not for three lanes.
define i16 @f13(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: f13:
; CHECK: vbperm [[PERM:%v[0-9]+]], {{%v[0-9]+}}, {{%v[0-9]+}}
; CHECK: vlgvh %r2, [[PERM]], 3
; CHECK: br %r14
  %cmp = icmp eq <16 x i8> %a, %b
  %mask = bitcast <16 x i1> %cmp to i16
  %res = and i16 %mask, 57344
  ret i16 %res
}
