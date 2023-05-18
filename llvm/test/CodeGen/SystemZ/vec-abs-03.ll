; Test v4i32 absolute.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test with slt.
define <4 x i32> @f1(<4 x i32> %val) {
; CHECK-LABEL: f1:
; CHECK: vlpf %v24, %v24
; CHECK: br %r14
  %cmp = icmp slt <4 x i32> %val, zeroinitializer
  %neg = sub <4 x i32> zeroinitializer, %val
  %ret = select <4 x i1> %cmp, <4 x i32> %neg, <4 x i32> %val
  ret <4 x i32> %ret
}

; Test with sle.
define <4 x i32> @f2(<4 x i32> %val) {
; CHECK-LABEL: f2:
; CHECK: vlpf %v24, %v24
; CHECK: br %r14
  %cmp = icmp sle <4 x i32> %val, zeroinitializer
  %neg = sub <4 x i32> zeroinitializer, %val
  %ret = select <4 x i1> %cmp, <4 x i32> %neg, <4 x i32> %val
  ret <4 x i32> %ret
}

; Test with sgt.
define <4 x i32> @f3(<4 x i32> %val) {
; CHECK-LABEL: f3:
; CHECK: vlpf %v24, %v24
; CHECK: br %r14
  %cmp = icmp sgt <4 x i32> %val, zeroinitializer
  %neg = sub <4 x i32> zeroinitializer, %val
  %ret = select <4 x i1> %cmp, <4 x i32> %val, <4 x i32> %neg
  ret <4 x i32> %ret
}

; Test with sge.
define <4 x i32> @f4(<4 x i32> %val) {
; CHECK-LABEL: f4:
; CHECK: vlpf %v24, %v24
; CHECK: br %r14
  %cmp = icmp sge <4 x i32> %val, zeroinitializer
  %neg = sub <4 x i32> zeroinitializer, %val
  %ret = select <4 x i1> %cmp, <4 x i32> %val, <4 x i32> %neg
  ret <4 x i32> %ret
}

; Test that negative absolute uses VLPF too.  There is no vector equivalent
; of LOAD NEGATIVE.
define <4 x i32> @f5(<4 x i32> %val) {
; CHECK-LABEL: f5:
; CHECK: vlpf [[REG:%v[0-9]+]], %v24
; CHECK: vlcf %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp slt <4 x i32> %val, zeroinitializer
  %neg = sub <4 x i32> zeroinitializer, %val
  %abs = select <4 x i1> %cmp, <4 x i32> %neg, <4 x i32> %val
  %ret = sub <4 x i32> zeroinitializer, %abs
  ret <4 x i32> %ret
}

; Try another form of negative absolute (slt version).
define <4 x i32> @f6(<4 x i32> %val) {
; CHECK-LABEL: f6:
; CHECK: vlpf [[REG:%v[0-9]+]], %v24
; CHECK: vlcf %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp slt <4 x i32> %val, zeroinitializer
  %neg = sub <4 x i32> zeroinitializer, %val
  %ret = select <4 x i1> %cmp, <4 x i32> %val, <4 x i32> %neg
  ret <4 x i32> %ret
}

; Test with sle.
define <4 x i32> @f7(<4 x i32> %val) {
; CHECK-LABEL: f7:
; CHECK: vlpf [[REG:%v[0-9]+]], %v24
; CHECK: vlcf %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp sle <4 x i32> %val, zeroinitializer
  %neg = sub <4 x i32> zeroinitializer, %val
  %ret = select <4 x i1> %cmp, <4 x i32> %val, <4 x i32> %neg
  ret <4 x i32> %ret
}

; Test with sgt.
define <4 x i32> @f8(<4 x i32> %val) {
; CHECK-LABEL: f8:
; CHECK: vlpf [[REG:%v[0-9]+]], %v24
; CHECK: vlcf %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp sgt <4 x i32> %val, zeroinitializer
  %neg = sub <4 x i32> zeroinitializer, %val
  %ret = select <4 x i1> %cmp, <4 x i32> %neg, <4 x i32> %val
  ret <4 x i32> %ret
}

; Test with sge.
define <4 x i32> @f9(<4 x i32> %val) {
; CHECK-LABEL: f9:
; CHECK: vlpf [[REG:%v[0-9]+]], %v24
; CHECK: vlcf %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp sge <4 x i32> %val, zeroinitializer
  %neg = sub <4 x i32> zeroinitializer, %val
  %ret = select <4 x i1> %cmp, <4 x i32> %neg, <4 x i32> %val
  ret <4 x i32> %ret
}

; Test with an SRA-based boolean vector.
define <4 x i32> @f10(<4 x i32> %val) {
; CHECK-LABEL: f10:
; CHECK: vlpf %v24, %v24
; CHECK: br %r14
  %shr = ashr <4 x i32> %val, <i32 31, i32 31, i32 31, i32 31>
  %neg = sub <4 x i32> zeroinitializer, %val
  %and1 = and <4 x i32> %shr, %neg
  %not = xor <4 x i32> %shr, <i32 -1, i32 -1, i32 -1, i32 -1>
  %and2 = and <4 x i32> %not, %val
  %ret = or <4 x i32> %and1, %and2
  ret <4 x i32> %ret
}

; ...and again in reverse
define <4 x i32> @f11(<4 x i32> %val) {
; CHECK-LABEL: f11:
; CHECK: vlpf [[REG:%v[0-9]+]], %v24
; CHECK: vlcf %v24, [[REG]]
; CHECK: br %r14
  %shr = ashr <4 x i32> %val, <i32 31, i32 31, i32 31, i32 31>
  %and1 = and <4 x i32> %shr, %val
  %not = xor <4 x i32> %shr, <i32 -1, i32 -1, i32 -1, i32 -1>
  %neg = sub <4 x i32> zeroinitializer, %val
  %and2 = and <4 x i32> %not, %neg
  %ret = or <4 x i32> %and1, %and2
  ret <4 x i32> %ret
}
