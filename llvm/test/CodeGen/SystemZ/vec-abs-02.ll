; Test v8i16 absolute.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test with slt.
define <8 x i16> @f1(<8 x i16> %val) {
; CHECK-LABEL: f1:
; CHECK: vlph %v24, %v24
; CHECK: br %r14
  %cmp = icmp slt <8 x i16> %val, zeroinitializer
  %neg = sub <8 x i16> zeroinitializer, %val
  %ret = select <8 x i1> %cmp, <8 x i16> %neg, <8 x i16> %val
  ret <8 x i16> %ret
}

; Test with sle.
define <8 x i16> @f2(<8 x i16> %val) {
; CHECK-LABEL: f2:
; CHECK: vlph %v24, %v24
; CHECK: br %r14
  %cmp = icmp sle <8 x i16> %val, zeroinitializer
  %neg = sub <8 x i16> zeroinitializer, %val
  %ret = select <8 x i1> %cmp, <8 x i16> %neg, <8 x i16> %val
  ret <8 x i16> %ret
}

; Test with sgt.
define <8 x i16> @f3(<8 x i16> %val) {
; CHECK-LABEL: f3:
; CHECK: vlph %v24, %v24
; CHECK: br %r14
  %cmp = icmp sgt <8 x i16> %val, zeroinitializer
  %neg = sub <8 x i16> zeroinitializer, %val
  %ret = select <8 x i1> %cmp, <8 x i16> %val, <8 x i16> %neg
  ret <8 x i16> %ret
}

; Test with sge.
define <8 x i16> @f4(<8 x i16> %val) {
; CHECK-LABEL: f4:
; CHECK: vlph %v24, %v24
; CHECK: br %r14
  %cmp = icmp sge <8 x i16> %val, zeroinitializer
  %neg = sub <8 x i16> zeroinitializer, %val
  %ret = select <8 x i1> %cmp, <8 x i16> %val, <8 x i16> %neg
  ret <8 x i16> %ret
}

; Test that negative absolute uses VLPH too.  There is no vector equivalent
; of LOAD NEGATIVE.
define <8 x i16> @f5(<8 x i16> %val) {
; CHECK-LABEL: f5:
; CHECK: vlph [[REG:%v[0-9]+]], %v24
; CHECK: vlch %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp slt <8 x i16> %val, zeroinitializer
  %neg = sub <8 x i16> zeroinitializer, %val
  %abs = select <8 x i1> %cmp, <8 x i16> %neg, <8 x i16> %val
  %ret = sub <8 x i16> zeroinitializer, %abs
  ret <8 x i16> %ret
}

; Try another form of negative absolute (slt version).
define <8 x i16> @f6(<8 x i16> %val) {
; CHECK-LABEL: f6:
; CHECK: vlph [[REG:%v[0-9]+]], %v24
; CHECK: vlch %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp slt <8 x i16> %val, zeroinitializer
  %neg = sub <8 x i16> zeroinitializer, %val
  %ret = select <8 x i1> %cmp, <8 x i16> %val, <8 x i16> %neg
  ret <8 x i16> %ret
}

; Test with sle.
define <8 x i16> @f7(<8 x i16> %val) {
; CHECK-LABEL: f7:
; CHECK: vlph [[REG:%v[0-9]+]], %v24
; CHECK: vlch %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp sle <8 x i16> %val, zeroinitializer
  %neg = sub <8 x i16> zeroinitializer, %val
  %ret = select <8 x i1> %cmp, <8 x i16> %val, <8 x i16> %neg
  ret <8 x i16> %ret
}

; Test with sgt.
define <8 x i16> @f8(<8 x i16> %val) {
; CHECK-LABEL: f8:
; CHECK: vlph [[REG:%v[0-9]+]], %v24
; CHECK: vlch %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp sgt <8 x i16> %val, zeroinitializer
  %neg = sub <8 x i16> zeroinitializer, %val
  %ret = select <8 x i1> %cmp, <8 x i16> %neg, <8 x i16> %val
  ret <8 x i16> %ret
}

; Test with sge.
define <8 x i16> @f9(<8 x i16> %val) {
; CHECK-LABEL: f9:
; CHECK: vlph [[REG:%v[0-9]+]], %v24
; CHECK: vlch %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp sge <8 x i16> %val, zeroinitializer
  %neg = sub <8 x i16> zeroinitializer, %val
  %ret = select <8 x i1> %cmp, <8 x i16> %neg, <8 x i16> %val
  ret <8 x i16> %ret
}

; Test with an SRA-based boolean vector.
define <8 x i16> @f10(<8 x i16> %val) {
; CHECK-LABEL: f10:
; CHECK: vlph %v24, %v24
; CHECK: br %r14
  %shr = ashr <8 x i16> %val,
              <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  %neg = sub <8 x i16> zeroinitializer, %val
  %and1 = and <8 x i16> %shr, %neg
  %not = xor <8 x i16> %shr,
             <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %and2 = and <8 x i16> %not, %val
  %ret = or <8 x i16> %and1, %and2
  ret <8 x i16> %ret
}

; ...and again in reverse
define <8 x i16> @f11(<8 x i16> %val) {
; CHECK-LABEL: f11:
; CHECK: vlph [[REG:%v[0-9]+]], %v24
; CHECK: vlch %v24, [[REG]]
; CHECK: br %r14
  %shr = ashr <8 x i16> %val,
              <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  %and1 = and <8 x i16> %shr, %val
  %not = xor <8 x i16> %shr,
             <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %neg = sub <8 x i16> zeroinitializer, %val
  %and2 = and <8 x i16> %not, %neg
  %ret = or <8 x i16> %and1, %and2
  ret <8 x i16> %ret
}
