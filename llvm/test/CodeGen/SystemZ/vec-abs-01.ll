; Test v16i8 absolute.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test with slt.
define <16 x i8> @f1(<16 x i8> %val) {
; CHECK-LABEL: f1:
; CHECK: vlpb %v24, %v24
; CHECK: br %r14
  %cmp = icmp slt <16 x i8> %val, zeroinitializer
  %neg = sub <16 x i8> zeroinitializer, %val
  %ret = select <16 x i1> %cmp, <16 x i8> %neg, <16 x i8> %val
  ret <16 x i8> %ret
}

; Test with sle.
define <16 x i8> @f2(<16 x i8> %val) {
; CHECK-LABEL: f2:
; CHECK: vlpb %v24, %v24
; CHECK: br %r14
  %cmp = icmp sle <16 x i8> %val, zeroinitializer
  %neg = sub <16 x i8> zeroinitializer, %val
  %ret = select <16 x i1> %cmp, <16 x i8> %neg, <16 x i8> %val
  ret <16 x i8> %ret
}

; Test with sgt.
define <16 x i8> @f3(<16 x i8> %val) {
; CHECK-LABEL: f3:
; CHECK: vlpb %v24, %v24
; CHECK: br %r14
  %cmp = icmp sgt <16 x i8> %val, zeroinitializer
  %neg = sub <16 x i8> zeroinitializer, %val
  %ret = select <16 x i1> %cmp, <16 x i8> %val, <16 x i8> %neg
  ret <16 x i8> %ret
}

; Test with sge.
define <16 x i8> @f4(<16 x i8> %val) {
; CHECK-LABEL: f4:
; CHECK: vlpb %v24, %v24
; CHECK: br %r14
  %cmp = icmp sge <16 x i8> %val, zeroinitializer
  %neg = sub <16 x i8> zeroinitializer, %val
  %ret = select <16 x i1> %cmp, <16 x i8> %val, <16 x i8> %neg
  ret <16 x i8> %ret
}

; Test that negative absolute uses VLPB too.  There is no vector equivalent
; of LOAD NEGATIVE.
define <16 x i8> @f5(<16 x i8> %val) {
; CHECK-LABEL: f5:
; CHECK: vlpb [[REG:%v[0-9]+]], %v24
; CHECK: vlcb %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp slt <16 x i8> %val, zeroinitializer
  %neg = sub <16 x i8> zeroinitializer, %val
  %abs = select <16 x i1> %cmp, <16 x i8> %neg, <16 x i8> %val
  %ret = sub <16 x i8> zeroinitializer, %abs
  ret <16 x i8> %ret
}

; Try another form of negative absolute (slt version).
define <16 x i8> @f6(<16 x i8> %val) {
; CHECK-LABEL: f6:
; CHECK: vlpb [[REG:%v[0-9]+]], %v24
; CHECK: vlcb %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp slt <16 x i8> %val, zeroinitializer
  %neg = sub <16 x i8> zeroinitializer, %val
  %ret = select <16 x i1> %cmp, <16 x i8> %val, <16 x i8> %neg
  ret <16 x i8> %ret
}

; Test with sle.
define <16 x i8> @f7(<16 x i8> %val) {
; CHECK-LABEL: f7:
; CHECK: vlpb [[REG:%v[0-9]+]], %v24
; CHECK: vlcb %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp sle <16 x i8> %val, zeroinitializer
  %neg = sub <16 x i8> zeroinitializer, %val
  %ret = select <16 x i1> %cmp, <16 x i8> %val, <16 x i8> %neg
  ret <16 x i8> %ret
}

; Test with sgt.
define <16 x i8> @f8(<16 x i8> %val) {
; CHECK-LABEL: f8:
; CHECK: vlpb [[REG:%v[0-9]+]], %v24
; CHECK: vlcb %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp sgt <16 x i8> %val, zeroinitializer
  %neg = sub <16 x i8> zeroinitializer, %val
  %ret = select <16 x i1> %cmp, <16 x i8> %neg, <16 x i8> %val
  ret <16 x i8> %ret
}

; Test with sge.
define <16 x i8> @f9(<16 x i8> %val) {
; CHECK-LABEL: f9:
; CHECK: vlpb [[REG:%v[0-9]+]], %v24
; CHECK: vlcb %v24, [[REG]]
; CHECK: br %r14
  %cmp = icmp sge <16 x i8> %val, zeroinitializer
  %neg = sub <16 x i8> zeroinitializer, %val
  %ret = select <16 x i1> %cmp, <16 x i8> %neg, <16 x i8> %val
  ret <16 x i8> %ret
}

; Test with an SRA-based boolean vector.
define <16 x i8> @f10(<16 x i8> %val) {
; CHECK-LABEL: f10:
; CHECK: vlpb %v24, %v24
; CHECK: br %r14
  %shr = ashr <16 x i8> %val,
              <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7,
               i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  %neg = sub <16 x i8> zeroinitializer, %val
  %and1 = and <16 x i8> %shr, %neg
  %not = xor <16 x i8> %shr,
             <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
              i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %and2 = and <16 x i8> %not, %val
  %ret = or <16 x i8> %and1, %and2
  ret <16 x i8> %ret
}

; ...and again in reverse
define <16 x i8> @f11(<16 x i8> %val) {
; CHECK-LABEL: f11:
; CHECK: vlpb [[REG:%v[0-9]+]], %v24
; CHECK: vlcb %v24, [[REG]]
; CHECK: br %r14
  %shr = ashr <16 x i8> %val,
              <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7,
               i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  %and1 = and <16 x i8> %shr, %val
  %not = xor <16 x i8> %shr,
             <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
              i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %neg = sub <16 x i8> zeroinitializer, %val
  %and2 = and <16 x i8> %not, %neg
  %ret = or <16 x i8> %and1, %and2
  ret <16 x i8> %ret
}
