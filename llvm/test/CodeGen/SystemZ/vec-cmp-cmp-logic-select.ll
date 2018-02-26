; Test that a vector select with a logic combination of two compares do not
; produce any unnecessary pack, unpack or shift instructions.
; And, Or and Xor are tested.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s -check-prefix=CHECK-Z14

define <2 x i8> @fun0(<2 x i8> %val1, <2 x i8> %val2, <2 x i8> %val3, <2 x i8> %val4, <2 x i8> %val5, <2 x i8> %val6) {
; CHECK-LABEL: fun0:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqb [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vceqb [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vn %v0, [[REG0]], [[REG1]]
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i8> %val1, %val2
  %cmp1 = icmp eq <2 x i8> %val3, %val4
  %and = and <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i8> %val5, <2 x i8> %val6
  ret <2 x i8> %sel
}

define <2 x i16> @fun1(<2 x i8> %val1, <2 x i8> %val2, <2 x i8> %val3, <2 x i8> %val4, <2 x i16> %val5, <2 x i16> %val6) {
; CHECK-LABEL: fun1:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqb [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vceqb [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vn %v0, [[REG0]], [[REG1]]
; CHECK-NEXT:    vuphb %v0, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i8> %val1, %val2
  %cmp1 = icmp eq <2 x i8> %val3, %val4
  %and = and <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i16> %val5, <2 x i16> %val6
  ret <2 x i16> %sel
}

define <16 x i8> @fun2(<16 x i8> %val1, <16 x i8> %val2, <16 x i16> %val3, <16 x i16> %val4, <16 x i8> %val5, <16 x i8> %val6) {
; CHECK-LABEL: fun2:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqh [[REG0:%v[0-9]+]], %v30, %v27
; CHECK-DAG:     vceqh [[REG1:%v[0-9]+]], %v28, %v25
; CHECK-DAG:     vceqb [[REG2:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vpkh [[REG3:%v[0-9]+]], [[REG1]], [[REG0]]
; CHECK-NEXT:    vo %v0, [[REG2]], [[REG3]]
; CHECK-NEXT:    vsel %v24, %v29, %v31, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <16 x i8> %val1, %val2
  %cmp1 = icmp eq <16 x i16> %val3, %val4
  %and = or <16 x i1> %cmp0, %cmp1
  %sel = select <16 x i1> %and, <16 x i8> %val5, <16 x i8> %val6
  ret <16 x i8> %sel
}

define <16 x i16> @fun3(<16 x i8> %val1, <16 x i8> %val2, <16 x i16> %val3, <16 x i16> %val4, <16 x i16> %val5, <16 x i16> %val6) {
; CHECK-LABEL: fun3:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqb [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vuphb [[REG2:%v[0-9]+]], [[REG0]]
; CHECK-DAG:     vmrlg [[REG1:%v[0-9]+]], [[REG0]], [[REG0]]
; CHECK-DAG:     vuphb [[REG1]], [[REG1]]
; CHECK-DAG:     vceqh [[REG3:%v[0-9]+]], %v28, %v25
; CHECK-DAG:     vceqh [[REG4:%v[0-9]+]], %v30, %v27
; CHECK-DAG:     vl [[REG5:%v[0-9]+]], 176(%r15)
; CHECK-DAG:     vl [[REG6:%v[0-9]+]], 160(%r15)
; CHECK-DAG:     vo [[REG7:%v[0-9]+]], %v2, [[REG4]]
; CHECK-DAG:     vo [[REG8:%v[0-9]+]], [[REG2]], [[REG3]]
; CHECK-DAG:     vsel %v24, %v29, [[REG6]], [[REG8]]
; CHECK-DAG:     vsel %v26, %v31, [[REG5]], [[REG7]]
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <16 x i8> %val1, %val2
  %cmp1 = icmp eq <16 x i16> %val3, %val4
  %and = or <16 x i1> %cmp0, %cmp1
  %sel = select <16 x i1> %and, <16 x i16> %val5, <16 x i16> %val6
  ret <16 x i16> %sel
}

define <32 x i8> @fun4(<32 x i8> %val1, <32 x i8> %val2, <32 x i8> %val3, <32 x i8> %val4, <32 x i8> %val5, <32 x i8> %val6) {
; CHECK-LABEL: fun4:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqb [[REG0:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vceqb [[REG1:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vceqb [[REG2:%v[0-9]+]], %v25, %v29
; CHECK-DAG:     vceqb [[REG3:%v[0-9]+]], %v27, %v31
; CHECK-DAG:     vl [[REG4:%v[0-9]+]], 208(%r15)
; CHECK-DAG:     vl [[REG5:%v[0-9]+]], 176(%r15)
; CHECK-DAG:     vl [[REG6:%v[0-9]+]], 192(%r15)
; CHECK-DAG:     vl [[REG7:%v[0-9]+]], 160(%r15)
; CHECK-DAG:     vx [[REG8:%v[0-9]+]], [[REG1]], [[REG3]]
; CHECK-DAG:     vx [[REG9:%v[0-9]+]], [[REG0]], [[REG2]]
; CHECK-DAG:     vsel %v24, [[REG7]], [[REG6]], [[REG9]]
; CHECK-DAG:     vsel %v26, [[REG5]], [[REG4]], [[REG8]]
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <32 x i8> %val1, %val2
  %cmp1 = icmp eq <32 x i8> %val3, %val4
  %and = xor <32 x i1> %cmp0, %cmp1
  %sel = select <32 x i1> %and, <32 x i8> %val5, <32 x i8> %val6
  ret <32 x i8> %sel
}

define <2 x i8> @fun5(<2 x i16> %val1, <2 x i16> %val2, <2 x i8> %val3, <2 x i8> %val4, <2 x i8> %val5, <2 x i8> %val6) {
; CHECK-LABEL: fun5:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqh [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vpkh [[REG1:%v[0-9]+]], [[REG0]], [[REG0]]
; CHECK-DAG:     vceqb [[REG2:%v[0-9]+]], %v28, %v30
; CHECK-DAG:     vo %v0, [[REG1]], [[REG2]]
; CHECK-DAG:     vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i16> %val1, %val2
  %cmp1 = icmp eq <2 x i8> %val3, %val4
  %and = or <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i8> %val5, <2 x i8> %val6
  ret <2 x i8> %sel
}

define <2 x i16> @fun6(<2 x i16> %val1, <2 x i16> %val2, <2 x i8> %val3, <2 x i8> %val4, <2 x i16> %val5, <2 x i16> %val6) {
; CHECK-LABEL: fun6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqb %v1, %v28, %v30
; CHECK-NEXT:    vceqh %v0, %v24, %v26
; CHECK-NEXT:    vuphb %v1, %v1
; CHECK-NEXT:    vo %v0, %v0, %v1
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i16> %val1, %val2
  %cmp1 = icmp eq <2 x i8> %val3, %val4
  %and = or <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i16> %val5, <2 x i16> %val6
  ret <2 x i16> %sel
}

define <2 x i32> @fun7(<2 x i16> %val1, <2 x i16> %val2, <2 x i8> %val3, <2 x i8> %val4, <2 x i32> %val5, <2 x i32> %val6) {
; CHECK-LABEL: fun7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqb %v1, %v28, %v30
; CHECK-NEXT:    vceqh %v0, %v24, %v26
; CHECK-NEXT:    vuphb %v1, %v1
; CHECK-NEXT:    vo %v0, %v0, %v1
; CHECK-NEXT:    vuphh %v0, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i16> %val1, %val2
  %cmp1 = icmp eq <2 x i8> %val3, %val4
  %and = or <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i32> %val5, <2 x i32> %val6
  ret <2 x i32> %sel
}

define <8 x i8> @fun8(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3, <8 x i16> %val4, <8 x i8> %val5, <8 x i8> %val6) {
; CHECK-LABEL: fun8:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqh [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vceqh [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vx %v0, [[REG0]], [[REG1]]
; CHECK-NEXT:    vpkh %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <8 x i16> %val1, %val2
  %cmp1 = icmp eq <8 x i16> %val3, %val4
  %and = xor <8 x i1> %cmp0, %cmp1
  %sel = select <8 x i1> %and, <8 x i8> %val5, <8 x i8> %val6
  ret <8 x i8> %sel
}

define <8 x i16> @fun9(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3, <8 x i16> %val4, <8 x i16> %val5, <8 x i16> %val6) {
; CHECK-LABEL: fun9:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqh [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vceqh [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vx %v0, [[REG0]], [[REG1]]
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <8 x i16> %val1, %val2
  %cmp1 = icmp eq <8 x i16> %val3, %val4
  %and = xor <8 x i1> %cmp0, %cmp1
  %sel = select <8 x i1> %and, <8 x i16> %val5, <8 x i16> %val6
  ret <8 x i16> %sel
}

define <8 x i32> @fun10(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3, <8 x i16> %val4, <8 x i32> %val5, <8 x i32> %val6) {
; CHECK-LABEL: fun10:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqh [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vceqh [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vx [[REG2:%v[0-9]+]], [[REG0]], [[REG1]]
; CHECK-DAG:     vuphh [[REG3:%v[0-9]+]], [[REG2]]
; CHECK-DAG:     vmrlg [[REG4:%v[0-9]+]], [[REG2]], [[REG2]]
; CHECK-DAG:     vuphh [[REG5:%v[0-9]+]], [[REG4]]
; CHECK-NEXT:    vsel %v24, %v25, %v29, [[REG3]]
; CHECK-NEXT:    vsel %v26, %v27, %v31, [[REG5]]
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <8 x i16> %val1, %val2
  %cmp1 = icmp eq <8 x i16> %val3, %val4
  %and = xor <8 x i1> %cmp0, %cmp1
  %sel = select <8 x i1> %and, <8 x i32> %val5, <8 x i32> %val6
  ret <8 x i32> %sel
}

define <16 x i8> @fun11(<16 x i16> %val1, <16 x i16> %val2, <16 x i32> %val3, <16 x i32> %val4, <16 x i8> %val5, <16 x i8> %val6) {
; CHECK-LABEL: fun11:
; CHECK:       # %bb.0:
; CHECK-DAG:     vl [[REG0:%v[0-9]+]], 192(%r15)
; CHECK-DAG:     vl [[REG1:%v[0-9]+]], 208(%r15)
; CHECK-DAG:     vl [[REG2:%v[0-9]+]], 160(%r15)
; CHECK-DAG:     vl [[REG3:%v[0-9]+]], 176(%r15)
; CHECK-DAG:     vceqf [[REG4:%v[0-9]+]], %v27, [[REG3]]
; CHECK-DAG:     vceqf [[REG5:%v[0-9]+]], %v25, [[REG2]]
; CHECK-DAG:     vceqf [[REG6:%v[0-9]+]], %v31, [[REG1]]
; CHECK-DAG:     vceqf [[REG7:%v[0-9]+]], %v29, [[REG0]]
; CHECK-DAG:     vceqh [[REG8:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vceqh [[REG9:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vpkf [[REG10:%v[0-9]+]], [[REG5]], [[REG4]]
; CHECK-DAG:     vpkf [[REG11:%v[0-9]+]], [[REG7]], [[REG6]]
; CHECK-DAG:     vn [[REG12:%v[0-9]+]], [[REG9]], [[REG11]]
; CHECK-DAG:     vn [[REG13:%v[0-9]+]], [[REG8]], [[REG10]]
; CHECK-DAG:     vl [[REG14:%v[0-9]+]], 240(%r15)
; CHECK-DAG:     vl [[REG15:%v[0-9]+]], 224(%r15)
; CHECK-DAG:     vpkh [[REG16:%v[0-9]+]], [[REG13]], [[REG12]]
; CHECK-NEXT:    vsel %v24, [[REG15]], [[REG14]], [[REG16]]
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <16 x i16> %val1, %val2
  %cmp1 = icmp eq <16 x i32> %val3, %val4
  %and = and <16 x i1> %cmp0, %cmp1
  %sel = select <16 x i1> %and, <16 x i8> %val5, <16 x i8> %val6
  ret <16 x i8> %sel
}

define <16 x i16> @fun12(<16 x i16> %val1, <16 x i16> %val2, <16 x i32> %val3, <16 x i32> %val4, <16 x i16> %val5, <16 x i16> %val6) {
; CHECK-LABEL: fun12:
; CHECK:       # %bb.0:
; CHECK-DAG:     vl [[REG0:%v[0-9]+]], 192(%r15)
; CHECK-DAG:     vl [[REG1:%v[0-9]+]], 208(%r15)
; CHECK-DAG:     vl [[REG2:%v[0-9]+]], 160(%r15)
; CHECK-DAG:     vl [[REG3:%v[0-9]+]], 176(%r15)
; CHECK-DAG:     vceqf [[REG4:%v[0-9]+]], %v27, [[REG3]]
; CHECK-DAG:     vceqf [[REG5:%v[0-9]+]], %v25, [[REG2]]
; CHECK-DAG:     vceqf [[REG6:%v[0-9]+]], %v31, [[REG1]]
; CHECK-DAG:     vceqf [[REG7:%v[0-9]+]], %v29, [[REG0]]
; CHECK-DAG:     vceqh [[REG8:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vceqh [[REG9:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vpkf [[REG10:%v[0-9]+]], [[REG5]], [[REG4]]
; CHECK-DAG:     vpkf [[REG11:%v[0-9]+]], [[REG7]], [[REG6]]
; CHECK-DAG:     vl [[REG12:%v[0-9]+]], 272(%r15)
; CHECK-DAG:     vl [[REG13:%v[0-9]+]], 240(%r15)
; CHECK-DAG:     vl [[REG14:%v[0-9]+]], 256(%r15)
; CHECK-DAG:     vl [[REG15:%v[0-9]+]], 224(%r15)
; CHECK-DAG:     vn [[REG16:%v[0-9]+]], [[REG9]], [[REG11]]
; CHECK-DAG:     vn [[REG17:%v[0-9]+]], [[REG8]], [[REG10]]
; CHECK-DAG:     vsel %v24, [[REG15]], [[REG14]], [[REG17]]
; CHECK-DAG:     vsel %v26, [[REG13]], [[REG12]], [[REG16]]
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <16 x i16> %val1, %val2
  %cmp1 = icmp eq <16 x i32> %val3, %val4
  %and = and <16 x i1> %cmp0, %cmp1
  %sel = select <16 x i1> %and, <16 x i16> %val5, <16 x i16> %val6
  ret <16 x i16> %sel
}

define <2 x i16> @fun13(<2 x i32> %val1, <2 x i32> %val2, <2 x i64> %val3, <2 x i64> %val4, <2 x i16> %val5, <2 x i16> %val6) {
; CHECK-LABEL: fun13:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqg %v1, %v28, %v30
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-NEXT:    vpkg %v1, %v1, %v1
; CHECK-NEXT:    vx %v0, %v0, %v1
; CHECK-NEXT:    vpkf %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i32> %val1, %val2
  %cmp1 = icmp eq <2 x i64> %val3, %val4
  %and = xor <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i16> %val5, <2 x i16> %val6
  ret <2 x i16> %sel
}

define <2 x i32> @fun14(<2 x i32> %val1, <2 x i32> %val2, <2 x i64> %val3, <2 x i64> %val4, <2 x i32> %val5, <2 x i32> %val6) {
; CHECK-LABEL: fun14:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqg %v1, %v28, %v30
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-NEXT:    vpkg %v1, %v1, %v1
; CHECK-NEXT:    vx %v0, %v0, %v1
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i32> %val1, %val2
  %cmp1 = icmp eq <2 x i64> %val3, %val4
  %and = xor <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i32> %val5, <2 x i32> %val6
  ret <2 x i32> %sel
}

define <2 x i64> @fun15(<2 x i32> %val1, <2 x i32> %val2, <2 x i64> %val3, <2 x i64> %val4, <2 x i64> %val5, <2 x i64> %val6) {
; CHECK-LABEL: fun15:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqf [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vuphf [[REG1:%v[0-9]+]], [[REG0]]
; CHECK-DAG:     vceqg [[REG2:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vx %v0, [[REG1]], [[REG2]]
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i32> %val1, %val2
  %cmp1 = icmp eq <2 x i64> %val3, %val4
  %and = xor <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i64> %val5, <2 x i64> %val6
  ret <2 x i64> %sel
}

define <4 x i16> @fun16(<4 x i32> %val1, <4 x i32> %val2, <4 x i16> %val3, <4 x i16> %val4, <4 x i16> %val5, <4 x i16> %val6) {
; CHECK-LABEL: fun16:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqf [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vpkf [[REG1:%v[0-9]+]], [[REG0]], [[REG0]]
; CHECK-DAG:     vceqh [[REG2:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vn %v0, [[REG1]], [[REG2]]
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <4 x i32> %val1, %val2
  %cmp1 = icmp eq <4 x i16> %val3, %val4
  %and = and <4 x i1> %cmp0, %cmp1
  %sel = select <4 x i1> %and, <4 x i16> %val5, <4 x i16> %val6
  ret <4 x i16> %sel
}

define <4 x i32> @fun17(<4 x i32> %val1, <4 x i32> %val2, <4 x i16> %val3, <4 x i16> %val4, <4 x i32> %val5, <4 x i32> %val6) {
; CHECK-LABEL: fun17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqh %v1, %v28, %v30
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-NEXT:    vuphh %v1, %v1
; CHECK-NEXT:    vn %v0, %v0, %v1
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <4 x i32> %val1, %val2
  %cmp1 = icmp eq <4 x i16> %val3, %val4
  %and = and <4 x i1> %cmp0, %cmp1
  %sel = select <4 x i1> %and, <4 x i32> %val5, <4 x i32> %val6
  ret <4 x i32> %sel
}

define <4 x i64> @fun18(<4 x i32> %val1, <4 x i32> %val2, <4 x i16> %val3, <4 x i16> %val4, <4 x i64> %val5, <4 x i64> %val6) {
; CHECK-LABEL: fun18:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqh %v1, %v28, %v30
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-NEXT:    vuphh %v1, %v1
; CHECK-NEXT:    vn %v0, %v0, %v1
; CHECK-DAG:     vuphf [[REG0:%v[0-9]+]], %v0
; CHECK-DAG:     vmrlg [[REG1:%v[0-9]+]], %v0, %v0
; CHECK-DAG:     vuphf [[REG2:%v[0-9]+]], [[REG1]]
; CHECK-NEXT:    vsel %v24, %v25, %v29, [[REG0]]
; CHECK-NEXT:    vsel %v26, %v27, %v31, [[REG2]]
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <4 x i32> %val1, %val2
  %cmp1 = icmp eq <4 x i16> %val3, %val4
  %and = and <4 x i1> %cmp0, %cmp1
  %sel = select <4 x i1> %and, <4 x i64> %val5, <4 x i64> %val6
  ret <4 x i64> %sel
}

define <8 x i16> @fun19(<8 x i32> %val1, <8 x i32> %val2, <8 x i32> %val3, <8 x i32> %val4, <8 x i16> %val5, <8 x i16> %val6) {
; CHECK-LABEL: fun19:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqf [[REG0:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vceqf [[REG1:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vceqf [[REG2:%v[0-9]+]], %v25, %v29
; CHECK-DAG:     vceqf [[REG3:%v[0-9]+]], %v27, %v31
; CHECK-DAG:     vo [[REG4:%v[0-9]+]], [[REG1]], [[REG3]]
; CHECK-DAG:     vo [[REG5:%v[0-9]+]], [[REG0]], [[REG2]]
; CHECK-DAG:     vl [[REG6:%v[0-9]+]], 176(%r15)
; CHECK-DAG:     vl [[REG7:%v[0-9]+]], 160(%r15)
; CHECK-DAG:     vpkf [[REG8:%v[0-9]+]], [[REG5]], [[REG4]]
; CHECK-NEXT:    vsel %v24, [[REG7]], [[REG6]], [[REG8]]
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <8 x i32> %val1, %val2
  %cmp1 = icmp eq <8 x i32> %val3, %val4
  %and = or <8 x i1> %cmp0, %cmp1
  %sel = select <8 x i1> %and, <8 x i16> %val5, <8 x i16> %val6
  ret <8 x i16> %sel
}

define <8 x i32> @fun20(<8 x i32> %val1, <8 x i32> %val2, <8 x i32> %val3, <8 x i32> %val4, <8 x i32> %val5, <8 x i32> %val6) {
; CHECK-LABEL: fun20:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqf [[REG0:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vceqf [[REG1:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vceqf [[REG2:%v[0-9]+]], %v25, %v29
; CHECK-DAG:     vceqf [[REG3:%v[0-9]+]], %v27, %v31
; CHECK-DAG:     vl [[REG4:%v[0-9]+]], 208(%r15)
; CHECK-DAG:     vl [[REG5:%v[0-9]+]], 176(%r15)
; CHECK-DAG:     vl [[REG6:%v[0-9]+]], 192(%r15)
; CHECK-DAG:     vl [[REG7:%v[0-9]+]], 160(%r15)
; CHECK-DAG:     vo [[REG8:%v[0-9]+]], [[REG1]], [[REG3]]
; CHECK-DAG:     vo [[REG9:%v[0-9]+]], [[REG0]], [[REG2]]
; CHECK-DAG:     vsel %v24, [[REG7]], [[REG6]], [[REG9]]
; CHECK-DAG:     vsel %v26, [[REG5]], [[REG4]], [[REG8]]
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <8 x i32> %val1, %val2
  %cmp1 = icmp eq <8 x i32> %val3, %val4
  %and = or <8 x i1> %cmp0, %cmp1
  %sel = select <8 x i1> %and, <8 x i32> %val5, <8 x i32> %val6
  ret <8 x i32> %sel
}

define <2 x i32> @fun21(<2 x i64> %val1, <2 x i64> %val2, <2 x i64> %val3, <2 x i64> %val4, <2 x i32> %val5, <2 x i32> %val6) {
; CHECK-LABEL: fun21:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqg [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vceqg [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vn %v0, [[REG0]], [[REG1]]
; CHECK-NEXT:    vpkg %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i64> %val1, %val2
  %cmp1 = icmp eq <2 x i64> %val3, %val4
  %and = and <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i32> %val5, <2 x i32> %val6
  ret <2 x i32> %sel
}

define <2 x i64> @fun22(<2 x i64> %val1, <2 x i64> %val2, <2 x i64> %val3, <2 x i64> %val4, <2 x i64> %val5, <2 x i64> %val6) {
; CHECK-LABEL: fun22:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqg [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vceqg [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vn %v0, [[REG0]], [[REG1]]
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i64> %val1, %val2
  %cmp1 = icmp eq <2 x i64> %val3, %val4
  %and = and <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x i64> %val5, <2 x i64> %val6
  ret <2 x i64> %sel
}

define <4 x i32> @fun23(<4 x i64> %val1, <4 x i64> %val2, <4 x i32> %val3, <4 x i32> %val4, <4 x i32> %val5, <4 x i32> %val6) {
; CHECK-LABEL: fun23:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqg %v0, %v26, %v30
; CHECK-NEXT:    vceqg %v1, %v24, %v28
; CHECK-NEXT:    vpkg %v0, %v1, %v0
; CHECK-NEXT:    vceqf %v1, %v25, %v27
; CHECK-NEXT:    vx %v0, %v0, %v1
; CHECK-NEXT:    vsel %v24, %v29, %v31, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <4 x i64> %val1, %val2
  %cmp1 = icmp eq <4 x i32> %val3, %val4
  %and = xor <4 x i1> %cmp0, %cmp1
  %sel = select <4 x i1> %and, <4 x i32> %val5, <4 x i32> %val6
  ret <4 x i32> %sel
}

define <4 x i64> @fun24(<4 x i64> %val1, <4 x i64> %val2, <4 x i32> %val3, <4 x i32> %val4, <4 x i64> %val5, <4 x i64> %val6) {
; CHECK-LABEL: fun24:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqf [[REG0:%v[0-9]+]], %v25, %v27
; CHECK-NEXT:    vuphf [[REG1:%v[0-9]+]], [[REG0]]
; CHECK-NEXT:    vmrlg [[REG2:%v[0-9]+]], [[REG0]], [[REG0]]
; CHECK-NEXT:    vceqg [[REG3:%v[0-9]+]], %v24, %v28
; CHECK-NEXT:    vceqg [[REG4:%v[0-9]+]], %v26, %v30
; CHECK-NEXT:    vuphf [[REG5:%v[0-9]+]], [[REG2]]
; CHECK-DAG:     vl [[REG6:%v[0-9]+]], 176(%r15)
; CHECK-DAG:     vl [[REG7:%v[0-9]+]], 160(%r15)
; CHECK-DAG:     vx [[REG8:%v[0-9]+]], [[REG4]], [[REG5]]
; CHECK-DAG:     vx [[REG9:%v[0-9]+]], [[REG3]], [[REG1]]
; CHECK-DAG:     vsel %v24, %v29, [[REG7]], [[REG9]]
; CHECK-DAG:     vsel %v26, %v31, [[REG6]], [[REG8]]
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <4 x i64> %val1, %val2
  %cmp1 = icmp eq <4 x i32> %val3, %val4
  %and = xor <4 x i1> %cmp0, %cmp1
  %sel = select <4 x i1> %and, <4 x i64> %val5, <4 x i64> %val6
  ret <4 x i64> %sel
}

define <2 x float> @fun25(<2 x float> %val1, <2 x float> %val2, <2 x double> %val3, <2 x double> %val4, <2 x float> %val5, <2 x float> %val6) {
; CHECK-LABEL: fun25:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmrlf %v0, %v26, %v26
; CHECK-NEXT:    vmrlf %v1, %v24, %v24
; CHECK-NEXT:    vldeb %v0, %v0
; CHECK-NEXT:    vldeb %v1, %v1
; CHECK-NEXT:    vfchdb %v0, %v1, %v0
; CHECK-NEXT:    vmrhf %v1, %v26, %v26
; CHECK-NEXT:    vmrhf %v2, %v24, %v24
; CHECK-NEXT:    vldeb %v1, %v1
; CHECK-NEXT:    vldeb %v2, %v2
; CHECK-NEXT:    vfchdb %v1, %v2, %v1
; CHECK-NEXT:    vpkg %v0, %v1, %v0
; CHECK-NEXT:    vfchdb %v1, %v28, %v30
; CHECK-NEXT:    vpkg %v1, %v1, %v1
; CHECK-NEXT:    vo %v0, %v0, %v1
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
;
; CHECK-Z14-LABEL: fun25:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchdb %v1, %v28, %v30
; CHECK-Z14-NEXT:    vfchsb %v0, %v24, %v26
; CHECK-Z14-NEXT:    vpkg %v1, %v1, %v1
; CHECK-Z14-NEXT:    vo %v0, %v0, %v1
; CHECK-Z14-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-Z14-NEXT:    br %r14
  %cmp0 = fcmp ogt <2 x float> %val1, %val2
  %cmp1 = fcmp ogt <2 x double> %val3, %val4
  %and = or <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x float> %val5, <2 x float> %val6
  ret <2 x float> %sel
}

define <2 x double> @fun26(<2 x float> %val1, <2 x float> %val2, <2 x double> %val3, <2 x double> %val4, <2 x double> %val5, <2 x double> %val6) {
; CHECK-LABEL: fun26:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmrlf %v0, %v26, %v26
; CHECK-NEXT:    vmrlf %v1, %v24, %v24
; CHECK-NEXT:    vldeb %v0, %v0
; CHECK-NEXT:    vldeb %v1, %v1
; CHECK-NEXT:    vfchdb %v0, %v1, %v0
; CHECK-NEXT:    vmrhf %v1, %v26, %v26
; CHECK-NEXT:    vmrhf %v2, %v24, %v24
; CHECK-NEXT:    vldeb %v1, %v1
; CHECK-NEXT:    vldeb %v2, %v2
; CHECK-NEXT:    vfchdb %v1, %v2, %v1
; CHECK-NEXT:    vpkg %v0, %v1, %v0
; CHECK-NEXT:    vuphf %v0, %v0
; CHECK-NEXT:    vfchdb %v1, %v28, %v30
; CHECK-NEXT:    vo %v0, %v0, %v1
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
;
; CHECK-Z14-LABEL: fun26:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchsb %v0, %v24, %v26
; CHECK-Z14-NEXT:    vuphf %v0, %v0
; CHECK-Z14-NEXT:    vfchdb %v1, %v28, %v30
; CHECK-Z14-NEXT:    vo %v0, %v0, %v1
; CHECK-Z14-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-Z14-NEXT:    br %r14
  %cmp0 = fcmp ogt <2 x float> %val1, %val2
  %cmp1 = fcmp ogt <2 x double> %val3, %val4
  %and = or <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x double> %val5, <2 x double> %val6
  ret <2 x double> %sel
}

; Also check a widening select of a vector of floats
define <2 x float> @fun27(<2 x i8> %val1, <2 x i8> %val2, <2 x i8> %val3, <2 x i8> %val4, <2 x float> %val5, <2 x float> %val6) {
; CHECK-LABEL: fun27:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqb [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vceqb [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vo %v0, [[REG0]], [[REG1]]
; CHECK-NEXT:    vuphb %v0, %v0
; CHECK-NEXT:    vuphh %v0, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = icmp eq <2 x i8> %val1, %val2
  %cmp1 = icmp eq <2 x i8> %val3, %val4
  %and = or <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x float> %val5, <2 x float> %val6
  ret <2 x float> %sel
}

define <4 x float> @fun28(<4 x float> %val1, <4 x float> %val2, <4 x float> %val3, <4 x float> %val4, <4 x float> %val5, <4 x float> %val6) {
; CHECK-LABEL: fun28:
; CHECK:       # %bb.0:
; CHECK-DAG:     vmrlf [[REG0:%v[0-9]+]], %v26, %v26
; CHECK-DAG:     vmrlf [[REG1:%v[0-9]+]], %v24, %v24
; CHECK-DAG:     vldeb [[REG2:%v[0-9]+]], [[REG0]]
; CHECK-DAG:     vldeb [[REG3:%v[0-9]+]], [[REG1]]
; CHECK-DAG:     vfchdb [[REG4:%v[0-9]+]], [[REG3]], [[REG2]]
; CHECK-DAG:     vmrhf [[REG5:%v[0-9]+]], %v26, %v26
; CHECK-DAG:     vmrhf [[REG6:%v[0-9]+]], %v24, %v24
; CHECK-DAG:     vldeb [[REG7:%v[0-9]+]], [[REG5]]
; CHECK-DAG:     vmrhf [[REG8:%v[0-9]+]], %v28, %v28
; CHECK-DAG:     vldeb [[REG9:%v[0-9]+]], [[REG6]]
; CHECK-DAG:     vfchdb [[REG10:%v[0-9]+]], [[REG9]], [[REG7]]
; CHECK-DAG:     vpkg [[REG11:%v[0-9]+]], [[REG10]], [[REG4]]
; CHECK-DAG:     vmrlf [[REG12:%v[0-9]+]], %v30, %v30
; CHECK-DAG:     vmrlf [[REG13:%v[0-9]+]], %v28, %v28
; CHECK-DAG:     vldeb [[REG14:%v[0-9]+]], [[REG12]]
; CHECK-DAG:     vldeb [[REG15:%v[0-9]+]], [[REG13]]
; CHECK-DAG:     vfchdb [[REG16:%v[0-9]+]], [[REG15]], [[REG14]]
; CHECK-DAG:     vmrhf [[REG17:%v[0-9]+]], %v30, %v30
; CHECK-DAG:     vldeb [[REG19:%v[0-9]+]], [[REG17]]
; CHECK-DAG:     vldeb [[REG20:%v[0-9]+]], [[REG8]]
; CHECK-NEXT:    vfchdb %v2, [[REG20]], [[REG19]]
; CHECK-NEXT:    vpkg [[REG21:%v[0-9]+]], %v2, [[REG16]]
; CHECK-NEXT:    vx %v0, [[REG11]], [[REG21]]
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
;
; CHECK-Z14-LABEL: fun28:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchsb %v0, %v24, %v26
; CHECK-Z14-NEXT:    vfchsb %v1, %v28, %v30
; CHECK-Z14-NEXT:    vx %v0, %v0, %v1
; CHECK-Z14-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-Z14-NEXT:    br %r14
  %cmp0 = fcmp ogt <4 x float> %val1, %val2
  %cmp1 = fcmp ogt <4 x float> %val3, %val4
  %and = xor <4 x i1> %cmp0, %cmp1
  %sel = select <4 x i1> %and, <4 x float> %val5, <4 x float> %val6
  ret <4 x float> %sel
}

define <4 x double> @fun29(<4 x float> %val1, <4 x float> %val2, <4 x float> %val3, <4 x float> %val4, <4 x double> %val5, <4 x double> %val6) {
; CHECK-LABEL: fun29:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmrlf %v0, %v26, %v26
; CHECK-NEXT:    vmrlf %v1, %v24, %v24
; CHECK-NEXT:    vldeb %v0, %v0
; CHECK-NEXT:    vldeb %v1, %v1
; CHECK-NEXT:    vfchdb %v0, %v1, %v0
; CHECK-NEXT:    vmrhf %v1, %v26, %v26
; CHECK-NEXT:    vmrhf %v2, %v24, %v24
; CHECK-NEXT:    vldeb %v1, %v1
; CHECK-NEXT:    vmrhf %v3, %v28, %v28
; CHECK-NEXT:    vldeb %v2, %v2
; CHECK-NEXT:    vfchdb %v1, %v2, %v1
; CHECK-NEXT:    vpkg %v0, %v1, %v0
; CHECK-NEXT:    vmrlf %v1, %v30, %v30
; CHECK-NEXT:    vmrlf %v2, %v28, %v28
; CHECK-NEXT:    vldeb %v1, %v1
; CHECK-NEXT:    vldeb %v2, %v2
; CHECK-NEXT:    vfchdb %v1, %v2, %v1
; CHECK-NEXT:    vmrhf %v2, %v30, %v30
; CHECK-NEXT:    vldeb %v2, %v2
; CHECK-NEXT:    vldeb %v3, %v3
; CHECK-NEXT:    vfchdb %v2, %v3, %v2
; CHECK-NEXT:    vpkg %v1, %v2, %v1
; CHECK-NEXT:    vx %v0, %v0, %v1
; CHECK-NEXT:    vmrlg %v1, %v0, %v0
; CHECK-NEXT:    vuphf %v1, %v1
; CHECK-NEXT:    vuphf %v0, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v29, %v0
; CHECK-NEXT:    vsel %v26, %v27, %v31, %v1
; CHECK-NEXT:    br %r14
;
; CHECK-Z14-LABEL: fun29:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchsb %v0, %v24, %v26
; CHECK-Z14-NEXT:    vfchsb %v1, %v28, %v30
; CHECK-Z14-NEXT:    vx %v0, %v0, %v1
; CHECK-Z14-NEXT:    vmrlg %v1, %v0, %v0
; CHECK-Z14-NEXT:    vuphf %v1, %v1
; CHECK-Z14-NEXT:    vuphf %v0, %v0
; CHECK-Z14-NEXT:    vsel %v24, %v25, %v29, %v0
; CHECK-Z14-NEXT:    vsel %v26, %v27, %v31, %v1
; CHECK-Z14-NEXT:    br %r14
  %cmp0 = fcmp ogt <4 x float> %val1, %val2
  %cmp1 = fcmp ogt <4 x float> %val3, %val4
  %and = xor <4 x i1> %cmp0, %cmp1
  %sel = select <4 x i1> %and, <4 x double> %val5, <4 x double> %val6
  ret <4 x double> %sel
}

define <8 x float> @fun30(<8 x float> %val1, <8 x float> %val2, <8 x double> %val3, <8 x double> %val4, <8 x float> %val5, <8 x float> %val6) {
; CHECK-LABEL: fun30:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmrlf %v16, %v28, %v28
; CHECK-NEXT:    vmrlf %v17, %v24, %v24
; CHECK-NEXT:    vldeb %v16, %v16
; CHECK-NEXT:    vldeb %v17, %v17
; CHECK-NEXT:    vfchdb %v16, %v17, %v16
; CHECK-NEXT:    vmrhf %v17, %v28, %v28
; CHECK-NEXT:    vmrhf %v18, %v24, %v24
; CHECK-NEXT:    vldeb %v17, %v17
; CHECK-NEXT:    vl %v4, 192(%r15)
; CHECK-NEXT:    vldeb %v18, %v18
; CHECK-NEXT:    vl %v5, 208(%r15)
; CHECK-NEXT:    vl %v6, 160(%r15)
; CHECK-NEXT:    vl %v7, 176(%r15)
; CHECK-NEXT:    vl %v0, 272(%r15)
; CHECK-NEXT:    vl %v1, 240(%r15)
; CHECK-NEXT:    vfchdb %v17, %v18, %v17
; CHECK-NEXT:    vl %v2, 256(%r15)
; CHECK-NEXT:    vl %v3, 224(%r15)
; CHECK-NEXT:    vpkg %v16, %v17, %v16
; CHECK-NEXT:    vmrlf %v17, %v30, %v30
; CHECK-NEXT:    vmrlf %v18, %v26, %v26
; CHECK-NEXT:    vmrhf %v19, %v26, %v26
; CHECK-NEXT:    vfchdb %v7, %v27, %v7
; CHECK-NEXT:    vfchdb %v6, %v25, %v6
; CHECK-NEXT:    vfchdb %v5, %v31, %v5
; CHECK-NEXT:    vfchdb %v4, %v29, %v4
; CHECK-NEXT:    vpkg %v6, %v6, %v7
; CHECK-NEXT:    vpkg %v4, %v4, %v5
; CHECK-NEXT:    vn %v5, %v16, %v6
; CHECK-NEXT:    vsel %v24, %v3, %v2, %v5
; CHECK-NEXT:    vldeb %v17, %v17
; CHECK-NEXT:    vldeb %v18, %v18
; CHECK-NEXT:    vfchdb %v17, %v18, %v17
; CHECK-NEXT:    vmrhf %v18, %v30, %v30
; CHECK-NEXT:    vldeb %v18, %v18
; CHECK-NEXT:    vldeb %v19, %v19
; CHECK-NEXT:    vfchdb %v18, %v19, %v18
; CHECK-NEXT:    vpkg %v17, %v18, %v17
; CHECK-NEXT:    vn %v4, %v17, %v4
; CHECK-NEXT:    vsel %v26, %v1, %v0, %v4
; CHECK-NEXT:    br %r14
;
; CHECK-Z14-LABEL: fun30:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vl %v4, 192(%r15)
; CHECK-Z14-NEXT:    vl %v5, 208(%r15)
; CHECK-Z14-NEXT:    vl %v6, 160(%r15)
; CHECK-Z14-NEXT:    vl %v7, 176(%r15)
; CHECK-Z14-NEXT:    vfchdb %v7, %v27, %v7
; CHECK-Z14-NEXT:    vfchdb %v6, %v25, %v6
; CHECK-Z14-NEXT:    vfchdb %v5, %v31, %v5
; CHECK-Z14-NEXT:    vfchdb %v4, %v29, %v4
; CHECK-Z14-NEXT:    vfchsb %v16, %v24, %v28
; CHECK-Z14-NEXT:    vfchsb %v17, %v26, %v30
; CHECK-Z14-NEXT:    vpkg %v6, %v6, %v7
; CHECK-Z14-NEXT:    vpkg %v4, %v4, %v5
; CHECK-Z14-NEXT:    vl %v0, 272(%r15)
; CHECK-Z14-NEXT:    vl %v1, 240(%r15)
; CHECK-Z14-NEXT:    vl %v2, 256(%r15)
; CHECK-Z14-NEXT:    vl %v3, 224(%r15)
; CHECK-Z14-NEXT:    vn %v4, %v17, %v4
; CHECK-Z14-NEXT:    vn %v5, %v16, %v6
; CHECK-Z14-NEXT:    vsel %v24, %v3, %v2, %v5
; CHECK-Z14-NEXT:    vsel %v26, %v1, %v0, %v4
; CHECK-Z14-NEXT:    br %r14
  %cmp0 = fcmp ogt <8 x float> %val1, %val2
  %cmp1 = fcmp ogt <8 x double> %val3, %val4
  %and = and <8 x i1> %cmp0, %cmp1
  %sel = select <8 x i1> %and, <8 x float> %val5, <8 x float> %val6
  ret <8 x float> %sel
}

define <2 x float> @fun31(<2 x double> %val1, <2 x double> %val2, <2 x double> %val3, <2 x double> %val4, <2 x float> %val5, <2 x float> %val6) {
; CHECK-LABEL: fun31:
; CHECK:       # %bb.0:
; CHECK-DAG:     vfchdb [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vfchdb [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vx %v0, [[REG0]], [[REG1]]
; CHECK-NEXT:    vpkg %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = fcmp ogt <2 x double> %val1, %val2
  %cmp1 = fcmp ogt <2 x double> %val3, %val4
  %and = xor <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x float> %val5, <2 x float> %val6
  ret <2 x float> %sel
}

define <2 x double> @fun32(<2 x double> %val1, <2 x double> %val2, <2 x double> %val3, <2 x double> %val4, <2 x double> %val5, <2 x double> %val6) {
; CHECK-LABEL: fun32:
; CHECK:       # %bb.0:
; CHECK-DAG:     vfchdb [[REG0:%v[0-9]+]], %v24, %v26
; CHECK-DAG:     vfchdb [[REG1:%v[0-9]+]], %v28, %v30
; CHECK-NEXT:    vx %v0, [[REG0]], [[REG1]]
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp0 = fcmp ogt <2 x double> %val1, %val2
  %cmp1 = fcmp ogt <2 x double> %val3, %val4
  %and = xor <2 x i1> %cmp0, %cmp1
  %sel = select <2 x i1> %and, <2 x double> %val5, <2 x double> %val6
  ret <2 x double> %sel
}

define <4 x float> @fun33(<4 x double> %val1, <4 x double> %val2, <4 x float> %val3, <4 x float> %val4, <4 x float> %val5, <4 x float> %val6) {
; CHECK-LABEL: fun33:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vfchdb %v0, %v26, %v30
; CHECK-NEXT:    vfchdb %v1, %v24, %v28
; CHECK-NEXT:    vpkg %v0, %v1, %v0
; CHECK-NEXT:    vmrlf %v1, %v27, %v27
; CHECK-NEXT:    vmrlf %v2, %v25, %v25
; CHECK-NEXT:    vldeb %v1, %v1
; CHECK-NEXT:    vldeb %v2, %v2
; CHECK-NEXT:    vfchdb %v1, %v2, %v1
; CHECK-NEXT:    vmrhf %v2, %v27, %v27
; CHECK-NEXT:    vmrhf %v3, %v25, %v25
; CHECK-NEXT:    vldeb %v2, %v2
; CHECK-NEXT:    vldeb %v3, %v3
; CHECK-NEXT:    vfchdb %v2, %v3, %v2
; CHECK-NEXT:    vpkg %v1, %v2, %v1
; CHECK-NEXT:    vn %v0, %v0, %v1
; CHECK-NEXT:    vsel %v24, %v29, %v31, %v0
; CHECK-NEXT:    br %r14
;
; CHECK-Z14-LABEL: fun33:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchdb %v0, %v26, %v30
; CHECK-Z14-NEXT:    vfchdb %v1, %v24, %v28
; CHECK-Z14-NEXT:    vpkg %v0, %v1, %v0
; CHECK-Z14-NEXT:    vfchsb %v1, %v25, %v27
; CHECK-Z14-NEXT:    vn %v0, %v0, %v1
; CHECK-Z14-NEXT:    vsel %v24, %v29, %v31, %v0
; CHECK-Z14-NEXT:    br %r14
  %cmp0 = fcmp ogt <4 x double> %val1, %val2
  %cmp1 = fcmp ogt <4 x float> %val3, %val4
  %and = and <4 x i1> %cmp0, %cmp1
  %sel = select <4 x i1> %and, <4 x float> %val5, <4 x float> %val6
  ret <4 x float> %sel
}

define <4 x double> @fun34(<4 x double> %val1, <4 x double> %val2, <4 x float> %val3, <4 x float> %val4, <4 x double> %val5, <4 x double> %val6) {
; CHECK-LABEL: fun34:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmrlf [[REG0:%v[0-9]+]], %v27, %v27
; CHECK-NEXT:    vmrlf [[REG1:%v[0-9]+]], %v25, %v25
; CHECK-NEXT:    vldeb [[REG2:%v[0-9]+]], [[REG0]]
; CHECK-NEXT:    vldeb [[REG3:%v[0-9]+]], [[REG1]]
; CHECK-NEXT:    vfchdb [[REG4:%v[0-9]+]], [[REG3]], [[REG2]]
; CHECK-NEXT:    vmrhf [[REG5:%v[0-9]+]], %v27, %v27
; CHECK-NEXT:    vmrhf [[REG6:%v[0-9]+]], %v25, %v25
; CHECK-DAG:     vldeb [[REG7:%v[0-9]+]], [[REG5]]
; CHECK-DAG:     vl [[REG8:%v[0-9]+]], 176(%r15)
; CHECK-DAG:     vldeb [[REG9:%v[0-9]+]], [[REG6]]
; CHECK-DAG:     vl [[REG10:%v[0-9]+]], 160(%r15)
; CHECK-DAG:     vfchdb [[REG11:%v[0-9]+]], [[REG9]], [[REG7]]
; CHECK-DAG:     vpkg [[REG12:%v[0-9]+]], [[REG11]], [[REG4]]
; CHECK-DAG:     vuphf [[REG13:%v[0-9]+]], [[REG12]]
; CHECK-DAG:     vmrlg [[REG14:%v[0-9]+]], [[REG12]], [[REG12]]
; CHECK-NEXT:    vfchdb [[REG15:%v[0-9]+]], %v24, %v28
; CHECK-NEXT:    vfchdb [[REG16:%v[0-9]+]], %v26, %v30
; CHECK-NEXT:    vuphf [[REG17:%v[0-9]+]], [[REG14]]
; CHECK-NEXT:    vn [[REG18:%v[0-9]+]], [[REG16]], [[REG17]]
; CHECK-NEXT:    vn [[REG19:%v[0-9]+]], [[REG15]], [[REG13]]
; CHECK-NEXT:    vsel %v24, %v29, [[REG10]], [[REG19]]
; CHECK-NEXT:    vsel %v26, %v31, [[REG8]], [[REG18]]
; CHECK-NEXT:    br %r14
;
; CHECK-Z14-LABEL: fun34:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchsb %v4, %v25, %v27
; CHECK-Z14-NEXT:    vuphf %v5, %v4
; CHECK-Z14-NEXT:    vmrlg %v4, %v4, %v4
; CHECK-Z14-NEXT:    vfchdb %v2, %v24, %v28
; CHECK-Z14-NEXT:    vfchdb %v3, %v26, %v30
; CHECK-Z14-NEXT:    vuphf %v4, %v4
; CHECK-Z14-NEXT:    vl %v0, 176(%r15)
; CHECK-Z14-NEXT:    vl %v1, 160(%r15)
; CHECK-Z14-NEXT:    vn %v3, %v3, %v4
; CHECK-Z14-NEXT:    vn %v2, %v2, %v5
; CHECK-Z14-NEXT:    vsel %v24, %v29, %v1, %v2
; CHECK-Z14-NEXT:    vsel %v26, %v31, %v0, %v3
; CHECK-Z14-NEXT:    br %r14
  %cmp0 = fcmp ogt <4 x double> %val1, %val2
  %cmp1 = fcmp ogt <4 x float> %val3, %val4
  %and = and <4 x i1> %cmp0, %cmp1
  %sel = select <4 x i1> %and, <4 x double> %val5, <4 x double> %val6
  ret <4 x double> %sel
}
