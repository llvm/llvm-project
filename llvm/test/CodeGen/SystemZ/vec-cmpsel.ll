; Test that vector compare / select combinations do not produce any
; unnecessary pack /unpack / shift instructions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s -check-prefix=CHECK-Z14

define <2 x i8> @fun0(<2 x i8> %val1, <2 x i8> %val2, <2 x i8> %val3, <2 x i8> %val4) {
; CHECK-LABEL: fun0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqb %v0, %v24, %v26
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i8> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
  ret <2 x i8> %sel
}

define <2 x i16> @fun1(<2 x i8> %val1, <2 x i8> %val2, <2 x i16> %val3, <2 x i16> %val4) {
; CHECK-LABEL: fun1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqb %v0, %v24, %v26
; CHECK-NEXT:    vuphb %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i8> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
  ret <2 x i16> %sel
}

define <16 x i8> @fun2(<16 x i8> %val1, <16 x i8> %val2, <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: fun2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqb %v0, %v24, %v26
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <16 x i8> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %sel
}

define <16 x i16> @fun3(<16 x i8> %val1, <16 x i8> %val2, <16 x i16> %val3, <16 x i16> %val4) {
; CHECK-LABEL: fun3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqb %v0, %v24, %v26
; CHECK-DAG:     vuphb [[REG0:%v[0-9]+]], %v0
; CHECK-DAG:     vmrlg [[REG1:%v[0-9]+]], %v0, %v0
; CHECK-DAG:     vuphb [[REG1]], [[REG1]]
; CHECK-NEXT:    vsel %v24, %v28, %v25, [[REG0]]
; CHECK-NEXT:    vsel %v26, %v30, %v27, [[REG1]]
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <16 x i8> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
  ret <16 x i16> %sel
}

define <32 x i8> @fun4(<32 x i8> %val1, <32 x i8> %val2, <32 x i8> %val3, <32 x i8> %val4) {
; CHECK-LABEL: fun4:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqb [[REG0:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vceqb [[REG1:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vsel %v24, %v25, %v29, [[REG1]]
; CHECK-DAG:     vsel %v26, %v27, %v31, [[REG0]]
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <32 x i8> %val1, %val2
  %sel = select <32 x i1> %cmp, <32 x i8> %val3, <32 x i8> %val4
  ret <32 x i8> %sel
}

define <2 x i8> @fun5(<2 x i16> %val1, <2 x i16> %val2, <2 x i8> %val3, <2 x i8> %val4) {
; CHECK-LABEL: fun5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqh %v0, %v24, %v26
; CHECK-NEXT:    vpkh %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i16> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
  ret <2 x i8> %sel
}

define <2 x i16> @fun6(<2 x i16> %val1, <2 x i16> %val2, <2 x i16> %val3, <2 x i16> %val4) {
; CHECK-LABEL: fun6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqh %v0, %v24, %v26
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i16> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
  ret <2 x i16> %sel
}

define <2 x i32> @fun7(<2 x i16> %val1, <2 x i16> %val2, <2 x i32> %val3, <2 x i32> %val4) {
; CHECK-LABEL: fun7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqh %v0, %v24, %v26
; CHECK-NEXT:    vuphh %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i16> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
  ret <2 x i32> %sel
}

define <8 x i8> @fun8(<8 x i16> %val1, <8 x i16> %val2, <8 x i8> %val3, <8 x i8> %val4) {
; CHECK-LABEL: fun8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqh %v0, %v24, %v26
; CHECK-NEXT:    vpkh %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <8 x i16> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
  ret <8 x i8> %sel
}

define <8 x i16> @fun9(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: fun9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqh %v0, %v24, %v26
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <8 x i16> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %sel
}

define <8 x i32> @fun10(<8 x i16> %val1, <8 x i16> %val2, <8 x i32> %val3, <8 x i32> %val4) {
; CHECK-LABEL: fun10:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqh %v0, %v24, %v26
; CHECK-DAG:     vuphh [[REG0:%v[0-9]+]], %v0
; CHECK-DAG:     vmrlg [[REG1:%v[0-9]+]], %v0, %v0
; CHECK-DAG:     vuphh [[REG1]], [[REG1]]
; CHECK-NEXT:    vsel %v24, %v28, %v25, [[REG0]]
; CHECK-NEXT:    vsel %v26, %v30, %v27, [[REG1]]
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <8 x i16> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
  ret <8 x i32> %sel
}

define <16 x i8> @fun11(<16 x i16> %val1, <16 x i16> %val2, <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: fun11:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqh %v0, %v26, %v30
; CHECK-NEXT:    vceqh %v1, %v24, %v28
; CHECK-NEXT:    vpkh %v0, %v1, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <16 x i16> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %sel
}

define <16 x i16> @fun12(<16 x i16> %val1, <16 x i16> %val2, <16 x i16> %val3, <16 x i16> %val4) {
; CHECK-LABEL: fun12:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqh [[REG0:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vceqh [[REG1:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vsel %v24, %v25, %v29, [[REG1]]
; CHECK-DAG:     vsel %v26, %v27, %v31, [[REG0]]
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <16 x i16> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
  ret <16 x i16> %sel
}

define <2 x i16> @fun13(<2 x i32> %val1, <2 x i32> %val2, <2 x i16> %val3, <2 x i16> %val4) {
; CHECK-LABEL: fun13:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-NEXT:    vpkf %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i32> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
  ret <2 x i16> %sel
}

define <2 x i32> @fun14(<2 x i32> %val1, <2 x i32> %val2, <2 x i32> %val3, <2 x i32> %val4) {
; CHECK-LABEL: fun14:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i32> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
  ret <2 x i32> %sel
}

define <2 x i64> @fun15(<2 x i32> %val1, <2 x i32> %val2, <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: fun15:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-NEXT:    vuphf %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i32> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %sel
}

define <4 x i16> @fun16(<4 x i32> %val1, <4 x i32> %val2, <4 x i16> %val3, <4 x i16> %val4) {
; CHECK-LABEL: fun16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-NEXT:    vpkf %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <4 x i32> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
  ret <4 x i16> %sel
}

define <4 x i32> @fun17(<4 x i32> %val1, <4 x i32> %val2, <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: fun17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <4 x i32> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %sel
}

define <4 x i64> @fun18(<4 x i32> %val1, <4 x i32> %val2, <4 x i64> %val3, <4 x i64> %val4) {
; CHECK-LABEL: fun18:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqf %v0, %v24, %v26
; CHECK-DAG:     vuphf [[REG0:%v[0-9]+]], %v0
; CHECK-DAG:     vmrlg [[REG1:%v[0-9]+]], %v0, %v0
; CHECK-DAG:     vuphf [[REG1]], [[REG1]]
; CHECK-NEXT:    vsel %v24, %v28, %v25, [[REG0]]
; CHECK-NEXT:    vsel %v26, %v30, %v27, [[REG1]]
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <4 x i32> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
  ret <4 x i64> %sel
}

define <8 x i16> @fun19(<8 x i32> %val1, <8 x i32> %val2, <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: fun19:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqf %v0, %v26, %v30
; CHECK-NEXT:    vceqf %v1, %v24, %v28
; CHECK-NEXT:    vpkf %v0, %v1, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <8 x i32> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %sel
}

define <8 x i32> @fun20(<8 x i32> %val1, <8 x i32> %val2, <8 x i32> %val3, <8 x i32> %val4) {
; CHECK-LABEL: fun20:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqf [[REG0:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vceqf [[REG1:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vsel %v24, %v25, %v29, [[REG1]]
; CHECK-DAG:     vsel %v26, %v27, %v31, [[REG0]]
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <8 x i32> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
  ret <8 x i32> %sel
}

define <2 x i32> @fun21(<2 x i64> %val1, <2 x i64> %val2, <2 x i32> %val3, <2 x i32> %val4) {
; CHECK-LABEL: fun21:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqg %v0, %v24, %v26
; CHECK-NEXT:    vpkg %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i64> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
  ret <2 x i32> %sel
}

define <2 x i64> @fun22(<2 x i64> %val1, <2 x i64> %val2, <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: fun22:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqg %v0, %v24, %v26
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <2 x i64> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %sel
}

define <4 x i32> @fun23(<4 x i64> %val1, <4 x i64> %val2, <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: fun23:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqg %v0, %v26, %v30
; CHECK-NEXT:    vceqg %v1, %v24, %v28
; CHECK-NEXT:    vpkg %v0, %v1, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <4 x i64> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %sel
}

define <4 x i64> @fun24(<4 x i64> %val1, <4 x i64> %val2, <4 x i64> %val3, <4 x i64> %val4) {
; CHECK-LABEL: fun24:
; CHECK:       # %bb.0:
; CHECK-DAG:     vceqg [[REG0:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vceqg [[REG1:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vsel %v24, %v25, %v29, [[REG1]]
; CHECK-DAG:     vsel %v26, %v27, %v31, [[REG0]]
; CHECK-NEXT:    br %r14
  %cmp = icmp eq <4 x i64> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
  ret <4 x i64> %sel
}

define <2 x float> @fun25(<2 x float> %val1, <2 x float> %val2, <2 x float> %val3, <2 x float> %val4) {
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
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14

; CHECK-Z14-LABEL: fun25:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchsb  %v0, %v24, %v26
; CHECK-Z14-NEXT:    vsel    %v24, %v28, %v30, %v0
; CHECK-Z14-NEXT:    br %r14

  %cmp = fcmp ogt <2 x float> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
  ret <2 x float> %sel
}

define <2 x double> @fun26(<2 x float> %val1, <2 x float> %val2, <2 x double> %val3, <2 x double> %val4) {
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
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14

; CHECK-Z14-LABEL: fun26:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchsb  %v0, %v24, %v26
; CHECK-Z14-NEXT:    vuphf   %v0, %v0
; CHECK-Z14-NEXT:    vsel    %v24, %v28, %v30, %v0
; CHECK-Z14-NEXT:    br %r14

  %cmp = fcmp ogt <2 x float> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %sel
}

; Test a widening select of floats.
define <2 x float> @fun27(<2 x i8> %val1, <2 x i8> %val2, <2 x float> %val3, <2 x float> %val4) {
; CHECK-LABEL: fun27:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vceqb %v0, %v24, %v26
; CHECK-NEXT:    vuphb %v0, %v0
; CHECK-NEXT:    vuphh %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14

  %cmp = icmp eq <2 x i8> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
  ret <2 x float> %sel
}

define <4 x float> @fun28(<4 x float> %val1, <4 x float> %val2, <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: fun28:
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
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14

; CHECK-Z14-LABEL: fun28:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchsb  %v0, %v24, %v26
; CHECK-Z14-NEXT:    vsel    %v24, %v28, %v30, %v0
; CHECK-Z14-NEXT:    br %r14

  %cmp = fcmp ogt <4 x float> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %sel
}

define <4 x double> @fun29(<4 x float> %val1, <4 x float> %val2, <4 x double> %val3, <4 x double> %val4) {
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
; CHECK-NEXT:    vldeb %v2, %v2
; CHECK-NEXT:    vfchdb %v1, %v2, %v1
; CHECK-NEXT:    vpkg [[REG0:%v[0-9]+]], %v1, %v0
; CHECK-DAG:     vmrlg [[REG1:%v[0-9]+]], [[REG0]], [[REG0]]
; CHECK-DAG:     vuphf [[REG1]], [[REG1]]
; CHECK-DAG:     vuphf [[REG2:%v[0-9]+]], [[REG0]]
; CHECK-NEXT:    vsel %v24, %v28, %v25, [[REG2]]
; CHECK-NEXT:    vsel %v26, %v30, %v27, [[REG1]]
; CHECK-NEXT:    br %r14

; CHECK-Z14-LABEL: fun29:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-NEXT:    vfchsb  %v0, %v24, %v26
; CHECK-Z14-DAG:     vuphf   [[REG0:%v[0-9]+]], %v0
; CHECK-Z14-DAG:     vmrlg   [[REG1:%v[0-9]+]], %v0, %v0
; CHECK-Z14-DAG:     vuphf   [[REG1]], [[REG1]]
; CHECK-Z14-NEXT:    vsel    %v24, %v28, %v25, [[REG0]]
; CHECK-Z14-NEXT:    vsel    %v26, %v30, %v27, [[REG1]]
; CHECK-Z14-NEXT:    br %r14

  %cmp = fcmp ogt <4 x float> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
  ret <4 x double> %sel
}

define <8 x float> @fun30(<8 x float> %val1, <8 x float> %val2, <8 x float> %val3, <8 x float> %val4) {
; CHECK-Z14-LABEL: fun30:
; CHECK-Z14:       # %bb.0:
; CHECK-Z14-DAG:     vfchsb  [[REG0:%v[0-9]+]], %v26, %v30
; CHECK-Z14-DAG:     vfchsb  [[REG1:%v[0-9]+]], %v24, %v28
; CHECK-Z14-DAG:     vsel    %v24, %v25, %v29, [[REG1]]
; CHECK-Z14-DAG:     vsel    %v26, %v27, %v31, [[REG0]]
; CHECK-Z14-NEXT:    br %r14
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
  ret <8 x float> %sel
}

define <2 x float> @fun31(<2 x double> %val1, <2 x double> %val2, <2 x float> %val3, <2 x float> %val4) {
; CHECK-LABEL: fun31:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vfchdb %v0, %v24, %v26
; CHECK-NEXT:    vpkg %v0, %v0, %v0
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14

  %cmp = fcmp ogt <2 x double> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
  ret <2 x float> %sel
}

define <2 x double> @fun32(<2 x double> %val1, <2 x double> %val2, <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: fun32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vfchdb %v0, %v24, %v26
; CHECK-NEXT:    vsel %v24, %v28, %v30, %v0
; CHECK-NEXT:    br %r14
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %sel
}

define <4 x float> @fun33(<4 x double> %val1, <4 x double> %val2, <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: fun33:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vfchdb %v0, %v26, %v30
; CHECK-NEXT:    vfchdb %v1, %v24, %v28
; CHECK-NEXT:    vpkg %v0, %v1, %v0
; CHECK-NEXT:    vsel %v24, %v25, %v27, %v0
; CHECK-NEXT:    br %r14
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %sel
}

define <4 x double> @fun34(<4 x double> %val1, <4 x double> %val2, <4 x double> %val3, <4 x double> %val4) {
; CHECK-LABEL: fun34:
; CHECK:       # %bb.0:
; CHECK-DAG:     vfchdb [[REG0:%v[0-9]+]], %v26, %v30
; CHECK-DAG:     vfchdb [[REG1:%v[0-9]+]], %v24, %v28
; CHECK-DAG:     vsel %v24, %v25, %v29, [[REG1]]
; CHECK-DAG:     vsel %v26, %v27, %v31, [[REG0]]
; CHECK-NEXT:    br %r14
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
  ret <4 x double> %sel
}
