; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s  | FileCheck %s

; Store a <4 x i31> vector.
define void @fun0(<4 x i31> %src, <4 x i31>* %p)
; CHECK-LABEL: fun0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vlgvf %r1, %v24, 0
; CHECK-NEXT:    vlgvf %r0, %v24, 1
; CHECK-NEXT:    sllg %r1, %r1, 29
; CHECK-NEXT:    rosbg %r1, %r0, 35, 63, 62
; CHECK-NEXT:    nihh %r1, 4095
; CHECK-NEXT:    stg %r1, 0(%r2)
; CHECK-NEXT:    vlgvf %r1, %v24, 2
; CHECK-NEXT:    risbgn %r0, %r0, 0, 129, 62
; CHECK-NEXT:    rosbg %r0, %r1, 2, 32, 31
; CHECK-NEXT:    vlgvf %r1, %v24, 3
; CHECK-NEXT:    rosbg %r0, %r1, 33, 63, 0
; CHECK-NEXT:    stg %r0, 8(%r2)
; CHECK-NEXT:    br %r14
{
  store <4 x i31> %src, <4 x i31>* %p
  ret void
}

; Store a <16 x i1> vector.
define i16 @fun1(<16 x i1> %src)
; CHECK-LABEL: fun1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    aghi %r15, -168
; CHECK-NEXT:    .cfi_def_cfa_offset 328
; CHECK-NEXT:    vlgvb %r0, %v24, 0
; CHECK-NEXT:    vlgvb %r1, %v24, 1
; CHECK-NEXT:    risblg %r0, %r0, 16, 144, 15
; CHECK-NEXT:    rosbg %r0, %r1, 49, 49, 14
; CHECK-NEXT:    vlgvb %r1, %v24, 2
; CHECK-NEXT:    rosbg %r0, %r1, 50, 50, 13
; CHECK-NEXT:    vlgvb %r1, %v24, 3
; CHECK-NEXT:    rosbg %r0, %r1, 51, 51, 12
; CHECK-NEXT:    vlgvb %r1, %v24, 4
; CHECK-NEXT:    rosbg %r0, %r1, 52, 52, 11
; CHECK-NEXT:    vlgvb %r1, %v24, 5
; CHECK-NEXT:    rosbg %r0, %r1, 53, 53, 10
; CHECK-NEXT:    vlgvb %r1, %v24, 6
; CHECK-NEXT:    rosbg %r0, %r1, 54, 54, 9
; CHECK-NEXT:    vlgvb %r1, %v24, 7
; CHECK-NEXT:    rosbg %r0, %r1, 55, 55, 8
; CHECK-NEXT:    vlgvb %r1, %v24, 8
; CHECK-NEXT:    rosbg %r0, %r1, 56, 56, 7
; CHECK-NEXT:    vlgvb %r1, %v24, 9
; CHECK-NEXT:    rosbg %r0, %r1, 57, 57, 6
; CHECK-NEXT:    vlgvb %r1, %v24, 10
; CHECK-NEXT:    rosbg %r0, %r1, 58, 58, 5
; CHECK-NEXT:    vlgvb %r1, %v24, 11
; CHECK-NEXT:    rosbg %r0, %r1, 59, 59, 4
; CHECK-NEXT:    vlgvb %r1, %v24, 12
; CHECK-NEXT:    rosbg %r0, %r1, 60, 60, 3
; CHECK-NEXT:    vlgvb %r1, %v24, 13
; CHECK-NEXT:    rosbg %r0, %r1, 61, 61, 2
; CHECK-NEXT:    vlgvb %r1, %v24, 14
; CHECK-NEXT:    rosbg %r0, %r1, 62, 62, 1
; CHECK-NEXT:    vlgvb %r1, %v24, 15
; CHECK-NEXT:    rosbg %r0, %r1, 63, 63, 0
; CHECK-NEXT:    llhr %r2, %r0
; CHECK-NEXT:    aghi %r15, 168
; CHECK-NEXT:    br %r14
{
  %res = bitcast <16 x i1> %src to i16
  ret i16 %res
}

; Truncate a <8 x i32> vector to <8 x i31> and store it (test splitting).
define void @fun2(<8 x i32> %src, <8 x i31>* %p)
; CHECK-LABEL: fun2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stmg %r14, %r15, 112(%r15)
; CHECK-NEXT:    .cfi_offset %r14, -48
; CHECK-NEXT:    .cfi_offset %r15, -40
; CHECK-NEXT:    vlgvf %r3, %v26, 1
; CHECK-NEXT:    vlgvf %r1, %v26, 2
; CHECK-NEXT:    risbgn %r4, %r3, 0, 129, 62
; CHECK-NEXT:    rosbg %r4, %r1, 2, 32, 31
; CHECK-DAG:     vlgvf %r0, %v26, 3
; CHECK-DAG:     rosbg %r4, %r0, 33, 63, 0
; CHECK-DAG:     stc %r0, 30(%r2)
; CHECK-DAG:     srl %r0, 8
; CHECK-DAG:     vlgvf [[REG0:%r[0-9]+]], %v24, 1
; CHECK-DAG:     vlgvf [[REG1:%r[0-9]+]], %v24, 0
; CHECK-DAG:     sth %r0, 28(%r2)
; CHECK-DAG:     vlgvf [[REG2:%r[0-9]+]], %v24, 2
; CHECK-DAG:     risbgn [[REG3:%r[0-9]+]], [[REG0]], 0, 133, 58
; CHECK-DAG:     rosbg [[REG3]], [[REG2]], 6, 36, 27
; CHECK-DAG:     sllg [[REG4:%r[0-9]+]], [[REG1]], 25
; CHECK-DAG:     rosbg [[REG4]], [[REG0]], 39, 63, 58
; CHECK-DAG:     vlgvf [[REG5:%r[0-9]+]], %v24, 3
; CHECK-DAG:     rosbg [[REG3]], [[REG5]], 37, 63, 60
; CHECK-DAG:     sllg [[REG6:%r[0-9]+]], [[REG4]], 8
; CHECK-DAG:     rosbg [[REG6]], [[REG3]], 56, 63, 8
; CHECK-NEXT:    stg [[REG6]], 0(%r2)
; CHECK-NEXT:    srlg [[REG7:%r[0-9]+]], %r4, 24
; CHECK-NEXT:    st [[REG7]], 24(%r2)
; CHECK-NEXT:    vlgvf [[REG8:%r[0-9]+]], %v26, 0
; CHECK-NEXT:    risbgn [[REG10:%r[0-9]+]], [[REG5]], 0, 131, 60
; CHECK-NEXT:    rosbg [[REG10]], [[REG8]], 4, 34, 29
; CHECK-NEXT:    sllg [[REG9:%r[0-9]+]], [[REG3]], 8
; CHECK-NEXT:    rosbg [[REG10]], %r3, 35, 63, 62
; CHECK-NEXT:    rosbg [[REG9]], [[REG10]], 56, 63, 8
; CHECK-NEXT:    stg [[REG9]], 8(%r2)
; CHECK-NEXT:    sllg %r0, [[REG10]], 8
; CHECK-NEXT:    rosbg %r0, %r4, 56, 63, 8
; CHECK-NEXT:    stg %r0, 16(%r2)
; CHECK-NEXT:    lmg %r14, %r15, 112(%r15)
; CHECK-NEXT:    br %r14
{
  %tmp = trunc <8 x i32> %src to <8 x i31>
  store <8 x i31> %tmp, <8 x i31>* %p
  ret void
}

; Load and store a <3 x i31> vector (test widening).
define void @fun3(<3 x i31>* %src, <3 x i31>* %p)
; CHECK-LABEL: fun3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    llgf %r1, 0(%r2)
; CHECK-NEXT:    llgf %r0, 3(%r2)
; CHECK-NEXT:    sllg %r4, %r1, 62
; CHECK-NEXT:    rosbg %r4, %r0, 0, 32, 31
; CHECK-NEXT:    llgf %r0, 6(%r2)
; CHECK-NEXT:    ogr %r0, %r4
; CHECK-NEXT:    st %r0, 8(%r3)
; CHECK-NEXT:    srlg %r0, %r4, 32
; CHECK-NEXT:    sllg %r1, %r1, 30
; CHECK-NEXT:    lr %r1, %r0
; CHECK-NEXT:    nihh %r1, 8191
; CHECK-NEXT:    stg %r1, 0(%r3)
; CHECK-NEXT:    br %r14
{
  %tmp = load <3 x i31>, <3 x i31>* %src
  store <3 x i31> %tmp, <3 x i31>* %p
  ret void
}
