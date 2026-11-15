; RUN: not llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx < %s 2>&1 | FileCheck %s

; Unsupported operation/type combinations tested:
; v2048i16 - abds, abdu, avgceils, avgceilu, avgfloors, avgflooru, mul
; v2048i8 - setcc
; v2048i8, v4096i8, v512i32 - abds, abdu, abs, add, avgceils, avgceilu,
;                                 avgfloors, avgflooru, mul, saddsat, shl,
;                                 smax, smin, splat_vector, sra, srl, ssubsat,
;                                 sub, uaddsat, umax, umin, usubsat, vselect

; CHECK-DAG: HLX does not support 'add' on vector type v4096i8
define void @add_v4096i8(ptr %in0, ptr %in1, ptr %out) #0 {
  %a = load <4096 x i8>, ptr %in0, align 4096
  %b = load <4096 x i8>, ptr %in1, align 4096
  %r = add <4096 x i8> %a, %b
  store <4096 x i8> %r, ptr %out, align 4096
  ret void
}

; CHECK-DAG: HLX does not support 'splat_vector' on vector type v2048i8
define void @splat_v2048i8(i8 %v, ptr %out) #0 {
  %ins = insertelement <2048 x i8> poison, i8 %v, i64 0
  %r = shufflevector <2048 x i8> %ins, <2048 x i8> poison,
                     <2048 x i32> zeroinitializer
  store <2048 x i8> %r, ptr %out, align 2048
  ret void
}

; CHECK-DAG: HLX does not support 'setcc' on vector type v2048i8; use an element type and vector length supported by the HLX ISA
define void @setcc_v2048i8(ptr %in0, ptr %in1, ptr %out) #0 {
  %a = load <2048 x i8>, ptr %in0, align 2048
  %b = load <2048 x i8>, ptr %in1, align 2048
  %p = icmp sgt <2048 x i8> %a, %b
  %q = freeze <2048 x i1> %p
  %r = select <2048 x i1> %q, <2048 x i8> %a, <2048 x i8> %b
  store <2048 x i8> %r, ptr %out, align 2048
  ret void
}

; CHECK-DAG: HLX does not support 'add' on vector type v512i32
define <512 x i32> @add_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {
  %r = add <512 x i32> %a, %b
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'sub' on vector type v512i32
define <512 x i32> @sub_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {
  %r = sub <512 x i32> %a, %b
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'shl' on vector type v512i32
define <512 x i32> @shl_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {
  %r = shl <512 x i32> %a, %b
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'srl' on vector type v512i32
define <512 x i32> @lshr_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {
  %r = lshr <512 x i32> %a, %b
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'sra' on vector type v512i32
define <512 x i32> @ashr_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {
  %r = ashr <512 x i32> %a, %b
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'splat_vector' on vector type v512i32
define void @splat_v512i32(i32 %v, ptr %out) #0 {
  %ins = insertelement <512 x i32> poison, i32 %v, i64 0
  %r = shufflevector <512 x i32> %ins, <512 x i32> poison,
                     <512 x i32> zeroinitializer
  store <512 x i32> %r, ptr %out, align 2048
  ret void
}

; CHECK-DAG: HLX does not support 'vselect' on vector type v512i32
define void @vselect_v512i32(ptr %incond, ptr %in0, ptr %in1, ptr %out) #0 {
  %cond = load <512 x i1>, ptr %incond, align 64
  %a = load <512 x i32>, ptr %in0, align 2048
  %b = load <512 x i32>, ptr %in1, align 2048
  %r = select <512 x i1> %cond, <512 x i32> %a, <512 x i32> %b
  store <512 x i32> %r, ptr %out, align 2048
  ret void
}

; CHECK-DAG: HLX does not support 'mul' on vector type v2048i16
define <2048 x i16> @mul_v2048i16(<2048 x i16> %a, <2048 x i16> %b) #0 {
  %r = mul <2048 x i16> %a, %b
  ret <2048 x i16> %r
}

; CHECK-DAG: HLX does not support 'abdu' on vector type v2048i16
define <2048 x i16> @abdu_v2048i16(<2048 x i16> %a, <2048 x i16> %b) #0 {
  %s1 = sub <2048 x i16> %a, %b
  %s2 = sub <2048 x i16> %b, %a
  %c = icmp ugt <2048 x i16> %a, %b
  %r = select <2048 x i1> %c, <2048 x i16> %s1, <2048 x i16> %s2
  ret <2048 x i16> %r
}

; CHECK-DAG: HLX does not support 'abds' on vector type v2048i16
define <2048 x i16> @abds_v2048i16(<2048 x i16> %a, <2048 x i16> %b) #0 {
  %s1 = sub <2048 x i16> %a, %b
  %s2 = sub <2048 x i16> %b, %a
  %c = icmp sgt <2048 x i16> %a, %b
  %r = select <2048 x i1> %c, <2048 x i16> %s1, <2048 x i16> %s2
  ret <2048 x i16> %r
}

; CHECK-DAG: HLX does not support 'uaddsat' on vector type v2048i8
define <2048 x i8> @uaddsat_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {
  %r = call <2048 x i8> @llvm.uadd.sat.v2048i8(<2048 x i8> %a,
                                                <2048 x i8> %b)
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'saddsat' on vector type v2048i8
define <2048 x i8> @saddsat_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {
  %r = call <2048 x i8> @llvm.sadd.sat.v2048i8(<2048 x i8> %a,
                                                <2048 x i8> %b)
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'usubsat' on vector type v2048i8
define <2048 x i8> @usubsat_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {
  %r = call <2048 x i8> @llvm.usub.sat.v2048i8(<2048 x i8> %a,
                                                <2048 x i8> %b)
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'ssubsat' on vector type v2048i8
define <2048 x i8> @ssubsat_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {
  %r = call <2048 x i8> @llvm.ssub.sat.v2048i8(<2048 x i8> %a,
                                                <2048 x i8> %b)
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'smin' on vector type v2048i8
define <2048 x i8> @smin_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {
  %r = call <2048 x i8> @llvm.smin.v2048i8(<2048 x i8> %a, <2048 x i8> %b)
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'smax' on vector type v2048i8
define <2048 x i8> @smax_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {
  %r = call <2048 x i8> @llvm.smax.v2048i8(<2048 x i8> %a, <2048 x i8> %b)
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'umin' on vector type v2048i8
define <2048 x i8> @umin_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {
  %r = call <2048 x i8> @llvm.umin.v2048i8(<2048 x i8> %a, <2048 x i8> %b)
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'umax' on vector type v2048i8
define <2048 x i8> @umax_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {
  %r = call <2048 x i8> @llvm.umax.v2048i8(<2048 x i8> %a, <2048 x i8> %b)
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'abs' on vector type v2048i8
define <2048 x i8> @abs_v2048i8(<2048 x i8> %a) #0 {
  %r = call <2048 x i8> @llvm.abs.v2048i8(<2048 x i8> %a, i1 false)
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'avgfloors' on vector type v2048i16
define <2048 x i16> @avgfloors_v2048i16(<2048 x i16> %a,
                                         <2048 x i16> %b) #0 {
  %and = and <2048 x i16> %a, %b
  %xor = xor <2048 x i16> %a, %b
  %half = ashr <2048 x i16> %xor, splat (i16 1)
  %r = add <2048 x i16> %and, %half
  ret <2048 x i16> %r
}

; CHECK-DAG: HLX does not support 'avgflooru' on vector type v2048i16
define <2048 x i16> @avgflooru_v2048i16(<2048 x i16> %a,
                                         <2048 x i16> %b) #0 {
  %and = and <2048 x i16> %a, %b
  %xor = xor <2048 x i16> %a, %b
  %half = lshr <2048 x i16> %xor, splat (i16 1)
  %r = add <2048 x i16> %and, %half
  ret <2048 x i16> %r
}

; CHECK-DAG: HLX does not support 'avgceils' on vector type v2048i16
define <2048 x i16> @avgceils_v2048i16(<2048 x i16> %a,
                                       <2048 x i16> %b) #0 {
  %or = or <2048 x i16> %a, %b
  %xor = xor <2048 x i16> %a, %b
  %half = ashr <2048 x i16> %xor, splat (i16 1)
  %r = sub <2048 x i16> %or, %half
  ret <2048 x i16> %r
}

; CHECK-DAG: HLX does not support 'avgceilu' on vector type v2048i16
define <2048 x i16> @avgceilu_v2048i16(<2048 x i16> %a,
                                       <2048 x i16> %b) #0 {
  %or = or <2048 x i16> %a, %b
  %xor = xor <2048 x i16> %a, %b
  %half = lshr <2048 x i16> %xor, splat (i16 1)
  %r = sub <2048 x i16> %or, %half
  ret <2048 x i16> %r
}

declare <2048 x i8> @llvm.uadd.sat.v2048i8(<2048 x i8>, <2048 x i8>)
declare <2048 x i8> @llvm.sadd.sat.v2048i8(<2048 x i8>, <2048 x i8>)
declare <2048 x i8> @llvm.usub.sat.v2048i8(<2048 x i8>, <2048 x i8>)
declare <2048 x i8> @llvm.ssub.sat.v2048i8(<2048 x i8>, <2048 x i8>)
declare <2048 x i8> @llvm.smin.v2048i8(<2048 x i8>, <2048 x i8>)
declare <2048 x i8> @llvm.smax.v2048i8(<2048 x i8>, <2048 x i8>)
declare <2048 x i8> @llvm.umin.v2048i8(<2048 x i8>, <2048 x i8>)
declare <2048 x i8> @llvm.umax.v2048i8(<2048 x i8>, <2048 x i8>)
declare <2048 x i8> @llvm.abs.v2048i8(<2048 x i8>, i1)

; CHECK-DAG: HLX does not support 'add' on vector type v2048i8
define <2048 x i8> @add_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %r = add <2048 x i8> %a, %b
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'sub' on vector type v2048i8
define <2048 x i8> @sub_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %r = sub <2048 x i8> %a, %b
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'sub' on vector type v4096i8
define <4096 x i8> @sub_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = sub <4096 x i8> %a, %b
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'uaddsat' on vector type v4096i8
define <4096 x i8> @uaddsat_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = call <4096 x i8> @llvm.uadd.sat.v4096i8(<4096 x i8> %a,
                                                <4096 x i8> %b)
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'uaddsat' on vector type v512i32
define <512 x i32> @uaddsat_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %r = call <512 x i32> @llvm.uadd.sat.v512i32(<512 x i32> %a,
                                                <512 x i32> %b)
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'saddsat' on vector type v4096i8
define <4096 x i8> @saddsat_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = call <4096 x i8> @llvm.sadd.sat.v4096i8(<4096 x i8> %a,
                                                <4096 x i8> %b)
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'saddsat' on vector type v512i32
define <512 x i32> @saddsat_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %r = call <512 x i32> @llvm.sadd.sat.v512i32(<512 x i32> %a,
                                                <512 x i32> %b)
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'usubsat' on vector type v4096i8
define <4096 x i8> @usubsat_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = call <4096 x i8> @llvm.usub.sat.v4096i8(<4096 x i8> %a,
                                                <4096 x i8> %b)
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'usubsat' on vector type v512i32
define <512 x i32> @usubsat_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %r = call <512 x i32> @llvm.usub.sat.v512i32(<512 x i32> %a,
                                                <512 x i32> %b)
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'ssubsat' on vector type v4096i8
define <4096 x i8> @ssubsat_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = call <4096 x i8> @llvm.ssub.sat.v4096i8(<4096 x i8> %a,
                                                <4096 x i8> %b)
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'ssubsat' on vector type v512i32
define <512 x i32> @ssubsat_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %r = call <512 x i32> @llvm.ssub.sat.v512i32(<512 x i32> %a,
                                                <512 x i32> %b)
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'mul' on vector type v2048i8
define <2048 x i8> @mul_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %r = mul <2048 x i8> %a, %b
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'mul' on vector type v4096i8
define <4096 x i8> @mul_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = mul <4096 x i8> %a, %b
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'mul' on vector type v512i32
define <512 x i32> @mul_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %r = mul <512 x i32> %a, %b
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'shl' on vector type v2048i8
define <2048 x i8> @shl_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %r = shl <2048 x i8> %a, %b
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'shl' on vector type v4096i8
define <4096 x i8> @shl_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = shl <4096 x i8> %a, %b
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'srl' on vector type v2048i8
define <2048 x i8> @lshr_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %r = lshr <2048 x i8> %a, %b
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'srl' on vector type v4096i8
define <4096 x i8> @lshr_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = lshr <4096 x i8> %a, %b
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'sra' on vector type v2048i8
define <2048 x i8> @ashr_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %r = ashr <2048 x i8> %a, %b
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'sra' on vector type v4096i8
define <4096 x i8> @ashr_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = ashr <4096 x i8> %a, %b
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'splat_vector' on vector type v4096i8
define void @splat_v4096i8(i8 %v, ptr %out) #0 {

  %ins = insertelement <4096 x i8> poison, i8 %v, i64 0
  %r = shufflevector <4096 x i8> %ins, <4096 x i8> poison,
                     <4096 x i32> zeroinitializer
  store <4096 x i8> %r, ptr %out, align 4096
  ret void
}

; CHECK-DAG: HLX does not support 'vselect' on vector type v2048i8
define void @vselect_v2048i8(ptr %incond, ptr %in0, ptr %in1, ptr %out) #0 {

  %cond = load <2048 x i1>, ptr %incond, align 256
  %a = load <2048 x i8>, ptr %in0, align 2048
  %b = load <2048 x i8>, ptr %in1, align 2048
  %r = select <2048 x i1> %cond, <2048 x i8> %a, <2048 x i8> %b
  store <2048 x i8> %r, ptr %out, align 2048
  ret void
}

; CHECK-DAG: HLX does not support 'vselect' on vector type v4096i8
define void @vselect_v4096i8(ptr %incond, ptr %in0, ptr %in1, ptr %out) #0 {

  %cond = load <4096 x i1>, ptr %incond, align 512
  %a = load <4096 x i8>, ptr %in0, align 4096
  %b = load <4096 x i8>, ptr %in1, align 4096
  %r = select <4096 x i1> %cond, <4096 x i8> %a, <4096 x i8> %b
  store <4096 x i8> %r, ptr %out, align 4096
  ret void
}

; CHECK-DAG: HLX does not support 'smin' on vector type v4096i8
define <4096 x i8> @smin_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = call <4096 x i8> @llvm.smin.v4096i8(<4096 x i8> %a, <4096 x i8> %b)
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'smin' on vector type v512i32
define <512 x i32> @smin_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %r = call <512 x i32> @llvm.smin.v512i32(<512 x i32> %a, <512 x i32> %b)
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'smax' on vector type v4096i8
define <4096 x i8> @smax_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = call <4096 x i8> @llvm.smax.v4096i8(<4096 x i8> %a, <4096 x i8> %b)
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'smax' on vector type v512i32
define <512 x i32> @smax_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %r = call <512 x i32> @llvm.smax.v512i32(<512 x i32> %a, <512 x i32> %b)
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'umin' on vector type v4096i8
define <4096 x i8> @umin_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = call <4096 x i8> @llvm.umin.v4096i8(<4096 x i8> %a, <4096 x i8> %b)
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'umin' on vector type v512i32
define <512 x i32> @umin_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %r = call <512 x i32> @llvm.umin.v512i32(<512 x i32> %a, <512 x i32> %b)
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'umax' on vector type v4096i8
define <4096 x i8> @umax_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %r = call <4096 x i8> @llvm.umax.v4096i8(<4096 x i8> %a, <4096 x i8> %b)
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'umax' on vector type v512i32
define <512 x i32> @umax_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %r = call <512 x i32> @llvm.umax.v512i32(<512 x i32> %a, <512 x i32> %b)
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'abs' on vector type v4096i8
define <4096 x i8> @abs_v4096i8(<4096 x i8> %a) #0 {

  %r = call <4096 x i8> @llvm.abs.v4096i8(<4096 x i8> %a, i1 false)
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'abs' on vector type v512i32
define <512 x i32> @abs_v512i32(<512 x i32> %a) #0 {

  %r = call <512 x i32> @llvm.abs.v512i32(<512 x i32> %a, i1 false)
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'abds' on vector type v2048i8
define <2048 x i8> @abds_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %s1 = sub <2048 x i8> %a, %b
  %s2 = sub <2048 x i8> %b, %a
  %c = icmp sgt <2048 x i8> %a, %b
  %r = select <2048 x i1> %c, <2048 x i8> %s1, <2048 x i8> %s2
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'abds' on vector type v4096i8
define <4096 x i8> @abds_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %s1 = sub <4096 x i8> %a, %b
  %s2 = sub <4096 x i8> %b, %a
  %c = icmp sgt <4096 x i8> %a, %b
  %r = select <4096 x i1> %c, <4096 x i8> %s1, <4096 x i8> %s2
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'abds' on vector type v512i32
define <512 x i32> @abds_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %s1 = sub <512 x i32> %a, %b
  %s2 = sub <512 x i32> %b, %a
  %c = icmp sgt <512 x i32> %a, %b
  %r = select <512 x i1> %c, <512 x i32> %s1, <512 x i32> %s2
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'abdu' on vector type v2048i8
define <2048 x i8> @abdu_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %s1 = sub <2048 x i8> %a, %b
  %s2 = sub <2048 x i8> %b, %a
  %c = icmp ugt <2048 x i8> %a, %b
  %r = select <2048 x i1> %c, <2048 x i8> %s1, <2048 x i8> %s2
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'abdu' on vector type v4096i8
define <4096 x i8> @abdu_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %s1 = sub <4096 x i8> %a, %b
  %s2 = sub <4096 x i8> %b, %a
  %c = icmp ugt <4096 x i8> %a, %b
  %r = select <4096 x i1> %c, <4096 x i8> %s1, <4096 x i8> %s2
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'abdu' on vector type v512i32
define <512 x i32> @abdu_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %s1 = sub <512 x i32> %a, %b
  %s2 = sub <512 x i32> %b, %a
  %c = icmp ugt <512 x i32> %a, %b
  %r = select <512 x i1> %c, <512 x i32> %s1, <512 x i32> %s2
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'avgfloors' on vector type v2048i8
define <2048 x i8> @avgfloors_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %first = and <2048 x i8> %a, %b
  %xor = xor <2048 x i8> %a, %b
  %half = ashr <2048 x i8> %xor, splat (i8 1)
  %r = add <2048 x i8> %first, %half
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'avgfloors' on vector type v4096i8
define <4096 x i8> @avgfloors_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %first = and <4096 x i8> %a, %b
  %xor = xor <4096 x i8> %a, %b
  %half = ashr <4096 x i8> %xor, splat (i8 1)
  %r = add <4096 x i8> %first, %half
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'avgfloors' on vector type v512i32
define <512 x i32> @avgfloors_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %first = and <512 x i32> %a, %b
  %xor = xor <512 x i32> %a, %b
  %half = ashr <512 x i32> %xor, splat (i32 1)
  %r = add <512 x i32> %first, %half
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'avgflooru' on vector type v2048i8
define <2048 x i8> @avgflooru_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %first = and <2048 x i8> %a, %b
  %xor = xor <2048 x i8> %a, %b
  %half = lshr <2048 x i8> %xor, splat (i8 1)
  %r = add <2048 x i8> %first, %half
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'avgflooru' on vector type v4096i8
define <4096 x i8> @avgflooru_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %first = and <4096 x i8> %a, %b
  %xor = xor <4096 x i8> %a, %b
  %half = lshr <4096 x i8> %xor, splat (i8 1)
  %r = add <4096 x i8> %first, %half
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'avgflooru' on vector type v512i32
define <512 x i32> @avgflooru_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %first = and <512 x i32> %a, %b
  %xor = xor <512 x i32> %a, %b
  %half = lshr <512 x i32> %xor, splat (i32 1)
  %r = add <512 x i32> %first, %half
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'avgceils' on vector type v2048i8
define <2048 x i8> @avgceils_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %first = or <2048 x i8> %a, %b
  %xor = xor <2048 x i8> %a, %b
  %half = ashr <2048 x i8> %xor, splat (i8 1)
  %r = sub <2048 x i8> %first, %half
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'avgceils' on vector type v4096i8
define <4096 x i8> @avgceils_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %first = or <4096 x i8> %a, %b
  %xor = xor <4096 x i8> %a, %b
  %half = ashr <4096 x i8> %xor, splat (i8 1)
  %r = sub <4096 x i8> %first, %half
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'avgceils' on vector type v512i32
define <512 x i32> @avgceils_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %first = or <512 x i32> %a, %b
  %xor = xor <512 x i32> %a, %b
  %half = ashr <512 x i32> %xor, splat (i32 1)
  %r = sub <512 x i32> %first, %half
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'avgceilu' on vector type v2048i8
define <2048 x i8> @avgceilu_v2048i8(<2048 x i8> %a, <2048 x i8> %b) #0 {

  %first = or <2048 x i8> %a, %b
  %xor = xor <2048 x i8> %a, %b
  %half = lshr <2048 x i8> %xor, splat (i8 1)
  %r = sub <2048 x i8> %first, %half
  ret <2048 x i8> %r
}

; CHECK-DAG: HLX does not support 'avgceilu' on vector type v4096i8
define <4096 x i8> @avgceilu_v4096i8(<4096 x i8> %a, <4096 x i8> %b) #0 {

  %first = or <4096 x i8> %a, %b
  %xor = xor <4096 x i8> %a, %b
  %half = lshr <4096 x i8> %xor, splat (i8 1)
  %r = sub <4096 x i8> %first, %half
  ret <4096 x i8> %r
}

; CHECK-DAG: HLX does not support 'avgceilu' on vector type v512i32
define <512 x i32> @avgceilu_v512i32(<512 x i32> %a, <512 x i32> %b) #0 {

  %first = or <512 x i32> %a, %b
  %xor = xor <512 x i32> %a, %b
  %half = lshr <512 x i32> %xor, splat (i32 1)
  %r = sub <512 x i32> %first, %half
  ret <512 x i32> %r
}

; CHECK-DAG: HLX does not support 'vselect' on vector type v4096i8; use an element type and vector length supported by the HLX ISA
define void @setcc_v4096i8(ptr %in0, ptr %in1, ptr %out) #0 {
  %a = load <4096 x i8>, ptr %in0, align 4096
  %b = load <4096 x i8>, ptr %in1, align 4096
  %p = icmp sgt <4096 x i8> %a, %b
  %q = freeze <4096 x i1> %p
  %r = select <4096 x i1> %q, <4096 x i8> %a, <4096 x i8> %b
  store <4096 x i8> %r, ptr %out, align 4096
  ret void
}

; CHECK-DAG: HLX does not support 'vselect' on vector type v512i32; use an element type and vector length supported by the HLX ISA
define void @setcc_v512i32(ptr %in0, ptr %in1, ptr %out) #0 {
  %a = load <512 x i32>, ptr %in0, align 2048
  %b = load <512 x i32>, ptr %in1, align 2048
  %p = icmp sgt <512 x i32> %a, %b
  %q = freeze <512 x i1> %p
  %r = select <512 x i1> %q, <512 x i32> %a, <512 x i32> %b
  store <512 x i32> %r, ptr %out, align 2048
  ret void
}

declare <4096 x i8> @llvm.uadd.sat.v4096i8(<4096 x i8>, <4096 x i8>)

declare <4096 x i8> @llvm.sadd.sat.v4096i8(<4096 x i8>, <4096 x i8>)

declare <4096 x i8> @llvm.usub.sat.v4096i8(<4096 x i8>, <4096 x i8>)

declare <4096 x i8> @llvm.ssub.sat.v4096i8(<4096 x i8>, <4096 x i8>)

declare <4096 x i8> @llvm.smin.v4096i8(<4096 x i8>, <4096 x i8>)

declare <4096 x i8> @llvm.smax.v4096i8(<4096 x i8>, <4096 x i8>)

declare <4096 x i8> @llvm.umin.v4096i8(<4096 x i8>, <4096 x i8>)

declare <4096 x i8> @llvm.umax.v4096i8(<4096 x i8>, <4096 x i8>)

declare <4096 x i8> @llvm.abs.v4096i8(<4096 x i8>, i1)

declare <512 x i32> @llvm.uadd.sat.v512i32(<512 x i32>, <512 x i32>)

declare <512 x i32> @llvm.sadd.sat.v512i32(<512 x i32>, <512 x i32>)

declare <512 x i32> @llvm.usub.sat.v512i32(<512 x i32>, <512 x i32>)

declare <512 x i32> @llvm.ssub.sat.v512i32(<512 x i32>, <512 x i32>)

declare <512 x i32> @llvm.smin.v512i32(<512 x i32>, <512 x i32>)

declare <512 x i32> @llvm.smax.v512i32(<512 x i32>, <512 x i32>)

declare <512 x i32> @llvm.umin.v512i32(<512 x i32>, <512 x i32>)

declare <512 x i32> @llvm.umax.v512i32(<512 x i32>, <512 x i32>)

declare <512 x i32> @llvm.abs.v512i32(<512 x i32>, i1)

attributes #0 = { "target-features"="+hlx,+v85" }
