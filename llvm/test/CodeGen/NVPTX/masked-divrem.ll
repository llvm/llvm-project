; RUN: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90a < %s | FileCheck %s
; RUN: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100a < %s | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

declare <2 x i32> @llvm.masked.sdiv.v2i32(<2 x i32>, <2 x i32>, <2 x i1>)
declare <2 x i32> @llvm.masked.udiv.v2i32(<2 x i32>, <2 x i32>, <2 x i1>)
declare <2 x i32> @llvm.masked.srem.v2i32(<2 x i32>, <2 x i32>, <2 x i1>)
declare <2 x i32> @llvm.masked.urem.v2i32(<2 x i32>, <2 x i32>, <2 x i1>)
declare <8 x i32> @llvm.masked.srem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>)

define <2 x i32> @masked_divrem_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK-LABEL: masked_divrem_v2i32(
; CHECK-DAG: selp.b32
; CHECK-DAG: div.s32
; CHECK-DAG: div.u32
; CHECK-DAG: rem.s32
; CHECK-DAG: rem.u32
; CHECK: ret;
  %mask = icmp sgt <2 x i32> %lhs, zeroinitializer
  %lhs2 = add <2 x i32> %lhs, <i32 1, i32 1>
  %rhs2 = add <2 x i32> %rhs, <i32 1, i32 1>
  %sdiv = call <2 x i32> @llvm.masked.sdiv.v2i32(
      <2 x i32> %lhs, <2 x i32> %rhs, <2 x i1> %mask)
  %udiv = call <2 x i32> @llvm.masked.udiv.v2i32(
      <2 x i32> %lhs, <2 x i32> %rhs2, <2 x i1> %mask)
  %srem = call <2 x i32> @llvm.masked.srem.v2i32(
      <2 x i32> %lhs2, <2 x i32> %rhs, <2 x i1> %mask)
  %urem = call <2 x i32> @llvm.masked.urem.v2i32(
      <2 x i32> %lhs2, <2 x i32> %rhs2, <2 x i1> %mask)
  %div = add <2 x i32> %sdiv, %udiv
  %rem = add <2 x i32> %srem, %urem
  %result = add <2 x i32> %div, %rem
  ret <2 x i32> %result
}

define <8 x i32> @masked_srem_v8i32(<8 x i32> %lhs, <8 x i32> %rhs) {
; CHECK-LABEL: masked_srem_v8i32(
; CHECK-COUNT-5: rem.s32
; CHECK: ret;
  %result = call <8 x i32> @llvm.masked.srem.v8i32(
      <8 x i32> %lhs, <8 x i32> %rhs,
      <8 x i1> <i1 true, i1 true, i1 true, i1 true,
                  i1 true, i1 false, i1 false, i1 false>)
  ret <8 x i32> %result
}
