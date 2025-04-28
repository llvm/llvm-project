; RUN: opt -S -passes=instcombine < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; DUPQ b8

define <vscale x 16 x i1> @dupq_b_0() #0 {
; CHECK-LABEL: @dupq_b_0(
; CHECK: ret <vscale x 16 x i1> zeroinitializer
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %2 = tail call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v16i8(<vscale x 16 x i8> undef,
    <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
               i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i64 0)
  %3 = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %3, <vscale x 2 x i64> %4)
  ret <vscale x 16 x i1> %5
}

define <vscale x 16 x i1> @dupq_b_d() #0 {
; CHECK-LABEL: @dupq_b_d(
; CHECK: %1 = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT: %2 = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %1)
; CHECK-NEXT: ret <vscale x 16 x i1> %2
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %2 = tail call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v16i8(<vscale x 16 x i8> undef,
    <16 x i8> <i8 1, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
               i8 1, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i64 0)
  %3 = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %3, <vscale x 2 x i64> %4)
  ret <vscale x 16 x i1> %5
}

define <vscale x 16 x i1> @dupq_b_w() #0 {
; CHECK-LABEL: @dupq_b_w(
; CHECK: %1 = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
; CHECK-NEXT: %2 = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %1)
; CHECK-NEXT: ret <vscale x 16 x i1> %2
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %2 = tail call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v16i8(<vscale x 16 x i8> undef,
    <16 x i8> <i8 1, i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0,
               i8 1, i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0>, i64 0)
  %3 = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %3, <vscale x 2 x i64> %4)
  ret <vscale x 16 x i1> %5
}

define <vscale x 16 x i1> @dupq_b_h() #0 {
; CHECK-LABEL: @dupq_b_h(
; CHECK: %1 = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
; CHECK-NEXT: %2 = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %1)
; CHECK-NEXT: ret <vscale x 16 x i1> %2
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %2 = tail call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v16i8(<vscale x 16 x i8> undef,
    <16 x i8> <i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0,
               i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0>, i64 0)
  %3 = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %3, <vscale x 2 x i64> %4)
  ret <vscale x 16 x i1> %5
}

define <vscale x 16 x i1> @dupq_b_b() #0 {
; CHECK-LABEL: @dupq_b_b(
; CHECK: %1 = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
; CHECK-NEXT: ret <vscale x 16 x i1> %1
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %2 = tail call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v16i8(<vscale x 16 x i8> undef,
    <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, i64 0)
  %3 = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %3, <vscale x 2 x i64> %4)
  ret <vscale x 16 x i1> %5
}

; DUPQ b16

define <vscale x 8 x i1> @dupq_h_0() #0 {
; CHECK-LABEL: @dupq_h_0(
; CHECK: ret <vscale x 8 x i1> zeroinitializer
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %2 = tail call <vscale x 8 x i16> @llvm.vector.insert.nxv8i16.v8i16(<vscale x 8 x i16> undef,
    <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, i64 0)
  %3 = tail call <vscale x 8 x i16> @llvm.aarch64.sve.dupq.lane.nxv8i16(<vscale x 8 x i16> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1> %1, <vscale x 8 x i16> %3, <vscale x 2 x i64> %4)
  ret <vscale x 8 x i1> %5
}

define <vscale x 8 x i1> @dupq_h_d() #0 {
; CHECK-LABEL: @dupq_h_d(
; CHECK: %1 = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT: %2 = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %1)
; CHECK-NEXT: %3 = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %2)
; CHECK-NEXT: ret <vscale x 8 x i1> %3
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %2 = tail call <vscale x 8 x i16> @llvm.vector.insert.nxv8i16.v8i16(<vscale x 8 x i16> undef,
    <8 x i16> <i16 1, i16 0, i16 0, i16 0, i16 1, i16 0, i16 0, i16 0>, i64 0)
  %3 = tail call <vscale x 8 x i16> @llvm.aarch64.sve.dupq.lane.nxv8i16(<vscale x 8 x i16> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1> %1, <vscale x 8 x i16> %3, <vscale x 2 x i64> %4)
  ret <vscale x 8 x i1> %5
}

define <vscale x 8 x i1> @dupq_h_w() #0 {
; CHECK-LABEL: @dupq_h_w(
; CHECK: %1 = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
; CHECK-NEXT: %2 = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %1)
; CHECK-NEXT: %3 = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %2)
; CHECK-NEXT: ret <vscale x 8 x i1> %3
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %2 = tail call <vscale x 8 x i16> @llvm.vector.insert.nxv8i16.v8i16(<vscale x 8 x i16> undef,
    <8 x i16> <i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0>, i64 0)
  %3 = tail call <vscale x 8 x i16> @llvm.aarch64.sve.dupq.lane.nxv8i16(<vscale x 8 x i16> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1> %1, <vscale x 8 x i16> %3, <vscale x 2 x i64> %4)
  ret <vscale x 8 x i1> %5
}

define <vscale x 8 x i1> @dupq_h_h() #0 {
; CHECK-LABEL: @dupq_h_h(
; CHECK: %1 = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
; CHECK-NEXT: ret <vscale x 8 x i1> %1
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %2 = tail call <vscale x 8 x i16> @llvm.vector.insert.nxv8i16.v8i16(<vscale x 8 x i16> undef,
    <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, i64 0)
  %3 = tail call <vscale x 8 x i16> @llvm.aarch64.sve.dupq.lane.nxv8i16(<vscale x 8 x i16> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1> %1, <vscale x 8 x i16> %3, <vscale x 2 x i64> %4)
  ret <vscale x 8 x i1> %5
}

; DUPQ b32

define <vscale x 4 x i1> @dupq_w_0() #0 {
; CHECK-LABEL: @dupq_w_0(
; CHECK: ret <vscale x 4 x i1> zeroinitializer
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32> undef,
    <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i64 0)
  %3 = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %1, <vscale x 4 x i32> %3, <vscale x 2 x i64> %4)
  ret <vscale x 4 x i1> %5
}

define <vscale x 4 x i1> @dupq_w_d() #0 {
; CHECK-LABEL: @dupq_w_d(
; CHECK: %1 = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT: %2 = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %1)
; CHECK-NEXT: %3 = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %2)
; CHECK-NEXT: ret <vscale x 4 x i1> %3
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32> undef,
    <4 x i32> <i32 1, i32 0, i32 1, i32 0>, i64 0)
  %3 = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %1, <vscale x 4 x i32> %3, <vscale x 2 x i64> %4)
  ret <vscale x 4 x i1> %5
}

define <vscale x 4 x i1> @dupq_w_w() #0 {
; CHECK-LABEL: @dupq_w_w(
; CHECK: %1 = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
; CHECK-NEXT: ret <vscale x 4 x i1> %1
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32> undef,
    <4 x i32> <i32 1, i32 1, i32 1, i32 1>, i64 0)
  %3 = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %1, <vscale x 4 x i32> %3, <vscale x 2 x i64> %4)
  ret <vscale x 4 x i1> %5
}

; DUPQ b64

define <vscale x 2 x i1> @dupq_d_0() #0 {
; CHECK-LABEL: @dupq_d_0(
; CHECK: ret <vscale x 2 x i1> zeroinitializer
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> undef,
    <2 x i64> <i64 0, i64 0>, i64 0)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %1, <vscale x 2 x i64> %3, <vscale x 2 x i64> %4)
  ret <vscale x 2 x i1> %5
}

define <vscale x 2 x i1> @dupq_d_d() #0 {
; CHECK-LABEL: @dupq_d_d(
; CHECK: %1 = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; CHECK-NEXT: ret <vscale x 2 x i1> %1
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> undef,
    <2 x i64> <i64 1, i64 1>, i64 0)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %1, <vscale x 2 x i64> %3, <vscale x 2 x i64> %4)
  ret <vscale x 2 x i1> %5
}

; Cases that cannot be converted

define <vscale x 2 x i1> @dupq_neg1() #0 {
; CHECK-LABEL: @dupq_neg1(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> undef,
    <2 x i64> <i64 1, i64 0>, i64 0)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %1, <vscale x 2 x i64> %3, <vscale x 2 x i64> %4)
  ret <vscale x 2 x i1> %5
}

define <vscale x 4 x i1> @dupq_neg2() #0 {
; CHECK-LABEL: @dupq_neg2(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32> undef,
    <4 x i32> <i32 1, i32 0, i32 0, i32 1>, i64 0)
  %3 = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %1, <vscale x 4 x i32> %3, <vscale x 2 x i64> %4)
  ret <vscale x 4 x i1> %5
}

define <vscale x 4 x i1> @dupq_neg3() #0 {
; CHECK-LABEL: @dupq_neg3(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32> undef,
    <4 x i32> <i32 0, i32 1, i32 0, i32 1>, i64 0)
  %3 = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %1, <vscale x 4 x i32> %3, <vscale x 2 x i64> %4)
  ret <vscale x 4 x i1> %5
}

define <vscale x 4 x i1> @dupq_neg4() #0 {
; CHECK-LABEL: @dupq_neg4(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32> undef,
    <4 x i32> <i32 1, i32 1, i32 0, i32 0>, i64 0)
  %3 = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %1, <vscale x 4 x i32> %3, <vscale x 2 x i64> %4)
  ret <vscale x 4 x i1> %5
}

define <vscale x 4 x i1> @dupq_neg5() #0 {
; CHECK-LABEL: @dupq_neg5(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32> undef,
    <4 x i32> <i32 0, i32 0, i32 0, i32 1>, i64 0)
  %3 = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %1, <vscale x 4 x i32> %3, <vscale x 2 x i64> %4)
  ret <vscale x 4 x i1> %5
}

define <vscale x 4 x i1> @dupq_neg6(i1 %a) #0 {
; CHECK-LABEL: @dupq_neg6(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = zext i1 %a to i32
  %3 = insertelement <4 x i32> <i32 1, i32 1, i32 1, i32 poison>, i32 %2, i32 3
  %4 = tail call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32> undef, <4 x i32> %3, i64 0)
  %5 = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %4 , i64 0)
  %6 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %7 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %1, <vscale x 4 x i32> %5, <vscale x 2 x i64> %6)
  ret <vscale x 4 x i1> %7
}

define <vscale x 2 x i1> @dupq_neg7() #0 {
; CHECK-LABEL: @dupq_neg7(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> undef,
    <2 x i64> <i64 1, i64 1>, i64 2)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %1, <vscale x 2 x i64> %3, <vscale x 2 x i64> %4)
  ret <vscale x 2 x i1> %5
}

define <vscale x 2 x i1> @dupq_neg8() #0 {
; CHECK-LABEL: @dupq_neg8(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> undef,
    <2 x i64> <i64 1, i64 1>, i64 0)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %2 , i64 1)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %1, <vscale x 2 x i64> %3, <vscale x 2 x i64> %4)
  ret <vscale x 2 x i1> %5
}

define <vscale x 2 x i1> @dupq_neg9(<vscale x 2 x i64> %x) #0 {
; CHECK-LABEL: @dupq_neg9(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> %x,
    <2 x i64> <i64 1, i64 1>, i64 0)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %1, <vscale x 2 x i64> %3, <vscale x 2 x i64> %4)
  ret <vscale x 2 x i1> %5
}

define <vscale x 2 x i1> @dupq_neg10() #0 {
; CHECK-LABEL: @dupq_neg10(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> undef,
    <2 x i64> <i64 1, i64 1>, i64 0)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 1)
  %5 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %1, <vscale x 2 x i64> %3, <vscale x 2 x i64> %4)
  ret <vscale x 2 x i1> %5
}

define <vscale x 2 x i1> @dupq_neg11(<vscale x 2 x i1> %pg) #0 {
; CHECK-LABEL: @dupq_neg11(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> undef,
    <2 x i64> <i64 1, i64 1>, i64 0)
  %2 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %1 , i64 0)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %4 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %2, <vscale x 2 x i64> %3)
  ret <vscale x 2 x i1> %4
}

define <vscale x 2 x i1> @dupq_neg12() #0 {
; CHECK-LABEL: @dupq_neg12(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 15)
  %2 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> undef,
    <2 x i64> <i64 1, i64 1>, i64 0)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %2 , i64 0)
  %4 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  %5 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %1, <vscale x 2 x i64> %3, <vscale x 2 x i64> %4)
  ret <vscale x 2 x i1> %5
}

define <vscale x 2 x i1> @dupq_neg13(<vscale x 2 x i64> %x) #0 {
; CHECK-LABEL: @dupq_neg13(
; CHECK: cmpne
; CHECK-NEXT: ret
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64> undef,
    <2 x i64> <i64 1, i64 1>, i64 0)
  %3 = tail call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %2 , i64 0)
  %4 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %1, <vscale x 2 x i64> %3, <vscale x 2 x i64> %x)
  ret <vscale x 2 x i1> %4
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32)

declare <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v16i8(<vscale x 16 x i8>, <16 x i8>, i64)
declare <vscale x 8 x i16> @llvm.vector.insert.nxv8i16.v8i16(<vscale x 8 x i16>, <8 x i16>, i64)
declare <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32>, <4 x i32>, i64)
declare <vscale x 2 x i64> @llvm.vector.insert.nxv2i64.v2i64(<vscale x 2 x i64>, <2 x i64>, i64)

declare <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8>, i64)
declare <vscale x 8 x i16> @llvm.aarch64.sve.dupq.lane.nxv8i16(<vscale x 8 x i16>, i64)
declare <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32>, i64)
declare <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64>, i64)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64)

attributes #0 = { "target-features"="+sve" }
