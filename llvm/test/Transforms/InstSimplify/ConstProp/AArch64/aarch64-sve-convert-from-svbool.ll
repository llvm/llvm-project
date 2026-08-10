; RUN: opt -passes=instsimplify -S -o - < %s | FileCheck %s

define <vscale x 2 x i1> @reinterpret_zero() {
; CHECK-LABEL:  @reinterpret_zero(
; CHECK: ret <vscale x 2 x i1> zeroinitializer
  %pg = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> zeroinitializer)
  ret <vscale x 2 x i1> %pg
}

declare <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1>)
