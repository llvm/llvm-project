; RUN: opt -S < %s | FileCheck %s
; RUN: llvm-dis < %s.bc | FileCheck %s

define <vscale x 16 x i8> @insert(<vscale x 16 x i8> %a, <4 x i8> %b) {
; CHECK-LABEL: @insert
; CHECK: %res = call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v4i8(<vscale x 16 x i8> %a, <4 x i8> %b, i64 0)
  %res = call <vscale x 16 x i8> @llvm.experimental.vector.insert.nxv16i8.v4i8(<vscale x 16 x i8> %a, <4 x i8> %b, i64 0)
  ret <vscale x 16 x i8> %res
}

define <4 x i8> @extract(<vscale x 16 x i8> %a) {
; CHECK-LABEL: @extract
; CHECK: %res = call <4 x i8> @llvm.vector.extract.v4i8.nxv16i8(<vscale x 16 x i8> %a, i64 0)
  %res = call <4 x i8> @llvm.experimental.vector.extract.v4i8.nxv16i8(<vscale x 16 x i8> %a, i64 0)
  ret <4 x i8> %res
}

declare <vscale x 16 x i8> @llvm.experimental.vector.insert.nxv16i8.v4i8(<vscale x 16 x i8>, <4 x i8>, i64 immarg)
; CHECK: declare <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v4i8(<vscale x 16 x i8>, <4 x i8>, i64 immarg)

declare <4 x i8> @llvm.experimental.vector.extract.v4i8.nxv16i8(<vscale x 16 x i8>, i64 immarg)
; CHECK: declare <4 x i8> @llvm.vector.extract.v4i8.nxv16i8(<vscale x 16 x i8>, i64 immarg)
