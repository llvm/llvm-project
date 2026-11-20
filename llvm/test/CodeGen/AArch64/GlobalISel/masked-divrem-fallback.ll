; RUN: llc -O0 -mtriple=aarch64 -global-isel -global-isel-abort=2 -start-after=pre-isel-intrinsic-lowering -stop-after=ir-translator -verify-machineinstrs %s -o - | FileCheck %s

define i128 @udiv(i128 %a, i128 %b, i1 %m) {
; CHECK-LABEL: name: udiv
; CHECK:       [[A:%[0-9]+]]:_(i128) = G_MERGE_VALUES
; CHECK:       [[B:%[0-9]+]]:_(i128) = G_MERGE_VALUES
; CHECK:       [[M:%[0-9]+]]:_(i1) = G_TRUNC
; CHECK-NEXT:  [[FM:%[0-9]+]]:_(i1) = G_FREEZE [[M]]
; CHECK-NEXT:  [[ONE:%[0-9]+]]:_(i128) = G_CONSTANT i128 1
; CHECK-NEXT:  [[DIV:%[0-9]+]]:_(i128) = G_SELECT [[FM]](i1), [[B]], [[ONE]]
; CHECK-NEXT:  [[RES:%[0-9]+]]:_(i128) = G_UDIV [[A]], [[DIV]]
; CHECK-NEXT:  G_UNMERGE_VALUES [[RES]](i128)
  %r = call i128 @llvm.masked.udiv.i128(i128 %a, i128 %b, i1 %m)
  ret i128 %r
}

define i128 @sdiv(i128 %a, i128 %b, i1 %m) {
; CHECK-LABEL: name: sdiv
; CHECK:       [[A:%[0-9]+]]:_(i128) = G_MERGE_VALUES
; CHECK:       [[B:%[0-9]+]]:_(i128) = G_MERGE_VALUES
; CHECK:       [[M:%[0-9]+]]:_(i1) = G_TRUNC
; CHECK-NEXT:  [[FM:%[0-9]+]]:_(i1) = G_FREEZE [[M]]
; CHECK-NEXT:  [[ONE:%[0-9]+]]:_(i128) = G_CONSTANT i128 1
; CHECK-NEXT:  [[DIV:%[0-9]+]]:_(i128) = G_SELECT [[FM]](i1), [[B]], [[ONE]]
; CHECK-NEXT:  [[RES:%[0-9]+]]:_(i128) = G_SDIV [[A]], [[DIV]]
; CHECK-NEXT:  G_UNMERGE_VALUES [[RES]](i128)
  %r = call i128 @llvm.masked.sdiv.i128(i128 %a, i128 %b, i1 %m)
  ret i128 %r
}

define i128 @urem(i128 %a, i128 %b, i1 %m) {
; CHECK-LABEL: name: urem
; CHECK:       [[A:%[0-9]+]]:_(i128) = G_MERGE_VALUES
; CHECK:       [[B:%[0-9]+]]:_(i128) = G_MERGE_VALUES
; CHECK:       [[M:%[0-9]+]]:_(i1) = G_TRUNC
; CHECK-NEXT:  [[FM:%[0-9]+]]:_(i1) = G_FREEZE [[M]]
; CHECK-NEXT:  [[ONE:%[0-9]+]]:_(i128) = G_CONSTANT i128 1
; CHECK-NEXT:  [[DIV:%[0-9]+]]:_(i128) = G_SELECT [[FM]](i1), [[B]], [[ONE]]
; CHECK-NEXT:  [[RES:%[0-9]+]]:_(i128) = G_UREM [[A]], [[DIV]]
; CHECK-NEXT:  G_UNMERGE_VALUES [[RES]](i128)
  %r = call i128 @llvm.masked.urem.i128(i128 %a, i128 %b, i1 %m)
  ret i128 %r
}

define i128 @srem(i128 %a, i128 %b, i1 %m) {
; CHECK-LABEL: name: srem
; CHECK:       [[A:%[0-9]+]]:_(i128) = G_MERGE_VALUES
; CHECK:       [[B:%[0-9]+]]:_(i128) = G_MERGE_VALUES
; CHECK:       [[M:%[0-9]+]]:_(i1) = G_TRUNC
; CHECK-NEXT:  [[FM:%[0-9]+]]:_(i1) = G_FREEZE [[M]]
; CHECK-NEXT:  [[ONE:%[0-9]+]]:_(i128) = G_CONSTANT i128 1
; CHECK-NEXT:  [[DIV:%[0-9]+]]:_(i128) = G_SELECT [[FM]](i1), [[B]], [[ONE]]
; CHECK-NEXT:  [[RES:%[0-9]+]]:_(i128) = G_SREM [[A]], [[DIV]]
; CHECK-NEXT:  G_UNMERGE_VALUES [[RES]](i128)
  %r = call i128 @llvm.masked.srem.i128(i128 %a, i128 %b, i1 %m)
  ret i128 %r
}

define i1 @udiv_i1(i1 %a, i1 %b, i1 %m) {
; CHECK-LABEL: name: udiv_i1
; CHECK:       [[A:%[0-9]+]]:_(i1) = G_TRUNC
; CHECK:       [[B:%[0-9]+]]:_(i1) = G_TRUNC
; CHECK:       [[M:%[0-9]+]]:_(i1) = G_TRUNC
; CHECK-NEXT:  [[FM:%[0-9]+]]:_(i1) = G_FREEZE [[M]]
; CHECK-NEXT:  [[ONE:%[0-9]+]]:_(i1) = G_CONSTANT i1 true
; CHECK-NEXT:  [[DIV:%[0-9]+]]:_(i1) = G_SELECT [[FM]](i1), [[B]], [[ONE]]
; CHECK-NEXT:  [[RES:%[0-9]+]]:_(i1) = G_UDIV [[A]], [[DIV]]
; CHECK-NEXT:  G_ZEXT [[RES]](i1)
  %r = call i1 @llvm.masked.udiv.i1(i1 %a, i1 %b, i1 %m)
  ret i1 %r
}

define i1 @signed_i1(i1 %a, i1 %b, i1 %m) {
; CHECK-LABEL: name: signed_i1
; CHECK-NOT:   G_SDIV
; CHECK:       [[ZERO:%[0-9]+]]:_(i1) = G_CONSTANT i1 false
; CHECK-NEXT:  [[ZEXT:%[0-9]+]]:_(i8) = G_ZEXT [[ZERO]](i1)
; CHECK-NEXT:  [[EXT:%[0-9]+]]:_(i32) = G_ANYEXT [[ZEXT]](i8)
; CHECK-NEXT:  $w0 = COPY [[EXT]](i32)
; CHECK-NEXT:  RET_ReallyLR implicit $w0
  %r = call i1 @llvm.masked.sdiv.i1(i1 %a, i1 %b, i1 %m)
  ret i1 %r
}

define i1 @srem_i1(i1 %a, i1 %b, i1 %m) {
; CHECK-LABEL: name: srem_i1
; CHECK-NOT:   G_SREM
; CHECK:       [[ZERO:%[0-9]+]]:_(i1) = G_CONSTANT i1 false
; CHECK-NEXT:  [[ZEXT:%[0-9]+]]:_(i8) = G_ZEXT [[ZERO]](i1)
; CHECK-NEXT:  [[EXT:%[0-9]+]]:_(i32) = G_ANYEXT [[ZEXT]](i8)
; CHECK-NEXT:  $w0 = COPY [[EXT]](i32)
; CHECK-NEXT:  RET_ReallyLR implicit $w0
  %r = call i1 @llvm.masked.srem.i1(i1 %a, i1 %b, i1 %m)
  ret i1 %r
}

define <2 x i1> @sdiv_v2i1(<2 x i1> %a, <2 x i1> %b, <2 x i1> %m) {
; CHECK-LABEL: name: sdiv_v2i1
; CHECK-NOT:   G_SDIV
; CHECK:       [[ZERO:%[0-9]+]]:_(i1) = G_CONSTANT i1 false
; CHECK-NEXT:  G_BUILD_VECTOR [[ZERO]](i1), [[ZERO]](i1)
; CHECK:       RET_ReallyLR
  %r = call <2 x i1> @llvm.masked.sdiv.v2i1(<2 x i1> %a, <2 x i1> %b, <2 x i1> %m)
  ret <2 x i1> %r
}

define <2 x i128> @udiv_v2i128(<2 x i128> %a, <2 x i128> %b, <2 x i1> %m) {
; CHECK-LABEL: name: udiv_v2i128
; CHECK-COUNT-2: G_UDIV
; CHECK:       RET_ReallyLR
  %r = call <2 x i128> @llvm.masked.udiv.v2i128(<2 x i128> %a, <2 x i128> %b, <2 x i1> %m)
  ret <2 x i128> %r
}

define void @udiv_nxv4i32() "target-features"="+sve" {
; CHECK-LABEL: name: udiv_nxv4i32
; CHECK:       failedISel: true
  %r = call <vscale x 4 x i32> @llvm.masked.udiv.nxv4i32(<vscale x 4 x i32> poison, <vscale x 4 x i32> poison, <vscale x 4 x i1> poison)
  ret void
}
