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

define <4 x i32> @udiv_v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i1> %m) {
; CHECK-LABEL: name: udiv_v4i32
; CHECK:       [[A:%[0-9]+]]:_(<4 x i32>) = COPY $q0
; CHECK-NEXT:  [[B:%[0-9]+]]:_(<4 x i32>) = COPY $q1
; CHECK-NEXT:  [[M16:%[0-9]+]]:_(<4 x i16>) = COPY $d2
; CHECK-NEXT:  [[M:%[0-9]+]]:_(<4 x i1>) = G_TRUNC [[M16]](<4 x i16>)
; CHECK-NEXT:  [[RES:%[0-9]+]]:_(<4 x i32>) = G_INTRINSIC intrinsic(@llvm.masked.udiv), [[A]](<4 x i32>), [[B]](<4 x i32>), [[M]](<4 x i1>)
; CHECK-NEXT:  $q0 = COPY [[RES]](<4 x i32>)
; CHECK-NEXT:  RET_ReallyLR implicit $q0
  %r = call <4 x i32> @llvm.masked.udiv.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i1> %m)
  ret <4 x i32> %r
}
