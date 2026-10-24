; RUN: opt -S -mtriple=x86_64-- -passes='require<libcall-lowering-info>,expand-ir-insts' < %s | FileCheck %s

define bfloat @ui129tobf16(i129 %a) {
; CHECK-LABEL: define bfloat @ui129tobf16(
; CHECK:       [[LOW:%.*]] = and i129 {{.*}}, 3
; CHECK:       [[STICKY:%.*]] = icmp ne i129 [[LOW]], 0
; CHECK:       [[STICKY_ZEXT:%.*]] = zext i1 [[STICKY]] to i129
; CHECK:       [[SHIFTED:%.*]] = lshr i129 {{.*}}, 2
; CHECK:       or i129 [[SHIFTED]], [[STICKY_ZEXT]]
; CHECK:       fptrunc float {{.*}} to bfloat
  %r = uitofp i129 %a to bfloat
  ret bfloat %r
}
