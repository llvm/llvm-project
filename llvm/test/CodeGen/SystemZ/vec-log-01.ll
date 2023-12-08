; Test v2f64 logarithm.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare <2 x double> @llvm.log.v2f64(<2 x double>)

define <2 x double> @f1(<2 x double> %val) {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, log@PLT
; CHECK: brasl %r14, log@PLT
; CHECK: vmrhg %v24,
; CHECK: br %r14
  %ret = call <2 x double> @llvm.log.v2f64(<2 x double> %val)
  ret <2 x double> %ret
}
