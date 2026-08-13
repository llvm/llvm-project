; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 %s -o - | FileCheck %s

define <4 x i1> @fp(<4 x fp128> %0)  {
; CHECK-LABEL: fp:
; CHECK-COUNT-4: bl __eqkf2
; CHECK: blr
Entry:
  %1 = fcmp oeq <4 x fp128> %0, zeroinitializer
  ret <4 x i1> %1
}

define <4 x i1> @foo(<4 x fp128> %0) strictfp {
; CHECK-LABEL: foo:
; CHECK-COUNT-4: bl __eqkf2
; CHECK: blr
Entry:
  %1 = call <4 x i1> @llvm.experimental.constrained.fcmp.v4fp128(<4 x fp128> %0, <4 x fp128> zeroinitializer, metadata !"oeq", metadata !"fpexcept.strict")
  ret <4 x i1> %1
}

declare <4 x i1> @llvm.experimental.constrained.fcmp.v4fp128(<4 x fp128>, <4 x fp128>, metadata, metadata)

