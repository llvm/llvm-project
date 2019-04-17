; We need this pipeline because to trigger dominator info verification
; we have to compute the dominator before libcalls-shrinkwrap and
; have a pass which requires the dominator tree after.
; RUN: opt -domtree -libcalls-shrinkwrap -instcombine -verify-dom-info %s

define void @main() {
  %_tmp31 = call float @acosf(float 2.000000e+00)
  ret void
}

declare float @acosf(float)
