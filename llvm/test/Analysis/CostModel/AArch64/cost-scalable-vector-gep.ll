; RUN: opt -passes='print<cost-model>' 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

; This regression test is verifying that a GEP instruction performed on a
; scalable vector does not produce a 'assumption that TypeSize is not scalable'
; warning when performing cost analysis.

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %retval = getelementptr
define <vscale x 16 x i8>* @gep_scalable_vector(<vscale x 16 x i8>* %ptr) {
  %retval = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %ptr, i32 2
  ret <vscale x 16 x i8>* %retval
}
