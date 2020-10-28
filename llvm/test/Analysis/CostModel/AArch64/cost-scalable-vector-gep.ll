; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s < %t

; This regression test is verifying that a GEP instruction performed on a
; scalable vector does not produce a 'assumption that TypeSize is not scalable'
; warning when performing cost analysis.

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning: {{.*}}TypeSize is not scalable

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %retval = getelementptr
define <vscale x 16 x i8>* @gep_scalable_vector(<vscale x 16 x i8>* %ptr) {
  %retval = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %ptr, i32 2
  ret <vscale x 16 x i8>* %retval
}
