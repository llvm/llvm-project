; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

; This regression test is verifying that a GEP instruction performed on a
; scalable vector does not produce a 'assumption that TypeSize is not scalable'
; warning when performing cost analysis.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %retval = getelementptr
define ptr @gep_scalable_vector(ptr %ptr) {
  %retval = getelementptr <vscale x 16 x i8>, ptr %ptr, i32 2
  ret ptr %retval
}
