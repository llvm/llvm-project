; Check getIntrinsicInstrCost in BasicTTIImpl.h with SVE for vector.reduce.<operand>
; Checks legal and not legal vector size

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s 2>%t | FileCheck %s


; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define i32 @add.i32.nxv4i32(<vscale x 4 x i32> %v) {
; CHECK-LABEL: 'add.i32.nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> %v)
; CHECK-NEXT:Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %r

  %r = call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> %v)
  ret i32 %r
}

define i64 @add.i64.nxv4i64(<vscale x 4 x i64> %v) {
; CHECK-LABEL: 'add.i64.nxv4i64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %r = call i64 @llvm.vector.reduce.add.nxv4i64(<vscale x 4 x i64> %v)
; CHECK-NEXT:Cost Model: Found an estimated cost of 0 for instruction:   ret i64 %r

  %r = call i64 @llvm.vector.reduce.add.nxv4i64(<vscale x 4 x i64> %v)
  ret i64 %r
}

define i32 @mul.i32.nxv4i32(<vscale x 4 x i32> %v) {
; CHECK-LABEL: 'mul.i32.nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 16 for instruction:   %r = call i32 @llvm.vector.reduce.mul.nxv4i32(<vscale x 4 x i32> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %r

  %r = call i32 @llvm.vector.reduce.mul.nxv4i32(<vscale x 4 x i32> %v)
  ret i32 %r
}

define i64 @mul.i64.nxv4i64(<vscale x 4 x i64> %v) {
; CHECK-LABEL: 'mul.i64.nxv4i64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 16 for instruction:   %r = call i64 @llvm.vector.reduce.mul.nxv4i64(<vscale x 4 x i64> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i64 %r

  %r = call i64 @llvm.vector.reduce.mul.nxv4i64(<vscale x 4 x i64> %v)
  ret i64 %r
}

define i32 @and.i32.nxv4i32(<vscale x 4 x i32> %v) {
; CHECK-LABEL: 'and.i32.nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call i32 @llvm.vector.reduce.and.nxv4i32(<vscale x 4 x i32> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %r

  %r = call i32 @llvm.vector.reduce.and.nxv4i32(<vscale x 4 x i32> %v)
  ret i32 %r
}

define i64 @and.i64.nxv4i64(<vscale x 4 x i64> %v) {
; CHECK-LABEL: 'and.i64.nxv4i64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %r = call i64 @llvm.vector.reduce.and.nxv4i64(<vscale x 4 x i64> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i64 %r

  %r = call i64 @llvm.vector.reduce.and.nxv4i64(<vscale x 4 x i64> %v)
  ret i64 %r
}

define i32 @or.i32.nxv4i32(<vscale x 4 x i32> %v) {
; CHECK-LABEL: 'or.i32.nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call i32 @llvm.vector.reduce.or.nxv4i32(<vscale x 4 x i32> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %r

  %r = call i32 @llvm.vector.reduce.or.nxv4i32(<vscale x 4 x i32> %v)
  ret i32 %r
}

define i64 @or.i64.nxv4i64(<vscale x 4 x i64> %v) {
; CHECK-LABEL: 'or.i64.nxv4i64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %r = call i64 @llvm.vector.reduce.or.nxv4i64(<vscale x 4 x i64> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i64 %r

  %r = call i64 @llvm.vector.reduce.or.nxv4i64(<vscale x 4 x i64> %v)
  ret i64 %r
}

define i32 @xor.i32.nxv4i32(<vscale x 4 x i32> %v) {
; CHECK-LABEL: 'xor.i32.nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call i32 @llvm.vector.reduce.xor.nxv4i32(<vscale x 4 x i32> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %r

  %r = call i32 @llvm.vector.reduce.xor.nxv4i32(<vscale x 4 x i32> %v)
  ret i32 %r
}

define i64 @xor.i64.nxv4i64(<vscale x 4 x i64> %v) {
; CHECK-LABEL: 'xor.i64.nxv4i64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %r = call i64 @llvm.vector.reduce.xor.nxv4i64(<vscale x 4 x i64> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i64 %r

  %r = call i64 @llvm.vector.reduce.xor.nxv4i64(<vscale x 4 x i64> %v)
  ret i64 %r
}

define i32 @umin.i32.nxv4i32(<vscale x 4 x i32> %v) {
; CHECK-LABEL: 'umin.i32.nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call i32 @llvm.vector.reduce.umin.nxv4i32(<vscale x 4 x i32> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %r

  %r = call i32 @llvm.vector.reduce.umin.nxv4i32(<vscale x 4 x i32> %v)
  ret i32 %r
}

define i64 @umin.i64.nxv4i64(<vscale x 4 x i64> %v) {
; CHECK-LABEL: 'umin.i64.nxv4i64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %r = call i64 @llvm.vector.reduce.umin.nxv4i64(<vscale x 4 x i64> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i64 %r

  %r = call i64 @llvm.vector.reduce.umin.nxv4i64(<vscale x 4 x i64> %v)
  ret i64 %r
}

define float @fmax.f32.nxv4f32(<vscale x 4 x float> %v) {
; CHECK-LABEL: 'fmax.f32.nxv4f32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call float @llvm.vector.reduce.fmax.nxv4f32(<vscale x 4 x float> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret float %r

  %r = call float @llvm.vector.reduce.fmax.nxv4f32(<vscale x 4 x float> %v)
  ret float %r
}

define double @fmax.f64.nxv4f64(<vscale x 4 x double> %v) {
; CHECK-LABEL: 'fmax.f64.nxv4f64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %r = call double @llvm.vector.reduce.fmax.nxv4f64(<vscale x 4 x double> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret double %r

  %r = call double @llvm.vector.reduce.fmax.nxv4f64(<vscale x 4 x double> %v)
  ret double %r
}

define float @fmin.f32.nxv4f32(<vscale x 4 x float> %v) {
; CHECK-LABEL: 'fmin.f32.nxv4f32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call float @llvm.vector.reduce.fmin.nxv4f32(<vscale x 4 x float> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret float %r

  %r = call float @llvm.vector.reduce.fmin.nxv4f32(<vscale x 4 x float> %v)
  ret float %r
}

define double @fmin.f64.nxv4f64(<vscale x 4 x double> %v) {
; CHECK-LABEL: 'fmin.f64.nxv4f64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %r = call double @llvm.vector.reduce.fmin.nxv4f64(<vscale x 4 x double> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret double %r

  %r = call double @llvm.vector.reduce.fmin.nxv4f64(<vscale x 4 x double> %v)
  ret double %r
}

define i32 @umax.i32.nxv4i32(<vscale x 4 x i32> %v) {
; CHECK-LABEL: 'umax.i32.nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call i32 @llvm.vector.reduce.umax.nxv4i32(<vscale x 4 x i32> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %r

  %r = call i32 @llvm.vector.reduce.umax.nxv4i32(<vscale x 4 x i32> %v)
  ret i32 %r
}

define i64 @umax.i64.nxv4i64(<vscale x 4 x i64> %v) {
; CHECK-LABEL: 'umax.i64.nxv4i64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %r = call i64 @llvm.vector.reduce.umax.nxv4i64(<vscale x 4 x i64> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i64 %r

  %r = call i64 @llvm.vector.reduce.umax.nxv4i64(<vscale x 4 x i64> %v)
  ret i64 %r
}

define i32 @smin.i32.nxv4i32(<vscale x 4 x i32> %v) {
; CHECK-LABEL: 'smin.i32.nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call i32 @llvm.vector.reduce.smin.nxv4i32(<vscale x 4 x i32> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %r

  %r = call i32 @llvm.vector.reduce.smin.nxv4i32(<vscale x 4 x i32> %v)
  ret i32 %r
}

define i64 @smin.i64.nxv4i64(<vscale x 4 x i64> %v) {
; CHECK-LABEL: 'smin.i64.nxv4i64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %r = call i64 @llvm.vector.reduce.smin.nxv4i64(<vscale x 4 x i64> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i64 %r

  %r = call i64 @llvm.vector.reduce.smin.nxv4i64(<vscale x 4 x i64> %v)
  ret i64 %r
}

define i32 @smax.i32.nxv4i32(<vscale x 4 x i32> %v) {
; CHECK-LABEL: 'smax.i32.nxv4i32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %r = call i32 @llvm.vector.reduce.smax.nxv4i32(<vscale x 4 x i32> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %r

  %r = call i32 @llvm.vector.reduce.smax.nxv4i32(<vscale x 4 x i32> %v)
  ret i32 %r
}

define i64 @smax.i64.nxv4i64(<vscale x 4 x i64> %v) {
; CHECK-LABEL: 'smax.i64.nxv4i64'
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction:   %r = call i64 @llvm.vector.reduce.smax.nxv4i64(<vscale x 4 x i64> %v)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret i64 %r

  %r = call i64 @llvm.vector.reduce.smax.nxv4i64(<vscale x 4 x i64> %v)
  ret i64 %r
}

define float @fadda_nxv4f32(float %start, <vscale x 4 x float> %a) #0 {
; CHECK-LABEL: 'fadda_nxv4f32
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %res = call float @llvm.vector.reduce.fadd.nxv4f32(float %start, <vscale x 4 x float> %a)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret float %res

  %res = call float @llvm.vector.reduce.fadd.nxv4f32(float %start, <vscale x 4 x float> %a)
  ret float %res
}

define double @fadda_nxv4f64(double %start, <vscale x 4 x double> %a) #0 {
; CHECK-LABEL: 'fadda_nxv4f64
; CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %res = call double @llvm.vector.reduce.fadd.nxv4f64(double %start, <vscale x 4 x double> %a)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret double %res

  %res = call double @llvm.vector.reduce.fadd.nxv4f64(double %start, <vscale x 4 x double> %a)
  ret double %res
}


declare i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32>)
declare i32 @llvm.vector.reduce.mul.nxv4i32(<vscale x 4 x i32>)
declare i32 @llvm.vector.reduce.and.nxv4i32(<vscale x 4 x i32>)
declare i32 @llvm.vector.reduce.or.nxv4i32(<vscale x 4 x i32>)
declare i32 @llvm.vector.reduce.xor.nxv4i32(<vscale x 4 x i32>)
declare float @llvm.vector.reduce.fmax.nxv4f32(<vscale x 4 x float>)
declare float @llvm.vector.reduce.fmin.nxv4f32(<vscale x 4 x float>)
declare i32 @llvm.vector.reduce.fmin.nxv4i32(<vscale x 4 x i32>)
declare i32 @llvm.vector.reduce.umin.nxv4i32(<vscale x 4 x i32>)
declare i32 @llvm.vector.reduce.umax.nxv4i32(<vscale x 4 x i32>)
declare i32 @llvm.vector.reduce.smin.nxv4i32(<vscale x 4 x i32>)
declare i32 @llvm.vector.reduce.smax.nxv4i32(<vscale x 4 x i32>)
declare float @llvm.vector.reduce.fadd.nxv4f32(float, <vscale x 4 x float>)
declare i64 @llvm.vector.reduce.add.nxv4i64(<vscale x 4 x i64>)
declare i64 @llvm.vector.reduce.mul.nxv4i64(<vscale x 4 x i64>)
declare i64 @llvm.vector.reduce.and.nxv4i64(<vscale x 4 x i64>)
declare i64 @llvm.vector.reduce.or.nxv4i64(<vscale x 4 x i64>)
declare i64 @llvm.vector.reduce.xor.nxv4i64(<vscale x 4 x i64>)
declare double @llvm.vector.reduce.fmax.nxv4f64(<vscale x 4 x double>)
declare double @llvm.vector.reduce.fmin.nxv4f64(<vscale x 4 x double>)
declare i64 @llvm.vector.reduce.umin.nxv4i64(<vscale x 4 x i64>)
declare i64 @llvm.vector.reduce.umax.nxv4i64(<vscale x 4 x i64>)
declare i64 @llvm.vector.reduce.smin.nxv4i64(<vscale x 4 x i64>)
declare i64 @llvm.vector.reduce.smax.nxv4i64(<vscale x 4 x i64>)
declare double @llvm.vector.reduce.fadd.nxv4f64(double, <vscale x 4 x double>)
