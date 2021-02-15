; Check getIntrinsicInstrCost in BasicTTIImpl.h for vector.reverse

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s 2>%t | FileCheck %s

; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define void @vector_reverse() #0 {
; CHECK-LABEL: 'vector_reverse':
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %1 = call <vscale x 16 x i8> @llvm.experimental.vector.reverse.nxv16i8(<vscale x 16 x i8> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %2 = call <vscale x 32 x i8> @llvm.experimental.vector.reverse.nxv32i8(<vscale x 32 x i8> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %3 = call <vscale x 8 x i16> @llvm.experimental.vector.reverse.nxv8i16(<vscale x 8 x i16> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %4 = call <vscale x 16 x i16> @llvm.experimental.vector.reverse.nxv16i16(<vscale x 16 x i16> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %5 = call <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %6 = call <vscale x 8 x i32> @llvm.experimental.vector.reverse.nxv8i32(<vscale x 8 x i32> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %7 = call <vscale x 2 x i64> @llvm.experimental.vector.reverse.nxv2i64(<vscale x 2 x i64> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %8 = call <vscale x 4 x i64> @llvm.experimental.vector.reverse.nxv4i64(<vscale x 4 x i64> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %9 = call <vscale x 8 x half> @llvm.experimental.vector.reverse.nxv8f16(<vscale x 8 x half> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %10 = call <vscale x 16 x half> @llvm.experimental.vector.reverse.nxv16f16(<vscale x 16 x half> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %11 = call <vscale x 4 x float> @llvm.experimental.vector.reverse.nxv4f32(<vscale x 4 x float> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %12 = call <vscale x 8 x float> @llvm.experimental.vector.reverse.nxv8f32(<vscale x 8 x float> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %13 = call <vscale x 2 x double> @llvm.experimental.vector.reverse.nxv2f64(<vscale x 2 x double> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %14 = call <vscale x 4 x double> @llvm.experimental.vector.reverse.nxv4f64(<vscale x 4 x double> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %15 = call <vscale x 8 x bfloat> @llvm.experimental.vector.reverse.nxv8bf16(<vscale x 8 x bfloat> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %16 = call <vscale x 16 x bfloat> @llvm.experimental.vector.reverse.nxv16bf16(<vscale x 16 x bfloat> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret void

  call <vscale x 16 x i8> @llvm.experimental.vector.reverse.nxv16i8(<vscale x 16 x i8> undef)
  call <vscale x 32 x i8> @llvm.experimental.vector.reverse.nxv32i8(<vscale x 32 x i8> undef)
  call <vscale x 8 x i16> @llvm.experimental.vector.reverse.nxv8i16(<vscale x 8 x i16> undef)
  call <vscale x 16 x i16> @llvm.experimental.vector.reverse.nxv16i16(<vscale x 16 x i16> undef)
  call <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32> undef)
  call <vscale x 8 x i32> @llvm.experimental.vector.reverse.nxv8i32(<vscale x 8 x i32> undef)
  call <vscale x 2 x i64> @llvm.experimental.vector.reverse.nxv2i64(<vscale x 2 x i64> undef)
  call <vscale x 4 x i64> @llvm.experimental.vector.reverse.nxv4i64(<vscale x 4 x i64> undef)
  call <vscale x 8 x half> @llvm.experimental.vector.reverse.nxv8f16(<vscale x 8 x half> undef)
  call <vscale x 16 x half> @llvm.experimental.vector.reverse.nxv16f16(<vscale x 16 x half> undef)
  call <vscale x 4 x float> @llvm.experimental.vector.reverse.nxv4f32(<vscale x 4 x float> undef)
  call <vscale x 8 x float> @llvm.experimental.vector.reverse.nxv8f32(<vscale x 8 x float> undef)
  call <vscale x 2 x double> @llvm.experimental.vector.reverse.nxv2f64(<vscale x 2 x double> undef)
  call <vscale x 4 x double> @llvm.experimental.vector.reverse.nxv4f64(<vscale x 4 x double> undef)
  call <vscale x 8 x bfloat> @llvm.experimental.vector.reverse.nxv8bf16(<vscale x 8 x bfloat> undef)
  call <vscale x 16 x bfloat> @llvm.experimental.vector.reverse.nxv16bf16(<vscale x 16 x bfloat> undef)
  ret void
}

attributes #0 = { "target-features"="+sve,+bf16" }

declare <vscale x 16 x i8> @llvm.experimental.vector.reverse.nxv16i8(<vscale x 16 x i8>)
declare <vscale x 32 x i8> @llvm.experimental.vector.reverse.nxv32i8(<vscale x 32 x i8>)
declare <vscale x 8 x i16> @llvm.experimental.vector.reverse.nxv8i16(<vscale x 8 x i16>)
declare <vscale x 16 x i16> @llvm.experimental.vector.reverse.nxv16i16(<vscale x 16 x i16>)
declare <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32>)
declare <vscale x 8 x i32> @llvm.experimental.vector.reverse.nxv8i32(<vscale x 8 x i32>)
declare <vscale x 2 x i64> @llvm.experimental.vector.reverse.nxv2i64(<vscale x 2 x i64>)
declare <vscale x 4 x i64> @llvm.experimental.vector.reverse.nxv4i64(<vscale x 4 x i64>)
declare <vscale x 8 x half> @llvm.experimental.vector.reverse.nxv8f16(<vscale x 8 x half>)
declare <vscale x 16 x half> @llvm.experimental.vector.reverse.nxv16f16(<vscale x 16 x half>)
declare <vscale x 4 x float> @llvm.experimental.vector.reverse.nxv4f32(<vscale x 4 x float>)
declare <vscale x 8 x float> @llvm.experimental.vector.reverse.nxv8f32(<vscale x 8 x float>)
declare <vscale x 2 x double> @llvm.experimental.vector.reverse.nxv2f64(<vscale x 2 x double>)
declare <vscale x 4 x double> @llvm.experimental.vector.reverse.nxv4f64(<vscale x 4 x double>)
declare <vscale x 8 x bfloat> @llvm.experimental.vector.reverse.nxv8bf16(<vscale x 8 x bfloat>)
declare <vscale x 16 x bfloat> @llvm.experimental.vector.reverse.nxv16bf16(<vscale x 16 x bfloat>)
