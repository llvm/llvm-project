; RUN: llc < %s -o /dev/null 2>&1 | FileCheck --allow-empty %s

; This regression test is defending against a ElementCount warning 'Possible incorrect use of
; EVT::getVectorNumElements() for scalable vector'. This warning appeared in
; DAGCombiner::visitINSERT_SUBVECTOR because of the use of getVectorNumElements() on scalable
; types.

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; CHECK-NOT: warning:

target triple = "aarch64-unknown-linux-gnu"
attributes #0 = {"target-features"="+sve"}

declare <16 x float> @llvm.experimental.vector.extract.v16f32.nxv4f32(<vscale x 4 x float>, i64)
declare <vscale x 2 x double> @llvm.experimental.vector.insert.nxv2f64.v8f64(<vscale x 2 x double>, <8 x double>, i64)

define <vscale x 2 x double> @reproducer_one(<vscale x 4 x float> %vec_a) #0 {
  %a = call <16 x float> @llvm.experimental.vector.extract.v16f32.nxv4f32(<vscale x 4 x float> %vec_a, i64 0)
  %b = bitcast <16 x float> %a to <8 x double>
  %retval = call <vscale x 2 x double> @llvm.experimental.vector.insert.nxv2f64.v8f64(<vscale x 2 x double> undef, <8 x double> %b, i64 0)
  ret <vscale x 2 x double> %retval
}

define <vscale x 2 x double> @reproducer_two(<4 x double> %a, <4 x double> %b) #0 {
  %concat = shufflevector <4 x double> %a, <4 x double> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  %retval = call <vscale x 2 x double> @llvm.experimental.vector.insert.nxv2f64.v8f64(<vscale x 2 x double> undef, <8 x double> %concat, i64 0)
  ret <vscale x 2 x double> %retval
}
