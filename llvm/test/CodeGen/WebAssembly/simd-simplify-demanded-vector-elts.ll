; RUN: llc < %s -mattr=+simd128 -verify-machineinstrs

target triple = "wasm32-unknown-unknown"

; After DAG legalization, in SelectionDAG optimization phase, ISel runs
; DAGCombiner on each node, among which SimplifyDemandedVectorElts turns unused
; vector elements into undefs. And in order to make sure the DAG is in a
; legalized state, it runs legalization again, which runs our custom
; LowerBUILD_VECTOR, which converts undefs into zeros, causing an infinite loop.
; We prevent this from happening by creating a custom hook , which allows us to
; bail out of SimplifyDemandedVectorElts after legalization.

; This is a reduced test case from a bug reproducer reported. This should not
; hang.
define void @test(i8 %0) {
  %2 = insertelement <4 x i8> <i8 -1, i8 -1, i8 -1, i8 poison>, i8 %0, i64 3
  %3 = zext <4 x i8> %2 to <4 x i32>
  %4 = mul nuw nsw <4 x i32> %3, <i32 257, i32 257, i32 257, i32 257>
  %5 = add nuw nsw <4 x i32> %4, <i32 1, i32 1, i32 1, i32 1>
  %6 = lshr <4 x i32> %5, <i32 1, i32 1, i32 1, i32 1>
  %7 = mul nuw nsw <4 x i32> %6, <i32 20000, i32 20000, i32 20000, i32 20000>
  %8 = add nuw nsw <4 x i32> %7, <i32 32768, i32 32768, i32 32768, i32 32768>
  %9 = and <4 x i32> %8, <i32 2147418112, i32 2147418112, i32 2147418112, i32 2147418112>
  %10 = sub nsw <4 x i32> <i32 655360000, i32 655360000, i32 655360000, i32 655360000>, %9
  %11 = ashr exact <4 x i32> %10, <i32 16, i32 16, i32 16, i32 16>
  %12 = trunc <4 x i32> %11 to <4 x i16>
  store <4 x i16> %12, ptr undef, align 4
  ret void
}
