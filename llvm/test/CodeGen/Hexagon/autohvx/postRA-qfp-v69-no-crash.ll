; Test that the PostRA QFP handle pass does not crash
; on <v79 with qfloat intrinsics.
;
; REQUIRES: asserts
; RUN: llc -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv69 \
; RUN:   -mattr=+hvx-length128b,+hvxv69 -O2 -filetype=obj \
; RUN:   < %s -o /dev/null

define <32 x i32> @main() {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.mix.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %puts = call i32 @puts()
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %0)
  ret <32 x i32> %1
}

declare <32 x i32> @llvm.hexagon.V6.vadd.qf32.mix.128B(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32>) #0
declare i32 @puts()

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
