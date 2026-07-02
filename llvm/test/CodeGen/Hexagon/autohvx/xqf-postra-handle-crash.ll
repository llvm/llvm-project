; Tests that the PostRA XQF handle pass does not crash on basic qf16 ops.
;
; RUN: llc -mtriple=hexagon-unknown-elf -mcpu=hexagonv79 \
; RUN:   -mattr=+hvx-length128b,+hvxv79,+hvx-ieee-fp,+hvx-qfloat \
; RUN:   < %s -o /dev/null

declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #0
declare <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32) #0
declare <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpy.qf16.128B(<32 x i32>, <32 x i32>) #0

define i32 @main() {
entry:
  %zero = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 0)
  %a = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 14336)
  %b = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 13312)
  %q1 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %zero, <32 x i32> %a)
  %q2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %zero, <32 x i32> %b)
  %q3 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf16.128B(<32 x i32> %q1, <32 x i32> %q2)
  ret i32 0
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
