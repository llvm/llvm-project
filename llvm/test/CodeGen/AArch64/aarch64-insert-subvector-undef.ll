; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s

; Check that this does not ICE.

@d = common dso_local local_unnamed_addr global <4 x i16> zeroinitializer, align 8

define <8 x i16> @c(i32 %e) {
entry:
  %0 = load <4 x i16>, ptr @d, align 8
  %vminv = tail call i32 @llvm.aarch64.neon.uminv.i32.v4i16(<4 x i16> %0)
  %1 = trunc i32 %vminv to i16
  %vecinit3 = insertelement <4 x i16> <i16 undef, i16 undef, i16 0, i16 0>, i16 %1, i32 1
  %call = tail call <8 x i16> @c(i32 0) #3
  %vgetq_lane = extractelement <8 x i16> %call, i32 0
  %vset_lane = insertelement <4 x i16> %vecinit3, i16 %vgetq_lane, i32 0
  %call4 = tail call i32 @k(<4 x i16> %vset_lane) #3
  ret <8 x i16> undef
}

declare i32 @llvm.aarch64.neon.uminv.i32.v4i16(<4 x i16>)
declare i32 @k(...)
