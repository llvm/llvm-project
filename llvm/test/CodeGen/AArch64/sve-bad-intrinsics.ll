; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t
; RUN: FileCheck --check-prefix=CHECK-ERROR %s <%t

declare <4 x float> @llvm.arm.neon.vcvthf2fp(<vscale x 4 x i16>)
declare <vscale x 4 x i16> @llvm.arm.neon.vcvtfp2hf(<vscale x 4 x float>)

; CHECK-ERROR: Intrinsic has incorrect return type!
define <vscale x 4 x i16> @bad1() {
  %r = call <vscale x 4 x i16> @llvm.arm.neon.vcvtfp2hf(<vscale x 4 x float> zeroinitializer)
  ret <vscale x 4 x i16> %r
}

; CHECK-ERROR: Intrinsic has incorrect argument type!
define <4 x float> @bad2() {
  %r = call <4 x float> @llvm.arm.neon.vcvthf2fp(<vscale x 4 x i16> zeroinitializer)
  ret <4 x float> %r
}
