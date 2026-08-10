; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %variable
; CHECK-NEXT: %ret = call <8 x i1> @llvm.vp.is.fpclass.v8f64(<8 x double> %x, i32 %variable, <8 x i1> %m, i32 %evl)
define <8 x i1> @test_mask_variable(<8 x double> %x, i32 %variable, <8 x i1> %m, i32 zeroext %evl) {
  %ret = call <8 x i1> @llvm.vp.is.fpclass.v8f64(<8 x double> %x, i32 %variable, <8 x i1> %m, i32 %evl) 
  ret <8 x i1> %ret
}

; CHECK: unsupported bits for llvm.vp.is.fpclass test mask
define <8 x i1> @test_mask_neg1(<8 x double> %x, <8 x i1> %m, i32 zeroext %evl) {
  %ret = call <8 x i1> @llvm.vp.is.fpclass.v8f64(<8 x double> %x, i32 -1, <8 x i1> %m, i32 %evl)
  ret <8 x i1> %ret
}

; CHECK: unsupported bits for llvm.vp.is.fpclass test mask
define <8 x i1> @test_mask_bit11(<8 x double> %x, <8 x i1> %m, i32 zeroext %evl) {
  %ret = call <8 x i1> @llvm.vp.is.fpclass.v8f64(<8 x double> %x, i32 2048, <8 x i1> %m, i32 %evl)
  ret <8 x i1> %ret
}

declare <8 x i1> @llvm.vp.is.fpclass.v8f64(<8 x double>, i32, <8 x i1>, i32)
