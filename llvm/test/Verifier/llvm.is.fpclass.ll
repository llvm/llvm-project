; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %variable
; CHECK-NEXT: %ret = call i1 @llvm.is.fpclass.f64(double %val, i32 %variable)
define i1 @test_mask_variable(double %val, i32 %variable) {
  %ret = call i1 @llvm.is.fpclass.f64(double %val, i32 %variable)
  ret i1 %ret
}

; CHECK: unsupported bits for llvm.is.fpclass test mask
define i1 @test_mask_neg1(double %val) {
  %ret = call i1 @llvm.is.fpclass.f64(double %val, i32 -1)
  ret i1 %ret
}

; CHECK: unsupported bits for llvm.is.fpclass test mask
define i1 @test_mask_bit11(double %val) {
  %ret = call i1 @llvm.is.fpclass.f64(double %val, i32 2048)
  ret i1 %ret
}

declare i1 @llvm.is.fpclass.f64(double, i32 immarg)
