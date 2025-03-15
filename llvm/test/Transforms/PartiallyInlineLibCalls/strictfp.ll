; RUN: opt -S -passes=partially-inline-libcalls -mtriple=x86_64-unknown-linux-gnu -pass-remarks-missed=partially-inline-libcalls 2>%t < %s | FileCheck %s
; RUN: cat %t | FileCheck %s -check-prefix=CHECK-REMARK

define float @f(float %val) strictfp {
; CHECK-LABEL: @f
; CHECK-REMARK: Could not consider library function for partial inlining: strict FP exception behavior is active
; CHECK: call{{.*}}@sqrtf
; CHECK-NOT: call{{.*}}@sqrtf
  %res = tail call float @sqrtf(float %val) strictfp
  ret float %res
}

declare float @sqrtf(float)
