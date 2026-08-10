; RUN: opt -S -passes=partially-inline-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define float @f(float %val) strictfp {
; CHECK-LABEL: @f
; CHECK: call{{.*}}@sqrtf
; CHECK-NOT: call{{.*}}@sqrtf
  %res = tail call float @sqrtf(float %val) strictfp
  ret float %res
}

declare float @sqrtf(float)
