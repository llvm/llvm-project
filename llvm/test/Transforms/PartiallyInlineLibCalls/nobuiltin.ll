; RUN: opt -S -passes=partially-inline-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define float @f(float %val) {
; CHECK-LABEL: @f
; CHECK: call{{.*}}@sqrtf
; CHECK-NOT: call{{.*}}@sqrtf
  %res = tail call float @sqrtf(float %val) nobuiltin
  ret float %res
}

declare float @sqrtf(float)
