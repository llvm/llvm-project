; RUN: opt < %s -cost-model -analyze -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+vsx | FileCheck %s
; RUN: opt < %s -cost-model -analyze -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr9 -mattr=+vsx | FileCheck --check-prefix=CHECK-P9 %s

define void @testi16(i16 %arg1, i16 %arg2, i16* %arg3) {

  %s1 = add i16 %arg1, %arg2
  %s2 = zext i16 %arg1 to i32
  %s3 = load i16, i16* %arg3
  store i16 %arg2, i16* %arg3
  %c = icmp eq i16 %arg1, %arg2

  ret void
  ; CHECK: cost of 1 {{.*}} add
  ; CHECK: cost of 1 {{.*}} zext
  ; CHECK: cost of 1 {{.*}} load
  ; CHECK: cost of 1 {{.*}} store
  ; CHECK: cost of 1 {{.*}} icmp
  ; CHECK-P9: cost of 1 {{.*}} add
  ; CHECK-P9: cost of 1 {{.*}} zext
  ; CHECK-P9: cost of 1 {{.*}} load
  ; CHECK-P9: cost of 1 {{.*}} store
  ; CHECK-P9: cost of 1 {{.*}} icmp
}

define void @test4xi16(<4 x i16> %arg1, <4 x i16> %arg2) {

  %v1 = add <4 x i16> %arg1, %arg2
  %v2 = zext <4 x i16> %arg1 to <4 x i32>
  %v3 = shufflevector <4 x i16> %arg1, <4 x i16> undef, <4 x i32> zeroinitializer
  %c = icmp eq <4 x i16> %arg1, %arg2

  ret void
  ; CHECK: cost of 1 {{.*}} add
  ; CHECK: cost of 1 {{.*}} zext
  ; CHECK: cost of 1 {{.*}} shufflevector
  ; CHECK: cost of 1 {{.*}} icmp
  ; CHECK-P9: cost of 2 {{.*}} add
  ; CHECK-P9: cost of 2 {{.*}} zext
  ; CHECK-P9: cost of 2 {{.*}} shufflevector
  ; CHECK-P9: cost of 2 {{.*}} icmp
}

define void @test4xi32(<4 x i32> %arg1, <4 x i32> %arg2, <4 x i32>* %arg3) {

  %v1 = load <4 x i32>, <4 x i32>* %arg3
  store <4 x i32> %arg2, <4 x i32>* %arg3

  ret void
  ; CHECK: cost of 1 {{.*}} load
  ; CHECK: cost of 1 {{.*}} store
  ; CHECK-P9: cost of 2 {{.*}} load
  ; CHECK-P9: cost of 2 {{.*}} store
}
