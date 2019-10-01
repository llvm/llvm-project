; RUN: not opt -verify -S < %s 2>&1 | FileCheck %s

; LD2 and LD2LANE

; CHECK: Intrinsic has incorrect return type
; CHECK-NEXT: llvm.aarch64.neon.ld2.v4i32
define { <4 x i64>, <4 x i32> } @test_ld2_ret(<4 x i32>* %ptr) {
  %res = call { <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32(<4 x i32>* %ptr)
  ret{ <4 x i64>, <4 x i32> } %res
}
declare { <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32(<4 x i32>* %ptr)

; CHECK: Intrinsic has incorrect return type
; CHECK-NEXT: llvm.aarch64.neon.ld2lane.v4i64
define { <4 x i64>, <4 x i32> } @test_ld2lane_ret(i8* %ptr, <4 x i64> %a, <4 x i64> %b) {
  %res = call { <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i64(<4 x i64> %a, <4 x i64> %b, i64 0, i8* %ptr)
  ret{ <4 x i64>, <4 x i32> } %res
}
declare { <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i64(<4 x i64>, <4 x i64>, i64, i8*)

; CHECK: Intrinsic has incorrect argument type
; CHECK-NEXT: llvm.aarch64.neon.ld2lane.v4i32
define { <4 x i32>, <4 x i32> } @test_ld2lane_arg(i8* %ptr, <4 x i64> %a, <4 x i32> %b) {
  %res = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i32(<4 x i64> %a, <4 x i32> %b, i64 0, i8* %ptr)
  ret{ <4 x i32>, <4 x i32> } %res
}
declare { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i32(<4 x i64>, <4 x i32>, i64, i8*)

; LD3 and LD3LANE

; CHECK: Intrinsic has incorrect return type
; CHECK-NEXT: llvm.aarch64.neon.ld3.v4i32
define { <4 x i32>, <4 x i64>, <4 x i32> } @test_ld3_ret(<4 x i32>* %ptr) {
  %res = call { <4 x i32>, <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32(<4 x i32>* %ptr)
  ret{ <4 x i32>, <4 x i64>, <4 x i32> } %res
}
declare { <4 x i32>, <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32(<4 x i32>* %ptr)

; CHECK: Intrinsic has incorrect return type
; CHECK-NEXT: llvm.aarch64.neon.ld3lane.v4i64
define { <4 x i64>, <4 x i32>, <4 x i64> } @test_ld3lane_ret(i8* %ptr, <4 x i64> %a, <4 x i64> %b, <4 x i64> %c) {
  %res = call { <4 x i64>, <4 x i32>, <4 x i64> } @llvm.aarch64.neon.ld3lane.v4i64(<4 x i64> %a, <4 x i64> %b, <4 x i64> %c, i64 0, i8* %ptr)
  ret{ <4 x i64>, <4 x i32>, <4 x i64> } %res
}
declare { <4 x i64>, <4 x i32>, <4 x i64> } @llvm.aarch64.neon.ld3lane.v4i64(<4 x i64>, <4 x i64>, <4 x i64>, i64, i8*)

; CHECK: Intrinsic has incorrect argument type
; CHECK-NEXT: llvm.aarch64.neon.ld3lane.v4i32
define { <4 x i32>, <4 x i32>, <4 x i32> } @test_ld3lane_arg(i8* %ptr, <4 x i64> %a, <4 x i32> %b, <4 x i32> %c) {
  %res = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3lane.v4i32(<4 x i64> %a, <4 x i32> %b, <4 x i32> %c, i64 0, i8* %ptr)
  ret{ <4 x i32>, <4 x i32>, <4 x i32> } %res
}
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3lane.v4i32(<4 x i64>, <4 x i32>, <4 x i32>, i64, i8*)

; LD4 and LD4LANE

; CHECK: Intrinsic has incorrect return type
; CHECK-NEXT: llvm.aarch64.neon.ld4.v4i32
define { <4 x i32>, <4 x i32>, <4 x i64>, <4 x i32> } @test_ld4_ret(<4 x i32>* %ptr) {
  %res = call { <4 x i32>, <4 x i32>, <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld4.v4i32(<4 x i32>* %ptr)
  ret{ <4 x i32>, <4 x i32>, <4 x i64>, <4 x i32> } %res
}
declare { <4 x i32>, <4 x i32>, <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld4.v4i32(<4 x i32>* %ptr)

; CHECK: Intrinsic has incorrect return type
; CHECK-NEXT: llvm.aarch64.neon.ld4lane.v4i64
define { <4 x i64>, <4 x i64>, <4 x i32>, <4 x i64> } @test_ld4lane_ret(i8* %ptr, <4 x i64> %a, <4 x i64> %b, <4 x i64> %c, <4 x i64> %d) {
  %res = call { <4 x i64>, <4 x i64>, <4 x i32>, <4 x i64> } @llvm.aarch64.neon.ld4lane.v4i64(<4 x i64> %a, <4 x i64> %b, <4 x i64> %c, <4 x i64> %d, i64 0, i8* %ptr)
  ret{ <4 x i64>, <4 x i64>, <4 x i32>, <4 x i64> } %res
}
declare { <4 x i64>, <4 x i64>, <4 x i32>, <4 x i64> } @llvm.aarch64.neon.ld4lane.v4i64(<4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, i64, i8*)

; CHECK: Intrinsic has incorrect argument type
; CHECK-NEXT: llvm.aarch64.neon.ld4lane.v4i32
define { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @test_ld4lane_arg(i8* %ptr, <4 x i64> %a, <4 x i32> %b, <4 x i32> %c, <4 x i32> %d) {
  %res = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4lane.v4i32(<4 x i64> %a, <4 x i32> %b, <4 x i32> %c, <4 x i32> %d, i64 0, i8* %ptr)
  ret{ <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %res
}
declare { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4lane.v4i32(<4 x i64>, <4 x i32>, <4 x i32>, <4 x i32>, i64, i8*)
