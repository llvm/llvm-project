; RUN: opt %s -passes='function(scalarizer<load-store>)' -S | FileCheck %s

; CHECK-LABEL: @test_vector_half_frexp_half
define noundef <2 x half> @test_vector_half_frexp_half(<2 x half> noundef %h) {
  ; CHECK: [[ee0:%.*]] = extractelement <2 x half> %h, i64 0
  ; CHECK-NEXT: [[ie0:%.*]] = call { half, i32 } @llvm.frexp.f16.i32(half [[ee0]])
  ; CHECK-NEXT: [[ee1:%.*]] = extractelement <2 x half> %h, i64 1
  ; CHECK-NEXT: [[ie1:%.*]] = call { half, i32 } @llvm.frexp.f16.i32(half [[ee1]])
  ; CHECK-NEXT: [[ev00:%.*]] = extractvalue { half, i32 } [[ie0]], 0
  ; CHECK-NEXT: [[ev01:%.*]] = extractvalue { half, i32 } [[ie1]], 0
  ; CHECK-NEXT: insertelement <2 x half> poison, half [[ev00]], i64 0
  ; CHECK-NEXT: insertelement <2 x half> %{{.*}}, half [[ev01]], i64 1
  %r =  call { <2 x half>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x half> %h)
  %e0 = extractvalue { <2 x half>, <2 x i32> } %r, 0
  ret <2 x half> %e0
}

; CHECK-LABEL: @test_vector_half_frexp_int
define noundef <2 x i32> @test_vector_half_frexp_int(<2 x half> noundef %h) {
  ; CHECK: [[ee0:%.*]] = extractelement <2 x half> %h, i64 0
  ; CHECK-NEXT: [[ie0:%.*]] = call { half, i32 } @llvm.frexp.f16.i32(half [[ee0]])
  ; CHECK-NEXT: [[ee1:%.*]] = extractelement <2 x half> %h, i64 1
  ; CHECK-NEXT: [[ie1:%.*]] = call { half, i32 } @llvm.frexp.f16.i32(half [[ee1]])
  ; CHECK-NEXT: [[ev10:%.*]] = extractvalue { half, i32 } [[ie0]], 1
  ; CHECK-NEXT: [[ev11:%.*]] = extractvalue { half, i32 } [[ie1]], 1
  ; CHECK-NEXT: insertelement <2 x i32> poison, i32 [[ev10]], i64 0
  ; CHECK-NEXT: insertelement <2 x i32> %{{.*}}, i32 [[ev11]], i64 1
  %r =  call { <2 x half>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x half> %h)
  %e1 = extractvalue { <2 x half>, <2 x i32> } %r, 1
  ret <2 x i32> %e1
}

; CHECK-LABEL: @test_vector_float_frexp_int
define noundef <2 x float> @test_vector_float_frexp_int(<2 x float> noundef %f) {
  ; CHECK: [[ee0:%.*]] = extractelement <2 x float> %f, i64 0
  ; CHECK-NEXT: [[ie0:%.*]] = call { float, i32 } @llvm.frexp.f32.i32(float [[ee0]])
  ; CHECK-NEXT: [[ee1:%.*]] = extractelement <2 x float> %f, i64 1
  ; CHECK-NEXT: [[ie1:%.*]] = call { float, i32 } @llvm.frexp.f32.i32(float [[ee1]])
  ; CHECK-NEXT: [[ev00:%.*]] = extractvalue { float, i32 } [[ie0]], 0
  ; CHECK-NEXT: [[ev01:%.*]] = extractvalue { float, i32 } [[ie1]], 0
  ; CHECK-NEXT: insertelement <2 x float> poison, float [[ev00]], i64 0
  ; CHECK-NEXT: insertelement <2 x float> %{{.*}}, float [[ev01]], i64 1
  ; CHECK-NEXT: extractvalue { float, i32 } [[ie0]], 1
  ; CHECK-NEXT: extractvalue { float, i32 } [[ie1]], 1
  %1 =  call { <2 x float>, <2 x i32> } @llvm.frexp.v2f16.v2i32(<2 x float> %f)
  %2 = extractvalue { <2 x float>, <2 x i32> } %1, 0
  %3 = extractvalue { <2 x float>, <2 x i32> } %1, 1
  ret <2 x float> %2
}

; CHECK-LABEL: @test_vector_double_frexp_int
define noundef <2 x double> @test_vector_double_frexp_int(<2 x double> noundef %d) {
  ; CHECK: [[ee0:%.*]] = extractelement <2 x double> %d, i64 0
  ; CHECK-NEXT: [[ie0:%.*]] = call { double, i32 } @llvm.frexp.f64.i32(double [[ee0]])
  ; CHECK-NEXT: [[ee1:%.*]] = extractelement <2 x double> %d, i64 1
  ; CHECK-NEXT: [[ie1:%.*]] = call { double, i32 } @llvm.frexp.f64.i32(double [[ee1]])
  ; CHECK-NEXT: [[ev00:%.*]] = extractvalue { double, i32 } [[ie0]], 0
  ; CHECK-NEXT: [[ev01:%.*]] = extractvalue { double, i32 } [[ie1]], 0
  ; CHECK-NEXT: insertelement <2 x double> poison, double [[ev00]], i64 0
  ; CHECK-NEXT: insertelement <2 x double> %{{.*}}, double [[ev01]], i64 1
  ; CHECK-NEXT: extractvalue { double, i32 } [[ie0]], 1
  ; CHECK-NEXT: extractvalue { double, i32 } [[ie1]], 1
  %1 =  call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %d)
  %2 = extractvalue { <2 x double>, <2 x i32> } %1, 0
  %3 = extractvalue { <2 x double>, <2 x i32> } %1, 1
  ret <2 x double> %2
}
