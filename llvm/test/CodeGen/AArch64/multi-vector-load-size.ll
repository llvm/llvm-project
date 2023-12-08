; RUN: llc -mtriple=aarch64-linux-gnu -stop-after=instruction-select < %s | FileCheck %s

%struct.__neon_float32x2x2_t = type { <2 x float>,  <2 x float> }
%struct.__neon_float32x2x3_t = type { <2 x float>,  <2 x float>,  <2 x float> }
%struct.__neon_float32x2x4_t = type { <2 x float>,  <2 x float>, <2 x float>,  <2 x float> }

declare %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld2.v2f32.p0f32(float*)
declare %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld3.v2f32.p0f32(float*)
declare %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld4.v2f32.p0f32(float*)

declare %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld1x2.v2f32.p0f32(float*)
declare %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld1x3.v2f32.p0f32(float*)
declare %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld1x4.v2f32.p0f32(float*)

declare %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld2r.v2f32.p0f32(float*)
declare %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld3r.v2f32.p0f32(float*)
declare %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld4r.v2f32.p0f32(float*)

declare %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld2lane.v2f32.p0f32(<2 x float>, <2 x float>, i64, float*)
declare %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld3lane.v2f32.p0f32(<2 x float>, <2 x float>, <2 x float>, i64, float*)
declare %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld4lane.v2f32.p0f32(<2 x float>, <2 x float>, <2 x float>, <2 x float>, i64, float*)


define %struct.__neon_float32x2x2_t @test_ld2(float* %addr) {
  ; CHECK-LABEL: name: test_ld2
  ; CHECK: LD2Twov2s {{.*}} :: (load (s128) {{.*}})
  %val = call %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld2.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x2_t %val
}

define %struct.__neon_float32x2x3_t @test_ld3(float* %addr) {
  ; CHECK-LABEL: name: test_ld3
  ; CHECK: LD3Threev2s {{.*}} :: (load (s192) {{.*}})
  %val = call %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld3.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x3_t %val
}

define %struct.__neon_float32x2x4_t @test_ld4(float* %addr) {
  ; CHECK-LABEL: name: test_ld4
  ; CHECK: LD4Fourv2s {{.*}} :: (load (s256) {{.*}})
  %val = call %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld4.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x4_t %val
}

define %struct.__neon_float32x2x2_t @test_ld1x2(float* %addr) {
  ; CHECK-LABEL: name: test_ld1x2
  ; CHECK: LD1Twov2s {{.*}} :: (load (s128) {{.*}})
  %val = call %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld1x2.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x2_t %val
}

define %struct.__neon_float32x2x3_t @test_ld1x3(float* %addr) {
  ; CHECK-LABEL: name: test_ld1x3
  ; CHECK: LD1Threev2s {{.*}} :: (load (s192) {{.*}})
  %val = call %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld1x3.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x3_t %val
}

define %struct.__neon_float32x2x4_t @test_ld1x4(float* %addr) {
  ; CHECK-LABEL: name: test_ld1x4
  ; CHECK: LD1Fourv2s {{.*}} :: (load (s256) {{.*}})
  %val = call %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld1x4.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x4_t %val
}

define %struct.__neon_float32x2x2_t @test_ld2r(float* %addr) {
  ; CHECK-LABEL: name: test_ld2r
  ; CHECK: LD2Rv2s {{.*}} :: (load (s64) {{.*}})
  %val = call %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld2r.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x2_t %val
}

define %struct.__neon_float32x2x3_t @test_ld3r(float* %addr) {
  ; CHECK-LABEL: name: test_ld3r
  ; CHECK: LD3Rv2s {{.*}} :: (load (s96) {{.*}})
  %val = call %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld3r.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x3_t %val
}

define %struct.__neon_float32x2x4_t @test_ld4r(float* %addr) {
  ; CHECK-LABEL: name: test_ld4r
  ; CHECK: LD4Rv2s {{.*}} :: (load (s128) {{.*}})
  %val = call %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld4r.v2f32.p0f32(float* %addr)
  ret %struct.__neon_float32x2x4_t %val
}

define %struct.__neon_float32x2x2_t @test_ld2lane(<2 x float> %a, <2 x float> %b, float* %addr) {
  ; CHECK-LABEL: name: test_ld2lane
  ; CHECK: {{.*}} LD2i32 {{.*}}
  %val = call %struct.__neon_float32x2x2_t @llvm.aarch64.neon.ld2lane.v2f32.p0f32(<2 x float> %a, <2 x float> %b, i64 1, float* %addr)
  ret %struct.__neon_float32x2x2_t %val
}

define %struct.__neon_float32x2x3_t @test_ld3lane(<2 x float> %a, <2 x float> %b, <2 x float> %c, float* %addr) {
  ; CHECK-LABEL: name: test_ld3lane
  ; CHECK: {{.*}} LD3i32 {{.*}}
  %val = call %struct.__neon_float32x2x3_t @llvm.aarch64.neon.ld3lane.v2f32.p0f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, i64 1, float* %addr)
  ret %struct.__neon_float32x2x3_t %val
}

define %struct.__neon_float32x2x4_t @test_ld4lane(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %d, float* %addr) {
  ; CHECK-LABEL: name: test_ld4lane
  ; CHECK: {{.*}} LD4i32 {{.*}}
  %val = call %struct.__neon_float32x2x4_t @llvm.aarch64.neon.ld4lane.v2f32.p0f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %d, i64 1, float* %addr)
  ret %struct.__neon_float32x2x4_t %val
}