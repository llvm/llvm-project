// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx906 --split-input-file -verify-diagnostics
// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx942 --split-input-file -verify-diagnostics

// fp8 dot4 is only available on gfx12+.
func.func @dot_fp8_requires_gfx12(%a: vector<4xf8E4M3FN>, %b: vector<4xf8E4M3FN>, %c: f32) -> f32 {
  // expected-error@below {{'amdgpu.dot' op no intrinsic matching dot on the given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.dot'}}
  %r = amdgpu.dot %a * %b + %c : vector<4xf8E4M3FN>, vector<4xf8E4M3FN>, f32
  func.return %r : f32
}

// -----

// fdot2.f16.f16 (f16 accumulator for f16 x f16) requires gfx11+.
func.func @dot_f16_f16_requires_gfx11(%a: vector<2xf16>, %b: vector<2xf16>, %c: f16) -> f16 {
  // expected-error@below {{'amdgpu.dot' op no intrinsic matching dot on the given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.dot'}}
  %r = amdgpu.dot %a * %b + %c : vector<2xf16>, vector<2xf16>, f16
  func.return %r : f16
}

// -----

// fdot2.f32.bf16 is available on gfx11+ and gfx950+.
func.func @dot_f32_bf16_requires_gfx11_or_gfx950(%a: vector<2xbf16>, %b: vector<2xbf16>, %c: f32) -> f32 {
  // expected-error@below {{'amdgpu.dot' op no intrinsic matching dot on the given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.dot'}}
  %r = amdgpu.dot %a * %b + %c : vector<2xbf16>, vector<2xbf16>, f32
  func.return %r : f32
}

// -----

// Mixed-sign integer dot (sudot) requires gfx11+.
func.func @dot_mixed_sign_requires_gfx11(%a: vector<4xi8>, %b: vector<4xi8>, %c: i32) -> i32 {
  // expected-error@below {{'amdgpu.dot' op no intrinsic matching dot on the given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.dot'}}
  %r = amdgpu.dot %a * %b + %c {unsignedB} : vector<4xi8>, vector<4xi8>, i32
  func.return %r : i32
}
