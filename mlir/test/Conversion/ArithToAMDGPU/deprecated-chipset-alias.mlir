// RUN: mlir-opt %s -convert-arith-to-amdgpu='arch=gfx1030 chipset=gfx942' \
// RUN: | FileCheck %s --check-prefix=ARCH-WINS
// RUN: mlir-opt %s -convert-arith-to-amdgpu='chipset=gfx942' \
// RUN: | FileCheck %s --check-prefix=ALIAS

// Errors name whichever option supplied the target, so a stale `chipset` is
// reported by its own value rather than by `arch`'s unusable default.
// RUN: not mlir-opt %s -convert-arith-to-amdgpu='chipset=gfx999' 2>&1 \
// RUN: | FileCheck %s --check-prefix=BAD-ALIAS
// BAD-ALIAS: 'gfx999' is not a valid AMDGPU architecture

// With neither given, the unusable default is still what gets reported.
// RUN: not mlir-opt %s -convert-arith-to-amdgpu 2>&1 \
// RUN: | FileCheck %s --check-prefix=NEITHER
// NEITHER: 'invalid' is not a valid AMDGPU architecture

// ARCH-WINS-LABEL: func @truncf_to_fp8
// ARCH-WINS: arith.truncf
// ARCH-WINS-NOT: amdgpu.packed_trunc_2xfp8

// ALIAS-LABEL: func @truncf_to_fp8
// ALIAS: amdgpu.packed_trunc_2xfp8
// ALIAS-NOT: arith.truncf
func.func @truncf_to_fp8(%x: f32) -> f8E4M3FNUZ {
  %r = arith.truncf %x : f32 to f8E4M3FNUZ
  return %r : f8E4M3FNUZ
}
