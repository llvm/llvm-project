// RUN: not llvm-mc -triple amdgcn-- -mcpu gfx900 %s -o - 2>&1 | FileCheck %s

// CHECK: :[[@LINE+1]]:27: error: size is too large
        .amdgpu_lds huge, 200000

// CHECK: :[[@LINE+1]]:30: error: size must be non-negative
        .amdgpu_lds negsize, -4

// CHECK: :[[@LINE+1]]:36: error: alignment must be a power of two
        .amdgpu_lds zero_align, 5, 0

// CHECK: :[[@LINE+1]]:39: error: alignment must be a power of two
        .amdgpu_lds non_pot_align, 0, 12

// CHECK: :[[@LINE+1]]:36: error: alignment is too large
        .amdgpu_lds huge_align, 0, 1099511627776

// CHECK: :[[@LINE+1]]:9: error: unknown directive
        .amdgpu_ldsnowhitespace, 8
