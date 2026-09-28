; RUN: opt -S -mtriple=amdgpu8.03-- -passes=amdgpu-codegenprepare %s | FileCheck -check-prefixes=CHECK,GFX8 %s
; RUN: opt -S -mtriple=amdgpu9.00-- -passes=amdgpu-codegenprepare %s | FileCheck -check-prefixes=CHECK,GFX9 %s

; On GFX9+ a divergent i64 mul whose only user is an add stays a plain mul, so
; the SDAG add combine can fold it into v_mad_[iu]64_[iu]32.

define i64 @umul24_i64_add(i64 %lhs, i64 %rhs, i64 %acc) {
; CHECK-LABEL: @umul24_i64_add(
; GFX8:          %mul = call i64 @llvm.amdgcn.mul.u24.i64(i32 %{{.*}}, i32 %{{.*}})
; GFX9-NOT:      @llvm.amdgcn.mul
; GFX9:          %mul = mul i64 %lhs24, %rhs24
; CHECK:         %add = add i64 %mul, %acc
  %lhs24 = and i64 %lhs, 16777215
  %rhs24 = and i64 %rhs, 16777215
  %mul = mul i64 %lhs24, %rhs24
  %add = add i64 %mul, %acc
  ret i64 %add
}

define i64 @smul24_i64_add(i64 %lhs, i64 %rhs, i64 %acc) {
; CHECK-LABEL: @smul24_i64_add(
; GFX8:          %mul = call i64 @llvm.amdgcn.mul.i24.i64(i32 %{{.*}}, i32 %{{.*}})
; GFX9-NOT:      @llvm.amdgcn.mul
; GFX9:          %mul = mul i64 %lhs24, %rhs24
; CHECK:         %add = add i64 %mul, %acc
  %lhs.shl = shl i64 %lhs, 40
  %lhs24 = ashr i64 %lhs.shl, 40
  %rhs.shl = shl i64 %rhs, 40
  %rhs24 = ashr i64 %rhs.shl, 40
  %mul = mul i64 %lhs24, %rhs24
  %add = add i64 %mul, %acc
  ret i64 %add
}

; A non-add user keeps the mul24 form on every target.
define void @umul24_i64_store(i64 %lhs, i64 %rhs, ptr addrspace(1) %out) {
; CHECK-LABEL: @umul24_i64_store(
; CHECK:         %mul = call i64 @llvm.amdgcn.mul.u24.i64(i32 %{{.*}}, i32 %{{.*}})
  %lhs24 = and i64 %lhs, 16777215
  %rhs24 = and i64 %rhs, 16777215
  %mul = mul i64 %lhs24, %rhs24
  store i64 %mul, ptr addrspace(1) %out
  ret void
}

; A second add user keeps the mul24 form on every target.
define i64 @umul24_i64_two_adds(i64 %lhs, i64 %rhs, i64 %a, i64 %b) {
; CHECK-LABEL: @umul24_i64_two_adds(
; CHECK:         %mul = call i64 @llvm.amdgcn.mul.u24.i64(i32 %{{.*}}, i32 %{{.*}})
  %lhs24 = and i64 %lhs, 16777215
  %rhs24 = and i64 %rhs, 16777215
  %mul = mul i64 %lhs24, %rhs24
  %add0 = add i64 %mul, %a
  %add1 = add i64 %mul, %b
  %r = xor i64 %add0, %add1
  ret i64 %r
}

; A divergent product that fits in 32 bits is not narrowed on GFX9 either.
define i64 @umul16_i64_add(i64 %lhs, i64 %rhs, i64 %acc) {
; CHECK-LABEL: @umul16_i64_add(
; GFX8:          %mul = call i64 @llvm.amdgcn.mul.u24.i64(i32 %{{.*}}, i32 %{{.*}})
; GFX9-NOT:      mul i32
; GFX9:          %mul = mul i64 %lhs16, %rhs16
; CHECK:         %add = add i64 %mul, %acc
  %lhs16 = and i64 %lhs, 65535
  %rhs16 = and i64 %rhs, 65535
  %mul = mul i64 %lhs16, %rhs16
  %add = add i64 %mul, %acc
  ret i64 %add
}

; Uniform values are still narrowed for the scalar unit.
define amdgpu_kernel void @umul16_i64_add_uniform(ptr addrspace(1) %out, i64 %lhs, i64 %rhs, i64 %acc) {
; CHECK-LABEL: @umul16_i64_add_uniform(
; CHECK:         [[MUL:%.*]] = mul i32 %{{.*}}, %{{.*}}
; CHECK:         zext i32 [[MUL]] to i64
  %lhs16 = and i64 %lhs, 65535
  %rhs16 = and i64 %rhs, 65535
  %mul = mul i64 %lhs16, %rhs16
  %add = add i64 %mul, %acc
  store i64 %add, ptr addrspace(1) %out
  ret void
}
