; RUN: llc -mtriple arm-unknown -mattr=+vfp2,+v6 -global-isel %s -o - | FileCheck %s

define void @test_void_return() {
; CHECK-LABEL: test_void_return:
; CHECK: bx lr
entry:
  ret void
}

define i32 @test_constant_return_i32() {
; CHECK-LABEL: test_constant_return_i32:
; CHECK: mov r0, #42
; CHECK: bx lr
entry:
  ret i32 42
}

define zeroext i1 @test_zext_i1(i1 %x) {
; CHECK-LABEL: test_zext_i1
; CHECK: and r0, r0, #1
; CHECK: bx lr
entry:
  ret i1 %x
}

define signext i1 @test_sext_i1(i1 %x) {
; CHECK-LABEL: test_sext_i1
; CHECK: and r0, r0, #1
; CHECK: rsb r0, r0, #0
; CHECK: bx lr
entry:
  ret i1 %x
}

define zeroext i8 @test_ext_i8(i8 %x) {
; CHECK-LABEL: test_ext_i8:
; CHECK: uxtb r0, r0
; CHECK: bx lr
entry:
  ret i8 %x
}

define signext i16 @test_ext_i16(i16 %x) {
; CHECK-LABEL: test_ext_i16:
; CHECK: sxth r0, r0
; CHECK: bx lr
entry:
  ret i16 %x
}

define void @test_trunc_i32_i16(i32 %v, i16 *%p) {
; CHECK-LABEL: test_trunc_i32_i16:
; The trunc doesn't result in any instructions, but we
; expect the store to be explicitly 16-bit.
; CHECK: strh r0, [r1]
; CHECK: bx lr
entry:
  %v16 = trunc i32 %v to i16
  store i16 %v16, i16 *%p
  ret void
}

define void @test_trunc_i32_i8(i32 %v, i8 *%p) {
; CHECK-LABEL: test_trunc_i32_i8:
; The trunc doesn't result in any instructions, but we
; expect the store to be explicitly 8-bit.
; CHECK: strb r0, [r1]
; CHECK: bx lr
entry:
  %v8 = trunc i32 %v to i8
  store i8 %v8, i8 *%p
  ret void
}

define i8 @test_add_i8(i8 %x, i8 %y) {
; CHECK-LABEL: test_add_i8:
; CHECK: add r0, r0, r1
; CHECK: bx lr
entry:
  %sum = add i8 %x, %y
  ret i8 %sum
}

define i16 @test_add_i16(i16 %x, i16 %y) {
; CHECK-LABEL: test_add_i16:
; CHECK: add r0, r0, r1
; CHECK: bx lr
entry:
  %sum = add i16 %x, %y
  ret i16 %sum
}

define i32 @test_add_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_add_i32:
; CHECK: add r0, r0, r1
; CHECK: bx lr
entry:
  %sum = add i32 %x, %y
  ret i32 %sum
}

define i8 @test_sub_i8(i8 %x, i8 %y) {
; CHECK-LABEL: test_sub_i8:
; CHECK: sub r0, r0, r1
; CHECK: bx lr
entry:
  %sum = sub i8 %x, %y
  ret i8 %sum
}

define i16 @test_sub_i16(i16 %x, i16 %y) {
; CHECK-LABEL: test_sub_i16:
; CHECK: sub r0, r0, r1
; CHECK: bx lr
entry:
  %sum = sub i16 %x, %y
  ret i16 %sum
}

define i32 @test_sub_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_sub_i32:
; CHECK: sub r0, r0, r1
; CHECK: bx lr
entry:
  %sum = sub i32 %x, %y
  ret i32 %sum
}

define i8 @test_mul_i8(i8 %x, i8 %y) {
; CHECK-LABEL: test_mul_i8:
; CHECK: mul r0, r0, r1
; CHECK: bx lr
entry:
  %sum = mul i8 %x, %y
  ret i8 %sum
}

define i16 @test_mul_i16(i16 %x, i16 %y) {
; CHECK-LABEL: test_mul_i16:
; CHECK: mul r0, r0, r1
; CHECK: bx lr
entry:
  %sum = mul i16 %x, %y
  ret i16 %sum
}

define i32 @test_mul_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_mul_i32:
; CHECK: mul r0, r0, r1
; CHECK: bx lr
entry:
  %sum = mul i32 %x, %y
  ret i32 %sum
}

define i8 @test_and_i8(i8 %x, i8 %y) {
; CHECK-LABEL: test_and_i8:
; CHECK: and r0, r0, r1
; CHECK: bx lr
entry:
  %sum = and i8 %x, %y
  ret i8 %sum
}

define i16 @test_and_i16(i16 %x, i16 %y) {
; CHECK-LABEL: test_and_i16:
; CHECK: and r0, r0, r1
; CHECK: bx lr
entry:
  %sum = and i16 %x, %y
  ret i16 %sum
}

define i32 @test_and_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_and_i32:
; CHECK: and r0, r0, r1
; CHECK: bx lr
entry:
  %sum = and i32 %x, %y
  ret i32 %sum
}

define i8 @test_or_i8(i8 %x, i8 %y) {
; CHECK-LABEL: test_or_i8:
; CHECK: orr r0, r0, r1
; CHECK: bx lr
entry:
  %sum = or i8 %x, %y
  ret i8 %sum
}

define i16 @test_or_i16(i16 %x, i16 %y) {
; CHECK-LABEL: test_or_i16:
; CHECK: orr r0, r0, r1
; CHECK: bx lr
entry:
  %sum = or i16 %x, %y
  ret i16 %sum
}

define i32 @test_or_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_or_i32:
; CHECK: orr r0, r0, r1
; CHECK: bx lr
entry:
  %sum = or i32 %x, %y
  ret i32 %sum
}

define i8 @test_xor_i8(i8 %x, i8 %y) {
; CHECK-LABEL: test_xor_i8:
; CHECK: eor r0, r0, r1
; CHECK: bx lr
entry:
  %sum = xor i8 %x, %y
  ret i8 %sum
}

define i16 @test_xor_i16(i16 %x, i16 %y) {
; CHECK-LABEL: test_xor_i16:
; CHECK: eor r0, r0, r1
; CHECK: bx lr
entry:
  %sum = xor i16 %x, %y
  ret i16 %sum
}

define i32 @test_xor_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_xor_i32:
; CHECK: eor r0, r0, r1
; CHECK: bx lr
entry:
  %sum = xor i32 %x, %y
  ret i32 %sum
}

define i32 @test_stack_args_i32(i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5) {
; CHECK-LABEL: test_stack_args_i32:
; CHECK: add [[P5ADDR:r[0-9]+]], sp, #4
; CHECK: ldr [[P5:r[0-9]+]], {{.*}}[[P5ADDR]]
; CHECK: add r0, r2, [[P5]]
; CHECK: bx lr
entry:
  %sum = add i32 %p2, %p5
  ret i32 %sum
}

define i16 @test_stack_args_mixed(i32 %p0, i16 %p1, i8 %p2, i1 %p3, i8 %p4, i16 %p5) {
; CHECK-LABEL: test_stack_args_mixed:
; CHECK: add [[P5ADDR:r[0-9]+]], sp, #4
; CHECK: ldrh [[P5:r[0-9]+]], {{.*}}[[P5ADDR]]
; CHECK: add r0, r1, [[P5]]
; CHECK: bx lr
entry:
  %sum = add i16 %p1, %p5
  ret i16 %sum
}

define i16 @test_stack_args_zeroext(i32 %p0, i16 %p1, i8 %p2, i1 %p3, i16 zeroext %p4) {
; CHECK-LABEL: test_stack_args_zeroext:
; CHECK: mov [[P4ADDR:r[0-9]+]], sp
; CHECK: ldr [[P4:r[0-9]+]], {{.*}}[[P4ADDR]]
; CHECK: add r0, r1, [[P4]]
; CHECK: bx lr
entry:
  %sum = add i16 %p1, %p4
  ret i16 %sum
}

define i8 @test_stack_args_signext(i32 %p0, i16 %p1, i8 %p2, i1 %p3, i8 signext %p4) {
; CHECK-LABEL: test_stack_args_signext:
; CHECK: mov [[P4ADDR:r[0-9]+]], sp
; CHECK: ldr [[P4:r[0-9]+]], {{.*}}[[P4ADDR]]
; CHECK: add r0, r2, [[P4]]
; CHECK: bx lr
entry:
  %sum = add i8 %p2, %p4
  ret i8 %sum
}

define i8 @test_stack_args_noext(i32 %p0, i16 %p1, i8 %p2, i1 %p3, i8 %p4) {
; CHECK-LABEL: test_stack_args_noext:
; CHECK: mov [[P4ADDR:r[0-9]+]], sp
; CHECK: ldrb [[P4:r[0-9]+]], {{.*}}[[P4ADDR]]
; CHECK: add r0, r2, [[P4]]
; CHECK: bx lr
entry:
  %sum = add i8 %p2, %p4
  ret i8 %sum
}

define i32 @test_ptr_arg_in_reg(i32* %p) {
; CHECK-LABEL: test_ptr_arg_in_reg:
; CHECK: ldr r0, [r0]
; CHECK: bx lr
entry:
  %v = load i32, i32* %p
  ret i32 %v
}

define i32 @test_ptr_arg_on_stack(i32 %f0, i32 %f1, i32 %f2, i32 %f3, i32* %p) {
; CHECK-LABEL: test_ptr_arg_on_stack:
; CHECK: mov r0, sp
; CHECK: ldr r0, [r0]
; CHECK: ldr r0, [r0]
; CHECK: bx lr
entry:
  %v = load i32, i32* %p
  ret i32 %v
}

define i8* @test_ptr_ret(i8** %p) {
; CHECK-LABEL: test_ptr_ret:
; CHECK: ldr r0, [r0]
; CHECK: bx lr
entry:
  %v = load i8*, i8** %p
  ret i8* %v
}

define arm_aapcs_vfpcc float @test_float_hard(float %f0, float %f1) {
; CHECK-LABEL: test_float_hard:
; CHECK: vadd.f32 s0, s0, s1
; CHECK: bx lr
entry:
  %v = fadd float %f0, %f1
  ret float %v
}

define arm_aapcscc float @test_float_softfp(float %f0, float %f1) {
; CHECK-LABEL: test_float_softfp:
; CHECK-DAG: vmov [[F0:s[0-9]+]], r0
; CHECK-DAG: vmov [[F1:s[0-9]+]], r1
; CHECK: vadd.f32 [[FV:s[0-9]+]], [[F0]], [[F1]]
; CHECK: vmov r0, [[FV]]
; CHECK: bx lr
entry:
  %v = fadd float %f0, %f1
  ret float %v
}

define arm_aapcs_vfpcc double @test_double_hard(double %f0, double %f1) {
; CHECK-LABEL: test_double_hard:
; CHECK: vadd.f64 d0, d0, d1
; CHECK: bx lr
entry:
  %v = fadd double %f0, %f1
  ret double %v
}

define arm_aapcscc double @test_double_softfp(double %f0, double %f1) {
; CHECK-LABEL: test_double_softfp:
; CHECK-DAG: vmov [[F0:d[0-9]+]], r0, r1
; CHECK-DAG: vmov [[F1:d[0-9]+]], r2, r3
; CHECK: vadd.f64 [[FV:d[0-9]+]], [[F0]], [[F1]]
; CHECK: vmov r0, r1, [[FV]]
; CHECK: bx lr
entry:
  %v = fadd double %f0, %f1
  ret double %v
}

define arm_aapcscc i32 @test_cmp_i32_eq(i32 %a, i32 %b) {
; CHECK-LABEL: test_cmp_i32_eq:
; CHECK: mov [[V:r[0-9]+]], #0
; CHECK: cmp r0, r1
; CHECK: moveq [[V]], #1
; CHECK: and r0, [[V]], #1
; CHECK: bx lr
entry:
  %v = icmp eq i32 %a, %b
  %r = zext i1 %v to i32
  ret i32 %r
}

define arm_aapcscc i32 @test_cmp_ptr_neq(double *%a, double *%b) {
; CHECK-LABEL: test_cmp_ptr_neq:
; CHECK: mov [[V:r[0-9]+]], #0
; CHECK: cmp r0, r1
; CHECK: movne [[V]], #1
; CHECK: and r0, [[V]], #1
; CHECK: bx lr
entry:
  %v = icmp ne double * %a, %b
  %r = zext i1 %v to i32
  ret i32 %r
}

define arm_aapcscc i32 @test_cmp_i16_slt(i16 %a, i16 %b) {
; CHECK-LABEL: test_cmp_i16_slt:
; CHECK-DAG: sxth r0, r0
; CHECK-DAG: sxth r1, r1
; CHECK-DAG: mov [[V:r[0-9]+]], #0
; CHECK: cmp r0, r1
; CHECK: movlt [[V]], #1
; CHECK: and r0, [[V]], #1
; CHECK: bx lr
entry:
  %v = icmp slt i16 %a, %b
  %r = zext i1 %v to i32
  ret i32 %r
}
