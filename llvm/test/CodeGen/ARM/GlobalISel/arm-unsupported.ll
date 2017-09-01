; RUN: llc -mtriple arm-unknown -verify-machineinstrs -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' %s -o - 2>&1 | FileCheck %s -check-prefixes=CHECK
; RUN: llc -mtriple arm-unknown -verify-machineinstrs -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -relocation-model=pic %s -o - 2>&1 | FileCheck %s -check-prefixes=PIC
; RUN: llc -mtriple arm-unknown -verify-machineinstrs -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -relocation-model=ropi %s -o - 2>&1 | FileCheck %s -check-prefixes=ROPI
; RUN: llc -mtriple arm-unknown -verify-machineinstrs -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -relocation-model=rwpi %s -o - 2>&1 | FileCheck %s -check-prefixes=RWPI
; RUN: llc -mtriple arm-unknown -verify-machineinstrs -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -relocation-model=ropi-rwpi %s -o - 2>&1 | FileCheck %s -check-prefixes=ROPI-RWPI

; This file checks that we use the fallback path for things that are known to
; be unsupported on the ARM target. It should progressively shrink in size.

define <4 x i32> @test_int_vectors(<4 x i32> %a, <4 x i32> %b) {
; CHECK: remark: {{.*}} unable to lower arguments: <4 x i32> (<4 x i32>, <4 x i32>)*
; CHECK-LABEL: warning: Instruction selection used fallback path for test_int_vectors
  %res = add <4 x i32> %a, %b
  ret <4 x i32> %res
}

define <4 x float> @test_float_vectors(<4 x float> %a, <4 x float> %b) {
; CHECK: remark: {{.*}} unable to lower arguments: <4 x float> (<4 x float>, <4 x float>)*
; CHECK-LABEL: warning: Instruction selection used fallback path for test_float_vectors
  %res = fadd <4 x float> %a, %b
  ret <4 x float> %res
}

define i64 @test_i64(i64 %a, i64 %b) {
; CHECK: remark: {{.*}} unable to lower arguments: i64 (i64, i64)*
; CHECK-LABEL: warning: Instruction selection used fallback path for test_i64
  %res = add i64 %a, %b
  ret i64 %res
}

define i128 @test_i128(i128 %a, i128 %b) {
; CHECK: remark: {{.*}} unable to lower arguments: i128 (i128, i128)*
; CHECK-LABEL: warning: Instruction selection used fallback path for test_i128
  %res = add i128 %a, %b
  ret i128 %res
}

define i17 @test_funny_ints(i17 %a, i17 %b) {
; CHECK: remark: {{.*}} unable to lower arguments: i17 (i17, i17)*
; CHECK-LABEL: warning: Instruction selection used fallback path for test_funny_ints
  %res = add i17 %a, %b
  ret i17 %res
}

define half @test_half(half %a, half %b) {
; CHECK: remark: {{.*}} unable to lower arguments: half (half, half)*
; CHECK-LABEL: warning: Instruction selection used fallback path for test_half
  %res = fadd half %a, %b
  ret half %res
}

declare [16 x i32] @ret_demotion_target()

define [16 x i32] @test_ret_demotion() {
; CHECK: remark: {{.*}} unable to translate instruction: call{{.*}} @ret_demotion_target
; CHECK-LABEL: warning: Instruction selection used fallback path for test_ret_demotion
  %res = call [16 x i32] @ret_demotion_target()
  ret [16 x i32] %res
}

%large.struct = type { i32, i32, i32, i32, i32} ; Doesn't fit in R0-R3

declare %large.struct @large_struct_return_target()

define %large.struct @test_large_struct_return() {
; CHECK: remark: {{.*}} unable to translate instruction: call{{.*}} @large_struct_return_target
; CHECK-LABEL: warning: Instruction selection used fallback path for test_large_struct_return
  %r = call %large.struct @large_struct_return_target()
  ret %large.struct %r
}

%mixed.struct = type {i32*, float, i32}

define %mixed.struct @test_mixed_struct(%mixed.struct %x) {
; CHECK: remark: {{.*}} unable to lower arguments: %mixed.struct (%mixed.struct)*
; CHECK-LABEL: warning: Instruction selection used fallback path for test_mixed_struct
  ret %mixed.struct %x
}

define void @test_vararg_definition(i32 %a, ...) {
; CHECK: remark: {{.*}} unable to lower arguments: void (i32, ...)*
; CHECK-LABEL: warning: Instruction selection used fallback path for test_vararg_definition
  ret void
}

define void @test_vararg_call(i32 %a) {
; CHECK: remark: {{.*}} unable to translate instruction: call
; CHECK-LABEL: warning: Instruction selection used fallback path for test_vararg_call
  call void(i32, ...) @test_vararg_definition(i32 %a, i32 %a, i32 %a)
  ret void
}

define i32 @test_thumb(i32 %a) #0 {
; CHECK: remark: {{.*}} unable to lower arguments: i32 (i32)*
; CHECK-LABEL: warning: Instruction selection used fallback path for test_thumb
  ret i32 %a
}

@thread_local_global = thread_local global i32 42

define i32 @test_thread_local_global() {
; CHECK: remark: {{.*}} cannot select: {{.*}} G_GLOBAL_VALUE
; CHECK-LABEL: warning: Instruction selection used fallback path for test_thread_local_global
; PIC: remark: {{.*}} cannot select: {{.*}} G_GLOBAL_VALUE
; PIC-LABEL: warning: Instruction selection used fallback path for test_thread_local_global
; ROPI: remark: {{.*}} cannot select: {{.*}} G_GLOBAL_VALUE
; ROPI-LABEL: warning: Instruction selection used fallback path for test_thread_local_global
; RWPI: remark: {{.*}} cannot select: {{.*}} G_GLOBAL_VALUE
; RWPI-LABEL: warning: Instruction selection used fallback path for test_thread_local_global
; ROPI-RWPI: remark: {{.*}} cannot select: {{.*}} G_GLOBAL_VALUE
; ROPI-RWPI-LABEL: warning: Instruction selection used fallback path for test_thread_local_global
  %v = load i32, i32* @thread_local_global
  ret i32 %v
}

@a_global = external global i32

define i32 @test_global_reloc_models() {
; This is only unsupported for the RWPI relocation modes.
; RWPI: remark: {{.*}} cannot select: {{.*}} G_GLOBAL_VALUE
; RWPI-LABEL: warning: Instruction selection used fallback path for test_global_reloc_models
; ROPI-RWPI: remark: {{.*}} cannot select: {{.*}} G_GLOBAL_VALUE
; ROPI-RWPI-LABEL: warning: Instruction selection used fallback path for test_global_reloc_models
  %v = load i32, i32* @a_global
  ret i32 %v
}

attributes #0 = { "target-features"="+thumb-mode" }
