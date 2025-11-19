; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -filetype=obj -o - | spirv-val %}
; XFAIL: *
;@llvm.sadd.with.overflow and @llvm.ssub.with.overflow has not been implemented.

define spir_func void @test_sadd_overflow(ptr %out_result, ptr %out_overflow, i32 %a, i32 %b) {
entry:
  %res = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %res, 0
  %ofl = extractvalue { i32, i1 } %res, 1
  store i32 %val, ptr %out_result
  %zext_ofl = zext i1 %ofl to i8
  store i8 %zext_ofl, ptr %out_overflow
  ret void
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

define spir_func void @test_ssub_overflow(ptr %out_result, ptr %out_overflow, i32 %a, i32 %b) {
entry:
  %res = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %res, 0
  %ofl = extractvalue { i32, i1 } %res, 1
  store i32 %val, ptr %out_result
  %zext_ofl = zext i1 %ofl to i8
  store i8 %zext_ofl, ptr %out_overflow
  ret void
}

declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)
