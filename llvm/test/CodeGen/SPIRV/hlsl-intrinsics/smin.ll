; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"


define noundef i16 @test_smin_i16(i16 noundef %a, i16 noundef %b) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] SMin %[[#]] %[[#]]
  %0 = call i16 @llvm.smin.i16(i16 %a, i16 %b)
  ret i16 %0
}


define noundef i32 @test_smin_i32(i32 noundef %a, i32 noundef %b) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] SMin %[[#]] %[[#]]
  %0 = call i32 @llvm.smin.i32(i32 %a, i32 %b)
  ret i32 %0
}


define noundef i64 @test_smin_i64(i64 noundef %a, i64 noundef %b) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] SMin %[[#]] %[[#]]
  %0 = call i64 @llvm.smin.i64(i64 %a, i64 %b)
  ret i64 %0
}

declare i16 @llvm.smin.i16(i16, i16)
declare i32 @llvm.smin.i32(i32, i32)
declare i64 @llvm.smin.i64(i64, i64)
