; RUN: not llc -global-isel=0 -mtriple=amdgpu12.5-amd-amdhsa -filetype=null < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -global-isel-abort=0 -mtriple=amdgpu12.5-amd-amdhsa -filetype=null < %s 2>&1 | FileCheck %s

; CHECK: error: {{.*}} llvm.amdgcn.sad.u8 requires target feature 'sad-insts'
define i32 @test_sad_u8(i32 %a, i32 %b, i32 %c) {
  %result = call i32 @llvm.amdgcn.sad.u8(i32 %a, i32 %b, i32 %c)
  ret i32 %result
}

; CHECK: error: {{.*}} llvm.amdgcn.sad.hi.u8 requires target feature 'sad-insts'
define i32 @test_sad_hi_u8(i32 %a, i32 %b, i32 %c) {
  %result = call i32 @llvm.amdgcn.sad.hi.u8(i32 %a, i32 %b, i32 %c)
  ret i32 %result
}

; CHECK: error: {{.*}} llvm.amdgcn.sad.u16 requires target feature 'sad-insts'
define i32 @test_sad_u16(i32 %a, i32 %b, i32 %c) {
  %result = call i32 @llvm.amdgcn.sad.u16(i32 %a, i32 %b, i32 %c)
  ret i32 %result
}

; CHECK: error: {{.*}} llvm.amdgcn.msad.u8 requires target feature 'msad-insts'
define i32 @test_msad_u8(i32 %a, i32 %b, i32 %c) {
  %result = call i32 @llvm.amdgcn.msad.u8(i32 %a, i32 %b, i32 %c)
  ret i32 %result
}

; CHECK: error: {{.*}} llvm.amdgcn.qsad.pk.u16.u8 requires target feature 'qsad-insts'
define i64 @test_qsad_pk_u16_u8(i64 %a, i32 %b, i64 %c) {
  %result = call i64 @llvm.amdgcn.qsad.pk.u16.u8(i64 %a, i32 %b, i64 %c)
  ret i64 %result
}

; CHECK: error: {{.*}} llvm.amdgcn.mqsad.pk.u16.u8 requires target feature 'mqsad-pk-insts'
define i64 @test_mqsad_pk_u16_u8(i64 %a, i32 %b, i64 %c) {
  %result = call i64 @llvm.amdgcn.mqsad.pk.u16.u8(i64 %a, i32 %b, i64 %c)
  ret i64 %result
}

; CHECK: error: {{.*}} llvm.amdgcn.mqsad.u32.u8 requires target feature 'mqsad-insts'
define <4 x i32> @test_mqsad_u32_u8(i64 %a, i32 %b, <4 x i32> %c) {
  %result = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %a, i32 %b, <4 x i32> %c)
  ret <4 x i32> %result
}
