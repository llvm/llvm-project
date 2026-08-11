; RUN: not llc -global-isel=0 -mtriple=amdgpu9.4 -mcpu=gfx942 -filetype=null < %s 2>&1 | FileCheck %s -check-prefix=GFX942
; RUN: not llc -global-isel=1 -global-isel-abort=0 -mtriple=amdgpu9.4 -mcpu=gfx942 -filetype=null < %s 2>&1 | FileCheck %s -check-prefix=GFX942
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.4 -mcpu=gfx950 -filetype=null < %s 2>&1 | FileCheck %s -check-prefix=GFX950
; RUN: not llc -global-isel=1 -global-isel-abort=0 -mtriple=amdgpu9.4 -mcpu=gfx950 -filetype=null < %s 2>&1 | FileCheck %s -check-prefix=GFX950

; The v_cube* ALU is graphics-pipe hardware, absent on the CDNA-3 compute-only
; parts gfx942 and gfx950.

; GFX942: error: {{.*}} llvm.amdgcn.cubeid requires target feature 'cube-insts'
; GFX950: error: {{.*}} llvm.amdgcn.cubeid requires target feature 'cube-insts'
define float @test_cubeid(float %a, float %b, float %c) {
  %result = call float @llvm.amdgcn.cubeid(float %a, float %b, float %c)
  ret float %result
}

; GFX942: error: {{.*}} llvm.amdgcn.cubema requires target feature 'cube-insts'
; GFX950: error: {{.*}} llvm.amdgcn.cubema requires target feature 'cube-insts'
define float @test_cubema(float %a, float %b, float %c) {
  %result = call float @llvm.amdgcn.cubema(float %a, float %b, float %c)
  ret float %result
}

; GFX942: error: {{.*}} llvm.amdgcn.cubesc requires target feature 'cube-insts'
; GFX950: error: {{.*}} llvm.amdgcn.cubesc requires target feature 'cube-insts'
define float @test_cubesc(float %a, float %b, float %c) {
  %result = call float @llvm.amdgcn.cubesc(float %a, float %b, float %c)
  ret float %result
}

; GFX942: error: {{.*}} llvm.amdgcn.cubetc requires target feature 'cube-insts'
; GFX950: error: {{.*}} llvm.amdgcn.cubetc requires target feature 'cube-insts'
define float @test_cubetc(float %a, float %b, float %c) {
  %result = call float @llvm.amdgcn.cubetc(float %a, float %b, float %c)
  ret float %result
}

declare float @llvm.amdgcn.cubeid(float, float, float)
declare float @llvm.amdgcn.cubema(float, float, float)
declare float @llvm.amdgcn.cubesc(float, float, float)
declare float @llvm.amdgcn.cubetc(float, float, float)
