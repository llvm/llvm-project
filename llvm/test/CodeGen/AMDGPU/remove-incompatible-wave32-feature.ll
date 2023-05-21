; RUN: llc -march=amdgcn -mcpu=gfx906 -mattr=+wavefrontsize64 -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX906 %s
; RUN: FileCheck --check-prefix=WARN-GFX906 %s < %t
; RUN: llc -march=amdgcn -mcpu=gfx906 -mattr=+wavefrontsize64 -verify-machineinstrs < %s

; RUN: llc -march=amdgcn -mcpu=gfx90a -mattr=+wavefrontsize64 -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX90A %s
; RUN: FileCheck --check-prefix=WARN-GFX90A %s < %t
; RUN: llc -march=amdgcn -mcpu=gfx90a -mattr=+wavefrontsize64 -verify-machineinstrs < %s

; RUN: llc -march=amdgcn -mcpu=gfx1011 -mattr=+wavefrontsize64 -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1011 -mattr=+wavefrontsize64 -verify-machineinstrs < %s

; RUN: llc -march=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64 -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX11 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64 -verify-machineinstrs < %s

; WARN-GFX906: removing function 'needs_wavefrontsize32': +wavefrontsize32 is not supported on the current target
; WARN-GFX906-NOT: not supported

; WARN-GFX90A: removing function 'needs_wavefrontsize32': +wavefrontsize32 is not supported on the current target
; WARN-GFX90A-NOT: not supported

define void @needs_wavefrontsize32(ptr %out) #0 {
; GFX906-NOT:   @needs_wavefrontsize32
; GFX90A-NOT:   @needs_wavefrontsize32
; GFX10:        define void @needs_wavefrontsize32(
; GFX11:        define void @needs_wavefrontsize32(
  %1 = tail call i32 @llvm.read_register.i32(metadata !0)
  %2 = tail call i32 @llvm.ctpop.i32(i32 %1)
  store i32 %2, ptr %out, align 4
  ret void
}

define void @caller(ptr %out) {
  ; GFX906: call void null(
  ; GFX90A: call void null(
  ; GFX10: call void @needs_wavefrontsize32(
  ; GFX11: call void @needs_wavefrontsize32(
  call void @needs_wavefrontsize32(ptr %out)
  ret void
}

declare i32 @llvm.read_register.i32(metadata)
declare i32 @llvm.ctpop.i32(i32)

!0 = !{!"exec_lo"}

attributes #0 = { "target-features"="+wavefrontsize32" }
