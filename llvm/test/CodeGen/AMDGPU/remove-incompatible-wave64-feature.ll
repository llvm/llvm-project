; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -mattr=+wavefrontsize32 -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX1250 %s
; RUN: FileCheck --check-prefix=WARN-GFX1250 %s < %t
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -mattr=+wavefrontsize32 < %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -mattr=+wavefrontsize32,-wavefrontsize64 -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX1200 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -mattr=+wavefrontsize32,-wavefrontsize64 < %s

; WARN-GFX1250: removing function 'needs_wavefrontsize64': +wavefrontsize64 is not supported on the current target
; WARN-GFX1250-NOT: not supported

define void @needs_wavefrontsize64(ptr %out) #0 {
; GFX1250-NOT:  @needs_wavefrontsize64
; GFX1200:      define void @needs_wavefrontsize64(
  %1 = tail call i64 @llvm.read_register.i64(metadata !0)
  %2 = tail call i64 @llvm.ctpop.i64(i64 %1)
  store i64 %2, ptr %out, align 4
  ret void
}

define void @caller(ptr %out) {
  ; GFX1250: call void null(
  ; GFX1200: call void @needs_wavefrontsize64(
  call void @needs_wavefrontsize64(ptr %out)
  ret void
}

declare i64 @llvm.read_register.i64(metadata)
declare i64 @llvm.ctpop.i64(i64)

!0 = !{!"exec"}

attributes #0 = { "target-features"="+wavefrontsize64" }
