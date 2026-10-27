; RUN: not llc -mtriple=amdgpu9.0a -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgpu11.00 -mattr=-real-true16 -filetype=null %s 2>&1 | FileCheck %s

; 16-bit physical registers require true16.

; CHECK: error: could not allocate input reg for constraint '{v1.l}'
define void @use_vgpr_lo16(i16 %x) {
  call void asm sideeffect "; use $0", "{v1.l}"(i16 %x)
  ret void
}

; CHECK: error: could not allocate input reg for constraint '{v1.h}'
define void @use_vgpr_hi16(i16 %x) {
  call void asm sideeffect "; use $0", "{v1.h}"(i16 %x)
  ret void
}

; CHECK: error: could not allocate input reg for constraint '{s1.l}'
define void @use_sgpr_lo16(i16 inreg %x) {
  call void asm sideeffect "; use $0", "{s1.l}"(i16 %x)
  ret void
}

; CHECK: error: could not allocate input reg for constraint '{s1.h}'
define void @use_sgpr_hi16(i16 inreg %x) {
  call void asm sideeffect "; use $0", "{s1.h}"(i16 %x)
  ret void
}

; CHECK: error: could not allocate input reg for constraint '{a1.l}'
define void @use_agpr_lo16(i16 %x) {
  call void asm sideeffect "; use $0", "{a1.l}"(i16 %x)
  ret void
}

; CHECK: error: could not allocate input reg for constraint '{a1.h}'
define void @use_agpr_hi16(i16 %x) {
  call void asm sideeffect "; use $0", "{a1.h}"(i16 %x)
  ret void
}

; CHECK: error: could not allocate output register for constraint '{v1.l}'
define i16 @def_vgpr_lo16() {
  %v = call i16 asm sideeffect "; def $0", "={v1.l}"()
  ret i16 %v
}

; CHECK: error: could not allocate output register for constraint '{v1.h}'
define half @def_vgpr_hi16() {
  %v = call half asm sideeffect "; def $0", "={v1.h}"()
  ret half %v
}

; CHECK: error: could not allocate output register for constraint '{v1.h}'
define i16 @tied_vgpr_hi16(i16 %x) {
  %v = call i16 asm sideeffect "; def $0", "={v1.h},0"(i16 %x)
  ret i16 %v
}
