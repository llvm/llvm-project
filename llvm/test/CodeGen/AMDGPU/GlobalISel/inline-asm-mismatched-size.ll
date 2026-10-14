; RUN: not llc -global-isel -mtriple=amdgpu8.03 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s

; This asm is broken because it's using a 5 element wide physical
; register for a 4 element wide value. Make sure we don't crash, and
; take the IR type as truth.

; ERR: error: could not allocate output register for constraint '{s[8:12]}'
define amdgpu_kernel void @return_type_is_too_big_vector() {
  %sgpr = call <4 x i32> asm sideeffect "; def $0", "={s[8:12]}" ()
  call void asm sideeffect "; use $0", "s"(<4 x i32> %sgpr) #0
  ret void
}

; This is broken because it requests 3 32-bit sgprs to handle a 4xi32 result.

; ERR: error: could not allocate output register for constraint '{s[8:10]}'
define amdgpu_kernel void @return_type_is_too_small_vector() {
  %sgpr = call <4 x i32> asm sideeffect "; def $0", "={s[8:10]}" ()
  call void asm sideeffect "; use $0", "s"(<4 x i32> %sgpr) #0
  ret void
}

; ERR: error: could not allocate output register for constraint '{v8}'
define i64 @return_type_is_too_big_scalar() {
  %reg = call i64 asm sideeffect "; def $0", "={v8}" ()
  ret i64 %reg
}

; ERR: error: could not allocate output register for constraint '{v[8:9]}'
define i32 @return_type_is_too_small_scalar() {
  %reg = call i32 asm sideeffect "; def $0", "={v[8:9]}" ()
  ret i32 %reg
}

; ERR: error: could not allocate output register for constraint '{v8}'
define ptr addrspace(1) @return_type_is_too_big_pointer() {
  %reg = call ptr addrspace(1) asm sideeffect "; def $0", "={v8}" ()
  ret ptr addrspace(1) %reg
}

; ERR: error: could not allocate output register for constraint '{v[8:9]}'
define ptr addrspace(3) @return_type_is_too_small_pointer() {
  %reg = call ptr addrspace(3) asm sideeffect "; def $0", "={v[8:9]}" ()
  ret ptr addrspace(3) %reg
}

define void @use_vector_too_small(<8 x i32> %arg) {
  call void asm sideeffect "; use $0", "{v[0:7]}"(<8 x i32> %arg)
  ret void
}

; ERR: error: could not allocate input reg for constraint '{v[0:9]}'
define void @use_vector_too_big(<8 x i32> %arg) {
  call void asm sideeffect "; use $0", "{v[0:9]}"(<8 x i32> %arg)
  ret void
}

; ERR: error: could not allocate input reg for constraint '{v0}'
define void @use_scalar_too_small(i64 %arg) {
  call void asm sideeffect "; use $0", "{v0}"(i64 %arg)
  ret void
}

; ERR: error: could not allocate input reg for constraint '{v[0:1]}'
define void @use_scalar_too_big(i32 %arg) {
  call void asm sideeffect "; use $0", "{v[0:1]}"(i32 %arg)
  ret void
}

; ERR: error: could not allocate input reg for constraint '{v0}'
define void @use_pointer_too_small(ptr addrspace(1) %arg) {
  call void asm sideeffect "; use $0", "{v0}"(ptr addrspace(1) %arg)
  ret void
}

; ERR: error: could not allocate input reg for constraint '{v[0:1]}'
define void @use_pointer_too_big(ptr addrspace(3) %arg) {
  call void asm sideeffect "; use $0", "{v[0:1]}"(ptr addrspace(3) %arg)
  ret void
}
