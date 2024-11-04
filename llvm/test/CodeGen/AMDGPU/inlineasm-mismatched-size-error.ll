; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s

; Diagnose register constraints that are not wide enough.

; ERR: error: couldn't allocate output register for constraint '{v[8:15]}'
define <9 x i32> @inline_asm_9xi32_in_8v_def() {
  %asm = call <9 x i32> asm sideeffect "; def $0", "={v[8:15]}"()
  ret <9 x i32> %asm
}

; ERR: error: couldn't allocate input reg for constraint '{v[8:15]}'
define void @inline_asm_9xi32_in_8v_use(<9 x i32> %val) {
  call void asm sideeffect "; use $0", "{v[8:15]}"(<9 x i32> %val)
  ret void
}

; ERR: error: couldn't allocate output register for constraint '{s[8:15]}'
define <9 x i32> @inline_asm_9xi32_in_8s_def() {
  %asm = call <9 x i32> asm sideeffect "; def $0", "={s[8:15]}"()
  ret <9 x i32> %asm
}


; Diagnose register constraints that are too wide.

; ERR: error: couldn't allocate output register for constraint '{v[8:16]}'
define <8 x i32> @inline_asm_8xi32_in_9v_def() {
  %asm = call <8 x i32> asm sideeffect "; def $0", "={v[8:16]}"()
  ret <8 x i32> %asm
}

; ERR: error: couldn't allocate input reg for constraint '{v[8:16]}'
define void @inline_asm_8xi32_in_9v_use(<8 x i32> %val) {
  call void asm sideeffect "; use $0", "{v[8:16]}"(<8 x i32> %val)
  ret void
}

; ERR: error: couldn't allocate output register for constraint '{s[8:16]}'
define <8 x i32> @inline_asm_8xi32_in_9s_def() {
  %asm = call <8 x i32> asm sideeffect "; def $0", "={s[8:16]}"()
  ret <8 x i32> %asm
}


; Diagnose mismatched scalars with register ranges

; ERR: error: couldn't allocate output register for constraint '{s[4:5]}'
define void @inline_asm_scalar_read_too_wide() {
  %asm = call i32 asm sideeffect "; def $0 ", "={s[4:5]}"()
  ret void
}

; ERR: error: couldn't allocate output register for constraint '{s[4:4]}'
define void @inline_asm_scalar_read_too_narrow() {
  %asm = call i64 asm sideeffect "; def $0 ", "={s[4:4]}"()
  ret void
}

; Single registers for vector types that are too wide or too narrow should be
; diagnosed.

; ERR: error: couldn't allocate input reg for constraint '{v8}'
define void @inline_asm_4xi32_in_v_use(<4 x i32> %val) {
  call void asm sideeffect "; use $0", "{v8}"(<4 x i32> %val)
  ret void
}

; ERR: error: couldn't allocate output register for constraint '{v8}'
define <4 x i32> @inline_asm_4xi32_in_v_def() {
  %asm = call <4 x i32> asm sideeffect "; def $0", "={v8}"()
  ret <4 x i32> %asm
}

; ERR: error: couldn't allocate output register for constraint '{s8}'
define <4 x i32> @inline_asm_4xi32_in_s_def() {
  %asm = call <4 x i32> asm sideeffect "; def $0", "={s8}"()
  ret <4 x i32> %asm
}

; ERR: error: couldn't allocate input reg for constraint '{v8}'
; ERR: error: couldn't allocate input reg for constraint 'v'
define void @inline_asm_2xi8_in_v_use(<2 x i8> %val) {
  call void asm sideeffect "; use $0", "{v8}"(<2 x i8> %val)
  call void asm sideeffect "; use $0", "v"(<2 x i8> %val)
  ret void
}

; ERR: error: couldn't allocate output register for constraint '{v8}'
; ERR: error: couldn't allocate output register for constraint 'v'
define <2 x i8> @inline_asm_2xi8_in_v_def() {
  %phys = call <2 x i8> asm sideeffect "; def $0", "={v8}"()
  %virt = call <2 x i8> asm sideeffect "; def $0", "=v"()
  %r = and <2 x i8> %phys, %virt
  ret <2 x i8> %r
}

; ERR: error: couldn't allocate output register for constraint '{s8}'
; ERR: error: couldn't allocate output register for constraint 's'
define <2 x i8> @inline_asm_2xi8_in_s_def() {
  %phys = call <2 x i8> asm sideeffect "; def $0", "={s8}"()
  %virt = call <2 x i8> asm sideeffect "; def $0", "=s"()
  %r = and <2 x i8> %phys, %virt
  ret <2 x i8> %r
}
