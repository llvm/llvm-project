; RUN: llc -march=r600 -mtriple=r600-- -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=EG %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=CM %s

; Loosely based on test/CodeGen/{X86,AArch64}/extract-lowbits.ll,
; but with all 64-bit tests, and tests with loads dropped.

; Patterns:
;   a) x &  (1 << nbits) - 1
;   b) x & ~(-1 << nbits)
;   c) x &  (-1 >> (32 - y))
;   d) x << (32 - y) >> (32 - y)
; are equivalent.

; ---------------------------------------------------------------------------- ;
; Pattern a. 32-bit
; ---------------------------------------------------------------------------- ;

; R600-LABEL: bzhi32_a0:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       BFE_UINT {{\*?}} [[RET]], KC0[2].Y, 0.0, KC0[2].Z
define amdgpu_kernel void @bzhi32_a0(i32 %val, i32 %numlowbits, i32 addrspace(1)* %out) {
  %onebit = shl i32 1, %numlowbits
  %mask = add nsw i32 %onebit, -1
  %masked = and i32 %mask, %val
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; R600-LABEL: bzhi32_a1_indexzext:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       BFE_UINT {{\*?}} [[RET]], KC0[2].Y, 0.0, KC0[2].Z
define amdgpu_kernel void @bzhi32_a1_indexzext(i32 %val, i8 zeroext %numlowbits, i32 addrspace(1)* %out) {
  %conv = zext i8 %numlowbits to i32
  %onebit = shl i32 1, %conv
  %mask = add nsw i32 %onebit, -1
  %masked = and i32 %mask, %val
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; R600-LABEL: bzhi32_a4_commutative:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       BFE_UINT {{\*?}} [[RET]], KC0[2].Y, 0.0, KC0[2].Z
define amdgpu_kernel void @bzhi32_a4_commutative(i32 %val, i32 %numlowbits, i32 addrspace(1)* %out) {
  %onebit = shl i32 1, %numlowbits
  %mask = add nsw i32 %onebit, -1
  %masked = and i32 %val, %mask ; swapped order
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; ---------------------------------------------------------------------------- ;
; Pattern b. 32-bit
; ---------------------------------------------------------------------------- ;

; R600-LABEL: bzhi32_b0:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       BFE_UINT {{\*?}} [[RET]], KC0[2].Y, 0.0, KC0[2].Z
define amdgpu_kernel void @bzhi32_b0(i32 %val, i32 %numlowbits, i32 addrspace(1)* %out) {
  %notmask = shl i32 -1, %numlowbits
  %mask = xor i32 %notmask, -1
  %masked = and i32 %mask, %val
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; R600-LABEL: bzhi32_b1_indexzext:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       BFE_UINT {{\*?}} [[RET]], KC0[2].Y, 0.0, KC0[2].Z
define amdgpu_kernel void @bzhi32_b1_indexzext(i32 %val, i8 zeroext %numlowbits, i32 addrspace(1)* %out) {
  %conv = zext i8 %numlowbits to i32
  %notmask = shl i32 -1, %conv
  %mask = xor i32 %notmask, -1
  %masked = and i32 %mask, %val
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; R600-LABEL: bzhi32_b4_commutative:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       BFE_UINT {{\*?}} [[RET]], KC0[2].Y, 0.0, KC0[2].Z
define amdgpu_kernel void @bzhi32_b4_commutative(i32 %val, i32 %numlowbits, i32 addrspace(1)* %out) {
  %notmask = shl i32 -1, %numlowbits
  %mask = xor i32 %notmask, -1
  %masked = and i32 %val, %mask ; swapped order
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; ---------------------------------------------------------------------------- ;
; Pattern c. 32-bit
; ---------------------------------------------------------------------------- ;

; R600-LABEL: bzhi32_c0:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       BFE_UINT {{\*?}} [[RET]], KC0[2].Y, 0.0, KC0[2].Z
define amdgpu_kernel void @bzhi32_c0(i32 %val, i32 %numlowbits, i32 addrspace(1)* %out) {
  %numhighbits = sub i32 32, %numlowbits
  %mask = lshr i32 -1, %numhighbits
  %masked = and i32 %mask, %val
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; R600-LABEL: bzhi32_c1_indexzext:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       SUB_INT {{\*?}} [[SUBR:T[0-9]+]].[[SUBC:[XYZW]]], literal.x, KC0[2].Z
; R600-NEXT:  32
; R600-NEXT:  AND_INT {{\*?}} {{T[0-9]+}}.[[AND1C:[XYZW]]], {{T[0-9]+|PV}}.[[SUBC]], literal.x
; R600-NEXT:  255
; R600:       LSHR {{\*?}} {{T[0-9]}}.[[LSHRC:[XYZW]]], literal.x, {{T[0-9]+|PV}}.[[AND1C]]
; R600-NEXT:  -1
; R600-NEXT:  AND_INT {{[* ]*}}[[RET]], {{T[0-9]+|PV}}.[[LSHRC]], KC0[2].Y
define amdgpu_kernel void @bzhi32_c1_indexzext(i32 %val, i8 %numlowbits, i32 addrspace(1)* %out) {
  %numhighbits = sub i8 32, %numlowbits
  %sh_prom = zext i8 %numhighbits to i32
  %mask = lshr i32 -1, %sh_prom
  %masked = and i32 %mask, %val
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; R600-LABEL: bzhi32_c4_commutative:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       BFE_UINT {{\*?}} [[RET]], KC0[2].Y, 0.0, KC0[2].Z
define amdgpu_kernel void @bzhi32_c4_commutative(i32 %val, i32 %numlowbits, i32 addrspace(1)* %out) {
  %numhighbits = sub i32 32, %numlowbits
  %mask = lshr i32 -1, %numhighbits
  %masked = and i32 %val, %mask ; swapped order
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; ---------------------------------------------------------------------------- ;
; Pattern d. 32-bit.
; ---------------------------------------------------------------------------- ;

; R600-LABEL: bzhi32_d0:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       BFE_UINT {{\*?}} [[RET]], KC0[2].Y, 0.0, KC0[2].Z
define amdgpu_kernel void @bzhi32_d0(i32 %val, i32 %numlowbits, i32 addrspace(1)* %out) {
  %numhighbits = sub i32 32, %numlowbits
  %highbitscleared = shl i32 %val, %numhighbits
  %masked = lshr i32 %highbitscleared, %numhighbits
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}

; R600-LABEL: bzhi32_d1_indexzext:
; EG:         MEM_RAT_CACHELESS STORE_RAW [[RET:T[0-1]+\.[XYZW]]]
; CM:         MEM_RAT_CACHELESS STORE_DWORD [[RET:T[0-1]+\.[XYZW]]]
; R600:       SUB_INT {{\*?}} [[SUBR:T[0-9]+]].[[SUBC:[XYZW]]], literal.x, KC0[2].Z
; R600-NEXT:  32
; R600-NEXT:  AND_INT {{\*?}} [[AND:T[0-9]+\.[XYZW]]], {{T[0-9]+|PV}}.[[SUBC]], literal.x
; R600-NEXT:  255
; R600:       LSHL {{\*?}} {{T[0-9]}}.[[LSHLC:[XYZW]]], KC0[2].Y, {{T[0-9]+|PV}}.[[AND1C]]
; R600:       LSHR {{[* ]*}}[[RET]], {{T[0-9]+|PV}}.[[LSHLC]], [[AND]]
define amdgpu_kernel void @bzhi32_d1_indexzext(i32 %val, i8 %numlowbits, i32 addrspace(1)* %out) {
  %numhighbits = sub i8 32, %numlowbits
  %sh_prom = zext i8 %numhighbits to i32
  %highbitscleared = shl i32 %val, %sh_prom
  %masked = lshr i32 %highbitscleared, %sh_prom
  store i32 %masked, i32 addrspace(1)* %out
  ret void
}
