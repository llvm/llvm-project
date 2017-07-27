; RUN: opt -S -mtriple=amdgcn-- -amdgpu-codegenprepare < %s | FileCheck -check-prefix=OPT %s

declare i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0

; OPT-LABEL: @constant_load_i1
; OPT: load i1
; OPT-NEXT: store i1
define amdgpu_kernel void @constant_load_i1(i1 addrspace(1)* %out, i1 addrspace(2)* %in) #0 {
  %val = load i1, i1 addrspace(2)* %in
  store i1 %val, i1 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @constant_load_i1_align2
; OPT: load i1
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_i1_align2(i1 addrspace(1)* %out, i1 addrspace(2)* %in) #0 {
  %val = load i1, i1 addrspace(2)* %in, align 2
  store i1 %val, i1 addrspace(1)* %out, align 2
  ret void
}

; OPT-LABEL: @constant_load_i1_align4
; OPT: bitcast
; OPT-NEXT: load i32
; OPT-NEXT: trunc
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_i1_align4(i1 addrspace(1)* %out, i1 addrspace(2)* %in) #0 {
  %val = load i1, i1 addrspace(2)* %in, align 4
  store i1 %val, i1 addrspace(1)* %out, align 4
  ret void
}

; OPT-LABEL: @constant_load_i8
; OPT: load i8
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_i8(i8 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
  %val = load i8, i8 addrspace(2)* %in
  store i8 %val, i8 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @constant_load_i8_align2
; OPT: load i8
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_i8_align2(i8 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
  %val = load i8, i8 addrspace(2)* %in, align 2
  store i8 %val, i8 addrspace(1)* %out, align 2
  ret void
}

; OPT-LABEL: @constant_load_i8align4
; OPT: bitcast
; OPT-NEXT: load i32
; OPT-NEXT: trunc
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_i8align4(i8 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
  %val = load i8, i8 addrspace(2)* %in, align 4
  store i8 %val, i8 addrspace(1)* %out, align 4
  ret void
}


; OPT-LABEL: @constant_load_v2i8
; OPT: load <2 x i8>
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_v2i8(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
  %ld = load <2 x i8>, <2 x i8> addrspace(2)* %in
  store <2 x i8> %ld, <2 x i8> addrspace(1)* %out
  ret void
}

; OPT-LABEL: @constant_load_v2i8_align4
; OPT: bitcast
; OPT-NEXT: load i32
; OPT-NEXT: trunc
; OPT-NEXT: bitcast
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_v2i8_align4(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
  %ld = load <2 x i8>, <2 x i8> addrspace(2)* %in, align 4
  store <2 x i8> %ld, <2 x i8> addrspace(1)* %out, align 4
  ret void
}

; OPT-LABEL: @constant_load_v3i8
; OPT: bitcast <3 x i8>
; OPT-NEXT: load i32, i32 addrspace(2)
; OPT-NEXT: trunc i32
; OPT-NEXT: bitcast i24
; OPT-NEXT: store <3 x i8>
define amdgpu_kernel void @constant_load_v3i8(<3 x i8> addrspace(1)* %out, <3 x i8> addrspace(2)* %in) #0 {
  %ld = load <3 x i8>, <3 x i8> addrspace(2)* %in
  store <3 x i8> %ld, <3 x i8> addrspace(1)* %out
  ret void
}

; OPT-LABEL: @constant_load_v3i8_align4
; OPT: bitcast <3 x i8>
; OPT-NEXT: load i32, i32 addrspace(2)
; OPT-NEXT: trunc i32
; OPT-NEXT: bitcast i24
; OPT-NEXT: store <3 x i8>
define amdgpu_kernel void @constant_load_v3i8_align4(<3 x i8> addrspace(1)* %out, <3 x i8> addrspace(2)* %in) #0 {
  %ld = load <3 x i8>, <3 x i8> addrspace(2)* %in, align 4
  store <3 x i8> %ld, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; OPT-LABEL: @constant_load_i16
; OPT: load i16
; OPT: sext
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_i16(i32 addrspace(1)* %out, i16 addrspace(2)* %in) #0 {
  %ld = load i16, i16 addrspace(2)* %in
  %ext = sext i16 %ld to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @constant_load_i16_align4
; OPT: bitcast
; OPT-NEXT: load i32
; OPT-NEXT: trunc
; OPT-NEXT: sext
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_i16_align4(i32 addrspace(1)* %out, i16 addrspace(2)* %in) #0 {
  %ld = load i16, i16 addrspace(2)* %in, align 4
  %ext = sext i16 %ld to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; OPT-LABEL: @constant_load_f16
; OPT: load half
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_f16(half addrspace(1)* %out, half addrspace(2)* %in) #0 {
  %ld = load half, half addrspace(2)* %in
  store half %ld, half addrspace(1)* %out
  ret void
}

; OPT-LABEL: @constant_load_v2f16
; OPT: load <2 x half>
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(2)* %in) #0 {
  %ld = load <2 x half>, <2 x half> addrspace(2)* %in
  store <2 x half> %ld, <2 x half> addrspace(1)* %out
  ret void
}

; OPT-LABEL: @load_volatile
; OPT: load volatile i16
; OPT-NEXT: store
define amdgpu_kernel void @load_volatile(i16 addrspace(1)* %out, i16 addrspace(2)* %in) {
  %a = load volatile i16, i16 addrspace(2)* %in
  store i16 %a, i16 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @constant_load_v2i8_volatile
; OPT: load volatile <2 x i8>
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_v2i8_volatile(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
  %ld = load volatile <2 x i8>, <2 x i8> addrspace(2)* %in
  store <2 x i8> %ld, <2 x i8> addrspace(1)* %out
  ret void
}

; OPT-LABEL: @constant_load_v2i8_addrspace1
; OPT: load <2 x i8>
; OPT-NEXT: store
define amdgpu_kernel void @constant_load_v2i8_addrspace1(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(1)* %in) #0 {
  %ld = load <2 x i8>, <2 x i8> addrspace(1)* %in
  store <2 x i8> %ld, <2 x i8> addrspace(1)* %out
  ret void
}

; OPT-LABEL: @use_dispatch_ptr
; OPT: bitcast
; OPT-NEXT: load i32
; OPT-NEXT: trunc
; OPT-NEXT: zext
; OPT-NEXT: store
define amdgpu_kernel void @use_dispatch_ptr(i32 addrspace(1)* %ptr) #1 {
  %dispatch.ptr = call i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  %val = load i8, i8 addrspace(2)* %dispatch.ptr, align 4
  %ld = zext i8 %val to i32
  store i32 %ld, i32 addrspace(1)* %ptr
  ret void
}

attributes #0 = { nounwind }
