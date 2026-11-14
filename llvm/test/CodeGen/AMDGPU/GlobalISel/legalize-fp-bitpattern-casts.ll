; RUN: llc -global-isel -mtriple=amdgpu7.00 -stop-after=legalizer -verify-machineinstrs -o - %s | FileCheck %s

; Widening a load or store preserves the memory value's bit pattern. In
; particular, an atomic load must keep its original memory width and ordering.
define half @atomic_load_half(ptr addrspace(1) %p) {
; CHECK-LABEL: name: atomic_load_half
; CHECK: [[LOAD:%[0-9]+]]:_(i32) = G_LOAD {{.*}} :: (load monotonic (f16) from %ir.p, addrspace 1)
; CHECK-NEXT: [[TRUNC:%[0-9]+]]:_(i16) = G_TRUNC [[LOAD]](i32)
; CHECK-NEXT: {{%[0-9]+}}:_(f16) = G_BITCAST [[TRUNC]](i16)
  %x = load atomic half, ptr addrspace(1) %p monotonic, align 2
  ret half %x
}

define bfloat @atomic_load_bfloat(ptr addrspace(1) %p) {
; CHECK-LABEL: name: atomic_load_bfloat
; CHECK: [[LOAD:%[0-9]+]]:_(i32) = G_LOAD {{.*}} :: (load monotonic (bf16) from %ir.p, addrspace 1)
; CHECK-NEXT: [[TRUNC:%[0-9]+]]:_(i16) = G_TRUNC [[LOAD]](i32)
; CHECK-NEXT: {{%[0-9]+}}:_(bf16) = G_BITCAST [[TRUNC]](i16)
  %x = load atomic bfloat, ptr addrspace(1) %p monotonic, align 2
  ret bfloat %x
}

define void @store_half(ptr addrspace(1) %p, half %x) {
; CHECK-LABEL: name: store_half
; CHECK: [[BITS:%[0-9]+]]:_(i16) = G_BITCAST {{%[0-9]+}}(f16)
; CHECK-NEXT: [[EXT:%[0-9]+]]:_(i32) = G_ANYEXT [[BITS]](i16)
; CHECK-NEXT: G_STORE [[EXT]](i32), {{.*}} :: (store (f16) into %ir.p, addrspace 1)
  store half %x, ptr addrspace(1) %p, align 2
  ret void
}

define void @store_bfloat(ptr addrspace(1) %p, bfloat %x) {
; CHECK-LABEL: name: store_bfloat
; CHECK: [[BITS:%[0-9]+]]:_(i16) = G_BITCAST {{%[0-9]+}}(bf16)
; CHECK-NEXT: [[EXT:%[0-9]+]]:_(i32) = G_ANYEXT [[BITS]](i16)
; CHECK-NEXT: G_STORE [[EXT]](i32), {{.*}} :: (store (bf16) into %ir.p, addrspace 1)
  store bfloat %x, ptr addrspace(1) %p, align 2
  ret void
}

; Odd-length vectors cannot use the packed i32 extraction path. Widening the
; elements to f32 must preserve the low 16 bits, without an FP conversion.
define half @extract_v3half(<3 x half> %v, i32 %idx) {
; CHECK-LABEL: name: extract_v3half
; CHECK: [[VEC:%[0-9]+]]:_(<3 x i32>) = G_BUILD_VECTOR
; CHECK-NEXT: [[WIDE:%[0-9]+]]:_(<3 x f32>) = G_BITCAST [[VEC]](<3 x i32>)
; CHECK-NEXT: [[ELT:%[0-9]+]]:_(f32) = G_EXTRACT_VECTOR_ELT [[WIDE]](<3 x f32>), {{%[0-9]+}}(i32)
; CHECK-NEXT: [[BITS:%[0-9]+]]:_(i32) = G_BITCAST [[ELT]](f32)
; CHECK-NEXT: [[TRUNC:%[0-9]+]]:_(i16) = G_TRUNC [[BITS]](i32)
; CHECK-NEXT: {{%[0-9]+}}:_(f16) = G_BITCAST [[TRUNC]](i16)
  %x = extractelement <3 x half> %v, i32 %idx
  ret half %x
}

define <3 x half> @insert_v3half(<3 x half> %v, half %elt, i32 %idx) {
; CHECK-LABEL: name: insert_v3half
; CHECK: [[VEC:%[0-9]+]]:_(<3 x i32>) = G_BUILD_VECTOR
; CHECK-NEXT: [[WIDE:%[0-9]+]]:_(<3 x f32>) = G_BITCAST [[VEC]](<3 x i32>)
; CHECK-NEXT: [[ELT_BITS:%[0-9]+]]:_(i16) = G_BITCAST {{%[0-9]+}}(f16)
; CHECK-NEXT: [[EXT:%[0-9]+]]:_(i32) = G_ANYEXT [[ELT_BITS]](i16)
; CHECK-NEXT: [[ELT:%[0-9]+]]:_(f32) = G_BITCAST [[EXT]](i32)
; CHECK-NEXT: [[INSERT:%[0-9]+]]:_(<3 x f32>) = G_INSERT_VECTOR_ELT [[WIDE]], [[ELT]](f32), {{%[0-9]+}}(i32)
; CHECK-NEXT: [[BITS:%[0-9]+]]:_(<3 x i32>) = G_BITCAST [[INSERT]](<3 x f32>)
; CHECK-NEXT: [[LO:%[0-9]+]]:_(i32), [[MID:%[0-9]+]]:_(i32), [[HI:%[0-9]+]]:_(i32) = G_UNMERGE_VALUES [[BITS]](<3 x i32>)
; CHECK-NEXT: [[MASK:%[0-9]+]]:_(i32) = G_CONSTANT i32 65535
; CHECK-NEXT: [[LO16:%[0-9]+]]:_(i32) = G_AND [[LO]], [[MASK]]
; CHECK-NEXT: [[MID16:%[0-9]+]]:_(i32) = G_AND [[MID]], [[MASK]]
; CHECK-NEXT: [[SHIFT:%[0-9]+]]:_(i32) = G_SHL [[MID16]], {{%[0-9]+}}(i32)
; CHECK-NEXT: [[PACKED:%[0-9]+]]:_(i32) = G_OR [[LO16]], [[SHIFT]]
; CHECK-NEXT: [[HI16:%[0-9]+]]:_(i32) = G_AND [[HI]], [[MASK]]
  %x = insertelement <3 x half> %v, half %elt, i32 %idx
  ret <3 x half> %x
}

; An even-length vector uses a packed integer extraction instead.
define float @extract_v2half(<2 x half> %v, i32 %idx) {
; CHECK-LABEL: name: extract_v2half
; CHECK: [[SHIFT:%[0-9]+]]:_(i32) = G_LSHR
; CHECK-NEXT: [[BITS:%[0-9]+]]:_(i16) = G_TRUNC [[SHIFT]](i32)
; CHECK-NEXT: [[ELT:%[0-9]+]]:_(f16) = G_BITCAST [[BITS]](i16)
; CHECK-NEXT: {{%[0-9]+}}:_(f32) = G_FPEXT [[ELT]](f16)
  %r = extractelement <2 x half> %v, i32 %idx
  %f = fpext half %r to float
  ret float %f
}

; Splitting a packed floating vector must truncate both components as integers.
define <2 x float> @fpext_v2bfloat(<2 x bfloat> %v) {
; CHECK-LABEL: name: fpext_v2bfloat
; CHECK: [[PACKED:%[0-9]+]]:_(i32) = G_BITCAST {{%[0-9]+}}(<2 x bf16>)
; CHECK-NEXT: [[LO:%[0-9]+]]:_(i16) = G_TRUNC [[PACKED]](i32)
; CHECK-NEXT: {{%[0-9]+}}:_(bf16) = G_BITCAST [[LO]](i16)
; CHECK-NEXT: [[AMT:%[0-9]+]]:_(i32) = G_CONSTANT i32 16
; CHECK-NEXT: [[SHIFT:%[0-9]+]]:_(i32) = G_LSHR [[PACKED]], [[AMT]](i32)
; CHECK-NEXT: [[HI:%[0-9]+]]:_(i16) = G_TRUNC [[SHIFT]](i32)
; CHECK-NEXT: {{%[0-9]+}}:_(bf16) = G_BITCAST [[HI]](i16)
  %r = fpext <2 x bfloat> %v to <2 x float>
  ret <2 x float> %r
}

; Software f64-to-f16 conversion computes the final bit pattern in i32.
define half @fptrunc_f64_to_f16(double %x) {
; CHECK-LABEL: name: fptrunc_f64_to_f16
; CHECK: [[BITS:%[0-9]+]]:_(i16) = G_TRUNC {{%[0-9]+}}(i32)
; CHECK-NEXT: {{%[0-9]+}}:_(f16) = G_BITCAST [[BITS]](i16)
  %r = fptrunc double %x to half
  ret half %r
}
