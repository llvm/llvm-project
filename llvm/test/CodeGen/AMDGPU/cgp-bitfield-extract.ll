; RUN: opt -S -mtriple=amdgcn-- -codegenprepare < %s | FileCheck -check-prefix=OPT %s
; RUN: opt -S -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -codegenprepare < %s | FileCheck -check-prefix=OPT %s
; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; This particular case will actually be worse in terms of code size
; from sinking into both.

; OPT-LABEL: @sink_ubfe_i32(
; OPT: entry:
; OPT-NEXT: br i1

; OPT: bb0:
; OPT: %0 = lshr i32 %arg1, 8
; OPT-NEXT: %val0 = and i32 %0, 255
; OPT: br label

; OPT: bb1:
; OPT: %1 = lshr i32 %arg1, 8
; OPT-NEXT: %val1 = and i32 %1, 127
; OPT: br label

; OPT: ret:
; OPT: store
; OPT: ret


; GCN-LABEL: {{^}}sink_ubfe_i32:
; GCN-NOT: lshr
; GCN: s_cbranch_scc{{[0-1]}}

; GCN: s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x70008
; GCN: .LBB0_3:
; GCN: s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80008

; GCN: buffer_store_dword
; GCN: s_endpgm
define amdgpu_kernel void @sink_ubfe_i32(ptr addrspace(1) %out, i32 %arg1) #0 {
entry:
  %shr = lshr i32 %arg1, 8
  br i1 undef, label %bb0, label %bb1

bb0:
  %val0 = and i32 %shr, 255
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

bb1:
  %val1 = and i32 %shr, 127
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

ret:
  %phi = phi i32 [ %val0, %bb0 ], [ %val1, %bb1 ]
  store i32 %phi, ptr addrspace(1) %out
  ret void
}

; OPT-LABEL: @sink_sbfe_i32(
; OPT: entry:
; OPT-NEXT: br i1

; OPT: bb0:
; OPT: %0 = ashr i32 %arg1, 8
; OPT-NEXT: %val0 = and i32 %0, 255
; OPT: br label

; OPT: bb1:
; OPT: %1 = ashr i32 %arg1, 8
; OPT-NEXT: %val1 = and i32 %1, 127
; OPT: br label

; OPT: ret:
; OPT: store
; OPT: ret

; GCN-LABEL: {{^}}sink_sbfe_i32:
define amdgpu_kernel void @sink_sbfe_i32(ptr addrspace(1) %out, i32 %arg1) #0 {
entry:
  %shr = ashr i32 %arg1, 8
  br i1 undef, label %bb0, label %bb1

bb0:
  %val0 = and i32 %shr, 255
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

bb1:
  %val1 = and i32 %shr, 127
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

ret:
  %phi = phi i32 [ %val0, %bb0 ], [ %val1, %bb1 ]
  store i32 %phi, ptr addrspace(1) %out
  ret void
}


; OPT-LABEL: @sink_ubfe_i16(
; OPT: entry:
; OPT-NEXT: br i1

; OPT: bb0:
; OPT: %0 = lshr i16 %arg1, 4
; OPT-NEXT: %val0 = and i16 %0, 255
; OPT: br label

; OPT: bb1:
; OPT: %1 = lshr i16 %arg1, 4
; OPT-NEXT: %val1 = and i16 %1, 127
; OPT: br label

; OPT: ret:
; OPT: store
; OPT: ret

; For GFX8: since i16 is legal type, we cannot sink lshr into .LBBs.

; GCN-LABEL: {{^}}sink_ubfe_i16:
; GCN-NOT: lshr
; VI: s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x2c
; VI: s_bfe_u32 [[BFE:s[0-9]+]], [[ARG]], 0xc0004
; GCN: s_cbranch_scc{{[0-1]}}

; SI: s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x70004
; VI: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x7f

; GCN: .LBB2_3:
; SI: s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80004
; VI: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0xff

; GCN: buffer_store_short
; GCN: s_endpgm
define amdgpu_kernel void @sink_ubfe_i16(ptr addrspace(1) %out, i16 %arg1) #0 {
entry:
  %shr = lshr i16 %arg1, 4
  br i1 undef, label %bb0, label %bb1

bb0:
  %val0 = and i16 %shr, 255
  store volatile i16 0, ptr addrspace(1) undef
  br label %ret

bb1:
  %val1 = and i16 %shr, 127
  store volatile i16 0, ptr addrspace(1) undef
  br label %ret

ret:
  %phi = phi i16 [ %val0, %bb0 ], [ %val1, %bb1 ]
  store i16 %phi, ptr addrspace(1) %out
  ret void
}

; We don't really want to sink this one since it isn't reducible to a
; 32-bit BFE on one half of the integer.

; OPT-LABEL: @sink_ubfe_i64_span_midpoint(
; OPT: entry:
; OPT-NOT: lshr
; OPT: br i1

; OPT: bb0:
; OPT: %0 = lshr i64 %arg1, 30
; OPT-NEXT: %val0 = and i64 %0, 255

; OPT: bb1:
; OPT: %1 = lshr i64 %arg1, 30
; OPT-NEXT: %val1 = and i64 %1, 127

; OPT: ret:
; OPT: store
; OPT: ret

; GCN-LABEL: {{^}}sink_ubfe_i64_span_midpoint:

; GCN: s_cbranch_scc{{[0-1]}} .LBB3_2
; GCN: v_alignbit_b32 v[[LO:[0-9]+]], s{{[0-9]+}}, v{{[0-9]+}}, 30
; GCN: v_and_b32_e32 v{{[0-9]+}}, 0x7f, v[[LO]]

; GCN: .LBB3_3:
; GCN: v_and_b32_e32 v{{[0-9]+}}, 0xff, v[[LO]]

; GCN: buffer_store_dwordx2
define amdgpu_kernel void @sink_ubfe_i64_span_midpoint(ptr addrspace(1) %out, i64 %arg1) #0 {
entry:
  %shr = lshr i64 %arg1, 30
  br i1 undef, label %bb0, label %bb1

bb0:
  %val0 = and i64 %shr, 255
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

bb1:
  %val1 = and i64 %shr, 127
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

ret:
  %phi = phi i64 [ %val0, %bb0 ], [ %val1, %bb1 ]
  store i64 %phi, ptr addrspace(1) %out
  ret void
}

; OPT-LABEL: @sink_ubfe_i64_low32(
; OPT: entry:
; OPT-NOT: lshr
; OPT: br i1

; OPT: bb0:
; OPT: %0 = lshr i64 %arg1, 15
; OPT-NEXT: %val0 = and i64 %0, 255

; OPT: bb1:
; OPT: %1 = lshr i64 %arg1, 15
; OPT-NEXT: %val1 = and i64 %1, 127

; OPT: ret:
; OPT: store
; OPT: ret

; GCN-LABEL: {{^}}sink_ubfe_i64_low32:

; GCN: s_cbranch_scc{{[0-1]}} .LBB4_2

; GCN: s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x7000f

; GCN: .LBB4_3:
; GCN: s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x8000f

; GCN: buffer_store_dwordx2
define amdgpu_kernel void @sink_ubfe_i64_low32(ptr addrspace(1) %out, i64 %arg1) #0 {
entry:
  %shr = lshr i64 %arg1, 15
  br i1 undef, label %bb0, label %bb1

bb0:
  %val0 = and i64 %shr, 255
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

bb1:
  %val1 = and i64 %shr, 127
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

ret:
  %phi = phi i64 [ %val0, %bb0 ], [ %val1, %bb1 ]
  store i64 %phi, ptr addrspace(1) %out
  ret void
}

; OPT-LABEL: @sink_ubfe_i64_high32(
; OPT: entry:
; OPT-NOT: lshr
; OPT: br i1

; OPT: bb0:
; OPT: %0 = lshr i64 %arg1, 35
; OPT-NEXT: %val0 = and i64 %0, 255

; OPT: bb1:
; OPT: %1 = lshr i64 %arg1, 35
; OPT-NEXT: %val1 = and i64 %1, 127

; OPT: ret:
; OPT: store
; OPT: ret

; GCN-LABEL: {{^}}sink_ubfe_i64_high32:
; GCN: s_cbranch_scc{{[0-1]}} .LBB5_2
; GCN: s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x70003

; GCN: .LBB5_3:
; GCN: s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80003

; GCN: buffer_store_dwordx2
define amdgpu_kernel void @sink_ubfe_i64_high32(ptr addrspace(1) %out, i64 %arg1) #0 {
entry:
  %shr = lshr i64 %arg1, 35
  br i1 undef, label %bb0, label %bb1

bb0:
  %val0 = and i64 %shr, 255
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

bb1:
  %val1 = and i64 %shr, 127
  store volatile i32 0, ptr addrspace(1) undef
  br label %ret

ret:
  %phi = phi i64 [ %val0, %bb0 ], [ %val1, %bb1 ]
  store i64 %phi, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
