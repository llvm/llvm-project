; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-misched=0 -post-RA-scheduler=0 -stress-regalloc=8 < %s | FileCheck -check-prefixes=GCN,MUBUF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-misched=0 -post-RA-scheduler=0 -stress-regalloc=8 -amdgpu-enable-flat-scratch < %s | FileCheck -check-prefixes=GCN,FLATSCR %s

; Test that the VGPR spiller correctly switches to SGPR offsets when the
; instruction offset field would overflow, and that it accounts for memory
; swizzling.

; GCN-LABEL: test_inst_offset_kernel
define amdgpu_kernel void @test_inst_offset_kernel() {
entry:
  ; Occupy 4092 bytes of scratch, so the offset of the spill of %a just fits in
  ; the instruction offset field.
  %alloca = alloca i8, i32 4088, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4092 ; 4-byte Folded Spill
  ; FLATSCR: scratch_store_dword off, v{{[0-9]+}}, s{{[0-9]+}} ; 4-byte Folded Spill
  %a = load volatile i32, i32 addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  store volatile i32 %a, i32 addrspace(5)* %outptr

  ret void
}

; GCN-LABEL: test_sgpr_offset_kernel
define amdgpu_kernel void @test_sgpr_offset_kernel() {
entry:
  ; Occupy 4096 bytes of scratch, so the offset of the spill of %a does not
  ; fit in the instruction, and has to live in the SGPR offset.
  %alloca = alloca i8, i32 4092, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  ; 0x40000 / 64 = 4096 (for wave64)
  ; MUBUF:   s_mov_b32 s6, 0x40000
  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s6 ; 4-byte Folded Spill
  ; FLATSCR: s_movk_i32 s2, 0x1000
  ; FLATSCR: scratch_store_dword off, v{{[0-9]+}}, s2 ; 4-byte Folded Spill
  %a = load volatile i32, i32 addrspace(5)* %aptr

  ; Force %a to spill
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  store volatile i32 %a, i32 addrspace(5)* %outptr

  ret void
}

; FIXME: If we fail to scavenge an SGPR in a kernel we don't have a stack
; pointer to temporarily update, so we just crash.

; GCN-LABEL: test_sgpr_offset_function_scavenge_fail
define void @test_sgpr_offset_function_scavenge_fail() #2 {
entry:
  ; Occupy 4096 bytes of scratch, so the offset of the spill of %a does not
  ; fit in the instruction, and has to live in the SGPR offset.
  %alloca = alloca i8, i32 4096, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1

  %asm.0 = call { i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "=s,=s,=s,=s,=s,=s,=s,=s"()
  %asm0.0 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm.0, 0
  %asm1.0 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm.0, 1
  %asm2.0 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm.0, 2
  %asm3.0 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm.0, 3
  %asm4.0 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm.0, 4
  %asm5.0 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm.0, 5
  %asm6.0 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm.0, 6
  %asm7.0 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm.0, 7

  ; 0x40000 / 64 = 4096 (for wave64)
  %a = load volatile i32, i32 addrspace(5)* %aptr

  ; MUBUF:   s_add_u32 s32, s32, 0x40000
  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s32 ; 4-byte Folded Spill
  ; MUBUF:   s_sub_u32 s32, s32, 0x40000
  ; FLATSCR: s_add_u32 [[SOFF:s[0-9+]]], s32, 0x1000
  ; FLATSCR: scratch_store_dword off, v{{[0-9]+}}, [[SOFF]] ; 4-byte Folded Spill
  call void asm sideeffect "", "s,s,s,s,s,s,s,s,v"(i32 %asm0.0, i32 %asm1.0, i32 %asm2.0, i32 %asm3.0, i32 %asm4.0, i32 %asm5.0, i32 %asm6.0, i32 %asm7.0, i32 %a)

  %asm = call { i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "=s,=s,=s,=s,=s,=s,=s,=s"()
  %asm0 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm, 0
  %asm1 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm, 1
  %asm2 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm, 2
  %asm3 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm, 3
  %asm4 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm, 4
  %asm5 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm, 5
  %asm6 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm, 6
  %asm7 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm, 7

  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"() #0

  ; MUBUF:   s_add_u32 s32, s32, 0x40000
  ; MUBUF:   buffer_load_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s32 ; 4-byte Folded Reload
  ; MUBUF:   s_sub_u32 s32, s32, 0x40000
  ; FLATSCR: s_add_u32 [[SOFF:s[0-9+]]], s32, 0x1000
  ; FLATSCR: scratch_load_dword v{{[0-9]+}}, off, [[SOFF]] ; 4-byte Folded Reload

   ; Force %a to spill with no free SGPRs
  call void asm sideeffect "", "s,s,s,s,s,s,s,s,v"(i32 %asm0, i32 %asm1, i32 %asm2, i32 %asm3, i32 %asm4, i32 %asm5, i32 %asm6, i32 %asm7, i32 %a)
  ret void
}

; GCN-LABEL: test_sgpr_offset_subregs_kernel
define amdgpu_kernel void @test_sgpr_offset_subregs_kernel() {
entry:
  ; Occupy 4088 bytes of scratch, so that the spill of the last subreg of %a
  ; still fits below offset 4096 (4088 + 8 - 4 = 4092), and can be placed in
  ; the instruction offset field.
  %alloca = alloca i8, i32 4084, align 4, addrspace(5)
  %bufv1 = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*
  %bufv2 = bitcast i8 addrspace(5)* %alloca to <2 x i32> addrspace(5)*

  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4088 ; 4-byte Folded Spill
  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4092 ; 4-byte Folded Spill
  ; FLATSCR: s_movk_i32 [[SOFF:s[0-9]+]], 0xff8
  ; FLATSCR: scratch_store_dwordx2 off, v[{{[0-9:]+}}], [[SOFF]]          ; 8-byte Folded Spill
  %aptr = getelementptr <2 x i32>, <2 x i32> addrspace(5)* %bufv2, i32 1
  %a = load volatile <2 x i32>, <2 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  ; Ensure the alloca sticks around.
  %bptr = getelementptr i32, i32 addrspace(5)* %bufv1, i32 1
  %b = load volatile i32, i32 addrspace(5)* %bptr

  ; Ensure the spill is of the full super-reg.
  call void asm sideeffect "; $0", "r"(<2 x i32> %a)

  ret void
}

; GCN-LABEL: test_inst_offset_subregs_kernel
define amdgpu_kernel void @test_inst_offset_subregs_kernel() {
entry:
  ; Occupy 4092 bytes of scratch, so that the spill of the last subreg of %a
  ; does not fit below offset 4096 (4092 + 8 - 4 = 4096), and has to live
  ; in the SGPR offset.
  %alloca = alloca i8, i32 4088, align 4, addrspace(5)
  %bufv1 = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*
  %bufv2 = bitcast i8 addrspace(5)* %alloca to <2 x i32> addrspace(5)*

  ; 0x3ff00 / 64 = 4092 (for wave64)
  ; MUBUF:   s_mov_b32 s6, 0x3ff00
  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s6 ; 4-byte Folded Spill
  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s6 offset:4 ; 4-byte Folded Spill
  ; FLATSCR: s_movk_i32 [[SOFF:s[0-9]+]], 0xffc
  ; FLATSCR: scratch_store_dwordx2 off, v[{{[0-9:]+}}], [[SOFF]]          ; 8-byte Folded Spill
  %aptr = getelementptr <2 x i32>, <2 x i32> addrspace(5)* %bufv2, i32 1
  %a = load volatile <2 x i32>, <2 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  ; Ensure the alloca sticks around.
  %bptr = getelementptr i32, i32 addrspace(5)* %bufv1, i32 1
  %b = load volatile i32, i32 addrspace(5)* %bptr

  ; Ensure the spill is of the full super-reg.
  call void asm sideeffect "; $0", "r"(<2 x i32> %a)

  ret void
}

; GCN-LABEL: test_inst_offset_function
define void @test_inst_offset_function() {
entry:
  ; Occupy 4092 bytes of scratch, so the offset of the spill of %a just fits in
  ; the instruction offset field.
  %alloca = alloca i8, i32 4092, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:4092 ; 4-byte Folded Spill
  ; FLATSCR: scratch_store_dword off, v{{[0-9]+}}, s{{[0-9]+}} offset:4092 ; 4-byte Folded Spill
  %a = load volatile i32, i32 addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  store volatile i32 %a, i32 addrspace(5)* %outptr

  ret void
}

; GCN-LABEL: test_sgpr_offset_function
define void @test_sgpr_offset_function() {
entry:
  ; Occupy 4096 bytes of scratch, so the offset of the spill of %a does not
  ; fit in the instruction, and has to live in the SGPR offset.
  %alloca = alloca i8, i32 4096, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  ; 0x40000 / 64 = 4096 (for wave64)
  ; MUBUF:   s_add_u32 s4, s32, 0x40000
  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s4 ; 4-byte Folded Spill
  ; FLATSCR: s_add_u32 s0, s32, 0x1000
  ; FLATSCR: scratch_store_dword off, v{{[0-9]+}}, s0 ; 4-byte Folded Spill
  %a = load volatile i32, i32 addrspace(5)* %aptr

  ; Force %a to spill
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  store volatile i32 %a, i32 addrspace(5)* %outptr

  ret void
}

; GCN-LABEL: test_sgpr_offset_subregs_function
define void @test_sgpr_offset_subregs_function() {
entry:
  ; Occupy 4088 bytes of scratch, so that the spill of the last subreg of %a
  ; still fits below offset 4096 (4088 + 8 - 4 = 4092), and can be placed in
  ; the instruction offset field.
  %alloca = alloca i8, i32 4088, align 4, addrspace(5)
  %bufv1 = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*
  %bufv2 = bitcast i8 addrspace(5)* %alloca to <2 x i32> addrspace(5)*

  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:4088 ; 4-byte Folded Spill
  ; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:4092 ; 4-byte Folded Spill
  ; FLATSCR: scratch_store_dwordx2 off, v[{{[0-9:]+}}], s32 offset:4088 ; 8-byte Folded Spill
  %aptr = getelementptr <2 x i32>, <2 x i32> addrspace(5)* %bufv2, i32 1
  %a = load volatile <2 x i32>, <2 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  ; Ensure the alloca sticks around.
  %bptr = getelementptr i32, i32 addrspace(5)* %bufv1, i32 1
  %b = load volatile i32, i32 addrspace(5)* %bptr

  ; Ensure the spill is of the full super-reg.
  call void asm sideeffect "; $0", "r"(<2 x i32> %a)

  ret void
}

; GCN-LABEL: test_inst_offset_subregs_function
define void @test_inst_offset_subregs_function() {
entry:
  ; Occupy 4092 bytes of scratch, so that the spill of the last subreg of %a
  ; does not fit below offset 4096 (4092 + 8 - 4 = 4096), and has to live
  ; in the SGPR offset.
  %alloca = alloca i8, i32 4092, align 4, addrspace(5)
  %bufv1 = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*
  %bufv2 = bitcast i8 addrspace(5)* %alloca to <2 x i32> addrspace(5)*

  ; 0x3ff00 / 64 = 4092 (for wave64)
  ; MUBUF: s_add_u32 s4, s32, 0x3ff00
  ; MUBUF: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s4 ; 4-byte Folded Spill
  ; MUBUF: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s4 offset:4 ; 4-byte Folded Spill
  ; FLATSCR: scratch_store_dwordx2 off, v[{{[0-9:]+}}], s32 offset:4092 ; 8-byte Folded Spill
  %aptr = getelementptr <2 x i32>, <2 x i32> addrspace(5)* %bufv2, i32 1
  %a = load volatile <2 x i32>, <2 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  ; Ensure the alloca sticks around.
  %bptr = getelementptr i32, i32 addrspace(5)* %bufv1, i32 1
  %b = load volatile i32, i32 addrspace(5)* %bptr

  ; Ensure the spill is of the full super-reg.
  call void asm sideeffect "; $0", "r"(<2 x i32> %a)

  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-num-sgpr"="17" "amdgpu-num-vgpr"="8" }
attributes #2 = { nounwind "amdgpu-num-sgpr"="14" "amdgpu-num-vgpr"="8" }
