; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-misched=0 -post-RA-scheduler=0 -stress-regalloc=8 < %s | FileCheck %s

; Test that the VGPR spiller correctly switches to SGPR offsets when the
; instruction offset field would overflow, and that it accounts for memory
; swizzling.

; CHECK-LABEL: test_inst_offset_kernel
define amdgpu_kernel void @test_inst_offset_kernel() {
entry:
  ; Occupy 4092 bytes of scratch, so the offset of the spill of %a just fits in
  ; the instruction offset field.
  %alloca = alloca i8, i32 4088, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:4092 ; 4-byte Folded Spill
  %a = load volatile i32, i32 addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  store volatile i32 %a, i32 addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: test_sgpr_offset_kernel
define amdgpu_kernel void @test_sgpr_offset_kernel() {
entry:
  ; Occupy 4096 bytes of scratch, so the offset of the spill of %a does not
  ; fit in the instruction, and has to live in the SGPR offset.
  %alloca = alloca i8, i32 4092, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  ; 0x40000 / 64 = 4096 (for wave64)
  ; CHECK: s_add_u32 s6, s7, 0x40000
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s6 ; 4-byte Folded Spill
  %a = load volatile i32, i32 addrspace(5)* %aptr

  ; Force %a to spill
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  store volatile i32 %a, i32 addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: test_sgpr_offset_kernel_scavenge_fail
define amdgpu_kernel void @test_sgpr_offset_kernel_scavenge_fail() #1 {
entry:
  ; Occupy 4096 bytes of scratch, so the offset of the spill of %a does not
  ; fit in the instruction, and has to live in the SGPR offset.
  %alloca = alloca i8, i32 4092, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1

  ; 0x40000 / 64 = 4096 (for wave64)
  %a = load volatile i32, i32 addrspace(5)* %aptr

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

  ; CHECK: s_add_u32 s7, s7, 0x40000
  ; CHECK: buffer_load_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s7 ; 4-byte Folded Reload
  ; CHECK: s_sub_u32 s7, s7, 0x40000

   ; Force %a to spill with no free SGPRs
  call void asm sideeffect "", "s,s,s,s,s,s,s,s,v"(i32 %asm0, i32 %asm1, i32 %asm2, i32 %asm3, i32 %asm4, i32 %asm5, i32 %asm6, i32 %asm7, i32 %a)
  ret void
}

; CHECK-LABEL: test_sgpr_offset_subregs_kernel
define amdgpu_kernel void @test_sgpr_offset_subregs_kernel() {
entry:
  ; Occupy 4088 bytes of scratch, so that the spill of the last subreg of %a
  ; still fits below offset 4096 (4088 + 8 - 4 = 4092), and can be placed in
  ; the instruction offset field.
  %alloca = alloca i8, i32 4084, align 4, addrspace(5)
  %bufv1 = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*
  %bufv2 = bitcast i8 addrspace(5)* %alloca to <2 x i32> addrspace(5)*

  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:4088 ; 4-byte Folded Spill
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:4092 ; 4-byte Folded Spill
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

; CHECK-LABEL: test_inst_offset_subregs_kernel
define amdgpu_kernel void @test_inst_offset_subregs_kernel() {
entry:
  ; Occupy 4092 bytes of scratch, so that the spill of the last subreg of %a
  ; does not fit below offset 4096 (4092 + 8 - 4 = 4096), and has to live
  ; in the SGPR offset.
  %alloca = alloca i8, i32 4088, align 4, addrspace(5)
  %bufv1 = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*
  %bufv2 = bitcast i8 addrspace(5)* %alloca to <2 x i32> addrspace(5)*

  ; 0x3ff00 / 64 = 4092 (for wave64)
  ; CHECK: s_add_u32 s6, s7, 0x3ff00
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s6 ; 4-byte Folded Spill
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s6 offset:4 ; 4-byte Folded Spill
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

; CHECK-LABEL: test_inst_offset_function
define void @test_inst_offset_function() {
entry:
  ; Occupy 4092 bytes of scratch, so the offset of the spill of %a just fits in
  ; the instruction offset field.
  %alloca = alloca i8, i32 4088, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:4092 ; 4-byte Folded Spill
  %a = load volatile i32, i32 addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  store volatile i32 %a, i32 addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: test_sgpr_offset_function
define void @test_sgpr_offset_function() {
entry:
  ; Occupy 4096 bytes of scratch, so the offset of the spill of %a does not
  ; fit in the instruction, and has to live in the SGPR offset.
  %alloca = alloca i8, i32 4092, align 4, addrspace(5)
  %buf = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*

  %aptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  ; 0x40000 / 64 = 4096 (for wave64)
  ; CHECK: s_add_u32 s6, s5, 0x40000
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s6 ; 4-byte Folded Spill
  %a = load volatile i32, i32 addrspace(5)* %aptr

  ; Force %a to spill
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr i32, i32 addrspace(5)* %buf, i32 1
  store volatile i32 %a, i32 addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: test_sgpr_offset_subregs_function
define void @test_sgpr_offset_subregs_function() {
entry:
  ; Occupy 4088 bytes of scratch, so that the spill of the last subreg of %a
  ; still fits below offset 4096 (4088 + 8 - 4 = 4092), and can be placed in
  ; the instruction offset field.
  %alloca = alloca i8, i32 4084, align 4, addrspace(5)
  %bufv1 = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*
  %bufv2 = bitcast i8 addrspace(5)* %alloca to <2 x i32> addrspace(5)*

  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:4088 ; 4-byte Folded Spill
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:4092 ; 4-byte Folded Spill
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

; CHECK-LABEL: test_inst_offset_subregs_function
define void @test_inst_offset_subregs_function() {
entry:
  ; Occupy 4092 bytes of scratch, so that the spill of the last subreg of %a
  ; does not fit below offset 4096 (4092 + 8 - 4 = 4096), and has to live
  ; in the SGPR offset.
  %alloca = alloca i8, i32 4088, align 4, addrspace(5)
  %bufv1 = bitcast i8 addrspace(5)* %alloca to i32 addrspace(5)*
  %bufv2 = bitcast i8 addrspace(5)* %alloca to <2 x i32> addrspace(5)*

  ; 0x3ff00 / 64 = 4092 (for wave64)
  ; CHECK: s_add_u32 s6, s5, 0x3ff00
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s6 ; 4-byte Folded Spill
  ; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], s6 offset:4 ; 4-byte Folded Spill
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
attributes #1 = { nounwind "amdgpu-num-sgpr"="18" "amdgpu-num-vgpr"="8" }
