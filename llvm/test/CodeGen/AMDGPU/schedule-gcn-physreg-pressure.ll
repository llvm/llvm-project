; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-use-amdgpu-trackers=1 -debug-only=machine-scheduler < %s 2>&1 | FileCheck --check-prefix=GCN-DEBUG %s
; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-use-amdgpu-trackers=0 -debug-only=machine-scheduler < %s 2>&1 | FileCheck --check-prefix=GENERIC-DEBUG %s
; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-use-amdgpu-trackers=1 -amdgpu-trackers-physical-register-tracking=0 -debug-only=machine-scheduler < %s 2>&1 | FileCheck --check-prefix=GCN-NOPHYS-DEBUG %s
; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-use-amdgpu-trackers=1 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-use-amdgpu-trackers=0 < %s | FileCheck --check-prefix=NO-GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-use-amdgpu-trackers=1 -amdgpu-trackers-physical-register-tracking=0 < %s | FileCheck --check-prefix=GCN-NOPHYS %s
; REQUIRES: asserts

; Test that GCN trackers correctly track physical register pressure from inline asm

; GCN-DEBUG-LABEL: test_single_physreg
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6

; GENERIC-DEBUG-LABEL: test_single_physreg
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6

; GCN-NOPHYS-DEBUG-LABEL: test_single_physreg
; GCN-NOPHYS-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6
; GCN-NOPHYS-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6

define amdgpu_kernel void @test_single_physreg(ptr addrspace(1) %out) {
entry:
  %val = call i32 asm sideeffect "s_mov_b32 $0, 0", "={s10}"()
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; Test multiple physical registers

; GCN-DEBUG-LABEL: test_multiple_physregs
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 9, LVGPR WT: 0, LSGPR WT: 6
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 9, LVGPR WT: 0, LSGPR WT: 6

; GENERIC-DEBUG-LABEL: test_multiple_physregs
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 7, LVGPR WT: 0, LSGPR WT: 6
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 7, LVGPR WT: 0, LSGPR WT: 6

; GCN-NOPHYS-DEBUG-LABEL: test_multiple_physregs
; GCN-NOPHYS-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 7, LVGPR WT: 0, LSGPR WT: 6
; GCN-NOPHYS-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 7, LVGPR WT: 0, LSGPR WT: 6

define amdgpu_kernel void @test_multiple_physregs(ptr addrspace(1) %out) {
entry:
  %result = call { i32, i32 } asm sideeffect "s_mov_b32 $0, 0; s_mov_b32 $1, 1", "={s10},={s11}"()
  %r0 = extractvalue { i32, i32 } %result, 0
  %r1 = extractvalue { i32, i32 } %result, 1
  %sum = add i32 %r0, %r1
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; Test physical register with virtual registers

; GCN-DEBUG-LABEL: test_physreg_with_vreg
; GCN-DEBUG: Region register pressure: VGPRs: 2 AGPRs: 0, SGPRs: 10, LVGPR WT: 0, LSGPR WT: 12
; GCN-DEBUG: Pressure after scheduling: VGPRs: 2 AGPRs: 0, SGPRs: 8, LVGPR WT: 0, LSGPR WT: 12

; GENERIC-DEBUG-LABEL: test_physreg_with_vreg
; GENERIC-DEBUG: Region register pressure: VGPRs: 2 AGPRs: 0, SGPRs: 9, LVGPR WT: 0, LSGPR WT: 12
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 2 AGPRs: 0, SGPRs: 7, LVGPR WT: 0, LSGPR WT: 12

; GCN-NOPHYS-DEBUG-LABEL: test_physreg_with_vreg
; GCN-NOPHYS-DEBUG: Region register pressure: VGPRs: 2 AGPRs: 0, SGPRs: 9, LVGPR WT: 0, LSGPR WT: 12
; GCN-NOPHYS-DEBUG: Pressure after scheduling: VGPRs: 2 AGPRs: 0, SGPRs: 7, LVGPR WT: 0, LSGPR WT: 12

define amdgpu_kernel void @test_physreg_with_vreg(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %asm_val = call i32 asm sideeffect "s_mov_b32 $0, 0", "={s10}"()
  %val = load i32, ptr addrspace(1) %in
  %sum = add i32 %asm_val, %val
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; Test early-clobber constraint

; GCN-DEBUG-LABEL: test_early_clobber
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6

; GENERIC-DEBUG-LABEL: test_early_clobber
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6

; GCN-NOPHYS-DEBUG-LABEL: test_early_clobber
; GCN-NOPHYS-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6
; GCN-NOPHYS-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6

define amdgpu_kernel void @test_early_clobber(ptr addrspace(1) %out) {
entry:
  %val = call i32 asm sideeffect "s_mov_b32 $0, 0", "=&{s10}"()
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; Test physical register input

; GCN-DEBUG-LABEL: test_physreg_input
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6

; GENERIC-DEBUG-LABEL: test_physreg_input
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6

; GCN-NOPHYS-DEBUG-LABEL: test_physreg_input
; GCN-NOPHYS-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6
; GCN-NOPHYS-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6

define amdgpu_kernel void @test_physreg_input(ptr addrspace(1) %out) {
entry:
  %val = call i32 asm sideeffect "s_mov_b32 s10, 5; s_add_u32 $0, s10, 1", "={s11}"()
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; Test physical register pressure for tuple (64-bit) registers.
; GCN tracker counts the 2 SGPRs.

; GCN-DEBUG-LABEL: test_tuple_physreg
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 8, LVGPR WT: 0, LSGPR WT: 8
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 8, LVGPR WT: 0, LSGPR WT: 8

; GENERIC-DEBUG-LABEL: test_tuple_physreg
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6

; GCN-NOPHYS-DEBUG-LABEL: test_tuple_physreg
; GCN-NOPHYS-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6
; GCN-NOPHYS-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6

define amdgpu_kernel void @test_tuple_physreg(ptr addrspace(1) %out) {
entry:
  %val = call i64 asm sideeffect "s_mov_b64 $0, 0", "={s[10:11]}"()
  %lo = trunc i64 %val to i32
  store i32 %lo, ptr addrspace(1) %out
  ret void
}

; Test physical register pressure for 128-bit tuple.
; GCN tracker counts the 4 SGPRs.

; GCN-DEBUG-LABEL: test_tuple128_physreg
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 12, LVGPR WT: 0, LSGPR WT: 12
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 12, LVGPR WT: 0, LSGPR WT: 12

; GENERIC-DEBUG-LABEL: test_tuple128_physreg
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 8, LVGPR WT: 0, LSGPR WT: 8
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 8, LVGPR WT: 0, LSGPR WT: 8

; GCN-NOPHYS-DEBUG-LABEL: test_tuple128_physreg
; GCN-NOPHYS-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 8, LVGPR WT: 0, LSGPR WT: 8
; GCN-NOPHYS-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 8, LVGPR WT: 0, LSGPR WT: 8

define amdgpu_kernel void @test_tuple128_physreg(ptr addrspace(1) %out) {
entry:
  %val = call i128 asm sideeffect "s_mov_b64 $0, 0; s_mov_b64 $0+2, 0", "={s[8:11]}"()
  %lo = trunc i128 %val to i32
  store i32 %lo, ptr addrspace(1) %out
  ret void
}

; Test virtual and physical register overlap

; GCN-DEBUG-LABEL: test_vreg_and_physreg_live_range_overlap
; GCN-DEBUG: Region register pressure: VGPRs: 3 AGPRs: 0, SGPRs: 16, LVGPR WT: 0, LSGPR WT: 16
; GCN-DEBUG: Pressure after scheduling: VGPRs: 3 AGPRs: 0, SGPRs: 14, LVGPR WT: 0, LSGPR WT: 16

; GENERIC-DEBUG-LABEL: test_vreg_and_physreg_live_range_overlap
; GENERIC-DEBUG: Region register pressure: VGPRs: 3 AGPRs: 0, SGPRs: 14, LVGPR WT: 0, LSGPR WT: 16
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 3 AGPRs: 0, SGPRs: 12, LVGPR WT: 0, LSGPR WT: 16

; GCN-NOPHYS-DEBUG-LABEL: test_vreg_and_physreg_live_range_overlap
; GCN-NOPHYS-DEBUG: Region register pressure: VGPRs: 3 AGPRs: 0, SGPRs: 14, LVGPR WT: 0, LSGPR WT: 16
; GCN-NOPHYS-DEBUG: Pressure after scheduling: VGPRs: 3 AGPRs: 0, SGPRs: 12, LVGPR WT: 0, LSGPR WT: 16

define amdgpu_kernel void @test_vreg_and_physreg_live_range_overlap(ptr addrspace(1) %in1, ptr addrspace(1) %in2, ptr addrspace(1) %out) {
entry:
  %result = call { i32, i32 } asm sideeffect "s_mov_b32 $0, 0; s_mov_b32 $1, 1", "={s10},={s11}"()
  %val1 = load i32, ptr addrspace(1) %in1
  %val2 = load i32, ptr addrspace(1) %in2
  %sum = add i32 %val1, %val2
  %r0 = extractvalue { i32, i32 } %result, 0
  %r1 = extractvalue { i32, i32 } %result, 1
  %with_asm = add i32 %sum, %r0
  %final = add i32 %with_asm, %r1
  store i32 %final, ptr addrspace(1) %out
  ret void
}

; Verify assembly output for GCN trackers
; GCN-LABEL: test_single_physreg:
; GCN-NEXT: ; %bb.0:
; GCN-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; GCN-NEXT: s_mov_b32 s3, 0xf000
; GCN-NEXT: s_mov_b32 s2, -1
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: s_mov_b32 s10, 0
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: v_mov_b32_e32 v0, s10
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], 0
; GCN-NEXT: s_endpgm
; GCN: .set test_single_physreg.numbered_sgpr, 11
; GCN: TotalNumSgprs: 11
; GCN: NumVgprs: 1

; GCN-LABEL: test_multiple_physregs:
; GCN-NEXT: ; %bb.0:
; GCN-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: s_mov_b32 s10, 0; s_mov_b32 s11, 1
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_add_i32 s4, s10, s11
; GCN-NEXT: s_mov_b32 s3, 0xf000
; GCN-NEXT: s_mov_b32 s2, -1
; GCN-NEXT: v_mov_b32_e32 v0, s4
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], 0
; GCN-NEXT: s_endpgm
; GCN: .set test_multiple_physregs.numbered_sgpr, 12
; GCN: TotalNumSgprs: 12
; GCN: NumVgprs: 1

; GCN-LABEL: test_physreg_with_vreg:
; GCN-NEXT: ; %bb.0:
; GCN-NEXT: s_load_dwordx4 s[0:3], s[4:5], 0x9
; GCN-NEXT: s_mov_b32 s7, 0xf000
; GCN-NEXT: s_mov_b32 s6, -1
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: s_mov_b32 s10, 0
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_mov_b32 s4, s0
; GCN-NEXT: s_mov_b32 s5, s1
; GCN-NEXT: buffer_load_dword v0, off, s[4:7], 0
; GCN-NEXT: s_mov_b32 s4, s2
; GCN-NEXT: s_mov_b32 s5, s3
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: v_add_i32_e32 v0, vcc, s10, v0
; GCN-NEXT: buffer_store_dword v0, off, s[4:7], 0
; GCN-NEXT: s_endpgm
; GCN: .set test_physreg_with_vreg.numbered_sgpr, 11
; GCN: TotalNumSgprs: 13
; GCN: NumVgprs: 1

; GCN-LABEL: test_early_clobber:
; GCN-NEXT: ; %bb.0:
; GCN-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; GCN-NEXT: s_mov_b32 s3, 0xf000
; GCN-NEXT: s_mov_b32 s2, -1
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: s_mov_b32 s10, 0
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: v_mov_b32_e32 v0, s10
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], 0
; GCN-NEXT: s_endpgm
; GCN: .set test_early_clobber.numbered_sgpr, 11
; GCN: TotalNumSgprs: 11
; GCN: NumVgprs: 1

; GCN-LABEL: test_physreg_input:
; GCN-NEXT: ; %bb.0:
; GCN-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; GCN-NEXT: s_mov_b32 s3, 0xf000
; GCN-NEXT: s_mov_b32 s2, -1
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: s_mov_b32 s10, 5; s_add_u32 s11, s10, 1
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: v_mov_b32_e32 v0, s11
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], 0
; GCN-NEXT: s_endpgm
; GCN: .set test_physreg_input.numbered_sgpr, 12
; GCN: TotalNumSgprs: 12
; GCN: NumVgprs: 1

; GCN-LABEL: test_tuple_physreg:
; GCN: ;;#ASMSTART
; GCN: s_mov_b64 s[10:11], 0
; GCN: ;;#ASMEND
; GCN: v_mov_b32_e32 v0, s10
; GCN: .set test_tuple_physreg.numbered_sgpr, 12
; GCN: TotalNumSgprs: 12
; GCN: NumVgprs: 1

; GCN-LABEL: test_tuple128_physreg:
; GCN: ;;#ASMSTART
; GCN: s_mov_b64 s[8:11], 0; s_mov_b64 s[8:11]+2, 0
; GCN: ;;#ASMEND
; GCN: v_mov_b32_e32 v0, s8
; GCN: .set test_tuple128_physreg.numbered_sgpr, 12
; GCN: TotalNumSgprs: 12
; GCN: NumVgprs: 1

; GCN-LABEL: test_vreg_and_physreg_live_range_overlap:
; GCN-NEXT: ; %bb.0:
; GCN-NEXT: s_load_dwordx4 s[0:3], s[4:5], 0x9
; GCN-NEXT: s_load_dwordx2 s[12:13], s[4:5], 0xd
; GCN-NEXT: s_mov_b32 s7, 0xf000
; GCN-NEXT: s_mov_b32 s6, -1
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: s_mov_b32 s10, 0; s_mov_b32 s11, 1
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_mov_b32 s4, s0
; GCN-NEXT: s_mov_b32 s5, s1
; GCN-NEXT: s_mov_b32 s0, s2
; GCN-NEXT: s_mov_b32 s1, s3
; GCN-NEXT: s_mov_b32 s2, s6
; GCN-NEXT: s_mov_b32 s3, s7
; GCN-NEXT: buffer_load_dword v0, off, s[4:7], 0
; GCN-NEXT: buffer_load_dword v1, off, s[0:3], 0
; GCN-NEXT: s_mov_b32 s14, s6
; GCN-NEXT: s_mov_b32 s15, s7
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: v_add_i32_e32 v0, vcc, v0, v1
; GCN-NEXT: v_add_i32_e32 v0, vcc, s10, v0
; GCN-NEXT: v_add_i32_e32 v0, vcc, s11, v0
; GCN-NEXT: buffer_store_dword v0, off, s[12:15], 0
; GCN-NEXT: s_endpgm
; GCN: .set test_vreg_and_physreg_live_range_overlap.numbered_sgpr, 16
; GCN: TotalNumSgprs: 18
; GCN: NumVgprs: 2

; Verify assembly output with GCN trackers but physical register tracking disabled (same as GCN)
; GCN-NOPHYS-LABEL: test_single_physreg:
; GCN-NOPHYS-NEXT: ; %bb.0:
; GCN-NOPHYS-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; GCN-NOPHYS-NEXT: s_mov_b32 s3, 0xf000
; GCN-NOPHYS-NEXT: s_mov_b32 s2, -1
; GCN-NOPHYS-NEXT: ;;#ASMSTART
; GCN-NOPHYS-NEXT: s_mov_b32 s10, 0
; GCN-NOPHYS-NEXT: ;;#ASMEND
; GCN-NOPHYS-NEXT: v_mov_b32_e32 v0, s10
; GCN-NOPHYS-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NOPHYS-NEXT: buffer_store_dword v0, off, s[0:3], 0
; GCN-NOPHYS-NEXT: s_endpgm
; GCN-NOPHYS: .set test_single_physreg.numbered_sgpr, 11
; GCN-NOPHYS: TotalNumSgprs: 11
; GCN-NOPHYS: NumVgprs: 1

; GCN-NOPHYS-LABEL: test_multiple_physregs:
; GCN-NOPHYS-NEXT: ; %bb.0:
; GCN-NOPHYS-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; GCN-NOPHYS-NEXT: ;;#ASMSTART
; GCN-NOPHYS-NEXT: s_mov_b32 s10, 0; s_mov_b32 s11, 1
; GCN-NOPHYS-NEXT: ;;#ASMEND
; GCN-NOPHYS-NEXT: s_add_i32 s4, s10, s11
; GCN-NOPHYS-NEXT: s_mov_b32 s3, 0xf000
; GCN-NOPHYS-NEXT: s_mov_b32 s2, -1
; GCN-NOPHYS-NEXT: v_mov_b32_e32 v0, s4
; GCN-NOPHYS-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NOPHYS-NEXT: buffer_store_dword v0, off, s[0:3], 0
; GCN-NOPHYS-NEXT: s_endpgm
; GCN-NOPHYS: .set test_multiple_physregs.numbered_sgpr, 12
; GCN-NOPHYS: TotalNumSgprs: 12
; GCN-NOPHYS: NumVgprs: 1

; GCN-NOPHYS-LABEL: test_physreg_with_vreg:
; GCN-NOPHYS-NEXT: ; %bb.0:
; GCN-NOPHYS-NEXT: s_load_dwordx4 s[0:3], s[4:5], 0x9
; GCN-NOPHYS-NEXT: s_mov_b32 s7, 0xf000
; GCN-NOPHYS-NEXT: s_mov_b32 s6, -1
; GCN-NOPHYS-NEXT: ;;#ASMSTART
; GCN-NOPHYS-NEXT: s_mov_b32 s10, 0
; GCN-NOPHYS-NEXT: ;;#ASMEND
; GCN-NOPHYS-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NOPHYS-NEXT: s_mov_b32 s4, s0
; GCN-NOPHYS-NEXT: s_mov_b32 s5, s1
; GCN-NOPHYS-NEXT: buffer_load_dword v0, off, s[4:7], 0
; GCN-NOPHYS-NEXT: s_mov_b32 s4, s2
; GCN-NOPHYS-NEXT: s_mov_b32 s5, s3
; GCN-NOPHYS-NEXT: s_waitcnt vmcnt(0)
; GCN-NOPHYS-NEXT: v_add_i32_e32 v0, vcc, s10, v0
; GCN-NOPHYS-NEXT: buffer_store_dword v0, off, s[4:7], 0
; GCN-NOPHYS-NEXT: s_endpgm
; GCN-NOPHYS: .set test_physreg_with_vreg.numbered_sgpr, 11
; GCN-NOPHYS: TotalNumSgprs: 13
; GCN-NOPHYS: NumVgprs: 1

; GCN-NOPHYS-LABEL: test_early_clobber:
; GCN-NOPHYS-NEXT: ; %bb.0:
; GCN-NOPHYS-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; GCN-NOPHYS-NEXT: s_mov_b32 s3, 0xf000
; GCN-NOPHYS-NEXT: s_mov_b32 s2, -1
; GCN-NOPHYS-NEXT: ;;#ASMSTART
; GCN-NOPHYS-NEXT: s_mov_b32 s10, 0
; GCN-NOPHYS-NEXT: ;;#ASMEND
; GCN-NOPHYS-NEXT: v_mov_b32_e32 v0, s10
; GCN-NOPHYS-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NOPHYS-NEXT: buffer_store_dword v0, off, s[0:3], 0
; GCN-NOPHYS-NEXT: s_endpgm
; GCN-NOPHYS: .set test_early_clobber.numbered_sgpr, 11
; GCN-NOPHYS: TotalNumSgprs: 11
; GCN-NOPHYS: NumVgprs: 1

; GCN-NOPHYS-LABEL: test_physreg_input:
; GCN-NOPHYS-NEXT: ; %bb.0:
; GCN-NOPHYS-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; GCN-NOPHYS-NEXT: s_mov_b32 s3, 0xf000
; GCN-NOPHYS-NEXT: s_mov_b32 s2, -1
; GCN-NOPHYS-NEXT: ;;#ASMSTART
; GCN-NOPHYS-NEXT: s_mov_b32 s10, 5; s_add_u32 s11, s10, 1
; GCN-NOPHYS-NEXT: ;;#ASMEND
; GCN-NOPHYS-NEXT: v_mov_b32_e32 v0, s11
; GCN-NOPHYS-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NOPHYS-NEXT: buffer_store_dword v0, off, s[0:3], 0
; GCN-NOPHYS-NEXT: s_endpgm
; GCN-NOPHYS: .set test_physreg_input.numbered_sgpr, 12
; GCN-NOPHYS: TotalNumSgprs: 12
; GCN-NOPHYS: NumVgprs: 1

; GCN-NOPHYS-LABEL: test_tuple_physreg:
; GCN-NOPHYS: ;;#ASMSTART
; GCN-NOPHYS: s_mov_b64 s[10:11], 0
; GCN-NOPHYS: ;;#ASMEND
; GCN-NOPHYS: v_mov_b32_e32 v0, s10
; GCN-NOPHYS: .set test_tuple_physreg.numbered_sgpr, 12
; GCN-NOPHYS: TotalNumSgprs: 12
; GCN-NOPHYS: NumVgprs: 1

; GCN-NOPHYS-LABEL: test_tuple128_physreg:
; GCN-NOPHYS: ;;#ASMSTART
; GCN-NOPHYS: s_mov_b64 s[8:11], 0; s_mov_b64 s[8:11]+2, 0
; GCN-NOPHYS: ;;#ASMEND
; GCN-NOPHYS: v_mov_b32_e32 v0, s8
; GCN-NOPHYS: .set test_tuple128_physreg.numbered_sgpr, 12
; GCN-NOPHYS: TotalNumSgprs: 12
; GCN-NOPHYS: NumVgprs: 1

; GCN-NOPHYS-LABEL: test_vreg_and_physreg_live_range_overlap:
; GCN-NOPHYS-NEXT: ; %bb.0:
; GCN-NOPHYS-NEXT: s_load_dwordx4 s[0:3], s[4:5], 0x9
; GCN-NOPHYS-NEXT: s_load_dwordx2 s[12:13], s[4:5], 0xd
; GCN-NOPHYS-NEXT: s_mov_b32 s7, 0xf000
; GCN-NOPHYS-NEXT: s_mov_b32 s6, -1
; GCN-NOPHYS-NEXT: ;;#ASMSTART
; GCN-NOPHYS-NEXT: s_mov_b32 s10, 0; s_mov_b32 s11, 1
; GCN-NOPHYS-NEXT: ;;#ASMEND
; GCN-NOPHYS-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NOPHYS-NEXT: s_mov_b32 s4, s0
; GCN-NOPHYS-NEXT: s_mov_b32 s5, s1
; GCN-NOPHYS-NEXT: s_mov_b32 s0, s2
; GCN-NOPHYS-NEXT: s_mov_b32 s1, s3
; GCN-NOPHYS-NEXT: s_mov_b32 s2, s6
; GCN-NOPHYS-NEXT: s_mov_b32 s3, s7
; GCN-NOPHYS-NEXT: buffer_load_dword v0, off, s[4:7], 0
; GCN-NOPHYS-NEXT: buffer_load_dword v1, off, s[0:3], 0
; GCN-NOPHYS-NEXT: s_mov_b32 s14, s6
; GCN-NOPHYS-NEXT: s_mov_b32 s15, s7
; GCN-NOPHYS-NEXT: s_waitcnt vmcnt(0)
; GCN-NOPHYS-NEXT: v_add_i32_e32 v0, vcc, v0, v1
; GCN-NOPHYS-NEXT: v_add_i32_e32 v0, vcc, s10, v0
; GCN-NOPHYS-NEXT: v_add_i32_e32 v0, vcc, s11, v0
; GCN-NOPHYS-NEXT: buffer_store_dword v0, off, s[12:15], 0
; GCN-NOPHYS-NEXT: s_endpgm
; GCN-NOPHYS: .set test_vreg_and_physreg_live_range_overlap.numbered_sgpr, 16
; GCN-NOPHYS: TotalNumSgprs: 18
; GCN-NOPHYS: NumVgprs: 2

; Verify assembly output without GCN trackers (should be identical)
; NO-GCN-LABEL: test_single_physreg:
; NO-GCN-NEXT: ; %bb.0:
; NO-GCN-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; NO-GCN-NEXT: s_mov_b32 s3, 0xf000
; NO-GCN-NEXT: s_mov_b32 s2, -1
; NO-GCN-NEXT: ;;#ASMSTART
; NO-GCN-NEXT: s_mov_b32 s10, 0
; NO-GCN-NEXT: ;;#ASMEND
; NO-GCN-NEXT: v_mov_b32_e32 v0, s10
; NO-GCN-NEXT: s_waitcnt lgkmcnt(0)
; NO-GCN-NEXT: buffer_store_dword v0, off, s[0:3], 0
; NO-GCN-NEXT: s_endpgm
; NO-GCN: .set test_single_physreg.numbered_sgpr, 11
; NO-GCN: TotalNumSgprs: 11
; NO-GCN: NumVgprs: 1

; NO-GCN-LABEL: test_multiple_physregs:
; NO-GCN-NEXT: ; %bb.0:
; NO-GCN-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; NO-GCN-NEXT: ;;#ASMSTART
; NO-GCN-NEXT: s_mov_b32 s10, 0; s_mov_b32 s11, 1
; NO-GCN-NEXT: ;;#ASMEND
; NO-GCN-NEXT: s_add_i32 s4, s10, s11
; NO-GCN-NEXT: s_mov_b32 s3, 0xf000
; NO-GCN-NEXT: s_mov_b32 s2, -1
; NO-GCN-NEXT: v_mov_b32_e32 v0, s4
; NO-GCN-NEXT: s_waitcnt lgkmcnt(0)
; NO-GCN-NEXT: buffer_store_dword v0, off, s[0:3], 0
; NO-GCN-NEXT: s_endpgm
; NO-GCN: .set test_multiple_physregs.numbered_sgpr, 12
; NO-GCN: TotalNumSgprs: 12
; NO-GCN: NumVgprs: 1

; NO-GCN-LABEL: test_physreg_with_vreg:
; NO-GCN-NEXT: ; %bb.0:
; NO-GCN-NEXT: s_load_dwordx4 s[0:3], s[4:5], 0x9
; NO-GCN-NEXT: s_mov_b32 s7, 0xf000
; NO-GCN-NEXT: s_mov_b32 s6, -1
; NO-GCN-NEXT: ;;#ASMSTART
; NO-GCN-NEXT: s_mov_b32 s10, 0
; NO-GCN-NEXT: ;;#ASMEND
; NO-GCN-NEXT: s_waitcnt lgkmcnt(0)
; NO-GCN-NEXT: s_mov_b32 s4, s0
; NO-GCN-NEXT: s_mov_b32 s5, s1
; NO-GCN-NEXT: buffer_load_dword v0, off, s[4:7], 0
; NO-GCN-NEXT: s_mov_b32 s4, s2
; NO-GCN-NEXT: s_mov_b32 s5, s3
; NO-GCN-NEXT: s_waitcnt vmcnt(0)
; NO-GCN-NEXT: v_add_i32_e32 v0, vcc, s10, v0
; NO-GCN-NEXT: buffer_store_dword v0, off, s[4:7], 0
; NO-GCN-NEXT: s_endpgm
; NO-GCN: .set test_physreg_with_vreg.numbered_sgpr, 11
; NO-GCN: TotalNumSgprs: 13
; NO-GCN: NumVgprs: 1

; NO-GCN-LABEL: test_early_clobber:
; NO-GCN-NEXT: ; %bb.0:
; NO-GCN-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; NO-GCN-NEXT: s_mov_b32 s3, 0xf000
; NO-GCN-NEXT: s_mov_b32 s2, -1
; NO-GCN-NEXT: ;;#ASMSTART
; NO-GCN-NEXT: s_mov_b32 s10, 0
; NO-GCN-NEXT: ;;#ASMEND
; NO-GCN-NEXT: v_mov_b32_e32 v0, s10
; NO-GCN-NEXT: s_waitcnt lgkmcnt(0)
; NO-GCN-NEXT: buffer_store_dword v0, off, s[0:3], 0
; NO-GCN-NEXT: s_endpgm
; NO-GCN: .set test_early_clobber.numbered_sgpr, 11
; NO-GCN: TotalNumSgprs: 11
; NO-GCN: NumVgprs: 1

; NO-GCN-LABEL: test_physreg_input:
; NO-GCN-NEXT: ; %bb.0:
; NO-GCN-NEXT: s_load_dwordx2 s[0:1], s[4:5], 0x9
; NO-GCN-NEXT: s_mov_b32 s3, 0xf000
; NO-GCN-NEXT: s_mov_b32 s2, -1
; NO-GCN-NEXT: ;;#ASMSTART
; NO-GCN-NEXT: s_mov_b32 s10, 5; s_add_u32 s11, s10, 1
; NO-GCN-NEXT: ;;#ASMEND
; NO-GCN-NEXT: v_mov_b32_e32 v0, s11
; NO-GCN-NEXT: s_waitcnt lgkmcnt(0)
; NO-GCN-NEXT: buffer_store_dword v0, off, s[0:3], 0
; NO-GCN-NEXT: s_endpgm
; NO-GCN: .set test_physreg_input.numbered_sgpr, 12
; NO-GCN: TotalNumSgprs: 12
; NO-GCN: NumVgprs: 1

; NO-GCN-LABEL: test_tuple_physreg:
; NO-GCN: ;;#ASMSTART
; NO-GCN: s_mov_b64 s[10:11], 0
; NO-GCN: ;;#ASMEND
; NO-GCN: v_mov_b32_e32 v0, s10
; NO-GCN: .set test_tuple_physreg.numbered_sgpr, 12
; NO-GCN: TotalNumSgprs: 12
; NO-GCN: NumVgprs: 1

; NO-GCN-LABEL: test_tuple128_physreg:
; NO-GCN: ;;#ASMSTART
; NO-GCN: s_mov_b64 s[8:11], 0; s_mov_b64 s[8:11]+2, 0
; NO-GCN: ;;#ASMEND
; NO-GCN: v_mov_b32_e32 v0, s8
; NO-GCN: .set test_tuple128_physreg.numbered_sgpr, 12
; NO-GCN: TotalNumSgprs: 12
; NO-GCN: NumVgprs: 1

; NO-GCN-LABEL: test_vreg_and_physreg_live_range_overlap:
; NO-GCN-NEXT: ; %bb.0:
; NO-GCN-NEXT: s_load_dwordx4 s[0:3], s[4:5], 0x9
; NO-GCN-NEXT: s_load_dwordx2 s[12:13], s[4:5], 0xd
; NO-GCN-NEXT: s_mov_b32 s7, 0xf000
; NO-GCN-NEXT: s_mov_b32 s6, -1
; NO-GCN-NEXT: ;;#ASMSTART
; NO-GCN-NEXT: s_mov_b32 s10, 0; s_mov_b32 s11, 1
; NO-GCN-NEXT: ;;#ASMEND
; NO-GCN-NEXT: s_waitcnt lgkmcnt(0)
; NO-GCN-NEXT: s_mov_b32 s4, s0
; NO-GCN-NEXT: s_mov_b32 s5, s1
; NO-GCN-NEXT: s_mov_b32 s0, s2
; NO-GCN-NEXT: s_mov_b32 s1, s3
; NO-GCN-NEXT: s_mov_b32 s2, s6
; NO-GCN-NEXT: s_mov_b32 s3, s7
; NO-GCN-NEXT: buffer_load_dword v0, off, s[4:7], 0
; NO-GCN-NEXT: buffer_load_dword v1, off, s[0:3], 0
; NO-GCN-NEXT: s_mov_b32 s14, s6
; NO-GCN-NEXT: s_mov_b32 s15, s7
; NO-GCN-NEXT: s_waitcnt vmcnt(0)
; NO-GCN-NEXT: v_add_i32_e32 v0, vcc, v0, v1
; NO-GCN-NEXT: v_add_i32_e32 v0, vcc, s10, v0
; NO-GCN-NEXT: v_add_i32_e32 v0, vcc, s11, v0
; NO-GCN-NEXT: buffer_store_dword v0, off, s[12:15], 0
; NO-GCN-NEXT: s_endpgm
; NO-GCN: .set test_vreg_and_physreg_live_range_overlap.numbered_sgpr, 16
; NO-GCN: TotalNumSgprs: 18
; NO-GCN: NumVgprs: 2
