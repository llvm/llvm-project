; RUN: llc -global-isel -stop-after=irtranslator -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GFX9 -enable-var-scope %s
; RUN: llc -global-isel -stop-after=irtranslator -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GFX11 -enable-var-scope %s

define i1 @i1_func_void() {
; GFX9-LABEL: name: i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    $sgpr4_sgpr5 = COPY [[LOAD]](s1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    $sgpr0 = COPY [[LOAD]](s1)
; GFX11-NEXT:    SI_RETURN
  %val = load i1, ptr addrspace(1) undef
  ret i1 %val
}

define void @test_call_i1_func_void() {
; GFX9-LABEL: name: test_call_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @i1_func_void
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @i1_func_void, csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @i1_func_void
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @i1_func_void, csr_amdgpu, implicit-def $sgpr0
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %val = call i1 @i1_func_void()
  store volatile i1 %val, ptr addrspace(1) undef
  ret void
}

define zeroext i1 @zeroext_i1_func_void() {
; GFX9-LABEL: name: zeroext_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    $sgpr4_sgpr5 = COPY [[LOAD]](s1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: zeroext_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    $sgpr0 = COPY [[LOAD]](s1)
; GFX11-NEXT:    SI_RETURN
  %val = load i1, ptr addrspace(1) undef
  ret i1 %val
}

define void @test_call_zeroext_i1_func_void() {
; GFX9-LABEL: name: test_call_zeroext_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @zeroext_i1_func_void
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @zeroext_i1_func_void, csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_zeroext_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @zeroext_i1_func_void
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @zeroext_i1_func_void, csr_amdgpu, implicit-def $sgpr0
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %val = call i1 @zeroext_i1_func_void()
  store volatile i1 %val, ptr addrspace(1) undef
  ret void
}

define signext i1 @signext_i1_func_void() {
; GFX9-LABEL: name: signext_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    $sgpr4_sgpr5 = COPY [[LOAD]](s1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: signext_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    $sgpr0 = COPY [[LOAD]](s1)
; GFX11-NEXT:    SI_RETURN
  %val = load i1, ptr addrspace(1) undef
  ret i1 %val
}

define void @test_call_signext_i1_func_void() {
; GFX9-LABEL: name: test_call_signext_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @signext_i1_func_void
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @signext_i1_func_void, csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_signext_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @signext_i1_func_void
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @signext_i1_func_void, csr_amdgpu, implicit-def $sgpr0
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %val = call i1 @signext_i1_func_void()
  store volatile i1 %val, ptr addrspace(1) undef
  ret void
}

define inreg i1 @inreg_i1_func_void() {
; GFX9-LABEL: name: inreg_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[ANYEXT:%[0-9]+]]:_(s32) = G_ANYEXT [[LOAD]](s1)
; GFX9-NEXT:    $vgpr0 = COPY [[ANYEXT]](s32)
; GFX9-NEXT:    SI_RETURN implicit $vgpr0
;
; GFX11-LABEL: name: inreg_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[ANYEXT:%[0-9]+]]:_(s32) = G_ANYEXT [[LOAD]](s1)
; GFX11-NEXT:    $vgpr0 = COPY [[ANYEXT]](s32)
; GFX11-NEXT:    SI_RETURN implicit $vgpr0
  %val = load i1, ptr addrspace(1) undef
  ret i1 %val
}

define void @test_call_inreg_i1_func_void() {
; GFX9-LABEL: name: test_call_inreg_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @inreg_i1_func_void
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @inreg_i1_func_void, csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $vgpr0
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s32)
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_inreg_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @inreg_i1_func_void
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @inreg_i1_func_void, csr_amdgpu, implicit-def $vgpr0
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %val = call i1 @inreg_i1_func_void()
  store volatile i1 %val, ptr addrspace(1) undef
  ret void
}

define signext inreg i1 @signext_inreg_i1_func_void() {
; GFX9-LABEL: name: signext_inreg_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[EXT:%[0-9]+]]:_(s32) = G_SEXT [[LOAD]](s1)
; GFX9-NEXT:    $vgpr0 = COPY [[EXT]](s32)
; GFX9-NEXT:    SI_RETURN implicit $vgpr0
;
; GFX11-LABEL: name: signext_inreg_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[EXT:%[0-9]+]]:_(s32) = G_SEXT [[LOAD]](s1)
; GFX11-NEXT:    $vgpr0 = COPY [[EXT]](s32)
; GFX11-NEXT:    SI_RETURN implicit $vgpr0
  %val = load i1, ptr addrspace(1) undef
  ret i1 %val
}

define void @test_call_signext_inreg_i1_func_void() {
; GFX9-LABEL: name: test_call_signext_inreg_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @[[CALLEE:signext_inreg_i1_func_void]]
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @[[CALLEE]], csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $vgpr0
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX9-NEXT:    [[ASSERTEXT:%[0-9]+]]:_(s32) = G_ASSERT_SEXT [[COPY2]], 1
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[ASSERTEXT]](s32)
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_signext_inreg_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @[[CALLEE:signext_inreg_i1_func_void]]
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @[[CALLEE]], csr_amdgpu, implicit-def $vgpr0
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX11-NEXT:    [[ASSERTEXT:%[0-9]+]]:_(s32) = G_ASSERT_SEXT [[COPY]], 1
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[ASSERTEXT]](s32)
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %val = call i1 @signext_inreg_i1_func_void()
  store volatile i1 %val, ptr addrspace(1) undef
  ret void
}

define zeroext inreg i1 @zeroext_inreg_i1_func_void() {
; GFX9-LABEL: name: zeroext_inreg_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[EXT:%[0-9]+]]:_(s32) = G_ZEXT [[LOAD]](s1)
; GFX9-NEXT:    $vgpr0 = COPY [[EXT]](s32)
; GFX9-NEXT:    SI_RETURN implicit $vgpr0
;
; GFX11-LABEL: name: zeroext_inreg_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[EXT:%[0-9]+]]:_(s32) = G_ZEXT [[LOAD]](s1)
; GFX11-NEXT:    $vgpr0 = COPY [[EXT]](s32)
; GFX11-NEXT:    SI_RETURN implicit $vgpr0
  %val = load i1, ptr addrspace(1) undef
  ret i1 %val
}

define void @test_call_zeroext_inreg_i1_func_void() {
; GFX9-LABEL: name: test_call_zeroext_inreg_i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @[[CALLEE:zeroext_inreg_i1_func_void]]
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @[[CALLEE]], csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $vgpr0
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX9-NEXT:    [[ASSERTEXT:%[0-9]+]]:_(s32) = G_ASSERT_ZEXT [[COPY2]], 1
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[ASSERTEXT]](s32)
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_zeroext_inreg_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @[[CALLEE:zeroext_inreg_i1_func_void]]
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @[[CALLEE]], csr_amdgpu, implicit-def $vgpr0
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX11-NEXT:    [[ASSERTEXT:%[0-9]+]]:_(s32) = G_ASSERT_ZEXT [[COPY]], 1
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[ASSERTEXT]](s32)
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %val = call i1 @zeroext_inreg_i1_func_void()
  store volatile i1 %val, ptr addrspace(1) undef
  ret void
}

define [2 x i1] @a2i1_func_void() {
; GFX9-LABEL: name: a2i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX9-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX9-NEXT:    [[LOAD2:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD]](p1) :: (load (s1) from `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX9-NEXT:    $sgpr4_sgpr5 = COPY [[LOAD]](s1)
; GFX9-NEXT:    $sgpr6_sgpr7 = COPY [[LOAD2]](s1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: a2i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX11-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX11-NEXT:    [[LOAD2:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD]](p1) :: (load (s1) from `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX11-NEXT:    $sgpr0 = COPY [[LOAD]](s1)
; GFX11-NEXT:    $sgpr1 = COPY [[LOAD2]](s1)
; GFX11-NEXT:    SI_RETURN
  %val = load [2 x i1], ptr addrspace(1) undef
  ret [2 x i1] %val
}

define void @test_call_a2i1_func_void() {
; GFX9-LABEL: name: test_call_a2i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @a2i1_func_void
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @a2i1_func_void, csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $sgpr4_sgpr5, implicit-def $sgpr6_sgpr7
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY3:%[0-9]+]]:sreg_64(s1) = COPY $sgpr6_sgpr7
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX9-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX9-NEXT:    G_STORE [[COPY3]](s1), [[PTRADD]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_a2i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @a2i1_func_void
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @a2i1_func_void, csr_amdgpu, implicit-def $sgpr0, implicit-def $sgpr1
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:sreg_32(s1) = COPY $sgpr1
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX11-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX11-NEXT:    G_STORE [[COPY2]](s1), [[PTRADD]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %val = call [2 x i1] @a2i1_func_void()
  store volatile [2 x i1] %val, ptr addrspace(1) undef
  ret void
}

define [16 x i1] @a16i1_func_void(ptr addrspace(1) %in) {
; GFX9-LABEL: name: a16i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $vgpr0, $vgpr1, $vgpr2
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(p5) = COPY $vgpr0
; GFX9-NEXT:    [[COPY1:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $vgpr2
; GFX9-NEXT:    [[MERGE:%[0-9]+]]:_(p1) = G_MERGE_VALUES [[COPY1]](s32), [[COPY2]](s32)
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[MERGE]](p1) :: (load (s1) from %ir.in, addrspace 1)

; GFX9-NEXT:    [[CONST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX9-NEXT:    [[PTRADD1:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST1]](s64)
; GFX9-NEXT:    [[LOAD1:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD1]](p1) :: (load (s1) from %ir.in + 1, addrspace 1)
; GFX9-NEXT:    [[CONST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 2
; GFX9-NEXT:    [[PTRADD2:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST2]](s64)
; GFX9-NEXT:    [[LOAD2:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD2]](p1) :: (load (s1) from %ir.in + 2, addrspace 1)
; GFX9-NEXT:    [[CONST3:%[0-9]+]]:_(s64) = G_CONSTANT i64 3
; GFX9-NEXT:    [[PTRADD3:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST3]](s64)
; GFX9-NEXT:    [[LOAD3:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD3]](p1) :: (load (s1) from %ir.in + 3, addrspace 1)
; GFX9-NEXT:    [[CONST4:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; GFX9-NEXT:    [[PTRADD4:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST4]](s64)
; GFX9-NEXT:    [[LOAD4:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD4]](p1) :: (load (s1) from %ir.in + 4, addrspace 1)
; GFX9-NEXT:    [[CONST5:%[0-9]+]]:_(s64) = G_CONSTANT i64 5
; GFX9-NEXT:    [[PTRADD5:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST5]](s64)
; GFX9-NEXT:    [[LOAD5:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD5]](p1) :: (load (s1) from %ir.in + 5, addrspace 1)
; GFX9-NEXT:    [[CONST6:%[0-9]+]]:_(s64) = G_CONSTANT i64 6
; GFX9-NEXT:    [[PTRADD6:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST6]](s64)
; GFX9-NEXT:    [[LOAD6:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD6]](p1) :: (load (s1) from %ir.in + 6, addrspace 1)
; GFX9-NEXT:    [[CONST7:%[0-9]+]]:_(s64) = G_CONSTANT i64 7
; GFX9-NEXT:    [[PTRADD7:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST7]](s64)
; GFX9-NEXT:    [[LOAD7:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD7]](p1) :: (load (s1) from %ir.in + 7, addrspace 1)
; GFX9-NEXT:    [[CONST8:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; GFX9-NEXT:    [[PTRADD8:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST8]](s64)
; GFX9-NEXT:    [[LOAD8:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD8]](p1) :: (load (s1) from %ir.in + 8, addrspace 1)
; GFX9-NEXT:    [[CONST9:%[0-9]+]]:_(s64) = G_CONSTANT i64 9
; GFX9-NEXT:    [[PTRADD9:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST9]](s64)
; GFX9-NEXT:    [[LOAD9:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD9]](p1) :: (load (s1) from %ir.in + 9, addrspace 1)
; GFX9-NEXT:    [[CONST10:%[0-9]+]]:_(s64) = G_CONSTANT i64 10
; GFX9-NEXT:    [[PTRADD10:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST10]](s64)
; GFX9-NEXT:    [[LOAD10:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD10]](p1) :: (load (s1) from %ir.in + 10, addrspace 1)
; GFX9-NEXT:    [[CONST11:%[0-9]+]]:_(s64) = G_CONSTANT i64 11
; GFX9-NEXT:    [[PTRADD11:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST11]](s64)
; GFX9-NEXT:    [[LOAD11:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD11]](p1) :: (load (s1) from %ir.in + 11, addrspace 1)
; GFX9-NEXT:    [[CONST12:%[0-9]+]]:_(s64) = G_CONSTANT i64 12
; GFX9-NEXT:    [[PTRADD12:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST12]](s64)
; GFX9-NEXT:    [[LOAD12:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD12]](p1) :: (load (s1) from %ir.in + 12, addrspace 1)
; GFX9-NEXT:    [[CONST13:%[0-9]+]]:_(s64) = G_CONSTANT i64 13
; GFX9-NEXT:    [[PTRADD13:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST13]](s64)
; GFX9-NEXT:    [[LOAD13:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD13]](p1) :: (load (s1) from %ir.in + 13, addrspace 1)
; GFX9-NEXT:    [[CONST14:%[0-9]+]]:_(s64) = G_CONSTANT i64 14
; GFX9-NEXT:    [[PTRADD14:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST14]](s64)
; GFX9-NEXT:    [[LOAD14:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD14]](p1) :: (load (s1) from %ir.in + 14, addrspace 1)
; GFX9-NEXT:    [[CONST15:%[0-9]+]]:_(s64) = G_CONSTANT i64 15
; GFX9-NEXT:    [[PTRADD15:%[0-9]+]]:_(p1) = G_PTR_ADD [[MERGE]], [[CONST15]](s64)
; GFX9-NEXT:    [[LOAD15:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD15]](p1) :: (load (s1) from %ir.in + 15, addrspace 1)

  %val = load [16 x i1], ptr addrspace(1) %in
  ret [16 x i1] %val
}

define void @test_call_a16i1_func_void(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; GFX9-LABEL: name: test_call_a16i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3
; GFX9-NEXT: {{  $}}
; GFX9:         [[FRAME:%[0-9]+]]:_(p5) = G_FRAME_INDEX %stack.0
; GFX9:         [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[FRAME]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; GFX9-NEXT:    [[PTRADD1:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST1]](s32)
; GFX9-NEXT:    [[LOAD1:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD1]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; GFX9-NEXT:    [[PTRADD2:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST2]](s32)
; GFX9-NEXT:    [[LOAD2:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD2]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
; GFX9-NEXT:    [[PTRADD3:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST3]](s32)
; GFX9-NEXT:    [[LOAD3:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD3]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST4:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
; GFX9-NEXT:    [[PTRADD4:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST4]](s32)
; GFX9-NEXT:    [[LOAD4:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD4]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST5:%[0-9]+]]:_(s32) = G_CONSTANT i32 5
; GFX9-NEXT:    [[PTRADD5:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST5]](s32)
; GFX9-NEXT:    [[LOAD5:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD5]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST6:%[0-9]+]]:_(s32) = G_CONSTANT i32 6
; GFX9-NEXT:    [[PTRADD6:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST6]](s32)
; GFX9-NEXT:    [[LOAD6:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD6]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST7:%[0-9]+]]:_(s32) = G_CONSTANT i32 7
; GFX9-NEXT:    [[PTRADD7:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST7]](s32)
; GFX9-NEXT:    [[LOAD7:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD7]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST8:%[0-9]+]]:_(s32) = G_CONSTANT i32 8
; GFX9-NEXT:    [[PTRADD8:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST8]](s32)
; GFX9-NEXT:    [[LOAD8:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD8]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST9:%[0-9]+]]:_(s32) = G_CONSTANT i32 9
; GFX9-NEXT:    [[PTRADD9:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST9]](s32)
; GFX9-NEXT:    [[LOAD9:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD9]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST10:%[0-9]+]]:_(s32) = G_CONSTANT i32 10
; GFX9-NEXT:    [[PTRADD10:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST10]](s32)
; GFX9-NEXT:    [[LOAD10:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD10]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST11:%[0-9]+]]:_(s32) = G_CONSTANT i32 11
; GFX9-NEXT:    [[PTRADD11:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST11]](s32)
; GFX9-NEXT:    [[LOAD11:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD11]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST12:%[0-9]+]]:_(s32) = G_CONSTANT i32 12
; GFX9-NEXT:    [[PTRADD12:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST12]](s32)
; GFX9-NEXT:    [[LOAD12:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD12]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST13:%[0-9]+]]:_(s32) = G_CONSTANT i32 13
; GFX9-NEXT:    [[PTRADD13:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST13]](s32)
; GFX9-NEXT:    [[LOAD13:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD13]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST14:%[0-9]+]]:_(s32) = G_CONSTANT i32 14
; GFX9-NEXT:    [[PTRADD14:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST14]](s32)
; GFX9-NEXT:    [[LOAD14:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD14]](p5) :: (load (s1) from %stack.0, addrspace 5)
; GFX9-NEXT:    [[CONST15:%[0-9]+]]:_(s32) = G_CONSTANT i32 15
; GFX9-NEXT:    [[PTRADD15:%[0-9]+]]:_(p5) = G_PTR_ADD [[FRAME]], [[CONST15]](s32)
; GFX9-NEXT:    [[LOAD15:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD15]](p5) :: (load (s1) from %stack.0, addrspace 5)

  %val = call [16 x i1] @a16i1_func_void(ptr addrspace(1) %in)
  store volatile [16 x i1] %val, ptr addrspace(1) %out
  ret void
}

define <2 x i1> @v2i1_func_void() {
; GFX9-LABEL: name: v2i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(<2 x s1>) = G_LOAD [[DEF]](p1) :: (load (<2 x s1>) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[UNMERGE:%[0-9]+]]:_(s1), [[UNMERGE1:%[0-9]+]]:_(s1) = G_UNMERGE_VALUES [[LOAD]](<2 x s1>)
; GFX9-NEXT:    [[EXT:%[0-9]+]]:_(s16) = G_ANYEXT [[UNMERGE]](s1)
; GFX9-NEXT:    [[EXT1:%[0-9]+]]:_(s16) = G_ANYEXT [[UNMERGE1]](s1)
; GFX9-NEXT:    [[EXT2:%[0-9]+]]:_(s32) = G_ANYEXT [[EXT]](s16)
; GFX9-NEXT:    $vgpr0 = COPY [[EXT2]](s32)
; GFX9-NEXT:    [[EXT3:%[0-9]+]]:_(s32) = G_ANYEXT [[EXT1]](s16)
; GFX9-NEXT:    $vgpr1 = COPY [[EXT3]](s32)
; GFX9-NEXT:    SI_RETURN implicit $vgpr0, implicit $vgpr1
;
; GFX11-LABEL: name: v2i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(<2 x s1>) = G_LOAD [[DEF]](p1) :: (load (<2 x s1>) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[UNMERGE:%[0-9]+]]:_(s1), [[UNMERGE1:%[0-9]+]]:_(s1) = G_UNMERGE_VALUES [[LOAD]](<2 x s1>)
; GFX11-NEXT:    [[EXT:%[0-9]+]]:_(s16) = G_ANYEXT [[UNMERGE]](s1)
; GFX11-NEXT:    [[EXT1:%[0-9]+]]:_(s16) = G_ANYEXT [[UNMERGE1]](s1)
; GFX11-NEXT:    [[EXT2:%[0-9]+]]:_(s32) = G_ANYEXT [[EXT]](s16)
; GFX11-NEXT:    $vgpr0 = COPY [[EXT2]](s32)
; GFX11-NEXT:    [[EXT3:%[0-9]+]]:_(s32) = G_ANYEXT [[EXT1]](s16)
; GFX11-NEXT:    $vgpr1 = COPY [[EXT3]](s32)
; GFX11-NEXT:    SI_RETURN implicit $vgpr0, implicit $vgpr1
  %val = load <2 x i1>, ptr addrspace(1) undef
  ret <2 x i1> %val
}

define void @test_call_v2i1_func_void() {
; GFX9-LABEL: name: test_call_v2i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @[[CALLEE:v2i1_func_void]]
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @[[CALLEE]], csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $vgpr0, implicit-def $vgpr1
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX9-NEXT:    [[TRUNC2:%[0-9]+]]:_(s16) = G_TRUNC [[COPY2]](s32)
; GFX9-NEXT:    [[COPY3:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX9-NEXT:    [[TRUNC3:%[0-9]+]]:_(s16) = G_TRUNC [[COPY3]](s32)
; GFX9-NEXT:    [[BUILDVEC:%[0-9]+]]:_(<2 x s16>) = G_BUILD_VECTOR [[TRUNC2]](s16), [[TRUNC3]](s16)
; GFX9-NEXT:    [[TRUNC4:%[0-9]+]]:_(<2 x s1>) = G_TRUNC [[BUILDVEC]](<2 x s16>)
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[TRUNC4]](<2 x s1>), [[DEF]](p1) :: (volatile store (<2 x s1>) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_v2i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @[[CALLEE:v2i1_func_void]]
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @[[CALLEE]], csr_amdgpu, implicit-def $vgpr0, implicit-def $vgpr1
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX11-NEXT:    [[TRUNC2:%[0-9]+]]:_(s16) = G_TRUNC [[COPY2]](s32)
; GFX11-NEXT:    [[COPY3:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX11-NEXT:    [[TRUNC3:%[0-9]+]]:_(s16) = G_TRUNC [[COPY3]](s32)
; GFX11-NEXT:    [[BUILDVEC:%[0-9]+]]:_(<2 x s16>) = G_BUILD_VECTOR [[TRUNC2]](s16), [[TRUNC3]](s16)
; GFX11-NEXT:    [[TRUNC4:%[0-9]+]]:_(<2 x s1>) = G_TRUNC [[BUILDVEC]](<2 x s16>)
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[TRUNC4]](<2 x s1>), [[DEF]](p1) :: (volatile store (<2 x s1>) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN

  %val = call <2 x i1> @v2i1_func_void()
  store volatile <2 x i1> %val, ptr addrspace(1) undef
  ret void
}
