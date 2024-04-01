; RUN: llc -global-isel -stop-after=irtranslator -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GFX9 -enable-var-scope %s
; RUN: llc -global-isel -stop-after=irtranslator -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GFX11 -enable-var-scope %s

define i1 @i1_func_void() {
; GFX9-LABEL: name: i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    $sgpr0_sgpr1 = COPY [[LOAD]](s1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[ANYEXT:%[0-9]+]]:_(s32) = G_ANYEXT [[LOAD]](s1)
; GFX11-NEXT:    [[INTRIN:%[0-9]+]]:_(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane), [[ANYEXT]](s32)
; GFX11-NEXT:    $sgpr0 = COPY [[INTRIN]](s32)
; GFX11-NEXT:    SI_RETURN implicit $sgpr0
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
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @i1_func_void, csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $sgpr0_sgpr1
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s64) = COPY $sgpr0_sgpr1
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s64)
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @i1_func_void
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @i1_func_void, csr_amdgpu, implicit-def $sgpr0
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr0
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
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
; GFX9-NEXT:    $sgpr0_sgpr1 = COPY [[LOAD]](s1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: zeroext_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[ANYEXT:%[0-9]+]]:_(s32) = G_ANYEXT [[LOAD]](s1)
; GFX11-NEXT:    [[INTRIN:%[0-9]+]]:_(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane), [[ANYEXT]](s32)
; GFX11-NEXT:    $sgpr0 = COPY [[INTRIN]](s32)
; GFX11-NEXT:    SI_RETURN implicit $sgpr0
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
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @zeroext_i1_func_void, csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $sgpr0_sgpr1
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s64) = COPY $sgpr0_sgpr1
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s64)
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_zeroext_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @zeroext_i1_func_void
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @zeroext_i1_func_void, csr_amdgpu, implicit-def $sgpr0
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr0
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
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
; GFX9-NEXT:    $sgpr0_sgpr1 = COPY [[LOAD]](s1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: signext_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[ANYEXT:%[0-9]+]]:_(s32) = G_ANYEXT [[LOAD]](s1)
; GFX11-NEXT:    [[INTRIN:%[0-9]+]]:_(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane), [[ANYEXT]](s32)
; GFX11-NEXT:    $sgpr0 = COPY [[INTRIN]](s32)
; GFX11-NEXT:    SI_RETURN implicit $sgpr0
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
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @signext_i1_func_void, csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $sgpr0_sgpr1
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s64) = COPY $sgpr0_sgpr1
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s64)
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_signext_i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @signext_i1_func_void
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @signext_i1_func_void, csr_amdgpu, implicit-def $sgpr0
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr0
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
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

define [2 x i1] @a2i1_func_void() {
; GFX9-LABEL: name: a2i1_func_void
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX9-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX9-NEXT:    [[LOAD2:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD]](p1) :: (load (s1) from `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX9-NEXT:    $sgpr0_sgpr1 = COPY [[LOAD]](s1)
; GFX9-NEXT:    $sgpr2_sgpr3 = COPY [[LOAD2]](s1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: a2i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX11-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX11-NEXT:    [[LOAD2:%[0-9]+]]:_(s1) = G_LOAD [[PTRADD]](p1) :: (load (s1) from `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX11-NEXT:    [[ANYEXT:%[0-9]+]]:_(s32) = G_ANYEXT [[LOAD]](s1)
; GFX11-NEXT:    [[INTRIN:%[0-9]+]]:_(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane), [[ANYEXT]](s32)
; GFX11-NEXT:    $sgpr0 = COPY [[INTRIN]](s32)
; GFX11-NEXT:    [[ANYEXT3:%[0-9]+]]:_(s32) = G_ANYEXT [[LOAD2]](s1)
; GFX11-NEXT:    [[INTRIN2:%[0-9]+]]:_(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane), [[ANYEXT3]](s32)
; GFX11-NEXT:    $sgpr1 = COPY [[INTRIN2]](s32)
; GFX11-NEXT:    SI_RETURN implicit $sgpr0, implicit $sgpr1
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
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @a2i1_func_void, csr_amdgpu, implicit $sgpr0_sgpr1_sgpr2_sgpr3, implicit-def $sgpr0_sgpr1, implicit-def $sgpr2_sgpr3
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s64) = COPY $sgpr0_sgpr1
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s64)
; GFX9-NEXT:    [[COPY3:%[0-9]+]]:_(s64) = COPY $sgpr2_sgpr3
; GFX9-NEXT:    [[TRUNC2:%[0-9]+]]:_(s1) = G_TRUNC [[COPY3]](s64)
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX9-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX9-NEXT:    G_STORE [[TRUNC2]](s1), [[PTRADD]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_a2i1_func_void
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @a2i1_func_void
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @a2i1_func_void, csr_amdgpu, implicit-def $sgpr0, implicit-def $sgpr1
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr0
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $sgpr1
; GFX11-NEXT:    [[TRUNC2:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s32)
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX11-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX11-NEXT:    G_STORE [[TRUNC2]](s1), [[PTRADD]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %val = call [2 x i1] @a2i1_func_void()
  store volatile [2 x i1] %val, ptr addrspace(1) undef
  ret void
}

