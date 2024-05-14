; RUN: llc -global-isel -stop-after=irtranslator -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GFX9 -enable-var-scope %s
; RUN: llc -global-isel -stop-after=irtranslator -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GFX11 -enable-var-scope %s

define void @void_func_i1(i1 %arg0) {
; GFX9-LABEL: name: void_func_i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:   liveins: $sgpr4_sgpr5
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:   liveins: $sgpr0
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store i1 %arg0, ptr addrspace(1) undef
  ret void
}

define void @test_call_void_func_i1() {
; GFX9-LABEL: name: test_call_void_func_i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_i1  
; GFX9-NEXT:    $sgpr4_sgpr5 = COPY [[LOAD]](s1)
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_i1, csr_amdgpu, implicit $sgpr4_sgpr5, implicit $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_void_func_i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_i1  
; GFX11-NEXT:    $sgpr0 = COPY [[LOAD]](s1)
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_i1, csr_amdgpu, implicit $sgpr0
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    SI_RETURN
  %val = load i1, ptr addrspace(1) undef
  call void @void_func_i1(i1 %val)
  ret void
}

define void @void_func_i1_zeroext(i1 zeroext %arg0) {
; GFX9-LABEL: name: void_func_i1_zeroext
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr4_sgpr5
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[CONST:%[0-9]+]]:_(s32) = G_CONSTANT i32 12
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[ZEXT:%[0-9]+]]:_(s32) = G_ZEXT [[COPY]](s1)
; GFX9-NEXT:    [[ADD:%[0-9]+]]:_(s32) = G_ADD [[ZEXT]], [[CONST]]
; GFX9-NEXT:    G_STORE [[ADD]](s32), [[DEF]](p1) :: (store (s32) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_i1_zeroext
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    [[CONST:%[0-9]+]]:_(s32) = G_CONSTANT i32 12
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[ZEXT:%[0-9]+]]:_(s32) = G_ZEXT [[COPY]](s1)
; GFX11-NEXT:    [[ADD:%[0-9]+]]:_(s32) = G_ADD [[ZEXT]], [[CONST]]
; GFX11-NEXT:    G_STORE [[ADD]](s32), [[DEF]](p1) :: (store (s32) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %ext = zext i1 %arg0 to i32
  %add = add i32 %ext, 12
  store i32 %add, ptr addrspace(1) undef
  ret void
}

define void @test_call_void_func_i1_zeroext() {
; GFX9-LABEL: name: test_call_void_func_i1_zeroext
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_i1_zeroext 
; GFX9-NEXT:    $sgpr4_sgpr5 = COPY [[LOAD]](s1)
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_i1_zeroext, csr_amdgpu, implicit $sgpr4_sgpr5, implicit $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_void_func_i1_zeroext
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_i1_zeroext 
; GFX11-NEXT:    $sgpr0 = COPY [[LOAD]](s1)
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_i1_zeroext, csr_amdgpu, implicit $sgpr0
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    SI_RETURN
  %val = load i1, ptr addrspace(1) undef
  call void @void_func_i1_zeroext(i1 %val)
  ret void
}

define void @void_func_i1_signext(i1 signext %arg0) {
; GFX9-LABEL: name: void_func_i1_signext
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr4_sgpr5
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[CONST:%[0-9]+]]:_(s32) = G_CONSTANT i32 12
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    [[SEXT:%[0-9]+]]:_(s32) = G_SEXT [[COPY]](s1)
; GFX9-NEXT:    [[ADD:%[0-9]+]]:_(s32) = G_ADD [[SEXT]], [[CONST]]
; GFX9-NEXT:    G_STORE [[ADD]](s32), [[DEF]](p1) :: (store (s32) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_i1_signext
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    [[CONST:%[0-9]+]]:_(s32) = G_CONSTANT i32 12
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:    [[SEXT:%[0-9]+]]:_(s32) = G_SEXT [[COPY]](s1)
; GFX11-NEXT:    [[ADD:%[0-9]+]]:_(s32) = G_ADD [[SEXT]], [[CONST]]
; GFX11-NEXT:    G_STORE [[ADD]](s32), [[DEF]](p1) :: (store (s32) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  %ext = sext i1 %arg0 to i32
  %add = add i32 %ext, 12
  store i32 %add, ptr addrspace(1) undef
  ret void
}

define void @test_call_void_func_i1_signext() {
; GFX9-LABEL: name: test_call_void_func_i1_signext
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_i1_signext 
; GFX9-NEXT:    $sgpr4_sgpr5 = COPY [[LOAD]](s1)
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_i1_signext, csr_amdgpu, implicit $sgpr4_sgpr5, implicit $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_void_func_i1_signext
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_i1_signext 
; GFX11-NEXT:    $sgpr0 = COPY [[LOAD]](s1)
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_i1_signext, csr_amdgpu, implicit $sgpr0
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    SI_RETURN
  %val = load i1, ptr addrspace(1) undef
  call void @void_func_i1_signext(i1 %val)
  ret void
}

define void @void_func_a2i1([2 x i1] %arg0) {
; GFX9-LABEL: name: void_func_a2i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr4_sgpr5, $sgpr6_sgpr7
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr6_sgpr7
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX9-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX9-NEXT:    G_STORE [[COPY2]](s1), [[PTRADD]](p1) :: (store (s1) into `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_a2i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0, $sgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:sreg_32(s1) = COPY $sgpr1
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    [[CONST:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX11-NEXT:    [[PTRADD:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST]](s64)
; GFX11-NEXT:    G_STORE [[COPY2]](s1), [[PTRADD]](p1) :: (store (s1) into `ptr addrspace(1) undef` + 1, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store [2 x i1] %arg0, ptr addrspace(1) undef
  ret void
}

define void @test_call_void_func_a2i1() {
; GFX9-LABEL: name: test_call_void_func_a2i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[CONST1:%[0-9]+]]:_(s1) = G_CONSTANT i1 false  
; GFX9-NEXT:    [[CONST2:%[0-9]+]]:_(s1) = G_CONSTANT i1 true  
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_a2i1 
; GFX9-NEXT:    $sgpr4_sgpr5 = COPY [[CONST1]](s1)
; GFX9-NEXT:    $sgpr6_sgpr7 = COPY [[CONST2]](s1)
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_a2i1, csr_amdgpu, implicit $sgpr4_sgpr5, implicit $sgpr6_sgpr7, implicit $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_void_func_a2i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[CONST1:%[0-9]+]]:_(s1) = G_CONSTANT i1 false  
; GFX11-NEXT:    [[CONST2:%[0-9]+]]:_(s1) = G_CONSTANT i1 true  
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_a2i1 
; GFX11-NEXT:    $sgpr0 = COPY [[CONST1]](s1)
; GFX11-NEXT:    $sgpr1 = COPY [[CONST2]](s1)
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_a2i1, csr_amdgpu, implicit $sgpr0, implicit $sgpr1
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    SI_RETURN
  %1 = insertvalue [2 x i1] undef, i1 0, 0
  %2 = insertvalue [2 x i1] %1, i1 1, 1
  call void @void_func_a2i1([2 x i1] %2)
  ret void
}

define void @void_func_v2i1(<2 x i1> %arg0) {
; GFX9-LABEL: name: void_func_v2i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $vgpr0, $vgpr1
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[COPY]](s32)
; GFX9-NEXT:    [[COPY1:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX9-NEXT:    [[TRUNC1:%[0-9]+]]:_(s16) = G_TRUNC [[COPY1]](s32)
; GFX9-NEXT:    [[BUILDVEC:%[0-9]+]]:_(<2 x s16>) = G_BUILD_VECTOR [[TRUNC]](s16), [[TRUNC1]](s16)
; GFX9-NEXT:    [[TRUNC2:%[0-9]+]]:_(<2 x s1>) = G_TRUNC [[BUILDVEC]](<2 x s16>)
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[TRUNC2]](<2 x s1>), [[DEF]](p1) :: (store (<2 x s1>) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_v2i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $vgpr0, $vgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[COPY]](s32)
; GFX11-NEXT:    [[COPY1:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX11-NEXT:    [[TRUNC1:%[0-9]+]]:_(s16) = G_TRUNC [[COPY1]](s32)
; GFX11-NEXT:    [[BUILDVEC:%[0-9]+]]:_(<2 x s16>) = G_BUILD_VECTOR [[TRUNC]](s16), [[TRUNC1]](s16)
; GFX11-NEXT:    [[TRUNC2:%[0-9]+]]:_(<2 x s1>) = G_TRUNC [[BUILDVEC]](<2 x s16>)
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[TRUNC2]](<2 x s1>), [[DEF]](p1) :: (store (<2 x s1>) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store <2 x i1> %arg0, ptr addrspace(1) undef
  ret void
}

define void @test_call_void_func_v2i1(ptr addrspace(1) %in) {
; GFX9-LABEL: name: test_call_void_func_v2i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $vgpr0, $vgpr1
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX9-NEXT:    [[COPY1:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX9-NEXT:    [[MERGE:%[0-9]+]]:_(p1) = G_MERGE_VALUES [[COPY]](s32), [[COPY1]](s32)
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(<2 x s1>) = G_LOAD [[MERGE]](p1) :: (load (<2 x s1>) from %ir.in, addrspace 1)
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @[[CALLEE:void_func_v2i1]]
; GFX9-NEXT:    [[UNMERGE:%[0-9]+]]:_(s1), [[UNMERGE1:%[0-9]+]]:_(s1) = G_UNMERGE_VALUES [[LOAD]](<2 x s1>)
; GFX9-NEXT:    [[EXT:%[0-9]+]]:_(s16) = G_ANYEXT [[UNMERGE]](s1)
; GFX9-NEXT:    [[EXT1:%[0-9]+]]:_(s16) = G_ANYEXT [[UNMERGE1]](s1)
; GFX9-NEXT:    [[EXT2:%[0-9]+]]:_(s32) = G_ANYEXT [[EXT]](s16)
; GFX9-NEXT:    $vgpr0 = COPY [[EXT2]](s32)
; GFX9-NEXT:    [[EXT3:%[0-9]+]]:_(s32) = G_ANYEXT [[EXT1]](s16)
; GFX9-NEXT:    $vgpr1 = COPY [[EXT3]](s32)
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY2]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @[[CALLEE]], csr_amdgpu, implicit $vgpr0, implicit $vgpr1, implicit $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_void_func_v2i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $vgpr0, $vgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX11-NEXT:    [[COPY1:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX11-NEXT:    [[MERGE:%[0-9]+]]:_(p1) = G_MERGE_VALUES [[COPY]](s32), [[COPY1]](s32)
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(<2 x s1>) = G_LOAD [[MERGE]](p1) :: (load (<2 x s1>) from %ir.in, addrspace 1)
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @[[CALLEE:void_func_v2i1]]
; GFX11-NEXT:    [[UNMERGE:%[0-9]+]]:_(s1), [[UNMERGE1:%[0-9]+]]:_(s1) = G_UNMERGE_VALUES [[LOAD]](<2 x s1>)
; GFX11-NEXT:    [[EXT:%[0-9]+]]:_(s16) = G_ANYEXT [[UNMERGE]](s1)
; GFX11-NEXT:    [[EXT1:%[0-9]+]]:_(s16) = G_ANYEXT [[UNMERGE1]](s1)
; GFX11-NEXT:    [[EXT2:%[0-9]+]]:_(s32) = G_ANYEXT [[EXT]](s16)
; GFX11-NEXT:    $vgpr0 = COPY [[EXT2]](s32)
; GFX11-NEXT:    [[EXT3:%[0-9]+]]:_(s32) = G_ANYEXT [[EXT1]](s16)
; GFX11-NEXT:    $vgpr1 = COPY [[EXT3]](s32)
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @[[CALLEE]], csr_amdgpu, implicit $vgpr0, implicit $vgpr1
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    SI_RETURN
  %a = load <2 x i1>, ptr addrspace(1) %in
  call void @void_func_v2i1(<2 x i1> %a)
  ret void
}

define void @void_func_i1_i1(i1 %arg0, i1 %arg1) {
; GFX9-LABEL: name: void_func_i1_i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr4_sgpr5, $sgpr6_sgpr7
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr6_sgpr7
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_i1_i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0, $sgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:sreg_32(s1) = COPY $sgpr1
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store volatile i1 %arg0, ptr addrspace(1) undef
  store volatile i1 %arg1, ptr addrspace(1) undef
  ret void
}

define void @test_call_void_func_i1_i1() {
; GFX9-LABEL: name: test_call_void_func_i1_i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    [[CONST:%[0-9]+]]:_(s1) = G_CONSTANT i1 true  
; GFX9-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX9-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_i1_i1
; GFX9-NEXT:    $sgpr4_sgpr5 = COPY [[LOAD]](s1)
; GFX9-NEXT:    $sgpr6_sgpr7 = COPY [[CONST]](s1)
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(<4 x s32>) = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    $sgpr0_sgpr1_sgpr2_sgpr3 = COPY [[COPY]](<4 x s32>)
; GFX9-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_i1_i1, csr_amdgpu, implicit $sgpr4_sgpr5, implicit $sgpr6_sgpr7, implicit $sgpr0_sgpr1_sgpr2_sgpr3
; GFX9-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: test_call_void_func_i1_i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    [[CONST:%[0-9]+]]:_(s1) = G_CONSTANT i1 true  
; GFX11-NEXT:    [[LOAD:%[0-9]+]]:_(s1) = G_LOAD [[DEF]](p1) :: (load (s1) from `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    ADJCALLSTACKUP 0, 0, implicit-def $scc
; GFX11-NEXT:    [[GLOBAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @void_func_i1_i1
; GFX11-NEXT:    $sgpr0 = COPY [[LOAD]](s1)
; GFX11-NEXT:    $sgpr1 = COPY [[CONST]](s1)
; GFX11-NEXT:    $sgpr30_sgpr31 = noconvergent G_SI_CALL [[GLOBAL]](p0), @void_func_i1_i1, csr_amdgpu, implicit $sgpr0, implicit $sgpr1
; GFX11-NEXT:    ADJCALLSTACKDOWN 0, 0, implicit-def $scc
; GFX11-NEXT:    SI_RETURN
  %val = load i1, ptr addrspace(1) undef
  call void @void_func_i1_i1(i1 %val, i1 true)
  ret void
}

define void @exhaust_sgprs_by_i1_args(
  i1 %arg0, i1 %arg1, i1 %arg2, i1 %arg3, i1 %arg4, i1 %arg5, i1 %arg6, i1 %arg7,
  i1 %arg8, i1 %arg9, i1 %arg10, i1 %arg11, i1 %arg12, i1 %arg13, i1 %arg14, i1 %arg15,
  i1 %arg16, i1 %arg17, i1 %arg18, i1 %arg19, i1 %arg20, i1 %arg21, i1 %arg22, i1 %arg23,
  i1 %arg24, i1 %arg25, i1 %arg26, i1 %arg27, i1 %arg28, i1 %arg29, i1 %arg30, i1 %arg31) {
; GFX9-LABEL: name: exhaust_sgprs_by_i1_args
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10, $vgpr11, $vgpr12, $vgpr13, $vgpr14, $vgpr15, $vgpr16, $vgpr17, $vgpr18, $sgpr4_sgpr5, $sgpr6_sgpr7, $sgpr8_sgpr9, $sgpr10_sgpr11, $sgpr12_sgpr13, $sgpr14_sgpr15, $sgpr16_sgpr17, $sgpr18_sgpr19, $sgpr20_sgpr21, $sgpr22_sgpr23, $sgpr24_sgpr25, $sgpr26_sgpr27, $sgpr28_sgpr29
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY1:%[0-9]+]]:sreg_64(s1) = COPY $sgpr6_sgpr7
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr8_sgpr9
; GFX9-NEXT:    [[COPY3:%[0-9]+]]:sreg_64(s1) = COPY $sgpr10_sgpr11
; GFX9-NEXT:    [[COPY4:%[0-9]+]]:sreg_64(s1) = COPY $sgpr12_sgpr13
; GFX9-NEXT:    [[COPY5:%[0-9]+]]:sreg_64(s1) = COPY $sgpr14_sgpr15
; GFX9-NEXT:    [[COPY6:%[0-9]+]]:sreg_64(s1) = COPY $sgpr16_sgpr17
; GFX9-NEXT:    [[COPY7:%[0-9]+]]:sreg_64(s1) = COPY $sgpr18_sgpr19
; GFX9-NEXT:    [[COPY8:%[0-9]+]]:sreg_64(s1) = COPY $sgpr20_sgpr21
; GFX9-NEXT:    [[COPY9:%[0-9]+]]:sreg_64(s1) = COPY $sgpr22_sgpr23
; GFX9-NEXT:    [[COPY10:%[0-9]+]]:sreg_64(s1) = COPY $sgpr24_sgpr25
; GFX9-NEXT:    [[COPY11:%[0-9]+]]:sreg_64(s1) = COPY $sgpr26_sgpr27
; GFX9-NEXT:    [[COPY12:%[0-9]+]]:sreg_64(s1) = COPY $sgpr28_sgpr29
; GFX9-NEXT:    [[COPY13:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX9-NEXT:    [[TRUNC13:%[0-9]+]]:_(s1) = G_TRUNC [[COPY13]](s32)
; GFX9-NEXT:    [[COPY14:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX9-NEXT:    [[TRUNC14:%[0-9]+]]:_(s1) = G_TRUNC [[COPY14]](s32)
; GFX9-NEXT:    [[COPY15:%[0-9]+]]:_(s32) = COPY $vgpr2
; GFX9-NEXT:    [[TRUNC15:%[0-9]+]]:_(s1) = G_TRUNC [[COPY15]](s32)
; GFX9-NEXT:    [[COPY16:%[0-9]+]]:_(s32) = COPY $vgpr3
; GFX9-NEXT:    [[TRUNC16:%[0-9]+]]:_(s1) = G_TRUNC [[COPY16]](s32)
; GFX9-NEXT:    [[COPY17:%[0-9]+]]:_(s32) = COPY $vgpr4
; GFX9-NEXT:    [[TRUNC17:%[0-9]+]]:_(s1) = G_TRUNC [[COPY17]](s32)
; GFX9-NEXT:    [[COPY18:%[0-9]+]]:_(s32) = COPY $vgpr5
; GFX9-NEXT:    [[TRUNC18:%[0-9]+]]:_(s1) = G_TRUNC [[COPY18]](s32)
; GFX9-NEXT:    [[COPY19:%[0-9]+]]:_(s32) = COPY $vgpr6
; GFX9-NEXT:    [[TRUNC19:%[0-9]+]]:_(s1) = G_TRUNC [[COPY19]](s32)
; GFX9-NEXT:    [[COPY20:%[0-9]+]]:_(s32) = COPY $vgpr7
; GFX9-NEXT:    [[TRUNC20:%[0-9]+]]:_(s1) = G_TRUNC [[COPY20]](s32)
; GFX9-NEXT:    [[COPY21:%[0-9]+]]:_(s32) = COPY $vgpr8
; GFX9-NEXT:    [[TRUNC21:%[0-9]+]]:_(s1) = G_TRUNC [[COPY21]](s32)
; GFX9-NEXT:    [[COPY22:%[0-9]+]]:_(s32) = COPY $vgpr9
; GFX9-NEXT:    [[TRUNC22:%[0-9]+]]:_(s1) = G_TRUNC [[COPY22]](s32)
; GFX9-NEXT:    [[COPY23:%[0-9]+]]:_(s32) = COPY $vgpr10
; GFX9-NEXT:    [[TRUNC23:%[0-9]+]]:_(s1) = G_TRUNC [[COPY23]](s32)
; GFX9-NEXT:    [[COPY24:%[0-9]+]]:_(s32) = COPY $vgpr11
; GFX9-NEXT:    [[TRUNC24:%[0-9]+]]:_(s1) = G_TRUNC [[COPY24]](s32)
; GFX9-NEXT:    [[COPY25:%[0-9]+]]:_(s32) = COPY $vgpr12
; GFX9-NEXT:    [[TRUNC25:%[0-9]+]]:_(s1) = G_TRUNC [[COPY25]](s32)
; GFX9-NEXT:    [[COPY26:%[0-9]+]]:_(s32) = COPY $vgpr13
; GFX9-NEXT:    [[TRUNC26:%[0-9]+]]:_(s1) = G_TRUNC [[COPY26]](s32)
; GFX9-NEXT:    [[COPY27:%[0-9]+]]:_(s32) = COPY $vgpr14
; GFX9-NEXT:    [[TRUNC27:%[0-9]+]]:_(s1) = G_TRUNC [[COPY27]](s32)
; GFX9-NEXT:    [[COPY28:%[0-9]+]]:_(s32) = COPY $vgpr15
; GFX9-NEXT:    [[TRUNC28:%[0-9]+]]:_(s1) = G_TRUNC [[COPY28]](s32)
; GFX9-NEXT:    [[COPY29:%[0-9]+]]:_(s32) = COPY $vgpr16
; GFX9-NEXT:    [[TRUNC29:%[0-9]+]]:_(s1) = G_TRUNC [[COPY29]](s32)
; GFX9-NEXT:    [[COPY30:%[0-9]+]]:_(s32) = COPY $vgpr17
; GFX9-NEXT:    [[TRUNC30:%[0-9]+]]:_(s1) = G_TRUNC [[COPY30]](s32)
; GFX9-NEXT:    [[COPY31:%[0-9]+]]:_(s32) = COPY $vgpr18
; GFX9-NEXT:    [[TRUNC31:%[0-9]+]]:_(s1) = G_TRUNC [[COPY31]](s32)
;
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; G_STOREs to COPY1-COPY11 omitted
; GFX9:         G_STORE [[COPY12]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    G_STORE [[TRUNC13]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; G_STOREs to TRUNC14-TRUNC30 omitted
; GFX9:         G_STORE [[TRUNC31]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
;
; GFX11-LABEL: name: exhaust_sgprs_by_i1_args
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT: liveins: $sgpr0, $sgpr1, $sgpr2, $sgpr3, $sgpr4, $sgpr5, $sgpr6, $sgpr7, $sgpr8, $sgpr9, $sgpr10, $sgpr11, $sgpr12, $sgpr13, $sgpr14, $sgpr15, $sgpr16, $sgpr17, $sgpr18, $sgpr19, $sgpr20, $sgpr21, $sgpr22, $sgpr23, $sgpr24, $sgpr25, $sgpr26, $sgpr27, $sgpr28, $sgpr29, $vgpr0, $vgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:   [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:   [[COPY1:%[0-9]+]]:sreg_32(s1) = COPY $sgpr1
; GFX11-NEXT:   [[COPY2:%[0-9]+]]:sreg_32(s1) = COPY $sgpr2
; GFX11-NEXT:   [[COPY3:%[0-9]+]]:sreg_32(s1) = COPY $sgpr3
; GFX11-NEXT:   [[COPY4:%[0-9]+]]:sreg_32(s1) = COPY $sgpr4
; GFX11-NEXT:   [[COPY5:%[0-9]+]]:sreg_32(s1) = COPY $sgpr5
; GFX11-NEXT:   [[COPY6:%[0-9]+]]:sreg_32(s1) = COPY $sgpr6
; GFX11-NEXT:   [[COPY7:%[0-9]+]]:sreg_32(s1) = COPY $sgpr7
; GFX11-NEXT:   [[COPY8:%[0-9]+]]:sreg_32(s1) = COPY $sgpr8
; GFX11-NEXT:   [[COPY9:%[0-9]+]]:sreg_32(s1) = COPY $sgpr9
; GFX11-NEXT:   [[COPY10:%[0-9]+]]:sreg_32(s1) = COPY $sgpr10
; GFX11-NEXT:   [[COPY11:%[0-9]+]]:sreg_32(s1) = COPY $sgpr11
; GFX11-NEXT:   [[COPY12:%[0-9]+]]:sreg_32(s1) = COPY $sgpr12
; GFX11-NEXT:   [[COPY13:%[0-9]+]]:sreg_32(s1) = COPY $sgpr13
; GFX11-NEXT:   [[COPY14:%[0-9]+]]:sreg_32(s1) = COPY $sgpr14
; GFX11-NEXT:   [[COPY15:%[0-9]+]]:sreg_32(s1) = COPY $sgpr15
; GFX11-NEXT:   [[COPY16:%[0-9]+]]:sreg_32(s1) = COPY $sgpr16
; GFX11-NEXT:   [[COPY17:%[0-9]+]]:sreg_32(s1) = COPY $sgpr17
; GFX11-NEXT:   [[COPY18:%[0-9]+]]:sreg_32(s1) = COPY $sgpr18
; GFX11-NEXT:   [[COPY19:%[0-9]+]]:sreg_32(s1) = COPY $sgpr19
; GFX11-NEXT:   [[COPY20:%[0-9]+]]:sreg_32(s1) = COPY $sgpr20
; GFX11-NEXT:   [[COPY21:%[0-9]+]]:sreg_32(s1) = COPY $sgpr21
; GFX11-NEXT:   [[COPY22:%[0-9]+]]:sreg_32(s1) = COPY $sgpr22
; GFX11-NEXT:   [[COPY23:%[0-9]+]]:sreg_32(s1) = COPY $sgpr23
; GFX11-NEXT:   [[COPY24:%[0-9]+]]:sreg_32(s1) = COPY $sgpr24
; GFX11-NEXT:   [[COPY25:%[0-9]+]]:sreg_32(s1) = COPY $sgpr25
; GFX11-NEXT:   [[COPY26:%[0-9]+]]:sreg_32(s1) = COPY $sgpr26
; GFX11-NEXT:   [[COPY27:%[0-9]+]]:sreg_32(s1) = COPY $sgpr27
; GFX11-NEXT:   [[COPY28:%[0-9]+]]:sreg_32(s1) = COPY $sgpr28
; GFX11-NEXT:   [[COPY29:%[0-9]+]]:sreg_32(s1) = COPY $sgpr29
; GFX11-NEXT:   [[COPY30:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX11-NEXT:   [[TRUNC30:%[0-9]+]]:_(s1) = G_TRUNC [[COPY30]](s32)
; GFX11-NEXT:   [[COPY31:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX11-NEXT:   [[TRUNC31:%[0-9]+]]:_(s1) = G_TRUNC [[COPY31]](s32)
;
; GFX11-NEXT:   [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX11-NEXT:   G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; G_STOREs to COPY1-COPY28 omitted
; GFX11:        G_STORE [[COPY29]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; G_STOREs to TRUNC1-TRUNC30 omitted
; GFX11-NEXT:   G_STORE [[TRUNC30]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11:        G_STORE [[TRUNC31]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
  store volatile i1 %arg0, ptr addrspace(1) undef
  store volatile i1 %arg1, ptr addrspace(1) undef
  store volatile i1 %arg2, ptr addrspace(1) undef
  store volatile i1 %arg3, ptr addrspace(1) undef
  store volatile i1 %arg4, ptr addrspace(1) undef
  store volatile i1 %arg5, ptr addrspace(1) undef
  store volatile i1 %arg6, ptr addrspace(1) undef
  store volatile i1 %arg7, ptr addrspace(1) undef

  store volatile i1 %arg8, ptr addrspace(1) undef
  store volatile i1 %arg9, ptr addrspace(1) undef
  store volatile i1 %arg10, ptr addrspace(1) undef
  store volatile i1 %arg11, ptr addrspace(1) undef
  store volatile i1 %arg12, ptr addrspace(1) undef
  store volatile i1 %arg13, ptr addrspace(1) undef
  store volatile i1 %arg14, ptr addrspace(1) undef
  store volatile i1 %arg15, ptr addrspace(1) undef

  store volatile i1 %arg16, ptr addrspace(1) undef
  store volatile i1 %arg17, ptr addrspace(1) undef
  store volatile i1 %arg18, ptr addrspace(1) undef
  store volatile i1 %arg19, ptr addrspace(1) undef
  store volatile i1 %arg20, ptr addrspace(1) undef
  store volatile i1 %arg21, ptr addrspace(1) undef
  store volatile i1 %arg22, ptr addrspace(1) undef
  store volatile i1 %arg23, ptr addrspace(1) undef

  store volatile i1 %arg24, ptr addrspace(1) undef
  store volatile i1 %arg25, ptr addrspace(1) undef
  store volatile i1 %arg26, ptr addrspace(1) undef
  store volatile i1 %arg27, ptr addrspace(1) undef
  store volatile i1 %arg28, ptr addrspace(1) undef
  store volatile i1 %arg29, ptr addrspace(1) undef
  store volatile i1 %arg30, ptr addrspace(1) undef
  store volatile i1 %arg31, ptr addrspace(1) undef

  ret void
}

define void @void_func_a48i1([48 x i1] %arg0) {
; GFX9-LABEL: name: void_func_a48i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10, $vgpr11, $vgpr12, $vgpr13, $vgpr14, $vgpr15, $vgpr16, $vgpr17, $vgpr18, $vgpr19, $vgpr20, $vgpr21, $vgpr22, $vgpr23, $vgpr24, $vgpr25, $vgpr26, $vgpr27, $vgpr28, $vgpr29, $vgpr30, $sgpr4_sgpr5, $sgpr6_sgpr7, $sgpr8_sgpr9, $sgpr10_sgpr11, $sgpr12_sgpr13, $sgpr14_sgpr15, $sgpr16_sgpr17, $sgpr18_sgpr19, $sgpr20_sgpr21, $sgpr22_sgpr23, $sgpr24_sgpr25, $sgpr26_sgpr27, $sgpr28_sgpr29
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY1:%[0-9]+]]:sreg_64(s1) = COPY $sgpr6_sgpr7
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr8_sgpr9
; GFX9-NEXT:    [[COPY3:%[0-9]+]]:sreg_64(s1) = COPY $sgpr10_sgpr11
; GFX9-NEXT:    [[COPY4:%[0-9]+]]:sreg_64(s1) = COPY $sgpr12_sgpr13
; GFX9-NEXT:    [[COPY5:%[0-9]+]]:sreg_64(s1) = COPY $sgpr14_sgpr15
; GFX9-NEXT:    [[COPY6:%[0-9]+]]:sreg_64(s1) = COPY $sgpr16_sgpr17
; GFX9-NEXT:    [[COPY7:%[0-9]+]]:sreg_64(s1) = COPY $sgpr18_sgpr19
; GFX9-NEXT:    [[COPY8:%[0-9]+]]:sreg_64(s1) = COPY $sgpr20_sgpr21
; GFX9-NEXT:    [[COPY9:%[0-9]+]]:sreg_64(s1) = COPY $sgpr22_sgpr23
; GFX9-NEXT:    [[COPY10:%[0-9]+]]:sreg_64(s1) = COPY $sgpr24_sgpr25
; GFX9-NEXT:    [[COPY11:%[0-9]+]]:sreg_64(s1) = COPY $sgpr26_sgpr27
; GFX9-NEXT:    [[COPY12:%[0-9]+]]:sreg_64(s1) = COPY $sgpr28_sgpr29
; GFX9-NEXT:    [[COPY13:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX9-NEXT:    [[TRUNC13:%[0-9]+]]:_(s1) = G_TRUNC [[COPY13]](s32)
; GFX9-NEXT:    [[COPY14:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX9-NEXT:    [[TRUNC14:%[0-9]+]]:_(s1) = G_TRUNC [[COPY14]](s32)
; GFX9-NEXT:    [[COPY15:%[0-9]+]]:_(s32) = COPY $vgpr2
; GFX9-NEXT:    [[TRUNC15:%[0-9]+]]:_(s1) = G_TRUNC [[COPY15]](s32)
; GFX9-NEXT:    [[COPY16:%[0-9]+]]:_(s32) = COPY $vgpr3
; GFX9-NEXT:    [[TRUNC16:%[0-9]+]]:_(s1) = G_TRUNC [[COPY16]](s32)
; GFX9-NEXT:    [[COPY17:%[0-9]+]]:_(s32) = COPY $vgpr4
; GFX9-NEXT:    [[TRUNC17:%[0-9]+]]:_(s1) = G_TRUNC [[COPY17]](s32)
; GFX9-NEXT:    [[COPY18:%[0-9]+]]:_(s32) = COPY $vgpr5
; GFX9-NEXT:    [[TRUNC18:%[0-9]+]]:_(s1) = G_TRUNC [[COPY18]](s32)
; GFX9-NEXT:    [[COPY19:%[0-9]+]]:_(s32) = COPY $vgpr6
; GFX9-NEXT:    [[TRUNC19:%[0-9]+]]:_(s1) = G_TRUNC [[COPY19]](s32)
; GFX9-NEXT:    [[COPY20:%[0-9]+]]:_(s32) = COPY $vgpr7
; GFX9-NEXT:    [[TRUNC20:%[0-9]+]]:_(s1) = G_TRUNC [[COPY20]](s32)
; GFX9-NEXT:    [[COPY21:%[0-9]+]]:_(s32) = COPY $vgpr8
; GFX9-NEXT:    [[TRUNC21:%[0-9]+]]:_(s1) = G_TRUNC [[COPY21]](s32)
; GFX9-NEXT:    [[COPY22:%[0-9]+]]:_(s32) = COPY $vgpr9
; GFX9-NEXT:    [[TRUNC22:%[0-9]+]]:_(s1) = G_TRUNC [[COPY22]](s32)
; GFX9-NEXT:    [[COPY23:%[0-9]+]]:_(s32) = COPY $vgpr10
; GFX9-NEXT:    [[TRUNC23:%[0-9]+]]:_(s1) = G_TRUNC [[COPY23]](s32)
; GFX9-NEXT:    [[COPY24:%[0-9]+]]:_(s32) = COPY $vgpr11
; GFX9-NEXT:    [[TRUNC24:%[0-9]+]]:_(s1) = G_TRUNC [[COPY24]](s32)
; GFX9-NEXT:    [[COPY25:%[0-9]+]]:_(s32) = COPY $vgpr12
; GFX9-NEXT:    [[TRUNC25:%[0-9]+]]:_(s1) = G_TRUNC [[COPY25]](s32)
; GFX9-NEXT:    [[COPY26:%[0-9]+]]:_(s32) = COPY $vgpr13
; GFX9-NEXT:    [[TRUNC26:%[0-9]+]]:_(s1) = G_TRUNC [[COPY26]](s32)
; GFX9-NEXT:    [[COPY27:%[0-9]+]]:_(s32) = COPY $vgpr14
; GFX9-NEXT:    [[TRUNC27:%[0-9]+]]:_(s1) = G_TRUNC [[COPY27]](s32)
; GFX9-NEXT:    [[COPY28:%[0-9]+]]:_(s32) = COPY $vgpr15
; GFX9-NEXT:    [[TRUNC28:%[0-9]+]]:_(s1) = G_TRUNC [[COPY28]](s32)
; GFX9-NEXT:    [[COPY29:%[0-9]+]]:_(s32) = COPY $vgpr16
; GFX9-NEXT:    [[TRUNC29:%[0-9]+]]:_(s1) = G_TRUNC [[COPY29]](s32)
; GFX9-NEXT:    [[COPY30:%[0-9]+]]:_(s32) = COPY $vgpr17
; GFX9-NEXT:    [[TRUNC30:%[0-9]+]]:_(s1) = G_TRUNC [[COPY30]](s32)
; GFX9-NEXT:    [[COPY31:%[0-9]+]]:_(s32) = COPY $vgpr18
; GFX9-NEXT:    [[TRUNC31:%[0-9]+]]:_(s1) = G_TRUNC [[COPY31]](s32)
; GFX9-NEXT:    [[COPY32:%[0-9]+]]:_(s32) = COPY $vgpr19
; GFX9-NEXT:    [[TRUNC32:%[0-9]+]]:_(s1) = G_TRUNC [[COPY32]](s32)
; GFX9-NEXT:    [[COPY33:%[0-9]+]]:_(s32) = COPY $vgpr20
; GFX9-NEXT:    [[TRUNC33:%[0-9]+]]:_(s1) = G_TRUNC [[COPY33]](s32)
; GFX9-NEXT:    [[COPY34:%[0-9]+]]:_(s32) = COPY $vgpr21
; GFX9-NEXT:    [[TRUNC34:%[0-9]+]]:_(s1) = G_TRUNC [[COPY34]](s32)
; GFX9-NEXT:    [[COPY35:%[0-9]+]]:_(s32) = COPY $vgpr22
; GFX9-NEXT:    [[TRUNC35:%[0-9]+]]:_(s1) = G_TRUNC [[COPY35]](s32)
; GFX9-NEXT:    [[COPY36:%[0-9]+]]:_(s32) = COPY $vgpr23
; GFX9-NEXT:    [[TRUNC36:%[0-9]+]]:_(s1) = G_TRUNC [[COPY36]](s32)
; GFX9-NEXT:    [[COPY37:%[0-9]+]]:_(s32) = COPY $vgpr24
; GFX9-NEXT:    [[TRUNC37:%[0-9]+]]:_(s1) = G_TRUNC [[COPY37]](s32)
; GFX9-NEXT:    [[COPY38:%[0-9]+]]:_(s32) = COPY $vgpr25
; GFX9-NEXT:    [[TRUNC38:%[0-9]+]]:_(s1) = G_TRUNC [[COPY38]](s32)
; GFX9-NEXT:    [[COPY39:%[0-9]+]]:_(s32) = COPY $vgpr26
; GFX9-NEXT:    [[TRUNC39:%[0-9]+]]:_(s1) = G_TRUNC [[COPY39]](s32)
; GFX9-NEXT:    [[COPY40:%[0-9]+]]:_(s32) = COPY $vgpr27
; GFX9-NEXT:    [[TRUNC40:%[0-9]+]]:_(s1) = G_TRUNC [[COPY40]](s32)
; GFX9-NEXT:    [[COPY41:%[0-9]+]]:_(s32) = COPY $vgpr28
; GFX9-NEXT:    [[TRUNC41:%[0-9]+]]:_(s1) = G_TRUNC [[COPY41]](s32)
; GFX9-NEXT:    [[COPY42:%[0-9]+]]:_(s32) = COPY $vgpr29
; GFX9-NEXT:    [[TRUNC42:%[0-9]+]]:_(s1) = G_TRUNC [[COPY42]](s32)
; GFX9-NEXT:    [[COPY43:%[0-9]+]]:_(s32) = COPY $vgpr30
; GFX9-NEXT:    [[TRUNC43:%[0-9]+]]:_(s1) = G_TRUNC [[COPY43]](s32)
; GFX9-NEXT:    [[FRAME1:%[0-9]+]]:_(p5) = G_FRAME_INDEX %fixed-stack.3
; GFX9-NEXT:    [[LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[FRAME1]](p5) :: (invariant load (s1) from %fixed-stack.3, align 16, addrspace 5)
; GFX9-NEXT:    [[TRUNC44:%[0-9]+]]:_(s1) = G_TRUNC [[LOAD1]](s32)
; GFX9-NEXT:    [[FRAME2:%[0-9]+]]:_(p5) = G_FRAME_INDEX %fixed-stack.2
; GFX9-NEXT:    [[LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[FRAME2]](p5) :: (invariant load (s1) from %fixed-stack.2, align 4, addrspace 5)
; GFX9-NEXT:    [[TRUNC45:%[0-9]+]]:_(s1) = G_TRUNC [[LOAD2]](s32)
; GFX9-NEXT:    [[FRAME3:%[0-9]+]]:_(p5) = G_FRAME_INDEX %fixed-stack.1
; GFX9-NEXT:    [[LOAD3:%[0-9]+]]:_(s32) = G_LOAD [[FRAME3]](p5) :: (invariant load (s1) from %fixed-stack.1, align 8, addrspace 5)
; GFX9-NEXT:    [[TRUNC46:%[0-9]+]]:_(s1) = G_TRUNC [[LOAD3]](s32)
; GFX9-NEXT:    [[FRAME4:%[0-9]+]]:_(p5) = G_FRAME_INDEX %fixed-stack.0
; GFX9-NEXT:    [[LOAD4:%[0-9]+]]:_(s32) = G_LOAD [[FRAME4]](p5) :: (invariant load (s1) from %fixed-stack.0, align 4, addrspace 5)
; GFX9-NEXT:    [[TRUNC47:%[0-9]+]]:_(s1) = G_TRUNC [[LOAD4]](s32)
;
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
; GFX9-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    [[CONST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX9-NEXT:    [[PTRADD1:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST1]](s64)
; GFX9-NEXT:    G_STORE [[COPY1]](s1), [[PTRADD1]](p1) :: (store (s1) into `ptr addrspace(1) undef` + 1, addrspace 1)
;
; G_STOREs to COPY2-COPY12, TRUNC13-TRUNC46 omitted
; GFX9:         [[CONST47:%[0-9]+]]:_(s64) = G_CONSTANT i64 47
; GFX9-NEXT:    [[PTRADD47:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST47]](s64)
; GFX9-NEXT:    G_STORE [[TRUNC47]](s1), [[PTRADD47]](p1) :: (store (s1) into `ptr addrspace(1) undef` + 47, addrspace 1)

  store [48 x i1] %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_a64i1([64 x i1] %arg0) {
; GFX11-LABEL: name: void_func_a64i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT: liveins: $sgpr0, $sgpr1, $sgpr2, $sgpr3, $sgpr4, $sgpr5, $sgpr6, $sgpr7, $sgpr8, $sgpr9, $sgpr10, $sgpr11, $sgpr12, $sgpr13, $sgpr14, $sgpr15, $sgpr16, $sgpr17, $sgpr18, $sgpr19, $sgpr20, $sgpr21, $sgpr22, $sgpr23, $sgpr24, $sgpr25, $sgpr26, $sgpr27, $sgpr28, $sgpr29, $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10, $vgpr11, $vgpr12, $vgpr13, $vgpr14, $vgpr15, $vgpr16, $vgpr17, $vgpr18, $vgpr19, $vgpr20, $vgpr21, $vgpr22, $vgpr23, $vgpr24, $vgpr25, $vgpr26, $vgpr27, $vgpr28, $vgpr29, $vgpr30
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:   [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:   [[COPY1:%[0-9]+]]:sreg_32(s1) = COPY $sgpr1
; GFX11-NEXT:   [[COPY2:%[0-9]+]]:sreg_32(s1) = COPY $sgpr2
; GFX11-NEXT:   [[COPY3:%[0-9]+]]:sreg_32(s1) = COPY $sgpr3
; GFX11-NEXT:   [[COPY4:%[0-9]+]]:sreg_32(s1) = COPY $sgpr4
; GFX11-NEXT:   [[COPY5:%[0-9]+]]:sreg_32(s1) = COPY $sgpr5
; GFX11-NEXT:   [[COPY6:%[0-9]+]]:sreg_32(s1) = COPY $sgpr6
; GFX11-NEXT:   [[COPY7:%[0-9]+]]:sreg_32(s1) = COPY $sgpr7
; GFX11-NEXT:   [[COPY8:%[0-9]+]]:sreg_32(s1) = COPY $sgpr8
; GFX11-NEXT:   [[COPY9:%[0-9]+]]:sreg_32(s1) = COPY $sgpr9
; GFX11-NEXT:   [[COPY10:%[0-9]+]]:sreg_32(s1) = COPY $sgpr10
; GFX11-NEXT:   [[COPY11:%[0-9]+]]:sreg_32(s1) = COPY $sgpr11
; GFX11-NEXT:   [[COPY12:%[0-9]+]]:sreg_32(s1) = COPY $sgpr12
; GFX11-NEXT:   [[COPY13:%[0-9]+]]:sreg_32(s1) = COPY $sgpr13
; GFX11-NEXT:   [[COPY14:%[0-9]+]]:sreg_32(s1) = COPY $sgpr14
; GFX11-NEXT:   [[COPY15:%[0-9]+]]:sreg_32(s1) = COPY $sgpr15
; GFX11-NEXT:   [[COPY16:%[0-9]+]]:sreg_32(s1) = COPY $sgpr16
; GFX11-NEXT:   [[COPY17:%[0-9]+]]:sreg_32(s1) = COPY $sgpr17
; GFX11-NEXT:   [[COPY18:%[0-9]+]]:sreg_32(s1) = COPY $sgpr18
; GFX11-NEXT:   [[COPY19:%[0-9]+]]:sreg_32(s1) = COPY $sgpr19
; GFX11-NEXT:   [[COPY20:%[0-9]+]]:sreg_32(s1) = COPY $sgpr20
; GFX11-NEXT:   [[COPY21:%[0-9]+]]:sreg_32(s1) = COPY $sgpr21
; GFX11-NEXT:   [[COPY22:%[0-9]+]]:sreg_32(s1) = COPY $sgpr22
; GFX11-NEXT:   [[COPY23:%[0-9]+]]:sreg_32(s1) = COPY $sgpr23
; GFX11-NEXT:   [[COPY24:%[0-9]+]]:sreg_32(s1) = COPY $sgpr24
; GFX11-NEXT:   [[COPY25:%[0-9]+]]:sreg_32(s1) = COPY $sgpr25
; GFX11-NEXT:   [[COPY26:%[0-9]+]]:sreg_32(s1) = COPY $sgpr26
; GFX11-NEXT:   [[COPY27:%[0-9]+]]:sreg_32(s1) = COPY $sgpr27
; GFX11-NEXT:   [[COPY28:%[0-9]+]]:sreg_32(s1) = COPY $sgpr28
; GFX11-NEXT:   [[COPY29:%[0-9]+]]:sreg_32(s1) = COPY $sgpr29
; GFX11-NEXT:   [[COPY30:%[0-9]+]]:_(s32) = COPY $vgpr0
; GFX11-NEXT:   [[TRUNC30:%[0-9]+]]:_(s1) = G_TRUNC [[COPY30]](s32)
; GFX11-NEXT:   [[COPY31:%[0-9]+]]:_(s32) = COPY $vgpr1
; GFX11-NEXT:   [[TRUNC31:%[0-9]+]]:_(s1) = G_TRUNC [[COPY31]](s32)
; GFX11-NEXT:   [[COPY32:%[0-9]+]]:_(s32) = COPY $vgpr2
; GFX11-NEXT:   [[TRUNC32:%[0-9]+]]:_(s1) = G_TRUNC [[COPY32]](s32)
; GFX11-NEXT:   [[COPY33:%[0-9]+]]:_(s32) = COPY $vgpr3
; GFX11-NEXT:   [[TRUNC33:%[0-9]+]]:_(s1) = G_TRUNC [[COPY33]](s32)
; GFX11-NEXT:   [[COPY34:%[0-9]+]]:_(s32) = COPY $vgpr4
; GFX11-NEXT:   [[TRUNC34:%[0-9]+]]:_(s1) = G_TRUNC [[COPY34]](s32)
; GFX11-NEXT:   [[COPY35:%[0-9]+]]:_(s32) = COPY $vgpr5
; GFX11-NEXT:   [[TRUNC35:%[0-9]+]]:_(s1) = G_TRUNC [[COPY35]](s32)
; GFX11-NEXT:   [[COPY36:%[0-9]+]]:_(s32) = COPY $vgpr6
; GFX11-NEXT:   [[TRUNC36:%[0-9]+]]:_(s1) = G_TRUNC [[COPY36]](s32)
; GFX11-NEXT:   [[COPY37:%[0-9]+]]:_(s32) = COPY $vgpr7
; GFX11-NEXT:   [[TRUNC37:%[0-9]+]]:_(s1) = G_TRUNC [[COPY37]](s32)
; GFX11-NEXT:   [[COPY38:%[0-9]+]]:_(s32) = COPY $vgpr8
; GFX11-NEXT:   [[TRUNC38:%[0-9]+]]:_(s1) = G_TRUNC [[COPY38]](s32)
; GFX11-NEXT:   [[COPY39:%[0-9]+]]:_(s32) = COPY $vgpr9
; GFX11-NEXT:   [[TRUNC39:%[0-9]+]]:_(s1) = G_TRUNC [[COPY39]](s32)
; GFX11-NEXT:   [[COPY40:%[0-9]+]]:_(s32) = COPY $vgpr10
; GFX11-NEXT:   [[TRUNC40:%[0-9]+]]:_(s1) = G_TRUNC [[COPY40]](s32)
; GFX11-NEXT:   [[COPY41:%[0-9]+]]:_(s32) = COPY $vgpr11
; GFX11-NEXT:   [[TRUNC41:%[0-9]+]]:_(s1) = G_TRUNC [[COPY41]](s32)
; GFX11-NEXT:   [[COPY42:%[0-9]+]]:_(s32) = COPY $vgpr12
; GFX11-NEXT:   [[TRUNC42:%[0-9]+]]:_(s1) = G_TRUNC [[COPY42]](s32)
; GFX11-NEXT:   [[COPY43:%[0-9]+]]:_(s32) = COPY $vgpr13
; GFX11-NEXT:   [[TRUNC43:%[0-9]+]]:_(s1) = G_TRUNC [[COPY43]](s32)
; GFX11-NEXT:   [[COPY44:%[0-9]+]]:_(s32) = COPY $vgpr14
; GFX11-NEXT:   [[TRUNC44:%[0-9]+]]:_(s1) = G_TRUNC [[COPY44]](s32)
; GFX11-NEXT:   [[COPY45:%[0-9]+]]:_(s32) = COPY $vgpr15
; GFX11-NEXT:   [[TRUNC45:%[0-9]+]]:_(s1) = G_TRUNC [[COPY45]](s32)
; GFX11-NEXT:   [[COPY46:%[0-9]+]]:_(s32) = COPY $vgpr16
; GFX11-NEXT:   [[TRUNC46:%[0-9]+]]:_(s1) = G_TRUNC [[COPY46]](s32)
; GFX11-NEXT:   [[COPY47:%[0-9]+]]:_(s32) = COPY $vgpr17
; GFX11-NEXT:   [[TRUNC47:%[0-9]+]]:_(s1) = G_TRUNC [[COPY47]](s32)
; GFX11-NEXT:   [[COPY48:%[0-9]+]]:_(s32) = COPY $vgpr18
; GFX11-NEXT:   [[TRUNC48:%[0-9]+]]:_(s1) = G_TRUNC [[COPY48]](s32)
; GFX11-NEXT:   [[COPY49:%[0-9]+]]:_(s32) = COPY $vgpr19
; GFX11-NEXT:   [[TRUNC49:%[0-9]+]]:_(s1) = G_TRUNC [[COPY49]](s32)
; GFX11-NEXT:   [[COPY50:%[0-9]+]]:_(s32) = COPY $vgpr20
; GFX11-NEXT:   [[TRUNC50:%[0-9]+]]:_(s1) = G_TRUNC [[COPY50]](s32)
; GFX11-NEXT:   [[COPY51:%[0-9]+]]:_(s32) = COPY $vgpr21
; GFX11-NEXT:   [[TRUNC51:%[0-9]+]]:_(s1) = G_TRUNC [[COPY51]](s32)
; GFX11-NEXT:   [[COPY52:%[0-9]+]]:_(s32) = COPY $vgpr22
; GFX11-NEXT:   [[TRUNC52:%[0-9]+]]:_(s1) = G_TRUNC [[COPY52]](s32)
; GFX11-NEXT:   [[COPY53:%[0-9]+]]:_(s32) = COPY $vgpr23
; GFX11-NEXT:   [[TRUNC53:%[0-9]+]]:_(s1) = G_TRUNC [[COPY53]](s32)
; GFX11-NEXT:   [[COPY54:%[0-9]+]]:_(s32) = COPY $vgpr24
; GFX11-NEXT:   [[TRUNC54:%[0-9]+]]:_(s1) = G_TRUNC [[COPY54]](s32)
; GFX11-NEXT:   [[COPY55:%[0-9]+]]:_(s32) = COPY $vgpr25
; GFX11-NEXT:   [[TRUNC55:%[0-9]+]]:_(s1) = G_TRUNC [[COPY55]](s32)
; GFX11-NEXT:   [[COPY56:%[0-9]+]]:_(s32) = COPY $vgpr26
; GFX11-NEXT:   [[TRUNC56:%[0-9]+]]:_(s1) = G_TRUNC [[COPY56]](s32)
; GFX11-NEXT:   [[COPY57:%[0-9]+]]:_(s32) = COPY $vgpr27
; GFX11-NEXT:   [[TRUNC57:%[0-9]+]]:_(s1) = G_TRUNC [[COPY57]](s32)
; GFX11-NEXT:   [[COPY58:%[0-9]+]]:_(s32) = COPY $vgpr28
; GFX11-NEXT:   [[TRUNC58:%[0-9]+]]:_(s1) = G_TRUNC [[COPY58]](s32)
; GFX11-NEXT:   [[COPY59:%[0-9]+]]:_(s32) = COPY $vgpr29
; GFX11-NEXT:   [[TRUNC59:%[0-9]+]]:_(s1) = G_TRUNC [[COPY59]](s32)
; GFX11-NEXT:   [[COPY60:%[0-9]+]]:_(s32) = COPY $vgpr30
; GFX11-NEXT:   [[TRUNC60:%[0-9]+]]:_(s1) = G_TRUNC [[COPY60]](s32)

; GFX11-NEXT:   [[FRAME0:%[0-9]+]]:_(p5) = G_FRAME_INDEX %fixed-stack.2
; GFX11-NEXT:   [[LOAD0:%[0-9]+]]:_(s32) = G_LOAD [[FRAME0]](p5) :: (invariant load (s1) from %fixed-stack.2, align 16, addrspace 5)
; GFX11-NEXT:   [[TRUNC61:%[0-9]+]]:_(s1) = G_TRUNC [[LOAD0]](s32)
; GFX11-NEXT:   [[FRAME1:%[0-9]+]]:_(p5) = G_FRAME_INDEX %fixed-stack.1
; GFX11-NEXT:   [[LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[FRAME1]](p5) :: (invariant load (s1) from %fixed-stack.1, align 4, addrspace 5)
; GFX11-NEXT:   [[TRUNC62:%[0-9]+]]:_(s1) = G_TRUNC [[LOAD1]](s32)

; GFX11-NEXT:   [[FRAME2:%[0-9]+]]:_(p5) = G_FRAME_INDEX %fixed-stack.0
; GFX11-NEXT:   [[LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[FRAME2]](p5) :: (invariant load (s1) from %fixed-stack.0, align 8, addrspace 5)
; GFX11-NEXT:   [[TRUNC63:%[0-9]+]]:_(s1) = G_TRUNC [[LOAD2]](s32)
; GFX11-NEXT:   [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF
;
; GFX11-NEXT:   G_STORE [[COPY]](s1), [[DEF]](p1) :: (store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:   [[CONST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; GFX11-NEXT:   [[PTRADD1:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST1]]
; GFX11-NEXT:   G_STORE [[COPY1]](s1), [[PTRADD1]](p1) :: (store (s1) into `ptr addrspace(1) undef` + 1, addrspace 1)
;
; GFX11:        [[CONST63:%[0-9]+]]:_(s64) = G_CONSTANT i64 63
; GFX11-NEXT:   [[PTRADD63:%[0-9]+]]:_(p1) = G_PTR_ADD [[DEF]], [[CONST63]]
; GFX11-NEXT:   G_STORE [[TRUNC63]](s1), [[PTRADD63]](p1) :: (store (s1) into `ptr addrspace(1) undef` + 63, addrspace 1)

  store [64 x i1] %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_i1_i1_inreg(i1 %arg0, i1 inreg %arg1) {
; GFX9-LABEL: name: void_func_i1_i1_inreg
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr6, $sgpr4_sgpr5
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $sgpr6
; GFX9-NEXT:    [[TRUNC2:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s32)
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    G_STORE [[TRUNC2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_i1_i1_inreg
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0, $sgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $sgpr1
; GFX11-NEXT:    [[TRUNC2:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s32)
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    G_STORE [[TRUNC2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store volatile i1 %arg0, ptr addrspace(1) undef
  store volatile i1 %arg1, ptr addrspace(1) undef
  ret void
}

define void @void_func_i1_inreg_i1(i1 inreg %arg0, i1 %arg1) {
; GFX9-LABEL: name: void_func_i1_inreg_i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr4, $sgpr6_sgpr7
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr4
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr6_sgpr7
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_i1_inreg_i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0, $sgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr0
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:sreg_32(s1) = COPY $sgpr1
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store volatile i1 %arg0, ptr addrspace(1) undef
  store volatile i1 %arg1, ptr addrspace(1) undef
  ret void
}

define void @void_func_zeroext_i1_i1_inreg(i1 zeroext %arg0, i1 inreg %arg1) {
; GFX9-LABEL: name: void_func_zeroext_i1_i1_inreg
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr6, $sgpr4_sgpr5
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $sgpr6
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s32)
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_zeroext_i1_i1_inreg
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0, $sgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $sgpr1
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s32)
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store volatile i1 %arg0, ptr addrspace(1) undef
  store volatile i1 %arg1, ptr addrspace(1) undef
  ret void
}

define void @void_func_i1_inreg_zeroext_i1(i1 inreg %arg0, i1 zeroext %arg1) {
; GFX9-LABEL: name: void_func_i1_inreg_zeroext_i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr4, $sgpr6_sgpr7
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr4
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr6_sgpr7
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_i1_inreg_zeroext_i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0, $sgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr0
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:sreg_32(s1) = COPY $sgpr1
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store volatile i1 %arg0, ptr addrspace(1) undef
  store volatile i1 %arg1, ptr addrspace(1) undef
  ret void
}

define void @void_func_signext_i1_i1_inreg(i1 signext %arg0, i1 inreg %arg1) {
; GFX9-LABEL: name: void_func_signext_i1_i1_inreg
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr6, $sgpr4_sgpr5
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:sreg_64(s1) = COPY $sgpr4_sgpr5
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $sgpr6
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s32)
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_signext_i1_i1_inreg
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0, $sgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:sreg_32(s1) = COPY $sgpr0
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:_(s32) = COPY $sgpr1
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY2]](s32)
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[COPY]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store volatile i1 %arg0, ptr addrspace(1) undef
  store volatile i1 %arg1, ptr addrspace(1) undef
  ret void
}

define void @void_func_i1_inreg_signext_i1(i1 inreg %arg0, i1 signext %arg1) {
; GFX9-LABEL: name: void_func_i1_inreg_signext_i1
; GFX9: bb.1 (%ir-block.0):
; GFX9-NEXT:    liveins: $sgpr4, $sgpr6_sgpr7
; GFX9-NEXT: {{  $}}
; GFX9-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr4
; GFX9-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX9-NEXT:    [[COPY2:%[0-9]+]]:sreg_64(s1) = COPY $sgpr6_sgpr7
; GFX9-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX9-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX9-NEXT:    SI_RETURN
;
; GFX11-LABEL: name: void_func_i1_inreg_signext_i1
; GFX11: bb.1 (%ir-block.0):
; GFX11-NEXT:    liveins: $sgpr0, $sgpr1
; GFX11-NEXT: {{  $}}
; GFX11-NEXT:    [[COPY:%[0-9]+]]:_(s32) = COPY $sgpr0
; GFX11-NEXT:    [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; GFX11-NEXT:    [[COPY2:%[0-9]+]]:sreg_32(s1) = COPY $sgpr1
; GFX11-NEXT:    [[DEF:%[0-9]+]]:_(p1) = G_IMPLICIT_DEF  
; GFX11-NEXT:    G_STORE [[TRUNC]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    G_STORE [[COPY2]](s1), [[DEF]](p1) :: (volatile store (s1) into `ptr addrspace(1) undef`, addrspace 1)
; GFX11-NEXT:    SI_RETURN
  store volatile i1 %arg0, ptr addrspace(1) undef
  store volatile i1 %arg1, ptr addrspace(1) undef
  ret void
}
