; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

@array1 = global [5 x i32] zeroinitializer, align 4
@array2 = global [5 x i32] zeroinitializer, align 4
@array3 = global [5 x i32] zeroinitializer, align 4
@array4 = global [5 x i32] zeroinitializer, align 4

;        bb.0.entry
;           /   |
;     bb.1.bb1  |
;           \   |
;         bb.2.bb2
;
define amdgpu_ps void @test10(ptr addrspace(1) %p1, ptr addrspace(3) %p2, i1 %cond1, ptr addrspace(1) %p3, ptr addrspace(1) %p4, ptr addrspace(1) %p5, i32 %arg1, i32 %arg2) {
; CHECK-LABEL: # Machine code for function test10: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]], $vgpr7 in [[Reg8:%[0-9]+]], $vgpr8 in [[Reg9:%[0-9]+]], $vgpr9 in [[Reg10:%[0-9]+]], $vgpr10 in [[Reg11:%[0-9]+]], $vgpr11 in [[Reg12:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x40000000), %bb.2(0x40000000); %bb.1(50.00%), %bb.2(50.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10, $vgpr11
; CHECK-NEXT:   [[Reg12]]:vgpr_32 = COPY killed $vgpr11
; CHECK-NEXT:   [[Reg11]]:vgpr_32 = COPY killed $vgpr10
; CHECK-NEXT:   [[Reg10]]:vgpr_32 = COPY killed $vgpr9
; CHECK-NEXT:   [[Reg9]]:vgpr_32 = COPY killed $vgpr8
; CHECK-NEXT:   [[Reg8]]:vgpr_32 = COPY killed $vgpr7
; CHECK-NEXT:   [[Reg7]]:vgpr_32 = COPY killed $vgpr6
; CHECK-NEXT:   [[Reg6]]:vgpr_32 = COPY killed $vgpr5
; CHECK-NEXT:   [[Reg5]]:vgpr_32 = COPY killed $vgpr4
; CHECK-NEXT:   [[Reg4]]:vgpr_32 = COPY killed $vgpr3
; CHECK-NEXT:   [[Reg3]]:vgpr_32 = COPY killed $vgpr2
; CHECK-NEXT:   [[Reg2]]:vgpr_32 = COPY killed $vgpr1
; CHECK-NEXT:   [[Reg1]]:vgpr_32 = COPY killed $vgpr0
; CHECK-NEXT:   [[Reg13:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg9]]:vgpr_32, %subreg.sub0, killed [[Reg10]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg14:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg7]]:vgpr_32, %subreg.sub0, killed [[Reg8]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg15:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg5]]:vgpr_32, %subreg.sub0, killed [[Reg6]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 1, killed [[Reg4]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:sreg_32 = V_CMP_EQ_U32_e64 1, killed [[Reg17]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg20:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg20]]:vgpr_32, 8, killed [[Reg19]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg23]]:vgpr_32, 8, killed [[Reg22]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg24]]:vgpr_32, 16, killed [[Reg21]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 12, 0, implicit $exec :: (load (s8) from %ir.gep1, addrspace 1)
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 13, 0, implicit $exec :: (load (s8) from %ir.gep1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg27]]:vgpr_32, 8, killed [[Reg26]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg16]]:vreg_64, 14, 0, implicit $exec :: (load (s8) from %ir.gep1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE killed [[Reg16]]:vreg_64, 15, 0, implicit $exec :: (load (s8) from %ir.gep1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg31:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg30]]:vgpr_32, 8, killed [[Reg29]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg32:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg31]]:vgpr_32, 16, killed [[Reg28]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg33:%[0-9]+]]:vreg_128 = GLOBAL_LOAD_DWORDX4 [[Reg15]]:vreg_64, 16, 0, implicit $exec :: (load (s128) from %ir.p3 + 16, align 4, addrspace 1)
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:vreg_128 = GLOBAL_LOAD_DWORDX4 [[Reg15]]:vreg_64, 0, 0, implicit $exec :: (load (s128) from %ir.p3, align 4, addrspace 1)
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg34]].sub0:vreg_128, [[Reg25]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg15]]:vreg_64, [[Reg35]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg35]]:vgpr_32, [[Reg25]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array1, target-flags(amdgpu-gotprel32-hi) @array1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg37]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:vreg_64 = COPY [[Reg38]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg39]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array1, i64 20)`)
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array3, target-flags(amdgpu-gotprel32-hi) @array3, implicit-def dead $scc
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg41]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:vreg_64 = COPY [[Reg42]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD killed [[Reg43]]:vreg_64, [[Reg40]]:vgpr_32, 4, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 4)`)
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 20, 0, implicit $exec :: (load (s8) from %ir.p4 + 20, addrspace 1)
; CHECK-NEXT:   [[Reg45:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 21, 0, implicit $exec :: (load (s8) from %ir.p4 + 21, addrspace 1)
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg45]]:vgpr_32, 8, killed [[Reg44]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 22, 0, implicit $exec :: (load (s8) from %ir.p4 + 22, addrspace 1)
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 23, 0, implicit $exec :: (load (s8) from %ir.p4 + 23, addrspace 1)
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg48]]:vgpr_32, 8, killed [[Reg47]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg49]]:vgpr_32, 16, killed [[Reg46]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 12, 0, implicit $exec :: (load (s8) from %ir.p4 + 12, addrspace 1)
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 13, 0, implicit $exec :: (load (s8) from %ir.p4 + 13, addrspace 1)
; CHECK-NEXT:   [[Reg53:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg52]]:vgpr_32, 8, killed [[Reg51]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg54:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 14, 0, implicit $exec :: (load (s8) from %ir.p4 + 14, addrspace 1)
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 15, 0, implicit $exec :: (load (s8) from %ir.p4 + 15, addrspace 1)
; CHECK-NEXT:   [[Reg56:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg55]]:vgpr_32, 8, killed [[Reg54]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg57:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg56]]:vgpr_32, 16, killed [[Reg53]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg58:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg14]]:vreg_64, 0, 0, implicit $exec :: (load (s16) from %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg59:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg14]]:vreg_64, 2, 0, implicit $exec :: (load (s16) from %ir.p4 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg60:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg59]]:vgpr_32, 16, killed [[Reg58]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg61:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg60]]:vgpr_32, [[Reg35]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg14]]:vreg_64, killed [[Reg61]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg62:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg18]]:sreg_32, %bb.2, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.1
; EMPTY:
; CHECK: bb.1.bb1:
; CHECK-NEXT: ; predecessors: %bb.0
; CHECK-NEXT:   successors: %bb.2(0x80000000); %bb.2(100.00%)
; EMPTY:
; CHECK:   [[Reg63:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 0, 0, implicit $exec :: (load (s8) from %ir.p2, addrspace 3)
; CHECK-NEXT:   [[Reg64:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 1, 0, implicit $exec :: (load (s8) from %ir.p2 + 1, addrspace 3)
; CHECK-NEXT:   [[Reg65:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 2, 0, implicit $exec :: (load (s8) from %ir.p2 + 2, addrspace 3)
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 killed [[Reg3]]:vgpr_32, 3, 0, implicit $exec :: (load (s8) from %ir.p2 + 3, addrspace 3)
; CHECK-NEXT:   [[Reg67:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg64]]:vgpr_32, 8, killed [[Reg63]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg68:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg66]]:vgpr_32, 8, killed [[Reg65]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg69:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg68]]:vgpr_32, 16, killed [[Reg67]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg70:%[0-9]+]]:vreg_64 = COPY killed [[Reg38]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg71:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg70]]:vreg_64, 28, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array1, i64 28)`)
; CHECK-NEXT:   [[Reg72:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array2, target-flags(amdgpu-gotprel32-hi) @array2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg73:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg72]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg74:%[0-9]+]]:vreg_64 = COPY killed [[Reg73]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD [[Reg74]]:vreg_64, [[Reg71]]:vgpr_32, 68, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array2, i64 68)`)
; CHECK-NEXT:   [[Reg75:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array4, target-flags(amdgpu-gotprel32-hi) @array4, implicit-def dead $scc
; CHECK-NEXT:   [[Reg76:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg75]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg77:%[0-9]+]]:vreg_64 = COPY killed [[Reg76]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg78:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg77]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array4, i64 20)`)
; CHECK-NEXT:   [[Reg79:%[0-9]+]]:vreg_64 = COPY killed [[Reg42]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD [[Reg79]]:vreg_64, [[Reg78]]:vgpr_32, 60, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 60)`)
; CHECK-NEXT:   [[Reg80:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg79]]:vreg_64, 84, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 84)`)
; CHECK-NEXT:   [[Reg81:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg79]]:vreg_64, 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from @array3)
; CHECK-NEXT:   [[Reg82:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg70]]:vreg_64, 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from @array1)
; CHECK-NEXT:   [[Reg83:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg74]]:vreg_64, 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from @array2)
; CHECK-NEXT:   [[Reg84:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg77]]:vreg_64, 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from @array4)
; CHECK-NEXT:   [[Reg85:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg83]]:vgpr_32, [[Reg84]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg86:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg85]]:vgpr_32, %subreg.sub0, undef [[Reg87:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg88:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 [[Reg81]]:vgpr_32, [[Reg82]]:vgpr_32, killed [[Reg86]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg89:%[0-9]+]]:vgpr_32 = COPY killed [[Reg88]].sub0:vreg_64
; CHECK-NEXT:   [[Reg90:%[0-9]+]]:vgpr_32 = COPY [[Reg34]].sub1:vreg_128
; CHECK-NEXT:   [[Reg91:%[0-9]+]]:vgpr_32 = COPY killed [[Reg34]].sub2:vreg_128
; CHECK-NEXT:   [[Reg92:%[0-9]+]]:vgpr_32 = COPY [[Reg33]].sub2:vreg_128
; CHECK-NEXT:   [[Reg93:%[0-9]+]]:vgpr_32 = COPY killed [[Reg33]].sub3:vreg_128
; EMPTY:
; CHECK: bb.2.bb2:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.1
; EMPTY:
; CHECK:   [[Reg94:%[0-9]+]]:vgpr_32 = PHI [[Reg25]]:vgpr_32, %bb.0, [[Reg71]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg95:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg78]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg96:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg80]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg97:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg81]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg98:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg82]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg99:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg83]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg100:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg84]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg101:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg89]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg102:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg90]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg103:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg91]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg104:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg92]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg105:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg93]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg106:%[0-9]+]]:vgpr_32 = PHI [[Reg32]]:vgpr_32, %bb.0, [[Reg25]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg107:%[0-9]+]]:vgpr_32 = PHI [[Reg40]]:vgpr_32, %bb.0, [[Reg11]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg108:%[0-9]+]]:vgpr_32 = PHI [[Reg36]]:vgpr_32, %bb.0, [[Reg69]]:vgpr_32, %bb.1
; CHECK-NEXT:   SI_END_CF killed [[Reg62]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg109:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg94]]:vgpr_32, killed [[Reg57]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg110:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg109]]:vgpr_32, killed [[Reg50]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg111:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg110]]:vgpr_32, killed [[Reg40]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg112:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg111]]:vgpr_32, killed [[Reg95]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg113:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg112]]:vgpr_32, killed [[Reg96]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg114:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg113]]:vgpr_32, killed [[Reg35]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg115:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg114]]:vgpr_32, killed [[Reg97]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg116:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg115]]:vgpr_32, killed [[Reg98]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg117:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg116]]:vgpr_32, killed [[Reg99]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg118:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg117]]:vgpr_32, killed [[Reg100]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg119:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg118]]:vgpr_32, killed [[Reg101]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg120:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg119]]:vgpr_32, killed [[Reg102]]:vgpr_32, killed [[Reg103]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg121:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg120]]:vgpr_32, killed [[Reg104]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg122:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg121]]:vgpr_32, killed [[Reg105]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg123:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg122]]:vgpr_32, killed [[Reg106]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg124:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg108]]:vgpr_32, %subreg.sub0, undef [[Reg125:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg126:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 killed [[Reg123]]:vgpr_32, killed [[Reg107]]:vgpr_32, killed [[Reg124]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg127:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg126]].sub0:vreg_64, killed [[Reg12]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_SHORT_D16_HI [[Reg13]]:vreg_64, [[Reg127]]:vgpr_32, 2, 0, implicit $exec :: (store (s16) into %ir.p5 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_SHORT killed [[Reg13]]:vreg_64, killed [[Reg127]]:vgpr_32, 0, 0, implicit $exec :: (store (s16) into %ir.p5, addrspace 1)
; CHECK-NEXT:   S_ENDPGM 0
; EMPTY:
; CHECK: # End machine code for function test10.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg12]] = 99.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 78.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 57.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 88.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 32.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 47.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 47.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 64.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 44.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 32.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 24.0
; CHECK-NEXT: Next-use distance of Register [[Reg58]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 17.0
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg64]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg67]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg68]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg69]] = 40.0
; CHECK-NEXT: Next-use distance of Register [[Reg70]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg71]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg72]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg73]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg74]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg75]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg76]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg77]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg78]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg79]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg80]] = 15.0
; CHECK-NEXT: Next-use distance of Register [[Reg81]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg82]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg83]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg84]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg85]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg86]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg88]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg89]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg90]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg91]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg92]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg93]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg94]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg95]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg96]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg97]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg98]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg99]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg100]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg101]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg102]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg103]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg104]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg105]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg106]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg107]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg108]] = 17.0
; CHECK-NEXT: Next-use distance of Register [[Reg109]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg110]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg111]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg112]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg113]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg114]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg115]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg116]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg117]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg118]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg119]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg120]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg121]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg122]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg123]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg124]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg126]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg127]] = 1.0
entry:
   %ld1 = load i32, ptr addrspace(1) %p1, align 1
   %gep1 = getelementptr inbounds i32, ptr addrspace(1) %p1, i64 3
   %ld2 = load i32, ptr addrspace(1) %gep1, align 1
   %load1 = load i32, ptr addrspace(1) %p3, align 4
   %tmp1 = add i32 %load1, %ld1
   %load2 = load <8 x i32>, ptr addrspace(1) %p3, align 1
   store i32 %tmp1, ptr addrspace(1) %p3
   %add1 = add i32 %ld1, %tmp1
   %idx10 = getelementptr inbounds [5 x i32], [5 x i32]* @array1, i64 1, i64 0
   %val0 = load i32, i32* %idx10, align 4
   %idx20 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 0, i64 1
   store i32 %val0, i32 *%idx20
   %load3 = load <8 x i32>, ptr addrspace(1) %p4, align 1
   %load4 = load i32, ptr addrspace(1) %p4, align 2
   %tmp2 = add i32 %load4, %tmp1
   store i32 %tmp2, ptr addrspace(1) %p4
   br i1 %cond1, label %bb1, label %bb2

bb1:
   %ld3 = load i32, ptr addrspace(3) %p2, align 1
   %idx12 = getelementptr inbounds [5 x i32], [5 x i32]* @array1, i64 1, i64 2
   %val2 = load i32, i32* %idx12, align 4
   %idx22 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 3, i64 2
   store i32 %val2, i32 *%idx22
   %idx13 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 1, i64 0
   %val3 = load i32, i32* %idx13, align 4
   %idx23 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 3, i64 0
   store i32 %val3, i32 *%idx23
   %idx14 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 4, i64 1
   %val4 = load i32, i32* %idx14, align 4
   %idx24 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 3, i64 0
   %idx15 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 0, i64 0
   %val5 = load i32, i32* %idx15, align 4
   %idx16 = getelementptr inbounds [5 x i32], [5 x i32]* @array1, i64 0, i64 0
   %val6 = load i32, i32* %idx16, align 4
   %idx17 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 0, i64 0
   %val7 = load i32, i32* %idx17, align 4
   %idx18 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 0, i64 0
   %val8 = load i32, i32* %idx18, align 4
   %mul10 = mul i32 %val5, %val6
   %mul11 = mul i32 %val7, %val8
   %add100 = add i32 %mul10, %mul11
   %extract1 = extractelement < 8 x i32> %load2, i32 1
   %extract2 = extractelement < 8 x i32> %load2, i32 2
   %extract3 = extractelement < 8 x i32> %load2, i32 6
   %extract4 = extractelement < 8 x i32> %load2, i32 7
   br label %bb2

bb2:
   %phi1 = phi i32 [ %ld1, %entry ], [ %val2, %bb1 ]
   %phi2 = phi i32 [ %val0, %entry ], [ %val3, %bb1 ]
   %phi3 = phi i32 [ %val0, %entry ], [ %val4, %bb1 ]
   %phi4 = phi i32 [ %val0, %entry ], [ %val5, %bb1 ]
   %phi5 = phi i32 [ %val0, %entry ], [ %val6, %bb1 ]
   %phi6 = phi i32 [ %val0, %entry ], [ %val7, %bb1 ]
   %phi7 = phi i32 [ %val0, %entry ], [ %val8, %bb1 ]
   %phi8 = phi i32 [ %val0, %entry ], [ %add100, %bb1 ]
   %phi9 = phi i32 [ %val0, %entry ], [ %extract1, %bb1 ]
   %phi10 = phi i32 [ %val0, %entry ], [ %extract2, %bb1 ]
   %phi11 = phi i32 [ %val0, %entry ], [ %extract3, %bb1 ]
   %phi12 = phi i32 [ %val0, %entry ], [ %extract4, %bb1 ]
   %phi13 = phi i32 [ %ld2, %entry ], [ %ld1, %bb1 ]
   %phi14 = phi i32 [ %val0, %entry ], [ %arg1, %bb1 ]
   %phi15 = phi i32 [ %add1, %entry ], [ %ld3, %bb1 ]
   %extract5 = extractelement < 8 x i32> %load3, i32 3
   %extract6 = extractelement < 8 x i32> %load3, i32 5
   %res1 = add i32 %phi1, %extract5
   %res2 = mul i32 %res1, %extract6
   %res3 = sub i32 %res2, %val0
   %res4 = sub i32 %res3, %phi2
   %res5 = add i32 %res4, %phi3
   %res6 = sub i32 %res5, %tmp1
   %res7 = mul i32 %res6, %phi4
   %res8 = mul i32 %res7, %phi5
   %res9 = sub i32 %res8, %phi6
   %res10 = add i32 %res9, %phi7
   %res11 = mul i32 %res10, %phi8
   %res12 = add i32 %res11, %phi9
   %res13 = add i32 %res12, %phi10
   %res14 = sub i32 %res13, %phi11
   %res15 = sub i32 %res14, %phi12
   %res16 = mul i32 %res15, %phi13
   %res17 = mul i32 %res16, %phi14
   %res18 = add i32 %res17, %phi15
   %res19 = sub i32 %res18, %arg2
   store i32 %res19, ptr addrspace(1) %p5, align 2
   ret void
}
