; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

@array2 = global [5 x i32] zeroinitializer, align 4
@array3 = global [5 x i32] zeroinitializer, align 4
@array4 = global [5 x i32] zeroinitializer, align 4
@array5 = global [5 x i32] zeroinitializer, align 4

@array6 = global [5 x i32] zeroinitializer, align 4
@array7 = global [5 x i32] zeroinitializer, align 4
@array8 = global [5 x i32] zeroinitializer, align 4
@array9 = global [5 x i32] zeroinitializer, align 4

;                bb.0.entry
;                  /    |
;            bb.1.bb1   |
;                  \    |
;                  bb.2.bb2
;                  |    \
;                  |   bb.3.bb4.preheader
;                  |           |
;                  |         bb.19<-+
;                  |           +----+
;                  |           |
;                  |     bb.20.bb14.loopexit
;                  |      /
;                bb.18.Flow17
;                 /        |
;              bb.4.bb3    |
;              /    |      |
;        bb.10.bb7  |      |
;              \    |      |
;           bb.5.Flow16    |
;              /    |      |
;        bb.6.bb6   |      |
;           /  |    |      |
;    bb.9.bb9  |    |      |
;           \  |    |      |
;     bb.7.Flow14   |      |
;           /  |    |      |
;     bb.8.bb8 |    |      |
;           \  |    |      |
;     bb.11.Flow15  |      |
;              \    |      |
;            bb.13.bb10    |
;              /    |      |
;     bb.16.bb12    |      |
;              \    |      |
;            bb.14.Flow    |
;              /    |      |
;      bb.15.bb11   |      |
;              \    |      |
;            bb.17.bb13    |
;                    \     |
;                  bb.12.Flow18
;                       |
;                  bb.21.bb14
;
define amdgpu_ps void @test(ptr addrspace(1) %p1, ptr addrspace(3) %p2, i1 %cond1, i1 %cond2, ptr addrspace(1) %p3, ptr addrspace(1) %p4, ptr addrspace(1) %p5, ptr addrspace(1) %p6, ptr addrspace(1) %p7, ptr addrspace(1) %p8, ptr addrspace(1) %p9, i32 %TC1) {
; CHECK-LABEL: # Machine code for function test: IsSSA, TracksLiveness
; CHECK-NEXT: Frame Objects:
; CHECK-NEXT:   fi#0: variable sized, align=1, at location [SP]
; CHECK-NEXT:   fi#1: variable sized, align=1, at location [SP]
; CHECK-NEXT: save/restore points:
; CHECK-NEXT: save points are empty
; CHECK-NEXT: restore points are empty
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]], $vgpr7 in [[Reg8:%[0-9]+]], $vgpr8 in [[Reg9:%[0-9]+]], $vgpr9 in [[Reg10:%[0-9]+]], $vgpr10 in [[Reg11:%[0-9]+]], $vgpr11 in [[Reg12:%[0-9]+]], $vgpr12 in [[Reg13:%[0-9]+]], $vgpr13 in [[Reg14:%[0-9]+]], $vgpr14 in [[Reg15:%[0-9]+]], $vgpr15 in [[Reg16:%[0-9]+]], $vgpr16 in [[Reg17:%[0-9]+]], $vgpr17 in [[Reg18:%[0-9]+]], $vgpr18 in [[Reg19:%[0-9]+]], $vgpr19 in [[Reg20:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x40000000), %bb.2(0x40000000); %bb.1(50.00%), %bb.2(50.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10, $vgpr11, $vgpr12, $vgpr13, $vgpr14, $vgpr15, $vgpr16, $vgpr17, $vgpr18, $vgpr19
; CHECK-NEXT:   [[Reg20]]:vgpr_32 = COPY killed $vgpr19
; CHECK-NEXT:   [[Reg19]]:vgpr_32 = COPY killed $vgpr18
; CHECK-NEXT:   [[Reg18]]:vgpr_32 = COPY killed $vgpr17
; CHECK-NEXT:   [[Reg17]]:vgpr_32 = COPY killed $vgpr16
; CHECK-NEXT:   [[Reg16]]:vgpr_32 = COPY killed $vgpr15
; CHECK-NEXT:   [[Reg15]]:vgpr_32 = COPY killed $vgpr14
; CHECK-NEXT:   [[Reg14]]:vgpr_32 = COPY killed $vgpr13
; CHECK-NEXT:   [[Reg13]]:vgpr_32 = COPY killed $vgpr12
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
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg6]]:vgpr_32, %subreg.sub0, killed [[Reg7]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 1, killed [[Reg4]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:sreg_32 = V_CMP_EQ_U32_e64 1, killed [[Reg23]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 1, killed [[Reg5]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:sreg_32 = V_CMP_EQ_U32_e64 1, killed [[Reg25]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:sreg_32 = S_XOR_B32 [[Reg26]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg22]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg22]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg29]]:vgpr_32, 8, killed [[Reg28]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg31:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg22]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg32:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg22]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg33:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg32]]:vgpr_32, 8, killed [[Reg31]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg33]]:vgpr_32, 16, killed [[Reg30]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg22]]:vreg_64, 12, 0, implicit $exec :: (load (s8) from %ir.gep1, addrspace 1)
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg22]]:vreg_64, 13, 0, implicit $exec :: (load (s8) from %ir.gep1 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg22]]:vreg_64, 14, 0, implicit $exec :: (load (s8) from %ir.gep1 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg22]]:vreg_64, 15, 0, implicit $exec :: (load (s8) from %ir.gep1 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:vreg_128 = GLOBAL_LOAD_DWORDX4 [[Reg21]]:vreg_64, 16, 0, implicit $exec :: (load (s128) from %ir.p3 + 16, align 4, addrspace 1)
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:vreg_128 = GLOBAL_LOAD_DWORDX4 [[Reg21]]:vreg_64, 0, 0, implicit $exec :: (load (s128) from %ir.p3, align 4, addrspace 1)
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg40]].sub0:vreg_128, [[Reg34]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg21]]:vreg_64, [[Reg41]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg41]]:vgpr_32, [[Reg34]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:sreg_32 = SI_IF [[Reg24]]:sreg_32, %bb.2, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.1
; EMPTY:
; CHECK: bb.1.bb1:
; CHECK-NEXT: ; predecessors: %bb.0
; CHECK-NEXT:   successors: %bb.2(0x80000000); %bb.2(100.00%)
; EMPTY:
; CHECK:   [[Reg44:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 100, implicit $exec
; EMPTY:
; CHECK: bb.2.bb2:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.1
; CHECK-NEXT:   successors: %bb.3(0x40000000), %bb.18(0x40000000); %bb.3(50.00%), %bb.18(50.00%)
; EMPTY:
; CHECK:   [[Reg45:%[0-9]+]]:vgpr_32 = PHI [[Reg42]]:vgpr_32, %bb.0, [[Reg44]]:vgpr_32, %bb.1
; CHECK-NEXT:   SI_END_CF killed [[Reg43]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg27]]:sreg_32, %bb.18, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.3
; EMPTY:
; CHECK: bb.3.bb4.preheader:
; CHECK-NEXT: ; predecessors: %bb.2
; CHECK-NEXT:   successors: %bb.19(0x80000000); %bb.19(100.00%)
; EMPTY:
; CHECK:   [[Reg47:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array6, target-flags(amdgpu-gotprel32-hi) @array6, implicit-def dead $scc
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg47]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vreg_64 = COPY killed [[Reg48]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg49]]:vreg_64, 28, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array6, i64 28)`)
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array8, target-flags(amdgpu-gotprel32-hi) @array8, implicit-def dead $scc
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg51]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg53:%[0-9]+]]:vreg_64 = COPY killed [[Reg52]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg54:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg53]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array8, i64 20)`)
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:sreg_32 = S_MOV_B32 -1
; CHECK-NEXT:   [[Reg56:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; CHECK-NEXT:   S_BRANCH %bb.19
; EMPTY:
; CHECK: bb.4.bb3:
; CHECK-NEXT: ; predecessors: %bb.18
; CHECK-NEXT:   successors: %bb.10(0x40000000), %bb.5(0x40000000); %bb.10(50.00%), %bb.5(50.00%)
; EMPTY:
; CHECK:   [[Reg57:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg58:%[0-9]+]]:vgpr_32, 8, killed [[Reg59:%[0-9]+]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg60:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg61:%[0-9]+]]:vgpr_32, 8, killed [[Reg62:%[0-9]+]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg63:%[0-9]+]]:vreg_64 = REG_SEQUENCE [[Reg64:%[0-9]+]]:vgpr_32, %subreg.sub0, [[Reg65:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg67:%[0-9]+]]:vgpr_32, %subreg.sub0, killed [[Reg68:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg69:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg60]]:vgpr_32, 16, killed [[Reg57]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg70:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg71:%[0-9]+]]:vgpr_32, 0, 0, implicit $exec :: (load (s8) from %ir.p2, addrspace 3)
; CHECK-NEXT:   [[Reg72:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg71]]:vgpr_32, 1, 0, implicit $exec :: (load (s8) from %ir.p2 + 1, addrspace 3)
; CHECK-NEXT:   [[Reg73:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg71]]:vgpr_32, 2, 0, implicit $exec :: (load (s8) from %ir.p2 + 2, addrspace 3)
; CHECK-NEXT:   [[Reg74:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg71]]:vgpr_32, 3, 0, implicit $exec :: (load (s8) from %ir.p2 + 3, addrspace 3)
; CHECK-NEXT:   [[Reg75:%[0-9]+]]:vgpr_32 = DS_READ_B32_gfx9 killed [[Reg71]]:vgpr_32, 12, 0, implicit $exec :: (load (s32) from %ir.gep2, align 8, addrspace 3)
; CHECK-NEXT:   [[Reg76:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg72]]:vgpr_32, 8, killed [[Reg70]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg77:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg74]]:vgpr_32, 8, killed [[Reg73]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg78:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 [[Reg77]]:vgpr_32, 16, [[Reg76]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg79:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 [[Reg80:%[0-9]+]]:vgpr_32, [[Reg45]]:vgpr_32, 1900, 0, implicit $exec
; CHECK-NEXT:   [[Reg81:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg75]]:vgpr_32, [[Reg79]].sub0:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg82:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 20, 0, implicit $exec :: (load (s8) from %ir.p4 + 20, addrspace 1)
; CHECK-NEXT:   [[Reg83:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 21, 0, implicit $exec :: (load (s8) from %ir.p4 + 21, addrspace 1)
; CHECK-NEXT:   [[Reg84:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg83]]:vgpr_32, 8, killed [[Reg82]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg85:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 22, 0, implicit $exec :: (load (s8) from %ir.p4 + 22, addrspace 1)
; CHECK-NEXT:   [[Reg86:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 23, 0, implicit $exec :: (load (s8) from %ir.p4 + 23, addrspace 1)
; CHECK-NEXT:   [[Reg87:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg86]]:vgpr_32, 8, killed [[Reg85]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg88:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 12, 0, implicit $exec :: (load (s8) from %ir.p4 + 12, addrspace 1)
; CHECK-NEXT:   [[Reg89:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 13, 0, implicit $exec :: (load (s8) from %ir.p4 + 13, addrspace 1)
; CHECK-NEXT:   [[Reg90:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg89]]:vgpr_32, 8, killed [[Reg88]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg91:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 14, 0, implicit $exec :: (load (s8) from %ir.p4 + 14, addrspace 1)
; CHECK-NEXT:   [[Reg92:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 15, 0, implicit $exec :: (load (s8) from %ir.p4 + 15, addrspace 1)
; CHECK-NEXT:   [[Reg93:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg92]]:vgpr_32, 8, killed [[Reg91]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg94:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 8, 0, implicit $exec :: (load (s8) from %ir.p4 + 8, addrspace 1)
; CHECK-NEXT:   [[Reg95:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 9, 0, implicit $exec :: (load (s8) from %ir.p4 + 9, addrspace 1)
; CHECK-NEXT:   [[Reg96:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg95]]:vgpr_32, 8, killed [[Reg94]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg97:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 10, 0, implicit $exec :: (load (s8) from %ir.p4 + 10, addrspace 1)
; CHECK-NEXT:   [[Reg98:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 11, 0, implicit $exec :: (load (s8) from %ir.p4 + 11, addrspace 1)
; CHECK-NEXT:   [[Reg99:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg98]]:vgpr_32, 8, killed [[Reg97]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg100:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 4, 0, implicit $exec :: (load (s8) from %ir.p4 + 4, addrspace 1)
; CHECK-NEXT:   [[Reg101:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 5, 0, implicit $exec :: (load (s8) from %ir.p4 + 5, addrspace 1)
; CHECK-NEXT:   [[Reg102:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg101]]:vgpr_32, 8, killed [[Reg100]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg103:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 6, 0, implicit $exec :: (load (s8) from %ir.p4 + 6, addrspace 1)
; CHECK-NEXT:   [[Reg104:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg66]]:vreg_64, 7, 0, implicit $exec :: (load (s8) from %ir.p4 + 7, addrspace 1)
; CHECK-NEXT:   [[Reg105:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg104]]:vgpr_32, 8, killed [[Reg103]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg106:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg66]]:vreg_64, 0, 0, implicit $exec :: (load (s16) from %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg107:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg66]]:vreg_64, 2, 0, implicit $exec :: (load (s16) from %ir.p4 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg108:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg107]]:vgpr_32, 16, killed [[Reg106]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg109:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg108]]:vgpr_32, killed [[Reg110:%[0-9]+]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg66]]:vreg_64, killed [[Reg109]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p4, addrspace 1)
; CHECK-NEXT:   ADJCALLSTACKUP 0, 0, implicit-def dead $scc, implicit-def $sgpr32, implicit $sgpr32
; CHECK-NEXT:   [[Reg111:%[0-9]+]]:sreg_32_xexec_hi = COPY $sgpr32
; CHECK-NEXT:   [[Reg112:%[0-9]+]]:sreg_32 = S_ADD_I32 [[Reg111]]:sreg_32_xexec_hi, 1024, implicit-def dead $scc
; CHECK-NEXT:   $sgpr32 = COPY killed [[Reg112]]:sreg_32
; CHECK-NEXT:   ADJCALLSTACKDOWN 0, 0, implicit-def dead $scc, implicit-def $sgpr32, implicit $sgpr32
; CHECK-NEXT:   [[Reg113:%[0-9]+]]:vreg_64 = GLOBAL_LOAD_DWORDX2 [[Reg63]]:vreg_64, 0, 0, implicit $exec :: (load (s64) from %ir.p6, align 4, addrspace 1)
; CHECK-NEXT:   [[Reg114:%[0-9]+]]:vgpr_32 = COPY [[Reg113]].sub0:vreg_64
; CHECK-NEXT:   [[Reg115:%[0-9]+]]:vgpr_32 = COPY [[Reg113]].sub1:vreg_64
; CHECK-NEXT:   [[Reg116:%[0-9]+]]:vgpr_32 = nsw V_LSHLREV_B32_e64 2, [[Reg113]].sub0:vreg_64, implicit $exec
; CHECK-NEXT:   SCRATCH_STORE_DWORD_SVS [[Reg45]]:vgpr_32, killed [[Reg116]]:vgpr_32, [[Reg111]]:sreg_32_xexec_hi, 0, 0, implicit $exec, implicit $flat_scr :: (store (s32) into %ir.arrayidx11, addrspace 5)
; CHECK-NEXT:   [[Reg117:%[0-9]+]]:vgpr_32 = nsw V_LSHLREV_B32_e64 2, killed [[Reg113]].sub1:vreg_64, implicit $exec
; CHECK-NEXT:   SCRATCH_STORE_SHORT_SVS killed [[Reg77]]:vgpr_32, [[Reg117]]:vgpr_32, [[Reg111]]:sreg_32_xexec_hi, 2, 0, implicit $exec, implicit $flat_scr :: (store (s16) into %ir.arrayidx33 + 2, addrspace 5)
; CHECK-NEXT:   SCRATCH_STORE_SHORT_SVS killed [[Reg76]]:vgpr_32, killed [[Reg117]]:vgpr_32, killed [[Reg111]]:sreg_32_xexec_hi, 0, 0, implicit $exec, implicit $flat_scr :: (store (s16) into %ir.arrayidx33, addrspace 5)
; CHECK-NEXT:   [[Reg118:%[0-9]+]]:sreg_32 = S_XOR_B32 [[Reg24]]:sreg_32, [[Reg26]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg119:%[0-9]+]]:sreg_32 = S_XOR_B32 killed [[Reg118]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg120:%[0-9]+]]:sreg_32 = SI_IF [[Reg119]]:sreg_32, %bb.5, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.10
; EMPTY:
; CHECK: bb.5.Flow18:
; CHECK-NEXT: ; predecessors: %bb.4, %bb.10
; CHECK-NEXT:   successors: %bb.6(0x40000000), %bb.13(0x40000000); %bb.6(50.00%), %bb.13(50.00%)
; EMPTY:
; CHECK:   [[Reg121:%[0-9]+]]:vgpr_32 = PHI undef [[Reg122:%[0-9]+]]:vgpr_32, %bb.4, [[Reg123:%[0-9]+]]:vgpr_32, %bb.10
; CHECK-NEXT:   [[Reg124:%[0-9]+]]:vgpr_32 = PHI [[Reg69]]:vgpr_32, %bb.4, undef [[Reg125:%[0-9]+]]:vgpr_32, %bb.10
; CHECK-NEXT:   [[Reg126:%[0-9]+]]:vgpr_32 = PHI [[Reg81]]:vgpr_32, %bb.4, undef [[Reg127:%[0-9]+]]:vgpr_32, %bb.10
; CHECK-NEXT:   [[Reg128:%[0-9]+]]:vreg_256 = REG_SEQUENCE undef [[Reg129:%[0-9]+]].sub0:vreg_128, %subreg.sub0, undef [[Reg129]].sub1:vreg_128, %subreg.sub1, undef [[Reg129]].sub2:vreg_128, %subreg.sub2, [[Reg129]].sub3:vreg_128, %subreg.sub3, undef [[Reg130:%[0-9]+]].sub0:vreg_128, %subreg.sub4, [[Reg130]].sub1:vreg_128, %subreg.sub5, undef [[Reg130]].sub2:vreg_128, %subreg.sub6, undef [[Reg130]].sub3:vreg_128, %subreg.sub7
; CHECK-NEXT:   [[Reg131:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg87]]:vgpr_32, 16, killed [[Reg84]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg132:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg93]]:vgpr_32, 16, killed [[Reg90]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg133:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg99]]:vgpr_32, 16, killed [[Reg96]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg134:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg105]]:vgpr_32, 16, killed [[Reg102]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg135:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg120]]:sreg_32, %bb.13, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.6
; EMPTY:
; CHECK: bb.6.bb6:
; CHECK-NEXT: ; predecessors: %bb.5
; CHECK-NEXT:   successors: %bb.9(0x40000000), %bb.7(0x40000000); %bb.9(50.00%), %bb.7(50.00%)
; EMPTY:
; CHECK:   [[Reg136:%[0-9]+]]:sreg_32 = S_AND_B32 killed [[Reg24]]:sreg_32, killed [[Reg26]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg137:%[0-9]+]]:sreg_32 = S_XOR_B32 killed [[Reg136]]:sreg_32, -1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg138:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array2, target-flags(amdgpu-gotprel32-hi) @array2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg139:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg138]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg140:%[0-9]+]]:vreg_64 = COPY killed [[Reg139]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg141:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg140]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array2, i64 20)`)
; CHECK-NEXT:   [[Reg142:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array4, target-flags(amdgpu-gotprel32-hi) @array4, implicit-def dead $scc
; CHECK-NEXT:   [[Reg143:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg142]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg144:%[0-9]+]]:vreg_64 = COPY [[Reg143]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD killed [[Reg144]]:vreg_64, killed [[Reg141]]:vgpr_32, 4, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array4, i64 4)`)
; CHECK-NEXT:   [[Reg145:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg137]]:sreg_32, %bb.7, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.9
; EMPTY:
; CHECK: bb.7.Flow16:
; CHECK-NEXT: ; predecessors: %bb.6, %bb.9
; CHECK-NEXT:   successors: %bb.8(0x40000000), %bb.11(0x40000000); %bb.8(50.00%), %bb.11(50.00%)
; EMPTY:
; CHECK:   [[Reg146:%[0-9]+]]:vgpr_32 = PHI undef [[Reg147:%[0-9]+]]:vgpr_32, %bb.6, [[Reg148:%[0-9]+]]:vgpr_32, %bb.9
; CHECK-NEXT:   [[Reg149:%[0-9]+]]:vgpr_32 = PHI [[Reg124]]:vgpr_32, %bb.6, undef [[Reg150:%[0-9]+]]:vgpr_32, %bb.9
; CHECK-NEXT:   [[Reg151:%[0-9]+]]:vgpr_32 = PHI [[Reg126]]:vgpr_32, %bb.6, undef [[Reg152:%[0-9]+]]:vgpr_32, %bb.9
; CHECK-NEXT:   [[Reg153:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg145]]:sreg_32, %bb.11, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.8
; EMPTY:
; CHECK: bb.8.bb8:
; CHECK-NEXT: ; predecessors: %bb.7
; CHECK-NEXT:   successors: %bb.11(0x80000000); %bb.11(100.00%)
; EMPTY:
; CHECK:   [[Reg154:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg149]]:vgpr_32, killed [[Reg151]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg155:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array2, target-flags(amdgpu-gotprel32-hi) @array2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg156:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg155]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg157:%[0-9]+]]:vreg_64 = COPY killed [[Reg156]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg158:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg157]]:vreg_64, 28, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array2, i64 28)`)
; CHECK-NEXT:   [[Reg159:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array3, target-flags(amdgpu-gotprel32-hi) @array3, implicit-def dead $scc
; CHECK-NEXT:   [[Reg160:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg159]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg161:%[0-9]+]]:vreg_64 = COPY killed [[Reg160]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD killed [[Reg161]]:vreg_64, killed [[Reg158]]:vgpr_32, 68, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 68)`)
; CHECK-NEXT:   S_BRANCH %bb.11
; EMPTY:
; CHECK: bb.9.bb9:
; CHECK-NEXT: ; predecessors: %bb.6
; CHECK-NEXT:   successors: %bb.7(0x80000000); %bb.7(100.00%)
; EMPTY:
; CHECK:   [[Reg148]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg124]]:vgpr_32, killed [[Reg126]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg162:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array5, target-flags(amdgpu-gotprel32-hi) @array5, implicit-def dead $scc
; CHECK-NEXT:   [[Reg163:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg162]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg164:%[0-9]+]]:vreg_64 = COPY killed [[Reg163]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg165:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg164]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array5, i64 20)`)
; CHECK-NEXT:   [[Reg166:%[0-9]+]]:vreg_64 = COPY killed [[Reg143]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD killed [[Reg166]]:vreg_64, killed [[Reg165]]:vgpr_32, 60, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array4, i64 60)`)
; CHECK-NEXT:   S_BRANCH %bb.7
; EMPTY:
; CHECK: bb.10.bb7:
; CHECK-NEXT: ; predecessors: %bb.4
; CHECK-NEXT:   successors: %bb.5(0x80000000); %bb.5(100.00%)
; EMPTY:
; CHECK:   [[Reg167:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg75]]:vgpr_32, killed [[Reg81]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg168:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg167]]:vgpr_32, [[Reg78]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg169:%[0-9]+]]:vgpr_32 = V_CVT_F32_U32_e64 [[Reg80]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg170:%[0-9]+]]:vgpr_32 = nofpexcept V_RCP_IFLAG_F32_e64 0, killed [[Reg169]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg171:%[0-9]+]]:vgpr_32 = nnan ninf nsz arcp contract afn reassoc nofpexcept V_MUL_F32_e64 0, 1333788670, 0, killed [[Reg170]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg172:%[0-9]+]]:vgpr_32 = nofpexcept V_CVT_U32_F32_e64 0, killed [[Reg171]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg173:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 0, [[Reg80]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg174:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg173]]:vgpr_32, [[Reg172]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg175:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg172]]:vgpr_32, killed [[Reg174]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg176:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg172]]:vgpr_32, killed [[Reg175]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg177:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg168]]:vgpr_32, killed [[Reg176]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg178:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg177]]:vgpr_32, [[Reg80]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg179:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg168]]:vgpr_32, killed [[Reg178]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg180:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 [[Reg179]]:vgpr_32, [[Reg80]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg181:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg177]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg182:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg177]]:vgpr_32, 0, killed [[Reg181]]:vgpr_32, [[Reg180]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg183:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg179]]:vgpr_32, [[Reg80]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg184:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg179]]:vgpr_32, 0, killed [[Reg183]]:vgpr_32, killed [[Reg180]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg185:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 killed [[Reg184]]:vgpr_32, killed [[Reg80]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg186:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg182]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg187:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg182]]:vgpr_32, 0, killed [[Reg186]]:vgpr_32, killed [[Reg185]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg123]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg187]]:vgpr_32, killed [[Reg69]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg188:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array3, target-flags(amdgpu-gotprel32-hi) @array3, implicit-def dead $scc
; CHECK-NEXT:   [[Reg189:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg188]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg190:%[0-9]+]]:vreg_64 = COPY killed [[Reg189]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg191:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg190]]:vreg_64, 84, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 84)`)
; CHECK-NEXT:   [[Reg192:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array4, target-flags(amdgpu-gotprel32-hi) @array4, implicit-def dead $scc
; CHECK-NEXT:   [[Reg193:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg192]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg194:%[0-9]+]]:vreg_64 = COPY killed [[Reg193]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD killed [[Reg194]]:vreg_64, killed [[Reg191]]:vgpr_32, 60, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array4, i64 60)`)
; CHECK-NEXT:   S_BRANCH %bb.5
; EMPTY:
; CHECK: bb.11.Flow17:
; CHECK-NEXT: ; predecessors: %bb.7, %bb.8
; CHECK-NEXT:   successors: %bb.13(0x80000000); %bb.13(100.00%)
; EMPTY:
; CHECK:   [[Reg195:%[0-9]+]]:vgpr_32 = PHI [[Reg146]]:vgpr_32, %bb.7, [[Reg154]]:vgpr_32, %bb.8
; CHECK-NEXT:   SI_END_CF killed [[Reg153]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.13
; EMPTY:
; CHECK: bb.12.Flow20:
; CHECK-NEXT: ; predecessors: %bb.18, %bb.17
; CHECK-NEXT:   successors: %bb.21(0x80000000); %bb.21(100.00%)
; EMPTY:
; CHECK:   [[Reg196:%[0-9]+]]:vgpr_32 = PHI [[Reg197:%[0-9]+]]:vgpr_32, %bb.18, [[Reg198:%[0-9]+]]:vgpr_32, %bb.17
; CHECK-NEXT:   [[Reg199:%[0-9]+]]:vgpr_32 = PHI [[Reg200:%[0-9]+]]:vgpr_32, %bb.18, [[Reg198]]:vgpr_32, %bb.17
; CHECK-NEXT:   [[Reg201:%[0-9]+]]:vgpr_32 = PHI [[Reg202:%[0-9]+]]:vgpr_32, %bb.18, [[Reg203:%[0-9]+]]:vgpr_32, %bb.17
; CHECK-NEXT:   [[Reg204:%[0-9]+]]:vgpr_32 = PHI [[Reg205:%[0-9]+]]:vgpr_32, %bb.18, [[Reg206:%[0-9]+]]:vgpr_32, %bb.17
; CHECK-NEXT:   [[Reg207:%[0-9]+]]:vgpr_32 = PHI [[Reg208:%[0-9]+]]:vgpr_32, %bb.18, [[Reg209:%[0-9]+]]:vgpr_32, %bb.17
; CHECK-NEXT:   [[Reg210:%[0-9]+]]:vgpr_32 = PHI [[Reg211:%[0-9]+]]:vgpr_32, %bb.18, [[Reg212:%[0-9]+]]:vgpr_32, %bb.17
; CHECK-NEXT:   [[Reg213:%[0-9]+]]:vgpr_32 = PHI [[Reg214:%[0-9]+]]:vgpr_32, %bb.18, [[Reg198]]:vgpr_32, %bb.17
; CHECK-NEXT:   [[Reg215:%[0-9]+]]:vgpr_32 = PHI [[Reg216:%[0-9]+]]:vgpr_32, %bb.18, [[Reg217:%[0-9]+]]:vgpr_32, %bb.17
; CHECK-NEXT:   SI_END_CF killed [[Reg218:%[0-9]+]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg219:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg18]]:vgpr_32, %subreg.sub0, killed [[Reg19]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg220:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg16]]:vgpr_32, %subreg.sub0, killed [[Reg17]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   S_BRANCH %bb.21
; EMPTY:
; CHECK: bb.13.bb10:
; CHECK-NEXT: ; predecessors: %bb.5, %bb.11
; CHECK-NEXT:   successors: %bb.16(0x40000000), %bb.14(0x40000000); %bb.16(50.00%), %bb.14(50.00%)
; EMPTY:
; CHECK:   [[Reg221:%[0-9]+]]:vgpr_32 = PHI [[Reg121]]:vgpr_32, %bb.5, [[Reg195]]:vgpr_32, %bb.11
; CHECK-NEXT:   SI_END_CF killed [[Reg135]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg222:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 100, killed [[Reg78]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg223:%[0-9]+]]:vgpr_32 = COPY [[Reg129]].sub1:vreg_128
; CHECK-NEXT:   [[Reg224:%[0-9]+]]:vgpr_32 = COPY [[Reg129]].sub2:vreg_128
; CHECK-NEXT:   [[Reg225:%[0-9]+]]:vgpr_32 = COPY [[Reg130]].sub2:vreg_128
; CHECK-NEXT:   [[Reg226:%[0-9]+]]:vgpr_32 = COPY [[Reg130]].sub3:vreg_128
; CHECK-NEXT:   [[Reg227:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 [[Reg129]].sub1:vreg_128, killed [[Reg130]].sub3:vreg_128, [[Reg130]].sub2:vreg_128, implicit $exec
; CHECK-NEXT:   [[Reg228:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
; CHECK-NEXT:   [[Reg229:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg129]].sub2:vreg_128, %subreg.sub0, [[Reg228]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg230:%[0-9]+]]:vreg_64 = nsw V_LSHLREV_B64_pseudo_e64 2, killed [[Reg229]]:vreg_64, implicit $exec
; CHECK-NEXT:   [[Reg231:%[0-9]+]]:vgpr_32, [[Reg232:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg233:%[0-9]+]].sub0:vreg_64, [[Reg230]].sub0:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg234:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg233]].sub1:vreg_64, killed [[Reg230]].sub1:vreg_64, killed [[Reg232]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg235:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg231]]:vgpr_32, %subreg.sub0, killed [[Reg234]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg235]]:vreg_64, killed [[Reg227]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.arrayidx1, addrspace 1)
; CHECK-NEXT:   [[Reg236:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg119]]:sreg_32, %bb.14, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.16
; EMPTY:
; CHECK: bb.14.Flow:
; CHECK-NEXT: ; predecessors: %bb.13, %bb.16
; CHECK-NEXT:   successors: %bb.15(0x40000000), %bb.17(0x40000000); %bb.15(50.00%), %bb.17(50.00%)
; EMPTY:
; CHECK:   [[Reg237:%[0-9]+]]:vgpr_32 = PHI undef [[Reg238:%[0-9]+]]:vgpr_32, %bb.13, [[Reg239:%[0-9]+]]:vgpr_32, %bb.16
; CHECK-NEXT:   [[Reg240:%[0-9]+]]:vgpr_32 = PHI undef [[Reg238]]:vgpr_32, %bb.13, [[Reg241:%[0-9]+]]:vgpr_32, %bb.16
; CHECK-NEXT:   [[Reg242:%[0-9]+]]:vgpr_32 = PHI undef [[Reg238]]:vgpr_32, %bb.13, [[Reg243:%[0-9]+]]:vgpr_32, %bb.16
; CHECK-NEXT:   [[Reg244:%[0-9]+]]:vgpr_32 = PHI [[Reg132]]:vgpr_32, %bb.13, undef [[Reg245:%[0-9]+]]:vgpr_32, %bb.16
; CHECK-NEXT:   [[Reg246:%[0-9]+]]:vgpr_32 = PHI [[Reg131]]:vgpr_32, %bb.13, undef [[Reg247:%[0-9]+]]:vgpr_32, %bb.16
; CHECK-NEXT:   [[Reg248:%[0-9]+]]:vgpr_32 = PHI [[Reg249:%[0-9]+]]:vgpr_32, %bb.13, undef [[Reg250:%[0-9]+]]:vgpr_32, %bb.16
; CHECK-NEXT:   [[Reg251:%[0-9]+]]:vgpr_32 = PHI [[Reg252:%[0-9]+]]:vgpr_32, %bb.13, undef [[Reg253:%[0-9]+]]:vgpr_32, %bb.16
; CHECK-NEXT:   [[Reg254:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg255:%[0-9]+]]:vgpr_32, %subreg.sub0, killed [[Reg256:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg257:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg236]]:sreg_32, %bb.17, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.15
; EMPTY:
; CHECK: bb.15.bb11:
; CHECK-NEXT: ; predecessors: %bb.14
; CHECK-NEXT:   successors: %bb.17(0x80000000); %bb.17(100.00%)
; EMPTY:
; CHECK:   [[Reg258:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg244]]:vgpr_32, [[Reg221]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg259:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
; CHECK-NEXT:   [[Reg260:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg246]]:vgpr_32, %subreg.sub0, killed [[Reg259]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg261:%[0-9]+]]:vreg_64 = nsw V_LSHLREV_B64_pseudo_e64 2, killed [[Reg260]]:vreg_64, implicit $exec
; CHECK-NEXT:   [[Reg262:%[0-9]+]]:vgpr_32, [[Reg263:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 killed [[Reg248]]:vgpr_32, [[Reg261]].sub0:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg264:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg251]]:vgpr_32, killed [[Reg261]].sub1:vreg_64, killed [[Reg263]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg265:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg262]]:vgpr_32, %subreg.sub0, killed [[Reg264]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg265]]:vreg_64, [[Reg258]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.arrayidx2, align 8, addrspace 1)
; CHECK-NEXT:   S_BRANCH %bb.17
; EMPTY:
; CHECK: bb.16.bb12:
; CHECK-NEXT: ; predecessors: %bb.13
; CHECK-NEXT:   successors: %bb.14(0x80000000); %bb.14(100.00%)
; EMPTY:
; CHECK:   [[Reg266:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 [[Reg222]]:vgpr_32, [[Reg221]]:vgpr_32, [[Reg128]].sub3:vreg_256, implicit $exec
; CHECK-NEXT:   [[Reg267:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg128]].sub5:vreg_256, %subreg.sub0, [[Reg228]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg268:%[0-9]+]]:vreg_64 = nsw V_LSHLREV_B64_pseudo_e64 2, killed [[Reg267]]:vreg_64, implicit $exec
; CHECK-NEXT:   [[Reg269:%[0-9]+]]:vgpr_32, [[Reg270:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 killed [[Reg249]]:vgpr_32, [[Reg268]].sub0:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg271:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg252]]:vgpr_32, killed [[Reg268]].sub1:vreg_64, killed [[Reg270]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg272:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg269]]:vgpr_32, %subreg.sub0, killed [[Reg271]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   GLOBAL_STORE_SHORT_D16_HI [[Reg272]]:vreg_64, [[Reg266]]:vgpr_32, 2, 0, implicit $exec :: (store (s16) into %ir.arrayidx3 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_SHORT killed [[Reg272]]:vreg_64, [[Reg266]]:vgpr_32, 0, 0, implicit $exec :: (store (s16) into %ir.arrayidx3, addrspace 1)
; CHECK-NEXT:   [[Reg273:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg134]]:vgpr_32, [[Reg266]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg274:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg133]]:vgpr_32, %subreg.sub0, killed [[Reg228]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg275:%[0-9]+]]:vreg_64 = nsw V_LSHLREV_B64_pseudo_e64 2, killed [[Reg274]]:vreg_64, implicit $exec
; CHECK-NEXT:   [[Reg276:%[0-9]+]]:vgpr_32, [[Reg277:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 killed [[Reg64]]:vgpr_32, [[Reg275]].sub0:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg278:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg65]]:vgpr_32, killed [[Reg275]].sub1:vreg_64, killed [[Reg277]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg279:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg276]]:vgpr_32, %subreg.sub0, killed [[Reg278]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   GLOBAL_STORE_SHORT_D16_HI [[Reg279]]:vreg_64, [[Reg273]]:vgpr_32, 2, 0, implicit $exec :: (store (s16) into %ir.arrayidx5 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_SHORT killed [[Reg279]]:vreg_64, killed [[Reg273]]:vgpr_32, 0, 0, implicit $exec :: (store (s16) into %ir.arrayidx5, addrspace 1)
; CHECK-NEXT:   ADJCALLSTACKUP 0, 0, implicit-def dead $scc, implicit-def $sgpr32, implicit $sgpr32
; CHECK-NEXT:   [[Reg280:%[0-9]+]]:sreg_32_xexec_hi = COPY $sgpr32
; CHECK-NEXT:   [[Reg281:%[0-9]+]]:sreg_32 = S_ADD_I32 [[Reg280]]:sreg_32_xexec_hi, 1024, implicit-def dead $scc
; CHECK-NEXT:   $sgpr32 = COPY killed [[Reg281]]:sreg_32
; CHECK-NEXT:   ADJCALLSTACKDOWN 0, 0, implicit-def dead $scc, implicit-def $sgpr32, implicit $sgpr32
; CHECK-NEXT:   [[Reg282:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD killed [[Reg63]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p6, addrspace 1)
; CHECK-NEXT:   [[Reg283:%[0-9]+]]:vgpr_32 = V_LSHLREV_B32_e64 2, killed [[Reg282]]:vgpr_32, implicit $exec
; CHECK-NEXT:   SCRATCH_STORE_DWORD_SVS killed [[Reg266]]:vgpr_32, killed [[Reg283]]:vgpr_32, killed [[Reg280]]:sreg_32_xexec_hi, 40, 0, implicit $exec, implicit $flat_scr :: (store (s32) into %ir.arrayidx1111, addrspace 5)
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg66]]:vreg_64, [[Reg79]].sub0:vreg_64, 4, 0, implicit $exec :: (store (s32) into %ir.arrayidx444, addrspace 1)
; CHECK-NEXT:   [[Reg284:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array4, target-flags(amdgpu-gotprel32-hi) @array4, implicit-def dead $scc
; CHECK-NEXT:   [[Reg285:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg284]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg286:%[0-9]+]]:vreg_64 = COPY killed [[Reg285]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg241]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg286]]:vreg_64, 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from @array4)
; CHECK-NEXT:   [[Reg287:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array2, target-flags(amdgpu-gotprel32-hi) @array2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg288:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg287]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg289:%[0-9]+]]:vreg_64 = COPY killed [[Reg288]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg290:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg289]]:vreg_64, 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from @array2)
; CHECK-NEXT:   [[Reg291:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array3, target-flags(amdgpu-gotprel32-hi) @array3, implicit-def dead $scc
; CHECK-NEXT:   [[Reg292:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg291]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg293:%[0-9]+]]:vreg_64 = COPY killed [[Reg292]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg294:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg293]]:vreg_64, 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from @array3)
; CHECK-NEXT:   [[Reg295:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array5, target-flags(amdgpu-gotprel32-hi) @array5, implicit-def dead $scc
; CHECK-NEXT:   [[Reg296:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg295]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg297:%[0-9]+]]:vreg_64 = COPY killed [[Reg296]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg239]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg297]]:vreg_64, 0, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from @array5)
; CHECK-NEXT:   [[Reg298:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg294]]:vgpr_32, [[Reg239]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg299:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg298]]:vgpr_32, %subreg.sub0, undef [[Reg300:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg301:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 [[Reg241]]:vgpr_32, killed [[Reg290]]:vgpr_32, killed [[Reg299]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg243]]:vgpr_32 = COPY killed [[Reg301]].sub0:vreg_64
; CHECK-NEXT:   S_BRANCH %bb.14
; EMPTY:
; CHECK: bb.17.bb13:
; CHECK-NEXT: ; predecessors: %bb.14, %bb.15
; CHECK-NEXT:   successors: %bb.12(0x80000000); %bb.12(100.00%)
; EMPTY:
; CHECK:   [[Reg302:%[0-9]+]]:vgpr_32 = PHI [[Reg242]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   [[Reg303:%[0-9]+]]:vgpr_32 = PHI [[Reg240]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   [[Reg304:%[0-9]+]]:vgpr_32 = PHI [[Reg115]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   [[Reg305:%[0-9]+]]:vgpr_32 = PHI [[Reg114]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   [[Reg306:%[0-9]+]]:vgpr_32 = PHI [[Reg237]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   [[Reg307:%[0-9]+]]:vgpr_32 = PHI [[Reg223]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   [[Reg308:%[0-9]+]]:vgpr_32 = PHI [[Reg224]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   [[Reg309:%[0-9]+]]:vgpr_32 = PHI [[Reg225]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   [[Reg310:%[0-9]+]]:vgpr_32 = PHI [[Reg226]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   [[Reg311:%[0-9]+]]:vgpr_32 = PHI [[Reg108]]:vgpr_32, %bb.14, [[Reg258]]:vgpr_32, %bb.15
; CHECK-NEXT:   SI_END_CF killed [[Reg257]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg217]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg302]]:vgpr_32, killed [[Reg221]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg209]]:vgpr_32 = V_SUB_U32_e64 [[Reg45]]:vgpr_32, killed [[Reg312:%[0-9]+]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg313:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg217]]:vgpr_32, [[Reg209]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg198]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg313]]:vgpr_32, killed [[Reg222]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg212]]:vgpr_32 = V_ADD_U32_e64 [[Reg198]]:vgpr_32, killed [[Reg79]].sub0:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg314:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg212]]:vgpr_32, killed [[Reg75]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg315:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg314]]:vgpr_32, killed [[Reg303]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg316:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg315]]:vgpr_32, killed [[Reg304]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg206]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg316]]:vgpr_32, killed [[Reg305]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg317:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg307]]:vgpr_32, %subreg.sub0, undef [[Reg318:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg319:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 [[Reg206]]:vgpr_32, killed [[Reg306]]:vgpr_32, killed [[Reg317]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg320:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg319]].sub0:vreg_64, killed [[Reg308]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg321:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg320]]:vgpr_32, killed [[Reg309]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg322:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg321]]:vgpr_32, killed [[Reg310]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg203]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg322]]:vgpr_32, killed [[Reg311]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_SHORT_D16_HI [[Reg254]]:vreg_64, [[Reg203]]:vgpr_32, 2, 0, implicit $exec :: (store (s16) into %ir.p7 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_SHORT killed [[Reg254]]:vreg_64, [[Reg203]]:vgpr_32, 0, 0, implicit $exec :: (store (s16) into %ir.p7, addrspace 1)
; CHECK-NEXT:   S_BRANCH %bb.12
; EMPTY:
; CHECK: bb.18.Flow19:
; CHECK-NEXT: ; predecessors: %bb.2, %bb.20
; CHECK-NEXT:   successors: %bb.4(0x40000000), %bb.12(0x40000000); %bb.4(50.00%), %bb.12(50.00%)
; EMPTY:
; CHECK:   [[Reg197]]:vgpr_32 = PHI undef [[Reg323:%[0-9]+]]:vgpr_32, %bb.2, [[Reg324:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg200]]:vgpr_32 = PHI undef [[Reg323]]:vgpr_32, %bb.2, [[Reg325:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg202]]:vgpr_32 = PHI undef [[Reg323]]:vgpr_32, %bb.2, [[Reg326:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg205]]:vgpr_32 = PHI undef [[Reg323]]:vgpr_32, %bb.2, [[Reg327:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg208]]:vgpr_32 = PHI undef [[Reg323]]:vgpr_32, %bb.2, [[Reg328:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg211]]:vgpr_32 = PHI undef [[Reg323]]:vgpr_32, %bb.2, [[Reg329:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg214]]:vgpr_32 = PHI undef [[Reg323]]:vgpr_32, %bb.2, [[Reg330:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg216]]:vgpr_32 = PHI undef [[Reg323]]:vgpr_32, %bb.2, [[Reg331:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg80]]:vgpr_32 = PHI [[Reg34]]:vgpr_32, %bb.2, undef [[Reg332:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg110]]:vgpr_32 = PHI [[Reg41]]:vgpr_32, %bb.2, undef [[Reg333:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg312]]:vgpr_32 = PHI [[Reg42]]:vgpr_32, %bb.2, undef [[Reg334:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg71]]:vgpr_32 = PHI [[Reg3]]:vgpr_32, %bb.2, undef [[Reg335:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg67]]:vgpr_32 = PHI [[Reg8]]:vgpr_32, %bb.2, undef [[Reg336:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg68]]:vgpr_32 = PHI [[Reg9]]:vgpr_32, %bb.2, undef [[Reg337:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg249]]:vgpr_32 = PHI [[Reg10]]:vgpr_32, %bb.2, undef [[Reg338:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg252]]:vgpr_32 = PHI [[Reg11]]:vgpr_32, %bb.2, undef [[Reg339:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg64]]:vgpr_32 = PHI [[Reg12]]:vgpr_32, %bb.2, undef [[Reg340:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg65]]:vgpr_32 = PHI [[Reg13]]:vgpr_32, %bb.2, undef [[Reg341:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg255]]:vgpr_32 = PHI [[Reg14]]:vgpr_32, %bb.2, undef [[Reg342:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg256]]:vgpr_32 = PHI [[Reg15]]:vgpr_32, %bb.2, undef [[Reg343:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg59]]:vgpr_32 = PHI [[Reg35]]:vgpr_32, %bb.2, undef [[Reg344:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg58]]:vgpr_32 = PHI [[Reg36]]:vgpr_32, %bb.2, undef [[Reg345:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg62]]:vgpr_32 = PHI [[Reg37]]:vgpr_32, %bb.2, undef [[Reg346:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg61]]:vgpr_32 = PHI [[Reg38]]:vgpr_32, %bb.2, undef [[Reg347:%[0-9]+]]:vgpr_32, %bb.20
; CHECK-NEXT:   [[Reg130]]:vreg_128 = PHI [[Reg39]]:vreg_128, %bb.2, undef [[Reg348:%[0-9]+]]:vreg_128, %bb.20
; CHECK-NEXT:   [[Reg129]]:vreg_128 = PHI [[Reg40]]:vreg_128, %bb.2, undef [[Reg349:%[0-9]+]]:vreg_128, %bb.20
; CHECK-NEXT:   [[Reg233]]:vreg_64 = PHI [[Reg22]]:vreg_64, %bb.2, undef [[Reg350:%[0-9]+]]:vreg_64, %bb.20
; CHECK-NEXT:   [[Reg218]]:sreg_32 = SI_ELSE killed [[Reg46]]:sreg_32, %bb.12, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.19.bb4:
; CHECK-NEXT: ; predecessors: %bb.3, %bb.19
; CHECK-NEXT:   successors: %bb.20(0x04000000), %bb.19(0x7c000000); %bb.20(3.12%), %bb.19(96.88%)
; EMPTY:
; CHECK:   [[Reg351:%[0-9]+]]:sreg_32 = PHI [[Reg56]]:sreg_32, %bb.3, [[Reg352:%[0-9]+]]:sreg_32, %bb.19
; CHECK-NEXT:   [[Reg353:%[0-9]+]]:sreg_32 = PHI [[Reg55]]:sreg_32, %bb.3, [[Reg354:%[0-9]+]]:sreg_32, %bb.19
; CHECK-NEXT:   [[Reg354]]:sreg_32 = S_ADD_I32 [[Reg353]]:sreg_32, 1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg355:%[0-9]+]]:sreg_32 = S_ADD_I32 killed [[Reg353]]:sreg_32, 2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg356:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 killed [[Reg355]]:sreg_32, [[Reg20]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg352]]:sreg_32 = SI_IF_BREAK killed [[Reg356]]:sreg_32, killed [[Reg351]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg357:%[0-9]+]]:vgpr_32 = COPY [[Reg354]]:sreg_32, implicit $exec
; CHECK-NEXT:   SI_LOOP [[Reg352]]:sreg_32, %bb.19, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.20
; EMPTY:
; CHECK: bb.20.bb14.loopexit:
; CHECK-NEXT: ; predecessors: %bb.19
; CHECK-NEXT:   successors: %bb.18(0x80000000); %bb.18(100.00%)
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg352]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg358:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg50]]:vgpr_32, [[Reg357]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg359:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg54]]:vgpr_32, [[Reg357]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg360:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array7, target-flags(amdgpu-gotprel32-hi) @array7, implicit-def dead $scc
; CHECK-NEXT:   [[Reg361:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg360]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg362:%[0-9]+]]:vreg_64 = COPY killed [[Reg361]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD [[Reg362]]:vreg_64, killed [[Reg358]]:vgpr_32, 68, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array7, i64 68)`)
; CHECK-NEXT:   [[Reg363:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array9, target-flags(amdgpu-gotprel32-hi) @array9, implicit-def dead $scc
; CHECK-NEXT:   [[Reg364:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg363]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg365:%[0-9]+]]:vreg_64 = COPY killed [[Reg364]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD [[Reg365]]:vreg_64, killed [[Reg359]]:vgpr_32, 60, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array9, i64 60)`)
; CHECK-NEXT:   [[Reg366:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array6, target-flags(amdgpu-gotprel32-hi) @array6, implicit-def dead $scc
; CHECK-NEXT:   [[Reg367:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg366]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg368:%[0-9]+]]:vreg_64 = COPY killed [[Reg367]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg369:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg368]]:vreg_64, 44, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array6, i64 44)`)
; CHECK-NEXT:   [[Reg331]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg369]]:vgpr_32, [[Reg357]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg370:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg362]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array7, i64 20)`)
; CHECK-NEXT:   [[Reg330]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg370]]:vgpr_32, [[Reg357]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg371:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array8, target-flags(amdgpu-gotprel32-hi) @array8, implicit-def dead $scc
; CHECK-NEXT:   [[Reg372:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg371]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg373:%[0-9]+]]:vreg_64 = COPY killed [[Reg372]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg374:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg373]]:vreg_64, 44, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array8, i64 44)`, align 8)
; CHECK-NEXT:   [[Reg329]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg374]]:vgpr_32, [[Reg357]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg375:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg365]]:vreg_64, 24, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array9, i64 24)`)
; CHECK-NEXT:   [[Reg328]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg375]]:vgpr_32, [[Reg357]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg376:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array2, target-flags(amdgpu-gotprel32-hi) @array2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg377:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg376]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg378:%[0-9]+]]:vreg_64 = COPY killed [[Reg377]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg379:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg378]]:vreg_64, 80, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array2, i64 80)`)
; CHECK-NEXT:   [[Reg327]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg379]]:vgpr_32, [[Reg357]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg380:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array3, target-flags(amdgpu-gotprel32-hi) @array3, implicit-def dead $scc
; CHECK-NEXT:   [[Reg381:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg380]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg382:%[0-9]+]]:vreg_64 = COPY killed [[Reg381]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg383:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg382]]:vreg_64, 80, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 80)`)
; CHECK-NEXT:   [[Reg326]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg383]]:vgpr_32, [[Reg357]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg384:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array4, target-flags(amdgpu-gotprel32-hi) @array4, implicit-def dead $scc
; CHECK-NEXT:   [[Reg385:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg384]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg386:%[0-9]+]]:vreg_64 = COPY killed [[Reg385]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg387:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg386]]:vreg_64, 80, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array4, i64 80)`, align 8)
; CHECK-NEXT:   [[Reg325]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg387]]:vgpr_32, [[Reg357]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg388:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array5, target-flags(amdgpu-gotprel32-hi) @array5, implicit-def dead $scc
; CHECK-NEXT:   [[Reg389:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg388]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg390:%[0-9]+]]:vreg_64 = COPY killed [[Reg389]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg391:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg390]]:vreg_64, 80, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array5, i64 80)`)
; CHECK-NEXT:   [[Reg324]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg391]]:vgpr_32, killed [[Reg357]]:vgpr_32, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.18
; EMPTY:
; CHECK: bb.21.bb14:
; CHECK-NEXT: ; predecessors: %bb.12
; EMPTY:
; CHECK:   [[Reg392:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg215]]:vgpr_32, killed [[Reg45]]:vgpr_32, killed [[Reg213]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg393:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg392]]:vgpr_32, killed [[Reg210]]:vgpr_32, killed [[Reg207]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg394:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg393]]:vgpr_32, killed [[Reg204]]:vgpr_32, 100, implicit $exec
; CHECK-NEXT:   [[Reg395:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg394]]:vgpr_32, killed [[Reg201]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg396:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg395]]:vgpr_32, killed [[Reg199]]:vgpr_32, killed [[Reg196]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg220]]:vreg_64, killed [[Reg396]]:vgpr_32, 4, 0, implicit $exec :: (store (s32) into %ir.gep3, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg219]]:vreg_64, killed [[Reg394]]:vgpr_32, 4, 0, implicit $exec :: (store (s32) into %ir.gep4, addrspace 1)
; CHECK-NEXT:   S_ENDPGM 0
; EMPTY:
; CHECK: # End machine code for function test.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg20]] = 64.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 86.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 85.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 85.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 84.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 63.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 61.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 59.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 57.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 55.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 53.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 51.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 49.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 43.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 18.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 20.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 21.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 35.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 35.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 35.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 35.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 35.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 45.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 9009.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 9006.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 47.0
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg69]] = 58.0
; CHECK-NEXT: Next-use distance of Register [[Reg70]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg72]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg73]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg74]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg75]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg76]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg77]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg78]] = 50.0
; CHECK-NEXT: Next-use distance of Register [[Reg79]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg81]] = 47.0
; CHECK-NEXT: Next-use distance of Register [[Reg82]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg83]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg84]] = 48.0
; CHECK-NEXT: Next-use distance of Register [[Reg85]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg86]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg87]] = 45.0
; CHECK-NEXT: Next-use distance of Register [[Reg88]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg89]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg90]] = 43.0
; CHECK-NEXT: Next-use distance of Register [[Reg91]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg92]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg93]] = 40.0
; CHECK-NEXT: Next-use distance of Register [[Reg94]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg95]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg96]] = 38.0
; CHECK-NEXT: Next-use distance of Register [[Reg97]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg98]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg99]] = 35.0
; CHECK-NEXT: Next-use distance of Register [[Reg100]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg101]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg102]] = 33.0
; CHECK-NEXT: Next-use distance of Register [[Reg103]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg104]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg105]] = 30.0
; CHECK-NEXT: Next-use distance of Register [[Reg106]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg107]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg108]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg109]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg111]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg112]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg113]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg114]] = 51.0
; CHECK-NEXT: Next-use distance of Register [[Reg115]] = 49.0
; CHECK-NEXT: Next-use distance of Register [[Reg116]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg117]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg118]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg119]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg120]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg121]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg124]] = 21.0
; CHECK-NEXT: Next-use distance of Register [[Reg126]] = 20.0
; CHECK-NEXT: Next-use distance of Register [[Reg128]] = 24.0
; CHECK-NEXT: Next-use distance of Register [[Reg131]] = 27.0
; CHECK-NEXT: Next-use distance of Register [[Reg132]] = 25.0
; CHECK-NEXT: Next-use distance of Register [[Reg133]] = 30.0
; CHECK-NEXT: Next-use distance of Register [[Reg134]] = 28.0
; CHECK-NEXT: Next-use distance of Register [[Reg135]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg136]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg137]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg138]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg139]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg140]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg141]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg142]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg143]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg144]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg145]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg146]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg149]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg151]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg153]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg154]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg155]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg156]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg157]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg158]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg159]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg160]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg161]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg148]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg162]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg163]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg164]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg165]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg166]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg167]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg168]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg169]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg170]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg171]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg172]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg173]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg174]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg175]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg176]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg177]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg178]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg179]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg180]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg181]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg182]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg183]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg184]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg185]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg186]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg187]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg123]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg188]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg189]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg190]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg191]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg192]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg193]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg194]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg195]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg196]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg199]] = 15.0
; CHECK-NEXT: Next-use distance of Register [[Reg201]] = 13.0
; CHECK-NEXT: Next-use distance of Register [[Reg204]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg207]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg210]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg213]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg215]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg219]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg220]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg221]] = 17.0
; CHECK-NEXT: Next-use distance of Register [[Reg222]] = 15.0
; CHECK-NEXT: Next-use distance of Register [[Reg223]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg224]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg225]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg226]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg227]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg228]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg229]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg230]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg231]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg232]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg234]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg235]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg236]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg237]] = 14.0
; CHECK-NEXT: Next-use distance of Register [[Reg240]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg242]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg244]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg246]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg248]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg251]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg254]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg257]] = 12.0
; CHECK-NEXT: Next-use distance of Register [[Reg258]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg259]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg260]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg261]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg262]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg263]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg264]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg265]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg266]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg267]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg268]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg269]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg270]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg271]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg272]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg273]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg274]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg275]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg276]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg277]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg278]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg279]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg280]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg281]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg282]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg283]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg284]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg285]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg286]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg241]] = 15.0
; CHECK-NEXT: Next-use distance of Register [[Reg287]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg288]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg289]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg290]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg291]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg292]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg293]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg294]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg295]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg296]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg297]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg239]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg298]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg299]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg301]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg243]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg302]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg303]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg304]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg305]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg306]] = 17.0
; CHECK-NEXT: Next-use distance of Register [[Reg307]] = 15.0
; CHECK-NEXT: Next-use distance of Register [[Reg308]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg309]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg310]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg311]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg217]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg209]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg313]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg198]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg212]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg314]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg315]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg316]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg206]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg317]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg319]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg320]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg321]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg322]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg203]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg197]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg200]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg202]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg205]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg208]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg211]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg214]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg216]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg80]] = 34.0
; CHECK-NEXT: Next-use distance of Register [[Reg110]] = 62.0
; CHECK-NEXT: Next-use distance of Register [[Reg312]] = 129.0
; CHECK-NEXT: Next-use distance of Register [[Reg71]] = 23.0
; CHECK-NEXT: Next-use distance of Register [[Reg67]] = 20.0
; CHECK-NEXT: Next-use distance of Register [[Reg68]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg249]] = 106.0
; CHECK-NEXT: Next-use distance of Register [[Reg252]] = 106.0
; CHECK-NEXT: Next-use distance of Register [[Reg64]] = 15.0
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 14.0
; CHECK-NEXT: Next-use distance of Register [[Reg255]] = 106.0
; CHECK-NEXT: Next-use distance of Register [[Reg256]] = 105.0
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg58]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg130]] = 69.0
; CHECK-NEXT: Next-use distance of Register [[Reg129]] = 68.0
; CHECK-NEXT: Next-use distance of Register [[Reg233]] = 85.0
; CHECK-NEXT: Next-use distance of Register [[Reg218]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg351]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg353]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg354]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg355]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg356]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg352]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg357]] = 9002.0
; CHECK-NEXT: Next-use distance of Register [[Reg358]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg359]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg360]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg361]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg362]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg363]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg364]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg365]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg366]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg367]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg368]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg369]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg331]] = 38.0
; CHECK-NEXT: Next-use distance of Register [[Reg370]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg330]] = 35.0
; CHECK-NEXT: Next-use distance of Register [[Reg371]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg372]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg373]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg374]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg329]] = 29.0
; CHECK-NEXT: Next-use distance of Register [[Reg375]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg328]] = 26.0
; CHECK-NEXT: Next-use distance of Register [[Reg376]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg377]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg378]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg379]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg327]] = 20.0
; CHECK-NEXT: Next-use distance of Register [[Reg380]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg381]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg382]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg383]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg326]] = 14.0
; CHECK-NEXT: Next-use distance of Register [[Reg384]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg385]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg386]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg387]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg325]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg388]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg389]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg390]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg391]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg324]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg392]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg393]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg394]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg395]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg396]] = 1.0
entry:
   %ld1 = load i32, ptr addrspace(1) %p1, align 1
   %gep1 = getelementptr inbounds i32, ptr addrspace(1) %p1, i64 3
   %ld2 = load i32, ptr addrspace(1) %gep1, align 1
   %load1 = load i32, ptr addrspace(1) %p3, align 4
   %tmp1 = add i32 %load1, %ld1
   %load2 = load <8 x i32>, ptr addrspace(1) %p3, align 1
   store i32 %tmp1, ptr addrspace(1) %p3
   %add1 = add i32 %ld1, %tmp1
   br i1 %cond1, label %bb1, label %bb2

bb1:
  br label %bb2

bb2:
  %phi0 = phi i32 [ 100, %bb1 ], [ %add1, %entry ]
  %ld3 = load i32, ptr addrspace(3) %p2, align 1
  %add2 = add i32 %ld3, 100
  br i1 %cond2, label %bb3, label %bb4

bb3:
  %mul1 = mul i32 %ld1, %phi0
  %add3 = add i32 %mul1, 1000
  br label %bb5

bb5:
  %add30 = add i32 %add3, 900
  %gep2 = getelementptr inbounds i32, ptr addrspace(3) %p2, i64 3
  %ld4 = load i32, ptr addrspace(3) %gep2, align 8
  %add5 = add i32 %ld4, %add30
  %load3 = load <8 x i32>, ptr addrspace(1) %p4, align 1
  %load4 = load i32, ptr addrspace(1) %p4, align 2
  %tmp2 = add i32 %load4, %tmp1
  store i32 %tmp2, ptr addrspace(1) %p4
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %load6 = load i32, ptr addrspace(1) %p6, align 4
  %arrayidx11 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 %load6
  store i32 %phi0, ptr addrspace(5) %arrayidx11, align 4
  %arrayidx22 = getelementptr inbounds i32, ptr addrspace(1) %p6, i32 1
  %load7 = load i32, ptr addrspace(1) %arrayidx22, align 4
  %arrayidx33 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 %load7
  store i32 %ld3, ptr addrspace(5) %arrayidx33, align 2
  %xor = xor i1 %cond1, %cond2
  br i1 %xor, label %bb6, label %bb7

bb6:
  %and = and i1 %cond1, %cond2
  %idx10 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 1, i64 0
  %val0 = load i32, i32* %idx10, align 4
  %idx20 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 0, i64 1
  store i32 %val0, i32 *%idx20
  br i1 %and, label %bb8, label %bb9

bb8:
  %add6 = add i32 %ld2, %add5
  %idx12 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 1, i64 2
  %val2 = load i32, i32* %idx12, align 4
  %idx22 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 3, i64 2
  store i32 %val2, i32 *%idx22
  br label %bb10

bb9:
  %mul2 = mul i32 %ld2, %add5
  %idx13 = getelementptr inbounds [5 x i32], [5 x i32]* @array5, i64 1, i64 0
  %val3 = load i32, i32* %idx13, align 4
  %idx23 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 3, i64 0
  store i32 %val3, i32 *%idx23
  br label %bb10

bb7:
  %sub1 = sub i32 %ld4, %add5
  %mul3 = mul i32 %sub1, %ld3
  %div = udiv i32 %mul3, %ld1
  %add7 = add i32 %div, %ld2
  %idx14 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 4, i64 1
  %val4 = load i32, i32* %idx14, align 4
  %idx24 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 3, i64 0
  store i32 %val4, i32 *%idx24
  br label %bb10

bb10:
  %phi2 = phi i32 [ %add6, %bb8 ], [ %mul2, %bb9], [ %add7, %bb7 ]
  %add8 = add i32 %add2, %phi2
  %extract1 = extractelement < 8 x i32> %load2, i32 1
  %extract2 = extractelement < 8 x i32> %load2, i32 2
  %extract3 = extractelement < 8 x i32> %load2, i32 6
  %extract4 = extractelement < 8 x i32> %load2, i32 7
  %add101 = add i32 %extract1, %extract4
  %add102 = add i32 %add101, %extract3
  %idx1 = zext i32 %extract2 to i64
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %p1, i64 %idx1
  store i32 %add102, ptr addrspace(1) %arrayidx1, align 4
  %cond3 = icmp ne i1 %cond1, %cond2
  br i1 %cond3, label %bb11, label %bb12

bb11:
  %extract5 = extractelement < 8 x i32> %load3, i32 3
  %extract6 = extractelement < 8 x i32> %load3, i32 5
  %tmp3 = add i32 %extract5, %phi2
  %idx2 = zext i32 %extract6 to i64
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %p5, i64 %idx2
  store i32 %tmp3, ptr addrspace(1) %arrayidx2, align 8
  br label %bb13

bb12:
  %extract7 = extractelement < 8 x i32> %load3, i32 1
  %extract8 = extractelement < 8 x i32> %load3, i32 2
  %extract9 = extractelement < 8 x i32> %load2, i32 3
  %extract10 = extractelement < 8 x i32> %load2, i32 5
  %tmp4 = add i32 %extract9, %add8
  %idx3 = zext i32 %extract10 to i64
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %p5, i64 %idx3
  store i32 %tmp4, ptr addrspace(1) %arrayidx3, align 2
  %tmp5 = add i32 %extract7, %tmp4
  %idx4 = zext i32 %extract8 to i64
  %arrayidx5 = getelementptr inbounds i32, ptr addrspace(1) %p6, i64 %idx4
  store i32 %tmp5, ptr addrspace(1) %arrayidx5, align 2
  %array1 = alloca [5 x i32], align 4, addrspace(5)
  %load8 = load i32, ptr addrspace(1) %p6, align 4
  %arrayidx111 = getelementptr inbounds [5 x i32], ptr addrspace(5) %array1, i32 2, i32 %load8
  store i32 %tmp4, ptr addrspace(5) %arrayidx111, align 4
  %arrayidx222 = getelementptr inbounds i32, ptr addrspace(1) %p6, i32 1
  %load9 = load i32, ptr addrspace(1) %arrayidx222, align 4
  %arrayidx333 = getelementptr inbounds [5 x i32], ptr addrspace(5) %array1, i32 1, i32 %load9
  %load10 = load i32, ptr addrspace(5) %arrayidx333
  %arrayidx444 = getelementptr inbounds i32, ptr addrspace(1) %p4, i32 1
  store i32 %add30, ptr addrspace(1) %arrayidx444
  %idx15 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 0, i64 0
  %val5 = load i32, i32* %idx15, align 4
  %idx16 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 0, i64 0
  %val6 = load i32, i32* %idx16, align 4
  %idx17 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 0, i64 0
  %val7 = load i32, i32* %idx17, align 4
  %idx18 = getelementptr inbounds [5 x i32], [5 x i32]* @array5, i64 0, i64 0
  %val8 = load i32, i32* %idx18, align 4
  %mul10 = mul i32 %val5, %val6
  %mul11 = mul i32 %val7, %val8
  %add100 = add i32 %mul10, %mul11
  br label %bb13

bb13:
  %phi3 = phi i32 [ %tmp3, %bb11 ], [ %add100, %bb12]
  %phi4 = phi i32 [ %tmp3, %bb11 ], [ %val5, %bb12]
  %phi5 = phi i32 [ %tmp3, %bb11 ], [ %load7, %bb12]
  %phi6 = phi i32 [ %tmp3, %bb11 ], [ %load6, %bb12]
  %phi7 = phi i32 [ %tmp3, %bb11 ], [ %val8, %bb12]
  %phi8 = phi i32 [ %tmp3, %bb11 ], [ %extract1, %bb12]
  %phi9 = phi i32 [ %tmp3, %bb11 ], [ %extract2, %bb12]
  %phi10 = phi i32 [ %tmp3, %bb11 ], [ %extract3, %bb12]
  %phi11 = phi i32 [ %tmp3, %bb11 ], [ %extract4, %bb12]
  %phi12 = phi i32 [ %tmp3, %bb11 ], [ %load4, %bb12]
  %add200 = add i32 %phi3, %phi2
  %add300 = sub i32 %phi0, %add1
  %add400 = add i32 %add200, %add300
  %add500 = mul i32 %add400, %add2
  %add600 = add i32 %add500, %add30
  %add700 = sub i32 %add600, %ld4
  %add800 = add i32 %add700, %phi4
  %add900 = mul i32 %add800, %phi5
  %add1000 = sub i32 %add900, %phi6
  %add1100 = mul i32 %add1000, %phi7
  %add1200 = add i32 %add1100, %phi8
  %add1300 = sub i32 %add1200, %phi9
  %add1400 = sub i32 %add1300, %phi10
  %add1500 = add i32 %add1400, %phi11
  %add1600 = mul i32 %add1500, %phi12
  store i32 %add1600, ptr addrspace(1) %p7, align 2
  br label %bb14

bb4:
   %phi13 = phi i32 [ 0, %bb2 ], [ %ind, %bb4 ]
   %idx600 = getelementptr inbounds [5 x i32], [5 x i32]* @array6, i64 1, i64 2
   %val600 = load i32, i32* %idx600, align 4
   %idx700 = getelementptr inbounds [5 x i32], [5 x i32]* @array7, i64 3, i64 2
   %addval600 = add i32 %val600, %phi13
   store i32 %addval600, i32 *%idx700
   %idx800 = getelementptr inbounds [5 x i32], [5 x i32]* @array8, i64 1, i64 0
   %val800 = load i32, i32* %idx800, align 4
   %idx900 = getelementptr inbounds [5 x i32], [5 x i32]* @array9, i64 3, i64 0
   %addval800 = add i32 %val800, %phi13
   store i32 %addval800, i32 *%idx900
   %idx601 = getelementptr inbounds [5 x i32], [5 x i32]* @array6, i64 2, i64 1
   %val601 = load i32, i32* %idx601, align 1
   %val611 = mul i32 %val601, %phi13
   %idx701 = getelementptr inbounds [5 x i32], [5 x i32]* @array7, i64 1, i64 0
   %val701 = load i32, i32* %idx701, align 2
   %val711 = sub i32 %val701, %phi13
   %idx801 = getelementptr inbounds [5 x i32], [5 x i32]* @array8, i64 2, i64 1
   %val801 = load i32, i32* %idx801, align 8
   %val811 = add i32 %val801, %phi13
   %idx901 = getelementptr inbounds [5 x i32], [5 x i32]* @array9, i64 1, i64 1
   %val901 = load i32, i32* %idx901, align 1
   %val911 = mul i32 %val901, %phi13
   %idx602 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 4, i64 0
   %val602 = load i32, i32* %idx602, align 1
   %val612 = add i32 %val602, %phi13
   %idx702 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 4, i64 0
   %val702 = load i32, i32* %idx702, align 2
   %val712 = sub i32 %val702, %phi13
   %idx802 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 4, i64 0
   %val802 = load i32, i32* %idx802, align 8
   %val812 = add i32 %val802, %phi13
   %idx902 = getelementptr inbounds [5 x i32], [5 x i32]* @array5, i64 4, i64 0
   %val902 = load i32, i32* %idx902, align 1
   %val912 = mul i32 %val902, %phi13
   %ind = add i32 %phi13, 1
   %loop.cond = icmp ult i32 %ind, %TC1
   br i1 %loop.cond, label %bb4, label %bb14

bb14:
  %phi14 = phi i32 [ %add200, %bb13 ], [ %val611, %bb4 ]
  %phi15 = phi i32 [ %add500, %bb13 ], [ %val711, %bb4 ]
  %phi16 = phi i32 [ %add600, %bb13 ], [ %val811, %bb4 ]
  %phi17 = phi i32 [ %add300, %bb13 ], [ %val911, %bb4 ]
  %phi18 = phi i32 [ %add1000, %bb13 ], [ %val612, %bb4 ]
  %phi19 = phi i32 [ %add1600, %bb13 ], [ %val712, %bb4 ]
  %phi20 = phi i32 [ %add500, %bb13 ], [ %val812, %bb4 ]
  %phi21 = phi i32 [ %add500, %bb13 ], [ %val912, %bb4 ]
  %addall1 = add i32 %phi14, %phi0
  %addall2 = add i32 %addall1, %phi15
  %addall3 = add i32 %addall2, 100
  %addall4 = add i32 %addall3, %phi16
  %addall5 = add i32 %addall4, %phi17
  %addall6 = add i32 %addall5, %phi18
  %addall7 = add i32 %addall6, %phi19
  %addall8 = add i32 %addall7, %phi20
  %addall9 = add i32 %addall8, %phi21
  %gep3 = getelementptr inbounds i32, ptr addrspace(1) %p8, i64 1
  store i32 %addall9, ptr addrspace(1) %gep3
  %gep4 = getelementptr inbounds i32, ptr addrspace(1) %p9, i64 1
  store i32 %addall6, ptr addrspace(1) %gep4
  ret void
}
