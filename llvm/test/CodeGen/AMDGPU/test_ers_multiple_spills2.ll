; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

@array2 = global [5 x i32] zeroinitializer, align 4
@array3 = global [5 x i32] zeroinitializer, align 4
@array4 = global [5 x i32] zeroinitializer, align 4
@array5 = global [5 x i32] zeroinitializer, align 4

@array6 = global [5 x i32] zeroinitializer, align 4
@array7 = global [5 x i32] zeroinitializer, align 4
@array8 = global [5 x i32] zeroinitializer, align 4
@array9 = global [5 x i32] zeroinitializer, align 4

;
;                 bb.0.entry
;                  /     |
;              bb.1.bb1  |
;                  \     |
;                 bb.2.bb2
;                  /     |
;              bb.5.bb8  |
;                  \     |
;                 bb.3.Flow
;                  /     |
;              bb.4.bb7  |
;                  \     |
;                 bb.6.Flow1
;                  /     |
;              bb.7.bb9  |
;                  \     |
;                 bb.8.bb10
;
define amdgpu_ps void @test(ptr addrspace(1) %p1, ptr addrspace(3) %p2, ptr addrspace(1) %p3, ptr addrspace(1) %p4, ptr addrspace(1) %p5, ptr addrspace(1) %p6, ptr addrspace(1) %p7, ptr addrspace(1) %p8, ptr addrspace(1) %p9, ptr addrspace(1) %p10, ptr addrspace(1) %p11, i32 %arg1, i32 %arg2) {
; CHECK-LABEL: # Machine code for function test: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]], $vgpr7 in [[Reg8:%[0-9]+]], $vgpr8 in [[Reg9:%[0-9]+]], $vgpr9 in [[Reg10:%[0-9]+]], $vgpr10 in [[Reg11:%[0-9]+]], $vgpr11 in [[Reg12:%[0-9]+]], $vgpr12 in [[Reg13:%[0-9]+]], $vgpr13 in [[Reg14:%[0-9]+]], $vgpr14 in [[Reg15:%[0-9]+]], $vgpr15 in [[Reg16:%[0-9]+]], $vgpr16 in [[Reg17:%[0-9]+]], $vgpr17 in [[Reg18:%[0-9]+]], $vgpr18 in [[Reg19:%[0-9]+]], $vgpr19 in [[Reg20:%[0-9]+]], $vgpr20 in [[Reg21:%[0-9]+]], $vgpr21 in [[Reg22:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x40000000), %bb.2(0x40000000); %bb.1(50.00%), %bb.2(50.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10, $vgpr11, $vgpr12, $vgpr13, $vgpr14, $vgpr15, $vgpr16, $vgpr17, $vgpr18, $vgpr19, $vgpr20, $vgpr21
; CHECK-NEXT:   [[Reg22]]:vgpr_32 = COPY killed $vgpr21
; CHECK-NEXT:   [[Reg21]]:vgpr_32 = COPY killed $vgpr20
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
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg10]]:vgpr_32, %subreg.sub0, killed [[Reg11]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg12]]:vgpr_32, %subreg.sub0, killed [[Reg13]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg4]]:vgpr_32, %subreg.sub0, killed [[Reg5]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg20]]:vgpr_32, %subreg.sub0, killed [[Reg21]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg18]]:vgpr_32, %subreg.sub0, killed [[Reg19]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg14]]:vgpr_32, %subreg.sub0, killed [[Reg15]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg16]]:vgpr_32, %subreg.sub0, killed [[Reg17]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg31:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg8]]:vgpr_32, %subreg.sub0, killed [[Reg9]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg32:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg6]]:vgpr_32, %subreg.sub0, killed [[Reg7]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg33:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg32]]:vreg_64, 0, 0, implicit $exec :: (load (s16) from %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg32]]:vreg_64, 2, 0, implicit $exec :: (load (s16) from %ir.p4 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg34]]:vgpr_32, 16, killed [[Reg33]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD killed [[Reg31]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p5, align 8, addrspace 1)
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg30]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p9, addrspace 1)
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p9 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg39]]:vgpr_32, 8, killed [[Reg38]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p9 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg29]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p9 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg42]]:vgpr_32, 8, killed [[Reg41]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg43]]:vgpr_32, 16, killed [[Reg40]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg45:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg28]]:vreg_64, 0, 0, implicit $exec :: (load (s16) from %ir.p8, addrspace 1)
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_USHORT [[Reg28]]:vreg_64, 2, 0, implicit $exec :: (load (s16) from %ir.p8 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p10, addrspace 1)
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p10 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p10 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p10 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg35]]:vgpr_32, [[Reg36]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg26]]:vreg_64, killed [[Reg51]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p11, addrspace 1)
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg35]]:vgpr_32, [[Reg36]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg30]]:vreg_64, [[Reg52]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p1, addrspace 1)
; CHECK-NEXT:   [[Reg53:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 0, 0, implicit $exec :: (load (s8) from %ir.p2, addrspace 3)
; CHECK-NEXT:   [[Reg54:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 1, 0, implicit $exec :: (load (s8) from %ir.p2 + 1, addrspace 3)
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 2, 0, implicit $exec :: (load (s8) from %ir.p2 + 2, addrspace 3)
; CHECK-NEXT:   [[Reg56:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg3]]:vgpr_32, 3, 0, implicit $exec :: (load (s8) from %ir.p2 + 3, addrspace 3)
; CHECK-NEXT:   [[Reg57:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg54]]:vgpr_32, 8, killed [[Reg53]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg58:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg56]]:vgpr_32, 8, killed [[Reg55]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg59:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg58]]:vgpr_32, 16, killed [[Reg57]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg60:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 8, 0, implicit $exec :: (load (s8) from %ir.p3 + 8, addrspace 1)
; CHECK-NEXT:   [[Reg61:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 9, 0, implicit $exec :: (load (s8) from %ir.p3 + 9, addrspace 1)
; CHECK-NEXT:   [[Reg62:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg61]]:vgpr_32, 8, killed [[Reg60]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg63:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 10, 0, implicit $exec :: (load (s8) from %ir.p3 + 10, addrspace 1)
; CHECK-NEXT:   [[Reg64:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 11, 0, implicit $exec :: (load (s8) from %ir.p3 + 11, addrspace 1)
; CHECK-NEXT:   [[Reg65:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg64]]:vgpr_32, 8, killed [[Reg63]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg66:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg65]]:vgpr_32, 16, killed [[Reg62]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg67:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 4, 0, implicit $exec :: (load (s8) from %ir.p3 + 4, addrspace 1)
; CHECK-NEXT:   [[Reg68:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 5, 0, implicit $exec :: (load (s8) from %ir.p3 + 5, addrspace 1)
; CHECK-NEXT:   [[Reg69:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg68]]:vgpr_32, 8, killed [[Reg67]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg70:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 6, 0, implicit $exec :: (load (s8) from %ir.p3 + 6, addrspace 1)
; CHECK-NEXT:   [[Reg71:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 7, 0, implicit $exec :: (load (s8) from %ir.p3 + 7, addrspace 1)
; CHECK-NEXT:   [[Reg72:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg71]]:vgpr_32, 8, killed [[Reg70]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg73:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg72]]:vgpr_32, 16, killed [[Reg69]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg74:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg75:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p3 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg76:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg75]]:vgpr_32, 8, killed [[Reg74]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg77:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p3 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg78:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg25]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p3 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg79:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg78]]:vgpr_32, 8, killed [[Reg77]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg80:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg79]]:vgpr_32, 16, killed [[Reg76]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg81:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg59]]:vgpr_32, killed [[Reg80]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg82:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg81]]:vgpr_32, killed [[Reg73]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg25]]:vreg_64, [[Reg82]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg83:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg29]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p9, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg24]]:vreg_64, [[Reg59]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p7, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg29]]:vreg_64, killed [[Reg81]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p9, addrspace 1)
; CHECK-NEXT:   [[Reg84:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg36]]:vgpr_32, killed [[Reg22]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg85:%[0-9]+]]:sreg_32 = SI_IF [[Reg84]]:sreg_32, %bb.2, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.1
; EMPTY:
; CHECK: bb.1.bb1:
; CHECK-NEXT: ; predecessors: %bb.0
; CHECK-NEXT:   successors: %bb.2(0x80000000); %bb.2(100.00%)
; EMPTY:
; CHECK:   [[Reg86:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg46]]:vgpr_32, 16, killed [[Reg45]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg87:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD killed [[Reg25]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg88:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 8, 0, implicit $exec :: (load (s8) from %ir.p10 + 8, addrspace 1)
; CHECK-NEXT:   [[Reg89:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 9, 0, implicit $exec :: (load (s8) from %ir.p10 + 9, addrspace 1)
; CHECK-NEXT:   [[Reg90:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg89]]:vgpr_32, 8, killed [[Reg88]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg91:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 10, 0, implicit $exec :: (load (s8) from %ir.p10 + 10, addrspace 1)
; CHECK-NEXT:   [[Reg92:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 11, 0, implicit $exec :: (load (s8) from %ir.p10 + 11, addrspace 1)
; CHECK-NEXT:   [[Reg93:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg92]]:vgpr_32, 8, killed [[Reg91]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg94:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg93]]:vgpr_32, 16, killed [[Reg90]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg95:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 4, 0, implicit $exec :: (load (s8) from %ir.p10 + 4, addrspace 1)
; CHECK-NEXT:   [[Reg96:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 5, 0, implicit $exec :: (load (s8) from %ir.p10 + 5, addrspace 1)
; CHECK-NEXT:   [[Reg97:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg96]]:vgpr_32, 8, killed [[Reg95]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg98:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 6, 0, implicit $exec :: (load (s8) from %ir.p10 + 6, addrspace 1)
; CHECK-NEXT:   [[Reg99:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg27]]:vreg_64, 7, 0, implicit $exec :: (load (s8) from %ir.p10 + 7, addrspace 1)
; CHECK-NEXT:   [[Reg100:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg99]]:vgpr_32, 8, killed [[Reg98]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg101:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg100]]:vgpr_32, 16, killed [[Reg97]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg102:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg87]]:vgpr_32, [[Reg101]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg103:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg102]]:vgpr_32, [[Reg94]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg23]]:vreg_64, killed [[Reg103]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p6, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg28]]:vreg_64, killed [[Reg102]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p8, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_DWORD [[Reg29]]:vreg_64, [[Reg101]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p9, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg27]]:vreg_64, [[Reg87]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p10, addrspace 1)
; CHECK-NEXT:   [[Reg104:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg87]]:vgpr_32, [[Reg87]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg105:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg87]]:vgpr_32, [[Reg101]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg106:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg101]]:vgpr_32, [[Reg94]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg107:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg87]]:vgpr_32, killed [[Reg94]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg108:%[0-9]+]]:vreg_64 = REG_SEQUENCE [[Reg35]]:vgpr_32, %subreg.sub0, undef [[Reg109:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg110:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 [[Reg104]]:vgpr_32, killed [[Reg36]]:vgpr_32, killed [[Reg108]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg111:%[0-9]+]]:vreg_64 = REG_SEQUENCE [[Reg105]]:vgpr_32, %subreg.sub0, undef [[Reg112:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg113:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 killed [[Reg110]].sub0:vreg_64, killed [[Reg52]]:vgpr_32, killed [[Reg111]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg114:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg113]].sub0:vreg_64, [[Reg106]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg115:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg113]].sub0:vreg_64, [[Reg107]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg116:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg104]]:vgpr_32, [[Reg106]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg117:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array2, target-flags(amdgpu-gotprel32-hi) @array2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg118:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg117]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg119:%[0-9]+]]:vreg_64 = COPY killed [[Reg118]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg120:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg119]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array2, i64 20)`)
; CHECK-NEXT:   [[Reg121:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg120]]:vgpr_32, [[Reg107]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg122:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array4, target-flags(amdgpu-gotprel32-hi) @array4, implicit-def dead $scc
; CHECK-NEXT:   [[Reg123:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg122]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg124:%[0-9]+]]:vreg_64 = COPY killed [[Reg123]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD [[Reg124]]:vreg_64, [[Reg121]]:vgpr_32, 4, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array4, i64 4)`)
; CHECK-NEXT:   [[Reg125:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p7, addrspace 1)
; CHECK-NEXT:   [[Reg126:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p7 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg127:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg126]]:vgpr_32, 8, killed [[Reg125]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg128:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p7 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg129:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p7 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg130:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg129]]:vgpr_32, 8, killed [[Reg128]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg131:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg130]]:vgpr_32, 16, killed [[Reg127]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg132:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 28, 0, implicit $exec :: (load (s8) from %ir.p7 + 28, addrspace 1)
; CHECK-NEXT:   [[Reg133:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 29, 0, implicit $exec :: (load (s8) from %ir.p7 + 29, addrspace 1)
; CHECK-NEXT:   [[Reg134:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg133]]:vgpr_32, 8, killed [[Reg132]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg135:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 30, 0, implicit $exec :: (load (s8) from %ir.p7 + 30, addrspace 1)
; CHECK-NEXT:   [[Reg136:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 31, 0, implicit $exec :: (load (s8) from %ir.p7 + 31, addrspace 1)
; CHECK-NEXT:   [[Reg137:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg136]]:vgpr_32, 8, killed [[Reg135]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg138:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg137]]:vgpr_32, 16, killed [[Reg134]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg139:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 24, 0, implicit $exec :: (load (s8) from %ir.p7 + 24, addrspace 1)
; CHECK-NEXT:   [[Reg140:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 25, 0, implicit $exec :: (load (s8) from %ir.p7 + 25, addrspace 1)
; CHECK-NEXT:   [[Reg141:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg140]]:vgpr_32, 8, killed [[Reg139]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg142:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 26, 0, implicit $exec :: (load (s8) from %ir.p7 + 26, addrspace 1)
; CHECK-NEXT:   [[Reg143:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg24]]:vreg_64, 27, 0, implicit $exec :: (load (s8) from %ir.p7 + 27, addrspace 1)
; CHECK-NEXT:   [[Reg144:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg143]]:vgpr_32, 8, killed [[Reg142]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg145:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg144]]:vgpr_32, 16, killed [[Reg141]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg146:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg119]]:vreg_64, 28, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array2, i64 28)`)
; CHECK-NEXT:   [[Reg147:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg146]]:vgpr_32, [[Reg116]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg148:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg121]]:vgpr_32, [[Reg106]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg149:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg107]]:vgpr_32, [[Reg120]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg150:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg149]]:vgpr_32, [[Reg147]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg151:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg148]]:vgpr_32, [[Reg150]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg152:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg151]]:vgpr_32, killed [[Reg145]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg153:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array3, target-flags(amdgpu-gotprel32-hi) @array3, implicit-def dead $scc
; CHECK-NEXT:   [[Reg154:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg153]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg155:%[0-9]+]]:vreg_64 = COPY killed [[Reg154]]:sreg_64_xexec
; CHECK-NEXT:   FLAT_STORE_DWORD [[Reg155]]:vreg_64, [[Reg152]]:vgpr_32, 68, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 68)`)
; CHECK-NEXT:   [[Reg156:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array5, target-flags(amdgpu-gotprel32-hi) @array5, implicit-def dead $scc
; CHECK-NEXT:   [[Reg157:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg156]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg158:%[0-9]+]]:vreg_64 = COPY killed [[Reg157]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg159:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg158]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array5, i64 20)`)
; CHECK-NEXT:   [[Reg160:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg159]]:vgpr_32, [[Reg138]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg161:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg160]]:vgpr_32, [[Reg105]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg162:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg120]]:vgpr_32, [[Reg138]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg163:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg107]]:vgpr_32, killed [[Reg106]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg164:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg161]]:vgpr_32, [[Reg148]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg165:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg160]]:vgpr_32, [[Reg151]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg166:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg138]]:vgpr_32, [[Reg107]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg167:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg162]]:vgpr_32, [[Reg146]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg168:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array6, target-flags(amdgpu-gotprel32-hi) @array6, implicit-def dead $scc
; CHECK-NEXT:   [[Reg169:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg168]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg170:%[0-9]+]]:vreg_64 = COPY killed [[Reg169]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg171:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg170]]:vreg_64, 44, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array6, i64 44)`)
; CHECK-NEXT:   [[Reg172:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg171]]:vgpr_32, [[Reg162]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg173:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array7, target-flags(amdgpu-gotprel32-hi) @array7, implicit-def dead $scc
; CHECK-NEXT:   [[Reg174:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg173]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg175:%[0-9]+]]:vreg_64 = COPY killed [[Reg174]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg176:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg175]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array7, i64 20)`)
; CHECK-NEXT:   [[Reg177:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array8, target-flags(amdgpu-gotprel32-hi) @array8, implicit-def dead $scc
; CHECK-NEXT:   [[Reg178:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg177]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg179:%[0-9]+]]:vreg_64 = COPY killed [[Reg178]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg180:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg179]]:vreg_64, 44, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array8, i64 44)`, align 8)
; CHECK-NEXT:   [[Reg181:%[0-9]+]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @array9, target-flags(amdgpu-gotprel32-hi) @array9, implicit-def dead $scc
; CHECK-NEXT:   [[Reg182:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM killed [[Reg181]]:sreg_64, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; CHECK-NEXT:   [[Reg183:%[0-9]+]]:vreg_64 = COPY killed [[Reg182]]:sreg_64_xexec
; CHECK-NEXT:   [[Reg184:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg183]]:vreg_64, 24, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array9, i64 24)`)
; CHECK-NEXT:   [[Reg185:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg184]]:vgpr_32, [[Reg152]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg186:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg119]]:vreg_64, 84, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array2, i64 84)`)
; CHECK-NEXT:   [[Reg187:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg186]]:vgpr_32, killed [[Reg151]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg188:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg155]]:vreg_64, 80, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 80)`)
; CHECK-NEXT:   [[Reg189:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg124]]:vreg_64, 80, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array4, i64 80)`, align 8)
; CHECK-NEXT:   [[Reg190:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg158]]:vreg_64, 88, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array5, i64 88)`)
; CHECK-NEXT:   [[Reg191:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg190]]:vgpr_32, killed [[Reg148]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg192:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg179]]:vreg_64, 20, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array8, i64 20)`)
; CHECK-NEXT:   [[Reg193:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg192]]:vgpr_32, killed [[Reg147]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg194:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg124]]:vreg_64, 8, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array4, i64 8)`)
; CHECK-NEXT:   [[Reg195:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg119]]:vreg_64, 12, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array2, i64 12)`)
; CHECK-NEXT:   [[Reg196:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD [[Reg155]]:vreg_64, 4, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 4)`)
; CHECK-NEXT:   [[Reg197:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg158]]:vreg_64, 4, 0, implicit $exec, implicit $flat_scr :: (dereferenceable load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array5, i64 4)`)
; CHECK-NEXT:   [[Reg198:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg197]]:vgpr_32, [[Reg160]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg199:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg32]]:vreg_64, 16, 0, implicit $exec :: (load (s8) from %ir.p4 + 16, addrspace 1)
; CHECK-NEXT:   [[Reg200:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg32]]:vreg_64, 17, 0, implicit $exec :: (load (s8) from %ir.p4 + 17, addrspace 1)
; CHECK-NEXT:   [[Reg201:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg200]]:vgpr_32, 8, killed [[Reg199]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg202:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg32]]:vreg_64, 18, 0, implicit $exec :: (load (s8) from %ir.p4 + 18, addrspace 1)
; CHECK-NEXT:   [[Reg203:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg32]]:vreg_64, 19, 0, implicit $exec :: (load (s8) from %ir.p4 + 19, addrspace 1)
; CHECK-NEXT:   [[Reg204:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg203]]:vgpr_32, 8, killed [[Reg202]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg205:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg204]]:vgpr_32, 16, killed [[Reg201]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg206:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg32]]:vreg_64, 12, 0, implicit $exec :: (load (s8) from %ir.p4 + 12, addrspace 1)
; CHECK-NEXT:   [[Reg207:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg32]]:vreg_64, 13, 0, implicit $exec :: (load (s8) from %ir.p4 + 13, addrspace 1)
; CHECK-NEXT:   [[Reg208:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg207]]:vgpr_32, 8, killed [[Reg206]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg209:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg32]]:vreg_64, 14, 0, implicit $exec :: (load (s8) from %ir.p4 + 14, addrspace 1)
; CHECK-NEXT:   [[Reg210:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE killed [[Reg32]]:vreg_64, 15, 0, implicit $exec :: (load (s8) from %ir.p4 + 15, addrspace 1)
; CHECK-NEXT:   [[Reg211:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg210]]:vgpr_32, 8, killed [[Reg209]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg212:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg211]]:vgpr_32, 16, killed [[Reg208]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg213:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg212]]:vgpr_32, killed [[Reg162]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg214:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 [[Reg205]]:vgpr_32, [[Reg161]]:vgpr_32, killed [[Reg213]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg215:%[0-9]+]]:vgpr_32 = V_CVT_F32_U32_e64 [[Reg198]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg216:%[0-9]+]]:vgpr_32 = nofpexcept V_RCP_IFLAG_F32_e64 0, killed [[Reg215]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg217:%[0-9]+]]:vgpr_32 = nnan ninf nsz arcp contract afn reassoc nofpexcept V_MUL_F32_e64 0, 1333788670, 0, killed [[Reg216]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg218:%[0-9]+]]:vgpr_32 = nofpexcept V_CVT_U32_F32_e64 0, killed [[Reg217]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg219:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 0, [[Reg198]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg220:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg219]]:vgpr_32, [[Reg218]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg221:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg218]]:vgpr_32, killed [[Reg220]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg222:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg218]]:vgpr_32, killed [[Reg221]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg223:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg214]]:vgpr_32, killed [[Reg222]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg224:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg223]]:vgpr_32, [[Reg198]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg225:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg214]]:vgpr_32, killed [[Reg224]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg226:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 [[Reg225]]:vgpr_32, [[Reg198]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg227:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg223]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg228:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg223]]:vgpr_32, 0, killed [[Reg227]]:vgpr_32, [[Reg226]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg229:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg225]]:vgpr_32, [[Reg198]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg230:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg225]]:vgpr_32, 0, killed [[Reg229]]:vgpr_32, killed [[Reg226]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg231:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 killed [[Reg230]]:vgpr_32, killed [[Reg198]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg232:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg228]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg233:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg228]]:vgpr_32, 0, killed [[Reg232]]:vgpr_32, killed [[Reg231]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg234:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg196]]:vgpr_32, [[Reg121]]:vgpr_32, killed [[Reg233]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg235:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg234]]:vgpr_32, killed [[Reg190]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg236:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg235]]:vgpr_32, killed [[Reg184]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg237:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg195]]:vgpr_32, [[Reg152]]:vgpr_32, killed [[Reg236]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg238:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg114]]:vgpr_32, %subreg.sub0, undef [[Reg239:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg240:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 killed [[Reg237]]:vgpr_32, killed [[Reg115]]:vgpr_32, killed [[Reg238]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg241:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg240]].sub0:vreg_64, [[Reg83]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg242:%[0-9]+]]:vgpr_32 = V_CVT_F32_U32_e64 [[Reg138]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg243:%[0-9]+]]:vgpr_32 = nofpexcept V_RCP_IFLAG_F32_e64 0, killed [[Reg242]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg244:%[0-9]+]]:vgpr_32 = nnan ninf nsz arcp contract afn reassoc nofpexcept V_MUL_F32_e64 0, 1333788670, 0, killed [[Reg243]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg245:%[0-9]+]]:vgpr_32 = nofpexcept V_CVT_U32_F32_e64 0, killed [[Reg244]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg246:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 0, [[Reg138]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg247:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg246]]:vgpr_32, [[Reg245]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg248:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg245]]:vgpr_32, killed [[Reg247]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg249:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg245]]:vgpr_32, killed [[Reg248]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg250:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg241]]:vgpr_32, killed [[Reg249]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg251:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg250]]:vgpr_32, [[Reg138]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg252:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg241]]:vgpr_32, killed [[Reg251]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg253:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 [[Reg252]]:vgpr_32, [[Reg138]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg254:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg250]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg255:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg250]]:vgpr_32, 0, killed [[Reg254]]:vgpr_32, [[Reg253]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg256:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg252]]:vgpr_32, [[Reg138]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg257:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg252]]:vgpr_32, 0, killed [[Reg256]]:vgpr_32, killed [[Reg253]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg258:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 killed [[Reg257]]:vgpr_32, killed [[Reg138]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg259:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg255]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg260:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg255]]:vgpr_32, 0, killed [[Reg259]]:vgpr_32, killed [[Reg258]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   DS_WRITE_B8_D16_HI [[Reg3]]:vgpr_32, [[Reg260]]:vgpr_32, 2, 0, implicit $exec :: (store (s8) into %ir.p2 + 2, addrspace 3)
; CHECK-NEXT:   DS_WRITE_B8_gfx9 [[Reg3]]:vgpr_32, [[Reg260]]:vgpr_32, 0, 0, implicit $exec :: (store (s8) into %ir.p2, addrspace 3)
; CHECK-NEXT:   [[Reg261:%[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 24, [[Reg260]]:vgpr_32, implicit $exec
; CHECK-NEXT:   DS_WRITE_B8_gfx9 [[Reg3]]:vgpr_32, killed [[Reg261]]:vgpr_32, 3, 0, implicit $exec :: (store (s8) into %ir.p2 + 3, addrspace 3)
; CHECK-NEXT:   [[Reg262:%[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 8, [[Reg260]]:vgpr_32, implicit $exec
; CHECK-NEXT:   DS_WRITE_B8_gfx9 killed [[Reg3]]:vgpr_32, killed [[Reg262]]:vgpr_32, 1, 0, implicit $exec :: (store (s8) into %ir.p2 + 1, addrspace 3)
; CHECK-NEXT:   [[Reg263:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg260]]:vgpr_32, killed [[Reg192]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg264:%[0-9]+]]:vreg_64 = REG_SEQUENCE [[Reg180]]:vgpr_32, %subreg.sub0, undef [[Reg265:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg266:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 killed [[Reg263]]:vgpr_32, [[Reg205]]:vgpr_32, killed [[Reg264]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg267:%[0-9]+]]:vreg_64 = REG_SEQUENCE [[Reg120]]:vgpr_32, %subreg.sub0, undef [[Reg268:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg269:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 killed [[Reg266]].sub0:vreg_64, killed [[Reg146]]:vgpr_32, killed [[Reg267]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg270:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg269]].sub0:vreg_64, [[Reg116]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg271:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg270]]:vgpr_32, %subreg.sub0, undef [[Reg272:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg273:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 killed [[Reg194]]:vgpr_32, [[Reg121]]:vgpr_32, killed [[Reg271]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg274:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg273]].sub0:vreg_64, killed [[Reg193]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg275:%[0-9]+]]:vgpr_32 = V_CVT_F32_U32_e64 [[Reg191]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg276:%[0-9]+]]:vgpr_32 = nofpexcept V_RCP_IFLAG_F32_e64 0, killed [[Reg275]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg277:%[0-9]+]]:vgpr_32 = nnan ninf nsz arcp contract afn reassoc nofpexcept V_MUL_F32_e64 0, 1333788670, 0, killed [[Reg276]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg278:%[0-9]+]]:vgpr_32 = nofpexcept V_CVT_U32_F32_e64 0, killed [[Reg277]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg279:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 0, [[Reg191]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg280:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg279]]:vgpr_32, [[Reg278]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg281:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg278]]:vgpr_32, killed [[Reg280]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg282:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg278]]:vgpr_32, killed [[Reg281]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg283:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg274]]:vgpr_32, killed [[Reg282]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg284:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg283]]:vgpr_32, [[Reg191]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg285:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg274]]:vgpr_32, killed [[Reg284]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg286:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 [[Reg285]]:vgpr_32, [[Reg191]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg287:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg283]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg288:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg283]]:vgpr_32, 0, killed [[Reg287]]:vgpr_32, [[Reg286]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg289:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg285]]:vgpr_32, [[Reg191]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg290:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg285]]:vgpr_32, 0, killed [[Reg289]]:vgpr_32, killed [[Reg286]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg291:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 killed [[Reg290]]:vgpr_32, killed [[Reg191]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg292:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg288]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg293:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg288]]:vgpr_32, 0, killed [[Reg292]]:vgpr_32, killed [[Reg291]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg294:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg189]]:vgpr_32, killed [[Reg149]]:vgpr_32, killed [[Reg293]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg295:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg150]]:vgpr_32, killed [[Reg188]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg296:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg294]]:vgpr_32, killed [[Reg295]]:vgpr_32, killed [[Reg187]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg297:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg296]]:vgpr_32, killed [[Reg185]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg298:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg180]]:vgpr_32, killed [[Reg160]]:vgpr_32, killed [[Reg297]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg299:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg161]]:vgpr_32, killed [[Reg176]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg300:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg298]]:vgpr_32, killed [[Reg299]]:vgpr_32, killed [[Reg172]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg301:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg300]]:vgpr_32, killed [[Reg121]]:vgpr_32, implicit $exec
; CHECK-NEXT:   FLAT_STORE_DWORD killed [[Reg175]]:vreg_64, [[Reg301]]:vgpr_32, 68, 0, implicit $exec, implicit $flat_scr :: (store (s32) into `ptr getelementptr inbounds nuw (i8, ptr @array7, i64 68)`)
; CHECK-NEXT:   [[Reg302:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg120]]:vgpr_32, killed [[Reg152]]:vgpr_32, killed [[Reg301]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg303:%[0-9]+]]:vgpr_32 = V_CVT_F32_U32_e64 [[Reg167]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg304:%[0-9]+]]:vgpr_32 = nofpexcept V_RCP_IFLAG_F32_e64 0, killed [[Reg303]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg305:%[0-9]+]]:vgpr_32 = nnan ninf nsz arcp contract afn reassoc nofpexcept V_MUL_F32_e64 0, 1333788670, 0, killed [[Reg304]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg306:%[0-9]+]]:vgpr_32 = nofpexcept V_CVT_U32_F32_e64 0, killed [[Reg305]]:vgpr_32, 0, 0, implicit $mode, implicit $exec
; CHECK-NEXT:   [[Reg307:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 0, [[Reg167]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg308:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg307]]:vgpr_32, [[Reg306]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg309:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg306]]:vgpr_32, killed [[Reg308]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg310:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg306]]:vgpr_32, killed [[Reg309]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg311:%[0-9]+]]:vgpr_32 = V_MUL_HI_U32_e64 [[Reg302]]:vgpr_32, killed [[Reg310]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg312:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 [[Reg311]]:vgpr_32, [[Reg167]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg313:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg302]]:vgpr_32, killed [[Reg312]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg314:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 [[Reg313]]:vgpr_32, [[Reg167]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg315:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg311]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg316:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg311]]:vgpr_32, 0, killed [[Reg315]]:vgpr_32, [[Reg314]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg317:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 [[Reg313]]:vgpr_32, [[Reg167]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg318:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg313]]:vgpr_32, 0, killed [[Reg317]]:vgpr_32, killed [[Reg314]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg319:%[0-9]+]]:sreg_32_xm0_xexec = V_CMP_GE_U32_e64 killed [[Reg318]]:vgpr_32, killed [[Reg167]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg320:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 1, [[Reg316]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg321:%[0-9]+]]:vgpr_32 = V_CNDMASK_B32_e64 0, killed [[Reg316]]:vgpr_32, 0, killed [[Reg320]]:vgpr_32, killed [[Reg319]]:sreg_32_xm0_xexec, implicit $exec
; CHECK-NEXT:   [[Reg322:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg321]]:vgpr_32, killed [[Reg166]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg323:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg322]]:vgpr_32, killed [[Reg165]]:vgpr_32, killed [[Reg164]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg324:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg323]]:vgpr_32, killed [[Reg163]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg325:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg324]]:vgpr_32, [[Reg44]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg326:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg155]]:vreg_64, 84, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array3, i64 84)`)
; CHECK-NEXT:   [[Reg327:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg326]]:vgpr_32, killed [[Reg205]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg328:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg327]]:vgpr_32, killed [[Reg116]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg329:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg325]]:vgpr_32, %subreg.sub0, killed [[Reg328]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   FLAT_STORE_DWORDX2 killed [[Reg124]]:vreg_64, killed [[Reg329]]:vreg_64, 76, 0, implicit $exec, implicit $flat_scr :: (store (s64) into `ptr getelementptr inbounds nuw (i8, ptr @array4, i64 76)`, align 4)
; CHECK-NEXT:   [[Reg330:%[0-9]+]]:vgpr_32 = FLAT_LOAD_DWORD killed [[Reg170]]:vreg_64, 28, 0, implicit $exec, implicit $flat_scr :: (load (s32) from `ptr getelementptr inbounds nuw (i8, ptr @array6, i64 28)`)
; CHECK-NEXT:   [[Reg331:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg330]]:vgpr_32, killed [[Reg131]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg332:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg331]]:vgpr_32, killed [[Reg86]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_SHORT_D16_HI [[Reg28]]:vreg_64, [[Reg332]]:vgpr_32, 2, 0, implicit $exec :: (store (s16) into %ir.p8 + 2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_SHORT killed [[Reg28]]:vreg_64, killed [[Reg332]]:vgpr_32, 0, 0, implicit $exec :: (store (s16) into %ir.p8, addrspace 1)
; EMPTY:
; CHECK: bb.2.bb2:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.1
; CHECK-NEXT:   successors: %bb.5(0x40000000), %bb.3(0x40000000); %bb.5(50.00%), %bb.3(50.00%)
; EMPTY:
; CHECK:   [[Reg333:%[0-9]+]]:vgpr_32 = PHI [[Reg59]]:vgpr_32, %bb.0, [[Reg105]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg334:%[0-9]+]]:vgpr_32 = PHI [[Reg66]]:vgpr_32, %bb.0, [[Reg326]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg335:%[0-9]+]]:vgpr_32 = PHI [[Reg82]]:vgpr_32, %bb.0, [[Reg107]]:vgpr_32, %bb.1
; CHECK-NEXT:   SI_END_CF killed [[Reg85]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg336:%[0-9]+]]:sreg_32 = V_CMP_GT_U32_e64 [[Reg35]]:vgpr_32, killed [[Reg334]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg337:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; CHECK-NEXT:   [[Reg338:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg336]]:sreg_32, %bb.3, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.5
; EMPTY:
; CHECK: bb.3.Flow:
; CHECK-NEXT: ; predecessors: %bb.2, %bb.5
; CHECK-NEXT:   successors: %bb.4(0x40000000), %bb.6(0x40000000); %bb.4(50.00%), %bb.6(50.00%)
; EMPTY:
; CHECK:   [[Reg339:%[0-9]+]]:sreg_32 = PHI [[Reg337]]:sreg_32, %bb.2, [[Reg340:%[0-9]+]]:sreg_32, %bb.5
; CHECK-NEXT:   [[Reg341:%[0-9]+]]:vgpr_32 = PHI [[Reg333]]:vgpr_32, %bb.2, undef [[Reg342:%[0-9]+]]:vgpr_32, %bb.5
; CHECK-NEXT:   [[Reg343:%[0-9]+]]:vreg_64 = PHI [[Reg23]]:vreg_64, %bb.2, undef [[Reg344:%[0-9]+]]:vreg_64, %bb.5
; CHECK-NEXT:   [[Reg345:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg48]]:vgpr_32, 8, killed [[Reg47]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg346:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg50]]:vgpr_32, 8, killed [[Reg49]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg347:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg338]]:sreg_32, %bb.6, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.4.bb7:
; CHECK-NEXT: ; predecessors: %bb.3
; CHECK-NEXT:   successors: %bb.6(0x80000000); %bb.6(100.00%)
; EMPTY:
; CHECK:   [[Reg348:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg341]]:vgpr_32, [[Reg44]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg343]]:vreg_64, killed [[Reg348]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p6, addrspace 1)
; CHECK-NEXT:   [[Reg349:%[0-9]+]]:sreg_32 = S_OR_B32 killed [[Reg339]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   S_BRANCH %bb.6
; EMPTY:
; CHECK: bb.5.bb8:
; CHECK-NEXT: ; predecessors: %bb.2
; CHECK-NEXT:   successors: %bb.3(0x80000000); %bb.3(100.00%)
; EMPTY:
; CHECK:   [[Reg350:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg335]]:vgpr_32, killed [[Reg35]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg24]]:vreg_64, killed [[Reg350]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p7, addrspace 1)
; CHECK-NEXT:   [[Reg351:%[0-9]+]]:sreg_32 = S_AND_B32 killed [[Reg84]]:sreg_32, $exec_lo, implicit-def dead $scc
; CHECK-NEXT:   [[Reg340]]:sreg_32 = COPY killed [[Reg351]]:sreg_32
; CHECK-NEXT:   S_BRANCH %bb.3
; EMPTY:
; CHECK: bb.6.Flow1:
; CHECK-NEXT: ; predecessors: %bb.3, %bb.4
; CHECK-NEXT:   successors: %bb.7(0x40000000), %bb.8(0x40000000); %bb.7(50.00%), %bb.8(50.00%)
; EMPTY:
; CHECK:   [[Reg352:%[0-9]+]]:sreg_32 = PHI [[Reg339]]:sreg_32, %bb.3, [[Reg349]]:sreg_32, %bb.4
; CHECK-NEXT:   SI_END_CF killed [[Reg347]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg353:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg346]]:vgpr_32, 16, killed [[Reg345]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg354:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg352]]:sreg_32, %bb.8, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.7
; EMPTY:
; CHECK: bb.7.bb9:
; CHECK-NEXT: ; predecessors: %bb.6
; CHECK-NEXT:   successors: %bb.8(0x80000000); %bb.8(100.00%)
; EMPTY:
; CHECK:   [[Reg355:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg37]]:vgpr_32, killed [[Reg44]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg30]]:vreg_64, killed [[Reg355]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p1, addrspace 1)
; EMPTY:
; CHECK: bb.8.bb10:
; CHECK-NEXT: ; predecessors: %bb.6, %bb.7
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg354]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg356:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 killed [[Reg353]]:vgpr_32, killed [[Reg83]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg29]]:vreg_64, killed [[Reg356]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p9, addrspace 1)
; CHECK-NEXT:   S_ENDPGM 0
; EMPTY:
; CHECK: # End machine code for function test.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg22]] = 88.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 24.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 23.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 23.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 22.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 23.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 22.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 20.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 14.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 13.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 17.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 15.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 35.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 79.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 63.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 37.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 26.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 20.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 17.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 15.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 75.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 63.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 47.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 46.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 56.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 55.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 55.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 54.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg56]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg57]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg58]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg59]] = 22.0
; CHECK-NEXT: Next-use distance of Register [[Reg60]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg61]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg62]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg63]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg64]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg65]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg66]] = 25.0
; CHECK-NEXT: Next-use distance of Register [[Reg67]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg68]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg69]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg70]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg71]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg72]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg73]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg74]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg75]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg76]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg77]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg78]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg79]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg80]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg81]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg82]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg83]] = 27.0
; CHECK-NEXT: Next-use distance of Register [[Reg84]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg85]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg86]] = 252.0
; CHECK-NEXT: Next-use distance of Register [[Reg87]] = 15.0
; CHECK-NEXT: Next-use distance of Register [[Reg88]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg89]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg90]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg91]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg92]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg93]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg94]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg95]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg96]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg97]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg98]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg99]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg100]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg101]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg102]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg103]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg104]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg105]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg106]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg107]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg108]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg110]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg111]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg113]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg114]] = 126.0
; CHECK-NEXT: Next-use distance of Register [[Reg115]] = 126.0
; CHECK-NEXT: Next-use distance of Register [[Reg116]] = 32.0
; CHECK-NEXT: Next-use distance of Register [[Reg117]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg118]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg119]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg120]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg121]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg122]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg123]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg124]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg125]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg126]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg127]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg128]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg129]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg130]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg131]] = 203.0
; CHECK-NEXT: Next-use distance of Register [[Reg132]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg133]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg134]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg135]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg136]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg137]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg138]] = 23.0
; CHECK-NEXT: Next-use distance of Register [[Reg139]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg140]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg141]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg142]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg143]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg144]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg145]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg146]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg147]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg148]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg149]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg150]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg151]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg152]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg153]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg154]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg155]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg156]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg157]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg158]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg159]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg160]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg161]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg162]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg163]] = 162.0
; CHECK-NEXT: Next-use distance of Register [[Reg164]] = 160.0
; CHECK-NEXT: Next-use distance of Register [[Reg165]] = 159.0
; CHECK-NEXT: Next-use distance of Register [[Reg166]] = 157.0
; CHECK-NEXT: Next-use distance of Register [[Reg167]] = 137.0
; CHECK-NEXT: Next-use distance of Register [[Reg168]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg169]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg170]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg171]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg172]] = 128.0
; CHECK-NEXT: Next-use distance of Register [[Reg173]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg174]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg175]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg176]] = 123.0
; CHECK-NEXT: Next-use distance of Register [[Reg177]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg178]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg179]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg180]] = 87.0
; CHECK-NEXT: Next-use distance of Register [[Reg181]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg182]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg183]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg184]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg185]] = 112.0
; CHECK-NEXT: Next-use distance of Register [[Reg186]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg187]] = 109.0
; CHECK-NEXT: Next-use distance of Register [[Reg188]] = 107.0
; CHECK-NEXT: Next-use distance of Register [[Reg189]] = 105.0
; CHECK-NEXT: Next-use distance of Register [[Reg190]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg191]] = 84.0
; CHECK-NEXT: Next-use distance of Register [[Reg192]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg193]] = 81.0
; CHECK-NEXT: Next-use distance of Register [[Reg194]] = 79.0
; CHECK-NEXT: Next-use distance of Register [[Reg195]] = 42.0
; CHECK-NEXT: Next-use distance of Register [[Reg196]] = 38.0
; CHECK-NEXT: Next-use distance of Register [[Reg197]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg198]] = 17.0
; CHECK-NEXT: Next-use distance of Register [[Reg199]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg200]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg201]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg202]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg203]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg204]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg205]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg206]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg207]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg208]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg209]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg210]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg211]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg212]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg213]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg214]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg215]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg216]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg217]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg218]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg219]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg220]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg221]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg222]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg223]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg224]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg225]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg226]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg227]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg228]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg229]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg230]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg231]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg232]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg233]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg234]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg235]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg236]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg237]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg238]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg240]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg241]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg242]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg243]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg244]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg245]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg246]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg247]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg248]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg249]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg250]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg251]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg252]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg253]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg254]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg255]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg256]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg257]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg258]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg259]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg260]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg261]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg262]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg263]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg264]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg266]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg267]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg269]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg270]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg271]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg273]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg274]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg275]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg276]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg277]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg278]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg279]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg280]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg281]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg282]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg283]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg284]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg285]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg286]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg287]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg288]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg289]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg290]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg291]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg292]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg293]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg294]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg295]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg296]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg297]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg298]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg299]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg300]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg301]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg302]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg303]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg304]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg305]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg306]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg307]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg308]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg309]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg310]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg311]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg312]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg313]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg314]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg315]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg316]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg317]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg318]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg319]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg320]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg321]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg322]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg323]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg324]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg325]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg326]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg327]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg328]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg329]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg330]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg331]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg332]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg333]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg334]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg335]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg336]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg337]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg338]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg339]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg341]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg343]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg345]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg346]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg347]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg348]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg349]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg350]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg351]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg340]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg352]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg353]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg354]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg355]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg356]] = 1.0
entry:
  %ld = load i32, ptr addrspace(1) %p4, align 2
  %ld0 = load i32, ptr addrspace(1) %p5, align 8
  %ld1 = load i32, ptr addrspace(1) %p1, align 4
  %ld2 = load i32, ptr addrspace(1) %p9, align 1
  %ld8 = load i32, ptr addrspace(1) %p8, align 2
  %ld6 = load i32, ptr addrspace(1) %p6, align 4
  %ld10 = load i32, ptr addrspace(1) %p10, align 1
  %ld11 = load i32, ptr addrspace(1) %p11, align 1
  %tmp0 = sub i32 %ld, %ld0
  store i32 %tmp0, ptr addrspace(1) %p11
  %tmp1 = add i32 %ld, %ld0
  store i32 %tmp1, ptr addrspace(1) %p1
  %ld3 = load i32, ptr addrspace(3) %p2, align 1
  %load4 = load <8 x i32>, ptr addrspace(1) %p3, align 1
  %extract11 = extractelement < 8 x i32> %load4, i32 0
  %extract12 = extractelement < 8 x i32> %load4, i32 1
  %extract13 = extractelement < 8 x i32> %load4, i32 2
  %extract14 = extractelement < 8 x i32> %load4, i32 3
  %extract15 = extractelement < 8 x i32> %load4, i32 4
  %extract16 = extractelement < 8 x i32> %load4, i32 5
  %extract17 = extractelement < 8 x i32> %load4, i32 6
  %extract18 = extractelement < 8 x i32> %load4, i32 7
  %tmp70 = mul i32 %ld3, %extract11
  %tmp71 = add i32 %tmp70, %extract12
  %tmp72 = sub i32 %tmp71, %ld0
  store i32 %tmp71, ptr addrspace(1) %p3
  %ld9 = load i32, ptr addrspace(1) %p9
  store i32 %ld3, ptr addrspace(1) %p7
  store i32 %tmp70, ptr addrspace(1) %p9
  %cond1 = icmp uge i32 %ld0, %arg1
  br i1 %cond1, label %bb1, label %bb2

bb1:
  %load1 = load i32, ptr addrspace(1) %p3, align 4
  %load2 = load <8 x i32>, ptr addrspace(1) %p10, align 1
  %extract1 = extractelement < 8 x i32> %load2, i32 1
  %extract2 = extractelement < 8 x i32> %load2, i32 2
  %tmp84 = add i32 %load1, %extract1
  %tmp85 = mul i32 %tmp84, %extract2
  store i32 %tmp85, ptr addrspace(1) %p6
  store i32 %tmp84, ptr addrspace(1) %p8
  store i32 %extract1, ptr addrspace(1) %p9
  store i32 %load1, ptr addrspace(1) %p10
  %tmp101 = mul i32 %load1, %load1
  %tmp102 = sub i32 %load1, %extract1
  %tmp103 = mul i32 %extract1, %extract2
  %tmp104 = sub i32 %load1, %extract2
  %tmp73 = mul i32 %tmp101, %ld0
  %tmp74 = add i32 %tmp73, %ld
  %tmp75 = mul i32 %tmp74, %tmp1
  %tmp76 = add i32 %tmp75, %tmp102
  %tmp77 = sub i32 %tmp76, %tmp103
  %tmp78 = mul i32 %tmp76, %tmp104
  %tmp2 = mul i32 %tmp101, %tmp103
  %idx10 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 1, i64 0
  %val0 = load i32, i32* %idx10, align 4
  %tmp3 = add i32 %val0, %tmp104
  %idx20 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 0, i64 1
  store i32 %tmp3, i32 *%idx20
  %load22 = load <8 x i32>, ptr addrspace(1) %p7, align 1
  %extract3 = extractelement < 8 x i32> %load22, i32 6
  %idx12 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 1, i64 2
  %val2 = load i32, i32* %idx12, align 4
  %tmp4 = mul i32 %val2, %tmp2
  %tmp5= add i32 %tmp3, %tmp103
  %tmp6 = mul i32 %tmp104, %val0
  %tmp7 = sub i32 %tmp6, %tmp4
  %tmp8 = mul i32 %tmp5, %tmp7
  %tmp9 = add i32 %tmp8, %extract3
  %idx22 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 3, i64 2
  store i32 %tmp9, i32 *%idx22
  %extract4 = extractelement < 8 x i32> %load22, i32 7
  %idx13 = getelementptr inbounds [5 x i32], [5 x i32]* @array5, i64 1, i64 0
  %val3 = load i32, i32* %idx13, align 4
  %tmp10 = mul i32 %val3, %extract4
  %tmp11 = add i32 %tmp10, %tmp102
  %tmp12 = sub i32 %val0, %extract4
  %tmp13 = mul i32 %tmp104, %tmp103
  %tmp14 = add i32 %tmp11, %tmp5
  %tmp15 = add i32 %tmp10, %tmp8
  %tmp16 = sub i32 %extract4, %tmp104
  %tmp17 = add i32 %tmp12, %val2
  %tmp18 = add i32 %val0, %tmp9
  %idx601 = getelementptr inbounds [5 x i32], [5 x i32]* @array6, i64 2, i64 1
  %val601 = load i32, i32* %idx601, align 1
  %tmp19 = mul i32 %val601, %tmp12
  %idx701 = getelementptr inbounds [5 x i32], [5 x i32]* @array7, i64 1, i64 0
  %val701 = load i32, i32* %idx701, align 2
  %tmp20 = sub i32 %val701, %tmp11
  %idx801 = getelementptr inbounds [5 x i32], [5 x i32]* @array8, i64 2, i64 1
  %val801 = load i32, i32* %idx801, align 8
  %tmp21 = add i32 %val801, %tmp10
  %idx901 = getelementptr inbounds [5 x i32], [5 x i32]* @array9, i64 1, i64 1
  %val901 = load i32, i32* %idx901, align 1
  %tmp22 = mul i32 %val901, %tmp9
  %idx602 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 4, i64 1
  %val602 = load i32, i32* %idx602, align 1
  %tmp23 = add i32 %val602, %tmp8
  %idx702 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 4, i64 0
  %val702 = load i32, i32* %idx702, align 2
  %tmp24 = sub i32 %val702, %tmp7
  %idx802 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 4, i64 0
  %val802 = load i32, i32* %idx802, align 8
  %tmp25 = add i32 %val802, %tmp6
  %idx902 = getelementptr inbounds [5 x i32], [5 x i32]* @array5, i64 4, i64 2
  %val902 = load i32, i32* %idx902, align 1
  %tmp26 = mul i32 %val902, %tmp5
  %idx800 = getelementptr inbounds [5 x i32], [5 x i32]* @array8, i64 1, i64 0
  %val800 = load i32, i32* %idx800, align 4
  %tmp27 = add i32 %val800, %tmp4
  %idx15 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 0, i64 2
  %val5 = load i32, i32* %idx15, align 4
  %tmp28 = mul i32 %val5, %tmp3
  %idx16 = getelementptr inbounds [5 x i32], [5 x i32]* @array2, i64 0, i64 3
  %val6 = load i32, i32* %idx16, align 4
  %tmp206 = add i32 %val6, %tmp9
  %idx17 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 0, i64 1
  %val7 = load i32, i32* %idx17, align 4
  %tmp207 = add i32 %val7, %tmp3
  %idx18 = getelementptr inbounds [5 x i32], [5 x i32]* @array5, i64 0, i64 1
  %val8 = load i32, i32* %idx18, align 4
  %tmp208 = mul i32 %val8, %tmp10
  %load3 = load <8 x i32>, ptr addrspace(1) %p4, align 1
  %extract7 = extractelement < 8 x i32> %load3, i32 4
  %tmp209 = add i32 %extract7, %tmp11
  %extract8 = extractelement < 8 x i32> %load3, i32 3
  %tmp30 = mul i32 %extract8, %tmp12
  %tmp31 = add i32 %tmp30, %tmp209
  %tmp32 = udiv i32 %tmp31, %tmp208
  %tmp33 = add i32 %tmp32, %tmp207
  %tmp34 = mul i32 %tmp33, %val902
  %tmp35 = sub i32 %tmp34, %val901
  %tmp36 = add i32 %tmp35, %tmp206
  %tmp37 = mul i32 %tmp36, %tmp78
  %tmp38 = add i32 %tmp37, %tmp77
  %tmp39 = sub i32 %tmp38, %ld9
  %tmp40 = udiv i32 %tmp39, %extract4
  store i32 %tmp40, ptr addrspace(3) %p2, align 1
  %tmp41 = sub i32 %tmp40, %val800
  %tmp42 = mul i32 %tmp41, %extract7
  %tmp43 = add i32 %tmp42, %val801
  %tmp44 = mul i32 %tmp43, %val2
  %tmp45 = add i32 %tmp44, %val0
  %tmp46 = sub i32 %tmp45, %tmp2
  %tmp47 = add i32 %tmp46, %tmp28
  %tmp48 = mul i32 %tmp47, %tmp27
  %tmp49 = udiv i32 %tmp48, %tmp26
  %tmp50 = add i32 %tmp49, %tmp25
  %tmp51 = sub i32 %tmp50, %tmp24
  %tmp52 = add i32 %tmp51, %tmp23
  %tmp53 = mul i32 %tmp52, %tmp22
  %tmp54 = add i32 %tmp53, %tmp21
  %tmp55 = sub i32 %tmp54, %tmp20
  %tmp56 = add i32 %tmp55, %tmp19
  %tmp57 = mul i32 %tmp56, %tmp3
  %idx700 = getelementptr inbounds [5 x i32], [5 x i32]* @array7, i64 3, i64 2
  store i32 %tmp57, i32 *%idx700
  %tmp58 = add i32 %tmp57, %tmp18
  %tmp59 = udiv i32 %tmp58, %tmp17
  %tmp60 = mul i32 %tmp59, %tmp16
  %tmp61 = add i32 %tmp60, %tmp15
  %tmp62 = add i32 %tmp61, %tmp14
  %tmp63 = mul i32 %tmp62, %tmp13
  %tmp64 = mul i32 %tmp63, %ld2
  %idx23 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 3, i64 4
  store i32 %tmp64, i32 *%idx23
  %extract27 = extractelement < 8 x i32> %load3, i32 4
  %idx14 = getelementptr inbounds [5 x i32], [5 x i32]* @array3, i64 4, i64 1
  %val4 = load i32, i32* %idx14, align 4
  %tmp65 = add i32 %val4, %extract27
  %tmp66 = sub i32 %tmp65, %tmp2
  %idx24 = getelementptr inbounds [5 x i32], [5 x i32]* @array4, i64 3, i64 5
  store i32 %tmp66, i32 *%idx24
  %extract9 = extractelement < 8 x i32> %load22, i32 0
  %idx600 = getelementptr inbounds [5 x i32], [5 x i32]* @array6, i64 1, i64 2
  %val600 = load i32, i32* %idx600, align 4
  %tmp67 = add i32 %val600, %extract9
  %extract10 = extractelement < 8 x i32> %load22, i32 5
  %tmp68 = sub i32 %extract10, %tmp3
  %tmp69 = add i32 %ld10, %ld6
  %tmp79 = mul i32 %tmp67, %ld8
  store i32 %tmp79, ptr addrspace(1) %p8, align 2
  br label %bb2

bb2:
  %phi1 = phi i32 [ %load1, %bb1 ], [ %tmp72, %entry ]
  %phi2 = phi i32 [ %tmp102, %bb1], [ %ld3, %entry ]
  %phi3 = phi i32 [ %val4, %bb1 ], [ %extract13, %entry ]
  %phi4 = phi i32 [ %tmp104, %bb1 ], [ %tmp71, %entry ]
  %tmp105 = add i32 %phi1, %phi2
  %tmp106 = add i32 %ld8, %phi4
  %tmp107 = mul i32 %tmp105, %tmp106
  %tmp108 = sub i32 %tmp107, %ld6
  %tmp80 = mul i32 %tmp108, %ld2
  %cond3 = icmp ule i32 %ld, %phi3
  br i1 %cond3, label %bb7, label %bb8

bb7:
  %tmp81 = add i32 %phi2, %ld2
  store i32 %tmp81, ptr addrspace(1) %p6
  br label %bb9

bb8:
  %tmp82 = add i32 %phi4, %ld
  store i32 %tmp82, ptr addrspace(1) %p7
  %xor = xor i1 %cond1, %cond3
  br i1 %xor, label %bb9, label %bb10

bb9:
  %phi5 = phi i32 [ %tmp81, %bb7], [%tmp82, %bb8]
  %tmp83 = add i32 %ld1, %ld2
  store i32 %tmp83, ptr addrspace(1) %p1
  br label %bb10

bb10:
  %tmp90 = add i32 %ld10, %ld9
  store i32 %tmp90, ptr addrspace(1) %p9, align 4
  ret void
}
