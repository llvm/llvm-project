; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

;
;       bb.0.entry
;        /    |
;   bb.3.bb2  |
;        \    |
;       bb.1.Flow
;        /    |
;   bb.2.bb1  |
;        \    |
;      bb.4.exit
define amdgpu_ps i64 @test1(ptr addrspace(3) %p1, ptr addrspace(3) %p2, i1 %cond1, i64 %val) {
; CHECK-LABEL: # Machine code for function test1: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.3(0x40000000), %bb.1(0x40000000); %bb.3(50.00%), %bb.1(50.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4
; CHECK-NEXT:   [[Reg5]]:vgpr_32 = COPY killed $vgpr4
; CHECK-NEXT:   [[Reg4]]:vgpr_32 = COPY killed $vgpr3
; CHECK-NEXT:   [[Reg3]]:vgpr_32 = COPY killed $vgpr2
; CHECK-NEXT:   [[Reg2]]:vgpr_32 = COPY killed $vgpr1
; CHECK-NEXT:   [[Reg1]]:vgpr_32 = COPY killed $vgpr0
; CHECK-NEXT:   [[Reg6:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 1, killed [[Reg3]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg7:%[0-9]+]]:sreg_32 = V_CMP_NE_U32_e64 1, killed [[Reg6]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg8:%[0-9]+]]:sreg_32 = SI_IF killed [[Reg7]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.3
; EMPTY:
; CHECK: bb.1.Flow:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.3
; CHECK-NEXT:   successors: %bb.2(0x40000000), %bb.4(0x40000000); %bb.2(50.00%), %bb.4(50.00%)
; EMPTY:
; CHECK:   [[Reg9:%[0-9]+]]:vreg_64 = PHI undef [[Reg10:%[0-9]+]]:vreg_64, %bb.0, [[Reg11:%[0-9]+]]:vreg_64, %bb.3
; CHECK-NEXT:   [[Reg12:%[0-9]+]]:vgpr_32 = PHI [[Reg1]]:vgpr_32, %bb.0, undef [[Reg13:%[0-9]+]]:vgpr_32, %bb.3
; CHECK-NEXT:   [[Reg14:%[0-9]+]]:sreg_32 = SI_ELSE killed [[Reg8]]:sreg_32, %bb.4, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.2
; EMPTY:
; CHECK: bb.2.bb1:
; CHECK-NEXT: ; predecessors: %bb.1
; CHECK-NEXT:   successors: %bb.4(0x80000000); %bb.4(100.00%)
; EMPTY:
; CHECK:   [[Reg15:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg12]]:vgpr_32, 0, 0, implicit $exec :: (load (s8) from %ir.p1, addrspace 3)
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg12]]:vgpr_32, 1, 0, implicit $exec :: (load (s8) from %ir.p1 + 1, addrspace 3)
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg12]]:vgpr_32, 2, 0, implicit $exec :: (load (s8) from %ir.p1 + 2, addrspace 3)
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg12]]:vgpr_32, 3, 0, implicit $exec :: (load (s8) from %ir.p1 + 3, addrspace 3)
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg12]]:vgpr_32, 4, 0, implicit $exec :: (load (s8) from %ir.p1 + 4, addrspace 3)
; CHECK-NEXT:   [[Reg20:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg12]]:vgpr_32, 5, 0, implicit $exec :: (load (s8) from %ir.p1 + 5, addrspace 3)
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg12]]:vgpr_32, 6, 0, implicit $exec :: (load (s8) from %ir.p1 + 6, addrspace 3)
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 killed [[Reg12]]:vgpr_32, 7, 0, implicit $exec :: (load (s8) from %ir.p1 + 7, addrspace 3)
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg20]]:vgpr_32, 8, killed [[Reg19]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg22]]:vgpr_32, 8, killed [[Reg21]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg24]]:vgpr_32, 16, killed [[Reg23]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg16]]:vgpr_32, 8, killed [[Reg15]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg18]]:vgpr_32, 8, killed [[Reg17]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg27]]:vgpr_32, 16, killed [[Reg26]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg28]]:vgpr_32, %subreg.sub0, killed [[Reg25]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vreg_64 = COPY killed [[Reg29]]:vreg_64
; CHECK-NEXT:   S_BRANCH %bb.4
; EMPTY:
; CHECK: bb.3.bb2:
; CHECK-NEXT: ; predecessors: %bb.0
; CHECK-NEXT:   successors: %bb.1(0x80000000); %bb.1(100.00%)
; EMPTY:
; CHECK:   [[Reg31:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 0, 0, implicit $exec :: (load (s8) from %ir.p2, addrspace 3)
; CHECK-NEXT:   [[Reg32:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 1, 0, implicit $exec :: (load (s8) from %ir.p2 + 1, addrspace 3)
; CHECK-NEXT:   [[Reg33:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 2, 0, implicit $exec :: (load (s8) from %ir.p2 + 2, addrspace 3)
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 3, 0, implicit $exec :: (load (s8) from %ir.p2 + 3, addrspace 3)
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 4, 0, implicit $exec :: (load (s8) from %ir.p2 + 4, addrspace 3)
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 5, 0, implicit $exec :: (load (s8) from %ir.p2 + 5, addrspace 3)
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 [[Reg2]]:vgpr_32, 6, 0, implicit $exec :: (load (s8) from %ir.p2 + 6, addrspace 3)
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:vgpr_32 = DS_READ_U8_gfx9 killed [[Reg2]]:vgpr_32, 7, 0, implicit $exec :: (load (s8) from %ir.p2 + 7, addrspace 3)
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg36]]:vgpr_32, 8, killed [[Reg35]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg38]]:vgpr_32, 8, killed [[Reg37]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg40]]:vgpr_32, 16, killed [[Reg39]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg32]]:vgpr_32, 8, killed [[Reg31]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg34]]:vgpr_32, 8, killed [[Reg33]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg43]]:vgpr_32, 16, killed [[Reg42]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg45:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg44]]:vgpr_32, %subreg.sub0, killed [[Reg41]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg11]]:vreg_64 = COPY killed [[Reg45]]:vreg_64
; CHECK-NEXT:   S_BRANCH %bb.1
; EMPTY:
; CHECK: bb.4.exit:
; CHECK-NEXT: ; predecessors: %bb.1, %bb.2
; EMPTY:
; CHECK:   [[Reg46:%[0-9]+]]:vreg_64 = PHI [[Reg9]]:vreg_64, %bb.1, [[Reg30]]:vreg_64, %bb.2
; CHECK-NEXT:   SI_END_CF killed [[Reg14]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32, [[Reg48:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg46]].sub0:vreg_64, killed [[Reg4]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg46]].sub1:vreg_64, killed [[Reg5]]:vgpr_32, killed [[Reg48]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 killed [[Reg47]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 killed [[Reg49]]:vgpr_32, implicit $exec
; CHECK-NEXT:   $sgpr0 = COPY killed [[Reg50]]:sreg_32_xm0
; CHECK-NEXT:   $sgpr1 = COPY killed [[Reg51]]:sreg_32_xm0
; CHECK-NEXT:   SI_RETURN_TO_EPILOG killed $sgpr0, killed $sgpr1
; EMPTY:
; CHECK: # End machine code for function test1.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg5]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 14.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 6.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg11]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 2.0
entry:
;    entry
;    /   \
;  bb1   bb2
;    \   /
;     exit
   br i1 %cond1, label %bb1, label %bb2

bb1:
  %ld1 = load i64, ptr addrspace(3) %p1, align 1
  br label %exit

bb2:
  %ld2 = load i64, ptr addrspace(3) %p2, align 1
  br label %exit

exit:
  %phi = phi i64 [ %ld1, %bb1 ], [ %ld2, %bb2 ]
  %add = add i64 %phi, %val
  ret i64 %add
}
