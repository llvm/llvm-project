; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -enable-next-use-analysis=true -verify-machineinstrs -dump-next-use-distance < %s 2>&1 | FileCheck %s

;
;       bb.0.entry
;           |
;           +<-----+
;       bb.1.loop  |
;           +------+
;           |
;       bb.2.exit
;
define amdgpu_ps void @test8(ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3, ptr addrspace(1) %p4, ptr addrspace(1) %p5, i32 %TC) {
; CHECK-LABEL: # Machine code for function test8: IsSSA, TracksLiveness
; CHECK-NEXT: Function Live Ins: $vgpr0 in [[Reg1:%[0-9]+]], $vgpr1 in [[Reg2:%[0-9]+]], $vgpr2 in [[Reg3:%[0-9]+]], $vgpr3 in [[Reg4:%[0-9]+]], $vgpr4 in [[Reg5:%[0-9]+]], $vgpr5 in [[Reg6:%[0-9]+]], $vgpr6 in [[Reg7:%[0-9]+]], $vgpr7 in [[Reg8:%[0-9]+]], $vgpr8 in [[Reg9:%[0-9]+]], $vgpr9 in [[Reg10:%[0-9]+]], $vgpr10 in [[Reg11:%[0-9]+]]
; EMPTY:
; CHECK: bb.0.entry:
; CHECK-NEXT:   successors: %bb.1(0x80000000); %bb.1(100.00%)
; CHECK-NEXT:   liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr4, $vgpr5, $vgpr6, $vgpr7, $vgpr8, $vgpr9, $vgpr10
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
; CHECK-NEXT:   [[Reg12:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg9]]:vgpr_32, %subreg.sub0, killed [[Reg10]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg13:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg7]]:vgpr_32, %subreg.sub0, killed [[Reg8]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg14:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg5]]:vgpr_32, %subreg.sub0, killed [[Reg6]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg15:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg3]]:vgpr_32, %subreg.sub0, killed [[Reg4]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg16]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p2, addrspace 1)
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg16]]:vreg_64, [[Reg11]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p2, addrspace 1)
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:sreg_32 = S_MOV_B32 -1
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.1.loop:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.1
; CHECK-NEXT:   successors: %bb.2(0x04000000), %bb.1(0x7c000000); %bb.2(3.12%), %bb.1(96.88%)
; EMPTY:
; CHECK:   [[Reg20:%[0-9]+]]:sreg_32 = PHI [[Reg19]]:sreg_32, %bb.0, [[Reg21:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:sreg_32 = PHI [[Reg18]]:sreg_32, %bb.0, [[Reg23:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg23]]:sreg_32 = S_ADD_I32 [[Reg22]]:sreg_32, 1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:sreg_32 = S_ADD_I32 killed [[Reg22]]:sreg_32, 2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg24]]:sreg_32, [[Reg17]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg21]]:sreg_32 = SI_IF_BREAK killed [[Reg25]]:sreg_32, killed [[Reg20]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:vgpr_32 = COPY [[Reg23]]:sreg_32, implicit $exec
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:vgpr_32 = COPY killed [[Reg24]]:sreg_32, implicit $exec
; CHECK-NEXT:   SI_LOOP [[Reg21]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.2
; EMPTY:
; CHECK: bb.2.exit:
; CHECK-NEXT: ; predecessors: %bb.1
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg21]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:vgpr_32 = V_ASHRREV_I32_e64 31, [[Reg26]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:vreg_64 = REG_SEQUENCE [[Reg26]]:vgpr_32, %subreg.sub0, killed [[Reg28]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vreg_64 = nsw V_LSHLREV_B64_pseudo_e64 2, killed [[Reg29]]:vreg_64, implicit $exec
; CHECK-NEXT:   [[Reg31:%[0-9]+]]:vgpr_32, [[Reg32:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg15]].sub0:vreg_64, [[Reg30]].sub0:vreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg33:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg15]].sub1:vreg_64, killed [[Reg30]].sub1:vreg_64, killed [[Reg32]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg31]]:vgpr_32, %subreg.sub0, killed [[Reg33]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg35:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg34]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.gep.le, addrspace 1)
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg34]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.gep.le + 1, addrspace 1)
; CHECK-NEXT:   [[Reg37:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg36]]:vgpr_32, 8, killed [[Reg35]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg34]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.gep.le + 2, addrspace 1)
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE killed [[Reg34]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.gep.le + 3, addrspace 1)
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg39]]:vgpr_32, 8, killed [[Reg38]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg40]]:vgpr_32, 16, killed [[Reg37]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg42:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p3 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg43]]:vgpr_32, 8, killed [[Reg42]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg45:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p3 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg14]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p3 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg46]]:vgpr_32, 8, killed [[Reg45]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg47]]:vgpr_32, 16, killed [[Reg44]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg13]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg49]]:vgpr_32, killed [[Reg27]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg49]]:vgpr_32, [[Reg50]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg14]]:vreg_64, [[Reg51]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg51]]:vgpr_32, killed [[Reg50]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg53:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg52]]:vgpr_32, killed [[Reg11]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg13]]:vreg_64, killed [[Reg53]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg54:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg17]]:vgpr_32, killed [[Reg48]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vgpr_32 = V_ADD3_U32_e64 killed [[Reg41]]:vgpr_32, killed [[Reg26]]:vgpr_32, killed [[Reg54]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg12]]:vreg_64, killed [[Reg55]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p5, addrspace 1)
; CHECK-NEXT:   S_ENDPGM 0
; EMPTY:
; CHECK: # End machine code for function test8.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg11]] = 17.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 10039.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 10029.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 10021.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 10010.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 5.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 10002.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 10023.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 16.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg54]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg55]] = 1.0
entry:
;     entry
;       |
;       +<----+
;      loop   |
;       +-----+
;       |
;      exit
   %ld1 = load i32, ptr addrspace(1) %p2, align 4
   store i32 %TC, ptr addrspace(1) %p2
   br label %loop

loop:
   %phi = phi i32 [ 100, %entry ], [ %add, %loop ]
   %phi.inc = phi i32 [ 0, %entry ], [ %inc, %loop ]
   %sext = sext i32 %phi.inc to i64
   %gep = getelementptr inbounds i32, ptr addrspace(1) %p1, i64 %sext
   %ld = load i32, ptr addrspace(1) %gep, align 1
   %add = add i32 %ld, %phi.inc
   %inc = add i32 %phi.inc, 1
   %cond = icmp ult i32 %inc, %ld1
   br i1 %cond, label %loop, label %exit

exit:
   %ld2 = load i32, ptr addrspace(1) %p3, align 1
   %ld3 = load i32, ptr addrspace(1) %p4
   %add1 = add i32 %ld3, %inc
   %mul1 = mul i32 %ld3, %add1
   store i32 %mul1, ptr addrspace(1) %p3
   %sub1 = sub i32 %mul1, %add1
   %mul2 = mul i32 %sub1, %TC
   store i32 %mul2, ptr addrspace(1) %p4
   %mul3 = mul i32 %ld1, %ld2
   %add2 = add i32 %mul3, %add
   store i32 %add2, ptr addrspace(1) %p5
   ret void
}
