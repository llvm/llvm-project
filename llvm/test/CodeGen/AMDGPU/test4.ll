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
define amdgpu_ps void @test4(ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3, ptr addrspace(1) %p4, ptr addrspace(1) %p5, i32 %TC) {
; CHECK-LABEL: # Machine code for function test4: IsSSA, TracksLiveness
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
; CHECK-NEXT:   [[Reg13:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg1]]:vgpr_32, %subreg.sub0, killed [[Reg2]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg14:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg7]]:vgpr_32, %subreg.sub0, killed [[Reg8]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg15:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg5]]:vgpr_32, %subreg.sub0, killed [[Reg6]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg16:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg3]]:vgpr_32, %subreg.sub0, killed [[Reg4]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg17:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg16]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p2, addrspace 1)
; CHECK-NEXT:   [[Reg18:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg15]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg19:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg15]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.p3 + 1, addrspace 1)
; CHECK-NEXT:   [[Reg20:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg19]]:vgpr_32, 8, killed [[Reg18]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg21:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg15]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.p3 + 2, addrspace 1)
; CHECK-NEXT:   [[Reg22:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg15]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.p3 + 3, addrspace 1)
; CHECK-NEXT:   [[Reg23:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg22]]:vgpr_32, 8, killed [[Reg21]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg24:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg23]]:vgpr_32, 16, killed [[Reg20]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg25:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD [[Reg14]]:vreg_64, 0, 0, implicit $exec :: (load (s32) from %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg26:%[0-9]+]]:vgpr_32 = V_ADD_U32_e64 [[Reg25]]:vgpr_32, [[Reg11]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg16]]:vreg_64, [[Reg26]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p2, addrspace 1)
; CHECK-NEXT:   [[Reg27:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg25]]:vgpr_32, [[Reg26]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg15]]:vreg_64, [[Reg27]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p3, addrspace 1)
; CHECK-NEXT:   [[Reg28:%[0-9]+]]:vgpr_32 = V_SUB_U32_e64 killed [[Reg27]]:vgpr_32, killed [[Reg26]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg29:%[0-9]+]]:vgpr_32 = V_MUL_LO_U32_e64 killed [[Reg28]]:vgpr_32, [[Reg11]]:vgpr_32, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg14]]:vreg_64, killed [[Reg29]]:vgpr_32, 0, 0, implicit $exec :: (store (s32) into %ir.p4, addrspace 1)
; CHECK-NEXT:   [[Reg30:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 100, implicit $exec
; CHECK-NEXT:   [[Reg31:%[0-9]+]]:sreg_32 = S_MOV_B32 0
; EMPTY:
; CHECK: bb.1.loop:
; CHECK-NEXT: ; predecessors: %bb.0, %bb.1
; CHECK-NEXT:   successors: %bb.2(0x04000000), %bb.1(0x7c000000); %bb.2(3.12%), %bb.1(96.88%)
; EMPTY:
; CHECK:   [[Reg32:%[0-9]+]]:sreg_32 = PHI [[Reg31]]:sreg_32, %bb.0, [[Reg33:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg34:%[0-9]+]]:vgpr_32 = PHI [[Reg30]]:vgpr_32, %bb.0, [[Reg35:%[0-9]+]]:vgpr_32, %bb.1
; CHECK-NEXT:   [[Reg36:%[0-9]+]]:sreg_32 = PHI [[Reg31]]:sreg_32, %bb.0, [[Reg37:%[0-9]+]]:sreg_32, %bb.1
; CHECK-NEXT:   [[Reg38:%[0-9]+]]:sreg_32_xm0 = S_ASHR_I32 [[Reg36]]:sreg_32, 31, implicit-def dead $scc
; CHECK-NEXT:   [[Reg39:%[0-9]+]]:sreg_64 = REG_SEQUENCE [[Reg36]]:sreg_32, %subreg.sub0, killed [[Reg38]]:sreg_32_xm0, %subreg.sub1
; CHECK-NEXT:   [[Reg40:%[0-9]+]]:sreg_64 = nsw S_LSHL_B64 killed [[Reg39]]:sreg_64, 2, implicit-def dead $scc
; CHECK-NEXT:   [[Reg41:%[0-9]+]]:vgpr_32, [[Reg42:%[0-9]+]]:sreg_32_xm0_xexec = V_ADD_CO_U32_e64 [[Reg13]].sub0:vreg_64, [[Reg40]].sub0:sreg_64, 0, implicit $exec
; CHECK-NEXT:   [[Reg43:%[0-9]+]]:vgpr_32, dead $sgpr_null = V_ADDC_U32_e64 killed [[Reg40]].sub1:sreg_64, [[Reg13]].sub1:vreg_64, killed [[Reg42]]:sreg_32_xm0_xexec, 0, implicit $exec
; CHECK-NEXT:   [[Reg44:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg41]]:vgpr_32, %subreg.sub0, killed [[Reg43]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg45:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg44]]:vreg_64, 0, 0, implicit $exec :: (load (s8) from %ir.gep, addrspace 1)
; CHECK-NEXT:   [[Reg46:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg44]]:vreg_64, 1, 0, implicit $exec :: (load (s8) from %ir.gep + 1, addrspace 1)
; CHECK-NEXT:   [[Reg47:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg46]]:vgpr_32, 8, killed [[Reg45]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg48:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE [[Reg44]]:vreg_64, 2, 0, implicit $exec :: (load (s8) from %ir.gep + 2, addrspace 1)
; CHECK-NEXT:   [[Reg49:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_UBYTE killed [[Reg44]]:vreg_64, 3, 0, implicit $exec :: (load (s8) from %ir.gep + 3, addrspace 1)
; CHECK-NEXT:   [[Reg50:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg49]]:vgpr_32, 8, killed [[Reg48]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg51:%[0-9]+]]:vgpr_32 = V_LSHL_OR_B32_e64 killed [[Reg50]]:vgpr_32, 16, killed [[Reg47]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg35]]:vgpr_32 = V_ADD_U32_e64 [[Reg36]]:sreg_32, killed [[Reg51]]:vgpr_32, 0, implicit $exec
; CHECK-NEXT:   [[Reg37]]:sreg_32 = S_ADD_I32 killed [[Reg36]]:sreg_32, 1, implicit-def dead $scc
; CHECK-NEXT:   [[Reg52:%[0-9]+]]:sreg_32 = V_CMP_GE_U32_e64 [[Reg37]]:sreg_32, [[Reg11]]:vgpr_32, implicit $exec
; CHECK-NEXT:   [[Reg33]]:sreg_32 = SI_IF_BREAK killed [[Reg52]]:sreg_32, killed [[Reg32]]:sreg_32, implicit-def dead $scc
; CHECK-NEXT:   SI_LOOP [[Reg33]]:sreg_32, %bb.1, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   S_BRANCH %bb.2
; EMPTY:
; CHECK: bb.2.exit:
; CHECK-NEXT: ; predecessors: %bb.1
; EMPTY:
; CHECK:   SI_END_CF killed [[Reg33]]:sreg_32, implicit-def dead $exec, implicit-def dead $scc, implicit $exec
; CHECK-NEXT:   [[Reg53:%[0-9]+]]:vreg_64 = REG_SEQUENCE killed [[Reg24]]:vgpr_32, %subreg.sub0, undef [[Reg54:%[0-9]+]]:vgpr_32, %subreg.sub1
; CHECK-NEXT:   [[Reg55:%[0-9]+]]:vreg_64, $sgpr_null = V_MAD_U64_U32_e64 killed [[Reg17]]:vgpr_32, killed [[Reg34]]:vgpr_32, killed [[Reg53]]:vreg_64, 0, implicit $exec
; CHECK-NEXT:   GLOBAL_STORE_DWORD killed [[Reg12]]:vreg_64, killed [[Reg55]].sub0:vreg_64, 0, 0, implicit $exec :: (store (s32) into %ir.p5, addrspace 1)
; CHECK-NEXT:   S_ENDPGM 0
; EMPTY:
; CHECK: # End machine code for function test4.
; EMPTY:
; CHECK: Next-use distance of Register [[Reg11]] = 25.0
; CHECK-NEXT: Next-use distance of Register [[Reg10]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg9]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg8]] = 10.0
; CHECK-NEXT: Next-use distance of Register [[Reg7]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg6]] = 9.0
; CHECK-NEXT: Next-use distance of Register [[Reg5]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg4]] = 8.0
; CHECK-NEXT: Next-use distance of Register [[Reg3]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg2]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg1]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg12]] = 22026.0
; CHECK-NEXT: Next-use distance of Register [[Reg13]] = 28.0
; CHECK-NEXT: Next-use distance of Register [[Reg14]] = 11.0
; CHECK-NEXT: Next-use distance of Register [[Reg15]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg16]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg17]] = 22020.0
; CHECK-NEXT: Next-use distance of Register [[Reg18]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg19]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg20]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg21]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg22]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg23]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg24]] = 22012.0
; CHECK-NEXT: Next-use distance of Register [[Reg25]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg26]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg27]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg28]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg29]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg30]] = 3.0
; CHECK-NEXT: Next-use distance of Register [[Reg31]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg32]] = 19.0
; CHECK-NEXT: Next-use distance of Register [[Reg34]] = 22003.0
; CHECK-NEXT: Next-use distance of Register [[Reg36]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg38]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg39]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg40]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg41]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg42]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg43]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg44]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg45]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg46]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg47]] = 4.0
; CHECK-NEXT: Next-use distance of Register [[Reg48]] = 2.0
; CHECK-NEXT: Next-use distance of Register [[Reg49]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg50]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg51]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg35]] = 7.0
; CHECK-NEXT: Next-use distance of Register [[Reg37]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg52]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg33]] = 1.0
; CHECK-NEXT: Next-use distance of Register [[Reg53]] = 1.0
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
   %ld2 = load i32, ptr addrspace(1) %p3, align 1
   %ld3 = load i32, ptr addrspace(1) %p4
   %add1 = add i32 %ld3, %TC
   store i32 %add1, ptr addrspace(1) %p2
   %mul1 = mul i32 %ld3, %add1
   store i32 %mul1, ptr addrspace(1) %p3
   %sub1 = sub i32 %mul1, %add1
   %mul2 = mul i32 %sub1, %TC
   store i32 %mul2, ptr addrspace(1) %p4
   br label %loop

loop:
   %phi = phi i32 [ 100, %entry ], [ %add, %loop ]
   %phi.inc = phi i32 [ 0, %entry ], [ %inc, %loop ]
   %sext = sext i32 %phi.inc to i64
   %gep = getelementptr inbounds i32, ptr addrspace(1) %p1, i64 %sext
   %ld = load i32, ptr addrspace(1) %gep, align 1
   %add = add i32 %ld, %phi.inc
   %inc = add i32 %phi.inc, 1
   %cond = icmp ult i32 %inc, %TC
   br i1 %cond, label %loop, label %exit

exit:
   %mul3 = mul i32 %ld1, %phi
   %add2 = add i32 %mul3, %ld2
   store i32 %add2, ptr addrspace(1) %p5
   ret void
}
