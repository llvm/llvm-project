target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"
; This file mainly tests the case that the two input registers of the ISEL instruction are the same register.
; The foldable ISEL in this test case is introduced at simple register coalescing stage.

; Before that stage we have:
; %vreg18<def> = ISEL8 %vreg5, %vreg2, %vreg15<undef>;

; At simple register coalescing stage, the register coalescer figures out it could remove the copy
; from %vreg2 to %vreg5, put the original value %X3 into %vreg5 directly
;  erased: 336r    %vreg5<def> = COPY %vreg2
;  updated: 288B   %vreg5<def> = COPY %X3;

; After that we have:
;   updated: 416B   %vreg18<def> = ISEL8 %vreg5, %vreg5, %vreg15<undef>;

; RUN: llc -verify-machineinstrs -O2 -ppc-asm-full-reg-names -mcpu=pwr7 -ppc-gen-isel=true < %s | FileCheck %s --check-prefix=CHECK-GEN-ISEL-TRUE
; RUN: llc -verify-machineinstrs -O2 -ppc-asm-full-reg-names -mcpu=pwr7 -ppc-gen-isel=false < %s | FileCheck %s --implicit-check-not isel
%"struct.pov::ot_block_struct" = type { %"struct.pov::ot_block_struct"*, [3 x double], [3 x double], float, float, float, float, float, float, float, float, float, [3 x float], float, float, [3 x double], i16 }
%"struct.pov::ot_node_struct" = type { %"struct.pov::ot_id_struct", %"struct.pov::ot_block_struct"*, [8 x %"struct.pov::ot_node_struct"*] }
%"struct.pov::ot_id_struct" = type { i32, i32, i32, i32 }

define void @_ZN3pov6ot_insEPPNS_14ot_node_structEPNS_15ot_block_structEPNS_12ot_id_structE(%"struct.pov::ot_block_struct"* %new_block) {
; CHECK-GEN-ISEL-TRUE-LABEL: _ZN3pov6ot_insEPPNS_14ot_node_structEPNS_15ot_block_structEPNS_12ot_id_structE:
; Note: the following line fold the original isel (isel r4, r3, r3)
; CHECK-GEN-ISEL-TRUE:    mr r4, r3
; CHECK-GEN-ISEL-TRUE:    isel r29, r5, r6, 4*cr5+lt
; CHECK-GEN-ISEL-TRUE:    blr
;
; CHECK-LABEL: _ZN3pov6ot_insEPPNS_14ot_node_structEPNS_15ot_block_structEPNS_12ot_id_structE:
; CHECK:    mr r4, r3
; CHECK:    bc 12, 4*cr5+lt, .LBB0_3
; CHECK:   # %bb.2:
; CHECK:    ori r29, r6, 0
; CHECK:    b .LBB0_4
; CHECK:  .LBB0_3:
; CHECK:    addi r29, r5, 0
; CHECK:  .LBB0_4:
; CHECK:    blr
entry:
  br label %while.cond11

while.cond11:
  %this_node.0250 = phi %"struct.pov::ot_node_struct"* [ undef, %entry ], [ %1, %cond.false21.i156 ], [ %1, %cond.true18.i153 ]
  %temp_id.sroa.21.1 = phi i32 [ undef, %entry ], [ %shr2039.i152, %cond.true18.i153 ], [ %div24.i155, %cond.false21.i156 ]
  %0 = load i32, i32* undef, align 4
  %cmp17 = icmp eq i32 0, %0
  br i1 %cmp17, label %lor.rhs, label %while.body21

lor.rhs:
  %Values = getelementptr inbounds %"struct.pov::ot_node_struct", %"struct.pov::ot_node_struct"* %this_node.0250, i64 0, i32 1
  store %"struct.pov::ot_block_struct"* %new_block, %"struct.pov::ot_block_struct"** %Values, align 8
  ret void

while.body21:
  %call.i84 = tail call i8* @ZN3pov10pov_callocEmmPKciS1_pov()
  store i8* %call.i84, i8** undef, align 8
  %1 = bitcast i8* %call.i84 to %"struct.pov::ot_node_struct"*
  br i1 undef, label %cond.true18.i153, label %cond.false21.i156

cond.true18.i153:
  %shr2039.i152 = lshr i32 %temp_id.sroa.21.1, 1
  br label %while.cond11

cond.false21.i156:
  %add23.i154 = add nsw i32 %temp_id.sroa.21.1, 1
  %div24.i155 = sdiv i32 %add23.i154, 2
  br label %while.cond11
}

declare i8* @ZN3pov10pov_callocEmmPKciS1_pov()
