; NOTE: This test case aims to test the sequence of spilling the CR[0-7]LT bits
; NOTE: on POWER9 using the setb instruction.

; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=pwr9 < %s \
; RUN:     | FileCheck %s --check-prefix=CHECK-P9
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=pwr9 < %s \
; RUN:     | FileCheck %s --check-prefix=CHECK-P9
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=pwr8 < %s \
; RUN:     | FileCheck %s --check-prefix=CHECK-P8
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=pwr8 < %s \
; RUN:     | FileCheck %s --check-prefix=CHECK-P8

define void @p9_setb_spill() {
; CHECK-P9-LABEL: p9_setb_spill:
; CHECK-P9:       # %bb.1: # %if.then
; CHECK-P9-DAG:    crnot 4*cr[[CREG:.*]]+lt, eq
; CHECK-P9-DAG:    setb [[REG1:.*]], cr[[CREG]]
; CHECK-P9-DAG:    stw [[REG1]]
; CHECK-P9:        blr
; CHECK-P9:        .LBB0_4: # %if.then1
;
; CHECK-P8-LABEL: p9_setb_spill:
; CHECK-P8:       # %bb.1: # %if.then
; CHECK-P8-DAG:    crnot 4*cr[[CREG2:.*]]+lt, eq
; CHECK-P8-DAG:    mfocrf [[REG2:.*]],
; CHECK-P8-DAG:    rlwinm [[REG2]], [[REG2]]
; CHECK-P8-DAG:    stw [[REG2]]
; CHECK-P8:        blr
; CHECK-P8:        .LBB0_4: # %if.then1
entry:
  br i1 undef, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call signext i32 @fn_call()
  %cmp1 = icmp ne i32 %call, 0
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %off0 = phi i1 [ %cmp1, %if.then ], [ false, %entry ]
  tail call void asm sideeffect "#Clobber", "~{cr0},~{cr1},~{cr2},~{cr3},~{cr4},~{cr5},~{cr6},~{cr7}"()
  %off0.not = xor i1 %off0, true
  %or = or i1 false, %off0.not
  br i1 %or, label %if.end2, label %if.then1

if.then1:                                         ; preds = %if.end
  unreachable

if.end2:                                         ; preds = %if.end
  ret void
}

declare signext i32 @fn_call(...)
