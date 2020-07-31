; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -target-abi=elfv2 -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s
define dso_local void @blockaddress() {
; CHECK-LABEL: blockaddress:
; CHECK:       # %bb.0: # %entry
; CHECK:       paddi r3, 0, .Ltmp0@PCREL, 1
; CHECK:       bl helper@notoc
entry:
  tail call void @helper(i8* blockaddress(@blockaddress, %label))
  br label %label

label:                                            ; preds = %entry
  ret void
}

declare void @helper(i8*)
