; RUN: llc -mattr=sram,eijmpcall < %s -mtriple=avr -verify-machineinstrs | FileCheck %s

@brind.k = private unnamed_addr constant [2 x ptr addrspace(1)] [ptr addrspace(1) blockaddress(@brind, %return), ptr addrspace(1) blockaddress(@brind, %b)], align 1

define i8 @brind(i8 %p) {
; CHECK-LABEL: brind:
; CHECK: ijmp
entry:
  %idxprom = sext i8 %p to i16
  %arrayidx = getelementptr inbounds [2 x ptr addrspace(1)], ptr @brind.k, i16 0, i16 %idxprom
  %s = load ptr addrspace(1), ptr %arrayidx
  indirectbr ptr addrspace(1) %s, [label %return, label %b]
b:
  br label %return
return:
  %retval.0 = phi i8 [ 4, %b ], [ 2, %entry ]
  ret i8 %retval.0
}
