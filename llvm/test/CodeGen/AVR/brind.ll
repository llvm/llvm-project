; RUN: llc -mattr=sram,eijmpcall < %s -march=avr -verify-machineinstrs | FileCheck %s

@brind.k = private unnamed_addr constant [2 x i8 addrspace(1)*] [i8 addrspace(1)* blockaddress(@brind, %return), i8 addrspace(1)* blockaddress(@brind, %b)], align 1

define i8 @brind(i8 %p) {
; CHECK-LABEL: brind:
; CHECK: ijmp
entry:
  %idxprom = sext i8 %p to i16
  %arrayidx = getelementptr inbounds [2 x i8 addrspace(1)*], [2 x i8 addrspace(1)*]* @brind.k, i16 0, i16 %idxprom
  %s = load i8 addrspace(1)*, i8 addrspace(1)** %arrayidx
  indirectbr i8 addrspace(1)* %s, [label %return, label %b]
b:
  br label %return
return:
  %retval.0 = phi i8 [ 4, %b ], [ 2, %entry ]
  ret i8 %retval.0
}
