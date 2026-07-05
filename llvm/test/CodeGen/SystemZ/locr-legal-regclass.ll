; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 -verify-machineinstrs | FileCheck %s
;
; Test that early if conversion produces LOCR with operands of the right
; register classes.

define void @autogen_SD4739(ptr) {
; CHECK-NOT: Expected a GR32Bit register, but got a GRX32Bit register
BB:
  %L34 = load i8, ptr %0
  %Cmp56 = icmp sgt i8 undef, %L34
  br label %CF246

CF246:                                            ; preds = %CF246, %BB
  %Sl163 = select i1 %Cmp56, i8 %L34, i8 undef
  br i1 undef, label %CF246, label %CF248

CF248:                                            ; preds = %CF248, %CF246
  store i8 %Sl163, ptr %0
  br label %CF248
}
