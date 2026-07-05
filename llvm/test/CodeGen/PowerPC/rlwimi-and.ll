; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-crbits -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s

define void @test() align 2 {
entry:
  br i1 undef, label %codeRepl1, label %codeRepl31

codeRepl1:                                        ; preds = %entry
  br i1 undef, label %codeRepl4, label %codeRepl29

codeRepl4:                                        ; preds = %codeRepl1
  br i1 undef, label %codeRepl12, label %codeRepl17

codeRepl12:                                       ; preds = %codeRepl4
  unreachable

codeRepl17:                                       ; preds = %codeRepl4
  %0 = load i8, ptr undef, align 2
  %1 = and i8 %0, 1
  %not.tobool.i.i.i = icmp eq i8 %1, 0
  %2 = select i1 %not.tobool.i.i.i, i16 0, i16 256
  %3 = load i8, ptr undef, align 1
  %4 = and i8 %3, 1
  %not.tobool.i.1.i.i = icmp eq i8 %4, 0
  %rvml38.sroa.1.1.insert.ext = select i1 %not.tobool.i.1.i.i, i16 0, i16 1
  %rvml38.sroa.0.0.insert.insert = or i16 %rvml38.sroa.1.1.insert.ext, %2
  store i16 %rvml38.sroa.0.0.insert.insert, ptr undef, align 2
  unreachable

; CHECK: @test
; CHECK: clrlwi [[R1:[0-9]+]], {{[0-9]+}}, 31
; CHECK: rlwimi [[R1]], {{[0-9]+}}, 8, 23, 23

codeRepl29:                                       ; preds = %codeRepl1
  unreachable

codeRepl31:                                       ; preds = %entry
  ret void
}

