; Report error when pass requires TargetMachine.
; RUN: not opt -passes=codegenprepare -disable-output %s 2>&1 | FileCheck %s
define void @foo() { ret void }
; CHECK: Pass 'codegenprepare' requires TargetMachine
