; Report error when pass requires TargetMachine.
; RUN: not opt -passes=atomic-expand -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=codegenprepare -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=complex-deinterleaving -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=dwarf-eh-prepare -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=expand-large-div-rem -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=expand-memcmp -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=indirectbr-expand -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=interleaved-access -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=interleaved-load-combine -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=safe-stack -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=select-optimize -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=stack-protector -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes=typepromotion -disable-output %s 2>&1 | FileCheck %s
; RUN: not opt -passes='expand-fp<O1>' -disable-output %s 2>&1 | FileCheck %s
define void @foo() { ret void }
; CHECK: pass '{{.+}}' requires TargetMachine
;requires TargetMachine
