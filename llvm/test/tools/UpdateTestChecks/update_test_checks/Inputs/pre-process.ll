; Test that update_test_checks.py can run pre-processing commands.
; RUN: opt < %s -passes=instsimplify -S | FileCheck %s --check-prefix=CHECK-AS200
; RUN: sed -e 's/addrspace(200)/addrspace(0)/g' -e 's/-A200-P200-G200//g' %s \
; RUN:   | opt -passes=instsimplify -S | FileCheck %s --check-prefix=CHECK-AS0
; Check that multiple pre-processing commands are handled
; RUN: sed 's/addrspace(200)/addrspace(1)/g' %s | sed 's/-A1-P1-G1//g' \
; RUN:   | opt -passes=instsimplify -S | FileCheck %s --check-prefix=CHECK-AS1
; More than two commands should also be fine
; RUN: cat %s | cat | cat | cat | opt < %s -passes=instsimplify -S \
; RUN:   | FileCheck %s --check-prefix=CHECK-AS200-NOOP-PRE-PROCESS

target datalayout = "e-m:e-p200:128:128:128:64-p:64:64-A200-P200-G200"

define ptr addrspace(200) @test_zerogep_in_different_as(ptr addrspace(200) %arg) addrspace(200) nounwind {
entry:
  ret ptr addrspace(200) %arg
}
