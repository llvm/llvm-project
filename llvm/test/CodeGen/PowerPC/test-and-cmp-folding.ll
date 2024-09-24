; RUN: llc < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 \
; RUN:   -verify-machineinstrs | FileCheck %s

; test folding and + cmp to and + bc

; CHECK-LABEL: test
define dso_local fastcc void @test(i64 %v1) {
entry:
; CHECK: andi.
; CHECK-NEXT: bc
  %and1 = and i64 %v1, 1
  %cmp1 = icmp eq i64 %and1, 0
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  call fastcc void @test2()
  ret void

if.end:
  call fastcc void @test3()
  ret void
}

declare dso_local fastcc void @test2()
declare dso_local fastcc void @test3()
