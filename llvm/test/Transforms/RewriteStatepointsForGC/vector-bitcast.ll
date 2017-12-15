; RUN: opt -S -rewrite-statepoints-for-gc < %s | FileCheck %s
; RUN: opt -S -passes=rewrite-statepoints-for-gc < %s | FileCheck %s
;
; A test to make sure that we can look through bitcasts of
; vector types when a base pointer is contained in a vector.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
define void @test() gc "statepoint-example" {
; CHECK-LABEL: @test
entry:
; CHECK-LABEL: entry
; CHECK: %bc = bitcast
; CHECK: %[[p1:[A-Za-z0-9_]+]] = extractelement
; CHECK: %[[p2:[A-Za-z0-9_]+]] = extractelement
; CHECK: llvm.experimental.gc.statepoint
; CHECK: %[[p2]].relocated = {{.+}} @llvm.experimental.gc.relocate
; CHECK: %[[p1]].relocated = {{.+}} @llvm.experimental.gc.relocate
; CHECK: load atomic
  %bc = bitcast <8 x i8 addrspace(1)*> undef to <8 x i32 addrspace(1)*>
  %ptr= extractelement <8 x i32 addrspace(1)*> %bc, i32 7
  %0 = call i8 addrspace(1)* undef() [ "deopt"() ]
  %1 = load atomic i32, i32 addrspace(1)* %ptr unordered, align 4
  unreachable
}
