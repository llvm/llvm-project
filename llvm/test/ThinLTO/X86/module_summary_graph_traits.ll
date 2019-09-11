; RUN: opt -module-summary %s -o %t1.bc
; RUN: llvm-lto2 run -print-summary-global-ids -dump-thin-cg-sccs %t1.bc -o %t.index.bc \
; RUN:     -r %t1.bc,external,px -r %t1.bc,l2,pl -r %t1.bc,l1,pl \
; RUN:     -r %t1.bc,simple,pl -r %t1.bc,root,pl 2>&1 | FileCheck %s

; CHECK: 5224464028922159466{{.*}} is external
; CHECK: 765152853862302398{{.*}} is l2
; CHECK: 17000277804057984823{{.*}} is l1
; CHECK: 15440740835768581517{{.*}} is simple
; CHECK: 5800840261926955363{{.*}} is root

; CHECK: SCC (2 nodes) {
; CHECK-NEXT: {{^}} 17000277804057984823 (has loop)
; CHECK-NEXT: {{^}} 765152853862302398 (has loop)
; CHECK-NEXT: }

; CHECK: SCC (1 node) {
; CHECK-NEXT: {{^}} 15440740835768581517{{$}}
; CHECK-NEXT: }

; CHECK: SCC (1 node) {
; CHECK-NEXT: External 5224464028922159466{{$}}
; CHECK-NEXT: }

; CHECK: SCC (1 node) {
; CHECK-NEXT: {{^}} 5800840261926955363{{$}}
; CHECK-NEXT: }

; Dummy call graph root that points at all roots of the callgraph.
; CHECK: SCC (1 node) {
; CHECK-NEXT: {{^}} 0{{$}}
; CHECK-NEXT: }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @external()

define void @l2() {
  call void @l1()
  ret void
}

define void @l1() {
  call void @l2()
  ret void
}

define i32 @simple() {
  ret i32 23
}

define void @root() {
  call void @l1()
  call i32 @simple()
  call void @external()
  ret void
}
