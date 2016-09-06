; RUN: llc -mtriple=i686-pc-gnu-linux < %s | FileCheck %s
; RUN: llc -mtriple=i686-pc-gnu-linux -print-machineinstrs=expand-isel-pseudos %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=CHECK-JT-PROB


; An unreachable default destination is replaced with the most popular case label.

define void @foo(i32 %x, i32* %to) {
; CHECK-LABEL: foo:
; CHECK: movl 4(%esp), [[REG:%e[a-z]{2}]]
; CHECK: cmpl $3, [[REG]]
; CHECK: ja .LBB0_6
; CHECK-NEXT: # BB#1:
; CHECK-NEXT: jmpl *.LJTI0_0(,[[REG]],4)
; CHECK: movl $4
; CHECK: retl

entry:
  switch i32 %x, label %default [
    i32 0, label %bb0
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb4
  ]
bb0:
  store i32 0, i32* %to
  br label %exit
bb1:
  store i32 1, i32* %to
  br label %exit
bb2:
  store i32 2, i32* %to
  br label %exit
bb3:
  store i32 3, i32* %to
  br label %exit
bb4:
  store i32 4, i32* %to
  br label %exit
exit:
  ret void
default:
  unreachable

; The jump table has four entries.
; CHECK-LABEL: .LJTI0_0:
; CHECK-NEXT: .long  .LBB0_2
; CHECK-NEXT: .long  .LBB0_3
; CHECK-NEXT: .long  .LBB0_4
; CHECK-NEXT: .long  .LBB0_5
}

; Check if branch probabilities are correctly assigned to the jump table.

define void @bar(i32 %x, i32* %to) {
; CHECK-JT-PROB-LABEL: bar:
; CHECK-JT-PROB: Successors according to CFG: BB#6({{[0-9a-fx/= ]+}}14.29%) BB#8({{[0-9a-fx/= ]+}}85.71%)
; CHECK-JT-PROB: Successors according to CFG: BB#1({{[0-9a-fx/= ]+}}16.67%) BB#2({{[0-9a-fx/= ]+}}16.67%) BB#3({{[0-9a-fx/= ]+}}16.67%) BB#4({{[0-9a-fx/= ]+}}16.67%) BB#5({{[0-9a-fx/= ]+}}33.33%)

entry:
  switch i32 %x, label %default [
    i32 0, label %bb0
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb4
  ], !prof !1
bb0:
  store i32 0, i32* %to
  br label %exit
bb1:
  store i32 1, i32* %to
  br label %exit
bb2:
  store i32 2, i32* %to
  br label %exit
bb3:
  store i32 3, i32* %to
  br label %exit
bb4:
  store i32 4, i32* %to
  br label %exit
default:
  store i32 5, i32* %to
  br label %exit
exit:
  ret void
}

!1 = !{!"branch_weights", i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16}
