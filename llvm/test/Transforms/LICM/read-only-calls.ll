; RUN: opt -S -basicaa -licm -licm-n2-threshold=0 %s | FileCheck %s
; RUN: opt -licm -basicaa -licm-n2-threshold=200 < %s -S | FileCheck %s --check-prefix=ALIAS-N2
; RUN: opt -aa-pipeline=basic-aa -licm-n2-threshold=0 -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -licm-n2-threshold=200 -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck %s --check-prefix=ALIAS-N2

; We should be able to hoist loads in presence of read only calls and stores
; that do not alias.

; Since LICM uses the AST mechanism for alias analysis, we will clump
; together all loads and stores in one set along with the read-only call.
; This prevents hoisting load that doesn't alias with any other memory
; operations.

declare void @foo(i64, i32*) readonly

; hoist the load out with the n2-threshold
; since it doesn't alias with the store.
; default AST mechanism clumps all memory locations in one set because of the
; readonly call
define void @test1(i32* %ptr) {
; CHECK-LABEL: @test1(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK: %val = load i32, i32* %ptr

; ALIAS-N2-LABEL: @test1(
; ALIAS-N2-LABEL: entry:
; ALIAS-N2:         %val = load i32, i32* %ptr
; ALIAS-N2-LABEL: loop:
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  %val = load i32, i32* %ptr
  call void @foo(i64 4, i32* %ptr)
  %p2 = getelementptr i32, i32* %ptr, i32 1
  store volatile i32 0, i32* %p2
  %x.inc = add i32 %x, %val
  br label %loop
}

; can hoist out load with the default AST and the alias analysis mechanism.
define void @test2(i32* %ptr) {
; CHECK-LABEL: @test2(
; CHECK-LABEL: entry:
; CHECK: %val = load i32, i32* %ptr
; CHECK-LABEL: loop:

; ALIAS-N2-LABEL: @test2(
; ALIAS-N2-LABEL: entry:
; ALIAS-N2:         %val = load i32, i32* %ptr
; ALIAS-N2-LABEL: loop:
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  %val = load i32, i32* %ptr
  call void @foo(i64 4, i32* %ptr)
  %x.inc = add i32 %x, %val
  br label %loop
}

; cannot hoist load since not guaranteed to execute
define void @test3(i32* %ptr) {
; CHECK-LABEL: @test3(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK: %val = load i32, i32* %ptr

; ALIAS-N2-LABEL: @test3(
; ALIAS-N2-LABEL: entry:
; ALIAS-N2-LABEL: loop:
; ALIAS-N2:         %val = load i32, i32* %ptr
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call void @foo(i64 4, i32* %ptr)
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}
