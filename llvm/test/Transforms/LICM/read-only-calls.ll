; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s

; We should be able to hoist loads in presence of read only calls and stores
; that do not alias.

; Since LICM uses the AST mechanism for alias analysis, we will clump
; together all loads and stores in one set along with the read-only call.
; This prevents hoisting load that doesn't alias with any other memory
; operations.

declare void @foo(i64, ptr) readonly

; hoist the load out with the n2-threshold
; since it doesn't alias with the store.
; default AST mechanism clumps all memory locations in one set because of the
; readonly call
define void @test1(ptr %ptr) {
; CHECK-LABEL: @test1(
; CHECK-LABEL: entry:
; CHECK:         %val = load i32, ptr %ptr
; CHECK-LABEL: loop:
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  %val = load i32, ptr %ptr
  call void @foo(i64 4, ptr %ptr)
  %p2 = getelementptr i32, ptr %ptr, i32 1
  store volatile i32 0, ptr %p2
  %x.inc = add i32 %x, %val
  br label %loop
}

; can hoist out load with the default AST and the alias analysis mechanism.
define void @test2(ptr %ptr) {
; CHECK-LABEL: @test2(
; CHECK-LABEL: entry:
; CHECK:         %val = load i32, ptr %ptr
; CHECK-LABEL: loop:
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  %val = load i32, ptr %ptr
  call void @foo(i64 4, ptr %ptr)
  %x.inc = add i32 %x, %val
  br label %loop
}

; cannot hoist load since not guaranteed to execute
define void @test3(ptr %ptr) {
; CHECK-LABEL: @test3(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK:         %val = load i32, ptr %ptr
entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call void @foo(i64 4, ptr %ptr)
  %val = load i32, ptr %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}
