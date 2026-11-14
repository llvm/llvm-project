;; Test that definitions in bitcode produced before GUID assignment existed
;; (i.e. without a MODULE_CODE_GUIDLIST record and !guid metadata) get GUIDs
;; assigned on load, matching what the assign-guid pass computes from the
;; original names and linkages.

;; legacy-guid-upgrade.ll.bc was assembled from this file with an LLVM 21
;; llvm-as.
; RUN: llvm-dis %s.bc -o - | FileCheck %s

;; The same GUIDs are assigned by the assign-guid pass.
; RUN: opt -passes=assign-guid -S %s | FileCheck %s

;; Bitcode produced by a current writer carries a (possibly empty) GUID list
;; record, so no GUIDs are assigned when it lacks !guid metadata.
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s --check-prefix=NOGUID
; NOGUID-NOT: !guid

; CHECK: @g = global i32 0, !guid ![[G:[0-9]+]]
; CHECK: @h = internal global i32 1, !guid ![[H:[0-9]+]]
; CHECK: @llvm.used = appending global [1 x ptr] [ptr @h], section "llvm.metadata", !guid ![[USED:[0-9]+]]
; CHECK: @d = external global i32{{$}}
; CHECK: @a = alias i32, ptr @g{{$}}
; CHECK: define i32 @foo() !guid ![[FOO:[0-9]+]] {
; CHECK: define internal i32 @bar() !guid ![[BAR:[0-9]+]] {
; CHECK: declare i32 @baz(){{$}}
; CHECK: ![[G]] = !{i64 -5300342847281564238}
; CHECK: ![[H]] = !{i64 3784812239522636055}
; CHECK: ![[USED]] = !{i64 -2781390103448774006}
; CHECK: ![[FOO]] = !{i64 6699318081062747564}
; CHECK: ![[BAR]] = !{i64 -2778452918715194291}

source_filename = "legacy-guid.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = global i32 0
@h = internal global i32 1
@llvm.used = appending global [1 x ptr] [ptr @h], section "llvm.metadata"
@a = alias i32, ptr @g
@d = external global i32

define i32 @foo() {
  %v = load i32, ptr @g
  %w = call i32 @bar()
  %x = add i32 %v, %w
  ret i32 %x
}

define internal i32 @bar() {
  %v = load i32, ptr @h
  %w = call i32 @baz()
  %x = add i32 %v, %w
  ret i32 %x
}

declare i32 @baz()
