; RUN: opt -passes=licm -verify-memoryssa %s -S | FileCheck %s
; REQUIRES: asserts
; Ensure verification doesn't fail with unreachable blocks.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

declare dso_local void @f()

; CHECK-LABEL: @foo
define dso_local void @foo(i1 %arg) {
entry:
  br i1 %arg, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %try.cont

if.end:                                           ; preds = %entry
; 1 = MemoryDef(liveOnEntry)
  call void @f()
  br label %try.cont

catch:                                            ; No predecessors!
; 2 = MemoryDef(liveOnEntry)
  call void @f()
  br label %try.cont

try.cont:                                         ; preds = %if.end, %catch, %if.then
; 3 = MemoryPhi({if.then,liveOnEntry},{if.end,1},{catch,liveOnEntry})
  ret void
}
