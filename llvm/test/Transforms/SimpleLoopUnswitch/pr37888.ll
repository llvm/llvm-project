; RUN: opt -simple-loop-unswitch -loop-deletion -S < %s | FileCheck %s
; RUN: opt -simple-loop-unswitch -enable-mssa-loop-dependency=true -verify-memoryssa -loop-deletion -S < %s | FileCheck %s
;
; Check that when we do unswitching where we re-enqueue the loop to be processed
; again, but manage to delete the loop before ever getting to iterate on it, it
; doesn't crash the legacy pass manager.

target triple = "x86_64-unknown-linux-gnu"

define void @pr37888() {
; CHECK-LABEL: define void @pr37888()
entry:
  %tobool = icmp ne i16 undef, 0
  br label %for.body
; CHECK:         %[[TOBOOL:.*]] = icmp ne
; CHECK-NEXT:    br i1 %[[TOBOOL]], label %if.then, label %[[ENTRY_SPLIT:.*]]
;
; CHECK:       [[ENTRY_SPLIT]]:
; CHECK-NEXT:    br label %for.end

for.body:
  br i1 %tobool, label %if.then, label %if.end

if.then:
  unreachable
; CHECK:       if.then:
; CHECK-NEXT:    unreachable

if.end:
  br label %for.inc

for.inc:
  br i1 undef, label %for.body, label %for.end

for.end:
  ret void
; CHECK:       for.end:
; CHECK-NEXT:    ret void
}
