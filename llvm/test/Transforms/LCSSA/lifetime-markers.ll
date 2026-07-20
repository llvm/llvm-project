; RUN: opt -passes='function(lcssa),verify' -S %s | FileCheck %s

; Lifetime markers must point directly to allocas. If an alloca is defined in
; a loop and a marker uses it outside the loop, LCSSA must drop all of the
; alloca's lifetime markers instead of rewriting one through a PHI.

declare void @llvm.lifetime.start.p0(ptr captures(none))
declare void @llvm.lifetime.end.p0(ptr captures(none))

define void @drop_lifetime_markers(i1 %repeat) {
; CHECK-LABEL: @drop_lifetime_markers(
; CHECK:       loop:
; CHECK-NEXT:    %storage = alloca i8, align 1
; CHECK-NOT:     call void @llvm.lifetime
; CHECK:       exit:
; CHECK-NEXT:    ret void
entry:
  br label %loop
loop:
  %storage = alloca i8
  call void @llvm.lifetime.start.p0(ptr %storage)
  br i1 %repeat, label %loop, label %exit
exit:
  call void @llvm.lifetime.end.p0(ptr %storage)
  ret void
}

; If all lifetime markers stay inside the loop, preserve them even when another
; use of the alloca requires an LCSSA PHI.
define ptr @preserve_lifetime_markers(i1 %repeat) {
; CHECK-LABEL: @preserve_lifetime_markers(
; CHECK:       loop:
; CHECK-NEXT:    %storage = alloca i8, align 1
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(ptr %storage)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(ptr %storage)
; CHECK:       exit:
; CHECK-NEXT:    %storage.lcssa = phi ptr [ %storage, %loop ]
; CHECK-NEXT:    ret ptr %storage.lcssa
entry:
  br label %loop
loop:
  %storage = alloca i8
  call void @llvm.lifetime.start.p0(ptr %storage)
  call void @llvm.lifetime.end.p0(ptr %storage)
  br i1 %repeat, label %loop, label %exit
exit:
  ret ptr %storage
}
