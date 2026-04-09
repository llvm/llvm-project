; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check that "r0 = rN" is packetized together with dealloc_return.
; CHECK: r0 = r
; CHECK-NOT: {
; CHECK: dealloc_return

target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind
define ptr @fred(ptr %user_context, i32 %x) #0 {
entry:
  %and14 = add i32 %x, 255
  %add1 = and i32 %and14, -128
  %call = tail call ptr @malloc(i32 %add1) #1
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  %0 = ptrtoint ptr %call to i32
  %sub4 = add i32 %0, 131
  %and5 = and i32 %sub4, -128
  %1 = inttoptr i32 %and5 to ptr
  %2 = inttoptr i32 %and5 to ptr
  %arrayidx = getelementptr inbounds ptr, ptr %2, i32 -1
  store ptr %call, ptr %arrayidx, align 4
  br label %cleanup

cleanup:                                          ; preds = %if.end, %entry
  %retval.0 = phi ptr [ %1, %if.end ], [ null, %entry ]
  ret ptr %retval.0
}

; Function Attrs: nounwind
declare noalias ptr @malloc(i32) local_unnamed_addr #1

attributes #0 = { nounwind }
attributes #1 = { nobuiltin nounwind }
