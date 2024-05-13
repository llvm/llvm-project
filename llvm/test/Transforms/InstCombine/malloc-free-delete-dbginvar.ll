; Check that the instcombine result is the same with/without debug info.
; This is a regression test for a function taken from malloc-free-delete.ll.

; RUN: opt < %s -passes=instcombine -S > %t.no_dbg.ll
; RUN: opt < %s -debugify-each -passes=instcombine -S > %t.ll
; RUN: diff %t.no_dbg.ll %t.ll

declare void @free(ptr)

define void @test12(ptr %foo) minsize {
entry:
  %tobool = icmp eq ptr %foo, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @free(ptr %foo)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}
