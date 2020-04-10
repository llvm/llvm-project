; Check that the instcombine result is the same with/without debug info.
; This is a regression test for a function taken from malloc-free-delete.ll.

; RUN: opt < %s -instcombine -S > %t.no_dbg.ll
; RUN: opt < %s -debugify-each -instcombine -S > %t.ll
; RUN: diff %t.no_dbg.ll %t.ll

declare void @free(i8*)

define void @test12(i32* %foo) minsize {
entry:
  %tobool = icmp eq i32* %foo, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %bitcast = bitcast i32* %foo to i8*
  tail call void @free(i8* %bitcast)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}
