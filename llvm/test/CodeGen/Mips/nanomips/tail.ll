; RUN: llc -mtriple=nanomips -verify-machineinstrs < %s | FileCheck %s
; CHECK-NOT: jrc ${{s[0-7]}}

define dso_local void @foo(i32 signext %i) local_unnamed_addr {
entry:
  %0 = tail call { void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()* } asm "nop", "={$16},={$17},={$18},={$19},={$20},={$21},={$22},={$23},~{$1}"()
  switch i32 %i, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb8
    i32 2, label %sw.bb9
    i32 3, label %sw.bb10
    i32 4, label %sw.bb11
    i32 5, label %sw.bb12
    i32 6, label %sw.bb13
  ]

sw.bb:
  %asmresult = extractvalue { void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()* } %0, 0
  tail call void %asmresult()
  ret void

sw.bb8:
  %asmresult1 = extractvalue { void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()* } %0, 1
  tail call void %asmresult1()
  ret void

sw.bb9:
  %asmresult2 = extractvalue { void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()* } %0, 2
  tail call void %asmresult2()
  ret void

sw.bb10:
  %asmresult3 = extractvalue { void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()* } %0, 3
  tail call void %asmresult3()
  ret void

sw.bb11:
  %asmresult4 = extractvalue { void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()* } %0, 4
  tail call void %asmresult4()
  ret void

sw.bb12:
  %asmresult5 = extractvalue { void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()* } %0, 5
  tail call void %asmresult5()
  ret void

sw.bb13:
  %asmresult6 = extractvalue { void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()* } %0, 6
  tail call void %asmresult6()
  ret void

sw.default:
  %asmresult7 = extractvalue { void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()* } %0, 7
  tail call void %asmresult7()
  ret void
}
