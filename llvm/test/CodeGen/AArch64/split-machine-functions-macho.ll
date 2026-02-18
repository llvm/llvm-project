; RUN: llc < %s -mtriple=arm64-apple-macos -split-machine-functions | FileCheck %s

define void @foo(i1 %cond) !prof !0 {
entry:
  br i1 %cond, label %hot, label %cold, !prof !1
hot:
  call void @bar()
  ret void
cold:
  call void @baz()
  ret void
}

declare void @bar()
declare void @baz()

!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"branch_weights", i32 1000, i32 0}

; CHECK: .section __TEXT,__text,regular,pure_instructions
; CHECK: _foo:
; CHECK: tbz w0, #0, [[TRAMPOLINE:LBB0_[0-9]+]]
; CHECK: bl _bar
; CHECK: ret
; CHECK: [[TRAMPOLINE]]:
; CHECK: b foo.cold

; CHECK: .section __TEXT,__text_cold,regular,pure_instructions
; CHECK: foo.cold:
; CHECK: bl _baz
; CHECK: ret
