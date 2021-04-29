; RUN: llc -O0 -mtriple=x86_64-linux -asm-verbose=false -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O0 -mtriple=x86_64-windows-itanium -asm-verbose=false -verify-machineinstrs < %s | FileCheck %s

; Fast-isel mustn't add a block to the MBB successor/predecessor list twice.
; The machine verifier will catch and complain about this case.
; CHECK-LABEL: baz
; CHECK: retq
define void @baz() {
entry:
  br i1 undef, label %exit, label %exit

exit:
  ret void
}

; rdar://8337108

; Fast-isel shouldn't try to look through the compare because it's in a
; different basic block, so its operands aren't necessarily exported
; for cross-block usage.

; CHECK: movb    %al, [[OFS:[0-9]*]](%rsp)
; CHECK: callq   {{_?}}bar
; CHECK: movb    [[OFS]](%rsp), %al

declare void @bar()

define void @foo(i32 %a, i32 %b) nounwind personality i32 (...)* @__gxx_personality_v0 {
entry:
  %q = add i32 %a, 7
  %r = add i32 %b, 9
  %t = icmp ult i32 %q, %r
  invoke void @bar() to label %next unwind label %unw
next:
  br i1 %t, label %true, label %return
true:
  call void @bar()
  br label %return
return:
  ret void
unw:
  %exn = landingpad {i8*, i32}
            cleanup
  unreachable
}

declare i32 @__gxx_personality_v0(...)
