; RUN: opt < %s -reassociate -disable-output

; It has been detected that dead loops like the one in this test case can be
; created by -jump-threading (it was detected by a csmith generated program).
;
; According to -verify this is valid input (even if it could be discussed if
; the dead loop really satisfies SSA form).
;
; The problem found was that the -reassociate pass ends up in an infinite loop
; when analysing the 'deadloop1' basic block. See "Bugzilla - Bug 30818".
define void @deadloop1() {
  br label %endlabel

deadloop1:
  %1 = xor i32 %2, 7
  %2 = xor i32 %1, 8
  br label %deadloop1

endlabel:
  ret void
}


; Another example showing that dead code could result in infinite loops in
; reassociate pass. See "Bugzilla - Bug 30818".
define void @deadloop2() {
  br label %endlabel

deadloop2:
  %1 = and i32 %2, 7
  %2 = and i32 %3, 8
  %3 = and i32 %1, 6
  br label %deadloop2

endlabel:
  ret void
}
