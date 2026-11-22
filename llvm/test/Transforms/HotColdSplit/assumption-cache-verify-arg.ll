; REQUIRES: asserts
; RUN: opt -passes='function(print<assumptions>),hotcoldsplit' -hotcoldsplit-threshold=-1 -disable-output < %s

; Check that the assumption cache can be verified when an assumption left in
; the original function has an argument as its condition.
define void @arg_condition(i1 %b, i1 %c) {
entry:
  call void @llvm.assume(i1 %b)
  br i1 %c, label %cold, label %exit

cold:
  call void @cold()
  br label %exit

exit:
  ret void
}

declare void @cold() cold
declare void @llvm.assume(i1)
