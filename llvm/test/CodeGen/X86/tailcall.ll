; RUN: llc < %s -mtriple=i686-- -tailcallopt | grep TAILCALL | count 7

; With -tailcallopt, CodeGen guarantees a tail call optimization
; for all of these.

declare fastcc i32 @tailcallee(i32 %a1, i32 %a2, i32 %a3, i32 %a4)

define fastcc i32 @tailcaller(i32 %in1, i32 %in2) nounwind {
entry:
  %tmp11 = tail call fastcc i32 @tailcallee(i32 %in1, i32 %in2, i32 %in1, i32 %in2)
  ret i32 %tmp11
}

declare fastcc i8* @alias_callee()

define fastcc noalias i8* @noalias_caller() nounwind {
  %p = tail call fastcc i8* @alias_callee()
  ret i8* %p
}

declare fastcc noalias i8* @noalias_callee()

define fastcc i8* @alias_caller() nounwind {
  %p = tail call fastcc noalias i8* @noalias_callee()
  ret i8* %p
}

declare fastcc i32 @i32_callee()

define fastcc i32 @ret_undef() nounwind {
  %p = tail call fastcc i32 @i32_callee()
  ret i32 undef
}

declare fastcc void @does_not_return()

define fastcc i32 @noret() nounwind {
  tail call fastcc void @does_not_return()
  unreachable
}

define fastcc void @void_test(i32, i32, i32, i32) {
  entry:
   tail call fastcc void @void_test( i32 %0, i32 %1, i32 %2, i32 %3)
   ret void 
}

define fastcc i1 @i1test(i32, i32, i32, i32) {
  entry:
  %4 = tail call fastcc i1 @i1test( i32 %0, i32 %1, i32 %2, i32 %3)
  ret i1 %4
}
