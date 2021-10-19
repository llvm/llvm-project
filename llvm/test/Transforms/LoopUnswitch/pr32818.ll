; Check that the call doesn't get removed even if
; it has no uses. It could have side-effects.
; RUN: opt -loop-unswitch -enable-new-pm=0 -verify-memoryssa -S %s | FileCheck %s

; CHECK-LABEL: @tinky
define i32 @tinkywinky(i8 %patatino) {
  %cmp1 = icmp slt i8 %patatino, 5
  br label %body
body:
  %i = select i1 %cmp1, i8 6, i8 undef
  br i1 true, label %body, label %end
end:
  %split = phi i8 [ %i, %body ]
  %conv4 = sext i8 %split to i32
; CHECK: tail call fastcc i32 @fn5(
  %call = tail call fastcc i32 @fn5(i32 %conv4)
  ret i32 0
}
declare fastcc i32 @fn5(i32 returned) unnamed_addr
