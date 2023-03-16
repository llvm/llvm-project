; RUN: llc < %s

; PR33094
; Make sure that a constant extractvalue doesn't cause a crash in
; SelectionDAGBuilder::visitExtractValue.

%A = type {}
%B = type {}
%Tuple = type { i64 }

@A_Inst = global %A zeroinitializer
@B_Inst = global %B zeroinitializer

define i64 @foo() {
  %s = select i1 icmp eq (ptr @A_Inst, ptr @B_Inst),
       %Tuple { i64 33 }, %Tuple { i64 42 }
  %e = extractvalue %Tuple %s, 0
  ret i64 %e
}
