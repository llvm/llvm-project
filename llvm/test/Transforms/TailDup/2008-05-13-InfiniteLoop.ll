; RUN: llvm-as < %s | opt -tailduplicate | llc
; PR2323

define i32 @func_27(i32 %p_28) nounwind  {
entry:
  %tmp125 = trunc i32 %p_28 to i8   ; <i8> [#uses=1]
  %tmp5.i = icmp eq i8 %tmp125, 0   ; <i1> [#uses=1]
  br i1 %tmp5.i, label %bb8.i, label %bb.i

bb.i:   ; preds = %entry
  br label %bb39.i

bb8.i:    ; preds = %entry
  br label %bb11.i

bb11.i:   ; preds = %bb39.i, %bb8.i
  %tmp126 = trunc i32 %p_28 to i8   ; <i8> [#uses=1]
  br label %bb39.i

bb39.i:   ; preds = %bb11.i, %bb.i
  %tmp127 = trunc i32 %p_28 to i8   ; <i8> [#uses=1]
  br label %bb11.i

func_29.exit:   ; No predecessors!
  ret i32 undef
}
