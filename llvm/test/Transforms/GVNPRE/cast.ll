; RUN: llvm-as < %s | opt -gvnpre | llvm-dis | grep b.gvnpre

define i32 @extract() {
entry:		; preds = %cond_false, %entry
	br i1 true, label %cond_true, label %cond_false

cond_true:
  br label %end

cond_false:
  %a = sext i16 0 to i32
  br label %end

end:
  %b = sext i16 0 to i32
  ret i32 %b
}
