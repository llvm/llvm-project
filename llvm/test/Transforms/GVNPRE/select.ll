; RUN: llvm-as < %s | opt -gvnpre | llvm-dis | grep b.gvnpre

define i32 @extract() {
entry:		; preds = %cond_false, %entry
	br i1 true, label %cond_true, label %cond_false

cond_true:
  br label %end

cond_false:
  %a = select i1 true, i32 0, i32 1
  br label %end

end:
  %b = select i1 true, i32 0, i32 1
  ret i32 %b
}
