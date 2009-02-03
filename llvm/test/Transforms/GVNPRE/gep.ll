; RUN: llvm-as < %s | opt -gvnpre | llvm-dis | grep b.gvnpre

define i32 @extract({ i32 }* %P) {
entry:		; preds = %cond_false, %entry
	br i1 true, label %cond_true, label %cond_false

cond_true:
  br label %end

cond_false:
  %a = getelementptr { i32 }* %P, i32 0, i32 0
  br label %end

end:
  %b = getelementptr { i32 }* %P, i32 0, i32 0
  ret i32 0
}
