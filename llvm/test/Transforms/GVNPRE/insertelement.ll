; RUN: llvm-as < %s | opt -gvnpre | llvm-dis | grep b.gvnpre

define i32 @extract() {
entry:		; preds = %cond_false, %entry
  %foo = add <2 x i32> < i32 1, i32 1 >, < i32 1, i32 1 >
	br i1 true, label %cond_true, label %cond_false

cond_true:
  br label %end

cond_false:
  %a = insertelement <2 x i32> %foo, i32 0, i32 3
  br label %end

end:
  %b = insertelement <2 x i32> %foo, i32 0, i32 3
  ret i32 0
}
